import os
import uuid
import asyncio
import logging
import aiofiles
from arq import ArqRedis
from fastapi import UploadFile
from anyio import Path as AnyioPath
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import infra_settings
from app.core.database import get_background_session
from app.models.document import Document, DocumentStatus
from app.schemas.document import DocumentCreate, DocumentListResponse
from app.repositories.document import document_repository
from app.integrations.vectorstores.chroma import VectorStore

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024  # 64 KB

class DocumentService:

    def __init__(self):
        self.repository = document_repository

    async def recover_stuck_documents(
        self,
        vector_store: VectorStore | None = None,
        job_queue: ArqRedis | None = None,
    ) -> None:
        """Reset PROCESSING documents, remove stale vectors, and requeue ingestion."""
        try:
            async with get_background_session() as session:
                reset_ids = await self.repository.reset_stuck_to_pending(session)
                await session.commit()
                
                if reset_ids:
                    logger.info("Crash recovery: reset %d stuck documents to PENDING.", len(reset_ids))

            if vector_store is not None:
                for doc_id in reset_ids:
                    try:
                        await asyncio.to_thread(vector_store.delete_by_doc_id, str(doc_id))
                    except Exception:
                        logger.warning(
                            "Crash recovery failed to clean vector chunks for doc %s",
                            doc_id,
                            exc_info=True,
                        )

            if job_queue is not None:
                for doc_id in reset_ids:
                    await job_queue.enqueue_job(
                        "ingest_document_job",
                        str(doc_id),
                        _queue_name=infra_settings.ingestion_queue_name,
                    )
        except Exception:
            logger.exception("Crash recovery failed — startup will continue.")

    async def save_upload_file_bytes(self, content: bytes, storage_filename: str) -> str:
        """Saves already-read file bytes to disk using a safe storage filename."""
        await AnyioPath(infra_settings.upload_dir).mkdir(parents=True, exist_ok=True)
        
        file_path = os.path.join(infra_settings.upload_dir, os.path.basename(storage_filename))
        real_path = os.path.realpath(file_path)
        base_dir = os.path.realpath(infra_settings.upload_dir) + os.sep
        if not real_path.startswith(base_dir):
            raise ValueError("Invalid filename — path traversal detected")
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            await out_file.write(content)
            
        return file_path

    async def upload_and_ingest(
        self,
        db: AsyncSession,
        file: UploadFile,
        content: bytes,
        job_queue: ArqRedis,
    ) -> Document:
        """Process an uploaded file, save it, create DB record, and enqueue ingestion."""
        original_filename = os.path.basename(file.filename) if file.filename else "unknown.pdf"
        storage_filename = f"{uuid.uuid4().hex}_{original_filename}"

        saved_path = await self.save_upload_file_bytes(content, storage_filename)

        doc_create = DocumentCreate(
            filename=original_filename,
            storage_filename=storage_filename,
            status=DocumentStatus.PENDING
        )
        try:
            new_doc = await self.repository.create(db, doc_create)
            await db.commit()
            await db.refresh(new_doc)
        except Exception:
            await db.rollback()
            # Remove orphaned file
            try:
                await AnyioPath(saved_path).unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to clean up orphaned file: %s", saved_path, exc_info=True)
            raise

        try:
            await job_queue.enqueue_job(
                "ingest_document_job",
                str(new_doc.id),
                _queue_name=infra_settings.ingestion_queue_name,
            )
        except Exception as exc:
            logger.exception("Failed to enqueue ingestion for document %s", new_doc.id)
            await self.repository.mark_failed(
                db,
                new_doc.id,
                f"Failed to enqueue ingestion job: {exc}",
            )
            await db.commit()
            raise
        
        return new_doc

    async def list_documents(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
    ) -> DocumentListResponse:
        """Retrieve a paginated list of all documents."""
        documents = await self.repository.get_multi(db, skip=skip, limit=limit)
        total = await self.repository.count(db)
        return DocumentListResponse(documents=documents, total=total)

    async def get_by_id(
        self,
        db: AsyncSession,
        document_id: uuid.UUID,
    ) -> Document | None:
        """Retrieve a single document by ID."""
        return await self.repository.get(db, document_id)

    async def delete_document(
        self,
        db: AsyncSession,
        document_id: uuid.UUID,
        vector_store: VectorStore | None = None,
    ) -> bool:
        """Delete a document from DB, Vector Store, and disk."""
        document = await self.repository.get(db, document_id)
        if not document:
            return False

        # Delete from vector store first
        try:
            if vector_store is not None:
                await asyncio.to_thread(vector_store.delete_by_doc_id, str(document_id))
        except Exception:
            logger.warning("Failed to delete chunks from vector store for doc %s", document_id, exc_info=True)

        # Delete from DB
        storage_name = document.storage_filename
        await self.repository.delete(db, document_id)
        await db.commit()

        # Delete from disk (with path traversal protection)
        resolved_upload_dir = os.path.realpath(infra_settings.upload_dir) + os.sep
        resolved_file = os.path.realpath(os.path.join(infra_settings.upload_dir, storage_name))
        if not resolved_file.startswith(resolved_upload_dir):
            logger.warning("Path traversal detected during delete, skipping: %s", storage_name)
            return True

        file_path = AnyioPath(resolved_file)
        if await file_path.exists():
            try:
                await file_path.unlink()
            except OSError:
                logger.warning("Failed to delete file from disk: %s", file_path, exc_info=True)
                
        return True

document_service = DocumentService()
