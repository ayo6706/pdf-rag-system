"""Service for managing the document lifecycle.

This module provides the DocumentService class, which orchestrates the
uploading, database registration, disk storage, and deletion of documents.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

import aiofiles
import arq
import fastapi
from anyio import Path as AnyioPath
from sqlalchemy.ext import asyncio as sa_asyncio

from app.core import config, database
from app.integrations.vectorstores import chroma
from app.models import document as document_model
from app.repositories import document as document_repo
from app.schemas import document as document_schema

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024  # 64 KB


class DocumentService:
    """Service for managing document lifecycle and orchestration.

    Attributes:
        repository: The repository used for database operations.
    """

    def __init__(self) -> None:
        """Initialize the document service."""
        self.repository = document_repo.document_repository

    async def recover_stuck_documents(
        self,
        vector_store: chroma.VectorStore | None = None,
        job_queue: arq.ArqRedis | None = None,
    ) -> None:
        """Reset PROCESSING documents, remove stale vectors, and requeue.

        Args:
            vector_store: Optional vector store to clean up stale chunks.
            job_queue: Optional job queue to requeue documents.
        """
        try:
            async with database.get_background_session() as session:
                reset_ids = await self.repository.reset_stuck_to_pending(session)
                await session.commit()

                if reset_ids:
                    logger.info(
                        "Crash recovery: reset %d stuck documents to PENDING.",
                        len(reset_ids)
                    )

            if vector_store is not None:
                for doc_id in reset_ids:
                    try:
                        await asyncio.to_thread(
                            vector_store.delete_by_doc_id, str(doc_id)
                        )
                    except Exception:  # pylint: disable=broad-except
                        logger.warning(
                            "Crash recovery failed to clean vectors for doc %s",
                            doc_id,
                            exc_info=True,
                        )

            if job_queue is not None:
                for doc_id in reset_ids:
                    await job_queue.enqueue_job(
                        "ingest_document_job",
                        str(doc_id),
                        _queue_name=config.infra_settings.ingestion_queue_name,
                    )
        except Exception:  # pylint: disable=broad-except
            logger.exception("Crash recovery failed — startup will continue.")

    async def save_upload_file_bytes(
        self,
        content: bytes,
        storage_filename: str
    ) -> str:
        """Saves already-read file bytes to disk using a safe storage filename.

        Args:
            content: The binary content of the file.
            storage_filename: The unique filename to use for storage.

        Returns:
            The absolute path to the saved file.

        Raises:
            ValueError: If path traversal is detected in the filename.
        """
        await AnyioPath(config.infra_settings.upload_dir).mkdir(
            parents=True, exist_ok=True
        )

        file_path = os.path.join(
            config.infra_settings.upload_dir,
            os.path.basename(storage_filename)
        )
        real_path = os.path.realpath(file_path)
        base_dir = os.path.realpath(config.infra_settings.upload_dir) + os.sep

        if not real_path.startswith(base_dir):
            raise ValueError("Invalid filename — path traversal detected")

        async with aiofiles.open(file_path, "wb") as out_file:
            await out_file.write(content)

        return file_path

    async def upload_and_ingest(
        self,
        db: sa_asyncio.AsyncSession,
        file: fastapi.UploadFile,
        content: bytes,
        job_queue: arq.ArqRedis,
    ) -> document_model.Document:
        """Process an uploaded file, save it, create DB record, and enqueue.

        Args:
            db: The database session.
            file: The uploaded file object.
            content: The file content in bytes.
            job_queue: The ARQ Redis queue instance.

        Returns:
            The newly created Document model instance.

        Raises:
            Exception: If database creation or queueing fails.
        """
        original_filename = (
            os.path.basename(file.filename) if file.filename else "unknown.pdf"
        )
        storage_filename = f"{uuid.uuid4().hex}_{original_filename}"

        saved_path = await self.save_upload_file_bytes(content, storage_filename)

        doc_create = document_schema.DocumentCreate(
            filename=original_filename,
            storage_filename=storage_filename,
            status=document_model.DocumentStatus.PENDING,
        )
        try:
            new_doc = await self.repository.create(db, doc_create)
            await db.commit()
            await db.refresh(new_doc)
        except Exception:  # pylint: disable=broad-except
            await db.rollback()
            # Remove orphaned file
            try:
                await AnyioPath(saved_path).unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Failed to clean up orphaned file: %s",
                    saved_path,
                    exc_info=True
                )
            raise

        try:
            await job_queue.enqueue_job(
                "ingest_document_job",
                str(new_doc.id),
                _queue_name=config.infra_settings.ingestion_queue_name,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(
                "Failed to enqueue ingestion for document %s", new_doc.id
            )
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
        db: sa_asyncio.AsyncSession,
        skip: int = 0,
        limit: int = 100,
    ) -> document_schema.DocumentListResponse:
        """Retrieve a paginated list of all documents.

        Args:
            db: The database session.
            skip: Number of documents to skip.
            limit: Maximum number of documents to return.

        Returns:
            A DocumentListResponse containing the documents and total count.
        """
        documents = await self.repository.get_multi(db, skip=skip, limit=limit)
        total = await self.repository.count(db)
        return document_schema.DocumentListResponse(
            documents=documents, total=total
        )

    async def get_by_id(
        self,
        db: sa_asyncio.AsyncSession,
        document_id: uuid.UUID,
    ) -> document_model.Document | None:
        """Retrieve a single document by ID.

        Args:
            db: The database session.
            document_id: The UUID of the document.

        Returns:
            The Document instance if found, None otherwise.
        """
        return await self.repository.get(db, document_id)

    async def delete_document(
        self,
        db: sa_asyncio.AsyncSession,
        document_id: uuid.UUID,
        vector_store: chroma.VectorStore | None = None,
    ) -> bool:
        """Delete a document from DB, Vector Store, and disk.

        Args:
            db: The database session.
            document_id: The UUID of the document to delete.
            vector_store: Optional vector store to remove chunks from.

        Returns:
            True if the document was found and deleted, False otherwise.
        """
        doc_record = await self.repository.get(db, document_id)
        if not doc_record:
            return False

        # Delete from vector store first
        try:
            if vector_store is not None:
                await asyncio.to_thread(
                    vector_store.delete_by_doc_id, str(document_id)
                )
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Failed to delete chunks from vector store for doc %s",
                document_id,
                exc_info=True,
            )

        # Delete from DB
        storage_name = doc_record.storage_filename
        await self.repository.delete(db, document_id)
        await db.commit()

        # Delete from disk (with path traversal protection)
        resolved_upload_dir = (
            os.path.realpath(config.infra_settings.upload_dir) + os.sep
        )
        resolved_file = os.path.realpath(
            os.path.join(config.infra_settings.upload_dir, storage_name)
        )
        if not resolved_file.startswith(resolved_upload_dir):
            logger.warning(
                "Path traversal detected during delete, skipping: %s",
                storage_name
            )
            return True

        file_path = AnyioPath(resolved_file)
        if await file_path.exists():
            try:
                await file_path.unlink()
            except OSError:
                logger.warning(
                    "Failed to delete file from disk: %s",
                    file_path,
                    exc_info=True
                )

        return True


document_service = DocumentService()
