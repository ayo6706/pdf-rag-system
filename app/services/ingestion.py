"""Ingestion orchestrator — wires the full PDF processing pipeline.

Coordinates: parse → chunk → embed → store → update DB.
"""

import os
import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import Document, DocumentStatus, Chunk
from app.exceptions import PasswordProtectedError, PDFParseError, EmbeddingError
from app.services.pdf_parser import parse_pdf
from app.services.chunker import chunk_pages
from app.services.llm import embed_chunks
from app.services.vector_store import VectorStore
from app.database import async_session_maker

logger = logging.getLogger(__name__)

MAX_ERROR_MESSAGE_LENGTH = 1000


def _truncate_error(exc: Exception, max_len: int = MAX_ERROR_MESSAGE_LENGTH) -> str:
    """Return a truncated string representation of an exception."""
    msg = str(exc)
    if len(msg) > max_len:
        return msg[:max_len - 3] + "..."
    return msg


async def ingest_document(
    doc_id: str,
    vector_store: VectorStore,
    settings: Settings,
) -> None:
    """Full ingestion pipeline: parse → chunk → embed → store → update DB.

    This coroutine is designed to run as a background task. It manages its
    own database session to avoid conflicts with the request session.

    Args:
        doc_id: UUID string of the document to ingest.
        vector_store: Initialized VectorStore instance.
        settings: Application settings (for file paths, model config).
    """
    async with async_session_maker() as db:
        try:
            # 1. Fetch the document record
            stmt = select(Document).where(Document.id == doc_id)
            result = await db.execute(stmt)
            document = result.scalar_one_or_none()

            if not document:
                logger.error(f"Document {doc_id} not found — skipping ingestion")
                return

            # 2. Transition to processing
            document.status = DocumentStatus.PROCESSING
            await db.commit()
            await db.refresh(document)

            # 3. Parse PDF (CPU-bound — run off the event loop)
            file_path = os.path.join(settings.upload_dir, document.storage_filename)
            pages = await asyncio.to_thread(parse_pdf, file_path)

            # 4. Check for empty pages
            non_empty_pages = [p for p in pages if p.text.strip()]
            total_pages = len(pages)
            warning_msg = None

            if total_pages == 0:
                document.status = DocumentStatus.FAILED
                document.error_message = "PDF has no pages"
                document.page_count = 0
                await db.commit()
                return

            if not non_empty_pages:
                # All pages are empty — likely a scanned/image-based PDF
                document.status = DocumentStatus.FAILED
                document.error_message = (
                    "No extractable text found — PDF may be scanned/image-based"
                )
                document.page_count = total_pages
                await db.commit()
                return

            if len(non_empty_pages) < total_pages:
                empty_count = total_pages - len(non_empty_pages)
                warning_msg = (
                    f"{empty_count} of {total_pages} pages had no extractable text"
                )
                logger.warning(f"Document {doc_id}: {warning_msg}")

            # 5. Chunk the text (only non-empty pages, CPU-bound)
            text_chunks = await asyncio.to_thread(chunk_pages, non_empty_pages)

            if not text_chunks:
                document.status = DocumentStatus.FAILED
                document.error_message = "No text chunks produced after splitting"
                document.page_count = total_pages
                await db.commit()
                return

            # 6. Generate embeddings
            chunks_with_embeddings = await embed_chunks(
                text_chunks, model=settings.default_embedding_model
            )

            # 7. Store in ChromaDB (sync call — run off the event loop)
            await asyncio.to_thread(
                vector_store.upsert_chunks, str(doc_id), chunks_with_embeddings
            )

            # 8. Create Chunk ORM records in the database
            for cwe in chunks_with_embeddings:
                chunk_record = Chunk(
                    document_id=doc_id,
                    text=cwe.chunk.text,
                    page_number=cwe.chunk.page_number,
                    chunk_index=cwe.chunk.chunk_index,
                    token_count=len(cwe.chunk.text) // 3,  # rough char→token estimate
                )
                db.add(chunk_record)

            # 9. Update document metadata
            document.page_count = total_pages
            document.chunk_count = len(chunks_with_embeddings)
            document.error_message = warning_msg  # None if no warnings
            document.status = DocumentStatus.READY
            await db.commit()

            logger.info(
                f"Ingestion complete for {doc_id}: "
                f"{total_pages} pages, {len(chunks_with_embeddings)} chunks"
            )

        except PasswordProtectedError as exc:
            logger.error(f"Document {doc_id}: {exc}")
            await _mark_failed(db, doc_id, "PDF is password-protected")

        except PDFParseError as exc:
            logger.error(f"Document {doc_id}: {exc}")
            await _mark_failed(db, doc_id, f"PDF parsing failed: {_truncate_error(exc)}")

        except EmbeddingError as exc:
            logger.error(f"Document {doc_id}: {exc}")
            await _mark_failed(db, doc_id, f"Embedding generation failed: {_truncate_error(exc)}")

        except Exception as exc:
            logger.exception(f"Unexpected error during ingestion of {doc_id}")
            await _mark_failed(
                db, doc_id, f"Unexpected error: {_truncate_error(exc)}"
            )


async def _mark_failed(db: AsyncSession, doc_id: str, error_message: str) -> None:
    """Set a document's status to FAILED with an error message.

    Handles the rollback + re-fetch pattern needed when the main
    transaction has already failed.
    """
    try:
        await db.rollback()
        stmt = select(Document).where(Document.id == doc_id)
        result = await db.execute(stmt)
        document = result.scalar_one_or_none()
        if document:
            document.status = DocumentStatus.FAILED
            document.error_message = error_message[:MAX_ERROR_MESSAGE_LENGTH]
            await db.commit()
    except Exception:
        logger.exception(f"Failed to mark document {doc_id} as FAILED")
