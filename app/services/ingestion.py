"""Ingestion orchestrator — wires the full PDF processing pipeline.

Coordinates: parse → chunk → embed → store → update DB.
"""

import os
import asyncio
import logging
import uuid

from app.core.config import InfraSettings, LLMSettings
from app.core.exceptions import PasswordProtectedError, PDFParseError, EmbeddingError
from app.schemas.embedding import ChunkWithEmbedding
from app.lib.document.base import BaseDocumentParser
from app.integrations.llm.base import BaseLLMProvider
from app.services.chunker import chunk_pages
from app.integrations.vectorstores.chroma import VectorStore
from app.repositories.document import document_repository
from app.core.database import get_background_session

logger = logging.getLogger(__name__)

MAX_ERROR_MESSAGE_LENGTH = 1000

def _truncate_error(exc: Exception, max_len: int = MAX_ERROR_MESSAGE_LENGTH) -> str:
    """Return a truncated string representation of an exception."""
    msg = str(exc)
    if len(msg) > max_len:
        return msg[:max_len - 3] + "..."
    return msg


async def _mark_failed_after_error(
    db,
    doc_id: uuid.UUID,
    error_message: str,
    vector_store: VectorStore | None = None,
    cleanup_vectors: bool = False,
) -> None:
    """Rollback current work, optionally compensate vector writes, and mark failed."""
    try:
        await db.rollback()
        if cleanup_vectors and vector_store is not None:
            await asyncio.to_thread(vector_store.delete_by_doc_id, str(doc_id))
        await document_repository.mark_failed(db, doc_id, error_message)
        await db.commit()
    except Exception:
        logger.exception("Failed to mark document %s as FAILED", doc_id)


async def ingest_document(
    doc_id: uuid.UUID,
    vector_store: VectorStore,
    infra_settings: InfraSettings,
    llm_settings: LLMSettings,
    document_parser: BaseDocumentParser,
    llm_provider: BaseLLMProvider,
) -> None:
    """Full ingestion pipeline: parse → chunk → embed → store → update DB.

    This coroutine is designed to run as a background task. It manages its
    own database session to avoid conflicts with the request session.

    Args:
        doc_id: UUID of the document to ingest.
        vector_store: Initialized VectorStore instance.
        infra_settings: Infrastructure settings.
        llm_settings: LLM settings.
        document_parser: The document parser provider.
        llm_provider: The LLM provider.
    """
    async with get_background_session() as db:
        vector_upserted = False
        try:
            # 1. Fetch and transition to PROCESSING
            document = await document_repository.mark_processing(db, doc_id)

            if not document:
                logger.error("Document %s not found — skipping ingestion", doc_id)
                return
            await db.commit()
            await db.refresh(document)

            # 2. Parse PDF (CPU-bound — run off the event loop)
            file_path = os.path.join(infra_settings.upload_dir, document.storage_filename)
            pages = await asyncio.to_thread(document_parser.parse, file_path)

            # 3. Check for empty pages
            non_empty_pages = [p for p in pages if p.text.strip()]
            total_pages = len(pages)
            warning_msg = None

            if total_pages == 0:
                await document_repository.mark_failed(db, doc_id, "PDF has no pages", page_count=0)
                await db.commit()
                return

            if not non_empty_pages:
                # All pages are empty — likely a scanned/image-based PDF
                await document_repository.mark_failed(
                    db, doc_id,
                    "No extractable text found — PDF may be scanned/image-based",
                    page_count=total_pages,
                )
                await db.commit()
                return

            if len(non_empty_pages) < total_pages:
                empty_count = total_pages - len(non_empty_pages)
                warning_msg = (
                    f"{empty_count} of {total_pages} pages had no extractable text"
                )
                logger.warning("Document %s: %s", doc_id, warning_msg)

            # 4. Chunk the text (only non-empty pages, CPU-bound)
            text_chunks = await asyncio.to_thread(chunk_pages, non_empty_pages)

            if not text_chunks:
                await document_repository.mark_failed(
                    db, doc_id,
                    "No text chunks produced after splitting",
                    page_count=total_pages,
                )
                await db.commit()
                return

            # 5. Generate embeddings (orchestrating batches)
            chunks_with_embeddings: list[ChunkWithEmbedding] = []
            batch_size = 100
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i : i + batch_size]
                texts = [c.text for c in batch]
                
                try:
                    embeddings = await llm_provider.embed_batch(texts, llm_settings.default_embedding_model)
                except Exception as exc:
                    error_detail = _truncate_error(exc)
                    logger.exception(
                        "Embedding API failed during ingestion for document %s: %s",
                        doc_id,
                        error_detail,
                    )
                    await _mark_failed_after_error(
                        db,
                        doc_id,
                        f"Embedding generation failed: {error_detail}",
                    )
                    return
                    
                for chunk, embedding in zip(batch, embeddings, strict=True):
                    chunks_with_embeddings.append(
                        ChunkWithEmbedding(chunk=chunk, embedding=embedding)
                    )

            # 6. Store in vector store (sync calls — run off the event loop)
            await asyncio.to_thread(vector_store.delete_by_doc_id, str(doc_id))
            await asyncio.to_thread(
                vector_store.upsert_chunks, str(doc_id), chunks_with_embeddings
            )
            vector_upserted = True

            # 7. Create Chunk records in the database
            chunks_data = [
                {
                    "text": cwe.chunk.text,
                    "page_number": cwe.chunk.page_number,
                    "chunk_index": cwe.chunk.chunk_index,
                    "token_count": len(cwe.chunk.text) // 3,
                }
                for cwe in chunks_with_embeddings
            ]
            await document_repository.delete_chunks(db, doc_id)
            await document_repository.add_chunks(db, doc_id, chunks_data)

            # 8. Mark document as READY
            await document_repository.mark_ready(
                db, doc_id,
                page_count=total_pages,
                chunk_count=len(chunks_with_embeddings),
                warning_msg=warning_msg,
            )
            await db.commit()

            logger.info(
                "Ingestion complete for %s: %d pages, %d chunks",
                doc_id, total_pages, len(chunks_with_embeddings),
            )

        except PasswordProtectedError as exc:
            logger.error("Document %s: %s", doc_id, exc)
            await _mark_failed_after_error(
                db, doc_id, "PDF is password-protected"
            )

        except PDFParseError as exc:
            logger.error("Document %s: %s", doc_id, exc)
            await _mark_failed_after_error(
                db, doc_id, f"PDF parsing failed: {_truncate_error(exc)}"
            )

        except EmbeddingError as exc:
            logger.error("Document %s: %s", doc_id, exc)
            await _mark_failed_after_error(
                db, doc_id, f"Embedding generation failed: {_truncate_error(exc)}"
            )

        except Exception as exc:
            logger.exception("Unexpected error during ingestion of %s", doc_id)
            await _mark_failed_after_error(
                db,
                doc_id,
                f"Unexpected error: {_truncate_error(exc)}",
                vector_store=vector_store,
                cleanup_vectors=vector_upserted,
            )
