"""Repository for document-related database operations.

This module provides the DocumentRepository class for managing Document and
Chunk records in the database, including status transitions and metadata updates.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional, List, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func, update, delete
from app.repositories.base import BaseRepository
from app.models.document import Document, DocumentStatus, Chunk
from app.schemas.document import DocumentCreate, DocumentUpdate

logger = logging.getLogger(__name__)

MAX_ERROR_MESSAGE_LENGTH = 1000


class DocumentRepository(BaseRepository[Document, DocumentCreate, DocumentUpdate]):
    """Document-specific repository."""

    async def get_by_status(
        self, db: AsyncSession, status: DocumentStatus
    ) -> List[Document]:
        """Get documents by status.

        Args:
            db: The database session.
            status: The status to filter by.

        Returns:
            A list of documents with the specified status.
        """
        result = await db.execute(
            select(Document).where(Document.status == status)
        )
        return list(result.scalars().all())

    async def reset_stuck_to_pending(self, db: AsyncSession) -> list[uuid.UUID]:
        """Reset PROCESSING documents to PENDING and return reset document IDs.

        Args:
            db: The database session.

        Returns:
            A list of UUIDs for the documents that were reset.
        """
        result = await db.execute(
            update(Document)
            .where(Document.status == DocumentStatus.PROCESSING)
            .values(status=DocumentStatus.PENDING)
            .returning(Document.id)
        )
        return list(result.scalars().all())

    async def get_processing_ids(self, db: AsyncSession) -> list[uuid.UUID]:
        """Return document IDs currently having the PROCESSING status.

        This method retrieves IDs where `Document.status == DocumentStatus.PROCESSING`.

        Args:
            db: The database session.

        Returns:
            A list of UUIDs for documents in PROCESSING status.
        """
        result = await db.execute(
            select(Document.id).where(Document.status == DocumentStatus.PROCESSING)
        )
        return list(result.scalars().all())

    async def get_by_ids(
        self, db: AsyncSession, ids: Sequence[uuid.UUID]
    ) -> List[Document]:
        """Batch fetch documents by a list of IDs.

        Args:
            db: The database session.
            ids: A sequence of document IDs to fetch.

        Returns:
            A list of found documents.
        """
        if not ids:
            return []
        result = await db.execute(
            select(Document).where(Document.id.in_(ids))
        )
        return list(result.scalars().all())

    async def get_ready_count(self, db: AsyncSession) -> int:
        """Get the count of documents that are READY.

        Args:
            db: The database session.

        Returns:
            The number of documents in READY status.
        """
        result = await db.execute(
            select(func.count(Document.id)).where(Document.status == DocumentStatus.READY)
        )
        return result.scalar_one()

    async def count(self, db: AsyncSession) -> int:
        """Get total document count.

        Args:
            db: The database session.

        Returns:
            The total number of document records.
        """
        result = await db.execute(select(func.count(Document.id)))
        return result.scalar_one()

    async def mark_processing(
        self, db: AsyncSession, doc_id: Any
    ) -> Optional[Document]:
        """Fetch a document and transition it to PROCESSING status.

        Args:
            db: The database session.
            doc_id: The ID of the document to mark.

        Returns:
            The updated document if found, None otherwise.
        """
        document = await self.get(db, doc_id)
        if document:
            document.status = DocumentStatus.PROCESSING
            await db.flush()
            await db.refresh(document)
        return document

    async def mark_failed(
        self,
        db: AsyncSession,
        doc_id: Any,
        error_message: str,
        page_count: int | None = None,
    ) -> None:
        """Set a document's status to FAILED with an error message.

        Args:
            db: The database session.
            doc_id: The ID of the document.
            error_message: The error description.
            page_count: Optional final page count.
        """
        document = await self.get(db, doc_id)
        if document:
            document.status = DocumentStatus.FAILED
            document.error_message = error_message[:MAX_ERROR_MESSAGE_LENGTH]
            if page_count is not None:
                document.page_count = page_count
            await db.flush()

    async def mark_ready(
        self,
        db: AsyncSession,
        doc_id: Any,
        page_count: int,
        chunk_count: int,
        warning_msg: str | None = None,
    ) -> None:
        """Transition a document to READY with its final metadata.

        Args:
            db: The database session.
            doc_id: The ID of the document.
            page_count: Total pages processed.
            chunk_count: Total chunks generated.
            warning_msg: Optional warning (e.g. "3 of 10 pages had no text").
                Stored in error_message for now; a dedicated warning_message
                field should be added in a future migration.
        """
        document = await self.get(db, doc_id)
        if document:
            document.page_count = page_count
            document.chunk_count = chunk_count
            document.error_message = warning_msg  # None clears any previous error
            document.status = DocumentStatus.READY
            await db.flush()

    async def add_chunks(
        self,
        db: AsyncSession,
        doc_id: Any,
        chunks_data: list[dict],
    ) -> None:
        """Bulk-add Chunk records for a document.

        Args:
            db: The database session.
            doc_id: The parent document's ID.
            chunks_data: List of dicts with keys: text, page_number, chunk_index,
                token_count.
        """
        for data in chunks_data:
            chunk_record = Chunk(document_id=doc_id, **data)
            db.add(chunk_record)

    async def delete_chunks(self, db: AsyncSession, doc_id: Any) -> None:
        """Delete persisted chunk rows for a document.

        Args:
            db: The database session.
            doc_id: The ID of the document whose chunks to delete.
        """
        await db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
        await db.flush()

    async def get_filenames_by_ids(
        self, db: AsyncSession, ids: Sequence[uuid.UUID]
    ) -> dict[str, str]:
        """Return a mapping of {str(doc_id): filename} for the given IDs.

        Args:
            db: The database session.
            ids: A sequence of document IDs.

        Returns:
            A dictionary mapping document ID strings to filenames.
        """
        if not ids:
            return {}
        result = await db.execute(
            select(Document.id, Document.filename).where(Document.id.in_(ids))
        )
        return {str(row.id): row.filename for row in result}


document_repository = DocumentRepository(Document)
