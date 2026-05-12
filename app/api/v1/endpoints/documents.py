"""API endpoints for document management.

This module provides routes for uploading, listing, retrieving, and deleting
PDF documents.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.api.dependencies import DbSession, JobQueueDep, VectorStoreDep
from app.core.config import infra_settings
from app.models.document import Document
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.document import document_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
    description=(
        "Uploads a PDF file, validates it, and starts background ingestion."
    ),
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    db: DbSession,
    job_queue: JobQueueDep,
) -> Document:
    """Uploads a PDF and enqueues it for ingestion.

    Args:
        file: The uploaded PDF file.
        db: The database session.
        job_queue: The ARQ job queue.

    Returns:
        The newly created Document record.

    Raises:
        HTTPException: If the file is not a PDF, too large, or invalid.
    """
    # Guard against missing or non-PDF filename
    is_valid_filename = (
        file.filename
        and isinstance(file.filename, str)
        and file.filename.lower().endswith(".pdf")
    )
    if not is_valid_filename:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Only PDF files are allowed",
                "error_code": "INVALID_FILE_TYPE",
            },
        )

    # Read file content for validation
    content = await file.read()
    max_bytes = infra_settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "message": (
                    f"File exceeds maximum size of "
                    f"{infra_settings.max_upload_size_mb}MB"
                ),
                "error_code": "FILE_TOO_LARGE",
            },
        )

    # Validate PDF magic bytes (%PDF-)
    if not content[:5].startswith(b"%PDF-"):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "File content is not a valid PDF",
                "error_code": "INVALID_FILE_TYPE",
            },
        )
    await file.seek(0)

    try:
        new_doc = await document_service.upload_and_ingest(
            db, file, content, job_queue
        )
        return new_doc
    except HTTPException:
        raise
    except OSError as exc:
        logger.error(
            "Failed to save uploaded file to disk: original=%s, error=%s",
            file.filename,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to save file to disk",
                "error_code": "DISK_ERROR",
            },
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Unexpected error during upload: original=%s, error=%s",
            file.filename,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An unexpected error occurred during upload",
                "error_code": "INTERNAL_ERROR",
            },
        )


@router.get(
    "",
    summary="List all documents",
    description="Retrieves a paginated list of all uploaded documents.",
)
async def list_documents(
    db: DbSession,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> DocumentListResponse:
    """Retrieves a paginated list of documents.

    Args:
        db: The database session.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        A DocumentListResponse with results and total count.
    """
    return await document_service.list_documents(db, skip=skip, limit=limit)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a document",
    description="Retrieves the details of a specific document by its ID.",
)
async def get_document(document_id: uuid.UUID, db: DbSession) -> Document:
    """Retrieves a single document by its UUID.

    Args:
        document_id: The UUID of the document.
        db: The database session.

    Returns:
        The Document record.

    Raises:
        HTTPException: If the document is not found.
    """
    document = await document_service.get_by_id(db, document_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Document not found",
                "error_code": "DOCUMENT_NOT_FOUND",
            },
        )

    return document


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Deletes a document from the database, disk, and vector store.",
)
async def delete_document(
    document_id: uuid.UUID, db: DbSession, vector_store: VectorStoreDep
) -> None:
    """Deletes a document and its associated data.

    Args:
        document_id: The UUID of the document to delete.
        db: The database session.
        vector_store: The vector store dependency.

    Returns:
        None (status 204).

    Raises:
        HTTPException: If the document is not found.
    """
    deleted = await document_service.delete_document(
        db, document_id, vector_store
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Document not found",
                "error_code": "DOCUMENT_NOT_FOUND",
            },
        )

    return None
