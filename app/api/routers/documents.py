import os
import uuid
import asyncio
import logging
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Document, DocumentStatus
from app.schemas import DocumentResponse, DocumentListResponse
from app.services.document_service import save_upload_file
from app.services.ingestion import ingest_document
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # Guard against missing or non-PDF filename
    if not file.filename or not isinstance(file.filename, str) or not file.filename.lower().endswith('.pdf'):
        return JSONResponse(
            status_code=422,
            content={"detail": "Only PDF files are allowed", "error_code": "INVALID_FILE_TYPE"}
        )
    
    # Check file size
    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        return JSONResponse(
            status_code=413,
            content={"detail": f"File exceeds maximum size of {settings.max_upload_size_mb}MB", "error_code": "FILE_TOO_LARGE"}
        )
    
    # Validate PDF magic bytes (%PDF-)
    if not content[:5].startswith(b"%PDF-"):
        return JSONResponse(
            status_code=422,
            content={"detail": "File content is not a valid PDF", "error_code": "INVALID_FILE_TYPE"}
        )
    await file.seek(0)
    
    # Sanitize filename: strip path components, generate unique storage name
    original_filename = os.path.basename(file.filename)
    storage_filename = f"{uuid.uuid4().hex}_{original_filename}"

    # Save file
    try:
        await save_upload_file(file, storage_filename)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"detail": "Failed to save file to disk", "error_code": "DISK_ERROR"}
        )

    # Create DB record with sanitized filename
    new_doc = Document(filename=original_filename, storage_filename=storage_filename, status=DocumentStatus.PENDING)
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)
    
    # Kick off background ingestion pipeline
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is None:
        logger.error("Vector store not initialized — ingestion will not run")
        raise HTTPException(
            status_code=503,
            detail="Vector store is unavailable. Please try again later.",
        )
    background_tasks.add_task(ingest_document, new_doc.id, vector_store, settings)
    
    return new_doc

@router.get("", response_model=DocumentListResponse)
async def list_documents(db: AsyncSession = Depends(get_db)):
    stmt = select(Document)
    result = await db.execute(stmt)
    documents = result.scalars().all()
    return {"documents": documents, "total": len(documents)}

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(Document).where(Document.id == document_id)
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    
    if not document:
        return JSONResponse(
            status_code=404,
            content={"detail": "Document not found", "error_code": "DOCUMENT_NOT_FOUND"}
        )
        
    return document

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, request: Request, db: AsyncSession = Depends(get_db)):
    stmt = select(Document).where(Document.id == document_id)
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    
    if not document:
        return JSONResponse(
            status_code=404,
            content={"detail": "Document not found", "error_code": "DOCUMENT_NOT_FOUND"}
        )
    
    # Delete from ChromaDB first (sync call — run off the event loop)
    try:
        vector_store = getattr(request.app.state, "vector_store", None)
        if vector_store is not None:
            await asyncio.to_thread(vector_store.delete_by_doc_id, str(document_id))
    except Exception:
        logger.warning(
            f"Failed to delete chunks from ChromaDB for doc {document_id}",
            exc_info=True,
        )

    # Delete from DB, then disk — avoids orphaned DB record if commit fails
    storage_name = document.storage_filename
    await db.delete(document)
    await db.commit()

    # Delete from disk after successful commit
    file_path = os.path.join(settings.upload_dir, storage_name)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            logger.warning(f"Failed to delete file from disk: {file_path}", exc_info=True)
    
    return None
