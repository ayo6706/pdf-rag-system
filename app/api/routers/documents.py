import os
import uuid
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Document, DocumentStatus
from app.schemas import DocumentResponse, DocumentListResponse
from app.services.document_service import save_upload_file, process_document_stub
from app.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
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
    
    # Kick off background task
    background_tasks.add_task(process_document_stub, new_doc.id)
    
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
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(Document).where(Document.id == document_id)
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    
    if not document:
        return JSONResponse(
            status_code=404,
            content={"detail": "Document not found", "error_code": "DOCUMENT_NOT_FOUND"}
        )
    
    # Delete from DB first, then disk — avoids orphaned DB record if commit fails
    storage_name = document.storage_filename
    await db.delete(document)
    await db.commit()

    # Delete from disk after successful commit
    file_path = os.path.join(settings.upload_dir, storage_name)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            # DB record already deleted; log and continue
            pass
    
    return None
