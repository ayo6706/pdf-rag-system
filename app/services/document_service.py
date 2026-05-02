import os
import uuid
import asyncio
import logging
import aiofiles
from fastapi import UploadFile
from sqlalchemy import select

from app.models import Document, DocumentStatus
from app.config import settings
from app.database import async_session_maker

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024  # 64 KB

async def process_document_stub(document_id: uuid.UUID):
    """Stub background task for processing a document."""
    async with async_session_maker() as session:
        # Fetch the document
        stmt = select(Document).where(Document.id == document_id)
        result = await session.execute(stmt)
        document = result.scalar_one_or_none()
        
        if not document:
            return

        try:
            # Simulate processing
            document.status = DocumentStatus.PROCESSING
            await session.commit()
            
            # Simulate time taken
            await asyncio.sleep(2)
            
            # Re-fetch to avoid expired instance after commit
            await session.refresh(document)
            
            # Simulate completion
            document.status = DocumentStatus.READY
            await session.commit()
        except Exception:
            logger.exception(f"Background processing failed for document {document_id}")
            await session.rollback()
            # Mark as failed
            stmt = select(Document).where(Document.id == document_id)
            result = await session.execute(stmt)
            document = result.scalar_one_or_none()
            if document:
                document.status = DocumentStatus.FAILED
                document.error_message = "Processing failed unexpectedly"
                await session.commit()

async def save_upload_file(upload_file: UploadFile, storage_filename: str) -> str:
    """Saves an uploaded file to disk using a safe storage filename. Streams in chunks to avoid OOM."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    
    # Sanitize: ensure the resolved path is inside upload_dir
    file_path = os.path.join(settings.upload_dir, os.path.basename(storage_filename))
    real_path = os.path.realpath(file_path)
    if not real_path.startswith(os.path.realpath(settings.upload_dir)):
        raise ValueError("Invalid filename — path traversal detected")
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        while True:
            chunk = await upload_file.read(CHUNK_SIZE)
            if not chunk:
                break
            await out_file.write(chunk)
        
    return file_path
