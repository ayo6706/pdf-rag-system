import os
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from app.models import DocumentStatus

class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    storage_filename: str
    status: DocumentStatus
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int

class ErrorResponse(BaseModel):
    detail: str
    error_code: str

class HealthResponse(BaseModel):
    status: str
