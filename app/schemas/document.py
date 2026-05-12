from uuid import UUID
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from app.models.document import DocumentBase, DocumentStatus

class DocumentCreate(BaseModel):
    filename: str = Field(min_length=1)
    storage_filename: str = Field(min_length=1)
    status: DocumentStatus = DocumentStatus.PENDING

class DocumentUpdate(BaseModel):
    status: Optional[DocumentStatus] = None
    page_count: Optional[int] = Field(default=None, ge=0)
    chunk_count: Optional[int] = Field(default=None, ge=0)
    error_message: Optional[str] = None

class DocumentResponse(DocumentBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int = Field(ge=0)
