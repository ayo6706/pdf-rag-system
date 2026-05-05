import os
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator
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

class QueryRequest(BaseModel):
    question: str
    doc_ids: Optional[list[uuid.UUID]] = None
    top_k: int = 5
    stream: bool = True

    @field_validator("question")
    @classmethod
    def question_must_be_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty")
        if len(v) > 8000:
            raise ValueError("Question must not exceed 8000 characters")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_in_range(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("top_k must be between 1 and 100")
        return v

class SourceReference(BaseModel):
    filename: str
    page_number: int
    relevance_score: float
    text_preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    confidence: str
