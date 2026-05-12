from uuid import UUID
from typing import Optional, List
from pydantic import BaseModel, field_validator

class QueryRequest(BaseModel):
    question: str
    doc_ids: Optional[List[UUID]] = None
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
    sources: List[SourceReference]
    confidence: str
