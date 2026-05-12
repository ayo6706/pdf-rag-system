from enum import StrEnum
from sqlmodel import SQLModel, Field, Relationship, Column
from sqlalchemy import ForeignKey
from typing import Optional, List
import uuid
from datetime import datetime, timezone

class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class ChunkBase(SQLModel):
    text: str
    page_number: int
    chunk_index: int
    token_count: int

class Chunk(ChunkBase, table=True):
    __tablename__ = "chunks"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(
        sa_column=Column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    )
    
    document: "Document" = Relationship(back_populates="chunks")

class DocumentBase(SQLModel):
    filename: str
    storage_filename: str = ""
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None

class Document(DocumentBase, table=True):
    __tablename__ = "documents"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    
    chunks: List["Chunk"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


from sqlalchemy import event

@event.listens_for(Document, "before_update")
def _set_updated_at(_mapper, _connection, target):
    """Automatically set updated_at on every ORM update."""
    target.updated_at = _utcnow()

