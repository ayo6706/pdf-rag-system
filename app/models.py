import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    filename: Mapped[str]
    storage_filename: Mapped[str] = mapped_column(default="")
    status: Mapped[DocumentStatus] = mapped_column(default=DocumentStatus.PENDING)
    page_count: Mapped[Optional[int]]
    chunk_count: Mapped[Optional[int]]
    error_message: Mapped[Optional[str]]
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    text: Mapped[str]
    page_number: Mapped[int]
    chunk_index: Mapped[int]
    token_count: Mapped[int]
    document: Mapped["Document"] = relationship(back_populates="chunks")
