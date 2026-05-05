"""Tests for the ingestion orchestrator."""

import os
import pytest
import fitz
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.ingestion import ingest_document

from app.models import Document, DocumentStatus, Chunk
from app.config import Settings
from sqlalchemy import select


@pytest.fixture
def test_settings(tmp_path):
    s = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        upload_dir=str(tmp_path / "uploads"),
    )
    os.makedirs(s.upload_dir, exist_ok=True)
    return s


@pytest.fixture
def test_vector_store():
    return MagicMock()


@pytest.fixture
def sample_pdf_in_uploads(test_settings):
    """Create a sample PDF in the uploads directory and return its storage filename."""
    storage_name = "test_sample.pdf"
    pdf_path = os.path.join(test_settings.upload_dir, storage_name)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 72), "This is test content for ingestion.")
    page2 = doc.new_page()
    page2.insert_text(fitz.Point(72, 72), "Second page of the document.")
    doc.save(pdf_path)
    doc.close()
    return storage_name


@pytest.fixture
def empty_pdf_in_uploads(test_settings):
    """Create an all-empty-pages PDF."""
    storage_name = "empty_pages.pdf"
    pdf_path = os.path.join(test_settings.upload_dir, storage_name)
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()
    return storage_name


class TestIngestionOrchestrator:
    @pytest.mark.asyncio
    async def test_successful_ingestion(
        self, test_settings, test_vector_store, sample_pdf_in_uploads
    ):
        """Full pipeline: PDF -> parse -> chunk -> embed -> store -> ready."""
        from tests.conftest import test_async_session_maker
        import app.services.ingestion as ing_mod

        original = ing_mod.async_session_maker
        ing_mod.async_session_maker = test_async_session_maker

        try:
            # Create a document record
            async with test_async_session_maker() as db:
                doc = Document(
                    filename="test.pdf",
                    storage_filename=sample_pdf_in_uploads,
                    status=DocumentStatus.PENDING,
                )
                db.add(doc)
                await db.commit()
                doc_id = doc.id

            # Mock the embedding call to return correct number of embeddings
            def mock_aembedding(*args, **kwargs):
                texts = kwargs.get("input", args[0] if args else [])
                response = MagicMock()
                response.data = [{"embedding": [0.1] * 768} for _ in texts]
                return response

            with patch("app.services.llm.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
                mock_embed.side_effect = lambda **kw: mock_aembedding(**kw)
                await ingest_document(doc_id, test_vector_store, test_settings)

            # Verify document status
            async with test_async_session_maker() as db:
                result = await db.execute(select(Document).where(Document.id == doc_id))
                doc = result.scalar_one()
                assert doc.status == DocumentStatus.READY
                assert doc.page_count == 2
                assert doc.chunk_count is not None
                assert doc.chunk_count > 0

                # Verify Chunk records created
                chunks = await db.execute(select(Chunk).where(Chunk.document_id == doc_id))
                chunk_list = chunks.scalars().all()
                assert len(chunk_list) == doc.chunk_count
        finally:
            ing_mod.async_session_maker = original

    @pytest.mark.asyncio
    async def test_scanned_pdf_fails(
        self, test_settings, test_vector_store, empty_pdf_in_uploads
    ):
        """All-empty-pages PDF should fail with scanned PDF message."""
        from tests.conftest import test_async_session_maker
        import app.services.ingestion as ing_mod

        original = ing_mod.async_session_maker
        ing_mod.async_session_maker = test_async_session_maker

        try:
            async with test_async_session_maker() as db:
                doc = Document(
                    filename="empty.pdf",
                    storage_filename=empty_pdf_in_uploads,
                    status=DocumentStatus.PENDING,
                )
                db.add(doc)
                await db.commit()
                doc_id = doc.id

            await ingest_document(doc_id, test_vector_store, test_settings)

            async with test_async_session_maker() as db:
                result = await db.execute(select(Document).where(Document.id == doc_id))
                doc = result.scalar_one()
                assert doc.status == DocumentStatus.FAILED
                assert doc.error_message is not None
                assert "scanned" in doc.error_message.lower()
        finally:
            ing_mod.async_session_maker = original

    @pytest.mark.asyncio
    async def test_embedding_failure_marks_failed(
        self, test_settings, test_vector_store, sample_pdf_in_uploads
    ):
        """Embedding API failure should transition to failed status."""
        from tests.conftest import test_async_session_maker
        import app.services.ingestion as ing_mod

        original = ing_mod.async_session_maker
        ing_mod.async_session_maker = test_async_session_maker

        try:
            async with test_async_session_maker() as db:
                doc = Document(
                    filename="test.pdf",
                    storage_filename=sample_pdf_in_uploads,
                    status=DocumentStatus.PENDING,
                )
                db.add(doc)
                await db.commit()
                doc_id = doc.id

            with patch("app.services.llm.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
                mock_embed.side_effect = ValueError("Invalid API key")
                await ingest_document(doc_id, test_vector_store, test_settings)

            async with test_async_session_maker() as db:
                result = await db.execute(select(Document).where(Document.id == doc_id))
                doc = result.scalar_one()
                assert doc.status == DocumentStatus.FAILED
                assert doc.error_message is not None
        finally:
            ing_mod.async_session_maker = original
