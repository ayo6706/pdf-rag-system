import pytest
import os
import json
import uuid
import fitz
from unittest.mock import AsyncMock, patch, MagicMock

from app.models import Document, DocumentStatus
from app.config import settings
from app.services.ingestion import ingest_document

pytestmark = pytest.mark.asyncio

@pytest.fixture
def sample_pdf_content():
    """Create a sample PDF in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 72), "The secret code is 42.")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

async def test_e2e_pipeline(async_client, tmp_upload_dir, sample_pdf_content):
    """End-to-end test: upload -> ingest -> query."""
    from app.main import app
    
    # 1. Setup mocks
    mock_vector_store = MagicMock()
    mock_vector_store.upsert_chunks = MagicMock()
    
    from app.services.vector_store import SearchResult
    app.state.vector_store = mock_vector_store
    
    async def mock_aembedding(*args, **kwargs):
        texts = kwargs.get("input", args[0] if args else [])
        response = MagicMock()
        response.data = [{"embedding": [0.1] * 768} for _ in texts]
        return response

    async def mock_acompletion(*args, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="The secret code is 42."))]
        return response

    # 2. Run the flow
    with patch("app.services.llm.litellm.aembedding", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.llm.litellm.acompletion", new_callable=AsyncMock) as mock_comp:
         
        mock_embed.side_effect = mock_aembedding
        mock_comp.side_effect = mock_acompletion
        
        # Upload
        files = {"file": ("secret.pdf", sample_pdf_content, "application/pdf")}
        response = await async_client.post("/documents/upload", files=files)
        assert response.status_code == 201
        doc_data = response.json()
        doc_id = uuid.UUID(doc_data["id"])
        doc_id_str = str(doc_id)
        
        from tests.conftest import test_async_session_maker
        import app.services.ingestion as ing_mod
        
        # Force ingestion to run
        await ingest_document(doc_id, mock_vector_store, settings)
        
        # Check it's ready
        async with test_async_session_maker() as db:
            from sqlalchemy import select
            doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one()
            assert doc.status == DocumentStatus.READY
        
        # Setup search mock to return the ingested chunk
        mock_vector_store.search.return_value = [
            SearchResult(chunk_text="The secret code is 42.", doc_id=doc_id_str, page_number=0, distance=0.1, similarity=0.95)
        ]
        
        # Query
        query_payload = {"question": "What is the secret code?", "stream": False}
        query_response = await async_client.post("/query", json=query_payload)
        assert query_response.status_code == 200
        result = query_response.json()
        
        assert result["confidence"] == "high"
        assert result["answer"] == "The secret code is 42."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["filename"] == "secret.pdf"
