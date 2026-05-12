import pytest
import uuid
import json
import logging
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models.document import Document, DocumentStatus
from app.api.dependencies import get_db
from app.repositories.document import document_repository
from tests.conftest import test_async_session_maker

pytestmark = pytest.mark.asyncio
logger = logging.getLogger(__name__)

@pytest.fixture
async def ready_doc(setup_db):
    """Create a READY document for query tests. Uses its own session for isolation."""
    async with test_async_session_maker() as db:
        doc = await document_repository.create(
            db, 
            obj_in={
                "filename": "test.pdf", 
                "storage_filename": "test.pdf", 
                "status": DocumentStatus.READY
            }
        )
        await db.commit()
        # Capture scalar attributes before leaving the session context
        doc_id = doc.id
    # Re-fetch in a clean session to return a non-detached instance
    async with test_async_session_maker() as db:
        fresh_doc = await document_repository.get(db, doc_id)
    try:
        yield fresh_doc
    finally:
        try:
            async with test_async_session_maker() as db:
                await document_repository.delete(db, doc_id)
                await db.commit()
        except Exception:
            logger.exception("Failed to clean up ready_doc fixture document %s", doc_id)

async def test_query_no_docs_available(async_client):
    response = await async_client.post("/api/v1/query", json={"question": "hello?", "stream": False})
    assert response.status_code == 400
    assert "NO_DOCUMENTS_AVAILABLE" in response.text

async def test_query_invalid_doc_ids(async_client, ready_doc):
    bad_id = str(uuid.uuid4())
    response = await async_client.post("/api/v1/query", json={"question": "hello?", "doc_ids": [bad_id], "stream": False})
    assert response.status_code == 400
    assert "INVALID_DOCUMENT_IDS" in response.text

@patch("app.api.v1.endpoints.query.query_documents", new_callable=AsyncMock)
async def test_query_non_streaming(mock_query_docs, async_client, ready_doc):
    from app.schemas.query import QueryResponse, SourceReference
    mock_query_docs.return_value = QueryResponse(
        answer="Hello",
        sources=[SourceReference(filename="test.pdf", page_number=1, relevance_score=0.9, text_preview="...")],
        confidence="high"
    )
    
    response = await async_client.post(
        "/api/v1/query", 
        json={"question": "hello?", "doc_ids": [str(ready_doc.id)], "stream": False}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Hello"
    assert data["confidence"] == "high"
    assert len(data["sources"]) == 1

@patch("app.api.v1.endpoints.query.query_documents")
async def test_query_streaming(mock_query_docs, async_client, ready_doc):
    expected_sources = [{"filename": "test.pdf", "page_number": 1, "relevance_score": 0.9, "text_preview": "..."}]

    # Mock the async generator returned by query_documents
    async def mock_stream():
        yield "hel"
        yield "lo"
        yield "\n\nSOURCES_JSON:\n" + json.dumps(expected_sources)
        
    mock_query_docs.return_value = mock_stream()
    
    response = await async_client.post(
        "/api/v1/query", 
        json={"question": "hello?", "doc_ids": [str(ready_doc.id)], "stream": True}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    content = response.text
    assert 'event: token\ndata: {"content": "hel"}\n\n' in content
    assert 'event: token\ndata: {"content": "lo"}\n\n' in content

    # Validate the complete done event: well-formed SSE with full JSON and trailing \n\n
    expected_compact = json.dumps(expected_sources, separators=(",", ":"))
    expected_done_event = f"event: done\ndata: {expected_compact}\n\n"
    assert expected_done_event in content
