import pytest
import uuid
import json
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models import DocumentStatus
from app.database import get_db

pytestmark = pytest.mark.asyncio

@pytest.fixture
def test_doc_id():
    return uuid.uuid4()

@pytest.fixture
def mock_db_with_docs(test_doc_id):
    """Mock the DB dependency to return a ready doc."""
    async def override_get_db():
        db = AsyncMock()
        # For the global count check
        count_result = MagicMock()
        count_result.scalar.return_value = 1
        
        # For the specific doc check
        doc_result = MagicMock()
        row = MagicMock()
        row.id = str(test_doc_id)
        row.status = DocumentStatus.READY
        doc_result.__iter__.return_value = [row]
        
        db.execute.side_effect = [doc_result, count_result]
        return db
        
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()

async def test_query_no_docs_available():
    async def override_get_db_empty():
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        db.execute.return_value = count_result
        yield db
    
    # We need to explicitly override the get_db used in query.py
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_db] = override_get_db_empty
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/query", json={"question": "hello?", "stream": False})
        
    app.dependency_overrides = original_overrides
    
    assert response.status_code == 400
    assert "NO_DOCUMENTS_AVAILABLE" in response.text

async def test_query_invalid_doc_ids():
    async def override_get_db_invalid():
        db = AsyncMock()
        # Returns empty, so the doc won't be found
        doc_result = MagicMock()
        doc_result.__iter__.return_value = []
        db.execute.return_value = doc_result
        yield db
    
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_db] = override_get_db_invalid
    
    bad_id = str(uuid.uuid4())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/query", json={"question": "hello?", "doc_ids": [bad_id], "stream": False})
        
    app.dependency_overrides = original_overrides
    
    assert response.status_code == 400
    assert "INVALID_DOCUMENT_IDS" in response.text

@patch("app.api.routers.query.query_documents", new_callable=AsyncMock)
async def test_query_non_streaming(mock_query_docs, test_doc_id):
    from app.database import get_db
    
    async def override_get_db_valid():
        db = AsyncMock()
        doc_result = MagicMock()
        row = MagicMock()
        row.id = str(test_doc_id)
        row.status = DocumentStatus.READY
        doc_result.__iter__.return_value = [row]
        db.execute.return_value = doc_result
        yield db
    
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_db] = override_get_db_valid
    
    from app.schemas import QueryResponse, SourceReference
    mock_query_docs.return_value = QueryResponse(
        answer="Hello",
        sources=[SourceReference(filename="a.pdf", page_number=1, relevance_score=0.9, text_preview="...")],
        confidence="high"
    )
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Mock vector store
        app.state.vector_store = MagicMock()
        response = await ac.post("/query", json={"question": "hello?", "doc_ids": [str(test_doc_id)], "stream": False})
        
    app.dependency_overrides = original_overrides
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Hello"
    assert data["confidence"] == "high"
    assert len(data["sources"]) == 1

@patch("app.api.routers.query.query_documents")
async def test_query_streaming(mock_query_docs, test_doc_id):
    from app.database import get_db
    
    async def override_get_db_valid():
        db = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 1
        db.execute.return_value = count_result
        yield db
        
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_db] = override_get_db_valid
    
    # Mock the async generator returned by query_documents
    async def mock_stream():
        yield "hel"
        yield "lo"
        yield "\n\nSOURCES_JSON:\n" + json.dumps([{"filename": "a.pdf", "page_number": 1, "relevance_score": 0.9, "text_preview": "..."}])
        
    mock_query_docs.return_value = mock_stream()
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        app.state.vector_store = MagicMock()
        response = await ac.post("/query", json={"question": "hello?", "stream": True})
        
    app.dependency_overrides = original_overrides
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    content = response.text
    assert 'event: token\ndata: {"content": "hel"}\n\n' in content
    assert 'event: token\ndata: {"content": "lo"}\n\n' in content
    assert 'event: done\ndata: [{"filename": "a.pdf"' in content
