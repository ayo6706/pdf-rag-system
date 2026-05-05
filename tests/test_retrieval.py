"""Tests for the retrieval orchestrator."""

import pytest
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.retrieval import assess_confidence, build_prompt, query_documents
from app.services.vector_store import SearchResult
from app.schemas import QueryResponse
from app.config import Settings

def test_assess_confidence():
    assert assess_confidence([], 0.3) == "low"
    
    results = [SearchResult(chunk_text="", doc_id="1", page_number=1, distance=0, similarity=0.2)]
    assert assess_confidence(results, 0.3) == "low"
    
    results.append(SearchResult(chunk_text="", doc_id="2", page_number=1, distance=0, similarity=0.4))
    assert assess_confidence(results, 0.3) == "high"

def test_build_prompt():
    results = [
        SearchResult(chunk_text="Chunk 1", doc_id="uuid1", page_number=1, distance=0, similarity=0.9),
        SearchResult(chunk_text="Chunk 2", doc_id="uuid2", page_number=5, distance=0, similarity=0.8),
    ]
    filenames = {"uuid1": "file1.pdf", "uuid2": "file2.pdf"}
    
    prompt = build_prompt(results, filenames, "What is this?")
    assert "Source: file1.pdf, Page 1" in prompt
    assert "Chunk 1" in prompt
    assert "Source: file2.pdf, Page 5" in prompt
    assert "Chunk 2" in prompt
    assert "Question: What is this?" in prompt

@pytest.fixture
def mock_settings():
    return Settings(google_api_key="test", default_embedding_model="test-embed", default_llm_model="test-llm", confidence_threshold=0.3)

@pytest.mark.asyncio
async def test_query_documents_low_confidence(mock_settings):
    vs = MagicMock()
    vs.search.return_value = [SearchResult(chunk_text="Test", doc_id="uuid1", page_number=1, distance=0, similarity=0.1)]
    
    with patch("app.services.retrieval.embed_text", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.retrieval._resolve_filenames", new_callable=AsyncMock) as mock_resolve:
        
        mock_embed.return_value = [0.1] * 768
        mock_resolve.return_value = {"uuid1": "test.pdf"}
        
        # Test non-streaming
        response = await query_documents("Q?", vs, mock_settings, stream=False)
        assert isinstance(response, QueryResponse)
        assert response.confidence == "low"
        assert "don't have enough information" in response.answer
        assert len(response.sources) == 1
        assert response.sources[0].filename == "test.pdf"

@pytest.mark.asyncio
async def test_query_documents_high_confidence(mock_settings):
    vs = MagicMock()
    vs.search.return_value = [SearchResult(chunk_text="Test context", doc_id="uuid1", page_number=1, distance=0, similarity=0.8)]
    
    with patch("app.services.retrieval.embed_text", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.retrieval._resolve_filenames", new_callable=AsyncMock) as mock_resolve, \
         patch("app.services.retrieval.completion", new_callable=AsyncMock) as mock_completion:
        
        mock_embed.return_value = [0.1] * 768
        mock_resolve.return_value = {"uuid1": "test.pdf"}
        mock_completion.return_value = "This is the answer."
        
        # Test non-streaming
        response = await query_documents("Q?", vs, mock_settings, stream=False)
        assert isinstance(response, QueryResponse)
        assert response.confidence == "high"
        assert response.answer == "This is the answer."
        assert len(response.sources) == 1
        
        # Ensure completion got the right prompt
        call_args = mock_completion.call_args[0]
        prompt = call_args[0]
        assert "Test context" in prompt
