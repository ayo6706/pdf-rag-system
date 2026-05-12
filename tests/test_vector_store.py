"""Tests for the vector store module."""

import pytest
from unittest.mock import patch, MagicMock

from app.integrations.vectorstores.chroma import VectorStore
from app.schemas.chunking import TextChunk
from app.schemas.embedding import ChunkWithEmbedding


def _make_cwe(text, page, idx):
    chunk = TextChunk(text=text, page_number=page, chunk_index=idx)
    return ChunkWithEmbedding(chunk=chunk, embedding=[0.1 * (idx + 1)] * 768)


@pytest.fixture
def mock_chroma_client():
    """Mock the chromadb.HttpClient to avoid native C++ crashes during testing."""
    with patch("app.integrations.vectorstores.chroma.chromadb.HttpClient") as mock_http_client:
        mock_instance = MagicMock()
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_http_client.return_value = mock_instance
        yield mock_http_client, mock_instance, mock_collection


@pytest.fixture
def vs(mock_chroma_client):
    """Return a VectorStore initialized with the mocked client."""
    _ = mock_chroma_client
    return VectorStore(host="localhost", port=8000, collection_name="test")


class TestVectorStore:
    def test_init(self, mock_chroma_client):
        mock_http_client_class, mock_client_instance, mock_collection = mock_chroma_client
        
        vs = VectorStore(host="custom-host", port=1234, collection_name="my_collection")
        
        mock_http_client_class.assert_called_once_with(host="custom-host", port=1234)
        mock_client_instance.get_or_create_collection.assert_called_once_with(
            name="my_collection", metadata={"hnsw:space": "cosine"}
        )

    def test_upsert_chunks(self, vs, mock_chroma_client):
        _, _, mock_collection = mock_chroma_client
        chunks = [_make_cwe("First.", 0, 0), _make_cwe("Second.", 0, 1)]
        
        vs.upsert_chunks("doc-1", chunks)
        
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        
        assert len(call_kwargs["ids"]) == 2
        assert len(call_kwargs["embeddings"]) == 2
        assert len(call_kwargs["documents"]) == 2
        assert len(call_kwargs["metadatas"]) == 2
        
        assert call_kwargs["documents"][0] == "First."
        assert call_kwargs["documents"][1] == "Second."
        assert call_kwargs["metadatas"][0]["doc_id"] == "doc-1"
        assert call_kwargs["metadatas"][1]["doc_id"] == "doc-1"

    def test_upsert_empty(self, vs, mock_chroma_client):
        _, _, mock_collection = mock_chroma_client
        vs.upsert_chunks("doc-1", [])
        mock_collection.upsert.assert_not_called()

    def test_delete_by_doc_id(self, vs, mock_chroma_client):
        _, _, mock_collection = mock_chroma_client
        vs.delete_by_doc_id("doc-1")
        mock_collection.delete.assert_called_once_with(where={"doc_id": "doc-1"})

    def test_delete_nonexistent_handles_error(self, vs, mock_chroma_client):
        _, _, mock_collection = mock_chroma_client
        # Simulate ChromaDB raising an exception when nothing is found
        mock_collection.delete.side_effect = Exception("Not found")
        
        # Should not raise
        vs.delete_by_doc_id("nonexistent")
        mock_collection.delete.assert_called_once_with(where={"doc_id": "nonexistent"})

    def test_search(self, vs, mock_chroma_client):
        _, _, mock_collection = mock_chroma_client
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "distances": [[0.2, 0.8]],
            "documents": [["text1", "text2"]],
            "metadatas": [[{"doc_id": "doc1", "page_number": 1}, {"doc_id": "doc2", "page_number": 2}]]
        }

        results = vs.search([0.1, 0.2], top_k=2)

        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=2
        )
        assert len(results) == 2
        assert results[0].chunk_text == "text1"
        assert results[0].doc_id == "doc1"
        assert results[0].page_number == 1
        assert results[0].distance == 0.2
        assert results[0].similarity == 0.9  # 1 - (0.2 / 2)

    def test_search_with_doc_ids(self, vs, mock_chroma_client):
        _, _, mock_collection = mock_chroma_client
        mock_collection.query.return_value = {"ids": []}

        vs.search([0.1, 0.2], top_k=5, doc_ids=["doc1", "doc2"])

        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=5,
            where={"doc_id": {"$in": ["doc1", "doc2"]}}
        )

    def test_search_with_single_doc_id(self, vs, mock_chroma_client):
        _, _, mock_collection = mock_chroma_client
        mock_collection.query.return_value = {"ids": []}

        vs.search([0.1], top_k=5, doc_ids=["doc1"])

        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1]],
            n_results=5,
            where={"doc_id": "doc1"}
        )
