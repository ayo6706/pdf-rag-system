"""Tests for the embedding service (LiteLLM wrapper)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.llm import embed_chunks, ChunkWithEmbedding, _embed_batch
from app.services.chunker import TextChunk
from app.exceptions import EmbeddingError


def _make_chunks(n: int) -> list[TextChunk]:
    """Create n dummy TextChunks for testing."""
    return [TextChunk(text=f"chunk {i}", page_number=0, chunk_index=i) for i in range(n)]


def _mock_embedding_response(texts: list[str]) -> MagicMock:
    """Create a mock LiteLLM embedding response."""
    response = MagicMock()
    response.data = [{"embedding": [0.1] * 768} for _ in texts]
    return response


class TestEmbedChunks:

    @pytest.mark.asyncio
    async def test_embed_single_batch(self):
        """Should embed all chunks in a single API call when under batch_size."""
        chunks = _make_chunks(5)

        with patch("app.services.llm.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = _mock_embedding_response([c.text for c in chunks])

            results = await embed_chunks(chunks, model="test-model", batch_size=100)

        assert len(results) == 5
        assert all(isinstance(r, ChunkWithEmbedding) for r in results)
        assert all(len(r.embedding) == 768 for r in results)
        mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_multiple_batches(self):
        """Should split into multiple API calls when chunks exceed batch_size."""
        chunks = _make_chunks(15)

        with patch("app.services.llm.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            # Each call returns embeddings for the batch
            mock_embed.side_effect = [
                _mock_embedding_response([""] * 5),
                _mock_embedding_response([""] * 5),
                _mock_embedding_response([""] * 5),
            ]

            results = await embed_chunks(chunks, model="test-model", batch_size=5)

        assert len(results) == 15
        assert mock_embed.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_preserves_chunk_order(self):
        """Embeddings should be paired with their source chunks in order."""
        chunks = _make_chunks(3)

        with patch("app.services.llm.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            response = MagicMock()
            response.data = [
                {"embedding": [float(i)] * 768} for i in range(3)
            ]
            mock_embed.return_value = response

            results = await embed_chunks(chunks, model="test-model")

        for i, result in enumerate(results):
            assert result.chunk.chunk_index == i
            assert result.embedding[0] == float(i)

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """Empty input should return empty output without API calls."""
        with patch("app.services.llm.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            results = await embed_chunks([], model="test-model")

        assert results == []
        mock_embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_raises_embedding_error_after_retries(self):
        """Should raise EmbeddingError when API fails after all retries."""
        chunks = _make_chunks(3)

        with patch("app.services.llm.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = Exception("503 Service Unavailable")

            with pytest.raises(EmbeddingError, match="Embedding API failed after retries"):
                await embed_chunks(chunks, model="test-model")

    @pytest.mark.asyncio
    async def test_embed_non_transient_error_no_retry(self):
        """Non-transient errors should fail immediately without retrying."""
        chunks = _make_chunks(3)

        with patch("app.services.llm.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = ValueError("Invalid input format")

            with pytest.raises(EmbeddingError):
                await embed_chunks(chunks, model="test-model")

            # Non-transient error should not trigger retries
            assert mock_embed.call_count == 1

    @pytest.mark.asyncio
    async def test_embed_retries_on_transient_error_then_succeeds(self):
        """Should retry on transient errors and succeed if a later attempt works."""
        chunks = _make_chunks(2)

        with patch("app.services.llm.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = [
                Exception("429 rate_limit exceeded"),
                _mock_embedding_response([""] * 2),
            ]

            results = await embed_chunks(chunks, model="test-model")

        assert len(results) == 2
        assert mock_embed.call_count == 2
