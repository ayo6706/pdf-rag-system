"""Embedding generation via LiteLLM.

Wraps LiteLLM's async embedding API with batching and retry logic.
"""

import logging
from dataclasses import dataclass

import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from typing import AsyncGenerator

from app.exceptions import EmbeddingError, LLMError
from app.services.chunker import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkWithEmbedding:
    """A text chunk paired with its embedding vector."""
    chunk: TextChunk
    embedding: list[float]


def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception represents a transient API error worth retrying."""
    error_str = str(exc).lower()
    # LiteLLM wraps HTTP errors with status codes in the message
    transient_codes = ["429", "500", "503", "rate_limit", "timeout", "overloaded"]
    return any(code in error_str for code in transient_codes)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_transient_error),
    reraise=True,
)
async def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    """Generate embeddings for a batch of texts with retry logic.

    Raises the original exception if all retries are exhausted.
    """
    response = await litellm.aembedding(model=model, input=texts)
    return [
        getattr(item, "embedding", None) or item["embedding"]
        for item in response.data
    ]


async def embed_chunks(
    chunks: list[TextChunk],
    model: str = "gemini/gemini-embedding-001",
    batch_size: int = 100,
) -> list[ChunkWithEmbedding]:
    """Generate embeddings for text chunks via LiteLLM.

    Handles batching (splits into groups of ``batch_size``) and retries
    on transient API errors.

    Args:
        chunks: Text chunks to embed.
        model: LiteLLM model identifier for the embedding model.
        batch_size: Maximum number of chunks per API call.

    Returns:
        A list of ChunkWithEmbedding, one per input chunk, in the same order.

    Raises:
        EmbeddingError: If an API call fails after all retries.
    """
    if not chunks:
        return []

    all_results: list[ChunkWithEmbedding] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]

        try:
            embeddings = await _embed_batch(texts, model)
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding API failed after retries: {exc}"
            ) from exc

        for chunk, embedding in zip(batch, embeddings):
            all_results.append(ChunkWithEmbedding(chunk=chunk, embedding=embedding))

    logger.info(f"Generated embeddings for {len(all_results)} chunks")
    return all_results


async def embed_text(text: str, model: str = "gemini/gemini-embedding-001") -> list[float]:
    """Embed a single text string (for questions)."""
    try:
        results = await _embed_batch([text], model)
        return results[0]
    except Exception as exc:
        raise EmbeddingError(f"Embedding API failed after retries: {exc}") from exc


async def completion(
    prompt: str,
    system_instruction: str,
    model: str = "gemini/gemini-2.5-flash",
) -> str:
    """Non-streaming LLM completion via LiteLLM."""
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        content = response.choices[0].message.content or ""
        return content.strip()
    except Exception as exc:
        raise LLMError(f"LLM API failed: {exc}") from exc


async def stream_completion(
    prompt: str,
    system_instruction: str,
    model: str = "gemini/gemini-2.5-flash",
) -> AsyncGenerator[str, None]:
    """Stream LLM completion via LiteLLM."""
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta_obj = chunk.choices[0].delta
                if delta_obj is None:
                    continue
                content = getattr(delta_obj, "content", None)
                if content:
                    yield content
    except Exception as exc:
        raise LLMError(f"LLM streaming failed: {exc}") from exc
