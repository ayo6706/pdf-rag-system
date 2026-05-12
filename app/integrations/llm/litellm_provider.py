"""LLM and embedding integration using LiteLLM implementing BaseLLMProvider."""

import logging
from typing import AsyncGenerator

import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from app.core.exceptions import EmbeddingError, LLMError
from app.integrations.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception represents a transient API error worth retrying."""
    error_str = str(exc).lower()
    # LiteLLM wraps HTTP errors with status codes in the message
    transient_codes = ["429", "500", "503", "rate_limit", "timeout", "overloaded"]
    return any(code in error_str for code in transient_codes)


class LiteLLMProvider(BaseLLMProvider):

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_is_transient_error),
        reraise=True,
    )
    async def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        """Generate embeddings for a batch of texts with retry logic.

        Raises the original exception if all retries are exhausted.
        """
        response = await litellm.aembedding(model=model, input=texts)
        if len(response.data) != len(texts):
            raise EmbeddingError(
                f"Embedding count mismatch: sent {len(texts)} texts, "
                f"received {len(response.data)} embeddings (model={model})"
            )
        return [
            getattr(item, "embedding", None) or item["embedding"]
            for item in response.data
        ]

    async def embed_text(self, text: str, model: str) -> list[float]:
        """Embed a single text string (for questions)."""
        try:
            results = await self.embed_batch([text], model)
            return results[0]
        except Exception as exc:
            raise EmbeddingError(f"Embedding API failed after retries: {exc}") from exc

    async def completion(
        self,
        prompt: str,
        system_instruction: str,
        model: str,
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
        self,
        prompt: str,
        system_instruction: str,
        model: str,
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
