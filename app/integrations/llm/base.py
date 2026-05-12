"""Abstract interface for LLM and embedding integration providers."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseLLMProvider(ABC):
    """Abstract base class for LLM and embedding integration providers."""

    @abstractmethod
    async def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.
            model: Model identifier string.

        Returns:
            A list of embedding vectors.
        """
        pass

    @abstractmethod
    async def embed_text(self, text: str, model: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Single text string to embed.
            model: Model identifier string.

        Returns:
            A single embedding vector.
        """
        pass

    @abstractmethod
    async def completion(self, prompt: str, system_instruction: str, model: str) -> str:
        """Non-streaming LLM completion.

        Args:
            prompt: User message / prompt text.
            system_instruction: System prompt.
            model: Model identifier.

        Returns:
            The generated response string.
        """
        pass

    @abstractmethod
    async def stream_completion(self, prompt: str, system_instruction: str, model: str) -> AsyncGenerator[str, None]:
        """Stream LLM completion.

        Args:
            prompt: User message / prompt text.
            system_instruction: System prompt.
            model: Model identifier.

        Returns:
            An async generator yielding response chunks as strings.
        """
        pass
