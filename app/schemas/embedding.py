"""Data structures for embedding pipeline outputs."""

from dataclasses import dataclass

from app.services.chunker import TextChunk


@dataclass
class ChunkWithEmbedding:
    """A text chunk paired with its embedding vector."""
    chunk: TextChunk
    embedding: list[float]
