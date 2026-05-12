"""Data structures for chunking pipeline outputs."""

from dataclasses import dataclass


@dataclass
class TextChunk:
    """A chunk of text with source metadata."""
    text: str
    page_number: int
    chunk_index: int  # sequential across the entire document
