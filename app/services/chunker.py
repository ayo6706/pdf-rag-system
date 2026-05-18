"""Recursive character text splitter for chunking PDF page content.

Splits text at semantic boundaries (paragraphs → lines → sentences → words)
to produce overlapping chunks that preserve coherence. Chunks never span
page boundaries and carry their source page number through the pipeline.
"""

from __future__ import annotations

import logging

from app.lib.document.base import PageContent
from app.schemas.chunking import TextChunk

logger = logging.getLogger(__name__)


# Ordered from coarsest to finest boundary
SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]


def _hard_split_by_character(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Force-split text into chunks by character count with overlap.

    Args:
        text: The text string to split.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Overlap character count.

    Returns:
        A list of hard-split chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        # Move forward by (chunk_size - overlap) to create overlap
        step = chunk_size - chunk_overlap
        start += step if chunk_overlap < chunk_size else chunk_size
    return chunks


def _merge_splits(
    splits: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Combine splits into chunks that fit chunk_size, building overlap.

    Args:
        splits: List of text substrings from splitting on the separator.
        separator: The separator string used.
        chunk_size: Maximum characters per merged chunk.
        chunk_overlap: Target overlap characters between consecutive chunks.

    Returns:
        A list of merged text chunks.
    """
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for split in splits:
        piece_length = len(split) + (len(separator) if current_chunk else 0)

        if current_length + piece_length > chunk_size and current_chunk:
            # Flush current chunk
            merged = separator.join(current_chunk)
            chunks.append(merged)

            # Start new chunk with overlap from the tail of the previous
            if chunk_overlap > 0:
                # Rebuild overlap by taking trailing pieces from current_chunk
                overlap_parts: list[str] = []
                overlap_len = 0
                for part in reversed(current_chunk):
                    part_len = len(part) + (len(separator) if overlap_parts else 0)
                    if overlap_len + part_len > chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_len += part_len
                if overlap_parts:
                    current_chunk = overlap_parts
                    current_length = overlap_len
                else:
                    current_chunk = [merged[-chunk_overlap:]]
                    current_length = len(current_chunk[0])
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(split)
        current_length += piece_length

    # Flush remaining
    if current_chunk:
        merged = separator.join(current_chunk)
        chunks.append(merged)

    return chunks


def _split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """Recursively split text at semantic boundaries.

    Tries the first separator; if any resulting piece is still too large,
    falls back to the next separator for that piece, and so on.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        separators: List of separators to try in order.

    Returns:
        A list of text strings representing the chunks.
    """
    if not text or not text.strip():
        return []

    # Base case: text fits in a single chunk
    if len(text) <= chunk_size:
        return [text]

    # Find the best separator to use at this level
    separator = separators[0] if separators else ""
    remaining_separators = separators[1:] if len(separators) > 1 else []

    if not separator:
        return _hard_split_by_character(text, chunk_size, chunk_overlap)

    # Split text and merge pieces into chunk_size groups
    splits = text.split(separator)
    chunks = _merge_splits(splits, separator, chunk_size, chunk_overlap)

    # Recursively split any chunks that are still too large
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) > chunk_size:
            sub_chunks = _split_text_recursive(
                chunk, chunk_size, chunk_overlap, remaining_separators
            )
            result.extend(sub_chunks)
        else:
            result.append(chunk)

    return result


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int = 1500,
    chunk_overlap: int = 190,
    separators: list[str] | None = None,
) -> list[TextChunk]:
    """Split page text into overlapping chunks using recursive splitting.

    Args:
        pages: List of PageContent objects from the PDF parser.
        chunk_size: Target maximum characters per chunk (~512 tokens).
        chunk_overlap: Number of characters to overlap between consecutive
            chunks from the same page.
        separators: Ordered list of separators to try. Defaults to
            ``SEPARATORS`` (paragraphs → lines → sentences → words).

    Returns:
        A list of TextChunk objects with sequential ``chunk_index`` values
        across the entire document. Empty pages produce no chunks.

    Raises:
        ValueError: If chunk_size is <= 0 or overlap is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    if separators is None:
        separators = SEPARATORS

    all_chunks: list[TextChunk] = []
    chunk_index = 0

    for page in pages:
        text = page.text.strip()
        if not text:
            # Skip empty pages — they produce no chunks
            continue

        raw_chunks = _split_text_recursive(
            text, chunk_size, chunk_overlap, separators
        )

        for raw in raw_chunks:
            stripped = raw.strip()
            if stripped:
                all_chunks.append(
                    TextChunk(
                        text=stripped,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

    logger.info("Created %d chunks from %d pages", len(all_chunks), len(pages))
    return all_chunks
