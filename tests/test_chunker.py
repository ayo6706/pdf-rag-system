"""Tests for the text chunker module."""

from app.lib.document.base import PageContent
from app.services.chunker import chunk_pages, SEPARATORS, _split_text_recursive
from app.schemas.chunking import TextChunk


class TestSplitTextRecursive:
    """Unit tests for the internal recursive splitting function."""

    def test_text_fits_in_one_chunk(self):
        """Text shorter than chunk_size should be returned as-is."""
        result = _split_text_recursive("Short text.", 100, 10, SEPARATORS)
        assert result == ["Short text."]

    def test_empty_text(self):
        """Empty or whitespace-only text should return an empty list."""
        assert _split_text_recursive("", 100, 10, SEPARATORS) == []
        assert _split_text_recursive("   ", 100, 10, SEPARATORS) == []

    def test_splits_on_paragraphs(self):
        """Should split on double newlines first."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = _split_text_recursive(text, 30, 0, SEPARATORS)
        assert len(result) >= 2
        assert "Paragraph one." in result[0]

    def test_splits_on_sentences_when_paragraphs_too_large(self):
        """Should fall back to sentence splitting when paragraphs exceed chunk_size."""
        # One big paragraph with sentences
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = _split_text_recursive(text, 30, 0, SEPARATORS)
        assert len(result) >= 2

    def test_hard_split_as_last_resort(self):
        """Should hard-split by characters when no separators work."""
        # Long word with no separators
        text = "a" * 200
        result = _split_text_recursive(text, 50, 10, [])
        assert all(len(chunk) <= 50 for chunk in result)


class TestChunkPages:
    """Tests for the main chunk_pages function."""

    def test_single_short_page(self):
        """A page shorter than chunk_size should produce one chunk."""
        pages = [PageContent(page_number=0, text="Hello world.")]
        chunks = chunk_pages(pages, chunk_size=1500, chunk_overlap=190)

        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].page_number == 0
        assert chunks[0].chunk_index == 0

    def test_empty_page_produces_no_chunks(self):
        """Pages with only whitespace should produce no chunks."""
        pages = [PageContent(page_number=0, text="   \n\n  ")]
        chunks = chunk_pages(pages)
        assert len(chunks) == 0

    def test_page_boundaries_respected(self):
        """Chunks should not span across page boundaries."""
        pages = [
            PageContent(page_number=0, text="Page zero content."),
            PageContent(page_number=1, text="Page one content."),
        ]
        chunks = chunk_pages(pages, chunk_size=1500)

        # Each chunk should belong to only one page
        page_zero_chunks = [c for c in chunks if c.page_number == 0]
        page_one_chunks = [c for c in chunks if c.page_number == 1]
        assert len(page_zero_chunks) >= 1
        assert len(page_one_chunks) >= 1

    def test_chunk_index_sequential_across_pages(self):
        """chunk_index should be sequential across the entire document."""
        pages = [
            PageContent(page_number=0, text="Page zero."),
            PageContent(page_number=1, text="Page one."),
            PageContent(page_number=2, text="Page two."),
        ]
        chunks = chunk_pages(pages, chunk_size=1500)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_long_text_produces_multiple_chunks(self):
        """Text longer than chunk_size should produce multiple chunks."""
        # Create text with natural paragraph breaks
        paragraphs = [f"This is paragraph number {i}. It has some content." for i in range(50)]
        long_text = "\n\n".join(paragraphs)
        pages = [PageContent(page_number=0, text=long_text)]

        chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)
        assert len(chunks) > 1

    def test_overlap_present(self):
        """Consecutive chunks from the same page should have overlapping content."""
        sentences = [f"Sentence number {i} with some filler text here." for i in range(20)]
        text = "\n\n".join(sentences)
        pages = [PageContent(page_number=0, text=text)]

        chunk_overlap = 30
        chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=chunk_overlap)

        # Chunking must produce at least 2 chunks for overlap to be testable
        assert len(chunks) >= 2, f"Expected >= 2 chunks, got {len(chunks)}"

        for i in range(len(chunks) - 1):
            tail_text = chunks[i].text[-chunk_overlap:]
            head_text = chunks[i + 1].text[:chunk_overlap]
            tail_words = [word for word in tail_text.split() if len(word) > 3]
            matching_words = [word for word in tail_words if word in head_text]
            if len(tail_text) <= chunk_overlap // 2:
                has_overlap = tail_text.strip() in head_text
            else:
                has_overlap = len(matching_words) / max(1, len(tail_words)) >= 0.5
            assert has_overlap, (
                f"No overlap found between chunk {i} and {i+1}: "
                f"tail='{tail_text}', head='{head_text}'"
            )

    def test_mixed_empty_and_non_empty_pages(self):
        """Empty pages should be skipped; non-empty pages chunked normally."""
        pages = [
            PageContent(page_number=0, text="Content here."),
            PageContent(page_number=1, text=""),
            PageContent(page_number=2, text="   "),
            PageContent(page_number=3, text="More content."),
        ]
        chunks = chunk_pages(pages, chunk_size=1500)

        page_numbers = {c.page_number for c in chunks}
        assert 0 in page_numbers
        assert 3 in page_numbers
        assert 1 not in page_numbers
        assert 2 not in page_numbers

    def test_chunk_dataclass_fields(self):
        """Each chunk should be a TextChunk with all required fields."""
        pages = [PageContent(page_number=0, text="Some content.")]
        chunks = chunk_pages(pages)

        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert isinstance(chunk.text, str)
            assert isinstance(chunk.page_number, int)
            assert isinstance(chunk.chunk_index, int)
            assert len(chunk.text) > 0

    def test_empty_input(self):
        """Empty page list should return empty chunk list."""
        chunks = chunk_pages([])
        assert chunks == []

    def test_custom_separators(self):
        """Should respect custom separator list."""
        text = "A|B|C|D|E"
        pages = [PageContent(page_number=0, text=text)]
        chunks = chunk_pages(pages, chunk_size=5, chunk_overlap=0, separators=["|"])
        assert len(chunks) >= 2
