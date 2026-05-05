"""Tests for the PDF parser module."""

import pytest
import fitz  # PyMuPDF

from app.services.pdf_parser import parse_pdf, PageContent
from app.exceptions import PasswordProtectedError, PDFParseError


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a small multi-page PDF with known text content."""
    pdf_path = str(tmp_path / "sample.pdf")
    doc = fitz.open()

    # Page 0
    page = doc.new_page()
    text_point = fitz.Point(72, 72)
    page.insert_text(text_point, "This is page one.\nIt has two lines.")

    # Page 1
    page = doc.new_page()
    text_point = fitz.Point(72, 72)
    page.insert_text(text_point, "Page two content here.")

    # Page 2 — empty (no text inserted)
    doc.new_page()

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path):
    """Create a PDF with pages but no extractable text."""
    pdf_path = str(tmp_path / "empty.pdf")
    doc = fitz.open()
    # PyMuPDF requires at least 1 page to save — create blank pages
    doc.new_page()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def password_pdf(tmp_path):
    """Create a password-protected PDF."""
    pdf_path = str(tmp_path / "protected.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 72), "Secret content")
    doc.save(pdf_path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="password123")
    doc.close()
    return pdf_path


class TestParsePDF:
    def test_valid_multi_page_pdf(self, sample_pdf):
        """Should extract text from each page, including empty pages."""
        pages = parse_pdf(sample_pdf)

        assert len(pages) == 3
        assert pages[0].page_number == 0
        assert "page one" in pages[0].text.lower()
        assert pages[1].page_number == 1
        assert "page two" in pages[1].text.lower()
        # Page 2 should have empty or whitespace-only text
        assert pages[2].page_number == 2
        assert pages[2].text.strip() == ""

    def test_empty_pages_pdf(self, empty_pdf):
        """A PDF with blank pages should return pages with empty text."""
        pages = parse_pdf(empty_pdf)
        assert len(pages) == 2
        for page in pages:
            assert page.text.strip() == ""

    def test_password_protected_pdf(self, password_pdf):
        """Should raise PasswordProtectedError for encrypted PDFs."""
        with pytest.raises(PasswordProtectedError):
            parse_pdf(password_pdf)

    def test_invalid_file(self, tmp_path):
        """Should raise PDFParseError for non-PDF files."""
        fake_path = str(tmp_path / "not_a_pdf.txt")
        with open(fake_path, "w") as f:
            f.write("This is not a PDF")

        with pytest.raises(PDFParseError):
            parse_pdf(fake_path)

    def test_nonexistent_file(self):
        """Should raise PDFParseError when file doesn't exist."""
        with pytest.raises(PDFParseError):
            parse_pdf("/nonexistent/path/file.pdf")

    def test_page_content_dataclass(self, sample_pdf):
        """Each item should be a PageContent instance."""
        pages = parse_pdf(sample_pdf)
        for page in pages:
            assert isinstance(page, PageContent)
            assert isinstance(page.page_number, int)
            assert isinstance(page.text, str)
