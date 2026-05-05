"""PDF text extraction using PyMuPDF (fitz).

Extracts text page-by-page from a PDF file, returning a list of PageContent
objects. Each page's text is preserved even if empty (not skipped), so that
page numbering remains accurate throughout the pipeline.
"""

import logging
from dataclasses import dataclass

import fitz  # PyMuPDF

from app.exceptions import PasswordProtectedError, PDFParseError

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Represents extracted text from a single PDF page."""
    page_number: int  # 0-indexed
    text: str


def parse_pdf(file_path: str) -> list[PageContent]:
    """Extract text from each page of a PDF.

    Args:
        file_path: Absolute path to the PDF file on disk.

    Returns:
        A list of PageContent objects, one per page. Pages with no
        extractable text will have an empty string for ``text``.

    Raises:
        PasswordProtectedError: If the PDF is encrypted/password-protected.
        PDFParseError: If the file is not a valid PDF or cannot be opened.
    """
    doc = None
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        error_msg = str(exc).lower()
        # PyMuPDF raises a generic RuntimeError for encrypted PDFs
        if "password" in error_msg or "encrypted" in error_msg:
            raise PasswordProtectedError(f"PDF is password-protected: {file_path}") from exc
        raise PDFParseError(f"Failed to open PDF: {exc}") from exc

    try:
        # Detect password-protected PDFs that opened but need authentication
        if doc.needs_pass:
            raise PasswordProtectedError(f"PDF is password-protected: {file_path}")

        # Detect non-PDF files (PyMuPDF can open some non-PDF formats silently)
        if not doc.is_pdf:
            raise PDFParseError(f"File is not a valid PDF: {file_path}")

        pages: list[PageContent] = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages.append(PageContent(page_number=page_num, text=text))

        logger.info(f"Parsed {len(pages)} pages from {file_path}")
        return pages
    finally:
        if doc:
            doc.close()
