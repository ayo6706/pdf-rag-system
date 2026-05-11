"""PDF text extraction using PyMuPDF (fitz) implementing BaseDocumentParser."""

import logging
import fitz  # PyMuPDF

from app.core.exceptions import PasswordProtectedError, PDFParseError
from app.lib.document.base import BaseDocumentParser, PageContent

logger = logging.getLogger(__name__)


class PyMuPDFParser(BaseDocumentParser):
    """Document parser implementation using PyMuPDF."""

    def parse(self, file_path: str) -> list[PageContent]:
        """Extract text from each page of a PDF.

        Args:
            file_path: Absolute path to the PDF file on disk.

        Returns:
            A list of PageContent objects, one per page. Pages with no
            extractable text will have an empty string for `text`.

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

            logger.info("Parsed %d pages from %s using PyMuPDF", len(pages), file_path)
            return pages
        finally:
            if doc:
                doc.close()
