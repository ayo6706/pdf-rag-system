"""Custom exceptions for the PDF ingestion pipeline."""


class PasswordProtectedError(Exception):
    """Raised when a PDF is password-protected and cannot be opened."""
    pass


class PDFParseError(Exception):
    """Raised when a file is not a valid PDF or cannot be parsed."""
    pass


class EmbeddingError(Exception):
    """Raised when the embedding API fails after exhausting all retries."""
    pass
