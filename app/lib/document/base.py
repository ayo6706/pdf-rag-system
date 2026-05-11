"""Abstract interface for document text extraction providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PageContent:
    """Represents extracted text from a single document page."""
    page_number: int  # 0-indexed
    text: str


class BaseDocumentParser(ABC):
    """Abstract base class for document text extraction providers."""

    @abstractmethod
    def parse(self, file_path: str) -> list[PageContent]:
        """Extract text from each page of a document.

        Args:
            file_path: Absolute path to the document file on disk.

        Returns:
            A list of PageContent objects, one per page. Pages with no
            extractable text will have an empty string for `text`.
        """
        pass
