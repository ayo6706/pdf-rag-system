"""API dependencies for FastAPI routes.

This module defines common dependencies used across the API, including
database sessions, vector store access, and provider instances.
"""

from typing import Annotated, AsyncGenerator
from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from arq import ArqRedis

from app.core.database import get_session_maker
from app.integrations.llm.base import BaseLLMProvider
from app.integrations.llm.litellm_provider import LiteLLMProvider
from app.integrations.vectorstores.chroma import VectorStore
from app.lib.document.base import BaseDocumentParser
from app.lib.document.pymupdf_provider import PyMuPDFParser


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to provide a request-scoped database session.

    Yields:
        An AsyncSession instance.
    """
    async with get_session_maker()() as session:
        try:
            yield session
        except Exception:  # pylint: disable=broad-except
            await session.rollback()
            raise


DbSession = Annotated[AsyncSession, Depends(get_db)]


async def get_vector_store(request: Request) -> VectorStore:
    """Dependency to retrieve the vector store from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The initialized VectorStore instance.

    Raises:
        HTTPException: If the vector store is not initialized.
    """
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Please try again later.",
        )
    return store


VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]


async def get_job_queue(request: Request) -> ArqRedis:
    """Dependency to retrieve the ingestion job queue from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The initialized ArqRedis job queue instance.

    Raises:
        HTTPException: If the job queue is not initialized.
    """
    queue = getattr(request.app.state, "job_queue", None)
    if queue is None:
        raise HTTPException(
            status_code=503,
            detail="Ingestion queue not initialized. Please try again later.",
        )
    return queue


JobQueueDep = Annotated[ArqRedis, Depends(get_job_queue)]


def get_llm_provider() -> BaseLLMProvider:
    """Dependency to retrieve the LLM provider.

    Returns:
        A concrete implementation of BaseLLMProvider.
    """
    return LiteLLMProvider()


LLMProviderDep = Annotated[BaseLLMProvider, Depends(get_llm_provider)]


def get_document_parser() -> BaseDocumentParser:
    """Dependency to retrieve the document parser.

    Returns:
        A concrete implementation of BaseDocumentParser.
    """
    return PyMuPDFParser()


DocumentParserDep = Annotated[BaseDocumentParser, Depends(get_document_parser)]
