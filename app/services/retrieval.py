"""Core retrieval orchestrator for the query pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import LLMSettings, RAGSettings
from app.integrations.llm.base import BaseLLMProvider
from app.integrations.vectorstores.chroma import SearchResult, VectorStore
from app.repositories.document import document_repository
from app.schemas.query import QueryResponse, SourceReference

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document assistant. Answer the question \
using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."
For each claim you make, cite the source as [Doc: {filename}, Page {page}].
Be concise and direct. Do not make up information.
"""


def assess_confidence(results: list[SearchResult], threshold: float) -> str:
    """Assess confidence based on the highest similarity score.

    Args:
        results: List of search results from the vector store.
        threshold: The similarity score threshold for "high" confidence.

    Returns:
        "high" if the best result is above the threshold, otherwise "low".
    """
    if not results:
        return "low"
    best_similarity = max(r.similarity for r in results)
    return "high" if best_similarity >= threshold else "low"


def build_prompt(
    results: list[SearchResult], filenames: dict[str, str], question: str
) -> str:
    """Build the prompt from retrieved chunks and the user's question.

    Args:
        results: List of search results.
        filenames: Mapping of document IDs to their original filenames.
        question: The user's original query.

    Returns:
        The formatted prompt string for the LLM.
    """
    context_blocks = []
    for r in results:
        filename = filenames.get(r.doc_id, "unknown.pdf")
        block = f"[Source: {filename}, Page {r.page_number}]\n{r.chunk_text}"
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    return f"Context:\n---\n{context_str}\n---\n\nQuestion: {question}"


async def _resolve_filenames(
    db: AsyncSession, doc_ids: set[str]
) -> dict[str, str]:
    """Batch resolve document IDs to filenames via the repository.

    Args:
        db: The database session.
        doc_ids: A set of document ID strings.

    Returns:
        A dictionary mapping document IDs to their filenames.
    """
    if not doc_ids:
        return {}

    doc_uuids: list[uuid.UUID] = []
    for d in doc_ids:
        if not d:
            continue
        try:
            doc_uuids.append(uuid.UUID(d))
        except ValueError:
            logger.warning(
                "Skipping invalid UUID from vector store result: %s", d
            )
    if not doc_uuids:
        return {}
    return await document_repository.get_filenames_by_ids(db, doc_uuids)


async def query_documents(
    question: str,
    db: AsyncSession,
    vector_store: VectorStore,
    llm_settings: LLMSettings,
    rag_settings: RAGSettings,
    llm_provider: BaseLLMProvider,
    doc_ids: list[str] | None = None,
    top_k: int = 5,
    stream: bool = True,
) -> QueryResponse | AsyncGenerator[str, None]:
    """Execute the full query pipeline.

    Args:
        question: The user's question.
        db: Request-scoped database session.
        vector_store: Initialized VectorStore instance.
        llm_settings: LLM configuration.
        rag_settings: RAG configuration.
        llm_provider: The LLM provider instance.
        doc_ids: Optional list of document UUID strings to filter by.
        top_k: Number of results to return.
        stream: Whether to stream the response.

    Returns:
        Either a QueryResponse object or an AsyncGenerator of strings.
    """

    # 1. Embed question
    query_embedding = await llm_provider.embed_text(
        question, model=llm_settings.default_embedding_model
    )

    # 2. Vector search (run in thread to avoid blocking event loop)
    results = await asyncio.to_thread(
        vector_store.search,
        query_embedding,
        top_k=top_k,
        doc_ids=doc_ids,
    )

    # 3. Resolve filenames
    unique_doc_ids = {r.doc_id for r in results}
    filenames = await _resolve_filenames(db, unique_doc_ids)

    sources = [
        SourceReference(
            filename=filenames.get(r.doc_id, "unknown.pdf"),
            page_number=r.page_number,
            relevance_score=r.similarity,
            text_preview=r.chunk_text[:200],
        )
        for r in results
    ]

    # 4. Assess confidence
    confidence = assess_confidence(results, rag_settings.confidence_threshold)

    if confidence == "low":
        msg = (
            "I don't have enough information in the provided documents "
            "to answer this question."
        )
        if not stream:
            return QueryResponse(
                answer=msg, sources=sources, confidence=confidence
            )

        async def mock_stream():
            yield msg
            yield "\n\nSOURCES_JSON:\n" + json.dumps(
                [s.model_dump() for s in sources]
            )

        return mock_stream()

    # 5. Build prompt
    prompt = build_prompt(results, filenames, question)

    # 6. Generate response
    if not stream:
        answer = await llm_provider.completion(
            prompt, SYSTEM_PROMPT, model=llm_settings.default_llm_model
        )
        return QueryResponse(
            answer=answer, sources=sources, confidence=confidence
        )

    async def stream_with_sources():
        async for chunk in llm_provider.stream_completion(
            prompt, SYSTEM_PROMPT, model=llm_settings.default_llm_model
        ):
            yield chunk
        # Send sources at the end
        yield "\n\nSOURCES_JSON:\n" + json.dumps(
            [s.model_dump() for s in sources]
        )

    return stream_with_sources()
