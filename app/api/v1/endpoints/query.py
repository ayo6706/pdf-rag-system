"""API endpoints for querying the knowledge base.

This module provides routes for executing RAG (Retrieval-Augmented Generation)
queries against ingested documents.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import DbSession, LLMProviderDep, VectorStoreDep
from app.core.config import llm_settings, rag_settings
from app.models.document import DocumentStatus
from app.repositories.document import document_repository
from app.schemas.query import QueryRequest, QueryResponse
from app.services.retrieval import query_documents

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


async def _validate_documents(
    db: AsyncSession, doc_ids: list[str] | None
) -> None:
    """Validate that requested documents exist and are ready.

    Args:
        db: The database session.
        doc_ids: Optional list of document ID strings to validate.

    Raises:
        HTTPException: If any document ID is invalid, not found, or not ready.
    """
    if doc_ids:
        # Parse UUIDs with explicit error handling
        doc_uuids = []
        for d in doc_ids:
            try:
                doc_uuids.append(uuid.UUID(d))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": f"Invalid doc_id: {d}",
                        "error_code": "INVALID_DOCUMENT_IDS",
                    },
                ) from None

        # Batch fetch instead of N+1 loop
        docs = await document_repository.get_by_ids(db, doc_uuids)
        found = {doc.id: doc for doc in docs}

        invalid_ids = []
        for d_id in doc_uuids:
            doc = found.get(d_id)
            if not doc or doc.status != DocumentStatus.READY:
                invalid_ids.append(str(d_id))

        if invalid_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Invalid or not-ready IDs: {invalid_ids}",
                    "error_code": "INVALID_DOCUMENT_IDS",
                },
            )
    else:
        # Check if there are any ready documents in the system
        count = await document_repository.get_ready_count(db)
        if count == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "No documents available for querying",
                    "error_code": "NO_DOCUMENTS_AVAILABLE",
                },
            )


async def _format_sse(
    stream: AsyncGenerator[str, None]
) -> AsyncGenerator[str, None]:
    """Format chunks as SSE events.

    Args:
        stream: The raw token/chunk stream from the LLM service.

    Yields:
        SSE-formatted strings.
    """
    try:
        async for chunk in stream:
            if chunk.startswith("\n\nSOURCES_JSON:\n"):
                # Strip marker precisely using slice
                marker = "\n\nSOURCES_JSON:\n"
                sources_json = chunk[len(marker) :]
                # Compact JSON to ensure no embedded newlines in SSE
                sources_raw = json.loads(sources_json)
                sources_compact = json.dumps(sources_raw, separators=(",", ":"))
                yield f"event: done\ndata: {sources_compact}\n\n"
            else:
                data = json.dumps({"content": chunk})
                yield f"event: token\ndata: {data}\n\n"
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error during SSE stream", exc_info=True)
        error_data = json.dumps({"error": str(exc)})
        yield f"event: error\ndata: {error_data}\n\n"


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    description=(
        "Queries ingested documents using semantic search and LLM answers."
    ),
)
async def query_endpoint(
    query: QueryRequest,
    db: DbSession,
    vector_store: VectorStoreDep,
    llm_provider: LLMProviderDep,
):
    """Executes a RAG query against the knowledge base.

    Args:
        query: The query request parameters.
        db: The database session.
        vector_store: The vector store dependency.
        llm_provider: The LLM provider dependency.

    Returns:
        A QueryResponse or a StreamingResponse.
    """
    # Extract string IDs
    doc_ids_str = [str(d) for d in query.doc_ids] if query.doc_ids else None

    await _validate_documents(db, doc_ids_str)

    response = await query_documents(
        question=query.question,
        db=db,
        vector_store=vector_store,
        llm_settings=llm_settings,
        rag_settings=rag_settings,
        llm_provider=llm_provider,
        doc_ids=doc_ids_str,
        top_k=query.top_k,
        stream=query.stream,
    )

    if query.stream:
        return StreamingResponse(
            _format_sse(response),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return response
