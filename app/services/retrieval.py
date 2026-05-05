"""Core retrieval orchestrator for the query pipeline."""

import json
import logging
import asyncio
from typing import AsyncGenerator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.database import async_session_maker
from app.models import Document
from app.schemas import SourceReference, QueryResponse
from app.services.vector_store import VectorStore, SearchResult
from app.services.llm import embed_text, stream_completion, completion

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."
For each claim you make, cite the source as [Doc: {filename}, Page {page}].
Be concise and direct. Do not make up information.
"""

def assess_confidence(results: list[SearchResult], threshold: float) -> str:
    """Assess confidence based on the highest similarity score."""
    if not results:
        return "low"
    best_similarity = max(r.similarity for r in results)
    return "high" if best_similarity >= threshold else "low"

def build_prompt(results: list[SearchResult], filenames: dict[str, str], question: str) -> str:
    """Build the prompt from retrieved chunks and the user's question."""
    context_blocks = []
    for r in results:
        filename = filenames.get(r.doc_id, "unknown.pdf")
        block = f"[Source: {filename}, Page {r.page_number}]\n{r.chunk_text}"
        context_blocks.append(block)
        
    context_str = "\n\n".join(context_blocks)
    
    return f"Context:\n---\n{context_str}\n---\n\nQuestion: {question}"

import uuid

async def _resolve_filenames(doc_ids: set[str]) -> dict[str, str]:
    """Batch resolve document IDs to filenames via database."""
    if not doc_ids:
        return {}
        
    doc_uuids = [uuid.UUID(d) for d in doc_ids if d]
        
    async with async_session_maker() as db:
        stmt = select(Document.id, Document.filename).where(Document.id.in_(doc_uuids))
        result = await db.execute(stmt)
        return {str(row.id): row.filename for row in result}

async def query_documents(
    question: str,
    vector_store: VectorStore,
    settings: Settings,
    doc_ids: list[str] | None = None,
    top_k: int = 5,
    stream: bool = True,
) -> QueryResponse | AsyncGenerator[str, None]:
    """Execute the full query pipeline."""
    
    # 1. Embed question
    query_embedding = await embed_text(question, model=settings.default_embedding_model)
    
    # 2. Vector search (run in thread to avoid blocking event loop)
    results = await asyncio.to_thread(
        vector_store.search,
        query_embedding,
        top_k=top_k,
        doc_ids=doc_ids,
    )
    
    # 3. Resolve filenames
    unique_doc_ids = {r.doc_id for r in results}
    filenames = await _resolve_filenames(unique_doc_ids)
    
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
    confidence = assess_confidence(results, settings.confidence_threshold)
    
    if confidence == "low":
        msg = "I don't have enough information in the provided documents to answer this question."
        if not stream:
            return QueryResponse(answer=msg, sources=sources, confidence=confidence)
            
        async def mock_stream():
            yield msg
            yield "\n\nSOURCES_JSON:\n" + json.dumps([s.model_dump() for s in sources])
        return mock_stream()
        
    # 5. Build prompt
    prompt = build_prompt(results, filenames, question)
    
    # 6. Generate response
    if not stream:
        answer = await completion(prompt, SYSTEM_PROMPT, model=settings.default_llm_model)
        return QueryResponse(answer=answer, sources=sources, confidence=confidence)
        
    async def stream_with_sources():
        async for chunk in stream_completion(prompt, SYSTEM_PROMPT, model=settings.default_llm_model):
            yield chunk
        # Send sources at the end
        yield "\n\nSOURCES_JSON:\n" + json.dumps([s.model_dump() for s in sources])
            
    return stream_with_sources()
