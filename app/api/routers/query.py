import json
import uuid
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Document, DocumentStatus
from app.schemas import QueryRequest, QueryResponse
from app.services.retrieval import query_documents

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])

async def _validate_documents(db: AsyncSession, doc_ids: list[str] | None) -> None:
    """Validate that requested documents exist and are ready."""
    if doc_ids:
        doc_uuids = [uuid.UUID(d) for d in doc_ids]
        stmt = select(Document.id, Document.status).where(Document.id.in_(doc_uuids))
        result = await db.execute(stmt)
        docs = {str(row.id): row.status for row in result}
        
        invalid_ids = []
        for d_id in doc_ids:
            d_id_str = str(d_id)
            if d_id_str not in docs or docs[d_id_str] != DocumentStatus.READY:
                invalid_ids.append(d_id_str)
                
        if invalid_ids:
            raise HTTPException(
                status_code=400,
                detail={"message": f"Invalid or not-ready document IDs: {invalid_ids}", "error_code": "INVALID_DOCUMENT_IDS"}
            )
    else:
        # Check if there are any ready documents in the system
        stmt = select(func.count(Document.id)).where(Document.status == DocumentStatus.READY)
        result = await db.execute(stmt)
        count = result.scalar()
        if count == 0:
            raise HTTPException(
                status_code=400,
                detail={"message": "No documents are available for querying", "error_code": "NO_DOCUMENTS_AVAILABLE"}
            )

async def _format_sse(stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Format chunks as SSE events."""
    async for chunk in stream:
        if chunk.startswith("\n\nSOURCES_JSON:\n"):
            sources_json = chunk.replace("\n\nSOURCES_JSON:\n", "")
            yield f"event: done\ndata: {sources_json}\n\n"
        else:
            data = json.dumps({"content": chunk})
            yield f"event: token\ndata: {data}\n\n"

@router.post("", response_model=QueryResponse)
async def query_endpoint(
    request: Request,
    query: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Query the knowledge base."""
    vector_store = request.app.state.vector_store
    
    # Extract string IDs
    doc_ids_str = [str(d) for d in query.doc_ids] if query.doc_ids else None
    
    await _validate_documents(db, doc_ids_str)
    
    response = await query_documents(
        question=query.question,
        vector_store=vector_store,
        settings=settings,
        doc_ids=doc_ids_str,
        top_k=query.top_k,
        stream=query.stream,
    )
    
    if query.stream:
        return StreamingResponse(
            _format_sse(response),
            media_type="text/event-stream"
        )
    else:
        return response
