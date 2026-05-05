from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy import select
import logging

from app.database import engine, async_session_maker
from app.models import Base, Document, DocumentStatus
from app.api.routers import health, documents, query
from app.services.vector_store import VectorStore
from app.config import settings

logger = logging.getLogger(__name__)

async def recover_stuck_documents():
    """Reset documents stuck in PROCESSING back to PENDING (crash recovery)."""
    try:
        async with async_session_maker() as session:
            stmt = select(Document).where(Document.status == DocumentStatus.PROCESSING)
            result = await session.execute(stmt)
            
            count = 0
            for doc in result.scalars():
                doc.status = DocumentStatus.PENDING
                count += 1
            
            if count:
                await session.commit()
                logger.info(f"Crash recovery: reset {count} stuck documents to PENDING.")
    except Exception:
        logger.exception("Crash recovery failed — startup will continue.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    # Crash recovery
    await recover_stuck_documents()

    # Initialize vector store
    app.state.vector_store = VectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port
    )

    yield
    
    await engine.dispose()

app = FastAPI(
    title="PDF Knowledge Base API",
    description="API for ingesting and querying PDF documents",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_code": "VALIDATION_ERROR"}
    )

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
