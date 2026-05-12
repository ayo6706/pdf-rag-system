from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from app.core.database import engine
from app.core.config import infra_settings
from sqlmodel import SQLModel
from app.api.v1.router import api_router
from app.core.vector_store import create_vector_store
from app.services.document import document_service

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        
    # Crash recovery
    await document_service.recover_stuck_documents()

    # Initialize vector store
    app.state.vector_store = create_vector_store(
        host=infra_settings.chroma_host,
        port=infra_settings.chroma_port,
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
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_code": "VALIDATION_ERROR"}
    )

app.include_router(api_router)
