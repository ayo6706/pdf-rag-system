from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from app.core.database import engine
from app.core.config import infra_settings
from app.api.v1.router import api_router
from app.core.vector_store import create_vector_store
from app.core.queue import close_job_queue, create_job_queue
from app.services.document import document_service

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Initialize vector store
        app.state.vector_store = create_vector_store(
            host=infra_settings.chroma_host,
            port=infra_settings.chroma_port,
        )
        app.state.job_queue = await create_job_queue()

        try:
            # Crash recovery after external resources are ready.
            await document_service.recover_stuck_documents(
                vector_store=app.state.vector_store,
                job_queue=app.state.job_queue,
            )
        except Exception:
            logger.exception("Startup crash recovery failed; continuing startup.")

        yield
    finally:
        job_queue = getattr(app.state, "job_queue", None)
        if job_queue is not None:
            await close_job_queue(job_queue)
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
