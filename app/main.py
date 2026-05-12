"""Main entry point for the PDF Knowledge Base API.

This module initializes the FastAPI application, sets up the request handlers,
and manages the application lifecycle including external resource connections.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, responses, exceptions
import uvicorn

from app.api.v1 import router
from app.core import config, database, queue, vector_store
from app.services import document

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages the application lifecycle and external resources.

    Args:
        app: The FastAPI application instance.

    Yields:
        None.
    """
    try:
        # Initialize vector store
        app.state.vector_store = vector_store.create_vector_store(
            host=config.infra_settings.chroma_host,
            port=config.infra_settings.chroma_port,
        )
        app.state.job_queue = await queue.create_job_queue()

        try:
            # Crash recovery after external resources are ready.
            await document.document_service.recover_stuck_documents(
                vector_store=app.state.vector_store,
                job_queue=app.state.job_queue,
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception("Startup crash recovery failed; continuing startup.")

        yield
    finally:
        job_queue = getattr(app.state, "job_queue", None)
        if job_queue is not None:
            await queue.close_job_queue(job_queue)
        await database.engine.dispose()


app = FastAPI(
    title="PDF Knowledge Base API",
    description="API for ingesting and querying PDF documents",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(exceptions.RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: exceptions.RequestValidationError
) -> responses.JSONResponse:
    """Handles validation errors and returns a structured JSON response.

    Args:
        _request: The incoming request object.
        exc: The validation error exception.

    Returns:
        A JSON response with 422 status and error details.
    """
    return responses.JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_code": "VALIDATION_ERROR"},
    )


app.include_router(router.api_router)


def main() -> None:
    """Entry point for running the application directly."""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
