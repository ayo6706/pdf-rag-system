from __future__ import annotations

import logging
import uuid
from typing import Any

from app.core.config import infra_settings, llm_settings
from app.core.queue import get_redis_settings
from app.core.vector_store import create_vector_store
from app.integrations.llm.litellm_provider import LiteLLMProvider
from app.lib.document.pymupdf_provider import PyMuPDFParser
from app.services.ingestion import ingest_document

logger = logging.getLogger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    try:
        ctx["vector_store"] = create_vector_store(
            host=infra_settings.chroma_host,
            port=infra_settings.chroma_port,
        )
        logger.info(
            "Worker vector_store initialized for Chroma at %s:%s",
            infra_settings.chroma_host,
            infra_settings.chroma_port,
        )
        ctx["document_parser"] = PyMuPDFParser()
        logger.info("Worker document_parser initialized")
        ctx["llm_provider"] = LiteLLMProvider()
        logger.info("Worker llm_provider initialized")
    except Exception:
        logger.exception(
            "Worker startup failed while initializing resources for Chroma at %s:%s",
            infra_settings.chroma_host,
            infra_settings.chroma_port,
        )
        raise


async def shutdown(ctx: dict[str, Any]) -> None:
    vector_store = ctx.get("vector_store")
    client = getattr(vector_store, "client", None)
    close = getattr(client, "close", None)
    if close is not None:
        close()
        logger.info("Worker vector_store client closed")


async def ingest_document_job(ctx: dict[str, Any], doc_id: str) -> None:
    """ARQ job entrypoint for document ingestion."""
    logger.info("Starting ingestion job for document %s", doc_id)
    try:
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id must be a non-empty string")
        parsed_doc_id = uuid.UUID(doc_id)
    except ValueError:
        logger.error("Invalid document ID for ingestion job: %r", doc_id)
        raise

    try:
        await ingest_document(
            doc_id=parsed_doc_id,
            vector_store=ctx["vector_store"],
            infra_settings=infra_settings,
            llm_settings=llm_settings,
            document_parser=ctx["document_parser"],
            llm_provider=ctx["llm_provider"],
        )
        logger.info("Ingestion job completed for document %s", doc_id)
    except Exception:
        logger.exception("Ingestion job failed for document %s", doc_id)
        raise


class WorkerSettings:
    functions = [ingest_document_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = get_redis_settings()
    queue_name = infra_settings.ingestion_queue_name
    max_jobs = 2
    job_timeout = 600
