"""API endpoints for health and readiness checks.

This module provides routes for monitoring the application's status and its
integration with external components like the database and vector store.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Response
from sqlalchemy import text
from starlette import status

from app.api.dependencies import DbSession, VectorStoreDep
from app.core.config import llm_settings
from app.schemas.health import ComponentHealth, HealthResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger(__name__)


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic liveness check.

    Returns:
        A HealthResponse with status "ok".
    """
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
)
async def readiness_check(
    response: Response,
    db: DbSession,
    vector_store: VectorStoreDep,
) -> ReadinessResponse:
    """Deep readiness check of all system components.

    Args:
        response: The FastAPI response object (used to set status code).
        db: The database session.
        vector_store: The vector store dependency.

    Returns:
        A ReadinessResponse with the status of each component.
    """
    db_health = await _check_db(db)
    vector_health = await _check_vector_store(vector_store)
    llm_health = _check_llm_config()

    checks = [db_health, vector_health, llm_health]
    is_ready = all(check.status == "ok" for check in checks)
    overall = "ready" if is_ready else "degraded"

    if overall != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status=overall,
        db=db_health,
        vector_store=vector_health,
        llm=llm_health,
    )


async def _check_db(db: DbSession) -> ComponentHealth:
    """Check database connectivity.

    Args:
        db: The database session.

    Returns:
        A ComponentHealth object.
    """
    try:
        await db.execute(text("SELECT 1"))
        return ComponentHealth(status="ok")
    except Exception:  # pylint: disable=broad-except
        logger.exception("_check_db failed")
        return ComponentHealth(status="error", detail="database error")


async def _check_vector_store(vector_store: VectorStoreDep) -> ComponentHealth:
    """Check vector store connectivity.

    Args:
        vector_store: The vector store dependency.

    Returns:
        A ComponentHealth object.
    """
    try:
        healthy = await asyncio.to_thread(vector_store.health_check)
        if not healthy:
            return ComponentHealth(
                status="error",
                detail="vector store health check failed",
            )
        return ComponentHealth(status="ok")
    except Exception:  # pylint: disable=broad-except
        logger.exception("_check_vector_store failed")
        return ComponentHealth(
            status="error",
            detail="vector store health check failed",
        )


def _check_llm_config() -> ComponentHealth:
    """Check if at least one LLM API key is configured.

    Returns:
        A ComponentHealth object.
    """
    provider_keys = (
        llm_settings.google_api_key,
        llm_settings.openai_api_key,
        llm_settings.anthropic_api_key,
    )
    has_key = any(
        k is not None and bool(k.get_secret_value()) for k in provider_keys
    )
    if not has_key:
        return ComponentHealth(
            status="error",
            detail="No supported LLM provider API key is configured.",
        )
    return ComponentHealth(status="ok")
