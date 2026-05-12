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
    db_health = await _check_db(db)
    vector_health = await _check_vector_store(vector_store)
    llm_health = _check_llm_config()

    checks = [db_health, vector_health, llm_health]
    overall = "ready" if all(check.status == "ok" for check in checks) else "degraded"
    if overall != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status=overall,
        db=db_health,
        vector_store=vector_health,
        llm=llm_health,
    )


async def _check_db(db: DbSession) -> ComponentHealth:
    try:
        await db.execute(text("SELECT 1"))
        return ComponentHealth(status="ok")
    except Exception:
        logger.exception("_check_db failed")
        return ComponentHealth(status="error", detail="database error")


async def _check_vector_store(vector_store: VectorStoreDep) -> ComponentHealth:
    try:
        healthy = await asyncio.to_thread(vector_store.health_check)
        if not healthy:
            return ComponentHealth(
                status="error",
                detail="vector store health check failed",
            )
        return ComponentHealth(status="ok")
    except Exception:
        logger.exception("_check_vector_store failed for VectorStoreDep")
        return ComponentHealth(
            status="error",
            detail="vector store health check failed",
        )


def _check_llm_config() -> ComponentHealth:
    provider_keys = (
        llm_settings.google_api_key,
        llm_settings.openai_api_key,
        llm_settings.anthropic_api_key,
    )
    has_provider_key = any(key is not None and bool(key.get_secret_value()) for key in provider_keys)
    if not has_provider_key:
        return ComponentHealth(
            status="error",
            detail="No supported LLM provider API key is configured.",
        )
    return ComponentHealth(status="ok")
