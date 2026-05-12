from arq import ArqRedis, create_pool
from arq.connections import RedisSettings
from urllib.parse import urlparse

from app.core.config import infra_settings


def get_redis_settings() -> RedisSettings:
    """Build ARQ Redis settings from application configuration."""
    parsed = urlparse(infra_settings.redis_url)
    if infra_settings.app_env.lower() == "production" and (
        not parsed.password or parsed.password == "REPLACE_ME"
    ):
        raise ValueError("Production REDIS_URL must include a real Redis password.")
    return RedisSettings.from_dsn(infra_settings.redis_url)


async def create_job_queue() -> ArqRedis:
    """Create the Redis connection pool used to enqueue ingestion jobs."""
    return await create_pool(get_redis_settings())


async def close_job_queue(pool: ArqRedis) -> None:
    """Close the Redis connection pool used for jobs."""
    await pool.aclose()
