from __future__ import annotations

from urllib import parse

import arq
from arq import connections

from app.core import config


def get_redis_settings() -> connections.RedisSettings:
    """Build ARQ Redis settings from application configuration.

    Returns:
        A RedisSettings object initialized with the configured Redis URL.

    Raises:
        ValueError: If in production and the Redis password is missing or default.
    """
    parsed = parse.urlparse(config.infra_settings.redis_url)
    if config.infra_settings.app_env.lower() == "production" and (
        not parsed.password or parsed.password == "REPLACE_ME"
    ):
        raise ValueError("Production REDIS_URL must include a real Redis password.")
    return connections.RedisSettings.from_dsn(config.infra_settings.redis_url)


async def create_job_queue() -> arq.ArqRedis:
    """Create the Redis connection pool used to enqueue ingestion jobs.

    Returns:
        An initialized ArqRedis pool.
    """
    return await arq.create_pool(get_redis_settings())


async def close_job_queue(pool: arq.ArqRedis) -> None:
    """Close the Redis connection pool used for jobs.

    Args:
        pool: The ArqRedis connection pool to close.
    """
    await pool.aclose()
