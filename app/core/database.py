from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import infra_settings

engine = create_async_engine(str(infra_settings.database_url), echo=False, pool_pre_ping=True)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

def get_session_maker():
    return async_session_maker


@asynccontextmanager
async def get_background_session() -> AsyncGenerator[AsyncSession]:
    """Provide a self-contained database session for background tasks."""
    async with get_session_maker()() as session:
        yield session
