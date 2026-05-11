from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import infra_settings

engine = create_async_engine(str(infra_settings.database_url), echo=False, pool_pre_ping=True)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

def get_session_maker():
    """Return the current async_session_maker. Override-friendly for tests."""
    return async_session_maker
