import os
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.config import settings

# Override settings for tests BEFORE importing app modules
settings.database_url = "sqlite+aiosqlite:///:memory:"
settings.upload_dir = "./test_uploads"

from app.main import app as fastapi_app
from app.database import get_db
from app.models import Base

from sqlalchemy.pool import StaticPool

# Setup test DB engine
test_engine = create_async_engine(
    settings.database_url, 
    echo=False, 
    poolclass=StaticPool,
    connect_args={"check_same_thread": False}
)
test_async_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)

async def override_get_db():
    async with test_async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

fastapi_app.dependency_overrides[get_db] = override_get_db

# Patch the background task session maker
import app.services.document_service
app.services.document_service.async_session_maker = test_async_session_maker

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

import pytest_asyncio

@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    # Create tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Drop tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def async_client():
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

@pytest.fixture
def tmp_upload_dir(tmp_path):
    """Creates a temporary upload directory and patches settings."""
    original_upload_dir = settings.upload_dir
    upload_path = tmp_path / "uploads"
    upload_path.mkdir(parents=True, exist_ok=True)
    settings.upload_dir = str(upload_path)
    yield upload_path
    settings.upload_dir = original_upload_dir
