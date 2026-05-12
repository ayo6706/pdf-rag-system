import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from unittest.mock import MagicMock

from app.core.config import infra_settings

# Override settings for tests BEFORE importing app modules
infra_settings.database_url = "sqlite+aiosqlite:///:memory:"
infra_settings.upload_dir = "./test_uploads"

from app.main import app as fastapi_app
from app.api.dependencies import get_db, get_job_queue
from sqlmodel import SQLModel

from sqlalchemy.pool import StaticPool

# Setup test DB engine
test_engine = create_async_engine(
    str(infra_settings.database_url), 
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


class FakeJobQueue:
    def __init__(self):
        self.enqueued: list[tuple[str, tuple, dict]] = []

    async def enqueue_job(self, name, *args, **kwargs):
        self.enqueued.append((name, args, kwargs))
        return MagicMock(job_id="test-job")


fake_job_queue = FakeJobQueue()


async def override_get_job_queue():
    return fake_job_queue


fastapi_app.dependency_overrides[get_job_queue] = override_get_job_queue


@pytest.fixture(autouse=True)
def reset_fake_job_queue():
    fake_job_queue.enqueued = []
    yield
    fake_job_queue.enqueued = []

# Patch the database session maker globally
import app.core.database
app.core.database.async_session_maker = test_async_session_maker

# Set up mocked VectorStore on app.state for API tests
fastapi_app.state.vector_store = MagicMock()
fastapi_app.state.vector_store.health_check.return_value = True

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

import pytest_asyncio

@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    # Create tables
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    # Drop tables
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)

@pytest_asyncio.fixture
async def async_client():
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

@pytest.fixture
def tmp_upload_dir(tmp_path):
    """Creates a temporary upload directory and patches settings."""
    original_upload_dir = infra_settings.upload_dir
    upload_path = tmp_path / "uploads"
    upload_path.mkdir(parents=True, exist_ok=True)
    infra_settings.upload_dir = str(upload_path)
    yield upload_path
    infra_settings.upload_dir = original_upload_dir
