from typing import Annotated, AsyncGenerator
from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_session_maker
from app.repositories.vector_store import VectorStore

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_maker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

DbSession = Annotated[AsyncSession, Depends(get_db)]

async def get_vector_store(request: Request) -> VectorStore:
    """Dependency to retrieve the vector store from app state."""
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Please try again later.",
        )
    return store

VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
