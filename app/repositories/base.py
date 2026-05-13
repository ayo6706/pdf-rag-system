"""Base repository providing common CRUD operations.

This module defines the BaseRepository class, which uses generics to provide
reusable database logic for any SQLModel-based entity.
"""

from __future__ import annotations

from typing import Generic, TypeVar, Type, Optional, List, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, select
from pydantic import BaseModel

ModelType = TypeVar("ModelType", bound=SQLModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base repository for CRUD operations."""

    def __init__(self, model: Type[ModelType]):
        """Initialize the repository.

        Args:
            model: The SQLModel class to be used by this repository.
        """
        self.model = model

    async def get(self, db: AsyncSession, id: Any) -> Optional[ModelType]:
        """Get a single record by ID.

        Args:
            db: The database session.
            id: The primary key of the record.

        Returns:
            The found record or None if it doesn't exist.
        """
        result = await db.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalars().first()

    async def get_multi(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Get multiple records with deterministic ordering.

        Args:
            db: The database session.
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            A list of records.
        """
        result = await db.execute(
            select(self.model).order_by(self.model.id).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def create(
        self,
        db: AsyncSession,
        obj_in: Union[CreateSchemaType, dict]
    ) -> ModelType:
        """Create a new record.

        Args:
            db: The database session.
            obj_in: The data to create the record with (schema or dict).

        Returns:
            The newly created record.
        """
        if isinstance(obj_in, dict):
            create_data = obj_in
        elif hasattr(obj_in, "model_dump"):
            create_data = obj_in.model_dump()
        else:
            create_data = obj_in.dict()

        db_obj = self.model(**create_data)
        db.add(db_obj)
        await db.flush()
        await db.refresh(db_obj)
        return db_obj

    async def update(
        self,
        db: AsyncSession,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, dict]
    ) -> ModelType:
        """Update an existing record.

        Note:
            This method only flushes the session (via `await db.flush()`) and
            refreshes the object (`await db.refresh(db_obj)`). It does not
            commit the transaction. Callers must call `db.commit()` or manage
            transactions externally.

        Args:
            db: The database session.
            db_obj: The existing database object to update.
            obj_in: The updated data (schema or dict).

        Returns:
            The updated record.
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        elif hasattr(obj_in, "model_dump"):
            update_data = obj_in.model_dump(exclude_unset=True)
        else:
            update_data = obj_in.dict(exclude_unset=True)

        for field, value in update_data.items():
            setattr(db_obj, field, value)

        await db.flush()
        await db.refresh(db_obj)
        return db_obj

    async def delete(self, db: AsyncSession, id: Any) -> bool:
        """Delete a record by ID.

        Args:
            db: The database session.
            id: The ID of the record to delete.

        Returns:
            True if deleted, False if not found.
        """
        obj = await self.get(db, id)
        if obj:
            await db.delete(obj)
            await db.flush()
            return True
        return False
