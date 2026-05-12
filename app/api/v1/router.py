from fastapi import APIRouter
from app.api.v1.endpoints import documents, query, health

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(query.router)
