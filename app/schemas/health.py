from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str


class ComponentHealth(BaseModel):
    status: str
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: str
    db: ComponentHealth
    vector_store: ComponentHealth
    llm: ComponentHealth
