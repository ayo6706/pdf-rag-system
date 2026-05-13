from typing import Optional
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class InfraSettings(BaseSettings):
    # ── Infrastructure ───────────────────────────────────────
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_url: str
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 50
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    redis_url: str = "redis://:REPLACE_ME@redis:6379/0"
    ingestion_queue_name: str = "ingestion"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class LLMSettings(BaseSettings):
    # ── LLM / Embedding (provider-agnostic via LiteLLM) ──────
    default_llm_model: str = "gemini/gemini-2.5-flash"
    default_embedding_model: str = "gemini/gemini-embedding-001"

    google_api_key: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class RAGSettings(BaseSettings):
    # ── RAG ──────────────────────────────────────────────────
    confidence_threshold: float = 0.3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


infra_settings = InfraSettings()
llm_settings = LLMSettings()
rag_settings = RAGSettings()
