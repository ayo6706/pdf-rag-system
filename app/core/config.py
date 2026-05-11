from typing import Optional
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class InfraSettings(BaseSettings):
    # ── Infrastructure ───────────────────────────────────────
    database_url: str
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 50
    chroma_host: str = "chroma"
    chroma_port: int = 8000

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
