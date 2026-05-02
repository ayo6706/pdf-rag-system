from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./dev.db"
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 50
    google_api_key: str = ""
    default_llm_model: str = "gemini/gemini-2.5-flash"
    default_embedding_model: str = "gemini/gemini-embedding-001"
    chroma_persist_dir: str = "./chroma_data"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
