from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql://mini_llm:mini_llm_dev@localhost:5432/mini_llm"
    ENABLE_EXACT_CACHE: bool = False
    ENABLE_SEMANTIC_CACHE: bool = False
    CACHE_TTL_SECONDS: int = 3600
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    SEMANTIC_CACHE_THRESHOLD: float = 0.82
    LOG_LEVEL: str = "INFO"
    PROMPT_TEMPLATE_VERSION: str = "v1"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 512

    @field_validator("GROQ_API_KEY")
    @classmethod
    def validate_groq_api_key(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("GROQ_API_KEY must be configured")
        return value.strip()

    @field_validator("CACHE_TTL_SECONDS")
    @classmethod
    def validate_cache_ttl(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("CACHE_TTL_SECONDS must be greater than zero")
        return value

    @field_validator("SEMANTIC_CACHE_THRESHOLD")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("SEMANTIC_CACHE_THRESHOLD must be between 0 and 1")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
