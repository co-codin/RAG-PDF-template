"""Application settings. Every value can be overridden via environment variables
(or a local, gitignored ``.env`` file). Backends are selected here, so swapping
infrastructure is a config change, not a code change."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    # --- App -----------------------------------------------------------------
    app_name: str = "rag-production"
    environment: Literal["dev", "test", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True
    api_key: SecretStr | None = None  # when set, all endpoints except /health need it
    api_key_header: str = "X-API-Key"
    cors_origins: list[str] = Field(default_factory=list)

    # --- Local storage (embedded backends) -----------------------------------
    data_dir: Path = Path("./storage")
    chroma_dir: Path | None = None  # default <data_dir>/chroma
    checkpoint_db: Path | None = None  # default <data_dir>/checkpoints.sqlite
    jobs_db: Path | None = None  # default <data_dir>/jobs.sqlite

    # --- Vector store --------------------------------------------------------
    vector_store_backend: Literal["chroma"] = "chroma"
    collection_name: str = "rag_documents"
    # Set CHROMA_HOST to use a Chroma *server* (HttpClient) shared by all workers/replicas.
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_auth_token: SecretStr | None = None
    chroma_connect_retries: int = 10

    # --- Checkpointer (conversation memory) ----------------------------------
    checkpointer_backend: Literal["sqlite", "postgres", "memory"] = "sqlite"
    postgres_url: SecretStr | None = None  # for checkpointer_backend=postgres

    # --- Background jobs -----------------------------------------------------
    # memory: per-process (single worker). sqlite: shared by workers on one host.
    job_store_backend: Literal["memory", "sqlite"] = "memory"
    max_jobs_in_memory: int = 1000

    # --- Ingestion -----------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    max_upload_mb: int = 25
    allowed_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".txt", ".md"])
    url_fetch_timeout_s: float = 20.0
    user_agent: str = "rag-production/0.2 (+https://example.invalid)"
    allow_private_urls: bool = False  # SSRF guard for /ingest/url

    # --- Embeddings ----------------------------------------------------------
    embeddings_provider: Literal["openai", "ollama", "hash"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    hash_embedding_dim: int = 512

    # --- Retrieval -----------------------------------------------------------
    retrieval_k: int = 6
    vector_k: int = 12
    bm25_k: int = 12
    rrf_k: int = 60
    bm25_refresh_s: float = 2.0  # how often a worker checks the store revision
    grader_mode: Literal["llm", "heuristic", "none"] = "llm"
    min_relevant_docs: int = 1

    # --- Generation ----------------------------------------------------------
    llm_provider: Literal["anthropic", "fake"] = "anthropic"
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    llm_temperature: float | None = None  # unset for models that reject non-default values
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    history_max_messages: int = 10
    query_rewrite: bool = True

    @field_validator("api_key", "openai_api_key", "anthropic_api_key", "chroma_auth_token",
                     "postgres_url", "chroma_host", mode="before")
    @classmethod
    def _empty_is_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("allowed_extensions", mode="after")
    @classmethod
    def _normalize_ext(cls, v: list[str]) -> list[str]:
        return [e.lower() if e.startswith(".") else f".{e.lower()}" for e in v]

    @property
    def resolved_chroma_dir(self) -> Path:
        return self.chroma_dir or (self.data_dir / "chroma")

    @property
    def resolved_checkpoint_db(self) -> Path:
        return self.checkpoint_db or (self.data_dir / "checkpoints.sqlite")

    @property
    def resolved_jobs_db(self) -> Path:
        return self.jobs_db or (self.data_dir / "jobs.sqlite")

    @property
    def chroma_mode(self) -> Literal["embedded", "http"]:
        return "http" if self.chroma_host else "embedded"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
