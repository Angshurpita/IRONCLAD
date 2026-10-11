"""Configuration management for IRONCLAD with secure credential handling."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def mask_secret(secret: Optional[SecretStr | str]) -> str:
    """Safely mask a secret for display or logging without exposing credentials."""
    if not secret:
        return "(not set)"
    raw = secret.get_secret_value() if isinstance(secret, SecretStr) else str(secret)
    if not raw or raw.strip() == "":
        return "(not set)"
    if len(raw) <= 8:
        return "****"
    return f"{raw[:4]}...{raw[-4:]}"


class IroncladSettings(BaseSettings):
    """Central configuration settings for IRONCLAD."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Core Directories & Metadata
    ironclad_dir_name: str = Field(default=".ironclad", validation_alias="IRONCLAD_DIR")
    project_root: Path = Field(default_factory=Path.cwd)

    # Inference Provider: 'openrouter' or 'ollama'
    inference_provider: str = Field(default="openrouter", validation_alias="IRONCLAD_INFERENCE_PROVIDER")

    # Hosted Inference (OpenRouter / OpenAI-compatible)
    openrouter_api_key: Optional[SecretStr] = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(
        default="google/gemma-4-26b-a4b", validation_alias="OPENROUTER_MODEL"
    )

    # Local Inference (Ollama)
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="gemma4:26b", validation_alias="OLLAMA_MODEL")

    # Local Embedding Settings
    embedding_provider: str = Field(default="local", validation_alias="IRONCLAD_EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", validation_alias="IRONCLAD_EMBEDDING_MODEL"
    )

    # Ingestion & Chunking Limits
    max_file_size_bytes: int = Field(default=10 * 1024 * 1024, validation_alias="IRONCLAD_MAX_FILE_SIZE_BYTES")  # 10 MB
    chunk_size: int = Field(default=500, validation_alias="IRONCLAD_CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, validation_alias="IRONCLAD_CHUNK_OVERLAP")
    max_context_tokens: int = Field(default=4000, validation_alias="IRONCLAD_MAX_CONTEXT_TOKENS")

    # Ignored directories and files (by default)
    default_ignore_patterns: List[str] = Field(
        default_factory=lambda: [
            ".git",
            ".svn",
            ".hg",
            ".ironclad",
            ".venv",
            "venv",
            "__pycache__",
            "*.pyc",
            "node_modules",
            "dist",
            "build",
            ".eggs",
            "*.egg-info",
            ".DS_Store",
            ".env",
            "*.pem",
            "*.key",
            "*.secret",
        ]
    )

    # Reliability & Logging
    log_level: str = Field(default="INFO", validation_alias="IRONCLAD_LOG_LEVEL")
    http_timeout_seconds: float = Field(default=60.0, validation_alias="IRONCLAD_HTTP_TIMEOUT")
    max_retries: int = Field(default=3, validation_alias="IRONCLAD_MAX_RETRIES")

    @property
    def ironclad_path(self) -> Path:
        """Returns absolute path to the target .ironclad knowledge base directory."""
        return (self.project_root / self.ironclad_dir_name).resolve()

    @property
    def metadata_db_path(self) -> Path:
        """Returns path to the transactional SQLite metadata database."""
        return self.ironclad_path / "metadata.db"

    @property
    def vector_store_path(self) -> Path:
        """Returns path to the embedded vector store."""
        return self.ironclad_path / "vectors"

    @property
    def lock_file_path(self) -> Path:
        """Returns path to the concurrency process lock file."""
        return self.ironclad_path / "ironclad.lock"

    @property
    def log_file_path(self) -> Path:
        """Returns path to the persistent debug log file."""
        return self.ironclad_path / "ironclad.log"

    def is_initialized(self) -> bool:
        """Check if target project has an initialized .ironclad directory and metadata DB."""
        return self.ironclad_path.is_dir() and self.metadata_db_path.is_file()

    def safe_dict(self) -> Dict[str, Any]:
        """Produce a dictionary representation with sensitive values masked."""
        return {
            "project_root": str(self.project_root),
            "ironclad_path": str(self.ironclad_path),
            "inference_provider": self.inference_provider,
            "openrouter_base_url": self.openrouter_base_url,
            "openrouter_model": self.openrouter_model,
            "openrouter_api_key_set": bool(self.openrouter_api_key),
            "openrouter_api_key_masked": mask_secret(self.openrouter_api_key),
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "max_file_size_bytes": self.max_file_size_bytes,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_context_tokens": self.max_context_tokens,
            "log_level": self.log_level,
            "is_initialized": self.is_initialized(),
        }


def get_settings(project_path: Optional[Path] = None) -> IroncladSettings:
    """
    Resolve and return IroncladSettings for a given path.
    If project_path is None, checks current directory or ancestors for .ironclad.
    """
    target = (project_path or Path.cwd()).resolve()

    # If the user pointed at a directory, find project root
    if not target.is_dir() and target.parent.is_dir():
        target = target.parent

    # Traverse upward to check if an existing .ironclad directory is present
    current = target
    detected_root = target
    for parent in [current] + list(current.parents):
        if (parent / ".ironclad").is_dir():
            detected_root = parent
            break

    # Look for .env file in detected root or current dir
    env_file = detected_root / ".env"
    if not env_file.exists():
        env_file = Path.cwd() / ".env"

    if env_file.exists():
        return IroncladSettings(
            project_root=detected_root,
            _env_file=str(env_file),  # type: ignore
        )
    return IroncladSettings(project_root=detected_root)
