"""Custom exceptions and exit codes for IRONCLAD."""

from typing import Optional


class IroncladError(Exception):
    """Base exception for all Ironclad errors."""

    exit_code: int = 1

    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class ConfigurationError(IroncladError):
    """Raised when configuration is missing or invalid."""
    exit_code = 2


class KnowledgeBaseNotFoundError(IroncladError):
    """Raised when a command requires an initialized knowledge base that does not exist."""
    exit_code = 3


class KnowledgeBaseCorruptedError(IroncladError):
    """Raised when the knowledge base SQLite or vector store state is corrupted."""
    exit_code = 4


class ConcurrentAccessError(IroncladError):
    """Raised when another Ironclad process holds the knowledge base lock."""
    exit_code = 5


class IngestionError(IroncladError):
    """Raised when file discovery, parsing, or chunking encounters an unrecoverable error."""
    exit_code = 6


class EmbeddingError(IroncladError):
    """Raised when local embedding generation fails or model is incompatible."""
    exit_code = 7


class InferenceError(IroncladError):
    """Raised when LLM generation fails (credentials, timeouts, rate limits, 5xx)."""
    exit_code = 8


class CitationValidationError(IroncladError):
    """Raised when generated citations cannot be verified against retrieved sources."""
    exit_code = 9


class SecurityError(IroncladError):
    """Raised when a path traversal, symlink loop, or unsafe execution is attempted."""
    exit_code = 10
