"""Unit tests for domain exceptions and standardized exit codes."""

from ironclad.errors import (
    CitationValidationError,
    ConcurrentAccessError,
    ConfigurationError,
    EmbeddingError,
    InferenceError,
    IngestionError,
    IroncladError,
    KnowledgeBaseCorruptedError,
    KnowledgeBaseNotFoundError,
    SecurityError,
)


def test_exception_exit_codes():
    """Verify that every domain exception has an assigned non-zero exit code."""
    assert IroncladError.exit_code == 1
    assert ConfigurationError.exit_code == 2
    assert KnowledgeBaseNotFoundError.exit_code == 3
    assert KnowledgeBaseCorruptedError.exit_code == 4
    assert ConcurrentAccessError.exit_code == 5
    assert IngestionError.exit_code == 6
    assert EmbeddingError.exit_code == 7
    assert InferenceError.exit_code == 8
    assert CitationValidationError.exit_code == 9
    assert SecurityError.exit_code == 10


def test_exception_formatting():
    """Verify error string formatting with and without actionable hints."""
    err_without_hint = IroncladError("Something broke")
    assert str(err_without_hint) == "Something broke"

    err_with_hint = ConfigurationError("Missing key", hint="Set OPENROUTER_API_KEY in .env")
    assert "Missing key" in str(err_with_hint)
    assert "Hint: Set OPENROUTER_API_KEY in .env" in str(err_with_hint)
