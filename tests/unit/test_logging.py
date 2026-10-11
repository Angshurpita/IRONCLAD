"""Unit tests for logging and credential sanitization filter."""

import logging

from ironclad.logging import CredentialSanitizingFilter, setup_logging


def test_credential_sanitizer_filter():
    """Verify that regex patterns reliably scrub API keys and bearer tokens."""
    sanitizer = CredentialSanitizingFilter()

    # Redacting sk-... style keys
    text = "Sending request with authorization: sk-or-v1-abcdef1234567890"
    cleaned = sanitizer.sanitize(text)
    assert "sk-or-v1-abcdef1234567890" not in cleaned
    assert "REDACTED" in cleaned

    # Redacting Bearer tokens
    text2 = "Header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    cleaned2 = sanitizer.sanitize(text2)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in cleaned2
    assert "REDACTED" in cleaned2


def test_logger_setup():
    """Verify setup_logging returns a configured logger instance."""
    logger = setup_logging(log_level="DEBUG")
    assert logger.name == "ironclad"
    assert logger.level == logging.DEBUG
