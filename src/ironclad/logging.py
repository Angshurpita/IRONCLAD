"""Logging infrastructure for IRONCLAD with credential sanitization."""

import logging
import re
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()
err_console = Console(stderr=True)

# Regex to redact probable keys and authorization tokens
KEY_REDACT_PATTERNS = [
    (re.compile(r"sk-[a-zA-Z0-9_\-]{8,}"), "***REDACTED***"),
    (re.compile(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{8,}", re.IGNORECASE), r"\1***REDACTED***"),
    (re.compile(r"((?:api[_-]?key|password|token)\s*[:=]\s*['\"]?)([^'\"\s]{8,})", re.IGNORECASE), r"\1***REDACTED***"),
]


class CredentialSanitizingFilter(logging.Filter):
    """Ensures no sensitive credentials or keys ever leak into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.sanitize(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self.sanitize(str(v)) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self.sanitize(str(v)) for v in record.args)
        return True

    @staticmethod
    def sanitize(text: str) -> str:
        res = text
        for pat, repl in KEY_REDACT_PATTERNS:
            res = pat.sub(repl, res)
        return res


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    verbose: bool = False,
) -> logging.Logger:
    """Configures project-wide logging with redaction filter and file rotation."""
    level = logging.DEBUG if verbose else getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger("ironclad")
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Filter
    sanitizer = CredentialSanitizingFilter()

    # Rich Console Handler (for warnings/errors or debug if verbose)
    console_handler = RichHandler(
        console=err_console,
        show_time=False,
        show_path=False,
        rich_tracebacks=verbose,
    )
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console_handler.addFilter(sanitizer)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(sanitizer)
            logger.addHandler(file_handler)
        except OSError:
            # If unable to write to log file, continue without file logging
            pass

    return logger
