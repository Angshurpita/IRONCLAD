"""File discovery engine with security protections, ignore rules, and type classification."""

import fnmatch
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from ironclad.config import IroncladSettings
from ironclad.models import DocumentType

logger = logging.getLogger("ironclad.discovery")

# Mapping of file extensions to DocumentType
EXTENSION_MAP: Dict[str, DocumentType] = {
    # Source Code
    ".py": DocumentType.SOURCE_CODE,
    ".js": DocumentType.SOURCE_CODE,
    ".jsx": DocumentType.SOURCE_CODE,
    ".ts": DocumentType.SOURCE_CODE,
    ".tsx": DocumentType.SOURCE_CODE,
    ".java": DocumentType.SOURCE_CODE,
    ".c": DocumentType.SOURCE_CODE,
    ".cpp": DocumentType.SOURCE_CODE,
    ".cc": DocumentType.SOURCE_CODE,
    ".cxx": DocumentType.SOURCE_CODE,
    ".h": DocumentType.SOURCE_CODE,
    ".hpp": DocumentType.SOURCE_CODE,
    ".cs": DocumentType.SOURCE_CODE,
    ".go": DocumentType.SOURCE_CODE,
    ".rs": DocumentType.SOURCE_CODE,
    ".rb": DocumentType.SOURCE_CODE,
    ".php": DocumentType.SOURCE_CODE,
    ".sh": DocumentType.SOURCE_CODE,
    ".bash": DocumentType.SOURCE_CODE,
    ".zsh": DocumentType.SOURCE_CODE,
    ".swift": DocumentType.SOURCE_CODE,
    ".kt": DocumentType.SOURCE_CODE,
    ".scala": DocumentType.SOURCE_CODE,
    ".sql": DocumentType.SOURCE_CODE,
    # Markdown & Documentation
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".mdown": DocumentType.MARKDOWN,
    # Plain Text
    ".txt": DocumentType.PLAIN_TEXT,
    ".text": DocumentType.PLAIN_TEXT,
    ".rst": DocumentType.PLAIN_TEXT,
    # PDF
    ".pdf": DocumentType.PDF,
    # Configurations & Structured Data
    ".json": DocumentType.CONFIG,
    ".yaml": DocumentType.CONFIG,
    ".yml": DocumentType.CONFIG,
    ".toml": DocumentType.CONFIG,
    ".ini": DocumentType.CONFIG,
    ".cfg": DocumentType.CONFIG,
    ".xml": DocumentType.CONFIG,
    ".env.example": DocumentType.CONFIG,
    ".csv": DocumentType.CONFIG,
}

# Special filenames without extensions
SPECIAL_FILENAMES: Dict[str, DocumentType] = {
    "dockerfile": DocumentType.CONFIG,
    "makefile": DocumentType.CONFIG,
    "gemfile": DocumentType.CONFIG,
    "procfile": DocumentType.CONFIG,
}


@dataclass
class DiscoveredFile:
    """Represents a discovered candidate file ready for ingestion."""

    path: Path
    rel_path: str
    file_hash: str
    size_bytes: int
    mtime: float
    doc_type: DocumentType


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash in 64KB blocks for memory safety."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def classify_file(path: Path) -> Optional[DocumentType]:
    """Determine DocumentType from file extension or base name."""
    name_lower = path.name.lower()
    if name_lower in SPECIAL_FILENAMES:
        return SPECIAL_FILENAMES[name_lower]

    suffix = path.suffix.lower()
    if suffix in EXTENSION_MAP:
        return EXTENSION_MAP[suffix]

    # Handle compound extensions like .env.example
    for ext, doc_type in EXTENSION_MAP.items():
        if name_lower.endswith(ext):
            return doc_type

    return None


def is_binary_file(path: Path) -> bool:
    """Heuristic check for binary content to avoid indexing binary non-PDFs."""
    if path.suffix.lower() == ".pdf":
        return False
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            # Null bytes strongly indicate binary files
            if b"\x00" in chunk:
                return True
    except OSError:
        return True
    return False


def load_ignore_patterns(root_dir: Path) -> List[str]:
    """Load exclusion patterns from .gitignore and .ironcladignore files."""
    patterns: List[str] = []
    for ignore_filename in [".gitignore", ".ironcladignore"]:
        ignore_file = root_dir / ignore_filename
        if ignore_file.is_file():
            try:
                with open(ignore_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except OSError as e:
                logger.warning(f"Could not read ignore file {ignore_file}: {e}")
    return patterns


def should_ignore(
    rel_path: str,
    patterns: List[str],
) -> bool:
    """Check if a relative path matches any ignore pattern."""
    parts = Path(rel_path).parts
    norm_path = rel_path.replace("\\", "/")

    for pattern in patterns:
        cleaned = pattern.rstrip("/")
        # Direct part match (e.g. node_modules, .git)
        for part in parts:
            if fnmatch.fnmatch(part, cleaned):
                return True
        # Path match (e.g. build/* or src/*.log)
        if fnmatch.fnmatch(norm_path, cleaned) or fnmatch.fnmatch(norm_path, pattern):
            return True
        # Directory prefix match
        if pattern.endswith("/") and (norm_path.startswith(cleaned + "/") or norm_path == cleaned):
            return True
    return False


def discover_files(
    root_dir: Path,
    settings: IroncladSettings,
    extra_exclusions: Optional[List[str]] = None,
) -> List[DiscoveredFile]:
    """
    Safely discover, filter, and classify files in target directory.
    Enforces path traversal safety, symlink loop protection, size bounds, and ignore rules.
    """
    resolved_root = root_dir.resolve()
    if not resolved_root.is_dir():
        logger.error(f"Discovery root {root_dir} is not a directory.")
        return []

    # Combine ignore patterns
    all_patterns = list(settings.default_ignore_patterns)
    all_patterns.extend(load_ignore_patterns(resolved_root))
    if extra_exclusions:
        all_patterns.extend(extra_exclusions)

    discovered: List[DiscoveredFile] = []
    visited_dirs: Set[str] = set()

    for dirpath, dirnames, filenames in os.walk(str(resolved_root), followlinks=False):
        current_dir = Path(dirpath).resolve()

        # Symlink loop protection
        dir_real = os.path.realpath(dirpath)
        if dir_real in visited_dirs:
            dirnames.clear()
            continue
        visited_dirs.add(dir_real)

        # Path traversal guard: must be strictly within root
        if not str(current_dir).startswith(str(resolved_root)):
            logger.warning(f"Skipping directory outside project root: {current_dir}")
            dirnames.clear()
            continue

        # Filter out directories to prune search tree early
        rel_dir = os.path.relpath(dirpath, str(resolved_root))
        if rel_dir != ".":
            if should_ignore(rel_dir, all_patterns):
                dirnames.clear()
                continue

        # Prune subdirectories in-place based on ignore patterns
        dirnames[:] = [
            d
            for d in dirnames
            if not should_ignore(
                os.path.relpath(Path(dirpath) / d, str(resolved_root)), all_patterns
            )
        ]

        for filename in filenames:
            file_path = Path(dirpath) / filename

            # Skip symlinks to prevent loops or pointing outside workspace
            if file_path.is_symlink():
                resolved_target = file_path.resolve()
                if not str(resolved_target).startswith(str(resolved_root)):
                    logger.debug(f"Skipping symlink pointing outside root: {file_path}")
                    continue

            rel_file_path = os.path.relpath(str(file_path), str(resolved_root))

            # Check ignore rules
            if should_ignore(rel_file_path, all_patterns):
                continue

            # Classify document type
            doc_type = classify_file(file_path)
            if not doc_type:
                continue

            try:
                stat = file_path.stat()
            except OSError as e:
                logger.warning(f"Cannot stat file {file_path}: {e}")
                continue

            # Size limits check
            if stat.st_size == 0:
                logger.debug(f"Skipping empty file: {rel_file_path}")
                continue
            if stat.st_size > settings.max_file_size_bytes:
                logger.warning(
                    f"Skipping file exceeding limit ({stat.st_size} > {settings.max_file_size_bytes}): {rel_file_path}"
                )
                continue

            # Heuristic binary check
            if is_binary_file(file_path):
                logger.debug(f"Skipping binary non-PDF file: {rel_file_path}")
                continue

            try:
                f_hash = compute_sha256(file_path)
            except OSError as e:
                logger.warning(f"Could not hash file {file_path}: {e}")
                continue

            discovered.append(
                DiscoveredFile(
                    path=file_path,
                    rel_path=rel_file_path,
                    file_hash=f_hash,
                    size_bytes=stat.st_size,
                    mtime=stat.st_mtime,
                    doc_type=doc_type,
                )
            )

    return discovered
