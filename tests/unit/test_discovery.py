"""Unit tests for file discovery, ignore rules, and security protections."""

from pathlib import Path

from ironclad.config import IroncladSettings
from ironclad.ingestion.discovery import (
    classify_file,
    compute_sha256,
    discover_files,
    is_binary_file,
    load_ignore_patterns,
    should_ignore,
)
from ironclad.models import DocumentType


def test_classify_file():
    """Verify correct classification of various extensions and special filenames."""
    assert classify_file(Path("main.py")) == DocumentType.SOURCE_CODE
    assert classify_file(Path("app.tsx")) == DocumentType.SOURCE_CODE
    assert classify_file(Path("Server.java")) == DocumentType.SOURCE_CODE
    assert classify_file(Path("README.md")) == DocumentType.MARKDOWN
    assert classify_file(Path("notes.txt")) == DocumentType.PLAIN_TEXT
    assert classify_file(Path("manual.pdf")) == DocumentType.PDF
    assert classify_file(Path("config.yaml")) == DocumentType.CONFIG
    assert classify_file(Path("settings.json")) == DocumentType.CONFIG
    assert classify_file(Path("Dockerfile")) == DocumentType.CONFIG
    assert classify_file(Path("unknown.xyz123")) is None


def test_binary_detection(temp_project: Path):
    """Verify binary files with null bytes are detected."""
    txt_file = temp_project / "clean.txt"
    txt_file.write_text("Hello World", encoding="utf-8")
    assert not is_binary_file(txt_file)

    bin_file = temp_project / "binary.dat"
    bin_file.write_bytes(b"\x00\x01\x02\x03\x04")
    assert is_binary_file(bin_file)


def test_compute_sha256(temp_project: Path):
    """Verify deterministic SHA-256 computation."""
    f = temp_project / "sample.txt"
    f.write_text("Ironclad Test Data", encoding="utf-8")
    hash1 = compute_sha256(f)
    hash2 = compute_sha256(f)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_ignore_rules(temp_project: Path):
    """Verify ignore pattern matching and .gitignore file loading."""
    gitignore = temp_project / ".gitignore"
    gitignore.write_text("*.log\nsecret_dir/\nbuild\n", encoding="utf-8")

    patterns = load_ignore_patterns(temp_project)
    assert "*.log" in patterns
    assert "secret_dir/" in patterns
    assert "build" in patterns

    assert should_ignore("app.log", patterns)
    assert should_ignore("logs/debug.log", patterns)
    assert should_ignore("secret_dir/data.txt", patterns)
    assert should_ignore("build/output.js", patterns)
    assert not should_ignore("src/main.py", patterns)


def test_discover_files_filters_and_limits(temp_project: Path):
    """Verify file discovery respects ignore patterns, size limits, and empty files."""
    settings = IroncladSettings(
        project_root=temp_project,
        max_file_size_bytes=1000,
    )

    # Valid file
    valid = temp_project / "main.py"
    valid.write_text("print('hello')", encoding="utf-8")

    # Empty file (should be skipped)
    empty = temp_project / "empty.py"
    empty.write_text("", encoding="utf-8")

    # Oversized file (should be skipped)
    big = temp_project / "big.txt"
    big.write_text("A" * 2000, encoding="utf-8")

    # Ignored directory
    git_dir = temp_project / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config", encoding="utf-8")

    files = discover_files(temp_project, settings)
    rel_paths = [f.rel_path for f in files]

    assert "main.py" in rel_paths
    assert "empty.py" not in rel_paths
    assert "big.txt" not in rel_paths
    assert ".git/config" not in rel_paths
