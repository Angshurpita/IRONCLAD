"""Phase 5 resilience, incremental updates, reindexing, and diagnostics tests."""

import json
from pathlib import Path

from typer.testing import CliRunner

from ironclad.cli import app
from ironclad.config import IroncladSettings
from ironclad.services.ingest import run_ingestion
from ironclad.services.status import get_status, run_diagnostics
from ironclad.storage.metadata import MetadataStore


def test_reindex_flag_wipes_and_rebuilds(cli_runner: CliRunner, temp_project: Path):
    """Verify `ironclad ingest --reindex` cleans up existing records and rebuilds cleanly."""
    file_a = temp_project / "file_a.py"
    file_a.write_text("def feature_a(): return 'alpha'\n", encoding="utf-8")

    cli_runner.invoke(app, ["init", str(temp_project)])
    cli_runner.invoke(app, ["ingest", str(temp_project)])

    settings = IroncladSettings(project_root=temp_project)
    meta_store = MetadataStore(settings.metadata_db_path)
    assert len(meta_store.get_indexed_documents()) == 1

    # Add file B, delete file A directly from disk
    file_a.unlink()
    file_b = temp_project / "file_b.py"
    file_b.write_text("def feature_b(): return 'beta'\n", encoding="utf-8")

    res = cli_runner.invoke(app, ["ingest", str(temp_project), "--reindex", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["mode"] == "reindex"
    assert data["total_files"] == 1

    docs = meta_store.get_indexed_documents()
    assert "file_b.py" in docs
    assert "file_a.py" not in docs


def test_embedding_model_change_triggers_reindex(cli_runner: CliRunner, temp_project: Path):
    """Verify that changing configured embedding model triggers re-indexing for dimensional consistency."""
    (temp_project / "module.py").write_text("def calculate(): pass\n", encoding="utf-8")

    cli_runner.invoke(app, ["init", str(temp_project)])
    settings_1 = IroncladSettings(project_root=temp_project, embedding_model="all-MiniLM-L6-v2")
    summary_1 = run_ingestion(settings_1, target_path=temp_project)
    assert summary_1.total_files == 1

    # Check status consistency
    status_1 = get_status(settings_1)
    assert status_1.is_consistent is True

    # Switch embedding model configuration
    settings_2 = IroncladSettings(project_root=temp_project, embedding_model="hash-projector-384")
    
    # Inconsistency is reported by status before ingest
    status_2 = get_status(settings_2)
    assert status_2.is_consistent is False
    assert any("differs from indexed" in err for err in status_2.inconsistencies)

    # Ingestion automatically handles model change and updates metadata
    summary_2 = run_ingestion(settings_2, target_path=temp_project)
    assert summary_2.mode == "reindex"

    status_3 = get_status(settings_2)
    assert status_3.is_consistent is True


def test_doctor_diagnostics_comprehensive(cli_runner: CliRunner, temp_project: Path):
    """Verify `ironclad doctor` returns complete report across all system categories."""
    cli_runner.invoke(app, ["init", str(temp_project)])
    cli_runner.invoke(app, ["ingest", str(temp_project)])

    settings = IroncladSettings(project_root=temp_project)
    report = run_diagnostics(settings)

    check_names = [c.name for c in report.checks]
    assert "Python Version" in check_names
    assert "Project Directory Writable" in check_names
    assert "Knowledge Base Initialized" in check_names
    assert "Metadata Database Integrity" in check_names
    assert "SQLite FTS5 Engine" in check_names
    assert "Vector Store Health" in check_names
    assert "Process Lock File" in check_names
    assert "Local Embedding Model Runtime" in check_names

    fts_check = next(c for c in report.checks if c.name == "SQLite FTS5 Engine")
    assert fts_check.passed is True

    vstore_check = next(c for c in report.checks if c.name == "Vector Store Health")
    assert vstore_check.passed is True


def test_edge_cases_empty_binary_unicode(cli_runner: CliRunner, temp_project: Path):
    """Verify ingestion gracefully handles zero-byte files, nested directories, unicode, and binary files."""
    # 1. Zero-byte file
    (temp_project / "empty.py").write_text("", encoding="utf-8")

    # 2. Binary file (must be skipped by discovery)
    (temp_project / "data.bin").write_bytes(b"\x00\x01\x02\xff\xfe\x00")

    # 3. Unicode and emoji in source code
    nested_dir = temp_project / "deep" / "nested" / "path"
    nested_dir.mkdir(parents=True)
    (nested_dir / "unicode_app.py").write_text(
        "# Unicode symbols: λ π Ω 日本語 🚀\n"
        "def greet(name: str) -> str:\n"
        "    return f'Hello, {name}! 🌍'\n",
        encoding="utf-8",
    )

    cli_runner.invoke(app, ["init", str(temp_project)])
    res = cli_runner.invoke(app, ["ingest", str(temp_project), "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)

    assert data["total_files"] == 1  # unicode_app.py indexed; empty.py and data.bin filtered
    assert data["errors"] == []


def test_stale_lock_cleanup_on_ingest(cli_runner: CliRunner, temp_project: Path):
    """Verify ingestion proceeds when an abandoned stale lock file is present."""
    cli_runner.invoke(app, ["init", str(temp_project)])
    (temp_project / "code.py").write_text("print('hello')\n", encoding="utf-8")

    settings = IroncladSettings(project_root=temp_project)
    lock_file = settings.lock_file_path

    # Simulate abandoned lock from a dead PID (e.g., PID 99999999)
    lock_file.write_text(
        json.dumps({"pid": 99999999, "timestamp": "2020-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    assert lock_file.exists()

    # Ingestion should detect staleness, remove it, and complete
    res = cli_runner.invoke(app, ["ingest", str(temp_project), "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["total_files"] == 1
