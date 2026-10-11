"""Integration tests for end-to-end ingestion, idempotence, and incremental updates."""

import json
from pathlib import Path

from typer.testing import CliRunner

from ironclad.cli import app


def test_full_ingestion_and_status(cli_runner: CliRunner, temp_project: Path):
    """Verify end-to-end init -> ingest -> status pipeline."""
    # Create sample project files
    src_dir = temp_project / "src"
    src_dir.mkdir()

    (src_dir / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n"
        "    return a - b\n",
        encoding="utf-8",
    )

    (temp_project / "README.md").write_text(
        "# Calculator Project\n"
        "Simple arithmetic library.\n\n"
        "## Usage\n"
        "Import calculator and call functions.\n",
        encoding="utf-8",
    )

    (temp_project / "config.yaml").write_text(
        "version: 1.0.0\nenvironment: production\n",
        encoding="utf-8",
    )

    # 1. Initialize
    init_res = cli_runner.invoke(app, ["init", str(temp_project)])
    assert init_res.exit_code == 0

    # 2. Ingest
    ingest_res = cli_runner.invoke(app, ["ingest", str(temp_project), "--json"])
    assert ingest_res.exit_code == 0
    ingest_data = json.loads(ingest_res.stdout)
    assert ingest_data["files_added"] == 3
    assert ingest_data["files_skipped"] == 0
    assert ingest_data["total_files"] == 3
    assert ingest_data["total_chunks"] >= 3

    # 3. Check status
    stat_res = cli_runner.invoke(app, ["status", "--path", str(temp_project), "--json"])
    assert stat_res.exit_code == 0
    stat_data = json.loads(stat_res.stdout)
    assert stat_data["initialized"] is True
    assert stat_data["total_files"] == 3
    assert stat_data["total_chunks"] == ingest_data["total_chunks"]
    assert stat_data["is_consistent"] is True

    # 4. Idempotent re-run: nothing changed, so all files must be skipped!
    re_ingest = cli_runner.invoke(app, ["ingest", str(temp_project), "--json"])
    assert re_ingest.exit_code == 0
    re_data = json.loads(re_ingest.stdout)
    assert re_data["files_added"] == 0
    assert re_data["files_updated"] == 0
    assert re_data["files_skipped"] == 3
    assert re_data["total_files"] == 3


def test_incremental_updates_and_deletions(cli_runner: CliRunner, temp_project: Path):
    """Verify incremental updates correctly handle modified and deleted files."""
    # Setup initial files
    doc1 = temp_project / "doc1.txt"
    doc1.write_text("Initial content for doc 1", encoding="utf-8")
    doc2 = temp_project / "doc2.txt"
    doc2.write_text("Initial content for doc 2", encoding="utf-8")

    cli_runner.invoke(app, ["init", str(temp_project)])
    cli_runner.invoke(app, ["ingest", str(temp_project)])

    # Modify doc1, keep doc2 unchanged, add doc3, delete nothing yet
    doc1.write_text("Updated content for doc 1 with extra details", encoding="utf-8")
    doc3 = temp_project / "doc3.txt"
    doc3.write_text("Brand new doc 3 content", encoding="utf-8")

    update_res = cli_runner.invoke(app, ["ingest", str(temp_project), "--update", "--json"])
    assert update_res.exit_code == 0
    up_data = json.loads(update_res.stdout)
    assert up_data["files_added"] == 1
    assert up_data["files_updated"] == 1
    assert up_data["files_skipped"] == 1  # doc2 unchanged
    assert up_data["total_files"] == 3

    # Now delete doc2
    doc2.unlink()
    del_res = cli_runner.invoke(app, ["ingest", str(temp_project), "--update", "--json"])
    assert del_res.exit_code == 0
    del_data = json.loads(del_res.stdout)
    assert del_data["files_deleted"] == 1
    assert del_data["total_files"] == 2
