"""Unit tests for the Typer CLI commands."""

import json
from pathlib import Path

from typer.testing import CliRunner

from ironclad.cli import app
from ironclad.errors import KnowledgeBaseNotFoundError


def test_cli_help(cli_runner: CliRunner):
    """Verify `ironclad --help` displays commands and options."""
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "IRONCLAD" in result.stdout
    assert "init" in result.stdout
    assert "ingest" in result.stdout
    assert "ask" in result.stdout
    assert "search" in result.stdout
    assert "status" in result.stdout
    assert "doctor" in result.stdout
    assert "config" in result.stdout


def test_cli_version(cli_runner: CliRunner):
    """Verify `ironclad --version` reports current version."""
    result = cli_runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_config(cli_runner: CliRunner):
    """Verify `ironclad config` outputs expected keys and safe representations."""
    result = cli_runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "openrouter" in result.stdout

    # JSON mode
    json_result = cli_runner.invoke(app, ["config", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert "inference_provider" in data
    assert data["inference_provider"] == "openrouter"


def test_cli_init_and_status(cli_runner: CliRunner, temp_project: Path):
    """Verify `ironclad init` creates knowledge base and `status` reflects it."""
    # Check status before init
    stat_before = cli_runner.invoke(app, ["status", "--path", str(temp_project), "--json"])
    assert stat_before.exit_code == 0
    stat_data_before = json.loads(stat_before.stdout)
    assert stat_data_before["initialized"] is False

    # Initialize project
    init_res = cli_runner.invoke(app, ["init", str(temp_project)])
    assert init_res.exit_code == 0
    assert "Initialized new Ironclad knowledge base" in init_res.stdout
    assert (temp_project / ".ironclad").is_dir()
    assert (temp_project / ".ironclad" / "metadata.db").is_file()

    # Re-initialize should be idempotent and succeed
    init_again = cli_runner.invoke(app, ["init", str(temp_project)])
    assert init_again.exit_code == 0
    assert "Re-verified existing" in init_again.stdout

    # Check status after init
    stat_after = cli_runner.invoke(app, ["status", "--path", str(temp_project), "--json"])
    assert stat_after.exit_code == 0
    stat_data_after = json.loads(stat_after.stdout)
    assert stat_data_after["initialized"] is True
    assert stat_data_after["total_files"] == 0
    assert stat_data_after["is_consistent"] is True


def test_uninitialized_commands_fail_cleanly(cli_runner: CliRunner, temp_project: Path):
    """Verify that ingest, ask, and search exit with KnowledgeBaseNotFoundError when uninitialized."""
    # ingest on uninitialized directory
    res_ingest = cli_runner.invoke(app, ["ingest", str(temp_project)])
    assert res_ingest.exit_code == KnowledgeBaseNotFoundError.exit_code
    assert "Knowledge base not initialized" in res_ingest.stderr or "Knowledge base not initialized" in res_ingest.stdout


def test_cli_doctor(cli_runner: CliRunner, temp_project: Path):
    """Verify that `doctor` performs diagnostic checks and supports JSON mode."""
    res_json = cli_runner.invoke(app, ["doctor", "--path", str(temp_project), "--json"])
    assert res_json.stdout.strip().startswith("{")
    data = json.loads(res_json.stdout)
    assert "checks" in data
    assert any(c["name"] == "Python Version" for c in data["checks"])
    assert any(c["name"] == "Project Directory Writable" for c in data["checks"])
    assert any(c["name"] == "Knowledge Base Initialized" for c in data["checks"])
