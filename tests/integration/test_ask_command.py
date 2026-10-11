"""Integration tests for `ironclad ask` command."""

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from ironclad.cli import app
from ironclad.generation.provider import MockGemmaProvider


def test_cli_ask_grounded_answer(cli_runner: CliRunner, temp_project: Path):
    """Verify `ironclad ask` retrieves evidence, generates answer, and verifies citations."""
    # Setup test file
    (temp_project / "auth.py").write_text(
        "def authenticate_user(username: str, password_hash: str) -> bool:\n"
        "    '''Validates user password hash against database record.'''\n"
        "    return password_hash == get_stored_hash(username)\n",
        encoding="utf-8",
    )

    cli_runner.invoke(app, ["init", str(temp_project)])
    cli_runner.invoke(app, ["ingest", str(temp_project)])

    mock_llm = MockGemmaProvider(
        canned_answer="User authentication compares the supplied password hash with the stored hash [1].",
        citations=[
            {
                "source_index": 1,
                "file_path": "auth.py",
                "start_line": 1,
                "end_line": 3,
                "excerpt": "return password_hash == get_stored_hash(username)",
            }
        ],
        insufficient_evidence=False,
    )

    with patch("ironclad.generation.provider.get_inference_provider", return_value=mock_llm):
        # 1. Human CLI mode
        res = cli_runner.invoke(
            app,
            ["ask", "How does user authentication work?", "-p", str(temp_project)],
        )
        assert res.exit_code == 0
        assert "IRONCLAD Answer" in res.stdout
        assert "compares the supplied password hash" in res.stdout
        assert "Verified Ground-Truth Citations" in res.stdout
        assert "auth.py" in res.stdout

        # 2. JSON mode
        res_json = cli_runner.invoke(
            app,
            ["ask", "How does user authentication work?", "-p", str(temp_project), "--json"],
        )
        assert res_json.exit_code == 0
        data = json.loads(res_json.stdout)
        assert data["insufficient_evidence"] is False
        assert len(data["citations"]) == 1
        assert data["citations"][0]["file_path"] == "auth.py"
        assert data["citations"][0]["start_line"] == 1


def test_cli_ask_abstention(cli_runner: CliRunner, temp_project: Path):
    """Verify `ironclad ask` properly abstains when evidence is insufficient."""
    (temp_project / "math_utils.py").write_text(
        "def add(a, b): return a + b\n",
        encoding="utf-8",
    )

    cli_runner.invoke(app, ["init", str(temp_project)])
    cli_runner.invoke(app, ["ingest", str(temp_project)])

    mock_llm = MockGemmaProvider(
        canned_answer="The knowledge base does not contain any information about database replication.",
        citations=[],
        insufficient_evidence=True,
    )

    with patch("ironclad.generation.provider.get_inference_provider", return_value=mock_llm):
        res = cli_runner.invoke(
            app,
            ["ask", "How is database replication configured?", "-p", str(temp_project), "--json"],
        )
        assert res.exit_code == 0
        data = json.loads(res.stdout)
        assert data["insufficient_evidence"] is True
