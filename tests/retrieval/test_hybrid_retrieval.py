"""Retrieval recall evaluation on a multi-file test corpus."""

import json
from pathlib import Path

from typer.testing import CliRunner

from ironclad.cli import app
from ironclad.config import IroncladSettings
from ironclad.services.query import execute_hybrid_search


def setup_evaluation_corpus(project_dir: Path) -> None:
    """Populate an evaluation corpus covering source code and documentation."""
    # 1. Auth module
    auth_dir = project_dir / "auth"
    auth_dir.mkdir(parents=True)
    (auth_dir / "jwt.py").write_text(
        "import jwt\n"
        "import datetime\n\n"
        "SECRET_KEY = 'secret-key'\n\n"
        "def create_access_token(user_id: int) -> str:\n"
        "    payload = {'sub': user_id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)}\n"
        "    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')\n\n"
        "def validate_jwt_token(token: str) -> dict:\n"
        "    '''Verifies HMAC-SHA256 signature and decodes bearer token.'''\n"
        "    try:\n"
        "        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\n"
        "        return payload\n"
        "    except jwt.PyJWTError as err:\n"
        "        raise ValueError(f'Invalid JWT token: {err}')\n",
        encoding="utf-8",
    )

    # 2. Database module
    db_dir = project_dir / "db"
    db_dir.mkdir(parents=True)
    (db_dir / "models.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class UserModel:\n"
        "    id: int\n"
        "    username: str\n"
        "    email: str\n\n"
        "def run_database_migrations():\n"
        "    '''Applies pending schema migrations to the PostgreSQL database cluster.'''\n"
        "    print('Applying migration 001_create_users_table')\n",
        encoding="utf-8",
    )

    # 3. Documentation
    docs_dir = project_dir / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "architecture.md").write_text(
        "# System Architecture\n"
        "This project uses a microservice architecture with PostgreSQL and Redis.\n\n"
        "## Authentication Flow\n"
        "Clients authenticate via the auth service to receive a signed JWT bearer token.\n\n"
        "## Database Replication\n"
        "PostgreSQL uses streaming replication with automatic failover.\n",
        encoding="utf-8",
    )


def test_retrieval_recall_on_corpus(cli_runner: CliRunner, temp_project: Path):
    """Evaluate hybrid retrieval recall on known questions against evaluation corpus."""
    setup_evaluation_corpus(temp_project)

    # 1. Initialize and ingest
    cli_runner.invoke(app, ["init", str(temp_project)])
    cli_runner.invoke(app, ["ingest", str(temp_project)])

    settings = IroncladSettings(project_root=temp_project)

    # Question 1: Specific function in auth/jwt.py
    selected1, all_fused1, _ = execute_hybrid_search(
        settings=settings,
        query="validate JWT token HMAC signature",
        top_k=5,
    )
    assert len(selected1) > 0
    top_file1 = selected1[0].file_path
    assert "jwt.py" in top_file1
    assert any("validate_jwt_token" in c.content for c in selected1)

    # Question 2: Specific function in db/models.py
    selected2, all_fused2, _ = execute_hybrid_search(
        settings=settings,
        query="database schema migrations postgresql",
        top_k=5,
    )
    assert len(selected2) > 0
    top_file2 = selected2[0].file_path
    assert "models.py" in top_file2

    # Question 3: System architecture in docs/architecture.md
    selected3, all_fused3, _ = execute_hybrid_search(
        settings=settings,
        query="system architecture streaming replication",
        top_k=5,
    )
    assert len(selected3) > 0
    top_file3 = selected3[0].file_path
    assert "architecture.md" in top_file3

    # Question 4: Out-of-corpus query has zero lexical matches
    selected4, all_fused4, _ = execute_hybrid_search(
        settings=settings,
        query="xylophone zeppelin kangaroo pterodactyl",
        top_k=5,
    )
    # Lexical rank must be None for all results since keywords don't exist
    assert all(r.lexical_rank is None for r in all_fused4)


def test_cli_search_command(cli_runner: CliRunner, temp_project: Path):
    """Verify `ironclad search` outputs JSON and formatted excerpts."""
    setup_evaluation_corpus(temp_project)
    cli_runner.invoke(app, ["init", str(temp_project)])
    cli_runner.invoke(app, ["ingest", str(temp_project)])

    # Search with JSON output
    res_json = cli_runner.invoke(
        app,
        ["search", "JWT validation", "-p", str(temp_project), "--json"],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["query"] == "JWT validation"
    assert data["total_results"] > 0
    assert "jwt.py" in data["results"][0]["file_path"]

    # Search with human-readable CLI output
    res_human = cli_runner.invoke(
        app,
        ["search", "JWT validation", "-p", str(temp_project)],
    )
    assert res_human.exit_code == 0
    assert "jwt.py" in res_human.stdout
    assert "validate_jwt_token" in res_human.stdout
