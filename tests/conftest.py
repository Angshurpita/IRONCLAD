"""Shared test fixtures for IRONCLAD test suite."""

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from typer.testing import CliRunner

from ironclad.config import IroncladSettings


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing a Typer CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_project() -> Generator[Path, None, None]:
    """Fixture providing an isolated temporary directory for test projects."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ironclad_test_"))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_settings(temp_project: Path) -> IroncladSettings:
    """Fixture providing an IroncladSettings instance bound to temp_project."""
    return IroncladSettings(project_root=temp_project)
