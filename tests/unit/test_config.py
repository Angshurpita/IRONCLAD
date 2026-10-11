"""Unit tests for configuration and credential protection."""

from pathlib import Path

from pydantic import SecretStr

from ironclad.config import IroncladSettings, get_settings, mask_secret


def test_mask_secret():
    """Verify secrets are masked safely and never leaked."""
    assert mask_secret(None) == "(not set)"
    assert mask_secret("") == "(not set)"
    assert mask_secret(SecretStr("")) == "(not set)"
    assert mask_secret("short") == "****"
    assert mask_secret(SecretStr("12345678")) == "****"

    masked = mask_secret("sk-or-v1-abcdef1234567890")
    assert masked.startswith("sk-o...")
    assert masked.endswith("7890")
    assert "abcdef" not in masked

    masked_secret = mask_secret(SecretStr("sk-or-v1-abcdef1234567890"))
    assert masked_secret.startswith("sk-o...")
    assert "abcdef" not in masked_secret


def test_default_settings(temp_project: Path):
    """Verify default settings initialization."""
    settings = IroncladSettings(project_root=temp_project, _env_file=None)
    assert settings.ironclad_dir_name == ".ironclad"
    assert settings.inference_provider == "openrouter"
    assert "gemma" in settings.openrouter_model
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.ironclad_path == temp_project.resolve() / ".ironclad"
    assert settings.metadata_db_path == temp_project.resolve() / ".ironclad" / "metadata.db"
    assert not settings.is_initialized()


def test_safe_dict_redaction(temp_project: Path):
    """Verify safe_dict masks sensitive credentials."""
    settings = IroncladSettings(
        project_root=temp_project,
        openrouter_api_key=SecretStr("sk-or-v1-supersecretkey123456789"),
    )
    data = settings.safe_dict()
    assert data["openrouter_api_key_set"] is True
    assert "supersecretkey" not in data["openrouter_api_key_masked"]
    assert data["openrouter_api_key_masked"].startswith("sk-o...")


def test_get_settings_ancestor_resolution(temp_project: Path):
    """Verify get_settings locates .ironclad directory in parent hierarchy."""
    resolved_root = temp_project.resolve()
    kb_dir = resolved_root / ".ironclad"
    kb_dir.mkdir()
    sub_dir = resolved_root / "src" / "deep"
    sub_dir.mkdir(parents=True)

    settings = get_settings(sub_dir)
    assert settings.project_root == resolved_root
    assert settings.ironclad_path == kb_dir
