"""Service implementing index status reports and system diagnostics (doctor)."""

import os
import sqlite3
import sys
from typing import List

import httpx

from ironclad.config import IroncladSettings
from ironclad.models import DoctorCheck, DoctorReport, IndexStatus


def get_status(settings: IroncladSettings) -> IndexStatus:
    """Inspect the target knowledge base and return an IndexStatus object."""
    initialized = settings.is_initialized()
    kb_path = str(settings.ironclad_path)

    if not initialized:
        return IndexStatus(
            initialized=False,
            knowledge_base_path=kb_path,
            total_files=0,
            total_chunks=0,
            index_version=1,
            inference_provider=settings.inference_provider,
            inference_model=(
                settings.openrouter_model
                if settings.inference_provider == "openrouter"
                else settings.ollama_model
            ),
            embedding_model=settings.embedding_model,
            is_consistent=True,
            inconsistencies=["Knowledge base is not initialized. Run `ironclad init`."],
        )

    # Inspect SQLite database
    inconsistencies: List[str] = []
    total_files = 0
    total_chunks = 0
    index_version = 1
    last_ingested_at = None
    stored_embedding_model = settings.embedding_model

    try:
        with sqlite3.connect(settings.metadata_db_path, timeout=5.0) as conn:
            cursor = conn.cursor()

            # Check metadata table
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata_info';"
            )
            if cursor.fetchone():
                cursor.execute("SELECT key, value FROM metadata_info;")
                meta = dict(cursor.fetchall())
                index_version = int(meta.get("schema_version", 1))
                last_ingested_at = meta.get("last_ingested_at")
                stored_embedding_model = meta.get("embedding_model", settings.embedding_model)

                if (
                    stored_embedding_model
                    and stored_embedding_model != settings.embedding_model
                ):
                    inconsistencies.append(
                        f"Configured embedding model ({settings.embedding_model}) differs "
                        f"from indexed embedding model ({stored_embedding_model})."
                    )

            # Check documents count
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='documents';"
            )
            if cursor.fetchone():
                cursor.execute("SELECT count(*) FROM documents;")
                row = cursor.fetchone()
                total_files = row[0] if row else 0

            # Check chunks count
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks';"
            )
            if cursor.fetchone():
                cursor.execute("SELECT count(*) FROM chunks;")
                row = cursor.fetchone()
                total_chunks = row[0] if row else 0

    except sqlite3.DatabaseError as e:
        inconsistencies.append(f"Database error reading knowledge base: {e}")

    # Check vector store directory
    if not settings.vector_store_path.exists():
        inconsistencies.append("Vector store directory is missing from .ironclad.")

    is_consistent = len(inconsistencies) == 0

    return IndexStatus(
        initialized=True,
        knowledge_base_path=kb_path,
        total_files=total_files,
        total_chunks=total_chunks,
        index_version=index_version,
        last_ingested_at=last_ingested_at,
        embedding_model=stored_embedding_model,
        inference_provider=settings.inference_provider,
        inference_model=(
            settings.openrouter_model
            if settings.inference_provider == "openrouter"
            else settings.ollama_model
        ),
        is_consistent=is_consistent,
        inconsistencies=inconsistencies,
    )


def run_diagnostics(settings: IroncladSettings) -> DoctorReport:
    """Run thorough environment, storage, embedding, and provider health checks."""
    checks: List[DoctorCheck] = []

    # 1. Python runtime check
    py_ver = sys.version_info
    py_ok = (py_ver.major == 3 and py_ver.minor >= 11)
    checks.append(
        DoctorCheck(
            category="Runtime",
            name="Python Version",
            passed=py_ok,
            message=f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}",
            hint=None if py_ok else "Ironclad requires Python 3.11 or newer.",
        )
    )

    # 2. Filesystem & permissions check
    target_dir = settings.project_root
    try:
        writable = os.access(target_dir, os.W_OK)
        # Test creating a temporary test file if writable
        if writable:
            test_file = target_dir / ".ironclad_perm_test"
            test_file.touch()
            test_file.unlink()
        checks.append(
            DoctorCheck(
                category="Storage",
                name="Project Directory Writable",
                passed=writable,
                message=f"Directory '{target_dir}' is writable.",
                hint=None if writable else "Check filesystem permissions for the target project.",
            )
        )
    except Exception as e:
        checks.append(
            DoctorCheck(
                category="Storage",
                name="Project Directory Writable",
                passed=False,
                message=f"Error accessing directory: {e}",
                hint="Verify directory write permissions.",
            )
        )

    # 3. Knowledge base initialization
    kb_init = settings.is_initialized()
    checks.append(
        DoctorCheck(
            category="Storage",
            name="Knowledge Base Initialized",
            passed=kb_init,
            message="Knowledge base is initialized." if kb_init else "Knowledge base not yet initialized.",
            hint=None if kb_init else "Run `ironclad init` to create the knowledge base.",
        )
    )

    # 4. SQLite integrity check & FTS5 support (if initialized)
    if kb_init:
        try:
            with sqlite3.connect(settings.metadata_db_path, timeout=3.0) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check;")
                res = cursor.fetchone()
                ok = res and res[0] == "ok"
                checks.append(
                    DoctorCheck(
                        category="Storage",
                        name="Metadata Database Integrity",
                        passed=bool(ok),
                        message="SQLite database passed PRAGMA integrity_check."
                        if ok
                        else f"Integrity check returned: {res}",
                        hint=None if ok else "Metadata database is corrupted. Re-initialize or recover.",
                    )
                )

                # Check FTS5 availability
                try:
                    cursor.execute("SELECT highlight(chunks_fts, 2, '', '') FROM chunks_fts LIMIT 0;")
                    fts_ok = True
                    fts_msg = "SQLite FTS5 full-text search index is operational."
                except Exception:
                    # Table might be empty or verify fts5 module directly
                    try:
                        cursor.execute("CREATE TEMP TABLE _test_fts USING fts5(c);")
                        cursor.execute("DROP TABLE _test_fts;")
                        fts_ok = True
                        fts_msg = "SQLite FTS5 full-text engine is operational."
                    except Exception as fts_err:
                        fts_ok = False
                        fts_msg = f"SQLite FTS5 not available: {fts_err}"

                checks.append(
                    DoctorCheck(
                        category="Storage",
                        name="SQLite FTS5 Engine",
                        passed=fts_ok,
                        message=fts_msg,
                        hint=None if fts_ok else "Install Python SQLite with FTS5 enabled.",
                    )
                )
        except Exception as e:
            checks.append(
                DoctorCheck(
                    category="Storage",
                    name="Metadata Database Integrity",
                    passed=False,
                    message=f"Could not open SQLite database: {e}",
                    hint="Check file locks and permissions.",
                )
            )

        # Vector store collection health
        if settings.vector_store_path.exists():
            try:
                from ironclad.retrieval.vector import ChromaVectorStore
                vstore = ChromaVectorStore(settings.vector_store_path)
                v_count = vstore.count()
                checks.append(
                    DoctorCheck(
                        category="Storage",
                        name="Vector Store Health",
                        passed=True,
                        message=f"ChromaDB persistent store active ({v_count} vectors indexed).",
                        hint=None,
                    )
                )
            except Exception as ve:
                checks.append(
                    DoctorCheck(
                        category="Storage",
                        name="Vector Store Health",
                        passed=False,
                        message=f"ChromaDB vector store error: {ve}",
                        hint="Try running `ironclad ingest --reindex` to repair the vector store.",
                    )
                )

    # 5. Lock file health check
    lock_file = settings.lock_file_path
    if lock_file.exists():
        from ironclad.storage.metadata import KnowledgeBaseLock
        kb_lock = KnowledgeBaseLock(lock_file)
        if kb_lock._is_stale_lock():
            checks.append(
                DoctorCheck(
                    category="Concurrency",
                    name="Process Lock File",
                    passed=True,
                    message="Found inactive lock file (will be automatically cleaned up on next run).",
                    hint=None,
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    category="Concurrency",
                    name="Process Lock File",
                    passed=False,
                    message="Active process lock detected. Another Ironclad command is currently executing.",
                    hint="Wait for the running command to finish or delete the lock file if aborted.",
                )
            )
    else:
        checks.append(
            DoctorCheck(
                category="Concurrency",
                name="Process Lock File",
                passed=True,
                message="No lock contention detected.",
                hint=None,
            )
        )

    # 6. Embedding runtime check
    try:
        from ironclad.embeddings.provider import get_embedding_provider
        embedder = get_embedding_provider(settings)
        test_vec = embedder.embed_query("ironclad test ping")
        checks.append(
            DoctorCheck(
                category="Embeddings",
                name="Local Embedding Model Runtime",
                passed=len(test_vec) == embedder.dimension,
                message=f"Embedding provider '{embedder.provider_name}' operational ({embedder.dimension}-dim).",
                hint=None,
            )
        )
    except Exception as ee:
        checks.append(
            DoctorCheck(
                category="Embeddings",
                name="Local Embedding Model Runtime",
                passed=False,
                message=f"Failed to initialize embedding model: {ee}",
                hint="Verify onnxruntime and tokenizers dependencies are installed.",
            )
        )

    # 6. Inference Provider Connectivity
    provider = settings.inference_provider.lower().strip()
    if provider in ("openrouter", "openrouter.ai"):
        has_key = bool(settings.openrouter_api_key and settings.openrouter_api_key.get_secret_value().strip())
        if not has_key:
            checks.append(
                DoctorCheck(
                    category="Inference",
                    name="OpenRouter API Key",
                    passed=False,
                    message="OPENROUTER_API_KEY is not set.",
                    hint="Set OPENROUTER_API_KEY in your environment or in .env file.",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    category="Inference",
                    name="OpenRouter API Key",
                    passed=True,
                    message="OPENROUTER_API_KEY is configured.",
                    hint=None,
                )
            )
            # Connectivity probe to OpenRouter
            try:
                # Masked check - querying endpoint
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(
                        f"{settings.openrouter_base_url.rstrip('/')}/models",
                        headers={
                            "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"
                        },
                    )
                    if resp.status_code == 200:
                        checks.append(
                            DoctorCheck(
                                category="Inference",
                                name="OpenRouter Connectivity",
                                passed=True,
                                message="Successfully connected to OpenRouter API.",
                                hint=None,
                            )
                        )
                        # Check model existence
                        try:
                            models_data = resp.json().get("data", [])
                            avail_ids = {m.get("id") for m in models_data if isinstance(m, dict)}
                            target_m = settings.openrouter_model
                            if target_m == "google/gemma-4-26b-a4b":
                                target_m = "google/gemma-4-26b-a4b-it"
                            model_found = target_m in avail_ids
                            checks.append(
                                DoctorCheck(
                                    category="Inference",
                                    name="Target Model Availability",
                                    passed=model_found,
                                    message=f"Model '{target_m}' is {'available' if model_found else 'not found'} on OpenRouter.",
                                    hint=None if model_found else f"Verify model slug '{target_m}' on openrouter.ai/models.",
                                )
                            )
                        except Exception:
                            pass
                    elif resp.status_code in (401, 403):
                        checks.append(
                            DoctorCheck(
                                category="Inference",
                                name="OpenRouter Connectivity",
                                passed=False,
                                message=f"Authentication failed (HTTP {resp.status_code}).",
                                hint="Verify your OPENROUTER_API_KEY is valid and has appropriate permissions.",
                            )
                        )
                    else:
                        checks.append(
                            DoctorCheck(
                                category="Inference",
                                name="OpenRouter Connectivity",
                                passed=False,
                                message=f"OpenRouter returned status HTTP {resp.status_code}.",
                                hint=f"Endpoint: {settings.openrouter_base_url}",
                            )
                        )
            except Exception as e:
                checks.append(
                    DoctorCheck(
                        category="Inference",
                        name="OpenRouter Connectivity",
                        passed=False,
                        message=f"Network error connecting to OpenRouter: {e.__class__.__name__}",
                        hint="Check your internet connection and proxy settings.",
                    )
                )

    elif provider == "ollama":
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
                if resp.status_code == 200:
                    models_data = resp.json()
                    model_names = [m.get("name", "") for m in models_data.get("models", [])]
                    model_available = any(settings.ollama_model in m for m in model_names)
                    checks.append(
                        DoctorCheck(
                            category="Inference",
                            name="Ollama Service Connectivity",
                            passed=True,
                            message=f"Connected to Ollama at {settings.ollama_base_url}.",
                            hint=None,
                        )
                    )
                    checks.append(
                        DoctorCheck(
                            category="Inference",
                            name="Ollama Model Availability",
                            passed=model_available,
                            message=f"Model '{settings.ollama_model}' {'is installed' if model_available else 'is NOT downloaded'}.",
                            hint=None if model_available else f"Run: ollama pull {settings.ollama_model}",
                        )
                    )
                else:
                    checks.append(
                        DoctorCheck(
                            category="Inference",
                            name="Ollama Service Connectivity",
                            passed=False,
                            message=f"Ollama server returned HTTP {resp.status_code}.",
                            hint=f"Check Ollama server at {settings.ollama_base_url}",
                        )
                    )
        except Exception as e:
            checks.append(
                DoctorCheck(
                    category="Inference",
                    name="Ollama Service Connectivity",
                    passed=False,
                    message=f"Cannot reach Ollama at {settings.ollama_base_url}: {e.__class__.__name__}",
                    hint="Ensure Ollama is running (`ollama serve`).",
                )
            )
    else:
        checks.append(
            DoctorCheck(
                category="Inference",
                name="Inference Provider",
                passed=False,
                message=f"Unknown inference provider: '{provider}'",
                hint="Supported providers: 'openrouter', 'ollama'.",
            )
        )

    all_passed = all(c.passed for c in checks)
    return DoctorReport(checks=checks, all_passed=all_passed)
