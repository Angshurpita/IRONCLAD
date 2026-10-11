"""Unit tests for SQLite transactional metadata store and concurrency locks."""

import json
from pathlib import Path

import pytest

from ironclad.errors import ConcurrentAccessError
from ironclad.models import DocumentChunk, IndexedDocument
from ironclad.storage.metadata import KnowledgeBaseLock, MetadataStore


def test_concurrency_lock_lifecycle(temp_project: Path):
    """Verify lock acquisition, re-entry conflict, and clean release."""
    lock_file = temp_project / "test.lock"

    # Acquire lock in context manager
    with KnowledgeBaseLock(lock_file, timeout_seconds=0.5):
        assert lock_file.exists()

        # Concurrent attempt should time out and raise ConcurrentAccessError
        with pytest.raises(ConcurrentAccessError):
            with KnowledgeBaseLock(lock_file, timeout_seconds=0.2):
                pass

    # Lock must be cleaned up on context exit
    assert not lock_file.exists()


def test_stale_lock_recovery(temp_project: Path):
    """Verify stale lock from a dead process PID is detected and automatically recovered."""
    lock_file = temp_project / "stale.lock"

    # Write a lock file with an invalid/dead PID (e.g. 9999999)
    with open(lock_file, "w") as f:
        f.write(json.dumps({"pid": 9999999, "timestamp": "2020-01-01T00:00:00Z"}))

    assert lock_file.exists()

    # Acquiring should detect stale lock, unlink it, and acquire successfully
    with KnowledgeBaseLock(lock_file, timeout_seconds=1.0) as lock:
        assert lock.acquired
        assert lock_file.exists()

    assert not lock_file.exists()


def test_metadata_store_crud_and_fts5(temp_project: Path):
    """Verify SQLite CRUD operations, cascade deletes, and FTS5 synchronization."""
    db_path = temp_project / "metadata.db"
    store = MetadataStore(db_path)

    # Initial state
    assert len(store.get_indexed_documents()) == 0
    assert len(store.get_all_chunks()) == 0

    # Add document with 2 chunks
    doc = IndexedDocument(
        file_path="src/auth.py",
        file_hash="hash_v1",
        size_bytes=512,
        mtime=1000.0,
        doc_type="source_code",
        chunk_count=2,
        indexed_at="2026-10-10T00:00:00Z",
    )

    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            file_path="src/auth.py",
            content="def validate_jwt(token: str):\n    return verify(token)",
            token_count=15,
            char_count=60,
            start_line=1,
            end_line=2,
            heading="def validate_jwt",
            file_hash="hash_v1",
            doc_type="source_code",
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            file_path="src/auth.py",
            content="def refresh_token(user_id: int):\n    return issue_token()",
            token_count=15,
            char_count=60,
            start_line=4,
            end_line=5,
            heading="def refresh_token",
            file_hash="hash_v1",
            doc_type="source_code",
        ),
    ]

    store.add_or_update_document(doc, chunks)

    # Verify storage
    stored_docs = store.get_indexed_documents()
    assert len(stored_docs) == 1
    assert "src/auth.py" in stored_docs
    assert stored_docs["src/auth.py"].file_hash == "hash_v1"

    stored_chunks = store.get_all_chunks()
    assert len(stored_chunks) == 2

    # Verify FTS5 indexed the content
    with store.connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, content FROM chunks_fts WHERE chunks_fts MATCH 'jwt';")
        matches = cursor.fetchall()
        assert len(matches) == 1
        assert matches[0]["chunk_id"] == "chunk_1"

    # Update document with new content and only 1 chunk
    doc_v2 = IndexedDocument(
        file_path="src/auth.py",
        file_hash="hash_v2",
        size_bytes=256,
        mtime=2000.0,
        doc_type="source_code",
        chunk_count=1,
        indexed_at="2026-10-10T01:00:00Z",
    )
    chunks_v2 = [
        DocumentChunk(
            chunk_id="chunk_3",
            file_path="src/auth.py",
            content="def validate_oauth(token: str):\n    return True",
            token_count=12,
            char_count=48,
            start_line=1,
            end_line=2,
            heading="def validate_oauth",
            file_hash="hash_v2",
            doc_type="source_code",
        )
    ]
    store.add_or_update_document(doc_v2, chunks_v2)

    # Verify old chunks were replaced
    updated_chunks = store.get_all_chunks()
    assert len(updated_chunks) == 1
    assert updated_chunks[0].chunk_id == "chunk_3"

    with store.connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'oauth';")
        assert len(cursor.fetchall()) == 1
        cursor.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'jwt';")
        assert len(cursor.fetchall()) == 0

    # Delete document
    store.delete_document("src/auth.py")
    assert len(store.get_indexed_documents()) == 0
    assert len(store.get_all_chunks()) == 0

    with store.connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM chunks_fts;")
        assert cursor.fetchone()[0] == 0
