"""Unit tests for SQLite FTS5 lexical search and query sanitization."""

from pathlib import Path

from ironclad.models import DocumentChunk, IndexedDocument
from ironclad.retrieval.lexical import sanitize_fts5_query, search_lexical
from ironclad.storage.metadata import MetadataStore


def test_sanitize_fts5_query():
    """Verify special FTS5 operators are safely sanitized into prefix clauses."""
    assert sanitize_fts5_query("") == ""
    assert sanitize_fts5_query("a") == ""  # Single char ignored

    raw = 'def test_auth(): jwt.verify("secret")*'
    cleaned = sanitize_fts5_query(raw)
    assert '"def"*' in cleaned
    assert '"jwt"*' in cleaned
    assert '"verify"*' in cleaned
    assert '"secret"*' in cleaned
    assert "(" not in cleaned
    assert ")" not in cleaned


def test_search_lexical(temp_project: Path):
    """Verify BM25 lexical ranking finds matching chunks."""
    db_path = temp_project / "metadata.db"
    store = MetadataStore(db_path)

    doc = IndexedDocument(
        file_path="src/jwt.py",
        file_hash="hash123",
        size_bytes=200,
        mtime=1.0,
        doc_type="source_code",
        chunk_count=1,
        indexed_at="2026-10-10T00:00:00Z",
    )
    chunks = [
        DocumentChunk(
            chunk_id="chunk_jwt_1",
            file_path="src/jwt.py",
            content="def validate_jwt_token(bearer_token: str):\n    return verify_signature(bearer_token)",
            token_count=16,
            char_count=85,
            start_line=1,
            end_line=2,
            heading="def validate_jwt_token",
            file_hash="hash123",
            doc_type="source_code",
        )
    ]
    store.add_or_update_document(doc, chunks)

    results = search_lexical(store, "jwt token", limit=5)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_jwt_1"
    assert results[0].lexical_rank == 1
    assert results[0].score > 0

    # Non-matching query
    nomatch = search_lexical(store, "database postgres migration", limit=5)
    assert len(nomatch) == 0
