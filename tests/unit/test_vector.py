"""Unit tests for ChromaVectorStore."""

from pathlib import Path

from ironclad.models import DocumentChunk
from ironclad.retrieval.vector import ChromaVectorStore


def test_chroma_vector_store_lifecycle(temp_project: Path):
    """Verify adding, searching, and deleting vectors in ChromaDB."""
    persist_dir = temp_project / "vectors"
    store = ChromaVectorStore(persist_dir)

    assert store.count() == 0

    chunk1 = DocumentChunk(
        chunk_id="chunk_vec_1",
        file_path="src/auth.py",
        content="JWT token validation logic",
        token_count=5,
        char_count=26,
        start_line=1,
        end_line=3,
        heading="class Auth",
        file_hash="hash1",
        doc_type="source_code",
    )
    chunk2 = DocumentChunk(
        chunk_id="chunk_vec_2",
        file_path="src/db.py",
        content="Postgres connection pool manager",
        token_count=5,
        char_count=32,
        start_line=1,
        end_line=5,
        heading="def connect_db",
        file_hash="hash2",
        doc_type="source_code",
    )

    # 384-dimensional synthetic unit vectors
    emb1 = [1.0] + [0.0] * 383
    emb2 = [0.0, 1.0] + [0.0] * 382

    store.add_or_update_chunks([chunk1, chunk2], [emb1, emb2])
    assert store.count() == 2

    # Query vector closest to emb1
    results = store.search_vector(query_embedding=emb1, limit=2)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_vec_1"
    assert results[0].vector_rank == 1
    assert results[0].score >= 0.99  # Exact match cosine similarity ~ 1.0

    # Delete chunk1
    store.delete_chunks(["chunk_vec_1"])
    assert store.count() == 1

    remaining = store.search_vector(query_embedding=emb1, limit=2)
    assert len(remaining) == 1
    assert remaining[0].chunk_id == "chunk_vec_2"
