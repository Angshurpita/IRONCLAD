"""Unit tests for Reciprocal Rank Fusion."""

from ironclad.models import SearchResult
from ironclad.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_combines_and_boosts_overlapping_candidates():
    """Verify RRF formula and score boost for documents appearing in both dense and sparse results."""
    # Chunk A is rank 1 in vector, rank 1 in lexical
    # Chunk B is rank 2 in vector, not in lexical
    # Chunk C is rank 2 in lexical, not in vector

    vector_results = [
        SearchResult(
            chunk_id="chunk_A",
            file_path="src/a.py",
            start_line=1,
            end_line=5,
            content="content A",
            score=0.9,
            vector_rank=1,
        ),
        SearchResult(
            chunk_id="chunk_B",
            file_path="src/b.py",
            start_line=1,
            end_line=5,
            content="content B",
            score=0.7,
            vector_rank=2,
        ),
    ]

    lexical_results = [
        SearchResult(
            chunk_id="chunk_A",
            file_path="src/a.py",
            start_line=1,
            end_line=5,
            content="content A",
            score=5.0,
            lexical_rank=1,
        ),
        SearchResult(
            chunk_id="chunk_C",
            file_path="src/c.py",
            start_line=1,
            end_line=5,
            content="content C",
            score=3.0,
            lexical_rank=2,
        ),
    ]

    k = 60
    fused = reciprocal_rank_fusion(vector_results, lexical_results, k=k)

    # Chunk A score should be 1/(60+1) + 1/(60+1) = 2/61 ~ 0.032787
    # Chunk B score should be 1/(60+2) = 1/62 ~ 0.016129
    # Chunk C score should be 1/(60+2) = 1/62 ~ 0.016129

    assert len(fused) == 3
    assert fused[0].chunk_id == "chunk_A"
    assert round(fused[0].score, 5) == round(2.0 / 61.0, 5)
    assert fused[0].vector_rank == 1
    assert fused[0].lexical_rank == 1

    # Chunk B and C have identical RRF scores
    assert round(fused[1].score, 5) == round(1.0 / 62.0, 5)
    assert round(fused[2].score, 5) == round(1.0 / 62.0, 5)
