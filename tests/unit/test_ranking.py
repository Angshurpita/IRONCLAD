"""Unit tests for context selection and token budgeting."""

from ironclad.models import SearchResult
from ironclad.retrieval.ranking import select_context_by_budget


def test_select_context_by_budget():
    """Verify evidence selection respects token budget limits."""
    # 3 chunks of roughly 100 tokens (~400 characters) each
    c1 = SearchResult(
        chunk_id="c1",
        file_path="src/a.py",
        start_line=1,
        end_line=10,
        content="A" * 400,
        score=0.9,
    )
    c2 = SearchResult(
        chunk_id="c2",
        file_path="src/b.py",
        start_line=1,
        end_line=10,
        content="B" * 400,
        score=0.8,
    )
    c3 = SearchResult(
        chunk_id="c3",
        file_path="src/c.py",
        start_line=1,
        end_line=10,
        content="C" * 400,
        score=0.7,
    )

    # Budget max 4000 with 3850 reserve leaves 150 tokens
    # (Only 1 chunk of 100 tokens can fit)
    selected, total_tokens = select_context_by_budget(
        [c1, c2, c3],
        max_context_tokens=1000,
        reserve_tokens=850,  # available budget = 150
    )

    assert len(selected) == 1
    assert selected[0].chunk_id == "c1"
    assert total_tokens <= 150
