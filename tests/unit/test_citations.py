"""Unit tests for independent citation verification and grounding checks."""

from ironclad.generation.citations import validate_citations
from ironclad.models import SearchResult


def test_validate_citations_success():
    """Verify valid citation is matched to retrieved chunk."""
    evidence = [
        SearchResult(
            chunk_id="chunk_1",
            file_path="src/auth.py",
            start_line=10,
            end_line=30,
            content="def validate_jwt(token):\n    return verify(token)",
            score=0.9,
        )
    ]
    raw_citations = [
        {
            "source_index": 1,
            "file_path": "src/auth.py",
            "start_line": 10,
            "end_line": 15,
            "excerpt": "def validate_jwt(token):",
        }
    ]

    report = validate_citations(raw_citations, evidence)
    assert report.is_fully_grounded is True
    assert report.rejected_citations_count == 0
    assert len(report.verified_citations) == 1
    assert report.verified_citations[0].file_path == "src/auth.py"
    assert report.verified_citations[0].start_line == 10


def test_validate_citations_rejects_hallucinations():
    """Verify citations pointing to un-retrieved files or non-existent lines are rejected."""
    evidence = [
        SearchResult(
            chunk_id="chunk_1",
            file_path="src/auth.py",
            start_line=10,
            end_line=30,
            content="def validate_jwt(token):\n    return verify(token)",
            score=0.9,
        )
    ]

    # Model hallucinated a completely different file
    raw_citations = [
        {
            "source_index": 1,
            "file_path": "src/billing_payments.py",
            "start_line": 1,
            "end_line": 5,
            "excerpt": "def charge_card(): pass",
        }
    ]

    report = validate_citations(raw_citations, evidence)
    assert report.is_fully_grounded is False
    assert report.rejected_citations_count == 1
    assert len(report.verified_citations) == 0
    # Since all citations were rejected, independent sufficiency check flags it
    assert report.insufficient_evidence is True


def test_validate_citations_abstention():
    """Verify model's abstention flag is preserved cleanly."""
    evidence = [
        SearchResult(
            chunk_id="chunk_1",
            file_path="src/auth.py",
            start_line=10,
            end_line=30,
            content="authentication code",
            score=0.5,
        )
    ]

    report = validate_citations([], evidence, model_claimed_insufficient=True)
    assert report.insufficient_evidence is True
    assert len(report.verified_citations) == 0
