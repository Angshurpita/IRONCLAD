"""Unit tests for system and user RAG prompts."""

from ironclad.generation.prompts import build_rag_user_prompt, build_system_prompt
from ironclad.models import SearchResult


def test_system_prompt_rules():
    """Verify system prompt contains essential grounding and injection defense instructions."""
    sp = build_system_prompt()
    assert "GROUNDING" in sp.upper()
    assert "UNTRUSTED" in sp.upper()
    assert "insufficient_evidence" in sp.lower()
    assert "json" in sp.lower()


def test_user_prompt_formatting():
    """Verify user prompt properly formats retrieved source evidence blocks."""
    evidence = [
        SearchResult(
            chunk_id="chunk_1",
            file_path="src/auth.py",
            start_line=10,
            end_line=25,
            content="def verify_token(t): return True",
            score=0.95,
        )
    ]
    prompt = build_rag_user_prompt("How does token verification work?", evidence)

    assert "RETRIEVED CONTEXT EVIDENCE:" in prompt
    assert "[Source 1]: src/auth.py (Lines 10-25)" in prompt
    assert "def verify_token(t): return True" in prompt
    assert "USER QUESTION: How does token verification work?" in prompt


def test_user_prompt_empty_evidence():
    """Verify user prompt indicates absence of evidence when list is empty."""
    prompt = build_rag_user_prompt("Unknown topic?", [])
    assert "No relevant documents found" in prompt
