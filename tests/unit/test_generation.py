"""Unit tests for answer generation and JSON parsing."""

import pytest

from ironclad.errors import InferenceError
from ironclad.generation.gemma import extract_json_payload, generate_grounded_answer
from ironclad.generation.provider import MockGemmaProvider
from ironclad.models import SearchResult


def test_extract_json_payload_variants():
    """Verify robust JSON extraction from direct JSON, markdown fences, and embedded commentary."""
    # 1. Direct JSON
    assert extract_json_payload('{"answer": "hello"}') == {"answer": "hello"}

    # 2. Markdown fenced JSON
    fenced = "Here is the response:\n```json\n{\n  \"answer\": \"hello from markdown\"\n}\n```\nHope that helps!"
    assert extract_json_payload(fenced)["answer"] == "hello from markdown"

    # 3. Leading and trailing commentary
    commentary = "Sure! Here is the output:\n{\n  \"answer\": \"raw braced\"\n}\nThanks!"
    assert extract_json_payload(commentary)["answer"] == "raw braced"

    # 4. Invalid non-JSON raises InferenceError
    with pytest.raises(InferenceError):
        extract_json_payload("This is purely plain text with no json whatsoever.")


def test_generate_grounded_answer_with_mock_provider():
    """Verify end-to-end generation with MockGemmaProvider."""
    evidence = [
        SearchResult(
            chunk_id="chunk_1",
            file_path="src/calc.py",
            start_line=1,
            end_line=5,
            content="def add(a, b): return a + b",
            score=0.9,
        )
    ]

    mock_provider = MockGemmaProvider(
        canned_answer="The add function adds two numbers [1].",
        citations=[
            {
                "source_index": 1,
                "file_path": "src/calc.py",
                "start_line": 1,
                "end_line": 2,
                "excerpt": "def add(a, b):",
            }
        ],
        insufficient_evidence=False,
    )

    response = generate_grounded_answer(
        query="What does add do?",
        evidence=evidence,
        provider=mock_provider,
    )

    assert response.query == "What does add do?"
    assert "adds two numbers" in response.answer
    assert response.insufficient_evidence is False
    assert len(response.citations) == 1
    assert response.citations[0].file_path == "src/calc.py"
    assert response.latency_ms >= 0


def test_generate_grounded_answer_empty_evidence_abstains_immediately():
    """Verify empty evidence abstains without invoking the LLM provider."""
    mock_provider = MockGemmaProvider()
    response = generate_grounded_answer(
        query="Something without evidence?",
        evidence=[],
        provider=mock_provider,
    )
    assert response.insufficient_evidence is True
    assert "unable to answer" in response.answer.lower()
    assert len(response.citations) == 0
    assert response.retrieved_chunks_count == 0
