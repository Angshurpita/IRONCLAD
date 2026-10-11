"""Unit tests for structure-aware chunker."""

from ironclad.ingestion.chunker import chunk_document, estimate_tokens
from ironclad.ingestion.parsers import DocumentSection, ParsedDocument
from ironclad.models import DocumentType


def test_estimate_tokens():
    """Verify heuristic token estimation."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("12345678") == 2


def test_chunk_document_line_ranges():
    """Verify structure-aware chunking preserves line numbers and headings."""
    content = "\n".join([f"line_{i} code implementation details here" for i in range(1, 101)])
    lines = content.splitlines()

    section = DocumentSection(
        heading="class ProcessHandler",
        start_line=1,
        end_line=100,
        content=content,
    )

    doc = ParsedDocument(
        rel_path="src/handler.py",
        file_hash="112233445566778899aabbccddeeff00",
        doc_type=DocumentType.SOURCE_CODE,
        lines=lines,
        sections=[section],
    )

    # Chunk with small target token size to force multiple chunks
    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1

    # Verify first chunk
    first = chunks[0]
    assert first.file_path == "src/handler.py"
    assert first.file_hash == "112233445566778899aabbccddeeff00"
    assert first.heading == "class ProcessHandler"
    assert first.start_line == 1
    assert first.end_line > 1
    assert first.chunk_id == "112233445566_0"

    # Verify second chunk continues line numbers with overlap
    second = chunks[1]
    assert second.chunk_id == "112233445566_1"
    assert second.start_line <= first.end_line
    assert second.end_line > first.end_line


def test_chunk_empty_document():
    """Verify empty document produces no chunks."""
    doc = ParsedDocument(
        rel_path="empty.py",
        file_hash="dummyhash",
        doc_type=DocumentType.SOURCE_CODE,
        lines=[],
        sections=[],
    )
    assert chunk_document(doc) == []
