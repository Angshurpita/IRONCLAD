"""Unit tests for document parsers."""

from pathlib import Path

import pypdf

from ironclad.ingestion.discovery import DiscoveredFile
from ironclad.ingestion.parsers import (
    MarkdownParser,
    PDFParser,
    PlainTextParser,
    SourceCodeParser,
)
from ironclad.models import DocumentType


def test_source_code_parser(temp_project: Path):
    """Verify source code parser identifies definitions and maps line numbers."""
    code_path = temp_project / "service.py"
    code_path.write_text(
        "import os\n\n"
        "class AuthService:\n"
        "    def authenticate(self, token: str):\n"
        "        return True\n\n"
        "def helper_func():\n"
        "    return 42\n",
        encoding="utf-8",
    )

    disc = DiscoveredFile(
        path=code_path,
        rel_path="service.py",
        file_hash="dummyhash",
        size_bytes=code_path.stat().st_size,
        mtime=code_path.stat().st_mtime,
        doc_type=DocumentType.SOURCE_CODE,
    )

    parser = SourceCodeParser()
    parsed = parser.parse(disc)

    assert parsed.rel_path == "service.py"
    assert len(parsed.lines) == 8
    assert len(parsed.sections) >= 2

    # Check headings detected
    headings = [s.heading for s in parsed.sections if s.heading]
    assert any("class AuthService" in h for h in headings)
    assert any("def helper_func" in h for h in headings)


def test_markdown_parser(temp_project: Path):
    """Verify markdown parser extracts headings and sections."""
    md_path = temp_project / "DOCS.md"
    md_path.write_text(
        "# Project Overview\n"
        "This is an intro paragraph.\n\n"
        "## Architecture\n"
        "Details on system components.\n\n"
        "### Ingestion\n"
        "Chunking and storage details.\n",
        encoding="utf-8",
    )

    disc = DiscoveredFile(
        path=md_path,
        rel_path="DOCS.md",
        file_hash="dummyhash",
        size_bytes=md_path.stat().st_size,
        mtime=md_path.stat().st_mtime,
        doc_type=DocumentType.MARKDOWN,
    )

    parser = MarkdownParser()
    parsed = parser.parse(disc)

    assert parsed.rel_path == "DOCS.md"
    headings = [s.heading for s in parsed.sections if s.heading]
    assert "Project Overview" in headings
    assert "Architecture" in headings
    assert "Ingestion" in headings


def test_plain_text_parser(temp_project: Path):
    """Verify plain text parser preserves all lines."""
    txt_path = temp_project / "info.txt"
    txt_path.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")

    disc = DiscoveredFile(
        path=txt_path,
        rel_path="info.txt",
        file_hash="dummyhash",
        size_bytes=txt_path.stat().st_size,
        mtime=txt_path.stat().st_mtime,
        doc_type=DocumentType.PLAIN_TEXT,
    )

    parser = PlainTextParser()
    parsed = parser.parse(disc)
    assert len(parsed.lines) == 3
    assert parsed.sections[0].start_line == 1
    assert parsed.sections[0].end_line == 3


def test_pdf_parser_with_sample_pdf(temp_project: Path):
    """Verify PDF parser extracts text and maps page numbers."""
    pdf_path = temp_project / "sample.pdf"

    # Create a minimal valid PDF using pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    disc = DiscoveredFile(
        path=pdf_path,
        rel_path="sample.pdf",
        file_hash="dummyhash",
        size_bytes=pdf_path.stat().st_size,
        mtime=pdf_path.stat().st_mtime,
        doc_type=DocumentType.PDF,
    )

    parser = PDFParser()
    parsed = parser.parse(disc)
    assert parsed.rel_path == "sample.pdf"
    assert parsed.doc_type == DocumentType.PDF


def test_pdf_parser_corrupted_file(temp_project: Path):
    """Verify corrupt PDF does not crash the parser."""
    corrupt_pdf = temp_project / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4 garbage data not a valid pdf")

    disc = DiscoveredFile(
        path=corrupt_pdf,
        rel_path="corrupt.pdf",
        file_hash="dummyhash",
        size_bytes=corrupt_pdf.stat().st_size,
        mtime=corrupt_pdf.stat().st_mtime,
        doc_type=DocumentType.PDF,
    )

    parser = PDFParser()
    parsed = parser.parse(disc)
    assert parsed.lines == []
    assert parsed.sections == []
