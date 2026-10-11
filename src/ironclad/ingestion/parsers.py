"""Explicit document parsers for source code, markdown, config, text, and PDF."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pypdf

from ironclad.ingestion.discovery import DiscoveredFile
from ironclad.models import DocumentType

logger = logging.getLogger("ironclad.parsers")


@dataclass
class DocumentSection:
    """A logical segment of a parsed document preserving heading and line boundaries."""

    heading: Optional[str]
    start_line: int
    end_line: int
    content: str


@dataclass
class ParsedDocument:
    """Standardized representation of a parsed document."""

    rel_path: str
    file_hash: str
    doc_type: DocumentType
    lines: List[str]
    sections: List[DocumentSection] = field(default_factory=list)


class BaseParser(ABC):
    """Abstract base parser interface."""

    @abstractmethod
    def parse(self, file: DiscoveredFile) -> ParsedDocument:
        """Parse the discovered file into a structured ParsedDocument."""
        pass


def read_text_safe(path: str) -> List[str]:
    """Read file content with UTF-8 encoding, falling back safely to latin-1."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="latin-1") as f:
                return f.readlines()
        except Exception as e:
            logger.warning(f"Could not read text from {path}: {e}")
            return []
    except Exception as e:
        logger.warning(f"Error opening {path}: {e}")
        return []


class SourceCodeParser(BaseParser):
    """
    Parser for source code files (Python, JS, TS, Java, C, C++, Go, Rust, etc.).
    Preserves exact line numbers and identifies functional headings (classes, functions).
    
    Documented Limitations:
    - Uses regex structure detection rather than a full language-specific AST compiler.
    - Non-UTF-8 source files fall back to latin-1 encoding.
    """

    DEF_PATTERN = re.compile(
        r"^\s*(?:async\s+def|def|class|function|interface|export\s+(?:default\s+)?(?:class|function|interface)|fn|pub\s+fn|func|public\s+(?:class|interface|void|\w+))\s+([a-zA-Z0-9_]+)",
        re.MULTILINE,
    )

    def parse(self, file: DiscoveredFile) -> ParsedDocument:
        raw_lines = read_text_safe(str(file.path))
        lines = [line.rstrip("\r\n") for line in raw_lines]
        sections: List[DocumentSection] = []

        current_heading: Optional[str] = None
        current_start = 1
        current_buffer: List[str] = []

        for idx, line in enumerate(lines, start=1):
            match = self.DEF_PATTERN.match(line)
            if match:
                # If we have an existing buffer, finalize section
                if current_buffer:
                    sections.append(
                        DocumentSection(
                            heading=current_heading,
                            start_line=current_start,
                            end_line=idx - 1,
                            content="\n".join(current_buffer),
                        )
                    )
                    current_buffer = []
                current_heading = match.group(0).strip()
                current_start = idx

            current_buffer.append(line)

        if current_buffer:
            sections.append(
                DocumentSection(
                    heading=current_heading,
                    start_line=current_start,
                    end_line=len(lines),
                    content="\n".join(current_buffer),
                )
            )

        return ParsedDocument(
            rel_path=file.rel_path,
            file_hash=file.file_hash,
            doc_type=file.doc_type,
            lines=lines,
            sections=sections,
        )


class MarkdownParser(BaseParser):
    """
    Parser for Markdown and documentation files.
    Extracts heading hierarchy (H1, H2, H3) and maps line ranges.
    
    Documented Limitations:
    - Parses GFM headings; does not execute embedded scripts, HTML macros, or MathJax.
    """

    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")

    def parse(self, file: DiscoveredFile) -> ParsedDocument:
        raw_lines = read_text_safe(str(file.path))
        lines = [line.rstrip("\r\n") for line in raw_lines]
        sections: List[DocumentSection] = []

        current_heading: Optional[str] = None
        current_start = 1
        current_buffer: List[str] = []

        for idx, line in enumerate(lines, start=1):
            match = self.HEADING_PATTERN.match(line)
            if match:
                if current_buffer:
                    sections.append(
                        DocumentSection(
                            heading=current_heading,
                            start_line=current_start,
                            end_line=idx - 1,
                            content="\n".join(current_buffer),
                        )
                    )
                    current_buffer = []
                current_heading = match.group(2).strip()
                current_start = idx

            current_buffer.append(line)

        if current_buffer:
            sections.append(
                DocumentSection(
                    heading=current_heading,
                    start_line=current_start,
                    end_line=len(lines),
                    content="\n".join(current_buffer),
                )
            )

        return ParsedDocument(
            rel_path=file.rel_path,
            file_hash=file.file_hash,
            doc_type=file.doc_type,
            lines=lines,
            sections=sections,
        )


class PlainTextParser(BaseParser):
    """
    Parser for plain text and configuration files (JSON, YAML, TOML, text).
    Preserves exact lines and treats contiguous content as natural sections.
    """

    def parse(self, file: DiscoveredFile) -> ParsedDocument:
        raw_lines = read_text_safe(str(file.path))
        lines = [line.rstrip("\r\n") for line in raw_lines]
        sections = [
            DocumentSection(
                heading=None,
                start_line=1,
                end_line=max(1, len(lines)),
                content="\n".join(lines),
            )
        ]
        return ParsedDocument(
            rel_path=file.rel_path,
            file_hash=file.file_hash,
            doc_type=file.doc_type,
            lines=lines,
            sections=sections,
        )


class PDFParser(BaseParser):
    """
    Safe parser for PDF documents using pypdf.
    Synthesizes clean page-based line references.
    
    Documented Limitations:
    - Extracts text from digital/searchable PDFs only. Does not OCR scanned images.
    - Encrypted, password-protected, or corrupted PDFs are caught and cleanly skipped.
    """

    def parse(self, file: DiscoveredFile) -> ParsedDocument:
        lines: List[str] = []
        sections: List[DocumentSection] = []

        try:
            reader = pypdf.PdfReader(str(file.path))
            if reader.is_encrypted:
                logger.warning(f"Skipping encrypted PDF: {file.rel_path}")
                return ParsedDocument(
                    rel_path=file.rel_path,
                    file_hash=file.file_hash,
                    doc_type=file.doc_type,
                    lines=[],
                    sections=[],
                )

            current_line_num = 1
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                page_lines = [pl.strip() for pl in page_text.splitlines() if pl.strip()]
                if not page_lines:
                    continue

                page_start_line = current_line_num
                for pl in page_lines:
                    lines.append(pl)
                    current_line_num += 1

                sections.append(
                    DocumentSection(
                        heading=f"Page {page_idx}",
                        start_line=page_start_line,
                        end_line=current_line_num - 1,
                        content="\n".join(page_lines),
                    )
                )

        except Exception as e:
            logger.warning(f"Error parsing PDF file {file.rel_path}: {e}")
            return ParsedDocument(
                rel_path=file.rel_path,
                file_hash=file.file_hash,
                doc_type=file.doc_type,
                lines=[],
                sections=[],
            )

        return ParsedDocument(
            rel_path=file.rel_path,
            file_hash=file.file_hash,
            doc_type=file.doc_type,
            lines=lines,
            sections=sections,
        )


# Parser registry
PARSER_REGISTRY: Dict[DocumentType, BaseParser] = {
    DocumentType.SOURCE_CODE: SourceCodeParser(),
    DocumentType.MARKDOWN: MarkdownParser(),
    DocumentType.PLAIN_TEXT: PlainTextParser(),
    DocumentType.CONFIG: PlainTextParser(),
    DocumentType.PDF: PDFParser(),
}


def get_parser_for_type(doc_type: DocumentType) -> BaseParser:
    """Return the designated parser for a given document type."""
    return PARSER_REGISTRY.get(doc_type, PlainTextParser())
