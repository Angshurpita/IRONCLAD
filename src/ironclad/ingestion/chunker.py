"""Structure-aware chunking engine preserving exact line ranges and headings."""

import logging
from typing import List

from ironclad.ingestion.parsers import DocumentSection, ParsedDocument
from ironclad.models import DocumentChunk

logger = logging.getLogger("ironclad.chunker")


def estimate_tokens(text: str) -> int:
    """Heuristic token estimation (~4 characters per token for source code/English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def chunk_section(
    section: DocumentSection,
    rel_path: str,
    file_hash: str,
    doc_type: str,
    target_tokens: int,
    overlap_tokens: int,
    start_chunk_idx: int,
) -> List[DocumentChunk]:
    """Break a document section into chunks respecting line numbers and overlap."""
    lines = section.content.splitlines()
    if not lines:
        return []

    chunks: List[DocumentChunk] = []
    current_lines: List[str] = []
    current_start_line = section.start_line
    current_tokens = 0
    chunk_idx = start_chunk_idx

    for line in lines:
        line_tokens = estimate_tokens(line + "\n")

        # If adding this line exceeds target and we already have content, finalize chunk
        if current_tokens + line_tokens > target_tokens and current_lines:
            content = "\n".join(current_lines)
            end_line = current_start_line + len(current_lines) - 1
            chunk_id = f"{file_hash[:12]}_{chunk_idx}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    file_path=rel_path,
                    content=content,
                    token_count=current_tokens,
                    char_count=len(content),
                    start_line=current_start_line,
                    end_line=end_line,
                    heading=section.heading,
                    file_hash=file_hash,
                    doc_type=doc_type,
                )
            )
            chunk_idx += 1

            # Compute overlap lines backwards
            overlap_accum = 0
            overlap_lines: List[str] = []
            for prev_line in reversed(current_lines):
                overlap_accum += estimate_tokens(prev_line + "\n")
                overlap_lines.insert(0, prev_line)
                if overlap_accum >= overlap_tokens:
                    break

            current_lines = list(overlap_lines)
            current_tokens = overlap_accum
            current_start_line = end_line - len(overlap_lines) + 1

        current_lines.append(line)
        current_tokens += line_tokens

    # Finalize leftover lines
    if current_lines:
        content = "\n".join(current_lines)
        end_line = section.end_line
        chunk_id = f"{file_hash[:12]}_{chunk_idx}"

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                file_path=rel_path,
                content=content,
                token_count=current_tokens,
                char_count=len(content),
                start_line=current_start_line,
                end_line=end_line,
                heading=section.heading,
                file_hash=file_hash,
                doc_type=doc_type,
            )
        )

    return chunks


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[DocumentChunk]:
    """
    Produce structure-aware chunks from a ParsedDocument.
    Preserves exact 1-indexed line spans and section heading hierarchy.
    """
    if not doc.lines or not doc.sections:
        return []

    chunks: List[DocumentChunk] = []
    chunk_counter = 0

    for section in doc.sections:
        section_chunks = chunk_section(
            section=section,
            rel_path=doc.rel_path,
            file_hash=doc.file_hash,
            doc_type=doc.doc_type.value,
            target_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
            start_chunk_idx=chunk_counter,
        )
        chunks.extend(section_chunks)
        chunk_counter += len(section_chunks)

    return chunks
