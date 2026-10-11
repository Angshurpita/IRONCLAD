"""Independent citation verification engine."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ironclad.models import SearchResult, SourceCitation

logger = logging.getLogger("ironclad.citations")


@dataclass
class CitationValidationReport:
    """Outcome of independent citation verification against retrieved evidence."""

    verified_citations: List[SourceCitation] = field(default_factory=list)
    rejected_citations_count: int = 0
    is_fully_grounded: bool = True
    insufficient_evidence: bool = False
    notes: List[str] = field(default_factory=list)


def normalize_path(p: str) -> str:
    """Normalize file path for robust matching."""
    return p.strip().replace("\\", "/").lstrip("/")


def validate_citations(
    raw_citations: List[Dict[str, Any]],
    evidence: List[SearchResult],
    model_claimed_insufficient: bool = False,
) -> CitationValidationReport:
    """
    Independently verify model citations against retrieved ground truth chunks.
    
    Ensures that every citation references a real retrieved source and valid line range.
    A model's claim that evidence is sufficient does NOT override independent validation.
    """
    if not evidence:
        return CitationValidationReport(
            verified_citations=[],
            rejected_citations_count=len(raw_citations),
            is_fully_grounded=False,
            insufficient_evidence=True,
            notes=["No evidence was retrieved from the knowledge base for this query."],
        )

    if model_claimed_insufficient:
        return CitationValidationReport(
            verified_citations=[],
            rejected_citations_count=0,
            is_fully_grounded=True,
            insufficient_evidence=True,
            notes=["Model correctly abstained due to insufficient context."],
        )

    verified: List[SourceCitation] = []
    rejected_count = 0
    notes: List[str] = []

    # Map evidence by normalized path and chunk
    evidence_by_path: Dict[str, List[SearchResult]] = {}
    for item in evidence:
        norm = normalize_path(item.file_path)
        evidence_by_path.setdefault(norm, []).append(item)

    for idx, raw in enumerate(raw_citations, start=1):
        raw_file = str(raw.get("file_path", "")).strip()
        norm_raw_file = normalize_path(raw_file)
        start_line = int(raw.get("start_line", 1))
        end_line = int(raw.get("end_line", start_line))
        excerpt = str(raw.get("excerpt", "")).strip()

        # Check if file exists in retrieved evidence
        matched_chunks = evidence_by_path.get(norm_raw_file)
        if not matched_chunks:
            # Check basename fallback (e.g. "jwt.py" vs "auth/jwt.py")
            matched_chunks = [
                c for c in evidence if normalize_path(c.file_path).endswith(norm_raw_file)
            ]

        if not matched_chunks:
            logger.warning(f"Rejected hallucinated citation: file '{raw_file}' was never retrieved.")
            rejected_count += 1
            notes.append(f"Citation #{idx} referenced un-retrieved file '{raw_file}'.")
            continue

        # Check line range alignment with retrieved chunks
        valid_chunk: Optional[SearchResult] = None
        for chunk in matched_chunks:
            # Overlap check: [start_line, end_line] overlaps with chunk's [start_line, end_line]
            # Allow a small 3-line margin of error in line numbering
            overlap = not (end_line < chunk.start_line - 3 or start_line > chunk.end_line + 3)
            if overlap:
                valid_chunk = chunk
                break

        if not valid_chunk:
            # If line numbers drifted but content excerpt exists in one of the chunks
            if excerpt:
                for chunk in matched_chunks:
                    if excerpt.lower() in chunk.content.lower():
                        valid_chunk = chunk
                        start_line = chunk.start_line
                        end_line = chunk.end_line
                        break

        if valid_chunk:
            verified.append(
                SourceCitation(
                    citation_index=len(verified) + 1,
                    file_path=valid_chunk.file_path,
                    start_line=max(1, start_line),
                    end_line=max(start_line, end_line),
                    excerpt=excerpt or valid_chunk.content[:150],
                    chunk_id=valid_chunk.chunk_id,
                )
            )
        else:
            rejected_count += 1
            notes.append(
                f"Citation #{idx} in '{raw_file}' specified invalid line span {start_line}-{end_line}."
            )

    # Independent sufficiency validation:
    # If the model claimed it has sufficient evidence but every single citation was rejected,
    # we cannot claim the answer is grounded.
    insufficient = len(verified) == 0 and len(raw_citations) > 0 and rejected_count == len(raw_citations)

    return CitationValidationReport(
        verified_citations=verified,
        rejected_citations_count=rejected_count,
        is_fully_grounded=rejected_count == 0 and len(verified) > 0,
        insufficient_evidence=insufficient,
        notes=notes,
    )
