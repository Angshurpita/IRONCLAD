"""Context ranking, evidence selection, and token budgeting."""

import logging
from typing import List, Tuple

from ironclad.ingestion.chunker import estimate_tokens
from ironclad.models import SearchResult

logger = logging.getLogger("ironclad.ranking")


def select_context_by_budget(
    candidates: List[SearchResult],
    max_context_tokens: int = 4000,
    reserve_tokens: int = 1000,
) -> Tuple[List[SearchResult], int]:
    """
    Select top-ranked evidence chunks according to a strict token budget.
    
    Ensures LLM context window is never exceeded while maximizing high-relevance evidence.
    Returns (selected_chunks, total_context_tokens).
    """
    available_budget = max(1, max_context_tokens - reserve_tokens)
    selected: List[SearchResult] = []
    total_tokens = 0

    for cand in candidates:
        cand_tokens = estimate_tokens(cand.content)

        if total_tokens + cand_tokens <= available_budget:
            selected.append(cand)
            total_tokens += cand_tokens
        elif not selected:
            # If the first chunk alone exceeds budget, truncate gracefully
            chars_budget = available_budget * 4
            truncated_content = cand.content[:chars_budget] + "\n... [truncated to fit token budget]"
            truncated_cand = SearchResult(
                chunk_id=cand.chunk_id,
                file_path=cand.file_path,
                start_line=cand.start_line,
                end_line=cand.end_line,
                content=truncated_content,
                score=cand.score,
                vector_rank=cand.vector_rank,
                lexical_rank=cand.lexical_rank,
                doc_type=cand.doc_type,
            )
            selected.append(truncated_cand)
            total_tokens = available_budget
            break
        else:
            # Cannot fit more whole chunks within budget
            break

    logger.debug(
        f"Selected {len(selected)} chunks totaling ~{total_tokens} tokens (budget: {available_budget})."
    )
    return selected, total_tokens
