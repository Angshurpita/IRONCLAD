"""Reciprocal Rank Fusion (RRF) for combining vector and lexical rankings."""

import logging
from typing import Dict, List, Optional

from ironclad.models import SearchResult

logger = logging.getLogger("ironclad.fusion")


def reciprocal_rank_fusion(
    vector_results: List[SearchResult],
    lexical_results: List[SearchResult],
    k: int = 60,
    top_n: Optional[int] = None,
) -> List[SearchResult]:
    """
    Merge and deduplicate semantic vector candidates and lexical candidates using RRF.
    
    Formula:
        RRF_Score(d) = sum_{m in {vector, lexical}} 1 / (k + rank_m(d))
    """
    scores: Dict[str, float] = {}
    combined_docs: Dict[str, SearchResult] = {}

    # Accumulate vector ranks
    for rank, res in enumerate(vector_results, start=1):
        cid = res.chunk_id
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
        if cid not in combined_docs:
            res.vector_rank = rank
            combined_docs[cid] = res
        else:
            combined_docs[cid].vector_rank = rank

    # Accumulate lexical ranks
    for rank, res in enumerate(lexical_results, start=1):
        cid = res.chunk_id
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
        if cid not in combined_docs:
            res.lexical_rank = rank
            combined_docs[cid] = res
        else:
            combined_docs[cid].lexical_rank = rank

    # Sort candidates by combined RRF score in descending order
    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused_results: List[SearchResult] = []
    for cid in sorted_chunk_ids:
        doc = combined_docs[cid]
        fused_score = scores[cid]
        # Return updated SearchResult copy with final score
        fused_results.append(
            SearchResult(
                chunk_id=doc.chunk_id,
                file_path=doc.file_path,
                start_line=doc.start_line,
                end_line=doc.end_line,
                content=doc.content,
                score=round(fused_score, 6),
                vector_rank=doc.vector_rank,
                lexical_rank=doc.lexical_rank,
                doc_type=doc.doc_type,
            )
        )

    if top_n is not None:
        return fused_results[:top_n]
    return fused_results
