"""Lexical search engine using SQLite FTS5 with BM25 ranking."""

import logging
import re
from typing import List

from ironclad.models import SearchResult
from ironclad.storage.metadata import MetadataStore

logger = logging.getLogger("ironclad.lexical")

# Characters with special syntax meaning in SQLite FTS5 or word delimiters
FTS5_SPECIAL_CHARS = re.compile(r'["\'\*\+\-\^\:\(\)\{\}\~\.\,\;\!\?\/\\\[\]]')


def sanitize_fts5_query(query: str) -> str:
    """
    Sanitize raw search terms to avoid FTS5 syntax errors while preserving keywords.
    Turns query terms into safe prefix match clauses.
    """
    # Replace special chars with whitespace
    cleaned = FTS5_SPECIAL_CHARS.sub(" ", query).strip()
    words = [w for w in cleaned.split() if len(w) > 1]
    if not words:
        return ""
    # Safe match with term prefixes or phrase matching
    terms = [f'"{w}"*' for w in words]
    return " OR ".join(terms)


def search_lexical(
    metadata_store: MetadataStore,
    query: str,
    limit: int = 20,
) -> List[SearchResult]:
    """
    Execute full-text lexical search using SQLite FTS5 with BM25 ranking.
    Returns ranked SearchResult objects.
    """
    sanitized = sanitize_fts5_query(query)
    if not sanitized:
        logger.debug(f"Query '{query}' produced empty sanitized FTS5 expression.")
        return []

    results: List[SearchResult] = []
    try:
        with metadata_store.connect() as conn:
            cursor = conn.cursor()
            sql = """
                SELECT c.chunk_id, c.file_path, c.start_line, c.end_line, c.content, c.doc_type,
                       bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts) ASC
                LIMIT ?;
            """
            cursor.execute(sql, (sanitized, limit))
            rows = cursor.fetchall()

            for rank, row in enumerate(rows, start=1):
                # In SQLite FTS5, lower bm25 score means higher relevance (e.g. -2.5 is better than -0.5)
                raw_bm25 = float(row["bm25_score"])
                normalized_score = abs(raw_bm25)

                results.append(
                    SearchResult(
                        chunk_id=row["chunk_id"],
                        file_path=row["file_path"],
                        start_line=row["start_line"],
                        end_line=row["end_line"],
                        content=row["content"],
                        score=normalized_score,
                        lexical_rank=rank,
                        doc_type=row["doc_type"],
                    )
                )
    except Exception as e:
        logger.error(f"Error during lexical search for '{query}': {e}")

    return results
