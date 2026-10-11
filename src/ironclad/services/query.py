"""Retrieval query service orchestrating hybrid search and evidence selection."""

import logging
from typing import List, Optional, Tuple

from ironclad.config import IroncladSettings
from ironclad.embeddings.provider import BaseEmbeddingProvider, get_embedding_provider
from ironclad.models import SearchResult
from ironclad.retrieval.fusion import reciprocal_rank_fusion
from ironclad.retrieval.lexical import search_lexical
from ironclad.retrieval.ranking import select_context_by_budget
from ironclad.retrieval.vector import ChromaVectorStore
from ironclad.storage.metadata import MetadataStore

logger = logging.getLogger("ironclad.query")


def execute_hybrid_search(
    settings: IroncladSettings,
    query: str,
    top_k: int = 15,
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    vector_store: Optional[ChromaVectorStore] = None,
    metadata_store: Optional[MetadataStore] = None,
) -> Tuple[List[SearchResult], List[SearchResult], int]:
    """
    Execute complete hybrid retrieval pipeline:
    1. Embed query with local embedding provider.
    2. Vector search via ChromaDB.
    3. Lexical BM25 search via SQLite FTS5.
    4. Reciprocal Rank Fusion (RRF) deduplication and combination.
    5. Context token budget pruning.
    
    Returns:
        (budget_selected_chunks, all_fused_chunks, total_context_tokens)
    """
    meta_store = metadata_store or MetadataStore(settings.metadata_db_path)
    vec_store = vector_store or ChromaVectorStore(settings.vector_store_path)
    embedder = embedding_provider or get_embedding_provider(settings)

    # 1. Lexical search
    lexical_results = search_lexical(meta_store, query, limit=top_k)

    # 2. Vector search
    vector_results: List[SearchResult] = []
    try:
        query_embedding = embedder.embed_query(query)
        vector_results = vec_store.search_vector(query_embedding, limit=top_k)
    except Exception as e:
        logger.warning(f"Vector search failed ({e}); falling back to lexical search only.")

    # 3. Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        lexical_results=lexical_results,
        k=60,
        top_n=top_k,
    )

    # 4. Token budget selection
    selected_context, total_tokens = select_context_by_budget(
        fused_results,
        max_context_tokens=settings.max_context_tokens,
    )

    return selected_context, fused_results, total_tokens
