"""Persistent vector store using embedded ChromaDB."""

import logging
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings as ChromaSettings

from ironclad.models import DocumentChunk, SearchResult

logger = logging.getLogger("ironclad.vector")


class ChromaVectorStore:
    """
    Embedded persistent vector store using ChromaDB.
    Maintains chunk vectors and document metadata across restarts.
    """

    COLLECTION_NAME = "ironclad_chunks"

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True,
            ),
        )
        # Cosine distance space
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return total number of vectors stored."""
        return self._collection.count()

    def add_or_update_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Upsert chunk embeddings and associated metadata."""
        if not chunks:
            return

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, object]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append(
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "heading": chunk.heading or "",
                    "file_hash": chunk.file_hash,
                    "doc_type": chunk.doc_type,
                }
            )

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete vectors by chunk IDs."""
        if not chunk_ids:
            return
        try:
            self._collection.delete(ids=chunk_ids)
        except Exception as e:
            logger.warning(f"Error deleting chunks from vector store: {e}")

    def clear(self) -> None:
        """Reset and wipe the vector collection."""
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.warning(f"Error resetting vector collection: {e}")

    def search_vector(
        self,
        query_embedding: List[float],
        limit: int = 20,
    ) -> List[SearchResult]:
        """
        Execute nearest-neighbor vector search in cosine space.
        Returns ranked SearchResult objects.
        """
        if self._collection.count() == 0:
            return []

        actual_limit = min(limit, self._collection.count())
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_limit,
            include=["documents", "metadatas", "distances"],
        )

        results: List[SearchResult] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        for rank, (chunk_id, doc_text, meta, dist) in enumerate(
            zip(ids, docs, metas, distances, strict=False), start=1
        ):
            # Chroma cosine distance is in [0, 2]; similarity = 1.0 - (dist / 2.0) or 1.0 - dist
            sim_score = max(0.0, 1.0 - float(dist))

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    file_path=str(meta.get("file_path", "")),
                    start_line=int(meta.get("start_line", 1)),
                    end_line=int(meta.get("end_line", 1)),
                    content=doc_text,
                    score=round(sim_score, 4),
                    vector_rank=rank,
                    doc_type=str(meta.get("doc_type", "unknown")),
                )
            )

        return results
