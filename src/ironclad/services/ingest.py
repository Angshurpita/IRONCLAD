"""Ingestion service coordinating discovery, parsing, chunking, and atomic storage."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ironclad.config import IroncladSettings
from ironclad.errors import KnowledgeBaseNotFoundError
from ironclad.ingestion.chunker import chunk_document
from ironclad.ingestion.discovery import DiscoveredFile, discover_files
from ironclad.ingestion.parsers import get_parser_for_type
from ironclad.models import IndexedDocument
from ironclad.storage.metadata import KnowledgeBaseLock, MetadataStore

logger = logging.getLogger("ironclad.ingest")


@dataclass
class IngestionSummary:
    """Summary metrics of an ingestion run."""

    mode: str
    target_path: str
    files_added: int = 0
    files_updated: int = 0
    files_deleted: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    total_files: int = 0
    total_chunks: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "target_path": self.target_path,
            "files_added": self.files_added,
            "files_updated": self.files_updated,
            "files_deleted": self.files_deleted,
            "files_skipped": self.files_skipped,
            "chunks_created": self.chunks_created,
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors,
        }


def run_ingestion(
    settings: IroncladSettings,
    target_path: Optional[Path] = None,
    update: bool = False,
    reindex: bool = False,
    extra_exclusions: Optional[List[str]] = None,
) -> IngestionSummary:
    """
    Execute document ingestion and index synchronization.
    Safely acquires lock, skips unchanged files, updates changed files, and purges deleted files.
    """
    if not settings.is_initialized():
        raise KnowledgeBaseNotFoundError(
            f"Knowledge base not initialized at {settings.project_root}.",
            hint=f"Run `ironclad init {target_path or settings.project_root}` first.",
        )

    scan_dir = (target_path or settings.project_root).resolve()
    start_time = datetime.now(timezone.utc)
    summary = IngestionSummary(
        mode="reindex" if reindex else ("incremental" if update else "full"),
        target_path=str(scan_dir),
    )

    metadata_store = MetadataStore(settings.metadata_db_path)
    from ironclad.embeddings.provider import get_embedding_provider
    from ironclad.retrieval.vector import ChromaVectorStore

    vector_store = ChromaVectorStore(settings.vector_store_path)
    embedder = get_embedding_provider(settings)

    # Acquire concurrency process lock
    with KnowledgeBaseLock(settings.lock_file_path):
        # Detect if configured embedding model has changed
        stored_emb = metadata_store.get_metadata("embedding_model")
        if stored_emb and stored_emb != settings.embedding_model:
            logger.warning(
                f"Embedding model changed from '{stored_emb}' to '{settings.embedding_model}'. "
                "Wiping previous vector index to ensure dimensional consistency."
            )
            reindex = True
            summary.mode = "reindex"

        if reindex:
            metadata_store.clear()
            vector_store.clear()
            summary.mode = "reindex"

        existing_docs = metadata_store.get_indexed_documents()
        discovered = discover_files(
            root_dir=scan_dir,
            settings=settings,
            extra_exclusions=extra_exclusions,
        )

        discovered_by_rel: Dict[str, DiscoveredFile] = {d.rel_path: d for d in discovered}

        # Identify changes
        files_to_process: List[DiscoveredFile] = []
        unchanged_count = 0

        for rel_path, disc_file in discovered_by_rel.items():
            if rel_path in existing_docs:
                existing = existing_docs[rel_path]
                if existing.file_hash == disc_file.file_hash:
                    # Unchanged file - skip
                    unchanged_count += 1
                else:
                    # Content modified
                    files_to_process.append(disc_file)
            else:
                # Newly added file
                files_to_process.append(disc_file)

        # Identify deleted files (in update mode or full sync)
        deleted_rel_paths = set(existing_docs.keys()) - set(discovered_by_rel.keys())

        summary.files_skipped = unchanged_count

        # Remove deleted files
        for del_path in deleted_rel_paths:
            try:
                # Find chunks to remove from vector store
                with metadata_store.connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT chunk_id FROM chunks WHERE file_path = ?;", (del_path,))
                    del_chunk_ids = [r[0] for r in cursor.fetchall()]

                metadata_store.delete_document(del_path)
                if del_chunk_ids:
                    vector_store.delete_chunks(del_chunk_ids)

                summary.files_deleted += 1
                logger.info(f"Removed deleted file: {del_path}")
            except Exception as e:
                err_msg = f"Failed to delete {del_path}: {e}"
                logger.error(err_msg)
                summary.errors.append(err_msg)

        # Process new and modified files
        for disc_file in files_to_process:
            is_new = disc_file.rel_path not in existing_docs
            try:
                parser = get_parser_for_type(disc_file.doc_type)
                parsed_doc = parser.parse(disc_file)

                # Generate structure-aware chunks
                chunks = chunk_document(
                    doc=parsed_doc,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )

                # Generate vector embeddings
                if chunks:
                    chunk_texts = [c.content for c in chunks]
                    embeddings = embedder.embed_documents(chunk_texts)
                    vector_store.add_or_update_chunks(chunks, embeddings)

                now_iso = datetime.now(timezone.utc).isoformat()
                indexed_doc = IndexedDocument(
                    file_path=disc_file.rel_path,
                    file_hash=disc_file.file_hash,
                    size_bytes=disc_file.size_bytes,
                    mtime=disc_file.mtime,
                    doc_type=disc_file.doc_type.value,
                    chunk_count=len(chunks),
                    indexed_at=now_iso,
                )

                metadata_store.add_or_update_document(indexed_doc, chunks)

                if is_new:
                    summary.files_added += 1
                else:
                    summary.files_updated += 1

                summary.chunks_created += len(chunks)

            except Exception as e:
                err_msg = f"Error indexing {disc_file.rel_path}: {e}"
                logger.error(err_msg)
                summary.errors.append(err_msg)

        # Record ingestion timestamp and embedding model in metadata
        now_str = datetime.now(timezone.utc).isoformat()
        metadata_store.update_metadata("last_ingested_at", now_str)
        metadata_store.update_metadata("embedding_model", settings.embedding_model)

        # Compute updated totals
        final_docs = metadata_store.get_indexed_documents()
        all_chunks = metadata_store.get_all_chunks()
        summary.total_files = len(final_docs)
        summary.total_chunks = len(all_chunks)
        summary.duration_seconds = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

    return summary
