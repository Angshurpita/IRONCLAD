"""Transactional SQLite metadata store with FTS5 lexical indexing and concurrency locking."""

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional

from ironclad.errors import ConcurrentAccessError, KnowledgeBaseCorruptedError
from ironclad.models import DocumentChunk, IndexedDocument

logger = logging.getLogger("ironclad.storage")


class KnowledgeBaseLock:
    """Process-level concurrency lock using a lockfile containing PID and timestamp."""

    def __init__(self, lock_file: Path, timeout_seconds: float = 10.0):
        self.lock_file = lock_file
        self.timeout_seconds = timeout_seconds
        self.acquired = False

    def acquire(self) -> None:
        """Acquire the lock, detecting and cleaning up stale locks if process died."""
        start_time = time.time()
        while time.time() - start_time < self.timeout_seconds:
            try:
                # O_CREAT | O_EXCL ensures atomic creation
                fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    lock_data = {
                        "pid": os.getpid(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    f.write(json.dumps(lock_data))
                self.acquired = True
                return
            except FileExistsError:
                # Inspect existing lock file for staleness
                if self._is_stale_lock():
                    logger.warning(f"Removing stale lock file at {self.lock_file}")
                    try:
                        self.lock_file.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue
                time.sleep(0.2)

        raise ConcurrentAccessError(
            f"Another Ironclad process holds the lock on {self.lock_file}.",
            hint="Wait for the running command to finish or delete the lock file if the process died.",
        )

    def release(self) -> None:
        """Release the lock if held by this instance."""
        if self.acquired and self.lock_file.exists():
            try:
                self.lock_file.unlink(missing_ok=True)
                self.acquired = False
            except OSError as e:
                logger.warning(f"Could not remove lock file {self.lock_file}: {e}")

    def _is_stale_lock(self) -> bool:
        """Check if the process that created the lock is no longer running."""
        try:
            with open(self.lock_file, "r") as f:
                data = json.load(f)
                pid = data.get("pid")
                if pid and isinstance(pid, int):
                    # Check if process exists (signal 0 does not terminate)
                    try:
                        os.kill(pid, 0)
                        return False  # Process is still alive
                    except OSError:
                        return True  # Process does not exist
        except Exception:
            return True
        return False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class MetadataStore:
    """Manages transactional SQLite storage for indexed documents and chunk metadata."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_schema()

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Provides a SQLite connection with foreign keys and WAL enabled."""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 10000;")
            yield conn
        except sqlite3.DatabaseError as e:
            raise KnowledgeBaseCorruptedError(
                f"SQLite error accessing knowledge base at {self.db_path}: {e}",
                hint="Verify database integrity using `ironclad doctor`.",
            ) from e
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Initialize SQLite database tables, FTS5 virtual table, and triggers."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metadata_info (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        file_path TEXT PRIMARY KEY,
                        file_hash TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        mtime REAL NOT NULL,
                        doc_type TEXT NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        indexed_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        content TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        char_count INTEGER NOT NULL,
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        heading TEXT,
                        file_hash TEXT NOT NULL,
                        doc_type TEXT NOT NULL,
                        FOREIGN KEY (file_path) REFERENCES documents(file_path) ON DELETE CASCADE
                    );
                    """
                )
                # FTS5 full-text lexical search table
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        file_path UNINDEXED,
                        content,
                        heading,
                        tokenize = 'porter unicode61'
                    );
                    """
                )
                # Index on documents file_hash for rapid change detection
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);"
                )

    def get_indexed_documents(self) -> Dict[str, IndexedDocument]:
        """Load all currently indexed documents mapped by relative file path."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_path, file_hash, size_bytes, mtime, doc_type, chunk_count, indexed_at FROM documents;"
            )
            docs: Dict[str, IndexedDocument] = {}
            for row in cursor.fetchall():
                docs[row["file_path"]] = IndexedDocument(
                    file_path=row["file_path"],
                    file_hash=row["file_hash"],
                    size_bytes=row["size_bytes"],
                    mtime=row["mtime"],
                    doc_type=row["doc_type"],
                    chunk_count=row["chunk_count"],
                    indexed_at=row["indexed_at"],
                )
            return docs

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Fetch all stored chunks across all documents."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT chunk_id, file_path, content, token_count, char_count,
                       start_line, end_line, heading, file_hash, doc_type
                FROM chunks;
                """
            )
            chunks: List[DocumentChunk] = []
            for row in cursor.fetchall():
                chunks.append(
                    DocumentChunk(
                        chunk_id=row["chunk_id"],
                        file_path=row["file_path"],
                        content=row["content"],
                        token_count=row["token_count"],
                        char_count=row["char_count"],
                        start_line=row["start_line"],
                        end_line=row["end_line"],
                        heading=row["heading"],
                        file_hash=row["file_hash"],
                        doc_type=row["doc_type"],
                    )
                )
            return chunks

    def add_or_update_document(
        self,
        doc: IndexedDocument,
        chunks: List[DocumentChunk],
    ) -> None:
        """Atomically update a document and its chunks in SQLite and FTS5."""
        with self.connect() as conn:
            with conn:
                cursor = conn.cursor()

                # Clean up old FTS5 and chunk records for this file path
                cursor.execute("SELECT chunk_id FROM chunks WHERE file_path = ?;", (doc.file_path,))
                old_chunk_ids = [r[0] for r in cursor.fetchall()]
                if old_chunk_ids:
                    cursor.execute("DELETE FROM chunks WHERE file_path = ?;", (doc.file_path,))
                    cursor.executemany(
                        "DELETE FROM chunks_fts WHERE chunk_id = ?;",
                        [(cid,) for cid in old_chunk_ids],
                    )

                # Upsert document record
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO documents 
                    (file_path, file_hash, size_bytes, mtime, doc_type, chunk_count, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        doc.file_path,
                        doc.file_hash,
                        doc.size_bytes,
                        doc.mtime,
                        doc.doc_type,
                        len(chunks),
                        doc.indexed_at,
                    ),
                )

                # Insert chunks
                if chunks:
                    chunk_records = [
                        (
                            c.chunk_id,
                            c.file_path,
                            c.content,
                            c.token_count,
                            c.char_count,
                            c.start_line,
                            c.end_line,
                            c.heading,
                            c.file_hash,
                            c.doc_type,
                        )
                        for c in chunks
                    ]
                    cursor.executemany(
                        """
                        INSERT INTO chunks
                        (chunk_id, file_path, content, token_count, char_count,
                         start_line, end_line, heading, file_hash, doc_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        chunk_records,
                    )

                    # Insert into FTS5
                    fts_records = [
                        (c.chunk_id, c.file_path, c.content, c.heading or "")
                        for c in chunks
                    ]
                    cursor.executemany(
                        """
                        INSERT INTO chunks_fts (chunk_id, file_path, content, heading)
                        VALUES (?, ?, ?, ?);
                        """,
                        fts_records,
                    )

    def delete_document(self, file_path: str) -> None:
        """Atomically delete a document and its chunks from SQLite and FTS5."""
        with self.connect() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id FROM chunks WHERE file_path = ?;", (file_path,))
                chunk_ids = [r[0] for r in cursor.fetchall()]

                cursor.execute("DELETE FROM documents WHERE file_path = ?;", (file_path,))
                cursor.execute("DELETE FROM chunks WHERE file_path = ?;", (file_path,))
                if chunk_ids:
                    cursor.executemany(
                        "DELETE FROM chunks_fts WHERE chunk_id = ?;",
                        [(cid,) for cid in chunk_ids],
                    )

    def update_metadata(self, key: str, value: str) -> None:
        """Update or insert a key-value pair in metadata_info table."""
        with self.connect() as conn:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO metadata_info (key, value) VALUES (?, ?);",
                    (key, value),
                )

    def get_metadata(self, key: str) -> Optional[str]:
        """Fetch metadata value by key."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata_info WHERE key = ?;", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None

    def clear(self) -> None:
        """Atomically clear all documents and chunks from the store."""
        with self.connect() as conn:
            with conn:
                conn.execute("DELETE FROM documents;")
                conn.execute("DELETE FROM chunks;")
                conn.execute("DELETE FROM chunks_fts;")
