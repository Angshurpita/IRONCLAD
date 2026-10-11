"""Data models and schemas for IRONCLAD."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    SOURCE_CODE = "source_code"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    PDF = "pdf"
    CONFIG = "config"
    UNKNOWN = "unknown"


class DocumentChunk(BaseModel):
    """Represents a discrete indexed fragment of a source document."""

    chunk_id: str
    file_path: str
    content: str
    token_count: int
    char_count: int
    start_line: int
    end_line: int
    heading: Optional[str] = None
    file_hash: str
    doc_type: str = DocumentType.UNKNOWN.value


class IndexedDocument(BaseModel):
    """Metadata representing an indexed file in the knowledge base."""

    file_path: str
    file_hash: str
    size_bytes: int
    mtime: float
    doc_type: str
    chunk_count: int
    indexed_at: str


class SourceCitation(BaseModel):
    """A verified citation mapping an answer claim back to ground truth."""

    citation_index: int
    file_path: str
    start_line: int
    end_line: int
    excerpt: str
    chunk_id: Optional[str] = None


class SearchResult(BaseModel):
    """A retrieved chunk from semantic, lexical, or hybrid search."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float
    vector_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    doc_type: Optional[str] = None


class RAGResponse(BaseModel):
    """The structured result of a RAG query answering pipeline."""

    query: str
    answer: str
    citations: List[SourceCitation] = Field(default_factory=list)
    insufficient_evidence: bool = False
    retrieved_chunks_count: int = 0
    model_used: str = ""
    latency_ms: float = 0.0


class IndexStatus(BaseModel):
    """Current state and health metrics of the Ironclad knowledge base."""

    initialized: bool
    knowledge_base_path: str
    total_files: int = 0
    total_chunks: int = 0
    index_version: int = 1
    last_ingested_at: Optional[str] = None
    embedding_model: str = ""
    embedding_dimension: Optional[int] = None
    inference_provider: str = ""
    inference_model: str = ""
    is_consistent: bool = True
    inconsistencies: List[str] = Field(default_factory=list)


class DoctorCheck(BaseModel):
    """Result of an individual diagnostic check."""

    category: str
    name: str
    passed: bool
    message: str
    hint: Optional[str] = None


class DoctorReport(BaseModel):
    """Comprehensive diagnostic report from `ironclad doctor`."""

    checks: List[DoctorCheck] = Field(default_factory=list)
    all_passed: bool = True
