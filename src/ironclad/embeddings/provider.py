"""Local embedding provider interface with Chroma ONNX runtime and offline fallback."""

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from typing import List

from ironclad.config import IroncladSettings
from ironclad.errors import EmbeddingError

logger = logging.getLogger("ironclad.embeddings")


class BaseEmbeddingProvider(ABC):
    """Abstract interface for local embedding generation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vector."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a batch of document texts."""
        pass


class ChromaONNXEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using Chroma's ONNX runtime (all-MiniLM-L6-v2).
    Runs 100% locally on CPU without requiring PyTorch or cloud APIs.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._dimension = 384
        self._ef = None

    def _get_ef(self):
        if self._ef is None:
            try:
                import chromadb.utils.embedding_functions as ef
                self._ef = ef.DefaultEmbeddingFunction()
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to initialize local Chroma embedding model: {e}",
                    hint="Ensure chromadb and onnxruntime are installed correctly.",
                ) from e
        return self._ef

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, query: str) -> List[float]:
        ef = self._get_ef()
        try:
            res = ef([query])
            return [float(x) for x in res[0]]
        except Exception as e:
            raise EmbeddingError(f"Error generating embedding for query: {e}") from e

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        if not documents:
            return []
        ef = self._get_ef()
        try:
            res = ef(documents)
            return [[float(x) for x in vec] for vec in res]
        except Exception as e:
            raise EmbeddingError(f"Error generating embeddings for document batch: {e}") from e


class DeterministicHashEmbeddingProvider(BaseEmbeddingProvider):
    """
    Offline deterministic n-gram hash projection embedding provider.
    Used for tests, airgapped offline environments, or rapid validation.
    """

    def __init__(self, dimension: int = 384, model_name: str = "hash-projector-384"):
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_single(self, text: str) -> List[float]:
        # Hash text words into dimension buckets and L2 normalize
        vec = [0.0] * self._dimension
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dimension
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec

    def embed_query(self, query: str) -> List[float]:
        return self._embed_single(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self._embed_single(doc) for doc in documents]


def get_embedding_provider(
    settings: IroncladSettings,
    force_fallback: bool = False,
) -> BaseEmbeddingProvider:
    """
    Resolve and instantiate the embedding provider based on configuration.
    Falls back gracefully if ONNX runtime is unavailable.
    """
    if force_fallback or settings.embedding_provider == "hash":
        return DeterministicHashEmbeddingProvider()

    try:
        provider = ChromaONNXEmbeddingProvider()
        return provider
    except Exception as e:
        logger.warning(
            f"Local ONNX embedding provider unavailable ({e}). Falling back to deterministic embedding."
        )
        return DeterministicHashEmbeddingProvider()
