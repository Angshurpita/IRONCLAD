"""Unit tests for embedding providers."""

from ironclad.config import IroncladSettings
from ironclad.embeddings.provider import (
    ChromaONNXEmbeddingProvider,
    DeterministicHashEmbeddingProvider,
    get_embedding_provider,
)


def test_deterministic_hash_provider():
    """Verify offline deterministic hash provider properties."""
    provider = DeterministicHashEmbeddingProvider(dimension=384)
    assert provider.dimension == 384

    vec = provider.embed_query("authentication token")
    assert len(vec) == 384
    assert isinstance(vec[0], float)

    batch = provider.embed_documents(["doc 1", "doc 2"])
    assert len(batch) == 2
    assert len(batch[0]) == 384


def test_chroma_onnx_provider():
    """Verify local Chroma ONNX model produces 384-dimensional embeddings."""
    provider = ChromaONNXEmbeddingProvider()
    assert provider.dimension == 384

    vec = provider.embed_query("retrieval augmented generation")
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)

    batch = provider.embed_documents(["hello world", "test code"])
    assert len(batch) == 2
    assert len(batch[0]) == 384


def test_get_embedding_provider_factory():
    """Verify factory returns provider based on settings."""
    settings = IroncladSettings(embedding_provider="hash")
    p1 = get_embedding_provider(settings)
    assert isinstance(p1, DeterministicHashEmbeddingProvider)

    settings2 = IroncladSettings(embedding_provider="local")
    p2 = get_embedding_provider(settings2)
    assert isinstance(p2, ChromaONNXEmbeddingProvider)
