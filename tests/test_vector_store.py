"""Tests for the Vector Store module."""

import pytest

from src.pdf_processor import TextChunk
from src.vector_store import EMBEDDING_DIM, VectorStore


# Fixtures


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore with a temp data directory."""
    return VectorStore(data_dir=tmp_path / "test_data")


@pytest.fixture
def sample_chunks():
    """Create a set of sample text chunks for testing."""
    return [
        TextChunk(
            text="Deep learning has revolutionized natural language processing.",
            filename="paper_nlp.pdf",
            page=1,
            chunk_index=0,
            start_char=0,
            end_char=60,
        ),
        TextChunk(
            text="Convolutional neural networks are effective for image classification.",
            filename="paper_cv.pdf",
            page=3,
            chunk_index=0,
            start_char=0,
            end_char=67,
        ),
        TextChunk(
            text="Reinforcement learning enables agents to learn optimal policies.",
            filename="paper_rl.pdf",
            page=2,
            chunk_index=0,
            start_char=0,
            end_char=63,
        ),
    ]


# Tests: Core Operations


class TestVectorStoreCore:
    def test_add_chunks(self, store, sample_chunks):
        added = store.add_chunks(sample_chunks)
        assert added == 3
        assert store.total_chunks == 3

    def test_add_empty_list(self, store):
        added = store.add_chunks([])
        assert added == 0
        assert store.total_chunks == 0

    def test_search_returns_results(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        results = store.search("natural language processing", top_k=3)
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_search_empty_store(self, store):
        results = store.search("anything")
        assert results == []

    def test_search_relevance(self, store, sample_chunks):
        """The NLP chunk should rank highest for an NLP query."""
        store.add_chunks(sample_chunks)
        results = store.search("deep learning for NLP")
        top_result = results[0]
        assert top_result["filename"] == "paper_nlp.pdf"

    def test_top_k_clamped(self, store, sample_chunks):
        """top_k should be clamped to the number of available chunks."""
        store.add_chunks(sample_chunks)
        results = store.search("test", top_k=100)
        assert len(results) <= 3


# Tests: Persistence


class TestPersistence:
    def test_save_and_load(self, tmp_path, sample_chunks):
        data_dir = tmp_path / "persist_test"

        # Save
        store1 = VectorStore(data_dir=data_dir)
        store1.add_chunks(sample_chunks)
        store1.save()

        # Load into a new instance
        store2 = VectorStore(data_dir=data_dir)
        loaded = store2.load()

        assert loaded is True
        assert store2.total_chunks == 3
        assert store2.ingested_files == {"paper_nlp.pdf", "paper_cv.pdf", "paper_rl.pdf"}

    def test_load_nonexistent(self, tmp_path):
        store = VectorStore(data_dir=tmp_path / "empty")
        assert store.load() is False


# Tests: Utilities


class TestUtilities:
    def test_clear(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        store.clear()
        assert store.total_chunks == 0
        assert store.ingested_files == set()

    def test_remove_file(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        removed = store.remove_file("paper_cv.pdf")
        assert removed == 1
        assert store.total_chunks == 2
        assert "paper_cv.pdf" not in store.ingested_files

    def test_remove_nonexistent_file(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        removed = store.remove_file("nonexistent.pdf")
        assert removed == 0

    def test_get_stats(self, store, sample_chunks):
        store.add_chunks(sample_chunks)
        stats = store.get_stats()
        assert stats["total_chunks"] == 3
        assert stats["num_files"] == 3
        assert stats["embedding_dim"] == EMBEDDING_DIM
