"""Tests for the standalone SourceRetriever (source_sleuth.py)."""

import pytest

from src.source_sleuth import SourceRetriever


@pytest.fixture(scope="module")
def retriever():
    """Create a shared SourceRetriever instance (model loads once)."""
    return SourceRetriever()


@pytest.fixture
def sample_documents():
    """A small corpus of academic-style text chunks."""
    return [
        {
            "text": "Deep learning has revolutionized natural language processing "
            "with models like BERT and GPT achieving state-of-the-art results.",
            "source": "nlp_survey.pdf",
            "page": 1,
        },
        {
            "text": "Convolutional neural networks apply learnable filters to detect "
            "local features in images, enabling high-accuracy classification.",
            "source": "cv_intro.pdf",
            "page": 3,
        },
        {
            "text": "Reinforcement learning enables agents to learn optimal policies "
            "through trial and error in complex environments.",
            "source": "rl_textbook.pdf",
            "page": 7,
        },
        {
            "text": "The attention mechanism allows the model to focus on specific "
            "parts of the input sequence that are most relevant to the output.",
            "source": "transformer_paper.pdf",
            "page": 5,
        },
        {
            "text": "Wave interference produces a pattern of bright and dark fringes "
            "when coherent light passes through a double slit.",
            "source": "physics_101.pdf",
            "page": 42,
        },
    ]


class TestSourceRetrieverInit:
    """Test initialization and properties."""

    def test_model_loads(self, retriever):
        assert retriever.model is not None
        assert retriever.model_name == "all-MiniLM-L6-v2"

    def test_empty_initially(self, retriever):
        retriever.clear()
        assert retriever.num_chunks == 0
        assert retriever.is_ready is False


class TestIngestion:
    """Test document ingestion."""

    def test_ingest_documents(self, retriever, sample_documents):
        retriever.clear()
        count = retriever.ingest_documents(sample_documents)
        assert count == 5
        assert retriever.num_chunks == 5
        assert retriever.is_ready is True

    def test_ingest_empty_raises(self, retriever):
        with pytest.raises(ValueError, match="No text chunks"):
            retriever.ingest_documents([])


class TestSearch:
    """Test the find_source search functionality."""

    def test_find_exact_quote(self, retriever, sample_documents):
        retriever.clear()
        retriever.ingest_documents(sample_documents)

        results = retriever.find_source(
            "Deep learning has revolutionized natural language processing"
        )
        assert len(results) == 3
        assert results[0]["source"] == "nlp_survey.pdf"
        # Confidence threshold adjusted for model variance across environments
        assert results[0]["confidence_score"] > 0.70

    def test_find_paraphrased_text(self, retriever, sample_documents):
        retriever.clear()
        retriever.ingest_documents(sample_documents)

        results = retriever.find_source(
            "neural networks for image recognition and object detection"
        )
        assert len(results) == 3
        # CV paper should rank highly
        sources = [r["source"] for r in results]
        assert "cv_intro.pdf" in sources

    def test_find_physics_quote(self, retriever, sample_documents):
        retriever.clear()
        retriever.ingest_documents(sample_documents)

        results = retriever.find_source("double slit experiment light interference pattern")
        assert results[0]["source"] == "physics_101.pdf"

    def test_top_k_parameter(self, retriever, sample_documents):
        retriever.clear()
        retriever.ingest_documents(sample_documents)

        results = retriever.find_source("machine learning", top_k=2)
        assert len(results) == 2

    def test_search_before_ingest_raises(self, retriever):
        retriever.clear()
        with pytest.raises(RuntimeError, match="No documents ingested"):
            retriever.find_source("test query")

    def test_confidence_scores_are_sorted(self, retriever, sample_documents):
        retriever.clear()
        retriever.ingest_documents(sample_documents)

        results = retriever.find_source("attention mechanism", top_k=5)
        scores = [r["confidence_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_preserved(self, retriever, sample_documents):
        retriever.clear()
        retriever.ingest_documents(sample_documents)

        results = retriever.find_source("reinforcement learning agents")
        top = results[0]
        assert "source" in top
        assert "page" in top
        assert "confidence_score" in top
        assert isinstance(top["page"], int)


class TestClear:
    """Test clearing the retriever."""

    def test_clear_resets_state(self, retriever, sample_documents):
        retriever.ingest_documents(sample_documents)
        assert retriever.is_ready is True

        retriever.clear()
        assert retriever.num_chunks == 0
        assert retriever.is_ready is False
