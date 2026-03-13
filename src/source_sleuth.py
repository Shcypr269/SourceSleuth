# ruff: noqa: RUF002  # Mathematical notation requires special Unicode chars
"""
SourceSleuth Core — Standalone Semantic Search Engine.

This module implements the ``SourceRetriever`` class, the core semantic
search engine that powers SourceSleuth. It is designed to be used
independently of the MCP server, making it easy to integrate into
scripts, notebooks, or other applications.

Model Architecture Documentation (AI/ML Hackathon Requirement):
    - Model: ``all-MiniLM-L6-v2`` from Sentence-Transformers
    - Why this model?
        1. Runs efficiently on CPU (~80 MB footprint).
        2. Produces 384-dimensional sentence embeddings.
        3. Trained on 1B+ sentence pairs — strong zero-shot performance.
        4. Ideal for a student laptop: no GPU, no API keys, no cloud.
    - Similarity metric: Cosine Similarity
        Cosine Similarity(A, B) = (A · B) / (||A|| × ||B||)
        Normalized dot-product captures semantic direction, not magnitude.
    - Chunking strategy:
        500-token chunks with 50-token overlap to preserve context at
        chunk boundaries. Approximate 4-char-per-token heuristic.

Usage:
    >>> from src.source_sleuth import SourceRetriever
    >>> retriever = SourceRetriever()
    >>> retriever.ingest_documents([
    ...     {"text": "Attention is all you need.", "source": "vaswani2017.pdf", "page": 1}
    ... ])
    >>> results = retriever.find_source("transformer self-attention mechanism")
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger("sourcesleuth.core")


class SourceRetriever:
    """
    Handles the embedding and retrieval of text chunks to find lost
    academic sources.

    Designed to be run locally to protect student privacy and avoid
    API costs. This is the standalone version of the search engine
    that does not require an MCP host or FAISS — it operates purely
    with NumPy and scikit-learn for maximum portability.

    Attributes:
        model: The SentenceTransformer embedding model.
        document_chunks: List of ingested document chunk metadata.
        document_embeddings: NumPy array of L2-normalized embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding model.

        Architectural Reasoning:
            We chose ``all-MiniLM-L6-v2`` because it balances speed and
            accuracy for semantic similarity tasks. It produces 384-dim
            embeddings, runs on CPU in ~50ms per sentence, and is small
            enough (~80 MB) for a student laptop. Larger models like
            ``all-mpnet-base-v2`` (768-dim) offer marginal accuracy
            gains but double the memory and inference time — a poor
            trade-off for local, interactive use.

        Args:
            model_name: The HuggingFace model identifier to use.
        """
        logger.info("Loading embedding model: %s …", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.document_chunks: list[dict] = []
        self.document_embeddings: np.ndarray | None = None
        logger.info("Model loaded successfully.")

    def ingest_documents(self, text_chunks: list[dict]) -> int:
        """
        Embed the text chunks from the student's semester readings.

        Each chunk should be a dictionary with at least::

            {
                "text": "The actual text content...",
                "source": "filename.pdf",
                "page": 1
            }

        Additional keys (e.g., ``chunk_index``, ``start_char``) are
        preserved in the metadata and returned with search results.

        Args:
            text_chunks: List of dicts containing chunk text and metadata.

        Returns:
            Number of chunks successfully ingested.

        Raises:
            ValueError: If no text chunks are provided.
        """
        if not text_chunks:
            raise ValueError("No text chunks provided for ingestion.")

        self.document_chunks = text_chunks
        raw_texts = [chunk["text"] for chunk in text_chunks]

        logger.info("Embedding %d document chunks …", len(raw_texts))
        self.document_embeddings = self.model.encode(
            raw_texts,
            show_progress_bar=len(raw_texts) > 100,
            normalize_embeddings=True,
            batch_size=64,
        )
        self.document_embeddings = np.asarray(self.document_embeddings, dtype=np.float32)
        logger.info("Ingestion complete: %d chunks embedded.", len(raw_texts))
        return len(raw_texts)

    def find_source(
        self,
        orphaned_quote: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Find the most likely original source for a given quote or paraphrase.

        Embeds the student's text and computes cosine similarity against
        all ingested document chunks. Returns the top-k matches ranked
        by confidence score.

        Evaluation Note (AI/ML Hackathon Requirement):
            - Exact quotes typically score > 0.85 cosine similarity.
            - Paraphrased text scores 0.55–0.80 depending on how heavily
              the wording was changed.
            - Unrelated text scores < 0.30.
            See ``EVALUATION.md`` for detailed benchmark results.

        Args:
            orphaned_quote: The text or paraphrase the student wrote.
            top_k: Number of top matching results to return.

        Returns:
            List of dicts, each containing the original chunk metadata
            plus a ``confidence_score`` key with the cosine similarity.

        Raises:
            RuntimeError: If no documents have been ingested yet.
        """
        if self.document_embeddings is None:
            raise RuntimeError("No documents ingested. Call ingest_documents first.")

        quote_embedding = self.model.encode(
            [orphaned_quote],
            normalize_embeddings=True,
        )
        quote_embedding = np.asarray(quote_embedding, dtype=np.float32)

        similarities = cosine_similarity(quote_embedding, self.document_embeddings)[0]

        # Get the indices of the highest similarity scores
        effective_k = min(top_k, len(self.document_chunks))
        top_indices = np.argsort(similarities)[-effective_k:][::-1]

        results = []
        for idx in top_indices:
            match = self.document_chunks[idx].copy()
            match["confidence_score"] = round(float(similarities[idx]), 4)
            results.append(match)

        return results

    @property
    def num_chunks(self) -> int:
        """Number of chunks currently ingested."""
        return len(self.document_chunks)

    @property
    def is_ready(self) -> bool:
        """Whether the retriever has ingested documents and is ready to search."""
        return self.document_embeddings is not None and len(self.document_chunks) > 0

    def clear(self) -> None:
        """Remove all ingested documents and embeddings."""
        self.document_chunks = []
        self.document_embeddings = None
        logger.info("SourceRetriever cleared.")
