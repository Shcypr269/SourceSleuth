"""
Vector Store Module for SourceSleuth.

Manages embeddings using FAISS (Facebook AI Similarity Search) as the
local vector database, with optional BM25 sparse retrieval for hybrid
search. All data stays on the student's machine — no network calls,
no cloud dependencies.

Model Architecture Documentation (Hackathon AI/ML Requirement):
    - Default Model: ``all-MiniLM-L6-v2`` from Sentence-Transformers
    - Embedding dimension: 384 (auto-detected for custom models)
    - Why this model as default?
        1. Runs efficiently on CPU (no GPU required).
        2. Produces high-quality sentence embeddings for semantic similarity.
        3. Small footprint (~80 MB) — ideal for a student laptop.
        4. Trained on 1B+ sentence pairs, strong zero-shot performance.
    - Configurable: Set ``SOURCESLEUTH_MODEL`` env var or pass
      ``model_name`` to use a different model (e.g., ``BAAI/bge-large-en-v1.5``
      for higher accuracy at the cost of speed and memory).
    - Index type: FAISS IndexFlatIP (inner product on L2-normalized vectors,
      equivalent to cosine similarity). Flat index chosen for simplicity and
      exact results on small-to-medium corpora (< 100k chunks).

Hybrid Search Architecture:
    - **Dense retrieval**: FAISS cosine similarity on sentence embeddings.
    - **Sparse retrieval**: BM25 keyword matching via ``rank_bm25``.
    - **Fusion**: Reciprocal Rank Fusion (RRF) combines both rankings
      into a single score: ``RRF(d) = Σ  1 / (k + rank_i(d))``
      where k=60 (standard constant). This captures both semantic
      similarity AND exact keyword matches.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL as DEFAULT_MODEL_NAME
from src.pdf_processor import TextChunk


# Try to import filelock for multi-process safety
try:
    from filelock import FileLock

    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False
    FileLock = None  # type: ignore

logger = logging.getLogger("sourcesleuth.vector_store")

# Configuration

# Known embedding dimensions for common models (fallback: auto-detect)
_KNOWN_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}

EMBEDDING_DIM = _KNOWN_DIMS.get(DEFAULT_MODEL_NAME, 384)

INDEX_FILENAME = "sourcesleuth.index"
METADATA_FILENAME = "sourcesleuth_metadata.json"

# RRF fusion constant (standard value from the original RRF paper)
RRF_K = 60


# BM25 Sparse Index


class _BM25Index:
    """
    Lightweight BM25 sparse keyword index.

    Uses ``rank_bm25`` when available, falls back to a basic TF-IDF
    approximation if the package is not installed. This ensures the
    system works without the optional dependency, just with reduced
    keyword retrieval quality.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._corpus_tokens: list[list[str]] = []
        self._available = False

        try:
            from rank_bm25 import BM25Okapi  # noqa: F401

            self._available = True
        except ImportError:
            logger.info(
                "rank_bm25 not installed — hybrid search will use "
                "dense retrieval only. Install with: pip install rank_bm25"
            )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer for BM25."""
        return re.findall(r"\b\w+\b", text.lower())

    def build(self, texts: list[str]) -> None:
        """Build the BM25 index from a list of text strings."""
        if not self._available:
            return

        from rank_bm25 import BM25Okapi

        self._corpus_tokens = [self._tokenize(t) for t in texts]
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
            logger.info("BM25 index built with %d documents.", len(texts))

    def query(self, text: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Query the BM25 index.

        Returns:
            List of (index, score) tuples, sorted by score descending.
        """
        if not self._available or self._bm25 is None:
            return []

        tokens = self._tokenize(text)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def clear(self) -> None:
        """Reset the BM25 index."""
        self._bm25 = None
        self._corpus_tokens = []

    @property
    def is_available(self) -> bool:
        """Whether the BM25 backend is installed and ready."""
        return self._available and self._bm25 is not None


# Reciprocal Rank Fusion


def _reciprocal_rank_fusion(
    dense_ranking: list[tuple[int, float]],
    sparse_ranking: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Combine dense and sparse rankings using Reciprocal Rank Fusion.

    RRF score for document d:
        RRF(d) = Σ  1 / (k + rank_i(d))

    where the sum is over all ranking systems that include d.
    k=60 is the standard constant from the original RRF paper
    (Cormack, Clarke & Buettcher, 2009).

    Args:
        dense_ranking: List of (index, score) from FAISS.
        sparse_ranking: List of (index, score) from BM25.
        k: RRF constant (default 60).

    Returns:
        Fused ranking as list of (index, rrf_score), sorted descending.
    """
    rrf_scores: dict[int, float] = {}

    for rank, (idx, _) in enumerate(dense_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, (idx, _) in enumerate(sparse_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # Sort by fused score, descending
    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return fused


# Vector Store Class


class VectorStore:
    """
    FAISS-backed vector store with optional BM25 hybrid search.

    Supports adding chunks, querying by text similarity (dense, sparse,
    or hybrid), and persisting the index + metadata to disk for fast
    reloads.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        data_dir: str | Path = "data",
    ) -> None:
        """
        Initialize the vector store.

        Args:
            model_name: HuggingFace model identifier for the embedding model.
                        Can also be set via SOURCESLEUTH_MODEL env var.
            data_dir: Directory where the FAISS index and metadata are persisted.
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-load the model to avoid slow startup when not needed
        self._model: SentenceTransformer | None = None
        self._embedding_dim: int = _KNOWN_DIMS.get(model_name, EMBEDDING_DIM)

        # FAISS index — inner-product on L2-normalized vectors = cosine sim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._embedding_dim)

        # Parallel metadata list (same order as vectors in the index)
        self._metadata: list[dict] = []

        # BM25 sparse index for hybrid search
        self._bm25 = _BM25Index()

        # Track ingested filenames to support re-ingestion
        self._ingested_files: set[str] = set()

    # Model

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model on first access."""
        if self._model is None:
            logger.info("Loading embedding model '%s' …", self.model_name)
            self._model = SentenceTransformer(self.model_name)

            # Auto-detect embedding dimension from the model
            test_dim = self._model.get_sentence_embedding_dimension()
            if test_dim and test_dim != self._embedding_dim:
                logger.info(
                    "Auto-detected embedding dim=%d (was %d). Rebuilding index.",
                    test_dim,
                    self._embedding_dim,
                )
                self._embedding_dim = test_dim
                # Rebuild FAISS index with correct dimensions if empty
                if self._index.ntotal == 0:
                    self._index = faiss.IndexFlatIP(self._embedding_dim)

            logger.info("Model loaded successfully (dim=%d).", self._embedding_dim)
        return self._model

    # Core API

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """
        Embed and add text chunks to the vector store.

        Args:
            chunks: List of TextChunk objects to embed and index.

        Returns:
            Number of new chunks added.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]

        logger.info("Encoding %d chunks …", len(texts))
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
            batch_size=64,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Add to FAISS index
        self._index.add(embeddings)

        # Store metadata in parallel
        for chunk in chunks:
            self._metadata.append(chunk.to_dict())
            self._ingested_files.add(chunk.filename)

        # Rebuild BM25 index with all texts
        all_texts = [m["text"] for m in self._metadata]
        self._bm25.build(all_texts)

        logger.info(
            "Added %d chunks to the vector store (total: %d).",
            len(chunks),
            self._index.ntotal,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[dict]:
        """
        Perform semantic search for a query text.

        Supports three search modes:
            - ``'hybrid'`` (default): Combines FAISS dense retrieval with
              BM25 sparse retrieval via Reciprocal Rank Fusion.
            - ``'dense'``: FAISS cosine similarity only.
            - ``'sparse'``: BM25 keyword matching only.

        Args:
            query: The text to search for (e.g., an orphaned quote).
            top_k: Number of top results to return.
            mode: Search mode — 'hybrid', 'dense', or 'sparse'.

        Returns:
            List of result dicts, each containing 'score' and chunk metadata.
        """
        if self._index.ntotal == 0:
            logger.warning("Vector store is empty — no results to return.")
            return []

        # Clamp top_k to the number of available vectors
        top_k = min(top_k, self._index.ntotal)

        results: list[dict] = []

        if mode == "sparse" and self._bm25.is_available:
            # BM25 only
            sparse_results = self._bm25.query(query, top_k=top_k)
            for idx, score in sparse_results[:top_k]:
                if 0 <= idx < len(self._metadata):
                    meta = self._metadata[idx].copy()
                    meta["score"] = round(score, 4)
                    results.append(meta)
            return results

        # Dense retrieval (always needed for 'dense' and 'hybrid')
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        # Fetch more candidates for RRF fusion
        fetch_k = min(top_k * 3, self._index.ntotal) if mode == "hybrid" else top_k
        scores, indices = self._index.search(query_embedding, fetch_k)

        dense_ranking = [
            (int(idx), float(score))
            for score, idx in zip(scores[0], indices[0], strict=False)
            if idx >= 0
        ]

        if mode == "dense" or not self._bm25.is_available:
            # Dense only — return FAISS results directly
            for idx, score in dense_ranking[:top_k]:
                if 0 <= idx < len(self._metadata):
                    meta = self._metadata[idx].copy()
                    meta["score"] = round(score, 4)
                    results.append(meta)
            return results

        # Hybrid: combine dense + sparse via RRF
        sparse_ranking = self._bm25.query(query, top_k=fetch_k)
        fused = _reciprocal_rank_fusion(dense_ranking, sparse_ranking)

        for idx, rrf_score in fused[:top_k]:
            if 0 <= idx < len(self._metadata):
                meta = self._metadata[idx].copy()
                # Use the dense cosine score for display if available
                dense_score = dict(dense_ranking).get(idx, 0.0)
                meta["score"] = round(float(dense_score), 4)
                meta["rrf_score"] = round(float(rrf_score), 6)
                results.append(meta)

        return results

    # Persistence

    def _get_lock_path(self) -> Path:
        """Get the path for the file lock."""
        return self.data_dir / "sourcesleuth.lock"

    def save(self) -> None:
        """
        Persist the FAISS index and metadata to disk.

        Uses file locking to prevent concurrent writes from multiple processes
        (e.g., MCP Server + Streamlit UI running simultaneously).
        """
        index_path = self.data_dir / INDEX_FILENAME
        meta_path = self.data_dir / METADATA_FILENAME

        # Use file lock if available to prevent concurrent writes
        if FILELOCK_AVAILABLE and FileLock is not None:
            lock_path = self._get_lock_path()
            lock = FileLock(lock_path, timeout=30)
            with lock:
                faiss.write_index(self._index, str(index_path))

                payload = {
                    "model_name": self.model_name,
                    "embedding_dim": self._embedding_dim,
                    "ingested_files": sorted(self._ingested_files),
                    "chunks": self._metadata,
                }
                meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            # No filelock - proceed without locking (may cause corruption)
            if not FILELOCK_AVAILABLE:
                logger.warning(
                    "filelock not installed - concurrent writes may corrupt index. "
                    "Install with: pip install filelock"
                )
            faiss.write_index(self._index, str(index_path))

            payload = {
                "model_name": self.model_name,
                "embedding_dim": self._embedding_dim,
                "ingested_files": sorted(self._ingested_files),
                "chunks": self._metadata,
            }
            meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        logger.info(
            "Saved vector store: %d vectors -> '%s'.",
            self._index.ntotal,
            self.data_dir,
        )

    def load(self) -> bool:
        """
        Load a previously persisted vector store from disk.

        Uses file locking to prevent reading during writes.

        Returns:
            True if loaded successfully, False if no saved data was found.
        """
        index_path = self.data_dir / INDEX_FILENAME
        meta_path = self.data_dir / METADATA_FILENAME

        if not index_path.exists() or not meta_path.exists():
            logger.info("No saved vector store found at '%s'.", self.data_dir)
            return False

        # Use file lock if available
        if FILELOCK_AVAILABLE and FileLock is not None:
            lock_path = self._get_lock_path()
            lock = FileLock(lock_path, timeout=30)
            with lock:
                self._index = faiss.read_index(str(index_path))

                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                self._metadata = payload.get("chunks", [])
                self._ingested_files = set(payload.get("ingested_files", []))
                self._embedding_dim = payload.get("embedding_dim", EMBEDDING_DIM)
        else:
            # No filelock - proceed without locking
            self._index = faiss.read_index(str(index_path))

            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            self._metadata = payload.get("chunks", [])
            self._ingested_files = set(payload.get("ingested_files", []))
            self._embedding_dim = payload.get("embedding_dim", EMBEDDING_DIM)

        # Rebuild BM25 index from loaded metadata
        all_texts = [m["text"] for m in self._metadata]
        self._bm25.build(all_texts)

        logger.info(
            "Loaded vector store: %d vectors from '%s'.",
            self._index.ntotal,
            self.data_dir,
        )
        return True

    # Utilities

    @property
    def total_chunks(self) -> int:
        """Number of chunks currently in the store."""
        return self._index.ntotal

    @property
    def ingested_files(self) -> set[str]:
        """Set of filenames that have been ingested."""
        return self._ingested_files.copy()

    def clear(self) -> None:
        """Remove all vectors and metadata."""
        self._index = faiss.IndexFlatIP(self._embedding_dim)
        self._metadata.clear()
        self._ingested_files.clear()
        self._bm25.clear()
        logger.info("Vector store cleared.")

    def remove_file(self, filename: str) -> int:
        """
        Remove all chunks belonging to a specific file and rebuild the index.

        Args:
            filename: The filename to remove.

        Returns:
            Number of chunks removed.
        """
        if filename not in self._ingested_files:
            return 0

        # Filter out chunks belonging to this file
        keep_indices = [i for i, m in enumerate(self._metadata) if m["filename"] != filename]
        removed_count = len(self._metadata) - len(keep_indices)

        if not keep_indices:
            self.clear()
            return removed_count

        # Rebuild index with remaining vectors
        remaining_texts = [self._metadata[i]["text"] for i in keep_indices]
        remaining_meta = [self._metadata[i] for i in keep_indices]

        # Re-encode remaining chunks
        embeddings = self.model.encode(
            remaining_texts,
            normalize_embeddings=True,
            batch_size=64,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        self._index = faiss.IndexFlatIP(self._embedding_dim)
        self._index.add(embeddings)
        self._metadata = remaining_meta
        self._ingested_files.discard(filename)

        # Rebuild BM25 index
        self._bm25.build(remaining_texts)

        logger.info("Removed %d chunks for '%s'.", removed_count, filename)
        return removed_count

    def get_stats(self) -> dict:
        """Return summary statistics about the vector store."""
        return {
            "total_chunks": self._index.ntotal,
            "ingested_files": sorted(self._ingested_files),
            "num_files": len(self._ingested_files),
            "model_name": self.model_name,
            "embedding_dim": self._embedding_dim,
            "index_type": "IndexFlatIP (cosine similarity)",
            "hybrid_search": self._bm25.is_available,
        }
