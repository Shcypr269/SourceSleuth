"""
SourceSleuth Configuration — Centralised Settings from .env.

This module is the single source of truth for all runtime configuration.
It loads the ``.env`` file at the project root via ``python-dotenv`` and
exposes every setting as a typed constant.  All other modules import
from here instead of reading ``os.environ`` directly.

Usage::

    from src.config import (
        PDF_DIR, DATA_DIR, EMBEDDING_MODEL, CHUNK_SIZE,
        CHUNK_OVERLAP, TOP_K, MIN_SCORE, LOG_LEVEL,
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# NLTK Data Initialization (Runtime Provisioning)
# ---------------------------------------------------------------------------

# Global flag to track NLTK availability (for graceful degradation)
NLTK_AVAILABLE = True

# Map NLTK packages to their correct subdirectories
_NLTK_PACKAGE_PATHS = {
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "omw-1.4": "corpora/omw-1.4",
}


def _ensure_nltk_data() -> None:
    """
    Ensure NLTK corpora are available, downloading if missing.

    This is critical for local installations where NLTK data hasn't been
    pre-downloaded. Docker/CI environments download data during build,
    but local pip installations need runtime provisioning.

    Handles offline scenarios gracefully by logging warnings instead of crashing.
    Sets global NLTK_AVAILABLE flag to False if downloads fail.
    """
    global NLTK_AVAILABLE

    try:
        import nltk

        for package, path in _NLTK_PACKAGE_PATHS.items():
            try:
                # Try to find the resource at its correct path
                nltk.data.find(path)
            except LookupError:
                # Not found, attempt to download
                try:
                    nltk.download(package, quiet=True)
                except Exception as download_exc:
                    # Handle offline/network errors gracefully
                    import logging

                    logger = logging.getLogger("sourcesleuth.config")
                    logger.warning(
                        f"Failed to download NLTK package '{package}': {download_exc}. "
                        f"Query expansion will be limited. "
                        f"To fix: run 'python -m nltk.downloader {package}' when online."
                    )
                    NLTK_AVAILABLE = False

    except ImportError:
        # NLTK not installed - will fail later with clear error
        NLTK_AVAILABLE = False
    except Exception as exc:
        # Unexpected error during initialization
        import logging

        logger = logging.getLogger("sourcesleuth.config")
        logger.warning(f"NLTK initialization failed: {exc}. Query expansion disabled.")
        NLTK_AVAILABLE = False


# Initialize NLTK data on module import
_ensure_nltk_data()

# ---------------------------------------------------------------------------
# Bootstrap: load .env BEFORE reading any os.environ values
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - graceful fallback
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    _env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)

# ---------------------------------------------------------------------------
# PATH settings
# ---------------------------------------------------------------------------

PDF_DIR = Path(os.environ.get("SOURCESLEUTH_PDF_DIR", str(PROJECT_ROOT / "student_pdfs")))
# Make relative paths relative to project root
if not PDF_DIR.is_absolute():
    PDF_DIR = PROJECT_ROOT / PDF_DIR

DATA_DIR = Path(os.environ.get("SOURCESLEUTH_DATA_DIR", str(PROJECT_ROOT / "data")))
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_ROOT / DATA_DIR

# ---------------------------------------------------------------------------
# MODEL settings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = os.environ.get("SOURCESLEUTH_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# CHUNKING settings
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = int(os.environ.get("SOURCESLEUTH_CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.environ.get("SOURCESLEUTH_CHUNK_OVERLAP", "50"))
CHUNK_STRATEGY: str = os.environ.get("SOURCESLEUTH_CHUNK_STRATEGY", "sentence")

# ---------------------------------------------------------------------------
# SEARCH settings
# ---------------------------------------------------------------------------

TOP_K: int = int(os.environ.get("SOURCESLEUTH_TOP_K", "5"))
# FIXED: Lower default from 0.65 to 0.35
# all-MiniLM-L6-v2 produces compressed cosine similarity scores
# Relevant paraphrases often score 0.45-0.55, not 0.7+
MIN_SCORE: float = float(os.environ.get("SOURCESLEUTH_MIN_SCORE", "0.35"))
SEARCH_MODE: str = os.environ.get("SOURCESLEUTH_SEARCH_MODE", "hybrid")

# ---------------------------------------------------------------------------
# ARXIV settings
# ---------------------------------------------------------------------------

ARXIV_MAX_RECORDS: int = int(os.environ.get("SOURCESLEUTH_ARXIV_MAX_RECORDS", "5000"))

# ---------------------------------------------------------------------------
# LOGGING settings
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.environ.get("SOURCESLEUTH_LOG_LEVEL", "INFO").upper()
LOG_FILE: str | None = os.environ.get("SOURCESLEUTH_LOG_FILE") or None

# ---------------------------------------------------------------------------
# WEB UI settings
# ---------------------------------------------------------------------------

WEB_PORT: int = int(os.environ.get("SOURCESLEUTH_WEB_PORT", "8501"))
WEB_ADDRESS: str = os.environ.get("SOURCESLEUTH_WEB_ADDRESS", "localhost")

# ---------------------------------------------------------------------------
# Logging bootstrap — configure root logger according to .env
# ---------------------------------------------------------------------------

_log_level_num = getattr(logging, LOG_LEVEL, logging.INFO)

_handlers: list[logging.Handler] = [logging.StreamHandler()]
if LOG_FILE:
    _handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))

logging.basicConfig(
    level=_log_level_num,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    handlers=_handlers,
    force=True,
)

logger = logging.getLogger("sourcesleuth.config")
logger.debug("Configuration loaded from: %s", PROJECT_ROOT / ".env")
logger.debug("PDF_DIR         = %s", PDF_DIR)
logger.debug("DATA_DIR        = %s", DATA_DIR)
logger.debug("EMBEDDING_MODEL = %s", EMBEDDING_MODEL)
logger.debug("CHUNK_SIZE      = %d", CHUNK_SIZE)
logger.debug("CHUNK_OVERLAP   = %d", CHUNK_OVERLAP)
logger.debug("CHUNK_STRATEGY  = %s", CHUNK_STRATEGY)
logger.debug("TOP_K           = %d", TOP_K)
logger.debug("MIN_SCORE       = %.2f", MIN_SCORE)
logger.debug("SEARCH_MODE     = %s", SEARCH_MODE)
logger.debug("LOG_LEVEL       = %s", LOG_LEVEL)
