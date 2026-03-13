"""Tests for the MCP Server tool functions."""

import pytest

from src.pdf_processor import TextChunk
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Fixtures - Provide isolated test environments
# ---------------------------------------------------------------------------


@pytest.fixture
def vector_store(tmp_path):
    """Create a fresh VectorStore instance for each test."""
    store = VectorStore(data_dir=tmp_path / "vector_data")
    yield store
    # Cleanup
    store.clear()


@pytest.fixture
def mock_mcp_store(monkeypatch, vector_store):
    """Patch the MCP server's global store with a fresh test instance."""
    import src.mcp_server

    monkeypatch.setattr(src.mcp_server, "store", vector_store)
    return vector_store


# ---------------------------------------------------------------------------
# Tests: Tool Functions
# ---------------------------------------------------------------------------


class TestFindOrphanedQuote:
    """Test the find_orphaned_quote tool logic."""

    def test_empty_store_returns_warning(self, mock_mcp_store):
        """When no PDFs are ingested, should return a helpful message."""
        from src.mcp_server import find_orphaned_quote

        result = find_orphaned_quote("some random quote")
        assert "No PDFs have been ingested" in result

    def test_search_with_data(self, mock_mcp_store):
        """When data is present, should return formatted results."""
        from src.mcp_server import find_orphaned_quote

        chunks = [
            TextChunk(
                text="Machine learning is a subset of artificial intelligence.",
                filename="intro_to_ml.pdf",
                page=1,
                chunk_index=0,
                start_char=0,
                end_char=55,
                title="",
                authors="",
                creation_date="",
                publisher="",
                journal="",
                doi="",
            ),
        ]
        mock_mcp_store.add_chunks(chunks)

        result = find_orphaned_quote("artificial intelligence and machine learning")
        assert "intro_to_ml.pdf" in result
        assert "Confidence" in result


class TestGetStoreStats:
    """Test the get_store_stats tool."""

    def test_empty_stats(self, mock_mcp_store):
        from src.mcp_server import get_store_stats

        result = get_store_stats()
        assert "Empty" in result

    def test_stats_with_data(self, mock_mcp_store):
        from src.mcp_server import get_store_stats

        chunks = [
            TextChunk(
                text="Test chunk content.",
                filename="test.pdf",
                page=1,
                chunk_index=0,
                start_char=0,
                end_char=19,
                title="",
                authors="",
                creation_date="",
                publisher="",
                journal="",
                doi="",
            ),
        ]
        mock_mcp_store.add_chunks(chunks)
        result = get_store_stats()
        assert "test.pdf" in result
        assert "all-MiniLM-L6-v2" in result


class TestCiteRecoveredSource:
    """Test the cite_recovered_source prompt."""

    def test_prompt_output(self):
        from src.mcp_server import cite_recovered_source

        result = cite_recovered_source(
            quote="Deep learning has transformed NLP.",
            source_filename="smith2023_nlp.pdf",
            page_number=5,
            citation_style="APA",
        )
        assert "smith2023_nlp.pdf" in result
        assert "APA" in result
        assert "page 5" in result
        assert "Full Citation" in result


class TestIngestPdfs:
    """Test the ingest_pdfs tool."""

    def test_ingest_basic(self, tmp_path, mock_mcp_store):
        import fitz

        from src.mcp_server import ingest_pdfs

        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test content for ingestion.", fontsize=11)
        pdf_path = tmp_path / "test_ingest.pdf"
        doc.save(str(pdf_path))
        doc.close()

        result = ingest_pdfs(directory=str(tmp_path), enable_ocr=False, ocr_language="eng")

        assert "Ingestion complete" in result
        assert "test_ingest.pdf" in result
        assert "Chunks created" in result

    def test_ingest_directory_not_found(self, mock_mcp_store):
        from src.mcp_server import ingest_pdfs

        result = ingest_pdfs(directory="/nonexistent/path", enable_ocr=False, ocr_language="eng")
        assert "Directory not found" in result

    def test_ingest_no_pdfs(self, tmp_path, mock_mcp_store):
        from src.mcp_server import ingest_pdfs

        result = ingest_pdfs(directory=str(tmp_path), enable_ocr=False, ocr_language="eng")
        assert "No PDF files found" in result
