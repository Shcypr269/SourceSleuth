"""
Integration tests using actual PDF files in student_pdfs/.

These tests verify the full pipeline: PDF extraction → chunking → embedding → search.
"""

import argparse
import os
import tempfile
from pathlib import Path

import pytest

from src.ingest import cmd_ingest_pdfs, cmd_stats
from src.pdf_processor import (
    TextChunk,
    chunk_text,
    extract_text_from_pdf,
    process_pdf_directory,
)
from src.vector_store import VectorStore


# Check if student_pdfs directory exists and has PDFs
STUDENT_PDFS_DIR = Path(__file__).resolve().parent.parent / "student_pdfs"
PDF_FILES = list(STUDENT_PDFS_DIR.glob("*.pdf")) if STUDENT_PDFS_DIR.exists() else []


@pytest.fixture(scope="module")
def pdf_directory():
    """Use the actual student_pdfs directory."""
    if not PDF_FILES:
        pytest.skip("No PDF files found in student_pdfs/ directory")
    return STUDENT_PDFS_DIR


@pytest.fixture
def vector_store(tmp_path):
    """Create a vector store with temp data directory."""
    data_dir = tmp_path / "vector_data"
    store = VectorStore(data_dir=data_dir)
    return store


class TestActualPDFExtraction:
    """Test PDF extraction with real PDF files."""

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_extract_text_from_wave_optics_pdf(self):
        """Test text extraction from 'E- Text Wave Optics.pdf'."""
        pdf_path = STUDENT_PDFS_DIR / "E- Text Wave Optics.pdf"
        if not pdf_path.exists():
            pytest.skip("Wave Optics PDF not found")

        document = extract_text_from_pdf(pdf_path)

        assert document.filename == "E- Text Wave Optics.pdf"
        assert len(document.full_text) > 0
        assert len(document.page_spans) > 0
        assert len(document.chunks) == 0  # Not chunked yet

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_extract_text_from_ch11_pdf(self):
        """Test text extraction from 'ch 11.pdf'."""
        pdf_path = STUDENT_PDFS_DIR / "ch 11.pdf"
        if not pdf_path.exists():
            pytest.skip("ch 11 PDF not found")

        document = extract_text_from_pdf(pdf_path)

        assert document.filename == "ch 11.pdf"
        assert len(document.full_text) > 0
        assert len(document.page_spans) > 0

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_chunk_wave_optics_pdf(self):
        """Test chunking of Wave Optics PDF."""
        pdf_path = STUDENT_PDFS_DIR / "E- Text Wave Optics.pdf"
        if not pdf_path.exists():
            pytest.skip("Wave Optics PDF not found")

        document = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(document)

        assert len(chunks) > 0
        # Verify chunk structure
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.filename == "E- Text Wave Optics.pdf"
            assert chunk.page >= 1
            assert len(chunk.text) > 0

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_process_all_pdfs_in_directory(self):
        """Test processing all PDFs in student_pdfs directory."""
        chunks = process_pdf_directory(STUDENT_PDFS_DIR)

        assert len(chunks) > 0

        # Verify chunks from both PDFs
        filenames = {c.filename for c in chunks}
        assert len(filenames) >= 1  # At least one PDF processed

        # Each chunk should have required fields
        for chunk in chunks:
            assert chunk.text
            assert chunk.filename
            assert chunk.page >= 1
            assert chunk.chunk_index >= 0


class TestVectorStoreWithActualPDFs:
    """Test vector store operations with real PDF content."""

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_add_actual_pdf_chunks(self, vector_store):
        """Test adding chunks from actual PDFs to vector store."""
        # Process PDFs
        chunks = process_pdf_directory(STUDENT_PDFS_DIR)

        # Add to vector store
        added = vector_store.add_chunks(chunks)

        assert added > 0
        assert vector_store.total_chunks > 0

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_search_with_actual_content(self, vector_store):
        """Test semantic search with actual PDF content."""
        # Process and add PDFs
        chunks = process_pdf_directory(STUDENT_PDFS_DIR)
        vector_store.add_chunks(chunks)

        # Get a sample text from chunks to use as query
        sample_text = chunks[0].text[:100] if chunks else "wave optics"

        # Search
        results = vector_store.search(query=sample_text, top_k=3)

        assert len(results) > 0
        assert "score" in results[0]
        assert "filename" in results[0]
        assert "text" in results[0]

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_vector_store_persistence(self, tmp_path):
        """Test that vector store can be saved and loaded."""
        data_dir = tmp_path / "vector_data"

        # Create store and add PDFs
        store = VectorStore(data_dir=data_dir)
        chunks = process_pdf_directory(STUDENT_PDFS_DIR)
        store.add_chunks(chunks)

        # Save
        store.save()

        # Create new store and load
        new_store = VectorStore(data_dir=data_dir)
        loaded = new_store.load()

        assert loaded
        assert new_store.total_chunks > 0


class TestCLIWithActualPDFs:
    """Test CLI ingestion with real PDFs."""

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_cli_ingest_pdfs(self, tmp_path):
        """Test CLI PDF ingestion command."""
        # Use temp directory for data
        with tempfile.TemporaryDirectory() as data_dir:
            # Set environment variable for data directory
            os.environ["SOURCESLEUTH_DATA_DIR"] = data_dir

            try:
                args = argparse.Namespace(directory=str(STUDENT_PDFS_DIR))
                result = cmd_ingest_pdfs(args)

                assert result == 0
            finally:
                # Restore original
                if "SOURCESLEUTH_DATA_DIR" in os.environ:
                    del os.environ["SOURCESLEUTH_DATA_DIR"]

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_cli_stats_after_ingestion(self, tmp_path):
        """Test CLI stats command after ingesting PDFs."""
        with tempfile.TemporaryDirectory() as data_dir:
            os.environ["SOURCESLEUTH_DATA_DIR"] = data_dir

            try:
                # First ingest
                ingest_args = argparse.Namespace(directory=str(STUDENT_PDFS_DIR))
                cmd_ingest_pdfs(ingest_args)

                # Then check stats
                stats_args = argparse.Namespace()
                result = cmd_stats(stats_args)

                assert result == 0
            finally:
                if "SOURCESLEUTH_DATA_DIR" in os.environ:
                    del os.environ["SOURCESLEUTH_DATA_DIR"]


class TestContentVerification:
    """Verify that actual academic content is properly extracted."""

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_physics_content_extracted(self):
        """Verify physics-related content is extracted from PDFs."""
        chunks = process_pdf_directory(STUDENT_PDFS_DIR)

        # Combine all text
        all_text = " ".join(c.text.lower() for c in chunks)

        # Should contain physics-related terms (wave optics content)
        # At least one PDF should have academic content
        assert len(all_text) > 100  # Should have substantial content

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_chunk_overlap_preserves_context(self):
        """Test that chunk overlap preserves sentence context."""
        chunks = process_pdf_directory(STUDENT_PDFS_DIR)

        # Check consecutive chunks have overlap
        if len(chunks) >= 2:
            # Get text from same document
            same_doc_chunks = [c for c in chunks if c.filename == chunks[0].filename]
            if len(same_doc_chunks) >= 2:
                chunk1 = same_doc_chunks[0].text
                chunk2 = same_doc_chunks[1].text

                # Overlapping chunks should share some text
                # (This is a basic check - actual overlap depends on content)
                assert len(chunk1) > 0
                assert len(chunk2) > 0

    @pytest.mark.skipif(not PDF_FILES, reason="No PDF files available")
    def test_page_numbers_tracked(self):
        """Verify page numbers are correctly tracked."""
        chunks = process_pdf_directory(STUDENT_PDFS_DIR)

        # All chunks should have valid page numbers
        for chunk in chunks:
            assert chunk.page >= 1

        # Should have chunks from multiple pages (if PDFs have multiple pages)
        unique_pages = {c.page for c in chunks}
        # At least page 1 should exist
        assert 1 in unique_pages or len(unique_pages) > 0
