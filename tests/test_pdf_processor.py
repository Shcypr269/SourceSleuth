from pathlib import Path

import fitz
import pytest

from src.pdf_processor import (
    TextChunk,
    chunk_text,
    extract_text_from_pdf,
    process_pdf_directory,
)


def _create_test_pdf(path: Path, pages: list[str]) -> Path:
    """Helper: create a minimal PDF with the given text on each page."""
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        text_point = fitz.Point(72, 72)  # 1 inch from top-left
        page.insert_text(text_point, page_text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a sample PDF with three pages of text."""
    pages = [
        "This is the first page of an academic paper about machine learning. "
        "Deep learning has revolutionized natural language processing.",
        "The second page discusses transformer architectures and attention. "
        "Self-attention mechanisms compute relationships between all tokens.",
        "The third page covers evaluation metrics and future directions. "
        "BLEU scores measure the quality of machine-translated text.",
    ]
    return _create_test_pdf(tmp_path / "test_paper.pdf", pages)


@pytest.fixture
def pdf_directory(tmp_path):
    """Create a directory with multiple test PDFs."""
    _create_test_pdf(tmp_path / "paper_a.pdf", ["Content of paper A."])
    _create_test_pdf(tmp_path / "paper_b.pdf", ["Content of paper B."])
    return tmp_path


class TestTextChunk:
    def test_to_dict_round_trip(self):
        chunk = TextChunk(
            text="Hello world",
            filename="test.pdf",
            page=1,
            chunk_index=0,
            start_char=0,
            end_char=11,
            title="",
            authors="",
            creation_date="",
            publisher="",
            journal="",
            doi="",
        )
        reconstructed = TextChunk.from_dict(chunk.to_dict())
        assert reconstructed.text == chunk.text
        assert reconstructed.filename == chunk.filename
        assert reconstructed.page == chunk.page


class TestExtractText:
    def test_extract_basic(self, sample_pdf):
        doc = extract_text_from_pdf(sample_pdf)
        assert doc.filename == "test_paper.pdf"
        assert len(doc.full_text) > 0
        assert len(doc.page_spans) == 3

    def test_extract_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf(tmp_path / "nonexistent.pdf")

    def test_page_spans_are_contiguous(self, sample_pdf):
        doc = extract_text_from_pdf(sample_pdf)
        for i in range(1, len(doc.page_spans)):
            assert doc.page_spans[i].start_char == doc.page_spans[i - 1].end_char


class TestChunking:
    def test_chunk_produces_output(self, sample_pdf):
        doc = extract_text_from_pdf(sample_pdf)
        chunks = chunk_text(doc, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 0

    def test_chunk_metadata(self, sample_pdf):
        doc = extract_text_from_pdf(sample_pdf)
        chunks = chunk_text(doc, chunk_size=50, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.filename == "test_paper.pdf"
            assert chunk.page >= 1
            assert chunk.text.strip()

    def test_small_chunk_size_produces_more_chunks(self, sample_pdf):
        doc = extract_text_from_pdf(sample_pdf)
        big_chunks = chunk_text(doc, chunk_size=500, chunk_overlap=50)
        small_chunks = chunk_text(doc, chunk_size=50, chunk_overlap=10)
        assert len(small_chunks) >= len(big_chunks)

    def test_empty_text_returns_no_chunks(self, tmp_path):
        doc = fitz.open()
        doc.new_page()
        pdf_path = tmp_path / "blank.pdf"
        doc.save(str(pdf_path))
        doc.close()

        extracted = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(extracted)
        assert chunks == []


class TestBatchProcessing:
    def test_process_directory(self, pdf_directory):
        chunks = process_pdf_directory(pdf_directory, use_ocr=False, ocr_language="eng")
        filenames = {c.filename for c in chunks}
        assert "paper_a.pdf" in filenames
        assert "paper_b.pdf" in filenames

    def test_empty_directory(self, tmp_path):
        chunks = process_pdf_directory(tmp_path, use_ocr=False, ocr_language="eng")
        assert chunks == []

    def test_invalid_directory(self):
        with pytest.raises(NotADirectoryError):
            process_pdf_directory("/nonexistent/path", use_ocr=False, ocr_language="eng")
