"""Tests for the dataset preprocessor module."""

import json
from pathlib import Path

import pytest

from src.dataset_preprocessor import (
    ArxivRecord,
    clean_text,
    clean_title,
    format_authors,
    preprocess_dataset,
    stream_arxiv_records,
)


# Test Data

SAMPLE_RECORDS = [
    {
        "id": "2301.00001",
        "submitter": "Jane Doe",
        "authors": "Jane Doe and John Smith",
        "title": "Attention is all you need for\\n  sequence transduction",
        "comments": "10 pages",
        "journal-ref": "NeurIPS 2017",
        "doi": "10.1234/test",
        "report-no": None,
        "categories": "cs.CL cs.AI",
        "license": None,
        "abstract": (
            "  We propose a novel architecture based on \\textbf{attention mechanisms}.\n"
            "  The $\\alpha$-transformer achieves state-of-the-art results on\n"
            "  machine translation benchmarks without recurrence or convolution.\n"
        ),
        "versions": [{"version": "v1", "created": "Tue, 3 Jan 2023 00:00:00 GMT"}],
        "update_date": "2023-01-15",
        "authors_parsed": [["Doe", "Jane", ""], ["Smith", "John", ""]],
    },
    {
        "id": "2301.00002",
        "submitter": "Alice Bob",
        "authors": "Alice Bob",
        "title": "Deep reinforcement learning for robotics",
        "comments": None,
        "journal-ref": None,
        "doi": None,
        "report-no": None,
        "categories": "cs.RO cs.LG",
        "license": None,
        "abstract": (
            "  We apply deep RL techniques to robotic manipulation tasks.\n"
            "  Our method improves sample efficiency by 50%.\n"
        ),
        "versions": [{"version": "v1", "created": "Wed, 4 Jan 2023 00:00:00 GMT"}],
        "update_date": "2023-02-20",
        "authors_parsed": [["Bob", "Alice", ""]],
    },
    {
        "id": "2301.00003",
        "submitter": "Physics Person",
        "authors": "Physics Person",
        "title": "Quantum entanglement in many-body systems",
        "comments": None,
        "journal-ref": None,
        "doi": None,
        "report-no": None,
        "categories": "quant-ph",
        "license": None,
        "abstract": "  A study of quantum entanglement properties in condensed matter.\n",
        "versions": [{"version": "v1", "created": "Thu, 5 Jan 2023 00:00:00 GMT"}],
        "update_date": "2023-03-10",
        "authors_parsed": [["Person", "Physics", ""]],
    },
]


def _write_sample_jsonl(path: Path, records: list[dict] | None = None):
    """Helper to write sample records to a JSONL file."""
    if records is None:
        records = SAMPLE_RECORDS
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# Tests: Text Cleaning


class TestCleanText:
    """Test the clean_text utility."""

    def test_strips_leading_whitespace(self):
        text = "  Line one\n  Line two\n  Line three"
        result = clean_text(text)
        assert not result.startswith(" ")
        assert "Line one" in result

    def test_removes_textbf(self):
        text = r"This is \textbf{bold text} in a sentence."
        result = clean_text(text)
        assert "bold text" in result
        assert "\\textbf" not in result

    def test_handles_inline_math(self):
        text = "The value $\\alpha$ is important."
        result = clean_text(text)
        assert "alpha" in result
        assert "$" not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_normalises_whitespace(self):
        text = "Hello    world   test"
        result = clean_text(text)
        assert "  " not in result


class TestCleanTitle:
    """Test title cleaning."""

    def test_removes_newlines(self):
        title = "Attention is all you need for\n  sequence transduction"
        result = clean_title(title)
        assert "\n" not in result
        assert "Attention" in result


class TestFormatAuthors:
    """Test author name formatting."""

    def test_parsed_authors(self):
        parsed = [["Doe", "Jane", ""], ["Smith", "John", ""]]
        result = format_authors(parsed, "")
        assert "Jane Doe" in result
        assert "John Smith" in result

    def test_fallback_to_string(self):
        result = format_authors(None, "Jane Doe and John Smith")
        assert "Jane Doe" in result

    def test_author_with_suffix(self):
        parsed = [["Evans", "Neal J.", "II"]]
        result = format_authors(parsed, "")
        assert "Neal J. Evans II" in result


# Tests: Streaming Reader


class TestStreamArxivRecords:
    """Test the streaming JSONL reader."""

    def test_reads_all_records(self, tmp_path):
        path = tmp_path / "test.jsonl"
        _write_sample_jsonl(path)

        records = list(stream_arxiv_records(path))
        assert len(records) == 3

    def test_category_prefix_filter(self, tmp_path):
        path = tmp_path / "test.jsonl"
        _write_sample_jsonl(path)

        records = list(stream_arxiv_records(path, category_prefix_filter={"cs."}))
        assert len(records) == 2
        assert all("cs." in r.categories for r in records)

    def test_exact_category_filter(self, tmp_path):
        path = tmp_path / "test.jsonl"
        _write_sample_jsonl(path)

        records = list(stream_arxiv_records(path, categories_filter={"cs.CL"}))
        assert len(records) == 1
        assert records[0].arxiv_id == "2301.00001"

    def test_max_records(self, tmp_path):
        path = tmp_path / "test.jsonl"
        _write_sample_jsonl(path)

        records = list(stream_arxiv_records(path, max_records=1))
        assert len(records) == 1

    def test_date_filter(self, tmp_path):
        path = tmp_path / "test.jsonl"
        _write_sample_jsonl(path)

        records = list(stream_arxiv_records(path, start_date="2023-03-01"))
        assert len(records) == 1
        assert records[0].arxiv_id == "2301.00003"

    def test_record_fields(self, tmp_path):
        path = tmp_path / "test.jsonl"
        _write_sample_jsonl(path)

        records = list(stream_arxiv_records(path))
        rec = records[0]
        assert rec.arxiv_id == "2301.00001"
        assert "Attention" in rec.title
        assert "Jane Doe" in rec.authors
        assert "attention" in rec.abstract.lower()
        assert rec.doi == "10.1234/test"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(stream_arxiv_records(tmp_path / "nonexistent.jsonl"))


# Tests: Preprocessing Pipeline


class TestPreprocessDataset:
    """Test the full preprocessing pipeline."""

    def test_basic_preprocessing(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_sample_jsonl(input_path)

        stats = preprocess_dataset(input_path, output_path)

        assert stats.records_output == 3
        assert output_path.exists()

        # Verify output is valid JSONL
        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            rec = json.loads(line)
            assert "arxiv_id" in rec
            assert "title" in rec
            assert "abstract" in rec

    def test_filtered_preprocessing(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_sample_jsonl(input_path)

        stats = preprocess_dataset(
            input_path,
            output_path,
            category_prefix_filter={"cs."},
        )

        assert stats.records_output == 2
        assert len(stats.categories_seen) > 0


# Tests: ArxivRecord


class TestArxivRecord:
    """Test the ArxivRecord dataclass."""

    def test_searchable_text(self):
        rec = ArxivRecord(
            arxiv_id="test",
            title="My Great Paper",
            authors="Author",
            abstract="This is the abstract.",
            categories="cs.AI",
        )
        assert rec.searchable_text == "My Great Paper. This is the abstract."

    def test_to_dict(self):
        rec = ArxivRecord(
            arxiv_id="test",
            title="Title",
            authors="Author",
            abstract="Abstract",
            categories="cs.AI",
        )
        d = rec.to_dict()
        assert d["arxiv_id"] == "test"
        assert d["title"] == "Title"
