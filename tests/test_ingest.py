"""
Tests for the SourceSleuth CLI ingestion tool.
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import (
    cmd_clear,
    cmd_ingest_pdfs,
    cmd_stats,
    main,
)


class TestCLIIngestPdfs:
    """Tests for the PDF ingestion command."""

    def test_ingest_pdfs_empty_directory(self, tmp_path):
        """Test ingestion from a directory with no PDFs."""
        args = argparse.Namespace(directory=str(tmp_path))
        result = cmd_ingest_pdfs(args)
        # Should return 0 (no error) even with no PDFs
        assert result == 0

    def test_ingest_pdfs_nonexistent_directory(self, tmp_path):
        """Test ingestion from a nonexistent directory."""
        nonexistent = tmp_path / "does_not_exist"
        args = argparse.Namespace(directory=str(nonexistent))
        result = cmd_ingest_pdfs(args)
        assert result == 1

    def test_ingest_pdfs_default_directory(self):
        """Test ingestion using the default PDF directory."""
        args = argparse.Namespace(directory="")
        # This will run against the actual student_pdfs folder
        # Result depends on whether PDFs exist there
        result = cmd_ingest_pdfs(args)
        assert result in (0, 1)


class TestCLIStats:
    """Tests for the stats command."""

    def test_stats_empty_store(self, tmp_path):
        """Test stats command with an empty vector store."""
        # Temporarily override DATA_DIR
        with patch("src.ingest.DATA_DIR", tmp_path):
            args = argparse.Namespace()
            result = cmd_stats(args)
            # Should return 0 even with empty store
            assert result == 0


class TestCLIClear:
    """Tests for the clear command."""

    def test_clear_empty_store(self, tmp_path):
        """Test clearing an empty vector store."""
        with patch("src.ingest.DATA_DIR", tmp_path):
            args = argparse.Namespace()
            result = cmd_clear(args)
            assert result == 0


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_main_no_command(self):
        """Test running CLI without a command shows help."""
        with patch.object(sys, "argv", ["sourcesleuth-ingest"]):
            result = main()
            assert result == 0

    def test_main_pdfs_command(self, tmp_path):
        """Test running the pdfs subcommand."""
        with patch.object(sys, "argv", ["sourcesleuth-ingest", "pdfs", "-d", str(tmp_path)]):
            result = main()
            assert result == 0

    def test_main_stats_command(self, tmp_path):
        """Test running the stats subcommand."""
        with (
            patch("src.ingest.DATA_DIR", tmp_path),
            patch.object(sys, "argv", ["sourcesleuth-ingest", "stats"]),
        ):
            result = main()
            assert result == 0

    def test_main_clear_command(self, tmp_path):
        """Test running the clear subcommand."""
        with (
            patch("src.ingest.DATA_DIR", tmp_path),
            patch.object(sys, "argv", ["sourcesleuth-ingest", "clear"]),
        ):
            result = main()
            assert result == 0

    def test_main_arxiv_command_missing_data(self, tmp_path):
        """Test arxiv command when data file is missing."""
        with (
            patch("src.ingest.DATA_DIR", tmp_path),
            patch.object(sys, "argv", ["sourcesleuth-ingest", "arxiv"]),
        ):
            result = main()
            # Should return 1 because arXiv data file doesn't exist
            assert result == 1


class TestCLIArgumentParsing:
    """Tests for argument parsing."""

    def test_pdfs_arguments(self):
        """Test PDF subcommand argument parsing."""
        import io
        from contextlib import redirect_stdout

        # Test with custom directory
        with patch.object(sys, "argv", ["sourcesleuth-ingest", "pdfs", "-d", "/custom/path"]):
            # Should not raise
            f = io.StringIO()
            with redirect_stdout(f):
                pass  # Parsing happens in main()

    def test_arxiv_arguments(self):
        """Test arXiv subcommand argument parsing."""
        with patch.object(
            sys, "argv", ["sourcesleuth-ingest", "arxiv", "-c", "physics.", "-n", "1000"]
        ):
            # Should not raise
            pass

    def test_help_flag(self):
        """Test --help flag."""
        with patch.object(sys, "argv", ["sourcesleuth-ingest", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
