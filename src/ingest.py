"""
SourceSleuth CLI Ingestion Tool.

A standalone command-line interface for ingesting PDFs and arXiv data
into the vector store, without requiring an MCP host.

Usage:
    # Ingest local PDFs
    python -m src.ingest pdfs --directory /path/to/pdfs

    # Ingest arXiv papers
    python -m src.ingest arxiv --category cs. --max-records 5000

    # View store stats
    python -m src.ingest stats

    # Clear the vector store
    python -m src.ingest clear
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import DATA_DIR, EMBEDDING_MODEL, PDF_DIR
from src.dataset_preprocessor import preprocess_dataset, stream_arxiv_records
from src.pdf_processor import process_pdf_directory
from src.vector_store import VectorStore


# Logging (level configured centrally by src.config)
logger = logging.getLogger("sourcesleuth.ingest")


def cmd_ingest_pdfs(args: argparse.Namespace) -> int:
    """Ingest PDFs from a directory into the vector store."""
    target_dir = Path(args.directory) if args.directory else PDF_DIR

    if not target_dir.is_dir():
        logger.error("Directory not found: %s", target_dir)
        return 1

    pdf_files = list(target_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", target_dir)
        return 0

    logger.info("Found %d PDF(s) to ingest.", len(pdf_files))

    # Initialize vector store
    store = VectorStore(model_name=EMBEDDING_MODEL, data_dir=DATA_DIR)

    # Optionally load existing store
    if store.load():
        logger.info("Loaded existing vector store (%d chunks).", store.total_chunks)

    # Process PDFs
    chunks = process_pdf_directory(target_dir)
    if not chunks:
        logger.warning("No text could be extracted from PDFs.")
        return 0

    # Add to store and persist
    added = store.add_chunks(chunks)
    store.save()

    files_set = {c.filename for c in chunks}
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info("PDFs processed:     %d", len(files_set))
    logger.info("Chunks created:     %d", added)
    logger.info("Total chunks:       %d", store.total_chunks)
    logger.info("Files:              %s", ", ".join(sorted(files_set)))
    logger.info("=" * 60)

    return 0


def cmd_ingest_arxiv(args: argparse.Namespace) -> int:
    """Ingest arXiv paper abstracts into the vector store."""
    raw_path = DATA_DIR / "arxiv-metadata-oai-snapshot.json"
    if not raw_path.exists():
        logger.error(
            "arXiv dataset not found at: %s\n"
            "Download from: https://www.kaggle.com/Cornell-University/arxiv",
            raw_path,
        )
        return 1

    preprocessed_path = DATA_DIR / "arxiv_preprocessed.jsonl"

    logger.info(
        "Preprocessing arXiv dataset (prefix=%s, max=%d)...", args.category, args.max_records
    )

    prefixes = {p.strip() for p in args.category.split(",") if p.strip()}
    stats = preprocess_dataset(
        input_path=raw_path,
        output_path=preprocessed_path,
        category_prefix_filter=prefixes,
        max_records=args.max_records,
    )

    # Initialize vector store
    store = VectorStore(model_name=EMBEDDING_MODEL, data_dir=DATA_DIR)
    if store.load():
        logger.info("Loaded existing vector store (%d chunks).", store.total_chunks)

    # Convert to TextChunks and ingest
    from src.pdf_processor import TextChunk

    chunks = []
    for record in stream_arxiv_records(preprocessed_path, max_records=args.max_records):
        text = f"{record.title}. {record.abstract}"
        chunk = TextChunk(
            text=text,
            filename=f"arxiv:{record.arxiv_id}",
            page=0,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
        )
        chunks.append(chunk)

    if not chunks:
        logger.warning("No arXiv records matched the filter criteria.")
        return 0

    added = store.add_chunks(chunks)
    store.save()

    top_cats = sorted(stats.categories_seen.items(), key=lambda x: -x[1])[:10]
    cats_str = ", ".join(f"{cat} ({n})" for cat, n in top_cats)

    logger.info("=" * 60)
    logger.info("ARXIV INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Records preprocessed:   %d", stats.records_output)
    logger.info("Chunks added:           %d", added)
    logger.info("Total chunks:           %d", store.total_chunks)
    logger.info("Category filter:        %s", args.category)
    logger.info("Top categories:         %s", cats_str)
    logger.info("Preprocessing time:     %.1fs", stats.elapsed_seconds)
    logger.info("=" * 60)

    return 0


def cmd_stats(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Display vector store statistics."""
    store = VectorStore(model_name=EMBEDDING_MODEL, data_dir=DATA_DIR)

    if not store.load():
        logger.info("Vector store is empty. Use 'ingest pdfs' to add documents.")
        return 0

    stats = store.get_stats()

    print("\n" + "=" * 60)
    print("VECTOR STORE STATISTICS")
    print("=" * 60)
    print(f"Total chunks:         {stats['total_chunks']}")
    print(f"Number of files:      {stats['num_files']}")
    print(f"Embedding model:      {stats['model_name']}")
    print(f"Embedding dimensions: {stats['embedding_dim']}")
    print(f"Index type:           {stats['index_type']}")
    print("\nIngested files:")
    for f in stats["ingested_files"]:
        print(f"  - {f}")
    print("=" * 60)

    return 0


def cmd_clear(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Clear the vector store."""
    store = VectorStore(model_name=EMBEDDING_MODEL, data_dir=DATA_DIR)

    if store.load():
        store.clear()
        # Also remove persisted files
        index_path = DATA_DIR / "sourcesleuth.index"
        meta_path = DATA_DIR / "sourcesleuth_metadata.json"
        if index_path.exists():
            index_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        logger.info("Vector store cleared.")
    else:
        logger.info("Vector store was already empty.")

    return 0


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="sourcesleuth-ingest",
        description="SourceSleuth CLI - Ingest PDFs and arXiv data for semantic search.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # PDF ingestion command
    pdf_parser = subparsers.add_parser(
        "pdfs",
        help="Ingest PDFs from a directory",
    )
    pdf_parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="",
        help=f"Directory containing PDFs (default: {PDF_DIR})",
    )
    pdf_parser.set_defaults(func=cmd_ingest_pdfs)

    # arXiv ingestion command
    arxiv_parser = subparsers.add_parser(
        "arxiv",
        help="Ingest arXiv paper abstracts",
    )
    arxiv_parser.add_argument(
        "-c",
        "--category",
        type=str,
        default="cs.",
        help="arXiv category prefix (e.g., 'cs.', 'physics.')",
    )
    arxiv_parser.add_argument(
        "-n",
        "--max-records",
        type=int,
        default=5000,
        help="Maximum number of records to ingest",
    )
    arxiv_parser.set_defaults(func=cmd_ingest_arxiv)

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Display vector store statistics",
    )
    stats_parser.set_defaults(func=cmd_stats)

    # Clear command
    clear_parser = subparsers.add_parser(
        "clear",
        help="Clear the vector store",
    )
    clear_parser.set_defaults(func=cmd_clear)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
