from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


logger = logging.getLogger("sourcesleuth.preprocessor")


# Data Structures


@dataclass
class ArxivRecord:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    categories: str
    doi: str | None = None
    journal_ref: str | None = None
    update_date: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def searchable_text(self) -> str:
        """Concatenated text used for embedding."""
        return f"{self.title}. {self.abstract}"


# Text Cleaning Utilities

# Common LaTeX commands to strip
_LATEX_CMD_RE = re.compile(
    r"\\(?:textbf|textit|emph|text|mathrm|mathbf|mathcal|mathbb|operatorname)"
    r"\{([^}]*)\}",
)
_LATEX_ACCENT_RE = re.compile(r"\\['\"`~^=.uvHtcdb]\{?(\w)\}?")
_LATEX_DOLLAR_RE = re.compile(r"\$([^$]*)\$")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """
    Clean LaTeX-heavy academic text for embedding.

    Steps:
        1. Strip leading/trailing whitespace per line (arXiv abstracts
           are indented with two spaces).
        2. Remove common LaTeX formatting commands, keeping the content.
        3. Collapse inline math delimiters; leave the math content.
        4. Normalise whitespace.
    """
    if not text:
        return ""

    # Strip per-line leading whitespace (arXiv abstracts are indented)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # LaTeX formatting → keep inner text
    text = _LATEX_CMD_RE.sub(r"\1", text)
    # Accented characters → base character
    text = _LATEX_ACCENT_RE.sub(r"\1", text)
    # Inline math $...$ → keep content
    text = _LATEX_DOLLAR_RE.sub(r"\1", text)
    # Remove remaining backslash commands (e.g. \cite{...})
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    # Remove stray backslashes before letters
    text = re.sub(r"\\([a-zA-Z])", r"\1", text)

    # Normalise whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


def clean_title(title: str) -> str:
    """Clean a paper title (single-line, no newlines)."""
    title = title.replace("\n", " ").replace("\r", " ")
    return clean_text(title)


def format_authors(authors_parsed: list[list[str]] | None, authors_str: str) -> str:
    """
    Format author names from the parsed representation.

    The parsed form is a list of [last, first, suffix] triples.
    Falls back to the raw ``authors`` string if parsed is unavailable.
    """
    if not authors_parsed:
        return clean_text(authors_str)

    names = []
    for parts in authors_parsed:
        last = parts[0].strip() if len(parts) > 0 else ""
        first = parts[1].strip() if len(parts) > 1 else ""
        suffix = parts[2].strip() if len(parts) > 2 else ""
        if suffix:
            names.append(f"{first} {last} {suffix}".strip())
        elif first:
            names.append(f"{first} {last}".strip())
        else:
            names.append(last)
    return ", ".join(names)


# Streaming Reader


def stream_arxiv_records(
    filepath: str | Path,
    categories_filter: set[str] | None = None,
    category_prefix_filter: set[str] | None = None,
    start_date: str | None = None,
    max_records: int | None = None,
) -> Iterator[ArxivRecord]:
    """
    Stream-read the arXiv metadata JSON-Lines file, yielding cleaned records.

    This is **memory-efficient**: only one line is in memory at a time.

    Args:
        filepath: Path to ``arxiv-metadata-oai-snapshot.json``.
        categories_filter: If provided, only yield records whose categories
            contain at least one of these exact category strings
            (e.g. ``{"cs.AI", "cs.CL"}``).
        category_prefix_filter: If provided, only yield records whose
            categories start with one of these prefixes
            (e.g. ``{"cs.", "stat."}`` for all CS and Statistics papers).
        start_date: If provided, only yield records updated on or after
            this date (format: ``YYYY-MM-DD``).
        max_records: Stop after yielding this many records.

    Yields:
        Cleaned ``ArxivRecord`` objects.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    yielded = 0
    skipped = 0

    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_num)
                skipped += 1
                continue

            # Apply filters
            cats = raw.get("categories", "")

            if categories_filter:
                record_cats = set(cats.split())
                if not record_cats.intersection(categories_filter):
                    continue

            if category_prefix_filter and not any(
                cats.startswith(prefix) or f" {prefix}" in f" {cats}"
                for prefix in category_prefix_filter
            ):
                # More robust: check each individual category
                record_cats = cats.split()
                if not any(
                    cat.startswith(prefix)
                    for cat in record_cats
                    for prefix in category_prefix_filter
                ):
                    continue

            update_date = raw.get("update_date", "")
            if start_date and update_date < start_date:
                continue

            # Build cleaned record
            record = ArxivRecord(
                arxiv_id=raw.get("id", ""),
                title=clean_title(raw.get("title", "")),
                authors=format_authors(
                    raw.get("authors_parsed"),
                    raw.get("authors", ""),
                ),
                abstract=clean_text(raw.get("abstract", "")),
                categories=cats,
                doi=raw.get("doi"),
                journal_ref=raw.get("journal-ref"),
                update_date=update_date,
            )

            # Skip records with empty abstracts
            if not record.abstract:
                skipped += 1
                continue

            yield record
            yielded += 1

            if max_records and yielded >= max_records:
                break

            # Progress logging every 100k records
            if yielded % 100_000 == 0:
                logger.info("  … streamed %d records so far …", yielded)

    logger.info(
        "Streaming complete: %d records yielded, %d skipped, %d lines read.",
        yielded,
        skipped,
        line_num,
    )


# Preprocessing Pipeline


@dataclass
class PreprocessingStats:
    """Statistics from a preprocessing run."""

    total_input_lines: int = 0
    records_output: int = 0
    records_skipped: int = 0
    categories_seen: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        top_cats = sorted(self.categories_seen.items(), key=lambda x: -x[1])[:15]
        cat_lines = "\n".join(f"    {cat}: {count:,}" for cat, count in top_cats)
        return (
            f"Preprocessing Summary\n"
            f"{'=' * 40}\n"
            f"  Input lines read  : {self.total_input_lines:,}\n"
            f"  Records output    : {self.records_output:,}\n"
            f"  Records skipped   : {self.records_skipped:,}\n"
            f"  Unique categories : {len(self.categories_seen):,}\n"
            f"  Elapsed time      : {self.elapsed_seconds:.1f}s\n"
            f"\n  Top 15 categories:\n{cat_lines}"
        )


def preprocess_dataset(
    input_path: str | Path,
    output_path: str | Path,
    categories_filter: set[str] | None = None,
    category_prefix_filter: set[str] | None = None,
    start_date: str | None = None,
    max_records: int | None = None,
) -> PreprocessingStats:

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = PreprocessingStats()
    t0 = time.time()

    logger.info("Starting preprocessing: %s -> %s", input_path, output_path)
    if categories_filter:
        logger.info("  Category filter: %s", categories_filter)
    if category_prefix_filter:
        logger.info("  Category prefix filter: %s", category_prefix_filter)
    if start_date:
        logger.info("  Start date filter: %s", start_date)
    if max_records:
        logger.info("  Max records: %d", max_records)

    with open(output_path, "w", encoding="utf-8") as out:
        for record in stream_arxiv_records(
            input_path,
            categories_filter=categories_filter,
            category_prefix_filter=category_prefix_filter,
            start_date=start_date,
            max_records=max_records,
        ):
            out.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            stats.records_output += 1

            # Track categories
            for cat in record.categories.split():
                stats.categories_seen[cat] = stats.categories_seen.get(cat, 0) + 1

    # Count total input lines for stats
    with open(input_path, encoding="utf-8") as f:
        stats.total_input_lines = sum(1 for _ in f)

    stats.records_skipped = stats.total_input_lines - stats.records_output
    stats.elapsed_seconds = time.time() - t0

    logger.info(stats.summary())
    return stats


# CLI Entry Point


def main() -> None:
    """Run the preprocessor from the command line."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess arXiv metadata for SourceSleuth.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/arxiv-metadata-oai-snapshot.json",
        help="Path to raw arXiv JSON-Lines file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/arxiv_preprocessed.jsonl",
        help="Path for cleaned output file.",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="*",
        default=None,
        help="Exact arXiv categories to include (e.g. cs.AI cs.CL).",
    )
    parser.add_argument(
        "--category-prefix",
        "-p",
        nargs="*",
        default=None,
        help="Category prefixes to include (e.g. cs. stat. math.).",
    )
    parser.add_argument(
        "--start-date",
        "-d",
        default=None,
        help="Only include records updated on/after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--max-records",
        "-n",
        type=int,
        default=None,
        help="Maximum number of records to output.",
    )

    args = parser.parse_args()

    cats = set(args.categories) if args.categories else None
    prefixes = set(args.category_prefix) if args.category_prefix else None

    stats = preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        categories_filter=cats,
        category_prefix_filter=prefixes,
        start_date=args.start_date,
        max_records=args.max_records,
    )

    print("\n" + stats.summary())


if __name__ == "__main__":
    main()
