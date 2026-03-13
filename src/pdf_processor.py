from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


logger = logging.getLogger("sourcesleuth.pdf_processor")

# Configuration

DEFAULT_CHUNK_SIZE = 500  # tokens (≈ 375 words)
DEFAULT_CHUNK_OVERLAP = 50  # tokens overlap between consecutive chunks
APPROX_CHARS_PER_TOKEN = 4  # rough estimate for English text

# Sentence-window chunking defaults
DEFAULT_SENTENCES_PER_WINDOW = 4  # group 3-5 sentences per chunk
DEFAULT_SENTENCE_OVERLAP = 1  # 1-sentence overlap between windows

# Regex-based sentence splitter
# Uses a simpler approach: split on sentence-ending punctuation followed by space
# This avoids Python's fixed-width lookbehind requirement
_SENTENCE_END_RE = re.compile(r"([.!?]+)\s+")


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex-based segmentation.

    Handles common abbreviations by checking for known patterns.
    Falls back to newline-based splitting if regex produces fewer than 2 sentences.

    Args:
        text: The text to split into sentences.

    Returns:
        List of sentences.
    """
    # Common abbreviations that shouldn't end sentences
    abbreviations = {
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "Sr",
        "Jr",
        "vs",
        "etc",
        "al",
        "e.g",
        "i.e",
        "approx",
        "Inc",
        "Ltd",
        "Corp",
        "Co",
    }

    # Split on sentence-ending punctuation, keeping the punctuation
    parts = _SENTENCE_END_RE.split(text)

    # Reconstruct sentences (parts alternates between text and punctuation)
    sentences = []
    i = 0
    current_sentence = ""

    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue

        # Check if next part is punctuation
        if i + 1 < len(parts) and parts[i + 1] in [".", "!", "?"]:
            punct = parts[i + 1]
            # Check if this might be an abbreviation (last word before punct)
            words = part.split()
            last_word = words[-1].rstrip(".") if words else ""

            if last_word in abbreviations:
                # Abbreviation - continue building sentence
                current_sentence += part + punct + " "
                i += 2
                # Also consume the next part if it exists
                if i < len(parts):
                    current_sentence += parts[i] + " "
                    i += 1
            else:
                # End of sentence
                current_sentence += part + punct
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
                i += 2
        else:
            # No punctuation, continue building sentence
            current_sentence += part + " "
            i += 1

    # Add any remaining text
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    # Fallback: if we got very few sentences, try splitting on newlines
    if len(sentences) < 2 and "\n" in text:
        sentences = [s.strip() for s in text.split("\n") if s.strip()]

    return sentences


@dataclass
class TextChunk:
    """A single chunk of text extracted from a PDF."""

    text: str
    filename: str
    page: int  # 1-indexed page number
    chunk_index: int  # position within the document
    start_char: int  # character offset in the full document text
    end_char: int  # character offset in the full document text

    # Document-level metadata extracted from PDF
    title: str = ""
    authors: str = ""
    creation_date: str = ""
    publisher: str = ""
    journal: str = ""
    doi: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        return {
            "text": self.text,
            "filename": self.filename,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "title": self.title,
            "authors": self.authors,
            "creation_date": self.creation_date,
            "publisher": self.publisher,
            "journal": self.journal,
            "doi": self.doi,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TextChunk:
        """Reconstruct from a dictionary."""
        return cls(**data)


@dataclass
class PageSpan:
    """Tracks which page a character range belongs to."""

    page: int  # 1-indexed
    start_char: int
    end_char: int


@dataclass
class PDFDocument:
    """Represents a fully extracted PDF document."""

    filename: str
    full_text: str
    page_spans: list[PageSpan] = field(default_factory=list)
    chunks: list[TextChunk] = field(default_factory=list)

    # Document-level metadata extracted from PDF metadata dictionary
    title: str = ""
    authors: str = ""
    creation_date: str = ""
    publisher: str = ""
    journal: str = ""
    doi: str = ""


def _extract_pdf_metadata(doc: fitz.Document) -> dict:
    """
    Extract metadata from a PyMuPDF document object.

    PyMuPDF provides a metadata dictionary with keys like:
    - title, author, subject, keywords
    - creator, producer, creationDate, modDate
    - format (PDF version), encryption

    Args:
        doc: Open PyMuPDF document object.

    Returns:
        Dictionary with extracted metadata fields.
    """
    try:
        meta = doc.metadata
    except Exception:
        logger.warning("Could not extract metadata from PDF.")
        return {}

    # Map PyMuPDF metadata keys to our schema
    # Note: PyMuPDF uses 'author' (singular), but we store as 'authors'
    extracted = {
        "title": meta.get("title", "") or "",
        "authors": meta.get("author", "") or "",
        "creation_date": meta.get("creationDate", "") or "",
        "publisher": meta.get("creator", "") or meta.get("producer", "") or "",
        "journal": meta.get("subject", "") or "",  # Sometimes journal info is in subject
        "doi": "",  # DOI typically not in PDF metadata, may need to extract from text
    }

    # Clean up metadata - remove None values, strip whitespace
    for key in extracted:
        if extracted[key] is None:
            extracted[key] = ""
        else:
            extracted[key] = str(extracted[key]).strip()

    return extracted


# Extraction


def extract_text_from_pdf(pdf_path: str | Path) -> PDFDocument:
    """
    Extract all text from a PDF file, tracking per-page boundaries and metadata.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A PDFDocument with full_text, page_spans, and metadata populated.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If the PDF cannot be opened or parsed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{pdf_path.name}': {exc}") from exc

    # Extract metadata first
    metadata = _extract_pdf_metadata(doc)

    full_text_parts: list[str] = []
    page_spans: list[PageSpan] = []
    current_offset = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")

        if not page_text.strip():
            continue

        start = current_offset
        full_text_parts.append(page_text)
        current_offset += len(page_text)

        page_spans.append(
            PageSpan(
                page=page_num + 1,  # 1-indexed
                start_char=start,
                end_char=current_offset,
            )
        )

    doc.close()

    full_text = "".join(full_text_parts)
    logger.info(
        "Extracted %d characters from %d pages of '%s'",
        len(full_text),
        len(page_spans),
        pdf_path.name,
    )

    return PDFDocument(
        filename=pdf_path.name,
        full_text=full_text,
        page_spans=page_spans,
        title=metadata.get("title", ""),
        authors=metadata.get("authors", ""),
        creation_date=metadata.get("creation_date", ""),
        publisher=metadata.get("publisher", ""),
        journal=metadata.get("journal", ""),
        doi=metadata.get("doi", ""),
    )


# Chunking


def _char_size(token_count: int) -> int:
    """Convert a token count to an approximate character count."""
    return token_count * APPROX_CHARS_PER_TOKEN


def chunk_text(
    document: PDFDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split the document text into overlapping chunks.

    Uses a sliding-window approach over character offsets, then maps each
    chunk back to its originating page.

    Args:
        document: A previously extracted PDFDocument.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between consecutive chunks in tokens.

    Returns:
        A list of TextChunk objects with document metadata attached.
    """
    text = document.full_text
    if not text.strip():
        logger.warning("Document '%s' has no extractable text.", document.filename)
        return []

    char_chunk = _char_size(chunk_size)
    char_overlap = _char_size(chunk_overlap)
    stride = max(char_chunk - char_overlap, 1)

    chunks: list[TextChunk] = []
    idx = 0
    start = 0

    while start < len(text):
        end = min(start + char_chunk, len(text))
        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            page = _resolve_page(document.page_spans, start)
            chunks.append(
                TextChunk(
                    text=chunk_text_str,
                    filename=document.filename,
                    page=page,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    title=document.title,
                    authors=document.authors,
                    creation_date=document.creation_date,
                    publisher=document.publisher,
                    journal=document.journal,
                    doi=document.doi,
                )
            )
            idx += 1

        start += stride

    document.chunks = chunks
    logger.info(
        "Chunked '%s' into %d chunks (size=%d, overlap=%d tokens).",
        document.filename,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# Sentence-Window Chunking


def chunk_text_by_sentences(
    document: PDFDocument,
    sentences_per_window: int = DEFAULT_SENTENCES_PER_WINDOW,
    sentence_overlap: int = DEFAULT_SENTENCE_OVERLAP,
) -> list[TextChunk]:
    """
    Split document text into overlapping sentence-window chunks.

    Segments the text by sentence boundaries and groups them into
    sliding windows. This prevents short, dense quotes from being
    diluted by surrounding unrelated text (e.g., a 10-word quote
    inside 500 tokens of math).

    Args:
        document: A previously extracted PDFDocument.
        sentences_per_window: Number of sentences per chunk (default 4).
        sentence_overlap: Number of overlapping sentences between
            consecutive windows (default 1).

    Returns:
        A list of TextChunk objects with sentence-aligned boundaries
        and document metadata attached.
    """
    text = document.full_text
    if not text.strip():
        logger.warning("Document '%s' has no extractable text.", document.filename)
        return []

    sentences = _split_sentences(text)
    if not sentences:
        logger.warning("Could not segment '%s' into sentences.", document.filename)
        return []

    stride = max(sentences_per_window - sentence_overlap, 1)
    chunks: list[TextChunk] = []
    idx = 0

    for start_idx in range(0, len(sentences), stride):
        end_idx = min(start_idx + sentences_per_window, len(sentences))
        window_text = " ".join(sentences[start_idx:end_idx]).strip()

        if not window_text:
            continue

        # Compute char offsets by finding this text in the full document
        char_start = text.find(sentences[start_idx])
        char_end_sent = sentences[end_idx - 1]
        char_end = (
            text.find(char_end_sent) + len(char_end_sent)
            if char_end_sent
            else char_start + len(window_text)
        )
        if char_start < 0:
            char_start = 0
        if char_end < char_start:
            char_end = char_start + len(window_text)

        page = _resolve_page(document.page_spans, max(char_start, 0))

        chunks.append(
            TextChunk(
                text=window_text,
                filename=document.filename,
                page=page,
                chunk_index=idx,
                start_char=char_start,
                end_char=char_end,
                title=document.title,
                authors=document.authors,
                creation_date=document.creation_date,
                publisher=document.publisher,
                journal=document.journal,
                doi=document.doi,
            )
        )
        idx += 1

        if end_idx >= len(sentences):
            break

    document.chunks = chunks
    logger.info(
        "Sentence-chunked '%s' into %d chunks (window=%d, overlap=%d sentences).",
        document.filename,
        len(chunks),
        sentences_per_window,
        sentence_overlap,
    )
    return chunks


def _resolve_page(page_spans: list[PageSpan], char_offset: int) -> int:
    """Map a character offset back to its 1-indexed page number."""
    for span in page_spans:
        if span.start_char <= char_offset < span.end_char:
            return span.page
    # Fallback: return last page if offset is at the very end
    return page_spans[-1].page if page_spans else 1


# Batch Processing


def process_pdf_directory(
    directory: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = "sentence",
    use_ocr: bool = False,
    ocr_language: str = "eng",
) -> list[TextChunk]:
    """
    Process all PDFs in a directory and return a flat list of chunks.

    Args:
        directory: Path to the directory containing PDF files.
        chunk_size: Target chunk size in tokens (for 'fixed' strategy).
        chunk_overlap: Overlap in tokens (for 'fixed' strategy).
        strategy: Chunking strategy — 'sentence' (sentence-window,
                  default) or 'fixed' (sliding character window).
        use_ocr: If True, use OCR for scanned PDFs. If False, skip
                 OCR and only process text-based PDFs.
        ocr_language: Tesseract OCR language code (default: "eng").
                      Only used if use_ocr=True.

    Returns:
        Flat list of TextChunk objects from all PDFs in the directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", directory)
        return []

    all_chunks: list[TextChunk] = []

    for pdf_path in pdf_files:
        try:
            # Try standard extraction first
            document = extract_text_from_pdf(pdf_path)

            # If no text extracted and OCR is enabled, try OCR
            if not document.full_text.strip() and use_ocr:
                logger.info(
                    "No text found in '%s', attempting OCR (language: %s)...",
                    pdf_path.name,
                    ocr_language,
                )
                try:
                    from src.ocr_processor import process_pdf_with_ocr_fallback

                    ocr_text, used_ocr = process_pdf_with_ocr_fallback(
                        pdf_path, language=ocr_language
                    )
                    if used_ocr:
                        logger.info(
                            "OCR successful for '%s': %d chars extracted",
                            pdf_path.name,
                            len(ocr_text),
                        )
                        document.full_text = ocr_text
                    else:
                        logger.warning(
                            "OCR did not extract text from '%s'. Skipping.",
                            pdf_path.name,
                        )
                        continue
                except Exception as ocr_exc:
                    logger.error(
                        "OCR failed for '%s': %s. Skipping.",
                        pdf_path.name,
                        ocr_exc,
                    )
                    continue

            if strategy == "sentence":
                chunks = chunk_text_by_sentences(document)
            else:
                chunks = chunk_text(document, chunk_size, chunk_overlap)

            all_chunks.extend(chunks)
            logger.info(
                "Processed '%s' -> %d chunks (strategy=%s, ocr=%s, lang=%s)",
                pdf_path.name,
                len(chunks),
                strategy,
                use_ocr,
                ocr_language,
            )
        except Exception as exc:
            logger.error("Failed to process '%s': %s", pdf_path.name, exc)

    logger.info(
        "Total: processed %d PDFs -> %d chunks from '%s' (strategy=%s, ocr=%s, lang=%s).",
        len(pdf_files),
        len(all_chunks),
        directory,
        strategy,
        use_ocr,
        ocr_language,
    )
    return all_chunks
