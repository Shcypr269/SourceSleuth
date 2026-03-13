"""
OCR Processor Module for SourceSleuth.

Provides optical character recognition (OCR) capabilities for scanned PDFs
and image-based documents using Tesseract OCR.

This module extends SourceSleuth to support:
- Scanned academic papers (image-only PDFs)
- Historical documents
- Textbook pages captured as images
- Handwritten notes (limited accuracy)

Dependencies:
    - pytesseract: Python wrapper for Tesseract OCR
    - Pillow: Image processing library
    - pdf2image: PDF to image conversion
    - tesseract-ocr: System-level OCR engine (must be installed separately)

Installation:
    # Install Python dependencies
    pip install -e ".[ocr]"

    # Install Tesseract OCR engine
    # Ubuntu/Debian:
    sudo apt-get install tesseract-ocr

    # macOS:
    brew install tesseract

    # Windows:
    # Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("sourcesleuth.ocr_processor")

# Try to import optional OCR dependencies
try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR dependencies not installed. Install with: pip install sourcesleuth[ocr]")

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not installed. Install with: pip install sourcesleuth[ocr]")


@dataclass
class OCRResult:
    """Result from OCR processing of a single page."""

    page_number: int
    text: str
    confidence: float  # 0-100 confidence score from Tesseract
    language: str = "eng"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "page_number": self.page_number,
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
        }


def ocr_image(image_path: str | Path, language: str = "eng") -> OCRResult:
    """
    Perform OCR on a single image file.

    Args:
        image_path: Path to the image file (PNG, JPG, TIFF, etc.)
        language: OCR language code (default: "eng" for English)
                  Use "eng+fra" for multiple languages

    Returns:
        OCRResult with extracted text and confidence score.

    Raises:
        RuntimeError: If OCR dependencies are not installed.
        FileNotFoundError: If the image file does not exist.
    """
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR not available. Install with: pip install sourcesleuth[ocr]")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info("Performing OCR on %s (language: %s)", image_path.name, language)

    # Open image and run OCR
    image = Image.open(image_path)

    # Get OCR data with confidence scores
    data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)

    # Calculate average confidence (filter out low-confidence detections)
    confidences = [c for c in data["conf"] if c > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Extract full text
    text = pytesseract.image_to_string(image, lang=language)

    logger.info(
        "OCR complete for %s: %d chars, confidence: %.1f%%",
        image_path.name,
        len(text),
        avg_confidence,
    )

    return OCRResult(
        page_number=0,  # Single image has no page number
        text=text,
        confidence=avg_confidence,
        language=language,
    )


def ocr_pdf(pdf_path: str | Path, language: str = "eng", dpi: int = 300) -> list[OCRResult]:
    """
    Perform OCR on a scanned PDF (image-only PDF).

    Converts each PDF page to an image and runs Tesseract OCR.

    Args:
        pdf_path: Path to the PDF file.
        language: OCR language code (default: "eng")
        dpi: Resolution for PDF to image conversion (default: 300)
             Higher DPI = better accuracy but slower processing.

    Returns:
        List of OCRResult objects, one per page.

    Raises:
        RuntimeError: If OCR dependencies are not installed or language pack is missing.
        FileNotFoundError: If the PDF does not exist.
    """
    if not OCR_AVAILABLE or not PDF2IMAGE_AVAILABLE:
        raise RuntimeError(
            "OCR not available. Install with: pip install sourcesleuth[ocr]"
            " and ensure tesseract-ocr is installed on your system."
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(
        "Performing OCR on PDF: %s (language: %s, dpi: %d)",
        pdf_path.name,
        language,
        dpi,
    )

    # Convert PDF pages to images
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as exc:
        raise RuntimeError(f"Failed to convert PDF to images: {exc}") from exc

    logger.info("Converted %d pages to images", len(images))

    # Run OCR on each page
    results = []
    for page_num, image in enumerate(images, start=1):
        logger.info("Processing page %d/%d", page_num, len(images))

        try:
            # Get OCR data with confidence
            data = pytesseract.image_to_data(
                image, lang=language, output_type=pytesseract.Output.DICT
            )
        except Exception as lang_exc:
            # Handle missing language pack gracefully
            error_msg = str(lang_exc)
            if "data file" in error_msg.lower() or "tessdata" in error_msg.lower():
                raise RuntimeError(
                    f"OCR language pack '{language}' is not installed. "
                    f"The default Docker container only includes English ('eng'). "
                    f"To use '{language}', install the language pack: "
                    f"apt-get install tesseract-ocr-{language[:3]}"
                ) from lang_exc
            raise

        # Calculate average confidence
        confidences = [c for c in data["conf"] if c > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Extract text
        text = pytesseract.image_to_string(image, lang=language)

        results.append(
            OCRResult(
                page_number=page_num,
                text=text,
                confidence=avg_confidence,
                language=language,
            )
        )

        logger.info(
            "Page %d: %d chars, confidence: %.1f%%",
            page_num,
            len(text),
            avg_confidence,
        )

    logger.info(
        "OCR complete for %s: %d pages processed",
        pdf_path.name,
        len(results),
    )

    return results


def is_scanned_pdf(pdf_path: str | Path) -> bool:
    """
    Detect if a PDF is scanned (image-only) vs text-based.

    Uses a simple heuristic: if no extractable text is found,
    the PDF is likely scanned.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        True if the PDF appears to be scanned, False otherwise.
    """
    try:
        # Try to extract text using PyMuPDF
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        has_text = False

        for page_num in range(min(len(doc), 3)):  # Check first 3 pages
            page = doc[page_num]
            text = page.get_text("text").strip()
            if len(text) > 100:  # Found substantial text
                has_text = True
                break

        doc.close()
        return not has_text

    except Exception:
        # If we can't check, assume it's scanned
        return True


def process_pdf_with_ocr_fallback(
    pdf_path: str | Path,
    language: str = "eng",
    dpi: int = 300,
) -> tuple[str, bool]:
    """
    Process a PDF, using OCR only if needed.

    First attempts standard text extraction with PyMuPDF.
    If no text is found (scanned PDF), falls back to OCR.

    Args:
        pdf_path: Path to the PDF file.
        language: OCR language code (default: "eng").
                  Used only if OCR is needed.
        dpi: Resolution for OCR (used only if OCR is needed).

    Returns:
        Tuple of (extracted_text, used_ocr) where used_ocr indicates
        whether OCR was required.
    """
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)

    # Try standard text extraction first
    try:
        doc = fitz.open(pdf_path)
        full_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            full_text.append(text)

        doc.close()
        combined_text = "".join(full_text)

        # If we got substantial text, return it
        if len(combined_text.strip()) > 100:
            logger.info(
                "Standard extraction successful for %s (%d chars)",
                pdf_path.name,
                len(combined_text),
            )
            return combined_text, False

    except Exception as exc:
        logger.warning(
            "Standard extraction failed for %s: %s. Falling back to OCR.",
            pdf_path.name,
            exc,
        )

    # Fall back to OCR
    if not OCR_AVAILABLE or not PDF2IMAGE_AVAILABLE:
        raise RuntimeError(
            f"Standard extraction failed and OCR not available for {pdf_path.name}. "
            "Install OCR dependencies: pip install sourcesleuth[ocr]"
        )

    logger.info("Falling back to OCR for %s (language: %s)", pdf_path.name, language)
    ocr_results = ocr_pdf(pdf_path, language=language, dpi=dpi)
    combined_text = "".join(result.text for result in ocr_results)

    return combined_text, True
