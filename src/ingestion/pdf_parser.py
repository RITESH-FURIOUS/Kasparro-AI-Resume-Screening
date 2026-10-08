from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class ParsedPDF:
    """
    Result of parsing a PDF resume.

    pages:
        Text extracted from each page individually.

    text:
        Full normalized resume text with page boundaries preserved.
    """

    filename: str
    path: str
    pages: list[str]
    text: str
    page_count: int


class PDFParseError(Exception):
    """Raised when a PDF cannot be reliably parsed."""


def normalize_page_text(text: str) -> str:
    """
    Normalize extracted PDF text without destroying useful structure.

    We intentionally do NOT aggressively collapse whitespace because
    resumes often use spacing/newlines to separate sections.
    """

    if not text:
        return ""

    # Normalize different newline styles.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null characters occasionally present in malformed PDFs.
    text = text.replace("\x00", "")

    # Remove trailing spaces from individual lines.
    lines = [line.rstrip() for line in text.split("\n")]

    # Remove excessive blank lines while preserving section boundaries.
    normalized_lines: list[str] = []
    previous_blank = False

    for line in lines:
        is_blank = not line.strip()

        if is_blank:
            if previous_blank:
                continue

            normalized_lines.append("")
            previous_blank = True
            continue

        normalized_lines.append(line)
        previous_blank = False

    return "\n".join(normalized_lines).strip()


def parse_pdf(path: Path) -> ParsedPDF:
    """
    Extract text from every page of a PDF.

    The parser does not assume:
    - a fixed number of pages
    - a particular section order
    - a particular resume layout

    Every page is processed independently first and then combined.
    """

    if not path.exists():
        raise PDFParseError(f"File does not exist: {path}")

    if not path.is_file():
        raise PDFParseError(f"Path is not a file: {path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise PDFParseError(
            f"Could not open PDF: {exc}"
        ) from exc

    if not reader.pages:
        raise PDFParseError("PDF contains no pages.")

    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
            page_text = normalize_page_text(raw_text)

        except Exception as exc:
            # We don't immediately kill the entire PDF if one page
            # has an extraction problem.
            page_text = (
                f"[Page {page_number} could not be extracted: "
                f"{type(exc).__name__}]"
            )

        pages.append(page_text)

    # Keep explicit page boundaries.
    page_blocks: list[str] = []

    for index, page_text in enumerate(pages, start=1):
        if page_text:
            page_blocks.append(
                f"--- PAGE {index} ---\n{page_text}"
            )
        else:
            page_blocks.append(
                f"--- PAGE {index} ---\n[NO EXTRACTABLE TEXT]"
            )

    full_text = "\n\n".join(page_blocks).strip()

    # A PDF can technically open successfully but contain no
    # extractable text, e.g. a scanned/image-only resume.
    extracted_character_count = sum(
        len(page.strip()) for page in pages
    )

    if extracted_character_count == 0:
        raise PDFParseError(
            "PDF contains no extractable text. "
            "It may be a scanned/image-only resume."
        )

    return ParsedPDF(
        filename=path.name,
        path=str(path),
        pages=pages,
        text=full_text,
        page_count=len(reader.pages),
    )