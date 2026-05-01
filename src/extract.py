"""PDF extraction with PyMuPDF.

Per page: extract → NFKC normalize → strip gazette header / page numbers.
Concatenate with a marker that lets us track per-page offsets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from normalize import flatten, nfkc, strip_page_artifacts


@dataclass
class Page:
    page_num: int  # 1-indexed
    raw_text: str
    cleaned_text: str  # NFKC + header-stripped + flattened


def extract_pdf(pdf_path: str | Path) -> list[Page]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages: list[Page] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            cleaned = flatten(strip_page_artifacts(nfkc(raw)))
            pages.append(Page(page_num=i, raw_text=raw, cleaned_text=cleaned))
    finally:
        doc.close()

    return pages


def concatenate_pages(pages: list[Page]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate cleaned page text. Returns (full_text, page_offsets).

    page_offsets[i] = (start_char, end_char) in full_text for page i+1.
    Pages are joined with " " separator so word boundaries survive.
    """
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    sep = " "
    for idx, p in enumerate(pages):
        body = p.cleaned_text
        parts.append(body)
        offsets.append((cursor, cursor + len(body)))
        cursor += len(body)
        if idx < len(pages) - 1:
            parts.append(sep)
            cursor += len(sep)
    return "".join(parts), offsets


def page_span_for_offset(start: int, end: int, offsets: list[tuple[int, int]]) -> list[int]:
    """Map a (start, end) char range in concatenated text to source page numbers."""
    pages: list[int] = []
    for i, (ps, pe) in enumerate(offsets, start=1):
        if pe <= start:
            continue
        if ps >= end:
            break
        pages.append(i)
    return pages
