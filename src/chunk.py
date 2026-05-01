"""Chunking rules for Egyptian Labor Law.

One chunk per article, with three deviations:
  1. Article 1 (definitions): split per defined term into :Definition nodes,
     plus one whole-article :Article node.
  2. Long articles (>1500 chars): split on top-level enumeration boundaries
     into :ArticleSegment nodes.
  3. Short articles (<150 chars): keep as standalone :Article nodes.

Each chunk carries a breadcrumb header in the embedded text and full
hierarchical metadata in fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterator

import regex as re

from extract import page_span_for_offset
from normalize import (
    estimate_tokens,
    normalize_for_display,
    normalize_for_index,
    pdf_arabic_int,
)
from parse_structure import ParsedArticle

LAW_ID = "EG-LAW-14-2025"
LAW_TITLE = "قانون العمل"
LAW_ISSUED_DATE = "2025-05-03"
LAW_GAZETTE_ISSUE = "81 (تابع)"

LONG_ARTICLE_CHAR_THRESHOLD = 1500
ENUM_BOUNDARY_RX = re.compile(
    r"(?<![\d٠-٩۰-۹])([\d٠-٩۰-۹]{1,2})\s*[-–]\s*"
)
TERM_HEADER_RX = re.compile(
    r"(?<![\d٠-٩۰-۹])([\d٠-٩۰-۹]{1,2})\s*[-–]\s*([^:.]+?)\s*[:\.]"
)


@dataclass
class Chunk:
    node_id: str
    node_type: str  # Article | ArticleSegment | Definition
    law_id: str = LAW_ID
    law_title: str = LAW_TITLE
    book_number: int | None = None
    book_title: str = ""
    part_number: int | None = None
    part_title: str = ""
    chapter_number: int | None = None
    chapter_title: str = ""
    article_number: int | None = None
    is_definition: bool = False
    defined_term: str | None = None
    term_variants: list[str] = field(default_factory=list)
    segment_index: int | None = None
    segment_total: int | None = None
    text: str = ""              # display form, with breadcrumb prepended
    text_normalized: str = ""   # index/embedding form
    char_count: int = 0
    token_count_est: int = 0
    cross_refs: list[int] = field(default_factory=list)
    external_refs: list[str] = field(default_factory=list)
    defined_terms_used: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


def _breadcrumb(article: ParsedArticle, suffix: str = "") -> str:
    parts = []
    if article.book_number is not None:
        title = f": {article.book_title}" if article.book_title else ""
        parts.append(f"الكتاب {article.book_number}{title}")
    if article.part_number is not None:
        title = f": {article.part_title}" if article.part_title else ""
        parts.append(f"الباب {article.part_number}{title}")
    if article.chapter_number is not None:
        title = f": {article.chapter_title}" if article.chapter_title else ""
        parts.append(f"الفصل {article.chapter_number}{title}")
    parts.append(f"المادة ({article.article_number}){suffix}")
    return " ← ".join(parts)


def _build_text_pair(breadcrumb: str, body: str) -> tuple[str, str, int, int]:
    display_body = normalize_for_display(body)
    display = f"{breadcrumb}\n\n{display_body}"
    normalized = normalize_for_index(display)
    return display, normalized, len(display), estimate_tokens(display)


def _hierarchy_fields(article: ParsedArticle) -> dict[str, Any]:
    return dict(
        book_number=article.book_number,
        book_title=normalize_for_display(article.book_title),
        part_number=article.part_number,
        part_title=normalize_for_display(article.part_title),
        chapter_number=article.chapter_number,
        chapter_title=normalize_for_display(article.chapter_title),
    )


# Fallback for terms whose enumeration digit was dropped during extraction:
# matches " - TERM :" preceded by whitespace, with TERM being Arabic words.
UNNUMBERED_TERM_RX = re.compile(r"(?<=[\s.،])\s*[-–]\s+([ء-ي][ء-ي\s]{1,50}?)\s*[:\.]")


def _split_definitions(body: str) -> list[tuple[int | None, str, str]]:
    """Split Article 1 body into (term_number, term_name, definition_body) tuples.

    Walks left-to-right finding "N- term :" headers. If consecutive numbered
    terms have a gap (e.g., 10 then 12), scans the gap for an unnumbered
    " - TERM :" pattern to recover the missing term (assigns it the gap number).
    """
    numbered = list(TERM_HEADER_RX.finditer(body))
    boundaries: list[tuple[int, int, int | None, str]] = []  # (start, end, number, term)

    for i, m in enumerate(numbered):
        n = pdf_arabic_int(m.group(1))
        term = normalize_for_display(m.group(2)).strip()
        boundaries.append((m.start(), m.end(), n, term))

    # Gap-fill: between consecutive numbered terms, look for unnumbered "- term :"
    augmented: list[tuple[int, int, int | None, str]] = []
    for i, b in enumerate(boundaries):
        augmented.append(b)
        if i + 1 >= len(boundaries):
            continue
        cur_n = b[2]
        nxt_n = boundaries[i + 1][2]
        if cur_n is None or nxt_n is None or nxt_n <= cur_n + 1:
            continue
        # Gap of at least 1 missing number; scan the body slice.
        gap_start = b[1]
        gap_end = boundaries[i + 1][0]
        slot = cur_n + 1
        for u in UNNUMBERED_TERM_RX.finditer(body, pos=gap_start, endpos=gap_end):
            term = normalize_for_display(u.group(1)).strip()
            if term and slot < nxt_n:
                augmented.append((u.start(), u.end(), slot, term))
                slot += 1

    augmented.sort(key=lambda x: x[0])

    out: list[tuple[int | None, str, str]] = []
    for i, (start, end, n, term) in enumerate(augmented):
        next_start = augmented[i + 1][0] if i + 1 < len(augmented) else len(body)
        definition = body[end:next_start].strip(" :-–،.")
        if term:
            out.append((n, term, definition))
    return out


def _split_long_article(body: str) -> list[str]:
    """Split a long article body on top-level enumeration boundaries.

    Returns a list of segment bodies. The preamble (text before the first
    enumeration marker) is prepended to each segment as context.
    """
    boundaries = list(ENUM_BOUNDARY_RX.finditer(body))
    if len(boundaries) < 2:
        return [body]
    preamble = body[: boundaries[0].start()].strip()
    segments: list[str] = []
    for i, m in enumerate(boundaries):
        next_start = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(body)
        segment_body = body[m.start():next_start].strip()
        if preamble:
            segments.append(f"{preamble}\n\n{segment_body}")
        else:
            segments.append(segment_body)
    return segments


def build_chunks(
    articles: list[ParsedArticle],
    page_offsets: list[tuple[int, int]],
) -> tuple[list[Chunk], dict[str, str]]:
    """Apply chunking rules. Returns (chunks, glossary).

    glossary maps each defined term to its definition node_id, used at
    query time for definition-injection.
    """
    chunks: list[Chunk] = []
    glossary: dict[str, str] = {}

    for art in articles:
        pages = page_span_for_offset(art.char_start, art.char_end, page_offsets)

        if art.article_number == 1:
            chunks.extend(_chunk_article_1(art, pages, glossary))
            continue

        body = art.body_text.strip()
        if len(body) > LONG_ARTICLE_CHAR_THRESHOLD:
            chunks.extend(_chunk_long_article(art, body, pages))
        else:
            chunks.append(_chunk_normal_article(art, body, pages))

    return chunks, glossary


def _chunk_article_1(
    art: ParsedArticle,
    pages: list[int],
    glossary: dict[str, str],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    body = art.body_text.strip()

    # Whole-Article-1 chunk.
    breadcrumb = _breadcrumb(art)
    display, normalized, n_chars, n_tokens = _build_text_pair(breadcrumb, body)
    chunks.append(Chunk(
        node_id=f"{LAW_ID.lower()}-art{art.article_number:03d}",
        node_type="Article",
        article_number=art.article_number,
        text=display,
        text_normalized=normalized,
        char_count=n_chars,
        token_count_est=n_tokens,
        source_pages=pages,
        **_hierarchy_fields(art),
    ))

    # Per-term Definition chunks.
    for term_num, term_name, defn_body in _split_definitions(body):
        canonical = _canonical_term(term_name)
        node_id = f"{LAW_ID.lower()}-def-{_slug_term(canonical, term_num)}"
        suffix = f" → التعريف {term_num}: {term_name}" if term_num else f" → {term_name}"
        breadcrumb_def = _breadcrumb(art, suffix=suffix)
        display, normalized, n_chars, n_tokens = _build_text_pair(breadcrumb_def, defn_body)
        chunks.append(Chunk(
            node_id=node_id,
            node_type="Definition",
            article_number=art.article_number,
            is_definition=True,
            defined_term=canonical,
            term_variants=_term_variants(canonical),
            text=display,
            text_normalized=normalized,
            char_count=n_chars,
            token_count_est=n_tokens,
            source_pages=pages,
            **_hierarchy_fields(art),
        ))
        glossary[canonical] = node_id

    return chunks


def _chunk_long_article(
    art: ParsedArticle,
    body: str,
    pages: list[int],
) -> list[Chunk]:
    segments = _split_long_article(body)
    if len(segments) <= 1:
        return [_chunk_normal_article(art, body, pages)]

    chunks: list[Chunk] = []
    parent_id = f"{LAW_ID.lower()}-art{art.article_number:03d}"

    # Parent :Article holding the whole body.
    breadcrumb_parent = _breadcrumb(art)
    p_display, p_norm, p_chars, p_tokens = _build_text_pair(breadcrumb_parent, body)
    chunks.append(Chunk(
        node_id=parent_id,
        node_type="Article",
        article_number=art.article_number,
        text=p_display,
        text_normalized=p_norm,
        char_count=p_chars,
        token_count_est=p_tokens,
        source_pages=pages,
        **_hierarchy_fields(art),
    ))

    total = len(segments)
    for i, seg_body in enumerate(segments, start=1):
        seg_breadcrumb = _breadcrumb(art, suffix=f" → جزء {i}/{total}")
        display, normalized, n_chars, n_tokens = _build_text_pair(seg_breadcrumb, seg_body)
        chunks.append(Chunk(
            node_id=f"{parent_id}-seg{i:02d}",
            node_type="ArticleSegment",
            article_number=art.article_number,
            segment_index=i,
            segment_total=total,
            text=display,
            text_normalized=normalized,
            char_count=n_chars,
            token_count_est=n_tokens,
            source_pages=pages,
            **_hierarchy_fields(art),
        ))

    return chunks


def _chunk_normal_article(
    art: ParsedArticle,
    body: str,
    pages: list[int],
) -> Chunk:
    breadcrumb = _breadcrumb(art)
    display, normalized, n_chars, n_tokens = _build_text_pair(breadcrumb, body)
    return Chunk(
        node_id=f"{LAW_ID.lower()}-art{art.article_number:03d}",
        node_type="Article",
        article_number=art.article_number,
        text=display,
        text_normalized=normalized,
        char_count=n_chars,
        token_count_est=n_tokens,
        source_pages=pages,
        **_hierarchy_fields(art),
    )


# --- Term canonicalization ---

ARABIC_LETTERS_ONLY = re.compile(r"[ء-ي\s]+")


def _canonical_term(raw: str) -> str:
    s = normalize_for_display(raw).strip()
    s = re.sub(r"^\s*ال", "ال", s)  # ensure single leading "ال" if present
    return s


def _term_variants(canonical: str) -> list[str]:
    """Surface forms used to detect the term in article bodies."""
    variants = {canonical}
    if canonical.startswith("ال"):
        bare = canonical[2:]
        variants.add(bare)
        for prefix in ("لل", "بال", "وال", "فال", "كال"):
            variants.add(prefix + bare)
    return sorted(variants)


def _slug_term(term: str, n: int | None) -> str:
    """Stable, ASCII-safe slug for node_id."""
    if n is not None:
        return f"{n:03d}"
    return re.sub(r"\s+", "-", term)[:40]
