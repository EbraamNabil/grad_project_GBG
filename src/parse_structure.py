"""Structural parser for Egyptian Labor Law.

Operates on flattened (whitespace-collapsed) NFKC text from extract.py.

Hierarchy: الكتاب → الباب → الفصل (optional) → مادة

PDF quirks handled:
  - Cross-references "المادة (X)" and "المواد ..." filtered via lookbehind.
  - Multi-digit Eastern Arabic numerals are reversed in PDF storage; pdf_arabic_int
    handles the reversal heuristic.
  - Hierarchy ordinals are sometimes split across whitespace ("الثان ى" not "الثاني");
    the ordinal regex tolerates internal spaces.
"""
from __future__ import annotations

from dataclasses import dataclass

import regex as re

from normalize import pdf_arabic_int

ARABIC_PREFIX_LETTERS = "وفبلك"  # letters that, when prefixed to "مادة", form a different word

# Ordinals 1..13 with tolerance for inner whitespace and ya/yeh variants.
ORDINAL_PATTERNS: list[tuple[str, int]] = [
    (r"الأول\b|الاول\b|الأول\s*ى\b|الأولى\b", 1),
    (r"الثاني\b|الثانى\b|الثان\s*[يى]\b|الثانية\b", 2),
    (r"الثالث\b|الثالثة\b", 3),
    (r"الرابع\b|الرابعة\b", 4),
    (r"الخامس\b|الخامسة\b", 5),
    (r"السادس\b|السادسة\b", 6),
    (r"السابع\b|السابعة\b", 7),
    (r"الثامن\b|الثامنة\b", 8),
    (r"التاسع\b|التاسعة\b", 9),
    (r"العاشر\b|العاشرة\b", 10),
    (r"الحاد[يى]\s*عشر\b", 11),
    (r"الثان[يى]\s*عشر\b", 12),
    (r"الثالث\s*عشر\b", 13),
    (r"الرابع\s*عشر\b", 14),
    (r"الخامس\s*عشر\b", 15),
    (r"السادس\s*عشر\b", 16),
]

ORDINAL_RX = re.compile("|".join(f"({p})" for p, _ in ORDINAL_PATTERNS))


def _parse_ordinal_at(text: str, start: int, max_lookahead: int = 30) -> tuple[int, int] | None:
    """Try to match an ordinal starting at `start`. Returns (number, end_pos) or None."""
    m = ORDINAL_RX.match(text, pos=start, endpos=min(start + max_lookahead, len(text)))
    if not m:
        return None
    for i, group in enumerate(m.groups(), start=1):
        if group is not None:
            return ORDINAL_PATTERNS[i - 1][1], m.end()
    return None


# Article header. Lookbehind excludes prefix letters that turn "مادة" into another word
# ("المادة" cross-reference, "بمادة" etc.). The four letters of "مادة" may be split by
# at most one whitespace each (PDF rendering artifact: "ما دة", "م ادة", "ماد ة").
ARTICLE_RX = re.compile(
    r"(?<![اوفبلكم])م\s?ا\s?د\s?ة\s*[\(\)]\s*([\d٠-٩۰-۹]*)\s*[\(\)\:\.\،]"
)
# Hierarchy keyword regexes; lookahead requires a following ordinal-starting word "ال..."
BOOK_KW_RX = re.compile(r"(?<![اوفبلكم])الكتاب\s+(?=ال)")
PART_KW_RX = re.compile(r"(?<![اوفبلكم])الباب\s+(?=ال)")
CHAPTER_KW_RX = re.compile(r"(?<![اوفبلكم])الفصل\s+(?=ال)")


@dataclass
class Marker:
    level: str  # "book" | "part" | "chapter" | "article"
    number: int | None
    text: str
    start: int
    end: int  # end of header (start of body)


@dataclass
class ParsedArticle:
    article_number: int
    body_text: str
    book_number: int | None
    book_title: str
    part_number: int | None
    part_title: str
    chapter_number: int | None
    chapter_title: str
    char_start: int
    char_end: int


def _scan_articles(text: str) -> list[Marker]:
    out: list[Marker] = []
    for m in ARTICLE_RX.finditer(text):
        digit_token = m.group(1)
        num = pdf_arabic_int(digit_token) if digit_token else None
        out.append(Marker(level="article", number=num, text=m.group(0),
                          start=m.start(), end=m.end()))
    return out


def _scan_hierarchy(text: str, kw_rx: re.Pattern, level: str) -> list[Marker]:
    out: list[Marker] = []
    for m in kw_rx.finditer(text):
        ordinal_start = m.end()
        result = _parse_ordinal_at(text, ordinal_start)
        if result is None:
            continue
        num, ord_end = result
        out.append(Marker(level=level, number=num, text=text[m.start():ord_end],
                          start=m.start(), end=ord_end))
    return out


def _renumber_articles_sequentially(arts: list[Marker]) -> list[Marker]:
    """Resolve article numbers using detected digits + sequential constraint.

    Strategy: walk left to right; the canonical sequence is 1, 2, 3, ...
    For each detected article header, prefer (last + 1). If the parsed digit
    matches that, use it; if not, trust the sequence (digits may be missing
    or split across spans). Reset to detected number only on large gaps that
    look intentional (≥ +5) when the detected number is plausible.
    """
    out: list[Marker] = []
    expected = 1
    for m in arts:
        candidate = m.number
        if candidate is None:
            assigned = expected
        elif candidate == expected:
            assigned = expected
        elif 1 <= candidate <= expected + 3:
            # Tiny correction (e.g., a real article was missed); accept the digit
            assigned = candidate
        else:
            # Digits look bad; trust sequence
            assigned = expected
        out.append(Marker(level="article", number=assigned, text=m.text,
                          start=m.start, end=m.end))
        expected = assigned + 1
    return out


def parse(full_text: str) -> list[ParsedArticle]:
    article_markers = _renumber_articles_sequentially(_scan_articles(full_text))
    book_markers = _scan_hierarchy(full_text, BOOK_KW_RX, "book")
    part_markers = _scan_hierarchy(full_text, PART_KW_RX, "part")
    chapter_markers = _scan_hierarchy(full_text, CHAPTER_KW_RX, "chapter")

    all_markers = article_markers + book_markers + part_markers + chapter_markers
    all_markers.sort(key=lambda m: (m.start, _level_priority(m.level)))

    book_num: int | None = None
    book_title = ""
    part_num: int | None = None
    part_title = ""
    chap_num: int | None = None
    chap_title = ""

    articles: list[ParsedArticle] = []
    for idx, m in enumerate(all_markers):
        next_start = all_markers[idx + 1].start if idx + 1 < len(all_markers) else len(full_text)
        title_after = full_text[m.end:next_start].strip(" :-–،.()").split("مادة")[0]
        title_after = re.sub(r"\s+", " ", title_after).strip()[:100]

        if m.level == "book":
            book_num = m.number
            book_title = title_after
            part_num, part_title = None, ""
            chap_num, chap_title = None, ""
        elif m.level == "part":
            part_num = m.number
            part_title = title_after
            chap_num, chap_title = None, ""
        elif m.level == "chapter":
            chap_num = m.number
            chap_title = title_after
        elif m.level == "article":
            if m.number is None:
                continue
            body = full_text[m.end:next_start].strip(" :-–،.()")
            articles.append(ParsedArticle(
                article_number=m.number,
                body_text=body,
                book_number=book_num,
                book_title=book_title,
                part_number=part_num,
                part_title=part_title,
                chapter_number=chap_num,
                chapter_title=chap_title,
                char_start=m.start,
                char_end=next_start,
            ))

    return articles


def _level_priority(level: str) -> int:
    return {"book": 0, "part": 1, "chapter": 2, "article": 3}[level]
