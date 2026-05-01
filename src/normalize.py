"""Text normalization for Egyptian Labor Law chunks.

Two text forms per chunk:
  - display: faithful Arabic, NFKC-normalized (Presentation Forms folded to base),
             Eastern numerals, breadcrumb prepended
  - normalized: also NFKC, plus diacritics/tatweel removed, Western numerals
                used for embedding and BM25

PDF digit storage quirks for this source:
  - Eastern Arabic-Indic digits (٠-٩, U+0660-U+0669): MULTI-DIGIT values are stored
    REVERSED in this PDF (e.g., "٤١" displayed → bytes are stored as ١٤ but appear as ٤١
    when extracted, meaning the actual value is 14, not 41). Always reverse multi-digit
    Eastern Arabic numerals to get the logical value.
  - Persian / extended Arabic-Indic digits (۰-۹, U+06F0-U+06F9): stored in correct
    logical order. Do NOT reverse.
"""
from __future__ import annotations

import unicodedata

import regex as re

# Tashkeel + tatweel ONLY. The naive range [ً-ٰ] (U+064B-U+0670) accidentally
# includes the Eastern Arabic-Indic digits U+0660-U+0669, so we list ranges explicitly.
ARABIC_DIACRITICS = re.compile(
    r"[ً-ٰٟۖ-ۭـ]"
)
EASTERN_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ALL_DIGITS_PATTERN = "0-9" + EASTERN_ARABIC_DIGITS + PERSIAN_DIGITS
EASTERN_DIGITS = str.maketrans(EASTERN_ARABIC_DIGITS + PERSIAN_DIGITS, "0123456789" * 2)
WESTERN_DIGITS = str.maketrans("0123456789", EASTERN_ARABIC_DIGITS)

# Gazette header repeats on every page. Source PDF mixes Arabic ya (ي) and
# Persian ya (ی), so [يی] in critical spots.
GAZETTE_HEADER = re.compile(
    r"الجر\s*\S{0,5}\s*[يی]دة\s+الرسم[يی]ة[\s\S]{1,250}?سنة\s*[\d٠-٩۰-۹]+",
)
PAGE_NUMBER_LINE = re.compile(r"(?m)^\s*[\d٠-٩۰-۹]{1,4}\s*$")
WHITESPACE_RUNS = re.compile(r"[ \t]+")
ALL_WHITESPACE_RUNS = re.compile(r"\s+")
BLANK_LINE_RUNS = re.compile(r"\n{2,}")


def to_western_digits(s: str) -> str:
    return s.translate(EASTERN_DIGITS)


def to_eastern_digits(s: str) -> str:
    return s.translate(WESTERN_DIGITS)


def strip_diacritics(s: str) -> str:
    return ARABIC_DIACRITICS.sub("", s)


def strip_page_artifacts(s: str) -> str:
    s = GAZETTE_HEADER.sub(" ", s)
    s = PAGE_NUMBER_LINE.sub("", s)
    return s


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


PUA_CHARS = re.compile(r"[-]")  # PyMuPDF/font rendering artifacts


def strip_pua(s: str) -> str:
    return PUA_CHARS.sub("", s)


def flatten(s: str) -> str:
    s = strip_pua(s)
    return ALL_WHITESPACE_RUNS.sub(" ", s).strip()


def normalize_for_display(s: str) -> str:
    s = nfkc(s)
    s = re.sub(r"ـ", "", s)  # tatweel; keep diacritics
    s = flatten(s)
    return s


def normalize_for_index(s: str) -> str:
    s = nfkc(s)
    s = strip_diacritics(s)
    s = to_western_digits(s)
    s = flatten(s)
    return s


def parse_arabic_int(token: str) -> int | None:
    t = to_western_digits(token).strip()
    return int(t) if t.isdigit() else None


def pdf_arabic_int(token: str) -> int | None:
    """Parse a digit token from this PDF, applying the digit-reversal heuristic.

    Single digits and Persian digits: parsed as-is.
    Multi-digit Eastern Arabic-Indic strings: reversed (PDF stores them visually).
    """
    if token is None:
        return None
    s = token.strip()
    if not s:
        return None
    if any(c not in EASTERN_ARABIC_DIGITS + PERSIAN_DIGITS + "0123456789" for c in s):
        return None
    if len(s) > 1 and all(c in EASTERN_ARABIC_DIGITS for c in s):
        s = s[::-1]
    return int(s.translate(EASTERN_DIGITS))


def estimate_tokens(s: str) -> int:
    return max(1, round(len(s) / 3.4))


ARABIC_PREFIXES = ("ال", "و", "ف", "ب", "ل", "ك")
ARABIC_SUFFIXES = ("ين", "ون", "ها", "هم", "كم", "ات", "ان", "ية", "ه", "ا", "ي")


def stem_arabic(word: str) -> str:
    """Light stemmer for closed-vocabulary glossary matching."""
    w = word
    for p in sorted(ARABIC_PREFIXES, key=len, reverse=True):
        if w.startswith(p) and len(w) - len(p) >= 2:
            w = w[len(p):]
            break
    for s in sorted(ARABIC_SUFFIXES, key=len, reverse=True):
        if w.endswith(s) and len(w) - len(s) >= 2:
            w = w[: -len(s)]
            break
    return w
