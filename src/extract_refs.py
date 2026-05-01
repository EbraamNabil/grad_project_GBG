"""Extract cross-references, external-law references, and defined-term usage
from each chunk's normalized text.

Cross-refs become (:Article)-[:REFERENCES]->(:Article) edges in Neo4j.
External refs become (:Article)-[:EXTERNAL_REF {law: "..."}]->() edges.
Defined-term usage becomes (:Article)-[:USES_TERM]->(:Definition) edges.
"""
from __future__ import annotations

import regex as re

from chunk import Chunk
from normalize import pdf_arabic_int, stem_arabic, strip_diacritics, to_western_digits

# Cross-reference patterns. Operate on the de-diacriticized display form so we see
# Eastern Arabic digits as in the source (with the digit-reversal heuristic).
# The PDF often inserts a stray ي after "المادة"/"المواد" (residual from PUA chars)
# and uses either bracket direction "(" or ")", so accept both.
ARTICLE_REF_RX = re.compile(
    r"(?<![اوفبلكم])(?:ال|لل|بال)?(?:مادة|مادتين|مادتى|مادتي)[يى]?\s*[\)\(]?\s*([\d٠-٩۰-۹]{1,4})\s*[\)\(]?"
)
ARTICLES_LIST_REF_RX = re.compile(
    r"(?<![اوفبلكم])(?:ال|لل|بال)?مواد[يى]?\s*[\)\(]\s*([\d٠-٩۰-۹\s،,]+?)\s*[\)\(]\s*(?:من|،)?"
)
EXTERNAL_LAW_RX = re.compile(
    r"قانون[^،,.\n]{1,80}رقم\s*([\d٠-٩۰-۹]+)\s*لسنة\s*([\d٠-٩۰-۹]{4})"
)


def extract_cross_refs(
    text: str,
    max_article: int,
    exclude: int | None = None,
) -> list[int]:
    """Return article numbers referenced in `text`, excluding `exclude` (self-refs).

    Strips diacritics/tatweel before matching so "المـواد" matches "المواد",
    but keeps Eastern digits so the digit-reversal heuristic can be applied.
    """
    text = strip_diacritics(text)
    refs: set[int] = set()
    for m in ARTICLE_REF_RX.finditer(text):
        n = pdf_arabic_int(m.group(1))
        if n is not None and 1 <= n <= max_article:
            refs.add(n)
    for m in ARTICLES_LIST_REF_RX.finditer(text):
        for tok in re.split(r"[\s،,]+", m.group(1)):
            n = pdf_arabic_int(tok)
            if n is not None and 1 <= n <= max_article:
                refs.add(n)
    if exclude is not None:
        refs.discard(exclude)
    return sorted(refs)


def extract_external_refs(text: str) -> list[str]:
    """Return free-text descriptions of external-law references."""
    out: list[str] = []
    for m in EXTERNAL_LAW_RX.finditer(text):
        snippet = re.sub(r"\s+", " ", m.group(0)).strip()
        if snippet not in out:
            out.append(snippet)
    return out


def extract_defined_terms_used(text: str, term_variants: dict[str, list[str]]) -> list[str]:
    """Return the canonical defined terms whose variants appear in `text`.

    `term_variants` maps canonical term → list of surface variants to match.
    Order of canonical terms in the output is sorted for stability.
    """
    found: set[str] = set()
    tokens = set(re.findall(r"[ء-ي]+", text))
    stems = {stem_arabic(t) for t in tokens}
    for canonical, variants in term_variants.items():
        for v in variants:
            v_stem = stem_arabic(v)
            if v in tokens or v_stem in stems:
                found.add(canonical)
                break
    return sorted(found)


def annotate_chunks(
    chunks: list[Chunk],
    glossary: dict[str, str],
    max_article: int,
) -> None:
    """Mutate chunks in place: fill cross_refs, external_refs, defined_terms_used."""
    term_variants = {term: _build_variants(term) for term in glossary.keys()}
    for c in chunks:
        own_article = c.article_number
        if c.is_definition:
            c.cross_refs = extract_cross_refs(c.text, max_article, exclude=own_article)
            c.external_refs = extract_external_refs(c.text)
            c.defined_terms_used = []
        else:
            c.cross_refs = extract_cross_refs(c.text, max_article, exclude=own_article)
            c.external_refs = extract_external_refs(c.text)
            c.defined_terms_used = extract_defined_terms_used(c.text, term_variants)


def _build_variants(canonical: str) -> list[str]:
    variants = {canonical}
    if canonical.startswith("ال"):
        bare = canonical[2:]
        variants.add(bare)
        for prefix in ("لل", "بال", "وال", "فال", "كال", "بـال"):
            variants.add(prefix + bare)
    return sorted(variants)
