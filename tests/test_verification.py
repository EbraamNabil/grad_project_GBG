"""Verification checks on chunks.jsonl. Run after `build_index.py --stage chunk`.

Usage:
    python tests/test_verification.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
GLOSSARY_PATH = ROOT / "data" / "glossary.json"

EXPECTED_TOTAL_ARTICLES = 298
EXPECTED_DEFINITIONS = 38


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_article_gaps(chunks: list[dict]) -> tuple[bool, str]:
    nums = sorted({c["article_number"] for c in chunks if c["node_type"] == "Article"})
    if not nums:
        return False, "No article chunks found"
    expected = set(range(1, max(nums) + 1))
    missing = sorted(expected - set(nums))
    if missing:
        return False, f"Missing article numbers (first 10): {missing[:10]} (of {len(missing)})"
    counter = Counter(c["article_number"] for c in chunks if c["node_type"] == "Article")
    dups = [(n, c) for n, c in counter.items() if c > 1]
    if dups:
        return False, f"Duplicate article numbers: {dups[:10]}"
    return True, f"Articles 1..{max(nums)}, no gaps, no duplicates ({len(nums)} total)"


def check_definitions_count(chunks: list[dict]) -> tuple[bool, str]:
    defs = [c for c in chunks if c["node_type"] == "Definition"]
    art1 = [c for c in chunks if c["node_type"] == "Article" and c["article_number"] == 1]
    msg = f"Definitions: {len(defs)} (expected ~{EXPECTED_DEFINITIONS}), Article-1 wholes: {len(art1)}"
    ok = len(defs) >= 30 and len(art1) == 1
    return ok, msg


def check_total_articles(chunks: list[dict]) -> tuple[bool, str]:
    nums = {c["article_number"] for c in chunks if c["node_type"] == "Article"}
    msg = f"Distinct article numbers: {len(nums)} (expected {EXPECTED_TOTAL_ARTICLES})"
    return len(nums) == EXPECTED_TOTAL_ARTICLES, msg


def check_page_monotonicity(chunks: list[dict]) -> tuple[bool, str]:
    art = sorted(
        [c for c in chunks if c["node_type"] == "Article"],
        key=lambda c: c["article_number"],
    )
    last_max = 0
    bad = 0
    for c in art:
        pages = c.get("source_pages") or []
        if pages and pages[0] < last_max - 1:
            bad += 1
        if pages:
            last_max = max(pages)
    return bad == 0, f"Page-monotonicity violations: {bad} of {len(art)} articles"


def check_token_distribution(chunks: list[dict]) -> tuple[bool, str]:
    toks = [c["token_count_est"] for c in chunks if c["node_type"] in {"Article", "ArticleSegment"}]
    p50 = int(statistics.median(toks))
    p90 = sorted(toks)[int(len(toks) * 0.9)]
    p99 = sorted(toks)[int(len(toks) * 0.99)]
    msg = f"Token est: p50={p50}, p90={p90}, p99={p99}, max={max(toks)}"
    ok = 100 < p50 < 800 and p99 < 3500
    return ok, msg


def check_dangling_cross_refs(chunks: list[dict]) -> tuple[bool, str]:
    article_nums = {c["article_number"] for c in chunks if c["node_type"] == "Article"}
    dangling: list[tuple[str, int]] = []
    for c in chunks:
        for ref in c.get("cross_refs", []):
            if ref not in article_nums:
                dangling.append((c["node_id"], ref))
    msg = f"Dangling cross_refs: {len(dangling)} (first 5: {dangling[:5]})"
    return len(dangling) == 0, msg


def check_glossary_consistency(chunks: list[dict]) -> tuple[bool, str]:
    glossary = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    def_ids = {c["node_id"] for c in chunks if c["node_type"] == "Definition"}
    missing = [t for t, nid in glossary.items() if nid not in def_ids]
    msg = f"Glossary entries: {len(glossary)}, missing definition nodes: {len(missing)}"
    return not missing, msg


def main() -> int:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    checks = [
        ("article gaps",          check_article_gaps),
        ("article count",         check_total_articles),
        ("definitions count",     check_definitions_count),
        ("page monotonicity",     check_page_monotonicity),
        ("token distribution",    check_token_distribution),
        ("dangling cross-refs",   check_dangling_cross_refs),
        ("glossary consistency",  check_glossary_consistency),
    ]

    failed = 0
    for name, fn in checks:
        ok, msg = fn(chunks)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:30s} {msg}")
        if not ok:
            failed += 1

    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
