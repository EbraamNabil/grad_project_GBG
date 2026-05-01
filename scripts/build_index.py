"""Orchestrator: PDF -> chunks.jsonl -> (embedded) -> Neo4j.

Stages can be run independently with --stage.

Usage:
    python scripts/build_index.py --stage chunk      # local, no creds needed
    python scripts/build_index.py --stage embed      # needs Azure OpenAI creds
    python scripts/build_index.py --stage load       # needs Neo4j creds
    python scripts/build_index.py --stage all        # runs all three
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from extract import concatenate_pages, extract_pdf  # noqa: E402
from extract_refs import annotate_chunks            # noqa: E402
from parse_structure import parse                   # noqa: E402
from chunk import build_chunks                      # noqa: E402

PDF_PATH = ROOT / "data" / "source" / "law-14-2025.pdf"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
GLOSSARY_PATH = ROOT / "data" / "glossary.json"
EMBEDDED_PATH = ROOT / "data" / "chunks_embedded.jsonl"


def stage_chunk() -> None:
    print(f"[chunk] Reading PDF: {PDF_PATH}")
    pages = extract_pdf(PDF_PATH)
    print(f"[chunk] Extracted {len(pages)} pages")

    full, offsets = concatenate_pages(pages)
    print(f"[chunk] Total flat text: {len(full)} chars")

    articles = parse(full)
    print(f"[chunk] Parsed {len(articles)} articles")

    chunks, glossary = build_chunks(articles, offsets)
    print(f"[chunk] Built {len(chunks)} chunks ({sum(1 for c in chunks if c.is_definition)} definitions)")
    print(f"[chunk] Glossary: {len(glossary)} terms")

    max_article = max((a.article_number for a in articles), default=0)
    annotate_chunks(chunks, glossary, max_article)
    n_with_refs = sum(1 for c in chunks if c.cross_refs)
    n_with_terms = sum(1 for c in chunks if c.defined_terms_used)
    print(f"[chunk] Annotated: {n_with_refs} chunks have cross_refs, {n_with_terms} use defined terms")

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"[chunk] Wrote {CHUNKS_PATH}")

    GLOSSARY_PATH.write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[chunk] Wrote {GLOSSARY_PATH}")


def stage_embed() -> None:
    from embed_azure import embed_chunks
    print(f"[embed] Embedding {CHUNKS_PATH} -> {EMBEDDED_PATH}")
    embed_chunks(CHUNKS_PATH, EMBEDDED_PATH)


def stage_load() -> None:
    from load_neo4j import load
    in_path = EMBEDDED_PATH if EMBEDDED_PATH.exists() else CHUNKS_PATH
    print(f"[load] Loading {in_path} into Neo4j")
    load(in_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["chunk", "embed", "load", "all"], default="chunk")
    args = ap.parse_args()

    if args.stage in ("chunk", "all"):
        stage_chunk()
    if args.stage in ("embed", "all"):
        stage_embed()
    if args.stage in ("load", "all"):
        stage_load()


if __name__ == "__main__":
    main()
