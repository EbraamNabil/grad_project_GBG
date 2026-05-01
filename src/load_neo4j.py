"""Load chunks into Neo4j with vector indexes for GraphRAG.

Schema:
    (:Law)-[:HAS_BOOK]->(:Book)-[:HAS_PART]->(:Part)
    (:Part)-[:HAS_CHAPTER]->(:Chapter)?  -- optional
    (:Chapter|:Part)-[:HAS_ARTICLE]->(:Article)
    (:Article)-[:NEXT]->(:Article)
    (:Article)-[:HAS_SEGMENT {index, total}]->(:ArticleSegment)
    (:Article|:ArticleSegment)-[:REFERENCES]->(:Article)
    (:Article|:ArticleSegment)-[:USES_TERM]->(:Definition)

Vector indexes on :Article, :ArticleSegment, :Definition over `embedding`
(3072-dim, cosine, matching Azure text-embedding-3-large).

Usage:
    python src/load_neo4j.py --in data/chunks_embedded.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None  # type: ignore


def driver():
    if GraphDatabase is None:
        raise RuntimeError("neo4j package not installed. `pip install neo4j`")
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    pw = os.environ["NEO4J_PASSWORD"]
    # Suppress vector-procedure deprecation notifications. Neo4j 2026+ prefers SEARCH
    # syntax, but db.index.vector.queryNodes still works and the warning floods logs.
    return GraphDatabase.driver(uri, auth=(user, pw), notifications_min_severity="OFF")


def load(in_path: Path, embedding_dim: int | None = None) -> None:
    with driver() as drv, drv.session() as session:
        chunks = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        if embedding_dim is None:
            embedding_dim = _detect_embedding_dim(chunks)
            print(f"  [load] auto-detected embedding dim: {embedding_dim}")

        _ensure_constraints(session)
        _create_law_node(session, chunks[0])
        _create_hierarchy(session, chunks)
        _create_article_nodes(session, chunks)
        _create_segment_nodes(session, chunks)
        _create_definition_nodes(session, chunks)
        _link_hierarchy(session, chunks)
        _link_next_article(session, chunks)
        _link_segments_to_parents(session, chunks)
        _link_cross_refs(session, chunks)
        _link_external_refs(session, chunks)
        _link_term_usage(session, chunks)
        _create_vector_indexes(session, embedding_dim)

    print(f"OK: loaded {len(chunks)} chunks into Neo4j")


def _detect_embedding_dim(chunks: list[dict]) -> int:
    for c in chunks:
        emb = c.get("embedding")
        if isinstance(emb, list) and emb:
            return len(emb)
    raise RuntimeError(
        "No chunk has an embedding. Run `python scripts/build_index.py --stage embed` first."
    )


def _ensure_constraints(session) -> None:
    queries = [
        "CREATE CONSTRAINT law_id IF NOT EXISTS FOR (l:Law) REQUIRE l.id IS UNIQUE",
        "CREATE CONSTRAINT book_id IF NOT EXISTS FOR (b:Book) REQUIRE (b.law_id, b.number) IS UNIQUE",
        "CREATE CONSTRAINT part_id IF NOT EXISTS FOR (p:Part) REQUIRE (p.law_id, p.book_number, p.number) IS UNIQUE",
        "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE (c.law_id, c.book_number, c.part_number, c.number) IS UNIQUE",
        "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.node_id IS UNIQUE",
        "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:ArticleSegment) REQUIRE s.node_id IS UNIQUE",
        "CREATE CONSTRAINT definition_id IF NOT EXISTS FOR (d:Definition) REQUIRE d.node_id IS UNIQUE",
    ]
    for q in queries:
        session.run(q)


def _create_law_node(session, sample_chunk: dict) -> None:
    session.run(
        "MERGE (l:Law {id: $id}) "
        "SET l.title = $title, l.issued_date = $issued_date, l.gazette_issue = $gazette_issue",
        id=sample_chunk["law_id"],
        title=sample_chunk["law_title"],
        issued_date="2025-05-03",
        gazette_issue="81 (تابع)",
    )


def _create_hierarchy(session, chunks: list[dict]) -> None:
    seen_books, seen_parts, seen_chapters = set(), set(), set()
    for c in chunks:
        law_id = c["law_id"]
        if c["book_number"] is not None and (law_id, c["book_number"]) not in seen_books:
            session.run(
                "MERGE (b:Book {law_id: $law, number: $n}) SET b.title = $t",
                law=law_id, n=c["book_number"], t=c["book_title"],
            )
            seen_books.add((law_id, c["book_number"]))
        if c["part_number"] is not None:
            key = (law_id, c["book_number"], c["part_number"])
            if key not in seen_parts:
                session.run(
                    "MERGE (p:Part {law_id: $law, book_number: $b, number: $n}) "
                    "SET p.title = $t",
                    law=law_id, b=c["book_number"], n=c["part_number"], t=c["part_title"],
                )
                seen_parts.add(key)
        if c["chapter_number"] is not None:
            key = (law_id, c["book_number"], c["part_number"], c["chapter_number"])
            if key not in seen_chapters:
                session.run(
                    "MERGE (ch:Chapter {law_id: $law, book_number: $b, "
                    "part_number: $p, number: $n}) SET ch.title = $t",
                    law=law_id, b=c["book_number"], p=c["part_number"],
                    n=c["chapter_number"], t=c["chapter_title"],
                )
                seen_chapters.add(key)


def _create_article_nodes(session, chunks: list[dict]) -> None:
    for c in chunks:
        if c["node_type"] != "Article":
            continue
        session.run(
            """
            MERGE (a:Article {node_id: $node_id})
            SET a.law_id = $law_id, a.number = $num, a.text = $text,
                a.text_normalized = $tn, a.char_count = $cc,
                a.token_count_est = $tok, a.source_pages = $pages,
                a.embedding = $emb
            """,
            node_id=c["node_id"], law_id=c["law_id"], num=c["article_number"],
            text=c["text"], tn=c["text_normalized"], cc=c["char_count"],
            tok=c["token_count_est"], pages=c["source_pages"],
            emb=c.get("embedding"),
        )


def _create_segment_nodes(session, chunks: list[dict]) -> None:
    for c in chunks:
        if c["node_type"] != "ArticleSegment":
            continue
        session.run(
            """
            MERGE (s:ArticleSegment {node_id: $node_id})
            SET s.law_id = $law_id, s.parent_article = $pa, s.segment_index = $si,
                s.segment_total = $st, s.text = $text, s.text_normalized = $tn,
                s.char_count = $cc, s.token_count_est = $tok,
                s.source_pages = $pages, s.embedding = $emb
            """,
            node_id=c["node_id"], law_id=c["law_id"], pa=c["article_number"],
            si=c["segment_index"], st=c["segment_total"], text=c["text"],
            tn=c["text_normalized"], cc=c["char_count"],
            tok=c["token_count_est"], pages=c["source_pages"],
            emb=c.get("embedding"),
        )


def _create_definition_nodes(session, chunks: list[dict]) -> None:
    for c in chunks:
        if c["node_type"] != "Definition":
            continue
        session.run(
            """
            MERGE (d:Definition {node_id: $node_id})
            SET d.law_id = $law_id, d.term = $term, d.term_variants = $variants,
                d.definition_text = $text, d.text_normalized = $tn,
                d.char_count = $cc, d.token_count_est = $tok,
                d.source_pages = $pages, d.embedding = $emb
            """,
            node_id=c["node_id"], law_id=c["law_id"], term=c["defined_term"],
            variants=c["term_variants"], text=c["text"],
            tn=c["text_normalized"], cc=c["char_count"],
            tok=c["token_count_est"], pages=c["source_pages"],
            emb=c.get("embedding"),
        )


def _link_hierarchy(session, chunks: list[dict]) -> None:
    seen_book_part = set()
    seen_part_chapter = set()
    for c in chunks:
        if c["book_number"] is not None:
            session.run(
                "MATCH (l:Law {id: $law}), (b:Book {law_id: $law, number: $n}) "
                "MERGE (l)-[:HAS_BOOK]->(b)",
                law=c["law_id"], n=c["book_number"],
            )
        if c["part_number"] is not None:
            key = (c["law_id"], c["book_number"], c["part_number"])
            if key not in seen_book_part:
                session.run(
                    "MATCH (b:Book {law_id: $law, number: $bn}), "
                    "(p:Part {law_id: $law, book_number: $bn, number: $pn}) "
                    "MERGE (b)-[:HAS_PART]->(p)",
                    law=c["law_id"], bn=c["book_number"], pn=c["part_number"],
                )
                seen_book_part.add(key)
        if c["chapter_number"] is not None:
            key = (c["law_id"], c["book_number"], c["part_number"], c["chapter_number"])
            if key not in seen_part_chapter:
                session.run(
                    "MATCH (p:Part {law_id: $law, book_number: $bn, number: $pn}), "
                    "(ch:Chapter {law_id: $law, book_number: $bn, part_number: $pn, number: $cn}) "
                    "MERGE (p)-[:HAS_CHAPTER]->(ch)",
                    law=c["law_id"], bn=c["book_number"], pn=c["part_number"], cn=c["chapter_number"],
                )
                seen_part_chapter.add(key)
        if c["node_type"] == "Article":
            if c["chapter_number"] is not None:
                session.run(
                    "MATCH (ch:Chapter {law_id: $law, book_number: $bn, "
                    "part_number: $pn, number: $cn}), (a:Article {node_id: $aid}) "
                    "MERGE (ch)-[:HAS_ARTICLE]->(a)",
                    law=c["law_id"], bn=c["book_number"], pn=c["part_number"],
                    cn=c["chapter_number"], aid=c["node_id"],
                )
            elif c["part_number"] is not None:
                session.run(
                    "MATCH (p:Part {law_id: $law, book_number: $bn, number: $pn}), "
                    "(a:Article {node_id: $aid}) MERGE (p)-[:HAS_ARTICLE]->(a)",
                    law=c["law_id"], bn=c["book_number"], pn=c["part_number"], aid=c["node_id"],
                )


def _link_next_article(session, chunks: list[dict]) -> None:
    art_chunks = sorted(
        [c for c in chunks if c["node_type"] == "Article"],
        key=lambda c: c["article_number"],
    )
    for prev, nxt in zip(art_chunks, art_chunks[1:]):
        session.run(
            "MATCH (a:Article {node_id: $p}), (b:Article {node_id: $n}) "
            "MERGE (a)-[:NEXT]->(b)",
            p=prev["node_id"], n=nxt["node_id"],
        )


def _link_segments_to_parents(session, chunks: list[dict]) -> None:
    for c in chunks:
        if c["node_type"] != "ArticleSegment":
            continue
        parent_id = c["node_id"].rsplit("-seg", 1)[0]
        session.run(
            "MATCH (a:Article {node_id: $p}), (s:ArticleSegment {node_id: $s}) "
            "MERGE (a)-[r:HAS_SEGMENT]->(s) SET r.index = $i, r.total = $t",
            p=parent_id, s=c["node_id"],
            i=c["segment_index"], t=c["segment_total"],
        )


def _link_cross_refs(session, chunks: list[dict]) -> None:
    for c in chunks:
        if not c.get("cross_refs"):
            continue
        src_label = c["node_type"]
        for target_num in c["cross_refs"]:
            if target_num == c.get("article_number"):
                continue
            session.run(
                f"MATCH (s:{src_label} {{node_id: $sid}}), "
                "(t:Article {law_id: $law, number: $n}) "
                "MERGE (s)-[:REFERENCES]->(t)",
                sid=c["node_id"], law=c["law_id"], n=target_num,
            )


def _link_external_refs(session, chunks: list[dict]) -> None:
    for c in chunks:
        for ext in c.get("external_refs", []):
            src_label = c["node_type"]
            session.run(
                f"MERGE (e:ExternalLaw {{description: $d}}) "
                f"WITH e MATCH (s:{src_label} {{node_id: $sid}}) "
                "MERGE (s)-[:EXTERNAL_REF]->(e)",
                d=ext, sid=c["node_id"],
            )


def _link_term_usage(session, chunks: list[dict]) -> None:
    for c in chunks:
        if c["node_type"] == "Definition":
            continue
        for term in c.get("defined_terms_used", []):
            src_label = c["node_type"]
            session.run(
                f"MATCH (s:{src_label} {{node_id: $sid}}), "
                "(d:Definition {law_id: $law, term: $term}) "
                "MERGE (s)-[:USES_TERM]->(d)",
                sid=c["node_id"], law=c["law_id"], term=term,
            )


def _create_vector_indexes(session, dim: int) -> None:
    for label, name in [
        ("Article", "article_embedding"),
        ("ArticleSegment", "article_segment_embedding"),
        ("Definition", "definition_embedding"),
    ]:
        session.run(
            f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON n.embedding "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}",
            dim=dim,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--dim", type=int, default=None,
                    help="Embedding dimension; default auto-detected from first chunk")
    args = ap.parse_args()
    load(args.in_path, args.dim)


if __name__ == "__main__":
    main()
