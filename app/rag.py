"""GraphRAG retrieval + generation for Egyptian Labor Law.

Pipeline per query:
  1. Embed the question via Azure OpenAI text-embedding-3-large.
  2. Vector-search Neo4j across :Article, :ArticleSegment, :Definition (top PRIMARY_K each, merged).
  3. Graph-expand: for each primary :Article, follow [:USES_TERM] (top 3) and [:REFERENCES] (top 2).
  4. Dedupe + cap at MAX_CONTEXT_CHUNKS.
  5. Build Arabic system+user prompt; call GPT-4.1 (Azure deployment).
  6. Return RagResponse with answer, source chunks, and per-stage latencies.

The module is UI-agnostic. Run headless via:
    python -m app.rag "ما هي مدة الإجازة السنوية؟"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is importable so we can reuse src/embed_azure & src/load_neo4j.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from embed_azure import make_client as make_embed_client  # noqa: E402
from load_neo4j import driver as make_neo4j_driver       # noqa: E402

# ── Tunables ──────────────────────────────────────────────────────────────────
PRIMARY_K = 5            # vector-search top-K per index, merged then re-ranked by score
DEFINITIONS_PER_ARTICLE = 3   # graph-expansion budget for [:USES_TERM]
CROSSREFS_PER_ARTICLE = 2     # graph-expansion budget for [:REFERENCES]
MAX_CONTEXT_CHUNKS = 12  # hard cap on chunks sent to the LLM
LLM_TEMPERATURE = 0.0    # deterministic for legal text
LLM_MAX_TOKENS = 800     # answer should be concise


@dataclass
class RetrievedChunk:
    node_id: str
    node_type: str  # "Article" | "ArticleSegment" | "Definition"
    article_number: int | None
    breadcrumb: str
    text: str           # display form (faithful Arabic)
    score: float        # cosine similarity in [0, 1]; 0.0 for graph-expanded chunks
    source: str         # "primary" | "definition" | "cross_ref"


@dataclass
class RagResponse:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    elapsed_ms: dict[str, int] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _embed_query(question: str) -> list[float]:
    client = make_embed_client()
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    resp = client.embeddings.create(input=[question], model=deployment)
    return resp.data[0].embedding


VECTOR_INDEXES = [
    ("article_embedding", "Article"),
    ("article_segment_embedding", "ArticleSegment"),
    ("definition_embedding", "Definition"),
]


def _vector_search(session, embedding: list[float], k: int) -> list[RetrievedChunk]:
    """Run vector search across all three vector indexes; return top-k merged by score."""
    out: list[RetrievedChunk] = []
    for index_name, label in VECTOR_INDEXES:
        result = session.run(
            "CALL db.index.vector.queryNodes($idx, $k, $emb) YIELD node, score "
            "RETURN node, score, labels(node) AS labels",
            idx=index_name, k=k, emb=embedding,
        )
        for record in result:
            node = record["node"]
            out.append(_node_to_chunk(node, label, score=record["score"], source="primary"))

    out.sort(key=lambda c: c.score, reverse=True)
    return out[:k]


def _expand_via_uses_term(session, article_node_ids: list[str], per_article: int) -> list[RetrievedChunk]:
    """For each primary :Article, fetch up to `per_article` linked :Definition nodes."""
    if not article_node_ids:
        return []
    result = session.run(
        """
        UNWIND $ids AS aid
        MATCH (a:Article {node_id: aid})-[:USES_TERM]->(d:Definition)
        WITH aid, d LIMIT 100
        WITH aid, collect(d)[0..$per] AS defs
        UNWIND defs AS d
        RETURN DISTINCT d AS node, labels(d) AS labels
        """,
        ids=article_node_ids, per=per_article,
    )
    return [_node_to_chunk(r["node"], "Definition", score=0.0, source="definition")
            for r in result]


def _expand_via_references(session, article_node_ids: list[str], per_article: int) -> list[RetrievedChunk]:
    """For each primary :Article, fetch up to `per_article` referenced :Article nodes."""
    if not article_node_ids:
        return []
    result = session.run(
        """
        UNWIND $ids AS aid
        MATCH (a:Article {node_id: aid})-[:REFERENCES]->(target:Article)
        WITH aid, target LIMIT 100
        WITH aid, collect(target)[0..$per] AS refs
        UNWIND refs AS target
        RETURN DISTINCT target AS node, labels(target) AS labels
        """,
        ids=article_node_ids, per=per_article,
    )
    return [_node_to_chunk(r["node"], "Article", score=0.0, source="cross_ref")
            for r in result]


def _node_to_chunk(node, label: str, score: float, source: str) -> RetrievedChunk:
    """Convert a Neo4j node dict-like to a RetrievedChunk."""
    text = node.get("text") or node.get("definition_text", "")
    breadcrumb = text.split("\n\n", 1)[0] if "\n\n" in text else ""
    return RetrievedChunk(
        node_id=node.get("node_id", ""),
        node_type=label,
        article_number=node.get("number") or node.get("parent_article"),
        breadcrumb=breadcrumb,
        text=text,
        score=float(score),
        source=source,
    )


def _dedupe_and_cap(chunks: list[RetrievedChunk], cap: int) -> list[RetrievedChunk]:
    """Keep first occurrence per node_id (primary first, expansions after), cap at `cap`."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for c in chunks:
        if c.node_id in seen:
            continue
        seen.add(c.node_id)
        out.append(c)
        if len(out) >= cap:
            break
    return out


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في قانون العمل المصري رقم 14 لسنة 2025.
أجب فقط بناءً على النصوص المرفقة من القانون.
- اذكر رقم المادة في إجابتك (مثال: "وفقاً للمادة 47").
- إذا كانت الإجابة غير موجودة في النصوص المرفقة، قل: "لا تتوفر إجابة قاطعة في النصوص المتاحة".
- لا تخترع أرقام مواد ولا تفترض محتوى غير مذكور.
- أجب بالعربية الفصحى، وبشكل موجز ومباشر."""


def _build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    defs = [c for c in chunks if c.node_type == "Definition"]
    primary = [c for c in chunks if c.source == "primary" and c.node_type != "Definition"]
    refs = [c for c in chunks if c.source == "cross_ref"]

    parts: list[str] = [f"السؤال: {question}", "", "النصوص المرجعية:"]

    if defs:
        parts.append("\n[تعريفات]")
        for c in defs:
            parts.append(c.text)
            parts.append("---")

    if primary:
        parts.append("\n[المواد الأساسية]")
        for c in primary:
            parts.append(c.text)
            parts.append("---")

    if refs:
        parts.append("\n[المواد المُحال إليها]")
        for c in refs:
            parts.append(c.text)
            parts.append("---")

    return "\n".join(parts)


def _call_chat(question: str, chunks: list[RetrievedChunk]) -> str:
    client = make_embed_client()  # same Azure client works for chat too
    deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    user_msg = _build_user_message(question, chunks)
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


# ── Public entry point ────────────────────────────────────────────────────────

def answer_question(question: str, primary_k: int = PRIMARY_K) -> RagResponse:
    elapsed: dict[str, int] = {}

    t0 = time.perf_counter()
    embedding = _embed_query(question)
    elapsed["embed"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    with make_neo4j_driver() as drv, drv.session() as session:
        primary = _vector_search(session, embedding, primary_k)
        primary_article_ids = [c.node_id for c in primary if c.node_type == "Article"]
        defs = _expand_via_uses_term(session, primary_article_ids, DEFINITIONS_PER_ARTICLE)
        refs = _expand_via_references(session, primary_article_ids, CROSSREFS_PER_ARTICLE)
    elapsed["retrieve"] = int((time.perf_counter() - t0) * 1000)

    chunks = _dedupe_and_cap(primary + defs + refs, cap=MAX_CONTEXT_CHUNKS)

    t0 = time.perf_counter()
    answer = _call_chat(question, chunks)
    elapsed["generate"] = int((time.perf_counter() - t0) * 1000)

    return RagResponse(question=question, answer=answer, chunks=chunks, elapsed_ms=elapsed)


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless RAG query test")
    ap.add_argument("question", help="Arabic question about Egyptian Labor Law 14/2025")
    ap.add_argument("--k", type=int, default=PRIMARY_K)
    args = ap.parse_args()

    resp = answer_question(args.question, primary_k=args.k)
    print("=" * 70)
    print(f"السؤال: {resp.question}")
    print(f"\nزمن: embed={resp.elapsed_ms['embed']}ms retrieve={resp.elapsed_ms['retrieve']}ms "
          f"generate={resp.elapsed_ms['generate']}ms")
    print(f"\nالإجابة:\n{resp.answer}\n")
    print("─" * 70)
    print(f"المصادر ({len(resp.chunks)} قطعة):")
    for c in resp.chunks:
        print(f"  [{c.source:9s}] score={c.score:.3f}  {c.node_id}  ({c.breadcrumb[:80]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
