"""GraphRAG retrieval + generation for Egyptian Labor Law.

Pipeline per query:
  0. Query routing — regex-detect explicit article references in the question
     ("المادة 48", "المواد 70 و 71 و 72", "المواد 70 إلى 75", etc.).
     If found, fetch those articles directly from Neo4j by number.
  1. Embed the question via Azure OpenAI (text-embedding-3-small/large).
  2. Vector-search Neo4j across :Article, :ArticleSegment, :Definition (top PRIMARY_K, merged).
  3. Graph-expand: for each :Article from steps 0 and 2, follow [:USES_TERM] (top 3)
     and [:REFERENCES] (top 2).
  4. Dedupe + cap at MAX_CONTEXT_CHUNKS, ordering: explicit → primary → expansions.
  5. Build Arabic system+user prompt; call GPT-4.1.
  6. Return RagResponse with answer, source chunks, detected refs, per-stage latencies.

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

import regex as re

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
from normalize import parse_arabic_int                    # noqa: E402

# ── Tunables ──────────────────────────────────────────────────────────────────
PRIMARY_K = 5            # vector-search top-K per index, merged then re-ranked by score
DEFINITIONS_PER_ARTICLE = 3   # graph-expansion budget for [:USES_TERM]
CROSSREFS_PER_ARTICLE = 2     # graph-expansion budget for [:REFERENCES]
MAX_CONTEXT_CHUNKS = 12  # hard cap on chunks sent to the LLM
LLM_TEMPERATURE = 0.0    # deterministic for legal text
LLM_MAX_TOKENS = 2048     # answer should be concise
TOTAL_ARTICLES = 298     # used to validate detected refs
RANGE_MAX_SPAN = 50      # safety cap on "المواد X إلى Y" expansion


@dataclass
class RetrievedChunk:
    node_id: str
    node_type: str  # "Article" | "ArticleSegment" | "Definition"
    article_number: int | None
    breadcrumb: str
    text: str           # display form (faithful Arabic)
    score: float        # 1.0 = explicit lookup, [0,1] cosine for vector, 0.0 for graph-expansion
    source: str         # "explicit" | "primary" | "definition" | "cross_ref"


@dataclass
class RagResponse:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    detected_refs: list[int] = field(default_factory=list)  # explicit article numbers from question
    elapsed_ms: dict[str, int] = field(default_factory=dict)


# ── Query routing ─────────────────────────────────────────────────────────────
# User-typed Arabic numerals are in LOGICAL order (not visually reversed like the PDF).
# So we use parse_arabic_int (no reversal), not pdf_arabic_int.

# Singular: "المادة 48" / "للمادة ٤٨" / "بالمادة (48)" / "المادتين 47 و 48"
SINGULAR_REF_RX = re.compile(
    r"(?<![اوفبلكم])(?:ال|لل|بال)?(?:مادة|مادتين|مادتي|مادتى)\s*\(?\s*([\d٠-٩۰-۹]+)\s*\)?"
    r"(?:\s*و\s*\(?\s*([\d٠-٩۰-۹]+)\s*\)?)?"
)

# Plural with possible list/range: "المواد 70 و 71 و 72" or "المواد 70 إلى 75"
PLURAL_REF_RX = re.compile(
    r"(?<![اوفبلكم])(?:ال|لل|بال)?مواد\s+([\d٠-٩۰-۹]+(?:\s*(?:و|إلى|الى|،|,|-)\s*[\d٠-٩۰-۹]+)+)"
)

RANGE_TOKEN_RX = re.compile(r"([\d٠-٩۰-۹]+)\s*(?:إلى|الى|-)\s*([\d٠-٩۰-۹]+)")
NUMBER_RX = re.compile(r"[\d٠-٩۰-۹]+")


def extract_article_refs_from_query(question: str) -> list[int]:
    """Detect explicit article numbers the user named in the question.

    Returns a sorted list of unique article numbers in [1, TOTAL_ARTICLES].
    """
    refs: set[int] = set()

    # Pattern 1 — singular and dual ("المادة X" / "المادتين X و Y")
    for m in SINGULAR_REF_RX.finditer(question):
        for grp in m.groups():
            if grp is None:
                continue
            n = parse_arabic_int(grp)
            if n and 1 <= n <= TOTAL_ARTICLES:
                refs.add(n)

    # Pattern 2 — plural with explicit list or range ("المواد 70 و 71" / "المواد 70 إلى 75")
    for m in PLURAL_REF_RX.finditer(question):
        tail = m.group(1)
        # Try to interpret as range first.
        rmatch = RANGE_TOKEN_RX.search(tail)
        if rmatch:
            start = parse_arabic_int(rmatch.group(1))
            end = parse_arabic_int(rmatch.group(2))
            if (start and end and 1 <= start <= TOTAL_ARTICLES
                    and 1 <= end <= TOTAL_ARTICLES
                    and abs(end - start) <= RANGE_MAX_SPAN):
                lo, hi = sorted((start, end))
                refs.update(range(lo, hi + 1))
                continue
        # Fall back to list parsing.
        for tok in NUMBER_RX.findall(tail):
            n = parse_arabic_int(tok)
            if n and 1 <= n <= TOTAL_ARTICLES:
                refs.add(n)

    return sorted(refs)


def _fetch_articles_by_numbers(session, numbers: list[int]) -> list[RetrievedChunk]:
    """Direct Cypher lookup by article number — no fuzziness, no LLM cost."""
    if not numbers:
        return []
    result = session.run(
        "MATCH (a:Article) WHERE a.number IN $nums "
        "RETURN a AS node, labels(a) AS labels, a.number AS num "
        "ORDER BY a.number",
        nums=numbers,
    )
    return [_node_to_chunk(r["node"], "Article", score=1.0, source="explicit")
            for r in result]


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_query(question: str) -> list[float]:
    client = make_embed_client()
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    resp = client.embeddings.create(input=[question], model=deployment)
    return resp.data[0].embedding


# ── Vector search ─────────────────────────────────────────────────────────────

VECTOR_INDEXES = [
    ("article_embedding", "Article"),
    ("article_segment_embedding", "ArticleSegment"),
    ("definition_embedding", "Definition"),
]


def _vector_search(session, embedding: list[float], k: int) -> list[RetrievedChunk]:
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


# ── Graph expansion ───────────────────────────────────────────────────────────

def _expand_via_uses_term(session, article_node_ids: list[str], per_article: int) -> list[RetrievedChunk]:
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


# ── Node → chunk + dedupe ─────────────────────────────────────────────────────

def _node_to_chunk(node, label: str, score: float, source: str) -> RetrievedChunk:
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
    """Keep first occurrence per node_id; ordering is the caller's responsibility."""
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
- إذا ذكر السؤال مادة محددة، ركّز إجابتك على نص تلك المادة المعروض في قسم [المواد المطلوبة صراحة].
- إذا كانت الإجابة غير موجودة في النصوص المرفقة، قل: "لا تتوفر إجابة قاطعة في النصوص المتاحة".
- لا تخترع أرقام مواد ولا تفترض محتوى غير مذكور.
- أجب بالعربية الفصحى، وبشكل موجز ومباشر."""


def _build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    explicit = [c for c in chunks if c.source == "explicit"]
    defs = [c for c in chunks if c.node_type == "Definition" and c.source != "explicit"]
    primary = [c for c in chunks if c.source == "primary" and c.node_type != "Definition"]
    refs = [c for c in chunks if c.source == "cross_ref"]

    parts: list[str] = [f"السؤال: {question}", "", "النصوص المرجعية:"]

    if explicit:
        parts.append("\n[المواد المطلوبة صراحة]")
        for c in explicit:
            parts.append(c.text)
            parts.append("---")

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
    client = make_embed_client()
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

    # Step 0 — query routing (regex, no API calls)
    t0 = time.perf_counter()
    detected_refs = extract_article_refs_from_query(question)
    elapsed["route"] = max(1, int((time.perf_counter() - t0) * 1000))

    # Step 1 — embed
    t0 = time.perf_counter()
    embedding = _embed_query(question)
    elapsed["embed"] = int((time.perf_counter() - t0) * 1000)

    # Step 2 — retrieve (explicit + vector) + graph expansion
    t0 = time.perf_counter()
    with make_neo4j_driver() as drv, drv.session() as session:
        explicit = _fetch_articles_by_numbers(session, detected_refs) if detected_refs else []
        primary = _vector_search(session, embedding, primary_k)

        # Graph-expand from BOTH explicit and primary articles
        seed_ids = list({
            c.node_id for c in (explicit + primary) if c.node_type == "Article"
        })
        defs = _expand_via_uses_term(session, seed_ids, DEFINITIONS_PER_ARTICLE)
        refs = _expand_via_references(session, seed_ids, CROSSREFS_PER_ARTICLE)
    elapsed["retrieve"] = int((time.perf_counter() - t0) * 1000)

    # Order: explicit (highest priority) → primary → definitions → cross-refs
    chunks = _dedupe_and_cap(explicit + primary + defs + refs, cap=MAX_CONTEXT_CHUNKS)

    # Step 3 — generate
    t0 = time.perf_counter()
    answer = _call_chat(question, chunks)
    elapsed["generate"] = int((time.perf_counter() - t0) * 1000)

    return RagResponse(
        question=question,
        answer=answer,
        chunks=chunks,
        detected_refs=detected_refs,
        elapsed_ms=elapsed,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless RAG query test")
    ap.add_argument("question", help="Arabic question about Egyptian Labor Law 14/2025")
    ap.add_argument("--k", type=int, default=PRIMARY_K)
    args = ap.parse_args()

    resp = answer_question(args.question, primary_k=args.k)
    print("=" * 70)
    print(f"السؤال: {resp.question}")
    if resp.detected_refs:
        print(f"المواد المكتشفة في السؤال: {resp.detected_refs}")
    print(f"\nزمن: route={resp.elapsed_ms.get('route', 0)}ms "
          f"embed={resp.elapsed_ms['embed']}ms "
          f"retrieve={resp.elapsed_ms['retrieve']}ms "
          f"generate={resp.elapsed_ms['generate']}ms")
    print(f"\nالإجابة:\n{resp.answer}\n")
    print("─" * 70)
    print(f"المصادر ({len(resp.chunks)} قطعة):")
    for c in resp.chunks:
        print(f"  [{c.source:9s}] score={c.score:.3f}  {c.node_id}  ({c.breadcrumb[:80]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
