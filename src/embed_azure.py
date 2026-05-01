"""Azure OpenAI embedding for chunks.

Reads chunks.jsonl, calls Azure OpenAI text-embedding-3-large in batches,
writes chunks_embedded.jsonl with the `embedding` field populated.

Usage:
    python src/embed_azure.py --in data/chunks.jsonl --out data/chunks_embedded.jsonl

Env vars (load from .env or shell):
    AZURE_OPENAI_ENDPOINT     e.g. https://my-resource.openai.azure.com
    AZURE_OPENAI_KEY          API key
    AZURE_OPENAI_API_VERSION  e.g. 2024-10-21
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT   your deployment name for text-embedding-3-large
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Lazy import openai to keep the rest of the pipeline runnable without it.
try:
    from openai import AzureOpenAI, RateLimitError
except ImportError:  # pragma: no cover
    AzureOpenAI = None  # type: ignore
    RateLimitError = Exception  # type: ignore

BATCH_SIZE = 16
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0


def make_client() -> "AzureOpenAI":
    if AzureOpenAI is None:
        raise RuntimeError(
            "openai package not installed. `pip install openai python-dotenv`"
        )
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_KEY"]
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def embed_batch(client, deployment: str, texts: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.embeddings.create(input=texts, model=deployment)
            return [d.embedding for d in resp.data]
        except RateLimitError:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            time.sleep(delay)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    raise RuntimeError(f"Failed to embed batch after {MAX_RETRIES} retries")


def embed_chunks(
    in_path: Path,
    out_path: Path,
    deployment: str | None = None,
) -> None:
    deployment = deployment or os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    client = make_client()

    chunks = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    embedded = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [c["text_normalized"] for c in batch]
            vectors = embed_batch(client, deployment, texts)
            for c, v in zip(batch, vectors):
                c["embedding"] = v
                out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
            embedded += len(batch)
            print(f"  embedded {embedded}/{len(chunks)}")

    print(f"OK: wrote {out_path} with {embedded} embedded chunks")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--deployment", default=None,
                    help="Azure deployment name; default from AZURE_OPENAI_EMBEDDING_DEPLOYMENT env")
    args = ap.parse_args()
    embed_chunks(args.in_path, args.out_path, args.deployment)


if __name__ == "__main__":
    main()
