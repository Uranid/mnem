"""FinanceBench closed-haystack retrieval eval for MemPalace with bge-large.

Identical eval to financebench_mempalace.py but uses Ollama bge-large
(1024-dim) instead of MemPalace's default FastEmbed bge-small (384-dim).
This allows an embedding-controlled comparison:

  financebench_mempalace.py     , MemPalace default (bge-small, 384-dim)
  financebench_mempalace_bgelarge.py, same ChromaDB store, bge-large (1024-dim)

MemPalace hardcodes its embedding factory to ONNX MiniLM.  To swap the
model we bypass that factory and create the ChromaDB collection directly
using chromadb's OllamaEmbeddingFunction.  The metadata layout and
source_file encoding are identical to the bge-small run so results are
directly comparable.

Retrieval uses a pure vector cosine-similarity query (no BM25 rerank)
because BM25 operates on MemPalace's internal closets collection, which
we don't populate here.  mnem also uses pure vector retrieval as its
baseline, so this is a fair apples-to-apples comparison.

mnem reference score: hit@5 = 0.9733
  Embedder: bge-large (Ollama, 1024-dim)
  No LLM extraction; raw passage text stored verbatim.

Requirements
-----------
    pip install chromadb
    ollama pull bge-large   (Ollama must be running)

Usage
-----
    python benchmarks/harness/adapters/financebench_mempalace_bgelarge.py \\
        --dataset datasets/financebench/financebench_open_source.jsonl \\
        --out results/financebench-mempalace-bgelarge.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
import tempfile
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--embed-model", default="bge-large",
                    help="Ollama embedding model (default: bge-large)")
    ap.add_argument("--ollama-base", default="http://localhost:11434")
    ap.add_argument("--storage-path", default=None,
                    help="ChromaDB directory (default: temp dir)")
    return ap.parse_args()


def _key_to_source_file(doc_name: str, page_num: int) -> str:
    return f"{doc_name}__p{page_num}"


def _source_file_to_key(source_file: str) -> tuple[str, int] | None:
    m = re.match(r"^(.+)__p(\d+)$", source_file)
    if m:
        return (m.group(1), int(m.group(2)))
    return None


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    try:
        import chromadb
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    except ImportError:
        print("chromadb not installed.  Run: pip install chromadb", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    with args.dataset.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]
    print(f"loaded {len(records)} questions", file=sys.stderr)

    tmp_dir: str | None = None
    storage_path = args.storage_path
    if not storage_path:
        tmp_dir = tempfile.mkdtemp(prefix="mempalace_bgelarge_financebench_")
        storage_path = tmp_dir
    print(f"ChromaDB storage: {storage_path}", file=sys.stderr)
    print(
        f"Embedding: Ollama/{args.embed_model} (1024-dim) via {args.ollama_base}",
        file=sys.stderr,
    )

    ef = OllamaEmbeddingFunction(
        model_name=args.embed_model,
        url=f"{args.ollama_base}/api/embeddings",
    )

    t0 = time.time()
    try:
        client = chromadb.PersistentClient(path=storage_path)
        col = client.get_or_create_collection(
            name="financebench_bgelarge",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        # Build corpus
        ev_key_to_text: dict[tuple[str, int], str] = {}
        for rec in records:
            for ev in rec.get("evidence") or []:
                key = (ev["doc_name"], int(ev["evidence_page_num"]))
                if key not in ev_key_to_text:
                    ev_key_to_text[key] = (ev.get("evidence_text") or "").strip()

        ev_keys_ordered = list(ev_key_to_text.keys())
        print(
            f"ingesting {len(ev_keys_ordered)} passages into ChromaDB "
            f"(Ollama/{args.embed_model}, no LLM extraction)...",
            file=sys.stderr,
        )

        for i, key in enumerate(ev_keys_ordered):
            text = ev_key_to_text[key]
            source_file = _key_to_source_file(key[0], key[1])
            doc_id = f"fb__{source_file}"
            try:
                col.upsert(
                    documents=[text],
                    ids=[doc_id],
                    metadatas=[{"source_file": source_file}],
                )
            except Exception as exc:
                print(f"  [warn] upsert failed for {key}: {exc}", file=sys.stderr)
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(
                    f"  stored {i+1}/{len(ev_keys_ordered)} ({elapsed:.0f}s)...",
                    file=sys.stderr,
                )
        t_ingest = time.time() - t0
        print(f"ingest done in {t_ingest:.1f}s", file=sys.stderr)

        # Retrieve + score
        try:
            from tqdm import tqdm  # type: ignore
        except ImportError:
            tqdm = lambda x, **kw: x  # type: ignore

        hits1 = hits3 = hits5 = 0
        per_q: list[dict[str, Any]] = []
        t_retrieve_total = 0.0
        n_results = max(50, args.top_k * 10)

        for rec in tqdm(records, desc="questions"):
            qid = rec["financebench_id"]
            question = rec["question"]
            target_keys: set[tuple[str, int]] = {
                (ev["doc_name"], int(ev["evidence_page_num"]))
                for ev in (rec.get("evidence") or [])
                if "doc_name" in ev and "evidence_page_num" in ev
            }

            ts = time.time()
            try:
                results = col.query(
                    query_texts=[question],
                    n_results=min(n_results, col.count()),
                    include=["metadatas", "distances"],
                )
                metas = (results.get("metadatas") or [[]])[0]
            except Exception as exc:
                print(f"  [warn] search failed for {qid}: {exc}", file=sys.stderr)
                metas = []
            t_retrieve_total += time.time() - ts

            ranked_keys: list[tuple[str, int]] = []
            seen: set[tuple[str, int]] = set()
            for meta in metas:
                src = (meta or {}).get("source_file") or ""
                key = _source_file_to_key(src)
                if key and key not in seen:
                    ranked_keys.append(key)
                    seen.add(key)

            h1 = bool(target_keys) and any(k in target_keys for k in ranked_keys[:1])
            h3 = bool(target_keys) and any(k in target_keys for k in ranked_keys[:3])
            h5 = bool(target_keys) and any(k in target_keys for k in ranked_keys[:args.top_k])
            if h1: hits1 += 1
            if h3: hits3 += 1
            if h5: hits5 += 1

            per_q.append({
                "financebench_id": qid,
                "hit@1": h1,
                "hit@3": h3,
                f"hit@{args.top_k}": h5,
                "target": [list(k) for k in target_keys],
                "top5_retrieved": [list(k) for k in ranked_keys[:5]],
            })

    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    n = len(records)
    r1, r3, r5 = hits1 / n, hits3 / n, hits5 / n

    report: dict[str, Any] = {
        "harness": "mempalace-bgelarge-financebench",
        "system": "MemPalace-bge-large",
        "embed_model": f"Ollama/{args.embed_model} (1024-dim)",
        "note": (
            "ChromaDB store with Ollama bge-large (1024-dim), bypassing MemPalace's "
            "default ONNX MiniLM embedding. Raw passage text stored verbatim. "
            "Pure vector cosine-similarity retrieval (no BM25 rerank). "
            "mnem reference: hit@5=0.9733 (bge-large + hybrid-boost + doc-filter). "
            "MemPalace bge-small default: hit@5=0.6267."
        ),
        "corpus_size": len(ev_keys_ordered),
        "n_questions": n,
        "runtime_seconds": round(time.time() - t0, 2),
        "timing": {"retrieve_s": round(t_retrieve_total, 3)},
        "overall": {
            "hit@1": round(r1, 4),
            "hit@3": round(r3, 4),
            f"hit@{args.top_k}": round(r5, 4),
            "n": n,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    jsonl_out = args.out.with_suffix(".jsonl")
    jsonl_out.write_text(
        "\n".join(json.dumps(r) for r in per_q) + "\n", encoding="utf-8"
    )

    print(
        f"\n=== MemPalace-bge-large FinanceBench  hit@1={r1:.4f}  hit@3={r3:.4f}  "
        f"hit@{args.top_k}={r5:.4f}  n={n}  corpus={len(ev_keys_ordered)}",
        file=sys.stderr,
    )
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
