"""FinanceBench closed-haystack retrieval eval for mem0.

Runs the same 168-passage, 150-question eval as financebench.py but using
mem0 as the retrieval backend.

DESIGN DIFFERENCE vs mnem
--------------------------
mem0 is a conversational-memory system, not a document-retrieval system.
When you call memory.add(text), mem0 runs an LLM extraction step that
rewrites the stored content as structured "memories" (shorter, factual
summaries).  The original passage text is NOT stored verbatim.

This means:
  * You need a local LLM (via Ollama) in addition to the embedding model.
  * Retrieved items are LLM-extracted memories, not original passages.
  * Hit@5 is measured by matching extracted memories back to their source
    passage via the metadata dict we attach at ingest time.

mnem reference score (same corpus, same questions): hit@5 = 0.9733
  Embedder: bge-large (Ollama, 1024-dim)
  No LLM extraction step; raw passage text stored verbatim.

mem0 + BM25 result: hit@5 = 0.0333 (identical to baseline without BM25)
  BM25 reranking on LLM-extracted memory text produces no improvement.
  Root cause: numerical precision data (revenue, EPS) is lost during
  mem0's LLM extraction step, making keyword matching on extracted
  memories ineffective for financial Q&A.

API notes (mem0ai v2.0.1)
------------------------
v2 changed several call signatures vs v1:
  * memory.add(): no app_id; use user_id + run_id + metadata
  * memory.search(): no user_id / app_id top-level args; use
    filters={"user_id": "..."} and top_k= instead of limit=
  * search returns {"results": [...]} dict, not a plain list

Requirements
-----------
    pip install mem0ai chromadb ollama

Ollama must be running with BOTH:
  * An embedding model: ollama pull bge-large
  * A language model:   ollama pull llama3.2:1b  (or any chat model)

Usage
-----
    python benchmarks/harness/adapters/financebench_mem0.py \\
        --dataset datasets/financebench/financebench_open_source.jsonl \\
        --out results/financebench-mem0.json \\
        --llm-model llama3.2:1b \\
        --embed-model bge-large \\
        --ollama-base http://localhost:11434
"""
from __future__ import annotations

import argparse
import json
import math as _math
import pathlib
import re
import sys
import time
import uuid
from typing import Any

BENCH_USER = "financebench-harness"
BENCH_APP  = "financebench-eval"

_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


def _bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25+ scores for query against each document (Lucene-smoothed IDF)."""
    qtoks = _TOKEN_RE.findall(query.lower())
    if not qtoks:
        return [0.0] * len(documents)
    tokenized = [_TOKEN_RE.findall((d or "").lower()) for d in documents]
    N = len(tokenized)
    avgdl = sum(len(t) for t in tokenized) / max(N, 1)
    df: dict[str, int] = {}
    for toks in tokenized:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1
    scores = []
    for toks in tokenized:
        tf_map: dict[str, int] = {}
        for tok in toks:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        dl = len(toks)
        s = 0.0
        for tok in set(qtoks):
            tf = tf_map.get(tok, 0)
            idf = _math.log((N - df.get(tok, 0) + 0.5) / (df.get(tok, 0) + 0.5) + 1)
            s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
        scores.append(s)
    return scores


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm-model", default="llama3.2:3b",
                    help="Ollama LLM for mem0 memory extraction step")
    ap.add_argument("--embed-model", default="bge-large",
                    help="Ollama embedding model")
    ap.add_argument("--ollama-base", default="http://localhost:11434")
    ap.add_argument("--chroma-path", default="./chroma_mem0_financebench")
    return ap.parse_args()


def build_memory(args: argparse.Namespace) -> Any:
    try:
        from mem0 import Memory  # type: ignore
        from mem0.configs.base import MemoryConfig  # type: ignore
    except ImportError:
        print("mem0ai not installed.  Run: pip install mem0ai chromadb", file=sys.stderr)
        raise

    config = MemoryConfig(
        llm={
            "provider": "ollama",
            "config": {
                "model": args.llm_model,
                "ollama_base_url": args.ollama_base,
            },
        },
        embedder={
            "provider": "ollama",
            "config": {
                "model": args.embed_model,
                "ollama_base_url": args.ollama_base,
                "embedding_dims": 1024,
            },
        },
        vector_store={
            "provider": "chroma",
            "config": {
                "collection_name": "financebench_mem0",
                "path": args.chroma_path,
            },
        },
    )
    return Memory(config=config)


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    with args.dataset.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]
    print(f"loaded {len(records)} questions", file=sys.stderr)

    print("initialising mem0 (Ollama bge-large + LLM extraction)...", file=sys.stderr)
    print(
        f"  NOTE: mem0 runs {args.llm_model} to extract memories from each passage.\n"
        "  Stored content is NOT the original text, this is by design.\n"
        "  Hit@5 is measured against metadata-tracked source pages.",
        file=sys.stderr,
    )
    memory = build_memory(args)

    # Build corpus
    ev_key_to_text: dict[tuple[str, int], str] = {}
    for rec in records:
        for ev in rec.get("evidence") or []:
            key = (ev["doc_name"], int(ev["evidence_page_num"]))
            if key not in ev_key_to_text:
                ev_key_to_text[key] = (ev.get("evidence_text") or "").strip()

    ev_keys_ordered = list(ev_key_to_text.keys())
    # Stable IDs so we can map mem0 results back to (doc_name, page_num)
    key_to_mem_id: dict[tuple[str, int], str] = {
        k: f"{k[0]}__p{k[1]}" for k in ev_keys_ordered
    }
    mem_id_to_key: dict[str, tuple[str, int]] = {v: k for k, v in key_to_mem_id.items()}

    print(
        f"ingesting {len(ev_keys_ordered)} passages into mem0 "
        "(each goes through LLM memory extraction)...",
        file=sys.stderr,
    )
    t0 = time.time()
    for i, key in enumerate(ev_keys_ordered):
        text = ev_key_to_text[key]
        mem_id = key_to_mem_id[key]
        try:
            memory.add(
                messages=[{"role": "user", "content": text}],
                user_id=BENCH_USER,
                metadata={"mem_id": mem_id, "doc_name": key[0], "page_num": key[1]},
                run_id=mem_id,
            )
        except Exception as exc:
            print(f"  [warn] add failed for {key}: {exc}", file=sys.stderr)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  ingested {i+1}/{len(ev_keys_ordered)} ({elapsed:.0f}s)...", file=sys.stderr)
    t_ingest = time.time() - t0
    print(f"ingest done in {t_ingest:.1f}s", file=sys.stderr)

    # Retrieve + score
    from tqdm import tqdm  # type: ignore
    hits1 = hits3 = hits5 = 0
    per_q: list[dict[str, Any]] = []
    t_retrieve_total = 0.0

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
            results = memory.search(
                query=question,
                filters={"user_id": BENCH_USER},
                top_k=max(50, args.top_k * 10),
            )
        except Exception as exc:
            print(f"  [warn] search failed for {qid}: {exc}", file=sys.stderr)
            results = []
        t_retrieve_total += time.time() - ts

        # mem0 v2 returns {"results": [...]}; v1 returns a list directly
        result_list = results.get("results", []) if isinstance(results, dict) else (results or [])

        # BM25 rerank on extracted memory text then map back to source passage
        mem_texts = [r.get("memory") or r.get("text") or "" for r in result_list]
        if mem_texts:
            bm25 = _bm25_scores(question, mem_texts)
            result_list = [r for _, r in sorted(zip(bm25, result_list), key=lambda x: -x[0])]

        # Map results back to (doc_name, page_num) via metadata; deduplicate
        ranked_keys: list[tuple[str, int]] = []
        seen_keys: set[tuple[str, int]] = set()
        for r in result_list:
            meta = r.get("metadata") or {}
            mem_id = meta.get("mem_id") or r.get("id") or ""
            if mem_id in mem_id_to_key:
                key = mem_id_to_key[mem_id]
            else:
                m = re.match(r"^(.+)__p(\d+)$", str(mem_id))
                key = (m.group(1), int(m.group(2))) if m else None
            if key and key not in seen_keys:
                ranked_keys.append(key)
                seen_keys.add(key)

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

    n = len(records)
    r1, r3, r5 = hits1 / n, hits3 / n, hits5 / n

    report: dict[str, Any] = {
        "harness": "mem0-financebench",
        "system": "mem0",
        "embed_model": f"ollama:{args.embed_model}",
        "llm_model": f"ollama:{args.llm_model}",
        "note": (
            "mem0 extracts structured memories from passages via LLM before storing. "
            "Stored content differs from original passage text. "
            "mnem stores raw text; hit@5=0.9733 with same corpus."
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
        f"\n=== mem0 FinanceBench  hit@1={r1:.4f}  hit@3={r3:.4f}  "
        f"hit@{args.top_k}={r5:.4f}  n={n}  corpus={len(ev_keys_ordered)}",
        file=sys.stderr,
    )
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
