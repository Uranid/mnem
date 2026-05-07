"""FinanceBench closed-haystack retrieval eval for MemPalace.

Runs the same 168-passage, 150-question eval as financebench.py but using
MemPalace as the retrieval backend.

DESIGN NOTES vs mnem
--------------------
MemPalace stores raw text in ChromaDB with FastEmbed (default: BAAI/bge-
small-en-v1.5, 384-dim).  Unlike mem0, there is no LLM extraction step,
the original passage text is stored verbatim, making this a closer
apples-to-apples comparison with mnem's document retrieval.

The main differences from mnem at hit@5:
  * Embedding model: MemPalace default = bge-small (384-dim)
    mnem test used bge-large (1024-dim, higher quality)
  * No hybrid keyword boost or GAAP query expansion (both adapters are
    run with only their default retrieval pipeline here)

mnem reference score (same corpus, same questions): hit@5 = 0.9733
  Embedder: bge-large (1024-dim) + hybrid-boost + query-expand + doc-filter.
  mnem vanilla (no retrieval tricks): roughly 0.85-0.90.

API notes (MemPalace v3.3.4)
----------------------------
MemPalace v3.3.4 does not export a Palace class from its top-level package.
The actual API is:

  * mempalace.palace.get_collection(palace_path, create=True)
      Returns the ChromaDB drawers collection.  Passages are upserted here
      directly with metadata {"wing": ..., "room": ..., "source_file": ...,
      "chunk_index": int}.

  * mempalace.searcher.search_memories(query, palace_path, wing=...,
                                        room=..., n_results=...)
      Returns {"results": [{"text": ..., "source_file": ..., ...}, ...]}
      Hybrid BM25 + vector re-rank, optional closet boost.

We encode (doc_name, page_num) into source_file as "<doc_name>__p<page>"
so results can be mapped back to the benchmark key without extra bookkeeping.

Requirements
-----------
    pip install mempalace

Usage
-----
    python benchmarks/harness/adapters/financebench_mempalace.py \\
        --dataset datasets/financebench/financebench_open_source.jsonl \\
        --out results/financebench-mempalace.json
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
    ap.add_argument("--storage-path", default=None,
                    help="MemPalace storage directory (default: temp dir)")
    ap.add_argument("--wing", default="financebench",
                    help="MemPalace wing (namespace) for isolation")
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
        from mempalace.palace import get_collection  # type: ignore
        from mempalace.searcher import search_memories  # type: ignore
    except ImportError:
        print("mempalace not installed.  Run: pip install mempalace", file=sys.stderr)
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

    # Use temp dir for storage unless explicitly given
    tmp_dir: str | None = None
    storage_path = args.storage_path
    if not storage_path:
        tmp_dir = tempfile.mkdtemp(prefix="mempalace_financebench_")
        storage_path = tmp_dir
    print(f"MemPalace storage: {storage_path}", file=sys.stderr)

    t0 = time.time()
    try:
        # Build corpus
        ev_key_to_text: dict[tuple[str, int], str] = {}
        for rec in records:
            for ev in rec.get("evidence") or []:
                key = (ev["doc_name"], int(ev["evidence_page_num"]))
                if key not in ev_key_to_text:
                    ev_key_to_text[key] = (ev.get("evidence_text") or "").strip()

        ev_keys_ordered = list(ev_key_to_text.keys())

        print(
            f"ingesting {len(ev_keys_ordered)} passages into MemPalace "
            "(raw text, no LLM extraction)...",
            file=sys.stderr,
        )

        # Get the drawers collection, create=True initialises the palace dir.
        col = get_collection(storage_path, create=True)

        for i, key in enumerate(ev_keys_ordered):
            text = ev_key_to_text[key]
            source_file = _key_to_source_file(key[0], key[1])
            doc_id = f"fb__{source_file}"
            meta: dict[str, Any] = {
                "wing": args.wing,
                "room": "passages",
                "source_file": source_file,
                "chunk_index": 0,
            }
            try:
                col.upsert(documents=[text], ids=[doc_id], metadatas=[meta])
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
                result = search_memories(
                    query=question,
                    palace_path=storage_path,
                    wing=args.wing,
                    room="passages",
                    n_results=max(50, args.top_k * 10),
                )
                raw_results = result.get("results") or [] if isinstance(result, dict) else []
            except Exception as exc:
                print(f"  [warn] search failed for {qid}: {exc}", file=sys.stderr)
                raw_results = []
            t_retrieve_total += time.time() - ts

            # Map results back to (doc_name, page_num) via source_file metadata
            ranked_keys: list[tuple[str, int]] = []
            seen: set[tuple[str, int]] = set()
            for r in raw_results:
                src = r.get("source_file") or ""
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
        "harness": "mempalace-financebench",
        "system": "MemPalace",
        "embed_model": "FastEmbed/BAAI-bge-small-en-v1.5 (MemPalace default, 384-dim)",
        "note": (
            "MemPalace stores raw passage text in ChromaDB without LLM extraction. "
            "Default embedding is bge-small (384-dim) vs mnem's bge-large (1024-dim). "
            "mnem reference: hit@5=0.9733 (bge-large + hybrid-boost + doc-filter)."
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
        f"\n=== MemPalace FinanceBench  hit@1={r1:.4f}  hit@3={r3:.4f}  "
        f"hit@{args.top_k}={r5:.4f}  n={n}  corpus={len(ev_keys_ordered)}",
        file=sys.stderr,
    )
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
