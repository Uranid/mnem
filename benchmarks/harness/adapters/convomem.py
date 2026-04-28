"""mnem ConvoMem adapter.

Mirrors MemPalace's `convomem_bench.py` methodology so our numbers
land directly alongside theirs:

    For each evidence_item in the Salesforce/ConvoMem HF dataset:
      1. Ingest every message from the item's `conversations` as a
         per-item-labelled node (label = `ConvoMemI:<cat>:<idx>`).
      2. Retrieve the top-K messages for the `question`.
      3. recall = fraction of `message_evidences[*].text` that appear
         (substring-match either direction) in any retrieved text.

    Overall = mean recall over all items; per-category breakdown too.

Evidence files are downloaded from HuggingFace via direct ureq/urllib
(the MemPalace bench uses Python's urllib, we use the same).

Usage (from Lab root):
    python benchmarks/adapters/mnem/convomem.py \
        --mnem-http http://127.0.0.1:9876 \
        --limit 50 \
        --out ../results/per_dataset/convomem-mnem-onnx-bge-large-250.json

The default category list matches MemPalace's default (all 6); the
default --limit 50 and top_k 10 also match their headline config.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request
from collections import defaultdict
from typing import Any

import requests
from tqdm import tqdm


HF_BASE = (
    "https://huggingface.co/datasets/Salesforce/ConvoMem/resolve/main/"
    "core_benchmark/evidence_questions"
)

CATEGORIES = {
    "user_evidence": "User Facts",
    "assistant_facts_evidence": "Assistant Facts",
    "changing_evidence": "Changing Facts",
    "abstention_evidence": "Abstention",
    "preference_evidence": "Preferences",
    "implicit_connection_evidence": "Implicit Connections",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnem-http", default="http://127.0.0.1:9876")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=50,
                    help="Items per category (matches MemPalace's 50-per-cat headline)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--category", action="append", default=None,
        help="Repeatable. Default = all 6 categories."
    )
    ap.add_argument(
        "--cache-dir", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent.parent /
                "datasets" / "convomem",
        help="Where to stash downloaded HF evidence JSONs."
    )
    # C3 full-layer retrieval flags
    ap.add_argument("--graph-expand", type=int, default=None,
                    help="Multi-hop expansion budget (E2). Set with --graph-mode.")
    ap.add_argument("--community-filter", action="store_true",
                    help="Enable Leiden community filter (E1).")
    ap.add_argument("--graph-mode", choices=["decay", "ppr"], default="decay",
                    help="Graph expansion mode (E2).")
    ap.add_argument("--summarize", action="store_true",
                    help="Enable recursive summarize layer (E4).")
    # C3 ingest-time flag
    ap.add_argument("--extractor", choices=["none", "keybert"], default="none",
                    help="Ingest-time keyphrase extractor (E3).")
    return ap.parse_args()


def download_evidence_file(category: str, subpath: str, cache_dir: pathlib.Path) -> dict | None:
    cache_path = cache_dir / category / subpath.replace("/", "_")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = f"{HF_BASE}/{category}/{subpath}"
    try:
        urllib.request.urlretrieve(url, str(cache_path))
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[convomem] download fail {url}: {e}", file=sys.stderr)
        return None


def discover_files(category: str, cache_dir: pathlib.Path) -> list[str]:
    api_url = (
        f"https://huggingface.co/api/datasets/Salesforce/ConvoMem/tree/main/"
        f"core_benchmark/evidence_questions/{category}/1_evidence"
    )
    cache_path = cache_dir / f"{category}_filelist.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=20) as resp:
            files = json.loads(resp.read())
            paths = [
                f["path"].split(f"{category}/")[1]
                for f in files
                if f["path"].endswith(".json")
            ]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(paths), encoding="utf-8")
            return paths
    except Exception as e:
        print(f"[convomem] discover fail {category}: {e}", file=sys.stderr)
        return []


def load_evidence_items(categories: list[str], limit: int, cache_dir: pathlib.Path) -> list[dict]:
    all_items: list[dict] = []
    for cat in categories:
        files = discover_files(cat, cache_dir)
        if not files:
            print(f"[convomem] no files for {cat}; skipping", file=sys.stderr)
            continue
        items_for_cat: list[dict] = []
        for fpath in files:
            if len(items_for_cat) >= limit:
                break
            data = download_evidence_file(cat, fpath, cache_dir)
            if data and "evidence_items" in data:
                for it in data["evidence_items"]:
                    it["_category_key"] = cat
                    items_for_cat.append(it)
        all_items.extend(items_for_cat[:limit])
        print(f"  {CATEGORIES.get(cat, cat)}: loaded {len(items_for_cat[:limit])} items",
              file=sys.stderr)
    return all_items


def score_item(session: requests.Session, base_url: str, item: dict,
               top_k: int, idx: int,
               c3_flags: dict | None = None) -> tuple[float, dict]:
    # C3 FIX-3: return per-phase timing in `details` so the caller can
    # accumulate a `timing` bucket split alongside `runtime_seconds`.
    c3_flags = c3_flags or {}
    conversations = item.get("conversations") or []
    question = item["question"]
    evidence_texts = {
        (e.get("text") or "").strip().lower()
        for e in (item.get("message_evidences") or [])
    }
    if not evidence_texts:
        return 1.0, {"reason": "empty evidence"}

    cat = item.get("_category_key", "unknown")
    label = f"ConvoMemI:{cat}:{idx}"

    # Flatten all messages into the corpus; remember order so we can
    # map back to speaker/original text later if needed.
    nodes: list[dict[str, Any]] = []
    corpus_text: list[str] = []
    for conv in conversations:
        for msg in conv.get("messages") or []:
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            nodes.append({
                "label": label,
                "summary": text,
                "props": {"speaker": msg.get("speaker", "?"), "idx": len(corpus_text)},
                "author": "convomem-harness",
            })
            corpus_text.append(text)

    if not nodes:
        return 0.0, {"reason": "empty corpus"}

    # Ingest
    ingest_body: dict[str, Any] = {
        "nodes": nodes, "author": "convomem-harness",
        "message": f"convomem ingest {cat}/{idx}",
        "auto_embed": True,
    }
    if c3_flags.get("extractor", "none") != "none":
        ingest_body["extractor"] = c3_flags["extractor"]
    _ts = time.time()
    r = session.post(
        f"{base_url}/v1/nodes/bulk",
        json=ingest_body,
        timeout=900,
    )
    r.raise_for_status()
    ingest_s = time.time() - _ts

    # Retrieve (lift both vector_cap and output limit vs mnem's 256 default)
    retrieve_body: dict[str, Any] = {
        "text": question, "label": label,
        "limit": max(500, top_k * 50),
        "vector_cap": 10000,
    }
    if c3_flags.get("graph_expand") is not None:
        retrieve_body["graph_expand"] = c3_flags["graph_expand"]
    if c3_flags.get("community_filter"):
        retrieve_body["community_filter"] = True
        retrieve_body["community_min_coverage"] = 0.1
    if c3_flags.get("graph_mode", "decay") != "decay":
        retrieve_body["graph_mode"] = c3_flags["graph_mode"]
    if c3_flags.get("summarize"):
        retrieve_body["summarize"] = True
    _ts = time.time()
    r = session.post(
        f"{base_url}/v1/retrieve",
        json=retrieve_body,
        timeout=300,
    )
    r.raise_for_status()
    items_ret = r.json().get("items", [])
    retrieve_s = time.time() - _ts

    _ts = time.time()
    retrieved_texts = [(it.get("summary") or "").strip().lower() for it in items_ret]

    found = 0
    for ev in evidence_texts:
        for ret in retrieved_texts:
            if ev in ret or ret in ev:
                found += 1
                break

    recall = found / len(evidence_texts)
    score_s = time.time() - _ts
    return recall, {
        "retrieved_count": len(retrieved_texts),
        "evidence_count": len(evidence_texts),
        "found": found,
        "_timing": {
            "ingest_s": ingest_s,
            "retrieve_s": retrieve_s,
            "score_s": score_s,
        },
    }


def main() -> int:
    args = parse_args()
    cats = args.category or list(CATEGORIES.keys())
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    # C3 FIX-2: bump healthz timeout 5s -> 60s with retry to tolerate
    # MAX_PAR=6 ONNX-init contention (matrix v4 saw ReadTimeout=5s in
    # batch-of-6 convomem bringup).
    last_err: Exception | None = None
    ok = False
    for _attempt in range(10):
        try:
            h = session.get(f"{args.mnem_http}/v1/healthz", timeout=60).json()
            if h.get("ok"):
                ok = True
                break
        except Exception as e:
            last_err = e
            time.sleep(3)
            continue
        time.sleep(3)
    assert ok, f"mnem-http unhealthy after 10 attempts: {last_err}"

    items = load_evidence_items(cats, args.limit, args.cache_dir)
    if not items:
        print("No items loaded; aborting", file=sys.stderr)
        return 2

    print(f"total items: {len(items)} across {len(cats)} categories", file=sys.stderr)

    all_recall: list[float] = []
    per_category: dict[str, list[float]] = defaultdict(list)
    per_item: list[dict[str, Any]] = []

    t0 = time.time()
    # C3 FIX-3: per-phase timing accumulation alongside wall-clock.
    t_ingest = 0.0
    t_retrieve = 0.0
    t_score = 0.0
    for i, it in enumerate(tqdm(items, desc="items")):
        try:
            recall, details = score_item(
                session, args.mnem_http, it, args.top_k, i,
                c3_flags={
                    "graph_expand": args.graph_expand,
                    "community_filter": args.community_filter,
                    "graph_mode": args.graph_mode,
                    "summarize": args.summarize,
                    "extractor": args.extractor,
                },
            )
        except Exception as e:
            tqdm.write(f"  item {i} failed: {type(e).__name__}: {e!s:.120}")
            recall, details = 0.0, {"error": str(e)[:200]}
        _t = details.get("_timing") if isinstance(details, dict) else None
        if _t:
            t_ingest += _t.get("ingest_s", 0.0)
            t_retrieve += _t.get("retrieve_s", 0.0)
            t_score += _t.get("score_s", 0.0)
        all_recall.append(recall)
        per_category[it.get("_category_key", "unknown")].append(recall)
        per_item.append({
            "i": i, "category": it.get("_category_key", "unknown"),
            "question": it.get("question", "")[:200], "recall": recall,
            "details": details,
        })
        if (i + 1) % 20 == 0:
            so_far = sum(all_recall) / len(all_recall)
            tqdm.write(f"[progress {i + 1}/{len(items)}] avg_recall={so_far:.4f}")

    dt = time.time() - t0
    avg = sum(all_recall) / len(all_recall) if all_recall else 0.0
    perfect = sum(1 for r in all_recall if r >= 1.0)
    zero = sum(1 for r in all_recall if r == 0.0)

    report = {
        "harness": "mnem-convomem",
        "mnem_http": args.mnem_http,
        "top_k": args.top_k,
        "limit_per_category": args.limit,
        "n_items": len(all_recall),
        "runtime_seconds": round(dt, 1),
        "timing": {
            "ingest_s": round(t_ingest, 3),
            "retrieve_s": round(t_retrieve, 3),
            "score_s": round(t_score, 3),
        },
        "overall": {"avg_recall": avg,
                    "perfect_items": perfect,
                    "zero_items": zero},
        "by_category": {
            c: {"n": len(v), "avg_recall": sum(v) / len(v) if v else 0.0,
                "perfect": sum(1 for r in v if r >= 1.0)}
            for c, v in sorted(per_category.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_path = args.out.with_suffix(".jsonl")
    with log_path.open("w", encoding="utf-8") as f:
        for row in per_item:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"=== mnem ConvoMem avg_recall = {avg:.4f}  perfect={perfect}/{len(all_recall)}"
          f"  zero={zero}/{len(all_recall)}")
    for c in sorted(per_category):
        vals = per_category[c]
        label = CATEGORIES.get(c, c)
        print(f"  {label:25s}  n={len(vals):3d}  avg_recall={sum(vals) / len(vals):.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
