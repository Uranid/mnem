"""mnem LoCoMo adapter.

LoCoMo (Snap 2024) - 10 conversations * ~200 QA pairs each.
Dataset: `locomo10.json` from snap-research/LoCoMo.

Schema per conversation:
  {
    "qa": [{"question", "answer", "evidence": ["D2:5", ...], "category": int}],
    "conversation": {
      "session_1": [{"speaker", "text", "dia_id": "D1:3"}, ...],
      "session_1_date_time": "...",
      "session_N": [...], "session_N_date_time": "...",
    }
  }

Granularity:
  - dialog   (default): score evidence dia_ids vs retrieved turn dia_ids
  - session: score session_id of evidence vs retrieved turn session_id

We run per-conversation (fresh label `LoCoMoC:<sample_id>`) so
cross-conversation retrieval can't leak.

Usage (from Lab root):
    python benchmarks/adapters/mnem/locomo.py \
        --dataset ../datasets/locomo/locomo10.json \
        --mnem-http http://127.0.0.1:9876 \
        --out ../results/per_dataset/locomo-mnem-bge-large.json \
        --granularity dialog
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from collections import Counter
from typing import Any

import requests
from tqdm import tqdm

CATEGORY_NAMES = {
    1: "single-hop", 2: "temporal", 3: "temporal-inference",
    4: "open-domain", 5: "adversarial",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--mnem-http", default="http://127.0.0.1:9876")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--granularity", choices=("dialog", "session"), default="dialog")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="limit conversations processed")
    ap.add_argument("--rerank", type=str, default=None)
    ap.add_argument("--graph-expand", type=int, default=None)
    ap.add_argument("--summary-char-cap", type=int, default=2000)
    # C3 full-layer retrieval flags
    ap.add_argument("--community-filter", action="store_true",
                    help="Enable Leiden community filter (E1).")
    ap.add_argument("--graph-mode", choices=["decay", "ppr"], default="decay",
                    help="Graph expansion mode (E2).")
    ap.add_argument("--summarize", action="store_true",
                    help="Enable recursive summarize layer (E4).")
    # C3 ingest-time flag
    ap.add_argument("--extractor", choices=["none", "keybert"], default="none",
                    help="Ingest-time keyphrase extractor (E3).")
    ap.add_argument("--vector-cap", type=int, default=10000,
                    help="Candidate pool size in vector lane (lift mnem's 256 default)")
    return ap.parse_args()


def iter_sessions(conv: dict[str, Any]):
    # LoCoMo packs sessions as flat keys session_1, session_1_date_time, ...
    idx = 1
    while True:
        skey = f"session_{idx}"
        if skey not in conv:
            break
        dialogs = conv[skey] or []
        date = conv.get(f"{skey}_date_time", "")
        yield idx, date, dialogs
        idx += 1


def session_of(dia_id: str) -> str | None:
    # "D1:3" -> "session_1"
    m = re.match(r"^D(\d+):", dia_id or "")
    return f"session_{m.group(1)}" if m else None


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    session = requests.Session()
    # C3 FIX-2: bump healthz timeout 5s -> 60s and retry up to 10x
    # with 3s sleep so MAX_PAR=6 ONNX-init races do not kill the run.
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
    if not ok:
        print(f"mnem-http unreachable after 10 attempts: {last_err}", file=sys.stderr)
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)

    totals_by_cat: Counter[str] = Counter()
    hits5_by_cat: Counter[str] = Counter()
    hits10_by_cat: Counter[str] = Counter()
    per_question: list[dict[str, Any]] = []

    t0 = time.time()
    # C3 FIX-3: split wall-time into ingest / retrieve / score buckets so
    # we can attribute v0.1.0 summarize / community-filter overheads to the
    # correct phase. `runtime_seconds` stays the combined wall for
    # back-compat; a `timing` nested field carries the per-phase split.
    t_ingest = 0.0
    t_retrieve = 0.0
    t_score = 0.0
    for ci, conv_rec in enumerate(tqdm(data, desc="conversations")):
        sample_id = conv_rec.get("sample_id", f"conv_{ci}")
        label = f"LoCoMoC:{sample_id}"
        qa = conv_rec.get("qa") or []
        conv = conv_rec.get("conversation") or {}

        # --- Ingest ---
        nodes_body: list[dict[str, Any]] = []
        turn_keys: list[tuple[str, str]] = []  # (dia_id, session_key)
        for sidx, date, dialogs in iter_sessions(conv):
            skey = f"session_{sidx}"
            for d in dialogs:
                dia_id = d.get("dia_id") or f"D{sidx}:?"
                speaker = d.get("speaker", "?")
                text = (d.get("text") or "").strip()
                summary = f"{speaker}: {text}"
                if args.summary_char_cap > 0:
                    summary = summary[: args.summary_char_cap]
                nodes_body.append({
                    "label": label,
                    "summary": summary,
                    "props": {
                        "dia_id": dia_id, "session": skey, "date": date, "speaker": speaker,
                    },
                    "author": "locomo-harness",
                })
                turn_keys.append((dia_id, skey))

        if not nodes_body:
            continue

        ingest_body = {
            "nodes": nodes_body, "author": "locomo-harness",
            "message": f"locomo ingest {sample_id}", "auto_embed": True,
        }
        if args.extractor != "none":
            ingest_body["extractor"] = args.extractor
        try:
            _ts = time.time()
            r = session.post(f"{args.mnem_http}/v1/nodes/bulk", json=ingest_body, timeout=900)
            r.raise_for_status()
            results = r.json().get("results", [])
            t_ingest += time.time() - _ts
        except Exception as e:
            print(f"  ingest fail on {sample_id}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        node_to_dia: dict[str, str] = {}
        node_to_sess: dict[str, str] = {}
        for (dia_id, skey), rv in zip(turn_keys, results):
            node_to_dia[rv["id"]] = dia_id
            node_to_sess[rv["id"]] = skey

        # --- Retrieve + score per QA ---
        for q in qa:
            question = q.get("question") or ""
            evidence = q.get("evidence") or []
            cat = CATEGORY_NAMES.get(q.get("category", 0), "unknown")
            ev_dias = set(evidence)
            ev_sess = {session_of(d) for d in evidence if session_of(d)}

            body: dict[str, Any] = {
                "text": question, "label": label,
                "limit": max(500, args.top_k * 50),
                "vector_cap": args.vector_cap,
            }
            if args.rerank:
                body["rerank"] = args.rerank
            if args.graph_expand is not None:
                body["graph_expand"] = args.graph_expand
            if args.community_filter:
                body["community_filter"] = True
                body["community_min_coverage"] = 0.1
            if args.graph_mode != "decay":
                body["graph_mode"] = args.graph_mode
            if args.summarize:
                body["summarize"] = True
            try:
                _ts = time.time()
                r = session.post(f"{args.mnem_http}/v1/retrieve", json=body, timeout=300)
                r.raise_for_status()
                items = r.json().get("items", [])
                t_retrieve += time.time() - _ts
            except Exception as e:
                print(f"  retrieve fail in {sample_id}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                items = []

            _ts = time.time()
            if args.granularity == "dialog":
                seen: dict[str, float] = {}
                for it in items:
                    dia = node_to_dia.get(it["id"])
                    if not dia:
                        continue
                    sc = float(it["score"])
                    if sc > seen.get(dia, float("-inf")):
                        seen[dia] = sc
                ranked = [k for k, _ in sorted(seen.items(), key=lambda kv: -kv[1])]
                hit5 = bool(ev_dias) and any(d in ev_dias for d in ranked[:5])
                hit10 = bool(ev_dias) and any(d in ev_dias for d in ranked[:10])
                top5 = ranked[:5]
            else:  # session granularity: aggregate MAX per session_N
                seen = {}
                for it in items:
                    sk = node_to_sess.get(it["id"])
                    if not sk:
                        continue
                    sc = float(it["score"])
                    if sc > seen.get(sk, float("-inf")):
                        seen[sk] = sc
                ranked = [k for k, _ in sorted(seen.items(), key=lambda kv: -kv[1])]
                hit5 = bool(ev_sess) and any(s in ev_sess for s in ranked[:5])
                hit10 = bool(ev_sess) and any(s in ev_sess for s in ranked[:10])
                top5 = ranked[:5]

            totals_by_cat[cat] += 1
            if hit5:
                hits5_by_cat[cat] += 1
            if hit10:
                hits10_by_cat[cat] += 1
            per_question.append({
                "sample_id": sample_id, "question": question[:200],
                "category": cat, "evidence": list(ev_dias),
                "hit@5": hit5, "hit@10": hit10, "top5": top5,
            })
            t_score += time.time() - _ts

    dt = time.time() - t0
    total = sum(totals_by_cat.values())
    r5 = sum(hits5_by_cat.values()) / total if total else 0.0
    r10 = sum(hits10_by_cat.values()) / total if total else 0.0

    report = {
        "harness": "mnem-locomo",
        "mnem_http": args.mnem_http,
        "dataset": str(args.dataset),
        "granularity": args.granularity,
        "n_questions": total,
        "runtime_seconds": round(dt, 1),
        "timing": {
            "ingest_s": round(t_ingest, 3),
            "retrieve_s": round(t_retrieve, 3),
            "score_s": round(t_score, 3),
        },
        "overall": {"recall@5": r5, "recall@10": r10},
        "by_category": {
            c: {
                "n": totals_by_cat[c],
                "recall@5": hits5_by_cat[c] / totals_by_cat[c],
                "recall@10": hits10_by_cat[c] / totals_by_cat[c],
            }
            for c in sorted(totals_by_cat)
        },
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log = args.out.with_suffix(".jsonl")
    with log.open("w", encoding="utf-8") as f:
        for row in per_question:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"=== mnem LoCoMo ({args.granularity}) R@5={r5:.4f}  R@10={r10:.4f}  n={total}")
    for c in sorted(totals_by_cat):
        n = totals_by_cat[c]
        print(f"  {c:24s}  n={n:3d}  R@5={hits5_by_cat[c]/n:.4f}  R@10={hits10_by_cat[c]/n:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
