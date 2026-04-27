"""mnem MemBench adapter.

MemBench (ACL 2025) - import-myself/Membench. Multi-category
conversation memory with target-turn ground truth.

Per item:
  {
    "tid": int,
    "message_list": [[turn, ...], ...],  # list-of-lists of sessions
    "QA": {"question", "answer", "target_step_id": [[sid, turn_idx], ...],
           "choices", "ground_truth", ...}
  }

Each `turn` dict has `sid` (global turn id across all sessions),
`user_message`, `assistant_message`, `time`, `place`.

Scoring: target_step_id is a list of [sid, turn_idx] pairs. Recall@K
counts a hit if the retrieved top-K includes at least one turn whose
`sid` matches any target_step_id's first element.

Usage (from Lab root):
    python benchmarks/adapters/mnem/membench.py \
        --data-dir ../datasets/membench/FirstAgent \
        --mnem-http http://127.0.0.1:9876 \
        --out ../results/per_dataset/membench-mnem-bge-large.json \
        --category simple --topic roles --limit 100
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Any

import requests
from tqdm import tqdm


CATEGORY_FILES = {
    "simple": "simple.json",
    "highlevel": "highlevel.json",
    "knowledge_update": "knowledge_update.json",
    "comparative": "comparative.json",
    "conditional": "conditional.json",
    "noisy": "noisy.json",
    "aggregative": "aggregative.json",
    "highlevel_rec": "highlevel_rec.json",
    "lowlevel_rec": "lowlevel_rec.json",
    "RecMultiSession": "RecMultiSession.json",
    "post_processing": "post_processing.json",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=pathlib.Path, required=True)
    ap.add_argument("--mnem-http", default="http://127.0.0.1:9876")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--category", action="append", default=None,
                    help="Repeatable. Defaults to all.")
    ap.add_argument("--topic", default=None,
                    help="'movie' / 'roles' / 'events' / ... Defaults to all topics in file.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rerank", type=str, default=None)
    ap.add_argument("--summary-char-cap", type=int, default=2000)
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
    ap.add_argument("--vector-cap", type=int, default=10000,
                    help="Candidate pool size in vector lane (lift mnem's 256 default)")
    return ap.parse_args()


def load_items(data_dir: pathlib.Path, cats: list[str], topic: str | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for cat in cats:
        fname = CATEGORY_FILES.get(cat)
        if not fname:
            continue
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        raw = json.loads(fpath.read_text(encoding="utf-8"))
        for t, topic_items in raw.items():
            if topic and t != topic:
                continue
            for it in topic_items:
                turns = it.get("message_list", [])
                qa = it.get("QA", {})
                if not turns or not qa:
                    continue
                items.append({
                    "category": cat, "topic": t, "tid": it.get("tid", 0),
                    "turns": turns, "qa": qa,
                })
    return items


def flatten_turns(turns) -> list[tuple[int, int, int, dict]]:
    """Return [(global_idx, s_idx, t_idx, turn_dict), ...].

    Accepts either flat list-of-dicts or list-of-sessions.
    """
    flat: list[tuple[int, int, int, dict]] = []
    if turns and isinstance(turns[0], dict):
        sessions = [turns]
    else:
        sessions = turns
    g = 0
    for s_idx, sess in enumerate(sessions):
        if not isinstance(sess, list):
            continue
        for t_idx, turn in enumerate(sess):
            if isinstance(turn, dict):
                flat.append((g, s_idx, t_idx, turn))
                g += 1
    return flat


def render_turn(turn: dict[str, Any]) -> str:
    user = turn.get("user_message") or turn.get("user") or ""
    time_str = turn.get("time") or ""
    place = turn.get("place") or ""
    prefix = f"[{time_str}] " if time_str else ""
    tail = f" (@{place})" if place else ""
    return f"{prefix}{user}{tail}"


def main() -> int:
    args = parse_args()
    cats = args.category or list(CATEGORY_FILES.keys())
    items = load_items(args.data_dir, cats, args.topic)
    if args.limit:
        items = items[: args.limit]
    print(f"running {len(items)} membench items (cats={cats}, topic={args.topic})", file=sys.stderr)

    session = requests.Session()
    # C3 FIX-2: bump healthz timeout 5s -> 60s with retry to tolerate
    # MAX_PAR=6 ONNX-init contention.
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
    hits_by_cat: Counter[str] = Counter()
    per_item: list[dict[str, Any]] = []

    t0 = time.time()
    for ci, it in enumerate(tqdm(items, desc="items")):
        cat = it["category"]
        topic = it["topic"]
        tid = it["tid"]
        qa = it["qa"]
        question = qa.get("question") or ""
        target = qa.get("target_step_id") or []
        # target is [[sid, turn_idx], ...]; we match by first element (sid)
        target_sids: set[int] = set()
        for pair in target:
            if isinstance(pair, (list, tuple)) and pair:
                try:
                    target_sids.add(int(pair[0]))
                except (TypeError, ValueError):
                    pass

        label = f"MemBenchI:{cat}:{topic}:{ci}:{tid}"

        flat = flatten_turns(it["turns"])
        nodes_body: list[dict[str, Any]] = []
        turn_meta: list[tuple[int, int, int]] = []  # (global_idx, sid, s_idx)
        for (g, s_idx, t_idx, turn) in flat:
            sid = turn.get("sid", turn.get("mid", g))
            try:
                sid_i = int(sid)
            except (TypeError, ValueError):
                sid_i = g
            summary = render_turn(turn)
            if args.summary_char_cap > 0:
                summary = summary[: args.summary_char_cap]
            nodes_body.append({
                "label": label, "summary": summary,
                "props": {"sid": sid_i, "s_idx": s_idx, "t_idx": t_idx, "g": g},
                "author": "membench-harness",
            })
            turn_meta.append((g, sid_i, s_idx))

        if not nodes_body:
            continue

        ingest_body_mb: dict[str, Any] = {
            "nodes": nodes_body, "author": "membench-harness",
            "message": f"membench ingest {cat}/{topic}/{tid}",
            "auto_embed": True,
        }
        if args.extractor != "none":
            ingest_body_mb["extractor"] = args.extractor
        try:
            r = session.post(
                f"{args.mnem_http}/v1/nodes/bulk",
                json=ingest_body_mb,
                timeout=900,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as e:
            tqdm.write(f"  ingest fail on {cat}/{tid}: {type(e).__name__}: {str(e)[:160]}")
            continue
        node_to_sid: dict[str, int] = {}
        for (g, sid_i, s_idx), rv in zip(turn_meta, results):
            node_to_sid[rv["id"]] = sid_i

        body: dict[str, Any] = {"text": question, "label": label,
                                "limit": max(500, args.top_k * 50),
                                "vector_cap": args.vector_cap}
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
            r = session.post(f"{args.mnem_http}/v1/retrieve", json=body, timeout=300)
            r.raise_for_status()
            ret_items = r.json().get("items", [])
        except Exception as e:
            tqdm.write(f"  retrieve fail on {cat}/{tid}: {type(e).__name__}: {str(e)[:160]}")
            ret_items = []

        top_sids: list[int] = []
        seen: set[int] = set()
        for ret in ret_items:
            sid_i = node_to_sid.get(ret["id"])
            if sid_i is None or sid_i in seen:
                continue
            seen.add(sid_i)
            top_sids.append(sid_i)
            if len(top_sids) >= args.top_k:
                break

        hit_at_k = bool(target_sids) and any(s in target_sids for s in top_sids)
        totals_by_cat[cat] += 1
        if hit_at_k:
            hits_by_cat[cat] += 1

        per_item.append({
            "cat": cat, "topic": topic, "tid": tid,
            "question": question[:200],
            "target_sids": sorted(target_sids),
            f"hit@{args.top_k}": hit_at_k, f"top{args.top_k}": top_sids[: args.top_k],
        })

    dt = time.time() - t0
    total = sum(totals_by_cat.values())
    r_at_k = sum(hits_by_cat.values()) / total if total else 0.0

    report = {
        "harness": "mnem-membench",
        "mnem_http": args.mnem_http,
        "data_dir": str(args.data_dir),
        "category": cats, "topic": args.topic, "top_k": args.top_k,
        "n_items": total, "runtime_seconds": round(dt, 1),
        "overall": {f"recall@{args.top_k}": r_at_k},
        "by_category": {
            c: {"n": totals_by_cat[c],
                f"recall@{args.top_k}": hits_by_cat[c] / totals_by_cat[c]}
            for c in sorted(totals_by_cat)
        },
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log = args.out.with_suffix(".jsonl")
    with log.open("w", encoding="utf-8") as f:
        for row in per_item:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"=== mnem MemBench R@{args.top_k}={r_at_k:.4f}  n={total}")
    for c in sorted(totals_by_cat):
        n = totals_by_cat[c]
        print(f"  {c:24s}  n={n:3d}  R@{args.top_k}={hits_by_cat[c]/n:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
