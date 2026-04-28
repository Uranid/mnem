"""mnem LongMemEval adapter, *per-session chunking* variant.

Differs from `longmemeval.py` in exactly one dimension: instead of
ingesting each (session, turn) as its own Node, we concatenate every
turn of a session into a single Node. This matches MemPalace's raw
mode: one document per session. The prediction (Plan agent) is that
bge-large produces a stronger whole-session embedding than per-turn
fragments, and we lose the false-positives MAX-over-turns introduces
from short chit-chat turns.

All other logic (per-question fresh label, oversampled retrieve,
session-level R@5 / R@10) is identical to the per-turn variant.

Usage (from Lab root):
    python benchmarks/adapters/mnem/longmemeval_session.py \\
        --dataset ../datasets/longmemeval/longmemeval_s_cleaned.json \\
        --mnem-http http://127.0.0.1:9876 \\
        --out ../results/per_dataset/longmemeval-mnem-bge-large-session-100q.json \\
        --limit 100
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--mnem-http", default="http://127.0.0.1:9876")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rerank", type=str, default=None,
                    help="mnem-core reranker (cross-encoder). E.g. 'onnx:ms-marco-MiniLM-L-6-v2'")
    ap.add_argument("--graph-expand", type=int, default=None)
    # C3 full-layer retrieval flags (pass-through to /v1/retrieve)
    ap.add_argument("--community-filter", action="store_true",
                    help="Enable Leiden community filter (E1). Requires mnem-http c3-full-layer.")
    ap.add_argument("--graph-mode", choices=["decay", "ppr"], default="decay",
                    help="Graph expansion mode (E2). 'ppr' enables Personalized PageRank.")
    ap.add_argument("--summarize", action="store_true",
                    help="Enable recursive summarize layer (E4).")
    # C3 full-layer ingest-time flag (pass-through to /v1/nodes/bulk)
    ap.add_argument("--extractor", choices=["none", "keybert"], default="none",
                    help="Ingest-time keyphrase extractor (E3).")
    ap.add_argument("--session-char-cap", type=int, default=16000,
                    help="Char cap for the joined-session node.summary")
    ap.add_argument("--vector-cap", type=int, default=10000,
                    help="Candidate pool size in vector lane (lift mnem's 256 default)")
    # Bench-harness-only post-filters (not mnem core features):
    ap.add_argument("--llm-rerank", type=str, default=None,
                    help="Bench-only LLM rerank. Format 'ollama:MODEL' or 'openai:MODEL'. "
                         "Mirrors MemPalace's --llm-rerank benchmark flag. Not a mnem-core feature.")
    ap.add_argument("--llm-url", type=str, default="http://127.0.0.1:11434",
                    help="Ollama / OpenAI-compatible base URL for --llm-rerank.")
    ap.add_argument("--llm-rerank-pool", type=int, default=20,
                    help="How many top-K candidates the LLM sees (default 20).")
    ap.add_argument("--hybrid-v4-boost", action="store_true", default=False,
                    help="Bench-only: post-filter rescore with keyword + predicate boost. "
                         "Mirrors MemPalace's hybrid_v4 mode. Not a mnem-core feature.")
    ap.add_argument("--hybrid-boost-weight", type=float, default=0.3)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Bench-harness post-filters - NOT mnem core features. These mirror the
# same benchmark-only helpers MemPalace ships in
# baselines/mempalace/benchmarks/longmemeval_bench.py. Kept in this adapter
# so mnem-core stays free of BM25/LLM-rerank coupling
# ---------------------------------------------------------------------------

def hybrid_v4_boost(
    query: str,
    candidates: list[dict[str, Any]],
    boost_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """Keyword + predicate post-filter over mnem's top-K candidates.

    Each candidate keeps its original `score` as `dense_score` and gets a
    new `score = dense_score + w * overlap + predicate_bonus`. Re-ordered
    DESC. No LLM, no API calls, deterministic given same inputs.
    """
    q_lower = query.lower()
    q_tokens = set(re.findall(r"\w+", q_lower))
    want_date = bool(re.search(r"\b(when|what year|what month|date)\b", q_lower))
    want_num = bool(re.search(r"\b(how many|how much|number of|count)\b", q_lower))
    out: list[dict[str, Any]] = []
    for c in candidates:
        summ = (c.get("summary") or c.get("rendered") or "").lower()
        d_tokens = set(re.findall(r"\w+", summ))
        overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
        bonus = 0.0
        if want_date and re.search(r"\b(20\d\d|\d{1,2}/\d{1,2}|\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))", summ):
            bonus += 0.1
        if want_num and re.search(r"\b\d+\b", summ):
            bonus += 0.05
        c2 = dict(c)
        c2["dense_score"] = c.get("score", 0.0)
        c2["score"] = float(c.get("score", 0.0)) + boost_weight * overlap + bonus
        out.append(c2)
    out.sort(key=lambda x: -x["score"])
    return out


def llm_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    base_url: str,
    model: str,
    provider: str = "ollama",
    pool: int = 20,
    timeout: int = 60,
) -> list[dict[str, Any]]:
    """Bench-harness LLM rerank. Asks a chat model to rank candidates
    0..N-1 by relevance to `query`. Parses ranked integers, falls back
    to original order if the model output is unparseable.

    Mirrors the MemPalace `longmemeval_bench.py` --llm-rerank helper.
    """
    head = candidates[:pool]
    if not head:
        return candidates
    docs = "\n".join(
        f"[{i}] {(c.get('summary') or c.get('rendered') or '')[:300]}"
        for i, c in enumerate(head)
    )
    prompt = (
        f"Rank the following documents by how well they answer this question.\n"
        f"Question: {query}\n\n"
        f"Documents:\n{docs}\n\n"
        f"Return the document indices in order of relevance, most relevant first, "
        f"as a comma-separated list. Example: 3, 0, 5, 2. Nothing else."
    )
    try:
        if provider == "ollama":
            resp = requests.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0}},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
        else:
            # OpenAI-compatible chat endpoint
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json={"model": model, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        tqdm.write(f"  llm-rerank call failed: {type(e).__name__}: {str(e)[:120]}; keeping original order")
        return candidates

    nums = [int(n) for n in re.findall(r"\d+", text)]
    order: list[int] = []
    seen: set[int] = set()
    for n in nums:
        if 0 <= n < len(head) and n not in seen:
            order.append(n)
            seen.add(n)
    # Any head items the LLM skipped keep their original order at the tail of head.
    for i in range(len(head)):
        if i not in seen:
            order.append(i)
    reordered_head = [head[i] for i in order]
    return reordered_head + candidates[pool:]


def render_session(turns: list[dict[str, Any]], cap: int) -> str:
    lines: list[str] = []
    for turn in turns:
        if turn.get("role") != "user":
            continue
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(content)
    s = "\n".join(lines)
    return s[:cap] if cap > 0 else s


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]
    print(f"running {len(data)} questions (per-session chunking) against {args.mnem_http}",
          file=sys.stderr)

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

    totals_by_type: Counter[str] = Counter()
    hits5_by_type: Counter[str] = Counter()
    hits10_by_type: Counter[str] = Counter()
    per_q: list[dict[str, Any]] = []

    t0 = time.time()
    # C3 FIX-3: split wall-time into ingest / retrieve / score buckets so
    # we can attribute v0.1.0 summarize / community-filter overheads to the
    # correct phase. `runtime_seconds` stays the combined wall for
    # back-compat; a `timing` nested field carries the per-phase split.
    t_ingest = 0.0
    t_retrieve = 0.0
    t_score = 0.0
    for i, rec in enumerate(tqdm(data, desc="questions")):
        qid = rec["question_id"]
        qtype = rec.get("question_type", "unknown")
        question = rec["question"]
        answer_sids = set(rec.get("answer_session_ids") or [])
        hsids: list[str] = rec.get("haystack_session_ids") or []
        hsessions: list[list[dict]] = rec.get("haystack_sessions") or []

        label = f"LmeQs:{qid}"  # 's' = session-chunking to avoid label collision

        nodes_body: list[dict[str, Any]] = []
        sess_ids: list[str] = []
        for sid, turns in zip(hsids, hsessions):
            summary = render_session(turns, args.session_char_cap)
            if not summary:
                continue
            nodes_body.append({
                "label": label,
                "summary": summary,
                "props": {"session_id": sid},
                "author": "lme-session-harness",
            })
            sess_ids.append(sid)

        if not nodes_body:
            continue

        ingest_body_lme: dict[str, Any] = {
            "nodes": nodes_body, "author": "lme-session-harness",
            "message": f"lme-session ingest {qid}", "auto_embed": True,
        }
        if args.extractor != "none":
            ingest_body_lme["extractor"] = args.extractor
        _ts = time.time()
        r = session.post(
            f"{args.mnem_http}/v1/nodes/bulk",
            json=ingest_body_lme,
            timeout=900,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        t_ingest += time.time() - _ts
        node_to_sid: dict[str, str] = {
            rv["id"]: sid for sid, rv in zip(sess_ids, results)
        }

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
        _ts = time.time()
        r = session.post(f"{args.mnem_http}/v1/retrieve", json=body, timeout=300)
        r.raise_for_status()
        items = r.json().get("items", [])
        t_retrieve += time.time() - _ts

        _ts = time.time()
        # Optional bench-harness post-filters (not mnem core features).
        # Apply in order: hybrid-v4 boost -> LLM rerank. Matches the
        # MemPalace benchmark harness's optional-stage semantics.
        if args.hybrid_v4_boost:
            items = hybrid_v4_boost(question, items,
                                    boost_weight=args.hybrid_boost_weight)
        if args.llm_rerank:
            prov, _, model = args.llm_rerank.partition(":")
            items = llm_rerank(question, items,
                               base_url=args.llm_url,
                               model=model,
                               provider=prov,
                               pool=args.llm_rerank_pool)

        # Each item IS a session (no turn->session collapse needed)
        ranked = [node_to_sid.get(it["id"]) for it in items]
        ranked = [s for s in ranked if s]

        hit5 = any(s in answer_sids for s in ranked[:5])
        hit10 = any(s in answer_sids for s in ranked[:10])
        totals_by_type[qtype] += 1
        if hit5:
            hits5_by_type[qtype] += 1
        if hit10:
            hits10_by_type[qtype] += 1
        per_q.append({
            "qid": qid, "qtype": qtype,
            "hit@5": hit5, "hit@10": hit10,
            "top5_sessions": ranked[:5], "answer_sessions": list(answer_sids),
        })
        t_score += time.time() - _ts

        if (i + 1) % 20 == 0:
            so_far_5 = sum(hits5_by_type.values()) / sum(totals_by_type.values())
            tqdm.write(f"[progress {i+1}/{len(data)}] R@5={so_far_5:.4f}")

    dt = time.time() - t0
    total = sum(totals_by_type.values())
    r5 = sum(hits5_by_type.values()) / total if total else 0.0
    r10 = sum(hits10_by_type.values()) / total if total else 0.0

    report = {
        "harness": "mnem-lme-session",
        "mnem_http": args.mnem_http,
        "dataset": str(args.dataset),
        "chunking": "per-session",
        "n_questions": total,
        "runtime_seconds": round(dt, 1),
        "timing": {
            "ingest_s": round(t_ingest, 3),
            "retrieve_s": round(t_retrieve, 3),
            "score_s": round(t_score, 3),
        },
        "overall": {"recall@5": r5, "recall@10": r10},
        "by_type": {
            qt: {
                "n": totals_by_type[qt],
                "recall@5": hits5_by_type[qt] / totals_by_type[qt],
                "recall@10": hits10_by_type[qt] / totals_by_type[qt],
            }
            for qt in sorted(totals_by_type)
        },
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log = args.out.with_suffix(".jsonl")
    with log.open("w", encoding="utf-8") as f:
        for row in per_q:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"=== mnem LongMemEval (per-SESSION) R@5 = {r5:.4f}   R@10 = {r10:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
