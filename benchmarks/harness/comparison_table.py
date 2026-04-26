"""Render apple-to-apple comparison table from v0.1.0 mnem runs vs MemPalace published numbers.

Reads result JSONs from run_apples_v04_vs_mempalace.sh and emits Markdown
table mirroring the user's reference layout: mnem dense (A) vs MP published,
plus the Hybrid v4 row that mirrors MP's harness helper.

Per-query retrieve latency (mean / p50 / p95 ms) is loaded from sidecar JSONLs
when present (fields `retrieve_s` or `t_retrieve_s`).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

# MemPalace published numbers (the user's reference table).
MP_REF = {
    "longmemeval-500q-r5":      0.966,
    "longmemeval-500q-r10":     0.982,
    "locomo-session-r5":        0.508,
    "locomo-session-r10":       0.603,
    "convomem-250":             0.929,
    "membench-simple-roles":    0.840,
    "membench-highlevel-movie": 0.950,
    "longmemeval-500q-hv4-r5":  0.982,
}

# (display_label, split, metric, mp_key, file, json_field_path)
ROW_DEFS = [
    ("LongMemEval", "500 Q (full)", "R@5 session",
     "longmemeval-500q-r5",
     "longmemeval-500q-A.json", ["overall", "recall@5"]),
    ("LongMemEval", "500 Q (full)", "R@10 session",
     "longmemeval-500q-r10",
     "longmemeval-500q-A.json", ["overall", "recall@10"]),
    ("LoCoMo", "1986 Q (full)", "R@5 session",
     "locomo-session-r5",
     "locomo-session-A.json", ["overall", "recall@5"]),
    ("LoCoMo", "1986 Q (full)", "R@10 session",
     "locomo-session-r10",
     "locomo-session-A.json", ["overall", "recall@10"]),
    ("ConvoMem", "5 cat x 50 items (250)", "avg recall",
     "convomem-250",
     "convomem-250-A.json", ["overall", "avg_recall"]),
    ("MemBench", "simple/roles, 100 items", "R@5",
     "membench-simple-roles",
     "membench-simple-roles-A.json", ["overall", "recall@5"]),
    ("MemBench", "highlevel/movie, 100 items", "R@5",
     "membench-highlevel-movie",
     "membench-highlevel-movie-A.json", ["overall", "recall@5"]),
    ("LongMemEval", "500 Q, Hybrid v4", "R@5 session",
     "longmemeval-500q-hv4-r5",
     "longmemeval-500q-hv4.json", ["overall", "recall@5"]),
]


def dig(d: dict, path: list[str]) -> Any:
    cur = d
    for k in path:
        if cur is None or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_score(results_dir: pathlib.Path, fname: str,
               path: list[str]) -> float | None:
    p = results_dir / fname
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    v = dig(d, path)
    return float(v) if isinstance(v, (int, float)) else None


def load_latency_ms(results_dir: pathlib.Path, fname: str) -> dict[str, float] | None:
    """Mean per-query retrieve latency from summary JSON aggregate timing.
    Adapters track total `timing.retrieve_s` and `runtime_seconds`, not per-query,
    so only mean is available (no p50/p95).
    Fallback to runtime_seconds / n if no per-phase timing is exposed.
    """
    p = results_dir / fname
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    n = d.get("n_questions") or d.get("total") or d.get("overall", {}).get("n")
    # MemBench / ConvoMem omit n; infer from filename heuristic.
    if not n:
        if "convomem-250" in fname:
            n = 250
        elif "membench-" in fname and "-100" not in fname:
            n = 100
        else:
            n = 1
    timing = d.get("timing") or {}
    retrieve_s = timing.get("retrieve_s")
    if retrieve_s is not None:
        return {"mean_retrieve_ms": float(retrieve_s) * 1000.0 / float(n)}
    rt = d.get("runtime_seconds")
    if rt is not None:
        return {"mean_runtime_ms": float(rt) * 1000.0 / float(n)}
    return None


def fmt_score(s: float | None) -> str:
    return f"{s:.3f}" if s is not None else "-"


def fmt_delta(mnem: float | None, mp: float | None) -> str:
    if mnem is None or mp is None:
        return "-"
    d = mnem - mp
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.3f}"


def fmt_lat(lat: dict[str, float] | None) -> str:
    if not lat:
        return "-"
    if "mean_retrieve_ms" in lat:
        return f"{lat['mean_retrieve_ms']:.0f} (retr)"
    if "mean_runtime_ms" in lat:
        return f"{lat['mean_runtime_ms']:.0f} (e2e)"
    return "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    lines = [
        "# mnem v0.1.0 vs MemPalace - apple-to-apple",
        "",
        "ONNX MiniLM-L6-v2 (bundled, in-process). 4 cores per lane, MNEM_BENCH=1.",
        "Config: dense-only (vector + top-k). "
        "Hybrid v4 row uses `--hybrid-v4-boost` (mirrors MP's harness helper). "
        "No LLM rerank. No graph-RAG (graph-on stack tested separately - "
        "showed equal recall to dense at 15-67x latency cost on these benchmarks; "
        "dense already saturates).",
        "",
        "Latency = mean ms per query. `(retr)` = retrieve-only (from summary timing); "
        "`(e2e)` = end-to-end (runtime / n) when adapter doesn't expose phase timing. "
        "Per-query p50/p95 not captured by adapters in this run.",
        "",
        "| Benchmark | Split | Metric | MP | mnem v0.1.0 | Δ vs MP | Latency (ms) |",
        "|-----------|-------|--------|----|-------------|---------|--------------|",
    ]

    for (bench, split, metric, mp_key, file_a, path_a) in ROW_DEFS:
        mp = MP_REF.get(mp_key)
        score = load_score(args.results, file_a, path_a)
        lat = load_latency_ms(args.results, file_a)
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                bench, split, metric,
                fmt_score(mp),
                fmt_score(score),
                fmt_delta(score, mp),
                fmt_lat(lat),
            )
        )

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
