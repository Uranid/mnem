"""mnem FinanceBench adapter (closed-haystack retrieval evaluation).

FinanceBench (Patronus AI 2024) - 150 open-source Q&A pairs on SEC filings.

Each record has:
  - financebench_id  : unique question ID
  - company          : company name
  - doc_name         : document identifier (e.g. "3M_2018_10K")
  - question_type    : "metrics-generated" | "domain-relevant"
  - question         : natural-language financial question
  - answer           : human-annotated ground-truth answer
  - evidence         : list of {evidence_text, doc_name, evidence_page_num}

Evaluation (closed-haystack retrieval):
  Ingest all unique evidence passages across the 150 questions as a corpus
  (keyed by (doc_name, evidence_page_num) to deduplicate shared pages).
  For each question, retrieve from the full ~150-passage corpus and check
  whether the correct (doc_name, evidence_page_num) appears in top-K.

  Metric: hit@1, hit@3, hit@5 on evidence page match.
  Corpus size: ~150 unique passages (may be fewer after page dedup).

  This measures mnem's retrieval accuracy on financial domain text, NOT
  end-to-end QA accuracy. A full QA evaluation requires downloading the
  ~90 PDF files and running a separate LLM judging step.

Usage:
    python benchmarks/harness/adapters/financebench.py \\
        --dataset datasets/financebench/financebench_open_source.jsonl \\
        --mnem-http http://127.0.0.1:9876 \\
        --out results/financebench-mnem.json \\
        --hybrid-boost --query-expand
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

BENCH_LABEL = "FinBench"

# ---------------------------------------------------------------------------
# Doc-level company+year filter
# ---------------------------------------------------------------------------

# Company aliases (lowercase) → doc_name prefix.  Longer entries first so
# "american express" matches before "american".  Sorted at module load below.
_COMPANY_ALIASES: list[tuple[str, str]] = [
    ("activision blizzard", "ACTIVISIONBLIZZARD"),
    ("activision",          "ACTIVISIONBLIZZARD"),
    ("adobe",               "ADOBE"),
    ("aes",                 "AES"),
    ("amazon",              "AMAZON"),
    ("amcor",               "AMCOR"),
    ("amd",                 "AMD"),
    ("american express",    "AMERICANEXPRESS"),
    ("amex",                "AMERICANEXPRESS"),
    ("american water works","AMERICANWATERWORKS"),
    ("american water",      "AMERICANWATERWORKS"),
    ("best buy",            "BESTBUY"),
    ("block",               "BLOCK"),
    ("boeing",              "BOEING"),
    ("coca-cola",           "COCACOLA"),
    ("coca cola",           "COCACOLA"),
    ("cocacola",            "COCACOLA"),
    ("coke",                "COCACOLA"),
    ("corning",             "CORNING"),
    ("costco",              "COSTCO"),
    ("cvs health",          "CVSHEALTH"),
    ("cvs",                 "CVSHEALTH"),
    ("foot locker",         "FOOTLOCKER"),
    ("footlocker",          "FOOTLOCKER"),
    ("general mills",       "GENERALMILLS"),
    ("johnson & johnson",   "JOHNSON_JOHNSON"),
    ("johnson and johnson", "JOHNSON_JOHNSON"),
    ("j&j",                 "JOHNSON_JOHNSON"),
    ("jnj",                 "JOHNSON_JOHNSON"),
    ("j&j",                 "JOHNSON_JOHNSON"),
    ("jpmorgan chase",      "JPMORGAN"),
    ("jpmorgan",            "JPMORGAN"),
    ("jp morgan",           "JPMORGAN"),
    ("jpm",                 "JPMORGAN"),
    ("j.p. morgan",         "JPMORGAN"),
    ("kraft heinz",         "KRAFTHEINZ"),
    ("lockheed martin",     "LOCKHEEDMARTIN"),
    ("mgm resorts",         "MGMRESORTS"),
    ("mgm",                 "MGMRESORTS"),
    ("microsoft",           "MICROSOFT"),
    ("netflix",             "NETFLIX"),
    ("nike",                "NIKE"),
    ("paypal",              "PAYPAL"),
    ("pepsico",             "PEPSICO"),
    ("pepsi",               "PEPSICO"),
    ("pfizer",              "PFIZER"),
    ("ulta beauty",         "ULTABEAUTY"),
    ("ulta",                "ULTABEAUTY"),
    ("verizon",             "VERIZON"),
    ("walmart",             "WALMART"),
    ("wal-mart",            "WALMART"),
    ("3m",                  "3M"),
]
_COMPANY_ALIASES.sort(key=lambda x: -len(x[0]))

# Pre-compiled word-boundary patterns for each alias (expensive to rebuild per query)
_COMPANY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.I), prefix)
    for alias, prefix in _COMPANY_ALIASES
]

# Regex for fiscal year extraction from question text
_FY_FULL  = re.compile(r"\bfy\s*(\d{4})\b", re.I)      # FY2022, fy 2022
_FY_SHORT = re.compile(r"\bfy\s*(\d{2})\b", re.I)       # FY22, fy22
_YEAR_RAW = re.compile(r"\b(20\d{2}|19\d{2})\b")        # standalone 4-digit year


def _extract_doc_filter(question: str) -> tuple[str | None, set[int]]:
    """Extract (company_prefix, year_set) from a question string.

    Returns the first matching company prefix (longest alias wins) and all
    calendar years mentioned.  Returns (None, set()) when nothing found.
    """
    q = question.lower()
    company: str | None = None
    for pattern, prefix in _COMPANY_PATTERNS:
        if pattern.search(q):
            company = prefix
            break

    years: set[int] = set()
    for m in _FY_FULL.finditer(q):
        years.add(int(m.group(1)))
    for m in _FY_SHORT.finditer(q):
        y2 = int(m.group(1))
        years.add(2000 + y2)           # FY22 → 2022
    for m in _YEAR_RAW.finditer(question):
        years.add(int(m.group(1)))

    return company, years


def _doc_name_match_score(doc_name: str,
                          company: str | None,
                          years: set[int]) -> float:
    """Compute a compatibility score between a doc_name and the extracted filter.

    Returns 0.0 when the filter is empty (no degradation of original ranking).
    """
    if not company and not years:
        return 0.0

    # Extract the year embedded in the doc_name (_YYYY_ or _YYYYQx_)
    doc_year_m = re.search(r"_(\d{4})(?:[Qq]\d)?_", doc_name)
    doc_year = int(doc_year_m.group(1)) if doc_year_m else None

    dn_upper = doc_name.upper()
    company_ok = (company is None) or dn_upper.startswith(company + "_") or dn_upper.startswith(company)
    year_ok    = (not years) or (doc_year is not None and doc_year in years)

    if company_ok and year_ok:
        return 6.0          # exact company+year hit -- overwhelms vector score
    if company_ok and not years:
        return 4.0          # company-only match (question lacks a specific year)
    if company_ok:
        return 1.0          # right company, wrong year -- mild boost
    return 0.0


# Financial query expansion: maps concise question phrasing to the
# equivalent SEC/GAAP terminology found in 10-K tables.
# Metrics-generated questions use analyst shorthand; actual evidence
# text uses the full GAAP line-item names from the filing.
_QUERY_EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    # Capital expenditure / PP&E
    (re.compile(r"\bcap(?:ital)?\s*ex(?:penditures?)?\b", re.I),
     "capital expenditure capex purchases of property plant and equipment PP&E"),
    (re.compile(r"\bpurchases?\s+of\s+property\b", re.I),
     "purchases of property plant and equipment capital expenditure capex PP&E"),
    # Free cash flow
    (re.compile(r"\bfree\s+cash\s+flow\b", re.I),
     "free cash flow operating cash flow capital expenditures"),
    # EBITDA
    (re.compile(r"\bebitda\b", re.I),
     "EBITDA operating income earnings before interest taxes depreciation amortization"),
    # Net income / earnings
    (re.compile(r"\bnet\s+income\b", re.I),
     "net income net earnings consolidated net income"),
    (re.compile(r"\bearnings\s+per\s+share\b", re.I),
     "earnings per share EPS diluted net income per share"),
    # Revenue / sales
    (re.compile(r"\b(?:total\s+)?revenue\b", re.I),
     "revenue total revenue net revenue net sales total net sales"),
    (re.compile(r"\bnet\s+sales\b", re.I),
     "net sales revenue total revenue net revenue"),
    # Gross profit / margin
    (re.compile(r"\bgross\s+(?:profit|margin)\b", re.I),
     "gross profit gross margin net revenue cost of goods sold"),
    # Operating income / profit
    (re.compile(r"\boperating\s+(?:income|profit)\b", re.I),
     "operating income operating profit income from operations operating earnings"),
    # R&D
    (re.compile(r"\br\s*[&and]+\s*d\b", re.I),
     "research and development R&D research development expenses"),
    (re.compile(r"\bresearch\s+and\s+development\b", re.I),
     "research and development R&D expenses"),
    # Long-term debt
    (re.compile(r"\blong[\s-]term\s+debt\b", re.I),
     "long-term debt long term debt notes payable borrowings"),
    # Total assets / liabilities
    (re.compile(r"\btotal\s+assets\b", re.I),
     "total assets total consolidated assets balance sheet"),
    (re.compile(r"\btotal\s+liabilities\b", re.I),
     "total liabilities total debt obligations"),
    # Cash and equivalents
    (re.compile(r"\bcash\s+and\s+(?:cash\s+)?equivalents\b", re.I),
     "cash and cash equivalents cash equivalents short-term investments"),
    # Dividends
    (re.compile(r"\bdividend\b", re.I),
     "dividends per share cash dividends declared paid"),
    # Inventory
    (re.compile(r"\binventory(?:ies)?\b", re.I),
     "inventory inventories finished goods raw materials"),
    # Accounts receivable
    (re.compile(r"\baccounts?\s+receivable\b", re.I),
     "accounts receivable trade receivables net receivables"),
    # Depreciation / amortization
    (re.compile(r"\bdepreciation\b", re.I),
     "depreciation amortization depreciation and amortization"),
    # Tax rate / income tax
    (re.compile(r"\bincome\s+tax\b", re.I),
     "income tax provision income taxes effective tax rate"),
    (re.compile(r"\btax\s+rate\b", re.I),
     "effective tax rate income tax rate provision for income taxes"),
    # Shareholders equity
    (re.compile(r"\bshareholders?\s+equity\b", re.I),
     "shareholders equity stockholders equity total equity retained earnings"),
    # Operating cash flow
    (re.compile(r"\boperating\s+cash\s+flow\b", re.I),
     "cash flows from operating activities operating cash flow net cash provided"),
    # Share repurchase / buyback
    (re.compile(r"\b(?:share\s+)?buyback\b|\brepurchase\b", re.I),
     "repurchase shares buyback stock repurchase treasury stock"),
]


def expand_query(question: str) -> str:
    """Append SEC/GAAP synonym expansions for financial terms in the question.

    Each expansion appends additional terminology so that bge-large's dense
    vector better aligns the question embedding with the evidence passage
    embedding (which uses GAAP wording from the 10-K filing).
    """
    extra: list[str] = []
    seen: set[str] = set()
    for pattern, expansion in _QUERY_EXPANSIONS:
        if pattern.search(question) and expansion not in seen:
            extra.append(expansion)
            seen.add(expansion)
    if not extra:
        return question
    return question + " " + " ".join(extra)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--mnem-http", default="http://127.0.0.1:9876")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max questions to evaluate (default: all)")
    ap.add_argument("--rerank", type=str, default=None,
                    help="mnem reranker (e.g. 'onnx:ms-marco-MiniLM-L-6-v2')")
    ap.add_argument("--graph-expand", type=int, default=None)
    ap.add_argument("--vector-cap", type=int, default=10000)
    ap.add_argument("--hybrid-boost", action="store_true",
                    help="Bench-only keyword overlap post-filter. Boosts evidence nodes "
                         "whose text shares tokens with the question (company names, years). "
                         "Mirrors MemPalace hybrid_v4 harness helper. Not a mnem-core feature.")
    ap.add_argument("--hybrid-boost-weight", type=float, default=0.4)
    ap.add_argument("--query-expand", action="store_true",
                    help="Append SEC/GAAP synonym expansions for financial terms before "
                         "sending to the retriever. Bridges analyst shorthand ('capex', "
                         "'EBITDA') to 10-K filing language ('purchases of property, plant "
                         "and equipment'). Bench-harness only -- not a mnem-core feature.")
    ap.add_argument("--doc-filter", action="store_true",
                    help="Extract company name + fiscal year(s) from each question, then "
                         "boost candidates whose doc_name matches. With a 168-passage "
                         "closed corpus, this almost always selects the right document "
                         "before vector search ranks individual pages. Bench-harness only.")
    return ap.parse_args()


def hybrid_boost(
    query: str,
    candidates: list[dict[str, Any]],
    boost_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """Keyword overlap post-filter over mnem's top-K candidates.

    Critical for FinanceBench: financial tables are semantically identical
    across companies, so pure vector search retrieves wrong companies.
    Token overlap on company names ("3M", "Adobe") and years ("2018",
    "FY2022") re-ranks the correct document to the top.

    Bench-harness only -- not a mnem-core feature.
    """
    q_lower = query.lower()
    q_tokens = set(re.findall(r"\w+", q_lower))
    # Extra weight for company-name / fiscal-year tokens that vector search misses
    year_pattern = re.compile(r"\b(20\d{2}|19\d{2})\b")
    want_year = bool(year_pattern.search(q_lower))

    out: list[dict[str, Any]] = []
    for c in candidates:
        text = (c.get("summary") or c.get("rendered") or "").lower()
        d_tokens = set(re.findall(r"\w+", text))
        overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
        bonus = 0.0
        if want_year and year_pattern.search(text):
            bonus += 0.05
        c2 = dict(c)
        c2["dense_score"] = c.get("score", 0.0)
        c2["score"] = float(c.get("score", 0.0)) + boost_weight * overlap + bonus
        out.append(c2)
    out.sort(key=lambda x: -x["score"])
    return out


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
    print(f"loaded {len(records)} questions from {args.dataset}", file=sys.stderr)

    http = requests.Session()
    last_err: Exception | None = None
    ok = False
    for _attempt in range(10):
        try:
            h = http.get(f"{args.mnem_http}/v1/healthz", timeout=60).json()
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

    # ------------------------------------------------------------------
    # Build corpus: collect unique (doc_name, page_num) evidence passages
    # across the loaded records (--limit already applied above).
    # ------------------------------------------------------------------
    ev_key_to_text: dict[tuple[str, int], str] = {}
    for rec in records:
        for ev in rec.get("evidence") or []:
            key = (ev["doc_name"], int(ev["evidence_page_num"]))
            if key not in ev_key_to_text:
                ev_key_to_text[key] = (ev.get("evidence_text") or "").strip()

    ev_keys_ordered: list[tuple[str, int]] = list(ev_key_to_text.keys())
    nodes_body: list[dict[str, Any]] = [
        {
            "label": BENCH_LABEL,
            "summary": ev_key_to_text[key],
            "props": {"doc_name": key[0], "page_num": key[1]},
            "author": "financebench-harness",
        }
        for key in ev_keys_ordered
    ]

    print(
        f"ingesting {len(nodes_body)} unique evidence passages "
        f"(corpus from {len(records)} questions)...",
        file=sys.stderr,
    )
    ts = time.time()
    resp = http.post(
        f"{args.mnem_http}/v1/nodes/bulk",
        json={
            "nodes": nodes_body,
            "author": "financebench-harness",
            "message": "financebench corpus ingest",
            "auto_embed": True,
        },
        timeout=900,
    )
    resp.raise_for_status()
    ingest_results = resp.json().get("results", [])
    t_ingest = time.time() - ts
    print(f"ingest done in {t_ingest:.1f}s", file=sys.stderr)

    # UUID -> (doc_name, page_num) for scoring
    uuid_to_key: dict[str, tuple[str, int]] = {
        rv["id"]: key for key, rv in zip(ev_keys_ordered, ingest_results)
    }

    # ------------------------------------------------------------------
    # Wait for async embedding to complete before querying.
    # mnem's /v1/nodes/bulk returns immediately; ollama embeddings are
    # computed in a background task.  Querying before all 168 passages are
    # embedded yields an incomplete vector index (we observed 126/168 in
    # the first query when we started immediately).  Poll with a neutral
    # query until the full corpus is visible.
    # ------------------------------------------------------------------
    expected = len(nodes_body)
    probe_body = {
        "text": "revenue income assets balance sheet cash flow",
        "label": BENCH_LABEL,
        "limit": expected + 10,
        "vector_cap": args.vector_cap,
    }
    print(f"waiting for all {expected} passages to be embedded...", file=sys.stderr)
    t_wait_start = time.time()
    for _wait in range(600):       # up to 10 minutes
        probe = http.post(f"{args.mnem_http}/v1/retrieve", json=probe_body, timeout=60)
        got = len(probe.json().get("items", []))
        if got >= expected:
            break
        if _wait % 10 == 0:
            elapsed = time.time() - t_wait_start
            print(f"  embedded {got}/{expected} ({elapsed:.0f}s)...", file=sys.stderr)
        time.sleep(1)
    t_wait = time.time() - t_wait_start
    print(f"corpus fully embedded in {t_wait:.1f}s (total ingest+embed: {t_ingest+t_wait:.1f}s)",
          file=sys.stderr)

    # ------------------------------------------------------------------
    # Retrieve + score per question
    # ------------------------------------------------------------------
    totals_by_type: Counter[str] = Counter()
    hits1_by_type: Counter[str] = Counter()
    hits3_by_type: Counter[str] = Counter()
    hits5_by_type: Counter[str] = Counter()
    per_q: list[dict[str, Any]] = []

    t_retrieve = 0.0
    t_score = 0.0
    t0 = time.time()

    for rec in tqdm(records, desc="questions"):
        qid = rec["financebench_id"]
        qtype = rec.get("question_type", "unknown")
        question = rec["question"]

        target_keys: set[tuple[str, int]] = {
            (ev["doc_name"], int(ev["evidence_page_num"]))
            for ev in (rec.get("evidence") or [])
            if "doc_name" in ev and "evidence_page_num" in ev
        }

        query_text = expand_query(question) if args.query_expand else question

        body: dict[str, Any] = {
            "text": query_text,
            "label": BENCH_LABEL,
            "limit": max(500, args.top_k * 50),
            "vector_cap": args.vector_cap,
        }
        if args.rerank:
            body["rerank"] = args.rerank
        if args.graph_expand is not None:
            body["graph_expand"] = args.graph_expand

        ts = time.time()
        resp = http.post(f"{args.mnem_http}/v1/retrieve", json=body, timeout=300)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        t_retrieve += time.time() - ts

        ts = time.time()
        if args.hybrid_boost:
            items = hybrid_boost(question, items, boost_weight=args.hybrid_boost_weight)

        if args.doc_filter:
            company, years = _extract_doc_filter(question)
            if company or years:
                for it in items:
                    node_key = uuid_to_key.get(it["id"])
                    if node_key:
                        doc_boost = _doc_name_match_score(node_key[0], company, years)
                        it["score"] = float(it.get("score", 0.0)) + doc_boost
                items.sort(key=lambda x: -x.get("score", 0.0))

        ranked_keys: list[tuple[str, int]] = [
            uuid_to_key[it["id"]] for it in items if it["id"] in uuid_to_key
        ]

        hit1 = bool(target_keys) and any(k in target_keys for k in ranked_keys[:1])
        hit3 = bool(target_keys) and any(k in target_keys for k in ranked_keys[:3])
        hit5 = bool(target_keys) and any(k in target_keys for k in ranked_keys[:args.top_k])

        totals_by_type[qtype] += 1
        if hit1:
            hits1_by_type[qtype] += 1
        if hit3:
            hits3_by_type[qtype] += 1
        if hit5:
            hits5_by_type[qtype] += 1

        row: dict[str, Any] = {
            "financebench_id": qid,
            "question_type": qtype,
            "hit@1": hit1,
            "hit@3": hit3,
            f"hit@{args.top_k}": hit5,
            "target": [list(k) for k in target_keys],
            "top5_retrieved": [list(k) for k in ranked_keys[:5]],
        }
        if args.query_expand and query_text != question:
            row["expanded_query"] = query_text
        per_q.append(row)
        t_score += time.time() - ts

    dt = time.time() - t0
    total = sum(totals_by_type.values())
    r1 = sum(hits1_by_type.values()) / total if total else 0.0
    r3 = sum(hits3_by_type.values()) / total if total else 0.0
    r5 = sum(hits5_by_type.values()) / total if total else 0.0

    report: dict[str, Any] = {
        "harness": "mnem-financebench",
        "mnem_http": args.mnem_http,
        "dataset": str(args.dataset),
        "corpus_size": len(nodes_body),
        "n_questions": total,
        "runtime_seconds": round(dt, 1),
        "timing": {
            "ingest_s": round(t_ingest, 3),
            "retrieve_s": round(t_retrieve, 3),
            "score_s": round(t_score, 3),
        },
        "overall": {
            "hit@1": round(r1, 4),
            "hit@3": round(r3, 4),
            f"hit@{args.top_k}": round(r5, 4),
        },
        "by_type": {
            qt: {
                "n": totals_by_type[qt],
                "hit@1": round(hits1_by_type[qt] / totals_by_type[qt], 4),
                "hit@3": round(hits3_by_type[qt] / totals_by_type[qt], 4),
                f"hit@{args.top_k}": round(hits5_by_type[qt] / totals_by_type[qt], 4),
            }
            for qt in sorted(totals_by_type)
        },
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log = args.out.with_suffix(".jsonl")
    with log.open("w", encoding="utf-8") as fh:
        for row in per_q:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(
        f"=== mnem FinanceBench  "
        f"hit@1={r1:.4f}  hit@3={r3:.4f}  hit@{args.top_k}={r5:.4f}  "
        f"n={total}  corpus={len(nodes_body)}"
    )
    for qt in sorted(totals_by_type):
        n = totals_by_type[qt]
        print(
            f"  {qt:30s}  n={n:3d}  "
            f"hit@1={hits1_by_type[qt]/n:.4f}  "
            f"hit@3={hits3_by_type[qt]/n:.4f}  "
            f"hit@{args.top_k}={hits5_by_type[qt]/n:.4f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
