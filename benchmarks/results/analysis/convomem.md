# ConvoMem

5 categories × 50 items = 250 items total. Salesforce/ConvoMem dataset.
Substring-match recall (gold evidence text in retrieved item text).

## Summary

| System | avg_recall | Source |
|--------|-----------:|--------|
| MemPalace | 0.929 | reproduced under our harness |
| **mnem** | **0.976** | `v0.1.0/jsonl/convomem-250.jsonl` |

mnem +0.047 over MemPalace.

## Per-category

| Category | mnem |
|----------|-----------:|
| `assistant_facts_evidence` | 0.93 |
| `implicit_connection_evidence` | 0.92 |
| `preference_evidence` | 0.99 |
| `user_evidence` | 0.99 |
| `abstention_evidence` | 1.00 |

Range: 0.92 to 1.00. Lowest category is `implicit_connection_evidence`,
where item-text overlap with the gold reference is semantic rather than
literal.

## Configuration

- Embedder: ONNX MiniLM-L6-v2 (bundled, in-process)
- Retrieve: dense only (vector + top-10)
- `--limit 50` per category, default 5 categories -> 250 items
- `MNEM_BENCH=1` for per-item label scoping
- 4 cores, `MNEM_ORT_INTRA_THREADS=4`

## Reproduce

```bash
docker compose -f benchmarks/harness/compose.yml up -d mnem-bench-1

PYTHONUTF8=1 python benchmarks/harness/adapters/convomem.py \
    --mnem http http://127.0.0.1:9876 \
    --limit 50 --top-k 10 \
    --out benchmarks/results/v0.1.0/json/convomem-250.json

docker compose -f benchmarks/harness/compose.yml down
```

Expected: `avg_recall = 0.976` (within +/- 0.005 sample variance).

## Artifacts

| File | Description |
|------|-------------|
| `v0.1.0/json/convomem-250.json` | summary: overall + per-category avg recall |
| `v0.1.0/jsonl/convomem-250.jsonl` | per-item rows: gold IDs, retrieved IDs, per-item recall, category tag |

## Methodology notes

- Substring match in either direction (gold-in-retrieved or retrieved-in-gold)
  per ConvoMem evaluation harness.
- Per-item label scoping isolates each evidence question's candidate pool;
  cross-question leak suppressed.
- No LLM involved in scoring or retrieval.
