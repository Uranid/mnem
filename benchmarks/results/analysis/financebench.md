# FinanceBench

150 questions, 168 unique evidence passages from `financebench_open_source.jsonl`
(Patronus AI 2024 open-source split, SEC 10-K/10-Q filings).

Closed-haystack document retrieval: ingest all 168 passages as a corpus, then for
each question measure whether the correct `(doc_name, page_num)` passage appears in
the top-K results. Metric: hit@1, hit@3, hit@5. Three question types:
`metrics-generated`, `domain-relevant`, `novel-generated` (50 questions each).

## Summary

| System | hit@1 | hit@3 | hit@5 | Source |
|--------|------:|------:|------:|--------|
| mem0 | 0.007 | 0.020 | 0.033 | reproduced under our harness |
| MemPalace | 0.493 | 0.673 | 0.767 | reproduced under our harness |
| **mnem** | **0.673** | **0.887** | **0.973** | `v0.1.0/jsonl/financebench-bge-large-full.jsonl` |

MemPalace shown at best configuration (bge-large, see below). All systems use
Ollama bge-large (1024-dim) embeddings; this is not the MiniLM baseline used
in other benchmarks.

### mnem by question type

| Type | n | hit@1 | hit@3 | hit@5 |
|------|--:|------:|------:|------:|
| metrics-generated | 50 | 0.920 | 1.000 | 1.000 |
| novel-generated | 50 | 0.580 | 0.880 | 0.960 |
| domain-relevant | 50 | 0.520 | 0.780 | 0.960 |

## System notes

### mem0

mem0 runs an LLM extraction step (llama3.2:3b via Ollama) on every ingested passage
before storage. The extracted summary replaces the original text. Specific financial
figures - revenue, EPS, ratios - are paraphrased or dropped during extraction, making
both keyword and vector retrieval ineffective for precise financial Q&A.

BM25 reranking on the extracted memory text was also tested; hit@5 was unchanged at
0.033. The ceiling is set by the extraction step, not the retrieval strategy.

Adapter: `benchmarks/harness/adapters/financebench_mem0.py`
Result: `v0.1.0/json/financebench-mem0.json`

### MemPalace

MemPalace hardcodes ONNX MiniLM (all-MiniLM-L6-v2, 384-dim) with no public API to
swap models. Two configurations were tested:

| Configuration | hit@5 | Embedder | Retrieval |
|---------------|------:|----------|-----------|
| Default | 0.627 | FastEmbed bge-small 384-dim | hybrid BM25+vector |
| bge-large | 0.767 | Ollama bge-large 1024-dim | pure vector |

The bge-large run bypasses MemPalace's embedding factory by creating a ChromaDB
collection directly via `OllamaEmbeddingFunction`. BM25 post-hoc reranking on the
bge-large results was also tested and hurt retrieval (hit@5 fell to 0.467): financial
queries ask "what was revenue?" but passages contain the bare number, so lexical overlap
between query and passage is low. Pure vector wins here; best result is 0.767.

Adapter: `benchmarks/harness/adapters/financebench_mempalace_bgelarge.py` (bge-large)
         `benchmarks/harness/adapters/financebench_mempalace.py` (default bge-small)
Results: `v0.1.0/json/financebench-mempalace-bgelarge.json`, `v0.1.0/json/financebench-mempalace.json`

## Configuration (mnem)

- Embedder: Ollama bge-large (1024-dim)
- Retrieve: hybrid (vector + BM25 boost) + doc-filter (company/filing scoping)
- `--hybrid-boost --query-expand --top-k 5`
- 168 passages ingested in 10.4 s; 150 questions retrieved in 313 s

## Reproduce

```bash
# Prerequisites: Ollama running with bge-large
ollama pull bge-large

# mnem result (requires mnem server at :9876)
python benchmarks/harness/adapters/financebench.py \
    --dataset datasets/financebench/financebench_open_source.jsonl \
    --mnem-http http://127.0.0.1:9876 \
    --hybrid-boost --query-expand \
    --out benchmarks/results/v0.1.0/json/financebench-bge-large-full.json

# MemPalace bge-large
python benchmarks/harness/adapters/financebench_mempalace_bgelarge.py \
    --dataset datasets/financebench/financebench_open_source.jsonl \
    --out benchmarks/results/v0.1.0/json/financebench-mempalace-bgelarge.json

# mem0 (also needs Ollama LLM; ingest takes ~90 min)
python benchmarks/harness/adapters/financebench_mem0.py \
    --dataset datasets/financebench/financebench_open_source.jsonl \
    --llm-model llama3.2:3b \
    --out benchmarks/results/v0.1.0/json/financebench-mem0.json
```

Expected: mnem `hit@5 = 0.9733` (within +/-0.005 sample variance).

## Latency

| Metric | Value |
|--------|------:|
| Retrieve mean (mnem) | 2087 ms / question |
| Total wall - ingest | 10.4 s |
| Total wall - retrieve (150 Q) | 313 s (~5 min) |

Higher per-question latency than the conversational benchmarks (333-711 ms) because
FinanceBench uses a corpus-wide scan with no per-session label scope. Every query
searches all 168 passages rather than a scoped subset.

## Artifacts

| File | Description |
|------|-------------|
| `v0.1.0/json/financebench-bge-large-full.json` | mnem summary: overall + per-question-type hit@1/3/5 |
| `v0.1.0/jsonl/financebench-bge-large-full.jsonl` | per-question rows: qid, hit@1/3/5, target, top5_retrieved |
| `v0.1.0/json/financebench-mempalace-bgelarge.json` | MemPalace bge-large (best) summary |
| `v0.1.0/jsonl/financebench-mempalace-bgelarge.jsonl` | per-question rows |
| `v0.1.0/json/financebench-mempalace.json` | MemPalace default bge-small summary |
| `v0.1.0/jsonl/financebench-mempalace.jsonl` | per-question rows |
| `v0.1.0/json/financebench-mem0.json` | mem0 + BM25 rerank summary |
| `v0.1.0/jsonl/financebench-mem0.jsonl` | per-question rows |
