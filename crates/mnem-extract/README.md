# mnem-extract

Extraction strategies for mnem ingest pipelines.

Provides keyword extraction (KeyBERT-style cooccurrence) and LLM-assisted
extraction traits used by `mnem-ingest`. Trust filters live here too:
each extracted entity carries provenance + a confidence score so the
ingest layer can drop low-trust noise before it reaches the graph.

Part of [mnem](https://github.com/Uranid/mnem): versioned, mergeable,
content-addressed knowledge graph for AI agent memory.

## License

Apache-2.0.
