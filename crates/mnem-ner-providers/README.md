# mnem-ner-providers

NER provider adapters for mnem. Ships `RuleNer` (heuristic, zero-dependency) and `NullNer`.

Provides implementations of the `Ner` trait used by `mnem-ingest` during entity extraction.
`RuleNer` applies a fast regex-based rule set to tag person, organization, and location spans.
`NullNer` is a no-op pass-through used when entity extraction is disabled.

Part of the [mnem](https://github.com/Uranid/mnem) workspace: git for knowledge graphs.
