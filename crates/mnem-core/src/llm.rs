// HyDE, OpenAI, Ollama, BEIR, LangChain, Anthropic are well-known
// external identifiers; backticking every mention in the module doc
// and field docs would degrade readability in rendered rustdoc for
// no information gain.
#![allow(clippy::doc_markdown)]

//! Text-generation trait for HyDE and multi-query retrieval .
//!
//! # Why
//!
//! Two high-ROI retrieval techniques need an LLM in the read path, but
//! mnem-core is WASM-clean and tokio-free, so the HTTP call can't live
//! here:
//!
//! - **HyDE** (Gao et al. 2022, [arXiv:2212.10496]): instead of
//!   embedding the raw query, ask an LLM to generate a hypothetical
//!   answer and embed THAT. The encoder acts as a lossy compressor
//!   that filters hallucinated specifics back to real-corpus
//!   neighborhoods. On BEIR, HyDE beats plain Contriever on every
//!   dataset we care about.
//! - **Multi-query / RAG-Fusion** (Raudaschl 2023): ask the LLM to
//!   generate N paraphrases of the query, retrieve top-K for each, and
//!   fuse with RRF. Particularly strong when the user's phrasing is
//!   sharply different from stored phrasing.
//!
//! Both techniques share the same primitive: `(prompt) -> completion`.
//! The [`TextGenerator`] trait is that primitive. Adapter crates
//! (OpenAI chat completions, Ollama chat, Anthropic messages, local
//! llama.cpp) live outside `mnem-core` so the core stays tokio-free.
//!
//! # What this module provides
//!
//! A [`TextGenerator`] trait that adapter crates implement, plus a
//! [`LlmError`] surface and a deterministic mock for tests.
//!
//! # How it plugs in today
//!
//! The trait + adapters live here and in `mnem-llm-providers`. HyDE is
//! wired in the CLI layer (`mnem retrieve --hyde`) rather than as a
//! `Retriever` builder method: the LLM call produces a hypothetical
//! passage and the CLI feeds the passage into the embedder input. The
//! symmetric multi-query variant (generate N variations, retrieve
//! each, RRF-fuse) is planned. On LLM failure, the graceful-degrade
//! policy is the same as the rerank pass: fall back to the plain
//! query.
//!
//! [arXiv:2212.10496]: https://arxiv.org/abs/2212.10496

use std::fmt::Debug;

use thiserror::Error;

/// Error surface for text-generation adapters.
///
/// Marked `#[non_exhaustive]` so provider crates can grow their own
/// failure modes without a breaking change here.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LlmError {
    /// TLS / TCP / DNS / timeout failure reaching the provider.
    #[error("network error: {0}")]
    Network(String),
    /// Provider rejected credentials.
    #[error("authentication failed: {0}")]
    Auth(String),
    /// Provider rate-limited the request.
    #[error("rate limited: {0}")]
    RateLimited(String),
    /// 4xx from the provider.
    #[error("bad request ({status}): {body}")]
    BadRequest {
        /// HTTP status code.
        status: u16,
        /// Response body or best-effort error string.
        body: String,
    },
    /// 5xx from the provider.
    #[error("server error ({status}): {body}")]
    Server {
        /// HTTP status code.
        status: u16,
        /// Response body or best-effort error string.
        body: String,
    },
    /// Response decoder failed (malformed JSON, missing content field, ...).
    #[error("decode error: {0}")]
    Decode(String),
    /// Adapter config invalid (bad URL, missing env var, etc.).
    #[error("config error: {0}")]
    Config(String),
    /// Provider returned an empty completion.
    #[error("empty completion")]
    EmptyCompletion,
}

/// Generation options that the caller supplies per request. Kept
/// provider-agnostic; adapters map these onto their own APIs and
/// ignore fields they don't support (all adapters MUST tolerate a
/// `None` on every optional field without erroring).
#[derive(Debug, Clone)]
pub struct GenOptions {
    /// Maximum tokens in the completion. `None` means adapter default.
    pub max_tokens: Option<u32>,
    /// Sampling temperature. `None` means adapter default.
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability cutoff. `None` means adapter
    /// default. OpenAI and most providers accept `top_p`; Ollama maps
    /// it to its `options.top_p`.
    pub top_p: Option<f32>,
    /// Top-K sampling. `None` means adapter default. Not supported by
    /// every provider (OpenAI v1 chat does NOT accept top_k; Ollama
    /// does via `options.top_k`). Adapters that can't honour it
    /// silently drop the field.
    pub top_k: Option<u32>,
    /// Stop sequences. Completion halts when any of these strings
    /// appears. Most providers accept 1-4 stop strings. `None` or
    /// empty vec means no stop.
    pub stop: Option<Vec<String>>,
    /// Presence-penalty (discourage repeating tokens). OpenAI-family.
    pub presence_penalty: Option<f32>,
    /// Frequency-penalty (discourage high-frequency tokens).
    /// OpenAI-family.
    pub frequency_penalty: Option<f32>,
    /// Deterministic-sampling seed. OpenAI accepts a seed on the
    /// chat-completions endpoint; Ollama accepts `options.seed`.
    /// Useful for reproducing HyDE runs in benchmarks.
    pub seed: Option<u64>,
    /// Number of completions to sample. For multi-query this is the
    /// number of paraphrases; for HyDE this is usually 1 (or small-N
    /// averaged, per the paper).
    pub n: usize,
    /// Optional system prompt / role preamble.
    pub system: Option<String>,
}

impl GenOptions {
    /// Construct options with `n = 1` and everything else None.
    /// Equivalent to `GenOptions::default()`; provided for
    /// self-documentation at call sites ("I only want 1 completion").
    #[must_use]
    pub fn single() -> Self {
        Self::default()
    }
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            n: 1,
            system: None,
        }
    }
}

/// Text-generation primitive: given a user prompt (and optional system
/// preamble), return one or more completions.
///
/// The returned `Vec<String>` has length exactly `opts.n`; adapters
/// that only support `n=1` natively MUST batch-call n times and
/// surface a coherent error if one sub-call fails. Completion content
/// is implementation-defined: callers who need structure should
/// post-parse (e.g. split on newlines for multi-query).
pub trait TextGenerator: Send + Sync + Debug {
    /// Provider + model identifier. Lowercase, colon-separated by
    /// convention (e.g. `"openai:gpt-4o-mini"`, `"ollama:llama3.2:3b"`).
    fn model(&self) -> &str;

    /// Generate completions for `prompt`.
    ///
    /// # Errors
    ///
    /// Any [`LlmError`] the adapter surfaces. Callers that use this
    /// for HyDE / multi-query SHOULD fall back gracefully to the plain
    /// query on error (same policy as the reranker fallback), so an
    /// LLM outage does not break retrieval.
    fn generate(&self, prompt: &str, opts: &GenOptions) -> Result<Vec<String>, LlmError>;
}

/// Deterministic test-only generator. Returns a configured response
/// regardless of input. Useful for wiring HyDE / multi-query tests
/// without a live provider.
#[derive(Debug, Clone)]
pub struct MockTextGenerator {
    model: String,
    /// Fixed completions to return on every call. If `opts.n` is
    /// larger than `completions.len()`, the last one is repeated.
    completions: Vec<String>,
}

impl MockTextGenerator {
    /// Construct a mock with the given `(model, completions)`.
    #[must_use]
    pub fn new(model: impl Into<String>, completions: Vec<String>) -> Self {
        Self {
            model: model.into(),
            completions,
        }
    }
}

impl Default for MockTextGenerator {
    fn default() -> Self {
        Self::new("mock:echo", vec!["(mock completion)".to_string()])
    }
}

impl TextGenerator for MockTextGenerator {
    fn model(&self) -> &str {
        &self.model
    }

    fn generate(&self, _prompt: &str, opts: &GenOptions) -> Result<Vec<String>, LlmError> {
        if self.completions.is_empty() {
            return Err(LlmError::EmptyCompletion);
        }
        let mut out = Vec::with_capacity(opts.n);
        for i in 0..opts.n {
            let idx = i.min(self.completions.len() - 1);
            out.push(self.completions[idx].clone());
        }
        Ok(out)
    }
}

/// Test-only generator that always errors. Proves the graceful
/// fallback path in HyDE / multi-query callers.
#[derive(Debug, Clone, Default)]
pub struct AlwaysFailGenerator;

impl TextGenerator for AlwaysFailGenerator {
    fn model(&self) -> &str {
        "mock:always-fail"
    }

    fn generate(&self, _prompt: &str, _opts: &GenOptions) -> Result<Vec<String>, LlmError> {
        Err(LlmError::Network(
            "intentional failure for test".to_string(),
        ))
    }
}

/// Default HyDE prompt template for short-fact agent memory.
///
/// Tuned for mnem's typical payload: short node summaries, concrete
/// entities/relations/attributes. Avoids the LangChain BEIR-task-tuned
/// templates because mnem nodes are not BEIR docs.
pub const HYDE_PROMPT_TEMPLATE: &str =
    "Write a short, factual passage (2-4 sentences) that would answer the \
question below. Focus on concrete entities, relations, and attributes \
a note-taking system might have recorded. Omit hedging and meta-talk.

Question: {query}
Passage:";

/// Default multi-query prompt template, parameterised by `{n}` and
/// `{query}`. Both placeholders are replaced at call time via
/// [`fill_multi_query_template`]. The listed angles (broader,
/// narrower, synonymous, entity-centric) are suggestions; when
/// `n > 4` the model is expected to mix and extend them.
pub const MULTI_QUERY_PROMPT_TEMPLATE: &str =
    "You are rewriting a user's query into search variations for a personal \
knowledge graph. Generate {n} alternative queries that together cover \
different angles of the same intent. Suggested angles:
  - a broader/more abstract phrasing
  - a narrower/more specific phrasing
  - a synonymous rephrasing using different key terms
  - an entity-centric phrasing (noun-heavy, no verbs)

Do NOT output minor rewordings. Do NOT repeat the original query.
Output exactly {n} lines, one query per line, no numbering.

Original: {query}";

/// Fill `{query}` in a template with the user's query string.
///
/// The substitution is naive: the literal substring `{query}` is
/// replaced. Templates with more sophisticated placeholders should
/// use a real templating engine; HyDE and multi-query only need this.
#[must_use]
pub fn fill_template(template: &str, query: &str) -> String {
    template.replace("{query}", query)
}

/// Fill both `{query}` and `{n}` placeholders in a multi-query
/// template. Use this when generating N paraphrase variants so the
/// prompt honours the caller's requested count.
#[must_use]
pub fn fill_multi_query_template(template: &str, query: &str, n: usize) -> String {
    template
        .replace("{query}", query)
        .replace("{n}", &n.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_generates_n_completions() {
        let g = MockTextGenerator::new(
            "mock:echo",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );
        let out = g
            .generate(
                "ignored",
                &GenOptions {
                    n: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out, vec!["a", "b", "c"]);
    }

    #[test]
    fn mock_repeats_last_when_n_exceeds_completion_count() {
        let g = MockTextGenerator::new("mock:echo", vec!["only".to_string()]);
        let out = g
            .generate(
                "ignored",
                &GenOptions {
                    n: 4,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out, vec!["only"; 4]);
    }

    #[test]
    fn mock_empty_completions_errors() {
        let g = MockTextGenerator::new("mock:echo", vec![]);
        let e = g.generate("q", &GenOptions::default()).unwrap_err();
        assert!(matches!(e, LlmError::EmptyCompletion));
    }

    #[test]
    fn always_fail_generator_errors() {
        let g = AlwaysFailGenerator;
        assert!(g.generate("q", &GenOptions::default()).is_err());
    }

    #[test]
    fn model_id_has_provider_prefix() {
        let g = MockTextGenerator::default();
        assert!(g.model().contains(':'));
    }

    #[test]
    fn fill_template_substitutes_query() {
        let s = fill_template("ask {query} now", "why");
        assert_eq!(s, "ask why now");
    }

    #[test]
    fn fill_template_leaves_text_without_placeholder_alone() {
        let s = fill_template("no placeholder here", "ignored");
        assert_eq!(s, "no placeholder here");
    }

    #[test]
    fn default_hyde_prompt_has_query_placeholder() {
        assert!(HYDE_PROMPT_TEMPLATE.contains("{query}"));
    }

    #[test]
    fn default_multi_query_prompt_has_query_placeholder() {
        assert!(MULTI_QUERY_PROMPT_TEMPLATE.contains("{query}"));
    }

    #[test]
    fn gen_options_default_is_n1() {
        let o = GenOptions::default();
        assert_eq!(o.n, 1);
        assert!(o.temperature.is_none());
        assert!(o.max_tokens.is_none());
        assert!(o.system.is_none());
    }
}
