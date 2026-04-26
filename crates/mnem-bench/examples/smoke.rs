//! LongMemEval canary against the in-process mnem adapter.
//!
//! Defaults to a 5-question slice using the real
//! `all-MiniLM-L6-v2` ONNX embedder via `mnem-embed-providers`
//! (gated on the default-on `onnx-minilm` feature). The 5q slice is
//! the 0.1.0 pre-flight gate; pass `--limit 50` (or any N) to
//! validate the headline 50q canary without needing the full
//! `mnem bench` runner.
//!
//! ## Flags
//!
//! - `--limit N` - cap the question set at `N` (default 5). The
//!   slice always comes from the head of the dataset so reruns are
//!   reproducible.
//! - `--bag-of-tokens` - force the toy hashed embedder. Useful for
//!   confirming a regression is in the embedder path vs the
//!   retriever path.
//!
//! ## Dataset resolution
//!
//! 1. `MNEM_BENCH_DATA` env var, expecting
//!    `<dir>/longmemeval/longmemeval_s_cleaned.json` underneath.
//! 2. The standard cache `~/.mnem/bench-data/longmemeval/...`.
//! 3. If none exist, falls back to a tiny synthetic 5-question
//!    fixture so the gate still runs end-to-end.
//!
//! ## Pre-flight gate
//!
//! Exits non-zero if `recall@5 < 0.6` on a real-dataset run with
//! the ONNX embedder; exits non-zero if `recall@5 == 0` on the
//! synthetic fixture or with the bag-of-tokens flavour. The cutoffs
//! mirror the 0.1.0 release-readiness checklist.

use std::path::PathBuf;

use anyhow::Result;
use mnem_bench::adapters::MnemAdapter;
use mnem_bench::datasets::longmemeval::{self, Question, Turn};
use mnem_bench::embed::{BenchEmbedder, DEFAULT_DIM};
use mnem_bench::score::longmemeval as scorer;

fn main() -> Result<()> {
    let opts = parse_args()?;

    let path = resolve_dataset();
    let mut questions: Vec<Question> = match path.as_ref().filter(|p| p.is_file()) {
        Some(p) => {
            eprintln!("[smoke] using dataset at {}", p.display());
            longmemeval::load(p)?
        }
        None => {
            eprintln!("[smoke] no LongMemEval dataset on disk; falling back to synthetic fixture.");
            synthetic_questions()
        }
    };
    if questions.len() > opts.limit {
        questions.truncate(opts.limit);
    }
    if questions.is_empty() {
        anyhow::bail!("no questions to score");
    }

    let (embedder, used_real_minilm) = build_embedder(opts.bag_of_tokens)?;
    eprintln!(
        "[smoke] embedder = {} (dim {})",
        embedder.model(),
        embedder.dim()
    );
    let mut adapter = MnemAdapter::with_embedder(embedder)
        .map_err(|e| anyhow::anyhow!("constructing mnem adapter: {e}"))?;
    let dataset_label = path
        .clone()
        .unwrap_or_else(|| PathBuf::from("synthetic://longmemeval-5q"));
    let (report, rows) = scorer::run(&mut adapter, &questions, 10, &dataset_label)?;

    let r5 = report.overall.get("recall@5").copied().unwrap_or(0.0);
    let r10 = report.overall.get("recall@10").copied().unwrap_or(0.0);

    eprintln!();
    eprintln!(
        "=== mnem-bench smoke (LongMemEval, {}q) ===",
        report.n_questions
    );
    eprintln!("n_questions        : {}", report.n_questions);
    eprintln!("recall@5           : {r5:.4}");
    eprintln!("recall@10          : {r10:.4}");
    eprintln!("runtime_seconds    : {:.2}", report.runtime_seconds);
    eprintln!("ingest_s           : {:.2}", report.timing.ingest_s);
    eprintln!("retrieve_s         : {:.2}", report.timing.retrieve_s);
    eprintln!("score_s            : {:.2}", report.timing.score_s);
    eprintln!("rows               : {}", rows.len());
    println!("{}", serde_json::to_string_pretty(&report)?);

    // Gate selection:
    //   - real ONNX MiniLM on a real dataset slice -> recall@5 >= 0.6
    //   - everything else (synthetic fixture or toy embedder)
    //     -> recall@5 > 0
    let on_real_dataset = path.is_some();
    if used_real_minilm && on_real_dataset {
        if r5 < 0.6 {
            anyhow::bail!(
                "smoke FAILED: recall@5 = {r5:.4} (must be >= 0.6 with real MiniLM \
                 on a real-dataset slice). Embedder/retriever pipeline regressed."
            );
        }
        eprintln!("\n[smoke] PASS - recall@5 = {r5:.4} >= 0.6 (real MiniLM gate).");
    } else {
        if r5 <= 0.0 {
            anyhow::bail!(
                "smoke FAILED: recall@5 = {r5:.4} (must be > 0). \
                 Investigate the embedder + retriever pipeline before pushing."
            );
        }
        eprintln!("\n[smoke] PASS - recall@5 = {r5:.4} > 0 (fallback gate).");
    }
    Ok(())
}

struct Opts {
    limit: usize,
    bag_of_tokens: bool,
}

fn parse_args() -> Result<Opts> {
    // Tiny hand-rolled parser. Avoids a clap dep on what is a smoke
    // example. Recognises:
    //   --limit N
    //   --limit=N
    //   --bag-of-tokens
    //   -h / --help
    let mut limit: usize = 5;
    let mut bag_of_tokens = false;
    let mut args = std::env::args().skip(1).peekable();
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                eprintln!("Usage: smoke [--limit N] [--bag-of-tokens]");
                std::process::exit(0);
            }
            "--bag-of-tokens" => bag_of_tokens = true,
            "--limit" => {
                let v = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--limit requires a value"))?;
                limit = v.parse().map_err(|e| anyhow::anyhow!("--limit {v}: {e}"))?;
            }
            s if s.starts_with("--limit=") => {
                let v = &s["--limit=".len()..];
                limit = v.parse().map_err(|e| anyhow::anyhow!("--limit {v}: {e}"))?;
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    if limit == 0 {
        anyhow::bail!("--limit must be > 0");
    }
    Ok(Opts { limit, bag_of_tokens })
}

/// Construct the embedder for this run. Returns the embedder + a
/// flag indicating whether the real ONNX MiniLM was used (so the
/// gate-selection logic can pick the right cutoff).
fn build_embedder(force_bag: bool) -> Result<(BenchEmbedder, bool)> {
    if force_bag {
        return Ok((BenchEmbedder::bag_of_tokens(DEFAULT_DIM), false));
    }
    #[cfg(feature = "onnx-minilm")]
    {
        match BenchEmbedder::onnx_minilm() {
            Ok(e) => Ok((e, true)),
            Err(e) => {
                eprintln!(
                    "[smoke] WARNING: onnx-minilm init failed ({e}); \
                     falling back to bag-of-tokens."
                );
                Ok((BenchEmbedder::bag_of_tokens(DEFAULT_DIM), false))
            }
        }
    }
    #[cfg(not(feature = "onnx-minilm"))]
    {
        eprintln!(
            "[smoke] built without `onnx-minilm` feature; using bag-of-tokens. \
             Rebuild with `cargo run --example smoke -p mnem-bench --release \
             --features onnx-minilm` to exercise the real embedder."
        );
        Ok((BenchEmbedder::bag_of_tokens(DEFAULT_DIM), false))
    }
}

fn resolve_dataset() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MNEM_BENCH_DATA") {
        let cand = PathBuf::from(p)
            .join("longmemeval")
            .join("longmemeval_s_cleaned.json");
        if cand.is_file() {
            return Some(cand);
        }
    }
    let lab = PathBuf::from("./datasets/longmemeval/longmemeval_s_cleaned.json");
    if lab.is_file() {
        return Some(lab);
    }
    if let Ok(p) = mnem_bench::datasets::cached_path(mnem_bench::Bench::LongMemEval) {
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Tiny synthetic LongMemEval fixture for environments without the
/// real dataset on disk (CI containers, fresh contributor laptops).
/// Each question's gold session is the obvious lexical match, so
/// even the bag-of-tokens embedder achieves recall@5 = 1.0 here.
fn synthetic_questions() -> Vec<Question> {
    fn turn(content: &str) -> Turn {
        Turn { role: "user".to_string(), content: content.to_string() }
    }
    fn q(qid: &str, qtype: &str, question: &str, gold: &str, sessions: Vec<(&str, &str)>) -> Question {
        let session_ids: Vec<String> = sessions.iter().map(|(s, _)| (*s).to_string()).collect();
        let session_turns: Vec<Vec<Turn>> = sessions
            .iter()
            .map(|(_, txt)| vec![turn(txt)])
            .collect();
        Question {
            question_id: qid.to_string(),
            question_type: Some(qtype.to_string()),
            question: question.to_string(),
            answer_session_ids: vec![gold.to_string()],
            haystack_session_ids: session_ids,
            haystack_sessions: session_turns,
        }
    }
    vec![
        q(
            "syn1",
            "single-session",
            "Where does Alice go climbing on the weekend?",
            "s_alice_berlin",
            vec![
                ("s_alice_berlin", "Alice goes climbing in Berlin every weekend with friends."),
                ("s_bob_paris", "Bob loves Paris and visits the Eiffel Tower."),
                ("s_movies", "Yesterday I watched a great science fiction movie."),
                ("s_cooking", "My grandmother taught me how to bake apple pie."),
                ("s_garden", "The garden is full of red roses in summer."),
            ],
        ),
        q(
            "syn2",
            "single-session",
            "What did Bob visit in Paris?",
            "s_bob_paris",
            vec![
                ("s_alice_berlin", "Alice goes climbing in Berlin every weekend."),
                ("s_bob_paris", "Bob loves Paris and visits the Eiffel Tower every spring."),
                ("s_movies", "Yesterday I watched a great science fiction movie."),
                ("s_cooking", "My grandmother taught me how to bake apple pie."),
                ("s_garden", "The garden is full of red roses in summer."),
            ],
        ),
        q(
            "syn3",
            "single-session",
            "What kind of movie did the user watch yesterday?",
            "s_movies",
            vec![
                ("s_alice_berlin", "Alice goes climbing in Berlin every weekend."),
                ("s_bob_paris", "Bob loves Paris and visits the Eiffel Tower."),
                ("s_movies", "Yesterday I watched a great science fiction movie about robots."),
                ("s_cooking", "My grandmother taught me how to bake apple pie."),
                ("s_garden", "The garden is full of red roses in summer."),
            ],
        ),
        q(
            "syn4",
            "single-session",
            "What did the grandmother teach about baking?",
            "s_cooking",
            vec![
                ("s_alice_berlin", "Alice goes climbing in Berlin every weekend."),
                ("s_bob_paris", "Bob loves Paris and visits the Eiffel Tower."),
                ("s_movies", "Yesterday I watched a great science fiction movie."),
                ("s_cooking", "My grandmother taught me how to bake apple pie from scratch."),
                ("s_garden", "The garden is full of red roses in summer."),
            ],
        ),
        q(
            "syn5",
            "single-session",
            "What grows in the garden in summer?",
            "s_garden",
            vec![
                ("s_alice_berlin", "Alice goes climbing in Berlin every weekend."),
                ("s_bob_paris", "Bob loves Paris and visits the Eiffel Tower."),
                ("s_movies", "Yesterday I watched a great science fiction movie."),
                ("s_cooking", "My grandmother taught me how to bake apple pie."),
                ("s_garden", "The garden is full of beautiful red roses in summer."),
            ],
        ),
    ]
}
