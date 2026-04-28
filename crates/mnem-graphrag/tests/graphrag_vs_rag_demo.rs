//! End-to-end demonstration that contrasts vector-only retrieval ("vanilla
//! RAG") with community-aware retrieval ("graphRAG") on a controlled
//! polysemy dataset.
//!
//! The test is hermetic (no network, no LLM, no real embedding model): a
//! deterministic keyword-bag embedder produces vectors with intentional
//! semantic overlap so that vector cosine alone confuses two distinct
//! topics. Leiden community detection on the authored graph separates
//! them. We print:
//!
//!   1. Leiden output (community assignments, modularity, count).
//!   2. Per-query top-k for vector-only vs community-filtered retrieval.
//!   3. Precision@k against ground truth.
//!   4. Centroid+MMR community summaries (mnem-graphrag::summarize).
//!
//! Run with:
//!
//!   cargo test --release -p mnem-graphrag --test graphrag_vs_rag_demo \
//!       -- --nocapture
//!
//! The asserts at the bottom are sanity gates only; the *printed output*
//! is the artefact consumed by the verification report.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use mnem_core::id::{NodeId, StableId};
use mnem_core::index::AuthoredSliceAdjacency;
use mnem_embed_providers::manifest::EmbedderManifest;
use mnem_embed_providers::{EmbedError, Embedder};
use mnem_graphrag::{compute_communities, summarize_community};

const DIM: usize = 64;
const K: usize = 5;
const SEED_POOL: usize = 8;

// ---------------------------------------------------------------------
// Keyword-bag embedder
// ---------------------------------------------------------------------
//
// Each token in `vocab` gets a deterministic L2-normalised basis vector
// derived from a splitmix-style PRNG. A document's embedding is the
// L2-normalised mean of the basis vectors of its in-vocabulary tokens.
//
// Why this matters: the basis is *not* orthogonal, so co-occurring
// tokens push vectors toward shared subspaces. The polysemous word
// "apple" appears in two distinct topical communities (Apple Inc. and
// orchard apples), giving every node in either community a strong
// shared component along the "apple" basis. This is the realistic
// failure mode that motivates community-aware retrieval.

#[derive(Clone)]
struct KeywordEmbedder {
    basis: HashMap<String, [f32; DIM]>,
    model: String,
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn make_basis(seed: u64) -> [f32; DIM] {
    let mut v = [0.0_f32; DIM];
    let mut s = seed;
    for slot in &mut v {
        s = splitmix(s);
        let bits = (s as u32) as i32;
        *slot = (bits as f32) / (i32::MAX as f32);
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

impl KeywordEmbedder {
    fn new(vocab: &[&str], seed: u64) -> Self {
        let mut basis = HashMap::new();
        for (i, w) in vocab.iter().enumerate() {
            basis.insert(
                (*w).to_lowercase(),
                make_basis(seed.wrapping_add(i as u64).wrapping_mul(0x100000001B3)),
            );
        }
        Self {
            basis,
            model: "kwbag:demo".into(),
        }
    }

    fn embed_text(&self, text: &str) -> [f32; DIM] {
        let mut acc = [0.0_f32; DIM];
        let mut n: u32 = 0;
        for tok in text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
        {
            if let Some(v) = self.basis.get(tok) {
                for (i, x) in v.iter().enumerate() {
                    acc[i] += *x;
                }
                n += 1;
            }
        }
        if n > 0 {
            let invn = 1.0 / n as f32;
            for x in &mut acc {
                *x *= invn;
            }
        }
        let nn: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nn > 0.0 {
            for x in &mut acc {
                *x /= nn;
            }
        }
        acc
    }
}

impl Embedder for KeywordEmbedder {
    fn model(&self) -> &str {
        &self.model
    }
    fn dim(&self) -> u32 {
        DIM as u32
    }
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.embed_text(text).to_vec())
    }
    fn manifest(&self) -> EmbedderManifest {
        EmbedderManifest::new(self.model.clone(), DIM as u32, 0.0)
    }
}

fn cosine(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    let mut s = 0.0;
    for i in 0..DIM {
        s += a[i] * b[i];
    }
    s
}

fn make_id(i: usize) -> NodeId {
    let mut b = [0_u8; 16];
    b[15] = i as u8;
    StableId::from_bytes(&b).unwrap()
}

fn idx_of(nid: NodeId) -> usize {
    nid.as_bytes()[15] as usize
}

// ---------------------------------------------------------------------
// Dataset: 30 nodes across 3 ground-truth communities
// ---------------------------------------------------------------------

fn dataset() -> (Vec<&'static str>, Vec<&'static str>, Vec<(usize, usize)>) {
    let texts: Vec<&'static str> = vec![
        // Community A (0..9): Apple Inc. (technology)
        /* 0 */
        "Apple Inc unveiled the iPhone 17 at Cupertino headquarters",
        /* 1 */ "Tim Cook is the CEO of Apple Inc",
        /* 2 */ "Apple M series chips power Mac laptops",
        /* 3 */ "iOS developers build apps for the Apple App Store",
        /* 4 */ "Apple announced a new MacBook Pro with M4 chip",
        /* 5 */ "Cupertino campus houses thousands of Apple engineers",
        /* 6 */ "Apple stock surged after the iPhone launch event",
        /* 7 */ "Apple Vision Pro is a spatial computing headset",
        /* 8 */ "Apple Watch tracks fitness heart rate and health",
        /* 9 */ "Apple Park features a circular ring building",
        // Community B (10..19): Apple fruit (orchard)
        /*10 */
        "Apple orchards harvest fruit in early autumn",
        /*11 */ "Honeycrisp apples are crunchy and sweet",
        /*12 */ "Apple pie recipe needs fresh tart apples and cinnamon",
        /*13 */ "Apple varieties include Gala Fuji and Granny Smith",
        /*14 */ "Pruning apple trees improves the fruit yield",
        /*15 */ "Cider is made by pressing apples",
        /*16 */ "Apple blossoms attract pollinating bees in spring",
        /*17 */ "Storing apples in cold cellars preserves freshness",
        /*18 */ "Grafting apple rootstock creates new fruit varieties",
        /*19 */ "Washington State produces the most apples in the country",
        // Community C (20..29): Banking / finance
        /*20 */
        "Banks offer savings accounts with compound interest",
        /*21 */ "Credit cards charge variable APR rates monthly",
        /*22 */ "Mortgage loans require down payment and credit check",
        /*23 */ "Federal Reserve sets monetary policy and interest rates",
        /*24 */ "Investment portfolios diversify across asset classes",
        /*25 */ "Stock market trading volume hit new highs today",
        /*26 */ "Bonds pay a fixed coupon over the maturity period",
        /*27 */ "Retirement accounts grow tax advantaged over decades",
        /*28 */ "Bank tellers process deposits and withdrawals daily",
        /*29 */ "Online banking makes wire transfers convenient and fast",
    ];

    let vocab: Vec<&'static str> = vec![
        // shared polysemy
        "apple",
        "apples",
        // tech
        "inc",
        "iphone",
        "cupertino",
        "tim",
        "cook",
        "ceo",
        "ipad",
        "ios",
        "mac",
        "macbook",
        "chip",
        "chips",
        "developer",
        "developers",
        "app",
        "apps",
        "store",
        "stock",
        "watch",
        "vision",
        "park",
        "engineer",
        "engineers",
        "pro",
        "headset",
        "headquarters",
        "campus",
        "circular",
        "ring",
        "launch",
        "event",
        "spatial",
        "computing",
        "fitness",
        "heart",
        "rate",
        "tracks",
        "thousands",
        "series",
        "laptops",
        "build",
        "m4",
        "announced",
        "houses",
        "surged",
        "features",
        "building",
        "health",
        // fruit
        "orchard",
        "orchards",
        "harvest",
        "fruit",
        "autumn",
        "honeycrisp",
        "crunchy",
        "sweet",
        "pie",
        "recipe",
        "tart",
        "cinnamon",
        "gala",
        "fuji",
        "granny",
        "smith",
        "varieties",
        "pruning",
        "tree",
        "trees",
        "yield",
        "cider",
        "pressing",
        "blossom",
        "blossoms",
        "pollinator",
        "pollinating",
        "bee",
        "bees",
        "spring",
        "cold",
        "cellar",
        "cellars",
        "fresh",
        "freshness",
        "grafting",
        "rootstock",
        "washington",
        "state",
        "produces",
        "country",
        "early",
        // finance
        "bank",
        "banks",
        "saving",
        "savings",
        "account",
        "accounts",
        "compound",
        "interest",
        "credit",
        "card",
        "cards",
        "variable",
        "apr",
        "monthly",
        "mortgage",
        "loan",
        "loans",
        "down",
        "payment",
        "check",
        "federal",
        "reserve",
        "monetary",
        "policy",
        "investment",
        "portfolio",
        "portfolios",
        "diversify",
        "asset",
        "classes",
        "trading",
        "volume",
        "highs",
        "today",
        "bond",
        "bonds",
        "coupon",
        "maturity",
        "period",
        "retirement",
        "tax",
        "advantaged",
        "decades",
        "teller",
        "tellers",
        "deposit",
        "deposits",
        "withdrawal",
        "withdrawals",
        "daily",
        "online",
        "banking",
        "wire",
        "transfers",
        "convenient",
        "fast",
        "rates",
        "market",
        "fixed",
    ];

    // Edges: dense intra-community + 3 cross-community bridges.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for c in 0..3_usize {
        let base = c * 10;
        for i in 0..10 {
            for j in (i + 1)..10 {
                let d = j - i;
                if d <= 3 || d >= 8 {
                    edges.push((base + i, base + j));
                }
            }
        }
    }
    // 3 cross-community bridges (sparse)
    edges.push((2, 12)); // A-B (the polysemy bridge: both mention `apple`)
    edges.push((6, 25)); // A-C (Apple stock <-> stock market)
    edges.push((15, 20)); // B-C (cider business <-> banks/savings)

    (texts, vocab, edges)
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn entropy(counts: &BTreeMap<u32, usize>) -> f32 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }
    let mut h = 0.0_f32;
    for &c in counts.values() {
        if c == 0 {
            continue;
        }
        let p = c as f32 / total as f32;
        h -= p * p.ln();
    }
    h
}

fn print_vec(idxs: &[usize]) -> String {
    let s: Vec<String> = idxs.iter().map(|i| format!("n{i:02}")).collect();
    format!("[{}]", s.join(", "))
}

// ---------------------------------------------------------------------
// The demo as a test (with --nocapture, the printed output is the
// artefact consumed by the verification report).
// ---------------------------------------------------------------------

#[test]
fn graphrag_vs_rag_demo() {
    let (texts, vocab, edge_pairs) = dataset();
    let embedder = KeywordEmbedder::new(&vocab, 0xC0FFEE_u64);

    // Embed every node.
    let embs: Vec<[f32; DIM]> = texts.iter().map(|t| embedder.embed_text(t)).collect();

    // Build adjacency.
    let edges_nid: Vec<(NodeId, NodeId)> = edge_pairs
        .iter()
        .map(|&(a, b)| (make_id(a), make_id(b)))
        .collect();
    let adj = AuthoredSliceAdjacency::new(&edges_nid);

    // Run Leiden.
    let assignment = compute_communities(&adj, 42);

    println!("\n================================================================");
    println!("  mnem-graphrag verification - Leiden + community-aware retrieval");
    println!("================================================================\n");

    println!("== Dataset ==");
    println!("  nodes:                30 (3 ground-truth communities of 10)");
    println!("  edges:                {}", edge_pairs.len());
    println!("  cross-community:      3 (sparse bridges)");
    println!();

    println!("== Leiden community detection ==");
    println!("  modularity Q:         {:.4}", assignment.modularity);
    println!("  communities found:    {}", assignment.community_count());
    println!("  partition CID:        {}", assignment.content_cid());
    println!("  seed:                 {}", assignment.seed);
    println!();
    println!("  partition:");
    for cid in 0..(assignment.community_count() as u32) {
        let members = assignment.members_of(cid);
        let idxs: Vec<usize> = members.iter().map(|&n| idx_of(n)).collect();
        println!(
            "    C{} (|members|={:2}):  {}",
            cid,
            idxs.len(),
            print_vec(&idxs)
        );
    }
    println!();

    // Determinism check (re-run, same seed).
    let assignment2 = compute_communities(&adj, 42);
    assert_eq!(assignment.content_cid(), assignment2.content_cid());
    println!("  determinism re-run:   IDENTICAL CID (byte-equal partition)\n");

    // Recover ground-truth -> detected community mapping.
    // Greedy: for each ground-truth (0=A, 1=B, 2=C), pick detected
    // community with the largest overlap.
    let truth_groups: Vec<Vec<usize>> =
        vec![(0..10).collect(), (10..20).collect(), (20..30).collect()];
    let mut gt_to_detected: Vec<u32> = Vec::new();
    let mut used = BTreeSet::new();
    for tg in &truth_groups {
        let mut best: (u32, usize) = (0, 0);
        for cid in 0..(assignment.community_count() as u32) {
            if used.contains(&cid) {
                continue;
            }
            let members = assignment.members_of(cid);
            let idxs: BTreeSet<usize> = members.iter().map(|&n| idx_of(n)).collect();
            let overlap = tg.iter().filter(|i| idxs.contains(i)).count();
            if overlap > best.1 {
                best = (cid, overlap);
            }
        }
        used.insert(best.0);
        gt_to_detected.push(best.0);
    }
    println!(
        "  ground-truth -> detected: A->C{}, B->C{}, C->C{}",
        gt_to_detected[0], gt_to_detected[1], gt_to_detected[2]
    );

    // Purity: how many nodes per ground-truth group land in the
    // dominant detected community for that group.
    let mut purity_total: usize = 0;
    for (gid, tg) in truth_groups.iter().enumerate() {
        let target = gt_to_detected[gid];
        let correct = tg
            .iter()
            .filter(|&&i| assignment.community_of(make_id(i)) == Some(target))
            .count();
        purity_total += correct;
        println!(
            "    truth-{}: {}/10 nodes in C{}",
            ["A", "B", "C"][gid],
            correct,
            target
        );
    }
    let purity = purity_total as f32 / 30.0;
    println!("  partition purity:     {purity:.2}\n");

    // -----------------------------------------------------------------
    // Retrieval comparison
    // -----------------------------------------------------------------

    let queries: Vec<(&str, &str, BTreeSet<usize>, &str)> = vec![
        (
            "Q1",
            "Tim Cook iPhone Apple CEO Cupertino",
            (0..10).collect(),
            "unambiguous Apple-Inc query",
        ),
        (
            "Q2",
            "Honeycrisp apple orchard fruit harvest pie",
            (10..20).collect(),
            "unambiguous orchard query",
        ),
        (
            "Q3",
            "credit savings interest rate bank account",
            (20..30).collect(),
            "unambiguous finance query",
        ),
        (
            "Q4",
            "apple",
            (0..20).collect(),
            "polysemous query (either Apple-Inc OR orchard is valid; mix is bad)",
        ),
    ];

    println!("== Retrieval comparison (top-{K}) ==");
    println!("  Vector-only:  rank by cosine(q, node) over all 30 nodes.");
    println!(
        "  GraphRAG:     pull top-{SEED_POOL} vector seeds; pick dominant community via Leiden;"
    );
    println!("                restrict to that community, then re-rank by cosine.\n");

    let mut sum_prec_v = 0.0_f32;
    let mut sum_prec_g = 0.0_f32;
    let mut sum_div_v = 0.0_f32; // cross-community entropy of result list (lower=more focused)
    let mut sum_div_g = 0.0_f32;

    for (qid, qtext, truth, label) in &queries {
        let q_emb = embedder.embed_text(qtext);
        let mut scored: Vec<(usize, f32)> = (0..texts.len())
            .map(|i| (i, cosine(&q_emb, &embs[i])))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let vector_topk: Vec<usize> = scored.iter().take(K).map(|(i, _)| *i).collect();
        let prec_v = vector_topk.iter().filter(|i| truth.contains(i)).count() as f32 / K as f32;

        let seeds: Vec<usize> = scored.iter().take(SEED_POOL).map(|(i, _)| *i).collect();
        let mut comm_count: HashMap<u32, usize> = HashMap::new();
        for s in &seeds {
            if let Some(c) = assignment.community_of(make_id(*s)) {
                *comm_count.entry(c).or_insert(0) += 1;
            }
        }
        let dominant = comm_count
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(c, _)| *c)
            .unwrap_or(0);
        let coverage = comm_count.get(&dominant).copied().unwrap_or(0) as f32 / SEED_POOL as f32;
        let allowed: BTreeSet<usize> = (0..texts.len())
            .filter(|i| assignment.community_of(make_id(*i)) == Some(dominant))
            .collect();
        let graph_topk: Vec<usize> = scored
            .iter()
            .filter(|(i, _)| allowed.contains(i))
            .take(K)
            .map(|(i, _)| *i)
            .collect();
        let prec_g = graph_topk.iter().filter(|i| truth.contains(i)).count() as f32 / K as f32;

        // Cross-community entropy of the result list.
        let mut hv: BTreeMap<u32, usize> = BTreeMap::new();
        let mut hg: BTreeMap<u32, usize> = BTreeMap::new();
        for &i in &vector_topk {
            if let Some(c) = assignment.community_of(make_id(i)) {
                *hv.entry(c).or_insert(0) += 1;
            }
        }
        for &i in &graph_topk {
            if let Some(c) = assignment.community_of(make_id(i)) {
                *hg.entry(c).or_insert(0) += 1;
            }
        }
        let ent_v = entropy(&hv);
        let ent_g = entropy(&hg);

        sum_prec_v += prec_v;
        sum_prec_g += prec_g;
        sum_div_v += ent_v;
        sum_div_g += ent_g;

        println!("  {qid} ({label}):");
        println!("       q = \"{qtext}\"");
        println!(
            "       vector-only:    {}  precision@{}={:.2}  cross-comm entropy={:.3}",
            print_vec(&vector_topk),
            K,
            prec_v,
            ent_v
        );
        println!(
            "       graphRAG seeds: top-{} -> dominant=C{} (coverage {:.0}%)",
            SEED_POOL,
            dominant,
            coverage * 100.0
        );
        println!(
            "       graphRAG topk:  {}  precision@{}={:.2}  cross-comm entropy={:.3}",
            print_vec(&graph_topk),
            K,
            prec_g,
            ent_g
        );
        println!();
    }

    let n = queries.len() as f32;
    println!("== Aggregate ==");
    println!(
        "  mean precision@{}  vector-only:   {:.3}",
        K,
        sum_prec_v / n
    );
    println!(
        "  mean precision@{}  graphRAG:      {:.3}",
        K,
        sum_prec_g / n
    );
    println!(
        "  mean cross-comm entropy  vector:  {:.3}  (mixing bad on polysemy)",
        sum_div_v / n
    );
    println!(
        "  mean cross-comm entropy  graph:   {:.3}  (focused = correct)",
        sum_div_g / n
    );
    println!();

    // -----------------------------------------------------------------
    // Centroid + MMR community summaries
    // -----------------------------------------------------------------

    println!("== Centroid + MMR community summaries (extractive, no LLM) ==");
    let community_count = assignment.community_count() as u32;
    for cid in 0..community_count {
        let members = assignment.members_of(cid);
        if members.is_empty() {
            continue;
        }
        let sentences: Vec<String> = members
            .iter()
            .map(|&n| texts[idx_of(n)].to_string())
            .collect();
        // No query: pure centroid + MMR.
        let summary = summarize_community(&sentences, &embedder, None, &|_| 1.0, 3, 0.5)
            .expect("summarize must succeed");
        println!("  C{} ({} members) - top-3 extractive:", cid, members.len());
        for (i, s) in summary.sentences.iter().enumerate() {
            println!("     {}. {}", i + 1, s);
        }
        println!();
    }

    // -----------------------------------------------------------------
    // Sanity gates (assertions)
    // -----------------------------------------------------------------
    assert!(
        assignment.modularity > 0.30,
        "modularity {:.4} below threshold",
        assignment.modularity
    );
    assert!(
        assignment.community_count() >= 3,
        "expected >=3 communities"
    );
    assert!(
        purity >= 0.80,
        "partition purity {purity:.2} below 0.80 threshold"
    );
    assert!(
        sum_prec_g / n >= sum_prec_v / n,
        "graphRAG mean precision {:.3} should be >= vector-only {:.3}",
        sum_prec_g / n,
        sum_prec_v / n
    );
}

// ---------------------------------------------------------------------
// Karate Club benchmark - print modularity and partition explicitly so
// it lands in the verification report. Leiden ground truth: Q > 0.4,
// >= 2 communities, node 0 (Mr Hi) and node 33 (John A) separated.
// ---------------------------------------------------------------------

#[test]
fn karate_club_with_printed_metrics() {
    // Standard 78-edge Zachary Karate Club edge list.
    const KARATE_EDGES: &[(usize, usize)] = &[
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 17),
        (0, 19),
        (0, 21),
        (0, 31),
        (1, 2),
        (1, 3),
        (1, 7),
        (1, 13),
        (1, 17),
        (1, 19),
        (1, 21),
        (1, 30),
        (2, 3),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 13),
        (2, 27),
        (2, 28),
        (2, 32),
        (3, 7),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 10),
        (5, 16),
        (6, 16),
        (8, 30),
        (8, 32),
        (8, 33),
        (9, 33),
        (13, 33),
        (14, 32),
        (14, 33),
        (15, 32),
        (15, 33),
        (18, 32),
        (18, 33),
        (19, 33),
        (20, 32),
        (20, 33),
        (22, 32),
        (22, 33),
        (23, 25),
        (23, 27),
        (23, 29),
        (23, 32),
        (23, 33),
        (24, 25),
        (24, 27),
        (24, 31),
        (25, 31),
        (26, 29),
        (26, 33),
        (27, 33),
        (28, 31),
        (28, 33),
        (29, 32),
        (29, 33),
        (30, 32),
        (30, 33),
        (31, 32),
        (31, 33),
        (32, 33),
    ];
    let edges: Vec<(NodeId, NodeId)> = KARATE_EDGES
        .iter()
        .map(|&(a, b)| (make_id(a), make_id(b)))
        .collect();
    let adj = AuthoredSliceAdjacency::new(&edges);

    let assignment = compute_communities(&adj, 42);

    println!("\n================================================================");
    println!("  Zachary Karate Club benchmark");
    println!("================================================================");
    println!("  nodes:                34");
    println!("  edges:                {}", KARATE_EDGES.len());
    println!(
        "  modularity Q:         {:.4} (Newman 2006 ground truth: Q ~ 0.4)",
        assignment.modularity
    );
    println!("  communities found:    {}", assignment.community_count());
    println!("  partition CID:        {}", assignment.content_cid());
    println!(
        "  Mr Hi (node 0)        in community C{:?}",
        assignment.community_of(make_id(0)).unwrap()
    );
    println!(
        "  John A (node 33)      in community C{:?}",
        assignment.community_of(make_id(33)).unwrap()
    );

    for cid in 0..(assignment.community_count() as u32) {
        let members = assignment.members_of(cid);
        let idxs: Vec<usize> = members.iter().map(|&n| idx_of(n)).collect();
        println!("  C{} (|members|={:2}):  {:?}", cid, idxs.len(), idxs);
    }
    println!();

    assert!(assignment.modularity > 0.35);
    assert_ne!(
        assignment.community_of(make_id(0)),
        assignment.community_of(make_id(33))
    );
}

// ---------------------------------------------------------------------
// Multi-hop graph expansion demo: a query about an entity that is
// only reachable by traversing through an intermediate.
// ---------------------------------------------------------------------

#[test]
fn multihop_graph_expand_demo() {
    // Toy entity graph:
    //
    //   "Tim Cook"  --employed_at-->  "Apple Inc"  --makes-->  "iPhone"
    //   "Sundar Pichai" --employed_at--> "Google"  --makes-->  "Pixel"
    //
    // Query: "Who runs the company that makes the iPhone?"
    //
    // Vector-only RAG has nothing to traverse - it can find iPhone or
    // Tim Cook independently, but cannot bridge them. GraphRAG (1-hop
    // expansion) finds iPhone by vector, then expands along edges to
    // pick up "Apple Inc" then "Tim Cook" so the answer surfaces.

    let texts: Vec<&'static str> = vec![
        /* 0 */ "Tim Cook",
        /* 1 */ "Apple Inc",
        /* 2 */ "iPhone",
        /* 3 */ "Sundar Pichai",
        /* 4 */ "Google",
        /* 5 */ "Pixel",
    ];

    let vocab: Vec<&'static str> = vec![
        "tim", "cook", "apple", "inc", "iphone", "sundar", "pichai", "google", "pixel",
    ];
    let embedder = KeywordEmbedder::new(&vocab, 0xCAFEBABE);
    let embs: Vec<[f32; DIM]> = texts.iter().map(|t| embedder.embed_text(t)).collect();

    // Edges (undirected for Leiden): Cook-Apple, Apple-iPhone, Pichai-Google, Google-Pixel.
    let edges_idx: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (3, 4), (4, 5)];
    let edges: Vec<(NodeId, NodeId)> = edges_idx
        .iter()
        .map(|&(a, b)| (make_id(a), make_id(b)))
        .collect();
    let adj = AuthoredSliceAdjacency::new(&edges);
    let assignment = compute_communities(&adj, 7);

    println!("\n================================================================");
    println!("  Multi-hop expansion: \"Who runs the company that makes the iPhone?\"");
    println!("================================================================");
    println!("  modularity Q:         {:.4}", assignment.modularity);
    println!("  communities found:    {}", assignment.community_count());
    for cid in 0..(assignment.community_count() as u32) {
        let members: Vec<usize> = assignment
            .members_of(cid)
            .iter()
            .map(|&n| idx_of(n))
            .collect();
        let names: Vec<&str> = members.iter().map(|&i| texts[i]).collect();
        println!("  C{cid}: {names:?}");
    }

    let q = embedder.embed_text("Who runs the company that makes the iPhone");
    let mut scored: Vec<(usize, f32)> = (0..texts.len())
        .map(|i| (i, cosine(&q, &embs[i])))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n  Vector-only ranking:");
    for (i, s) in &scored {
        println!("    cos={:.3}  n{} = \"{}\"", s, i, texts[*i]);
    }
    let vector_top1 = scored[0].0;
    println!(
        "  vector-only top-1 = \"{}\"  (cannot bridge to \"{}\" without traversal)",
        texts[vector_top1], texts[0]
    );

    // 1-hop expansion: take the top-1 seed, walk neighbours, then 2-hop.
    let mut neighbours: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for &(a, b) in &edges_idx {
        neighbours.entry(a).or_default().insert(b);
        neighbours.entry(b).or_default().insert(a);
    }
    let seed = vector_top1;
    let one_hop: BTreeSet<usize> = neighbours.get(&seed).cloned().unwrap_or_default();
    let mut two_hop: BTreeSet<usize> = BTreeSet::new();
    for n in &one_hop {
        if let Some(nn) = neighbours.get(n) {
            two_hop.extend(nn.iter().copied());
        }
    }
    two_hop.remove(&seed);

    let candidates: BTreeSet<usize> = one_hop.union(&two_hop).copied().collect();

    println!(
        "\n  GraphRAG 2-hop expansion from seed n{} (\"{}\"):",
        seed, texts[seed]
    );
    println!(
        "    1-hop neighbours: {:?}",
        one_hop.iter().map(|&i| texts[i]).collect::<Vec<_>>()
    );
    println!(
        "    2-hop neighbours: {:?}",
        two_hop.iter().map(|&i| texts[i]).collect::<Vec<_>>()
    );
    println!(
        "    expanded candidate set: {:?}",
        candidates.iter().map(|&i| texts[i]).collect::<Vec<_>>()
    );

    // The seed's community is {Cook, Apple Inc, iPhone}. Vector-only
    // top-1 = iPhone. With expansion we can now answer "Tim Cook".
    let cook_in_expansion = candidates.contains(&0);
    println!(
        "    \"Tim Cook\" reachable via graph expansion?  {}",
        if cook_in_expansion { "YES" } else { "no" }
    );
    assert!(cook_in_expansion, "graph expansion must surface Tim Cook");

    // Confirm the seed's community matches the answer's community.
    let seed_c = assignment.community_of(make_id(seed));
    let cook_c = assignment.community_of(make_id(0));
    println!(
        "    seed community = {:?}, answer community = {:?}  (same? {})",
        seed_c,
        cook_c,
        seed_c == cook_c
    );
    assert_eq!(seed_c, cook_c);
}
