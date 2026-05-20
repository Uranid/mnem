//! `mnem log` - walk the op-log backwards from the current head.
//!
//! Three output formats:
//!
//! - default: multi-line per op (git log default). Human readable.
//! - `--oneline`: `<short-cid> <message>` one line per op, matching
//!   `git log --oneline`.
//! - `--format=json`: JSON Lines, one compact object per op. Each
//!   line is a `{ "cid", "timestamp", "author", "description", "parents"
//!   }` record. Stable across releases ; scripts can
//!   depend on it.

use std::io::{self, Write};

use serde::Serialize;

use super::*;

/// JSON-Lines record shape. Explicit struct so adding a field later
/// stays backward-compatible with existing consumers (`mnem log
/// --format=json | jq .author` keeps working).
#[derive(Serialize)]
struct LogRecord<'a> {
    cid: String,
    time: u64,
    timestamp: String,
    author: &'a str,
    description: &'a str,
    parents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_id: Option<&'a str>,
}

/// Output format for `mnem log`. Defaults to the human-readable
/// multi-line form.
#[derive(clap::ValueEnum, Clone, Debug)]
pub(crate) enum Format {
    /// Multi-line per op (default).
    Human,
    /// JSON Lines, one record per op. Stable wire contract.
    Json,
}

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem log                       # last 20 ops (default, human form)
  mnem log -n 5                  # last 5 ops
  mnem log --oneline             # short-cid + message per line
  mnem log --format=json | jq .  # JSON Lines, pipe through jq
  mnem log --format=json -n 100  # agent-facing op stream
")]
pub(crate) struct Args {
    /// Maximum number of operations to print.
    #[arg(long, short = 'n', default_value_t = 20)]
    pub limit: usize,
    /// Short-form output, one line per op: `<short-cid> <description>`.
    /// Conflicts with `--format=json`; `--oneline` wins.
    #[arg(long)]
    pub oneline: bool,
    /// Output format. Defaults to the human-readable multi-line
    /// shape. `json` emits JSON Lines.
    #[arg(long, value_enum, default_value = "human")]
    pub format: Format,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_dir, r, bs, _ohs) = repo::open_all(override_path)?;

    let stdout = io::stdout();
    let mut w = stdout.lock();

    let mut cur = r.op_id().clone();
    for i in 0..args.limit {
        let bytes = bs
            .get(&cur)?
            .ok_or_else(|| anyhow!("op {cur} missing from store"))?;
        let op: Operation = from_canonical_bytes(&bytes)?;

        if args.oneline {
            // Git's short-cid is 7 hex chars; mnem CIDs start with a
            // multibase prefix, so we clip after the prefix for a
            // similar visual effect. Falls back to the full CID if
            // the render is shorter than expected.
            let full = cur.to_string();
            let short = short_cid(&full);
            writeln!(w, "{short} {}", op.description)?;
        } else {
            match args.format {
                Format::Json => write_json_record(&mut w, &cur, &op)?,
                Format::Human => write_human_record(&mut w, &cur, &op)?,
            }
        }

        match op.parents.first() {
            Some(p) => cur = p.clone(),
            None => {
                // Already printed this op; don't print a stale "break"
                // marker. The underscore keeps clippy quiet about `i`.
                let _ = i;
                break;
            }
        }
    }
    Ok(())
}

fn write_human_record(w: &mut impl Write, cid: &mnem_core::id::Cid, op: &Operation) -> Result<()> {
    writeln!(w, "op {cid}")?;
    writeln!(w, "   time    {}us", op.time)?;
    if !op.author.is_empty() {
        writeln!(w, "   author  {}", op.author)?;
    }
    if let Some(agent) = &op.agent_id {
        writeln!(w, "   agent   {agent}")?;
    }
    if let Some(task) = &op.task_id {
        writeln!(w, "   task    {task}")?;
    }
    writeln!(w, "   message {}", op.description)?;
    writeln!(w)?;
    Ok(())
}

fn write_json_record(w: &mut impl Write, cid: &mnem_core::id::Cid, op: &Operation) -> Result<()> {
    let record = LogRecord {
        cid: cid.to_string(),
        time: op.time,
        timestamp: micros_to_rfc3339(op.time),
        author: &op.author,
        description: &op.description,
        parents: op.parents.iter().map(ToString::to_string).collect(),
        agent_id: op.agent_id.as_deref(),
        task_id: op.task_id.as_deref(),
    };
    let line = serde_json::to_string(&record).context("serialising log record")?;
    writeln!(w, "{line}")?;
    Ok(())
}

/// Convert microseconds-since-epoch to an RFC 3339 timestamp string.
/// Falls back to the raw integer (as a string) on overflow.
fn micros_to_rfc3339(micros: u64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let secs = micros / 1_000_000;
    let nanos = ((micros % 1_000_000) * 1_000) as u32;
    match UNIX_EPOCH.checked_add(Duration::new(secs, nanos)) {
        Some(_t) => {
            // Format as RFC 3339 UTC without pulling in `chrono` or `time`.
            let s = secs % 60;
            let m = (secs / 60) % 60;
            let h = (secs / 3600) % 24;
            let days = secs / 86400;
            let (year, month, day) = days_to_ymd(days);
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
                year,
                month,
                day,
                h,
                m,
                s,
                micros % 1_000_000,
            )
        }
        None => micros.to_string(),
    }
}

/// Convert days since Unix epoch to (year, month, day).
/// Implements the standard proleptic Gregorian calendar algorithm.
fn days_to_ymd(days: u64) -> (u64, u8, u8) {
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    // (civil_from_days, public domain). Adapted for unsigned input.
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u64, m as u8, d as u8)
}

/// Produce a short-hex prefix of a CID for `--oneline` output. mnem
/// CIDs start with a multibase prefix; trimming the first 2 bytes and
/// taking the next 8 gives a compact, still-unique-in-practice
/// rendering for typical commit counts.
fn short_cid(full: &str) -> String {
    if full.len() <= 10 {
        full.to_string()
    } else {
        // Skip multibase prefix byte (`b` etc) + typical codec byte.
        full.chars().skip(2).take(8).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(days: u64, expected_year: u64, expected_month: u8, expected_day: u8) {
        let (y, m, d) = days_to_ymd(days);
        assert_eq!(
            (y, m, d),
            (expected_year, expected_month, expected_day),
            "days_to_ymd({days}) = ({y},{m},{d}), want ({expected_year},{expected_month},{expected_day})"
        );
    }

    #[test]
    fn epoch() {
        check(0, 1970, 1, 1);
    }

    #[test]
    fn epoch_plus_one() {
        check(1, 1970, 1, 2);
    }

    #[test]
    fn start_of_february_1970() {
        check(31, 1970, 2, 1);
    }

    #[test]
    fn start_of_march_1970_non_leap() {
        check(59, 1970, 3, 1);
    }

    #[test]
    fn second_year() {
        check(365, 1971, 1, 1);
    }

    #[test]
    fn year_2000_leap_day() {
        check(11016, 2000, 2, 29);
    }

    #[test]
    fn year_2000_day_before_leap() {
        check(11015, 2000, 2, 28);
    }

    #[test]
    fn year_2000_day_after_leap() {
        check(11017, 2000, 3, 1);
    }

    #[test]
    fn year_2024_leap_day() {
        check(19782, 2024, 2, 29);
    }

    #[test]
    fn year_1972_leap_day() {
        check(789, 1972, 2, 29);
    }

    #[test]
    fn year_2024_day_before_leap() {
        check(19781, 2024, 2, 28);
    }

    #[test]
    fn year_2024_day_after_leap() {
        check(19783, 2024, 3, 1);
    }

    #[test]
    fn december_year_end_1970() {
        check(364, 1970, 12, 31);
    }

    #[test]
    fn year_2100_is_not_leap_feb_28() {
        check(47540, 2100, 2, 28);
    }

    #[test]
    fn year_2100_is_not_leap_next_day_is_march() {
        check(47541, 2100, 3, 1);
    }

    #[test]
    fn year_2100_start() {
        check(47482, 2100, 1, 1);
    }
}
