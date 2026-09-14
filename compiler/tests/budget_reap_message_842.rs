//! #842 — `budget-reap`'s live-child refusal (R2, the arena-budget design's
//! §3.1) must read identically on both backends, modulo the `line:col` a
//! located diagnostic is entitled to spell differently.
//!
//! Before this, `interp.rs` hardcoded the plural pronoun ("reap **them**
//! before reaping **their** parent") while `jit.rs` conjugated, so a single
//! live child read plural under `dmc run` and singular under `dmc jit`. That
//! drift was invisible to every gate: `make jit-probes` pinned only the
//! `budget-reap` tag, and `#867`'s `SAME_MESSAGE` stderr comparison had to
//! leave `budget_reap_live_child_refused_674` out of the set because of it.
//! Both backends now build the clause through
//! `arena::budget_live_child_clause` (unit-tested per count in
//! `arena_tests.rs`); this test is the end-to-end half — it drives the real
//! binary on both backends and compares their whole stderr byte-for-byte, so
//! a future edit to either message alone fails here even if it never reaches
//! a probe.
//!
//! Counts 1 and 2: one live child is where the two disagreed, two is the
//! plural control that was already identical. The carve/reap spelling is
//! `examples/tenant_budgets.dmc`'s.

use std::path::PathBuf;
use std::process::{Command, Output};

fn dmc_binary() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_dmc") {
        return PathBuf::from(path);
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    #[cfg(debug_assertions)]
    let profile = "debug";
    #[cfg(not(debug_assertions))]
    let profile = "release";
    crate_dir.join("target").join(profile).join("dmc")
}

struct Probe {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Probe {
    fn new(src: &str) -> Probe {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("probe.dmc");
        std::fs::write(&path, src).expect("write probe");
        Probe { _dir: dir, path }
    }

    fn backend(&self, subcommand: &str) -> Output {
        let mut cmd = Command::new(dmc_binary());
        cmd.arg(subcommand).arg(&self.path);
        cmd.output().expect("spawn dmc")
    }
}

/// A budget `p` carved from the (unbounded, §4.1) root, given `children`
/// live children, and then reaped — R2's refusal, at whatever count we ask
/// for.
fn reap_with_live_children(children: usize) -> String {
    let mut src = String::from("fn main() -> i64 {\n    let p = forge.carve(forge.root, 65536)\n");
    for i in 1..=children {
        src.push_str(&format!("    let c{} = forge.carve(p, 1024)\n", i));
    }
    src.push_str("    forge.reap(p)\n    1\n}\n");
    src
}

/// Fold out the one thing the two backends are entitled to spell
/// differently — the location on a located diagnostic — and nothing else.
/// This is `tools/jit_probes.py`'s `_normalize_stderr` rule, hand-rolled so
/// the test needs no regex dependency: a leading `runtime error at L:C: `
/// collapses to `runtime error: `, `=> value` harness echoes are dropped,
/// and surrounding whitespace is trimmed. The kind word before the location
/// and the message after it are NOT elective.
fn normalize(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for line in text.lines() {
        if line.trim_start().starts_with("=>") {
            continue;
        }
        let mut normalized = line.to_string();
        for kind in ["runtime error", "type error", "parse error"] {
            let prefix = format!("{} at ", kind);
            let Some(rest) = line.strip_prefix(&prefix) else {
                continue;
            };
            let Some((location, message)) = rest.split_once(": ") else {
                continue;
            };
            let mut halves = location.split(':');
            let line_no = halves.next().unwrap_or("");
            let col_no = halves.next().unwrap_or("");
            let located = halves.next().is_none()
                && !line_no.is_empty()
                && !col_no.is_empty()
                && line_no.chars().all(|c| c.is_ascii_digit())
                && col_no.chars().all(|c| c.is_ascii_digit());
            if located {
                normalized = format!("{}: {}", kind, message);
                break;
            }
        }
        out.push(normalized);
    }
    out.join("\n").trim().to_string()
}

fn assert_backends_agree(children: usize, expected: &str) {
    let probe = Probe::new(&reap_with_live_children(children));

    let run = probe.backend("run");
    let jit = probe.backend("jit");
    assert!(
        !run.status.success(),
        "`dmc run` reaped a budget with {} live child(ren)",
        children
    );
    assert!(
        !jit.status.success(),
        "`dmc jit` reaped a budget with {} live child(ren)",
        children
    );

    let run_err = normalize(&String::from_utf8_lossy(&run.stderr));
    let jit_err = normalize(&String::from_utf8_lossy(&jit.stderr));

    assert_eq!(
        run_err, jit_err,
        "#842: the two backends' `budget-reap` refusal diverged at {} live \
         child(ren)\n  dmc run: {}\n  dmc jit: {}",
        children, run_err, jit_err
    );
    assert_eq!(run_err, expected, "`dmc run` stderr");
    assert_eq!(jit_err, expected, "`dmc jit` stderr");
}

#[test]
fn the_reap_refusal_is_singular_and_identical_at_one_live_child() {
    assert_backends_agree(
        1,
        "runtime error: budget-reap: this budget has 1 live child budget; \
         reap it before reaping its parent",
    );
}

#[test]
fn the_reap_refusal_is_plural_and_identical_at_two_live_children() {
    assert_backends_agree(
        2,
        "runtime error: budget-reap: this budget has 2 live child budgets; \
         reap them before reaping their parent",
    );
}
