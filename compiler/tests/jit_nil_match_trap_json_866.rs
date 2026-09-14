//! #866: `dmc_nil_trap` and `dmc_match_trap` (`compiler/src/jit.rs`) exited
//! through a bare `eprintln!` + `std::process::exit(1)` — the exact defect
//! class #822 retired for the budget fatals, #845 for the chunk-allocation
//! failure, and #865 for the new `dmc_binop_nil_trap` sibling. Under `dmc jit
//! --json` a nil trap or a non-exhaustive match at runtime emitted bare
//! prose on stderr, with no schema-1 diagnostic and no summary object.
//!
//! Fix: both routed through `crate::diag::out_of_band_fatal`, exactly like
//! `dmc_binop_nil_trap`. Each call site already had a `Span` in scope
//! (`opt_unwrap_or_trap`'s callers — `coerce_to`/`coerce_scalar` — and
//! `emit_match_trap`'s callers — `lower_match_stmt`/`lower_match_expr`, via
//! the `MatchExpr`'s own `m.span`), so both traps are now LOCATED
//! (`runtime error at L:C: ...`) where they used to be bare, unlocated
//! prose — matching #480's doctrine for every other JIT runtime fatal.
//!
//! This crate has no JSON dep in tests (see `jit_map_len_751.rs`'s and
//! `jit_map_get_eq_absent_767.rs`'s identical note) — a flat substring walk
//! over the JSON Lines stderr, not a real parse.

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

    fn run(&self, args: &[&str]) -> Output {
        Command::new(dmc_binary())
            .args(args)
            .arg(&self.path)
            .output()
            .expect("spawn dmc")
    }
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).trim().to_string()
}

fn json_lines(out: &Output) -> Vec<String> {
    let e = String::from_utf8_lossy(&out.stderr).to_string();
    e.lines().filter(|l| !l.trim().is_empty()).map(|s| s.to_string()).collect()
}

// A `map_get` result (`OptI64`) is used where a concrete value is required —
// here, coerced to the function's declared `i64` return type. The interpreter
// raises a runtime error on this `nil`; the JIT decodes and traps (#300.4).
const NIL_TRAP_SRC: &str = "fn main() -> i64 {\n    \
    let !d = map_new()\n    \
    map_set(d, \"present\", 5)\n    \
    let v = map_get(d, \"missing\")\n    \
    v\n\
}\n";

// A `match` whose only arm carries a guard that is false at runtime: `has_
// catchall` (`check.rs`) treats a bare-ident-bound arm as exhaustive
// regardless of its guard, so `--check`/`--all` accept this statically, and
// the non-exhaustive case is a genuine runtime question — reaching
// `dmc_match_trap`, not the static "not exhaustive" check error an unguarded
// open scalar match would hit instead.
const MATCH_TRAP_SRC: &str = "fn main() -> i64 {\n    \
    let x = 5\n    \
    match x {\n        \
        n if n == 999 => 10,\n    \
    }\n\
}\n";

// ─── plain mode: located message ───────────────────────────────────────────

#[test]
fn nil_trap_is_a_located_runtime_error_in_plain_mode() {
    let probe = Probe::new(NIL_TRAP_SRC);
    let jit = probe.run(&["jit"]);
    assert_eq!(jit.status.code(), Some(1), "dmc jit: {}", stderr(&jit));
    assert_eq!(
        stderr(&jit),
        "runtime error at 5:5: nil value used where a value is required \
         (a missing map_get result — guard with map_contains or compare to nil)",
        "expected the interpreter's exact wording, now located (#866)",
    );
}

#[test]
fn match_trap_is_a_located_runtime_error_in_plain_mode() {
    let probe = Probe::new(MATCH_TRAP_SRC);
    let jit = probe.run(&["jit"]);
    assert_eq!(jit.status.code(), Some(1), "dmc jit: {}", stderr(&jit));
    assert_eq!(
        stderr(&jit),
        "runtime error at 3:5: match: no arm matched scrutinee",
        "expected the interpreter's exact wording, now located (#866)",
    );
}

// ─── --json mode: schema-1 diagnostic + summary, not bare prose ───────────

#[test]
fn nil_trap_emits_a_json_diagnostic_and_summary() {
    let probe = Probe::new(NIL_TRAP_SRC);
    let jit = probe.run(&["jit", "--json"]);
    assert_eq!(jit.status.code(), Some(1), "dmc jit --json: {}", stderr(&jit));

    let lines = json_lines(&jit);
    assert_eq!(lines.len(), 2,
        "expected exactly one diagnostic line + one summary line, got: {:#?}", lines);

    let diag = &lines[0];
    assert!(diag.contains("\"schema\":1"), "diagnostic missing schema 1: {}", diag);
    assert!(diag.contains("\"kind\":\"unstructured\""),
        "expected DIAGNOSTICS.md §5.1's `unstructured` interim kind (`runtime` \
         is reserved, not yet implemented), got: {}", diag);
    assert!(diag.contains(
        "nil value used where a value is required \
         (a missing map_get result — guard with map_contains or compare to nil)"),
        "diagnostic message missing expected wording: {}", diag);
    assert!(diag.contains("5:5"), "diagnostic message missing the located span: {}", diag);

    let summary = &lines[1];
    assert!(summary.contains("\"schema\":1"), "summary missing schema 1: {}", summary);
    assert!(summary.contains("\"kind\":\"summary\""), "expected a summary object: {}", summary);
    assert!(summary.contains("\"status\":\"failed\""), "expected a failed summary: {}", summary);
    assert!(summary.contains("\"exit\":1"), "expected exit 1 in the summary: {}", summary);
}

#[test]
fn match_trap_emits_a_json_diagnostic_and_summary() {
    let probe = Probe::new(MATCH_TRAP_SRC);
    let jit = probe.run(&["jit", "--json"]);
    assert_eq!(jit.status.code(), Some(1), "dmc jit --json: {}", stderr(&jit));

    let lines = json_lines(&jit);
    assert_eq!(lines.len(), 2,
        "expected exactly one diagnostic line + one summary line, got: {:#?}", lines);

    let diag = &lines[0];
    assert!(diag.contains("\"schema\":1"), "diagnostic missing schema 1: {}", diag);
    assert!(diag.contains("\"kind\":\"unstructured\""),
        "expected DIAGNOSTICS.md §5.1's `unstructured` interim kind (`runtime` \
         is reserved, not yet implemented), got: {}", diag);
    assert!(diag.contains("match: no arm matched scrutinee"),
        "diagnostic message missing expected wording: {}", diag);
    assert!(diag.contains("3:5"), "diagnostic message missing the located span: {}", diag);

    let summary = &lines[1];
    assert!(summary.contains("\"schema\":1"), "summary missing schema 1: {}", summary);
    assert!(summary.contains("\"kind\":\"summary\""), "expected a summary object: {}", summary);
    assert!(summary.contains("\"status\":\"failed\""), "expected a failed summary: {}", summary);
    assert!(summary.contains("\"exit\":1"), "expected exit 1 in the summary: {}", summary);
}
