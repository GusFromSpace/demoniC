//! #767: `map_get(d, "missing") == 3` — `dmc run` answers `false` (a missing
//! key is `Value::Nil`, and `nil == 3` is a legal comparison per
//! `scalar_compare`'s blanket `(Nil, _) | (_, Nil)` rule); `dmc jit` used to
//! trap with the SAME unlocated "nil value used where a value is required"
//! message every other operator on an absent `map_get` result gets — a
//! run/jit divergence `make parity` cannot catch (no probe exercised it).
//! `!=` diverged identically. Found by the review of PR #764 (2026-09-06).
//!
//! Fix (`compiler/src/jit.rs`'s `scalar_binop`): `==`/`!=` against a
//! present-bit-clear `OptI64` now answer directly, matching the
//! interpreter, before the shared decode-and-trap step ever runs. Every
//! OTHER operator (`<` here) still traps on an absent value — the
//! interpreter errors there too — but the trap is now a LOCATED runtime
//! error (`line:col`) instead of the old unlocated one (#480).
//!
//! `compiler/src/jit.rs`'s own unit tests (`jit_767_*`) cover the decision
//! tree directly (both-absent, one-absent, both-present, a genuine kind
//! mismatch); this file drives the real `dmc` binary end to end to pin the
//! actual process exit code and stderr text for both backends, the way
//! `jit_map_get_tensor_param_769.rs` does for its own issue.

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

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).trim().to_string()
}

const SETUP: &str = "let !d = map_new()\n    map_set(d, \"present\", 5)\n    ";

fn eq_src(op: &str, rhs: &str) -> String {
    format!(
        "fn main() -> i64 {{\n    {}let v = map_get(d, \"missing\")\n    if v {} {} {{ 1 }} else {{ 0 }}\n}}\n",
        SETUP, op, rhs,
    )
}

#[test]
fn eq_against_an_absent_key_answers_false_both_backends() {
    let src = eq_src("==", "3");
    let probe = Probe::new(&src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run), "\n=> 0\n");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(),
        "dmc jit should answer, not trap: {}", stderr(&jit));
    assert_eq!(stdout(&jit), "=> 0\n");
}

#[test]
fn neq_against_an_absent_key_answers_true_both_backends() {
    let src = eq_src("!=", "3");
    let probe = Probe::new(&src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run), "\n=> 1\n");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(),
        "dmc jit should answer, not trap: {}", stderr(&jit));
    assert_eq!(stdout(&jit), "=> 1\n");
}

#[test]
fn eq_against_a_present_key_still_compares_the_payload_both_backends() {
    let src = "fn main() -> i64 {\n    let !d = map_new()\n    map_set(d, \"k\", 3)\n    \
               let v = map_get(d, \"k\")\n    if v == 3 { 1 } else { 0 }\n}\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run), "\n=> 1\n");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", stderr(&jit));
    assert_eq!(stdout(&jit), "=> 1\n");
}

// `<` against an absent key: the interpreter errors here too (`scalar_compare`
// has no `nil`-vs-concrete rule for the ordering operators), so this is NOT a
// divergence to erase — but the JIT's trap must be located (#480), where it
// used to be the bare, unlocated "nil value used where a value is required".
#[test]
fn ordering_on_an_absent_key_is_a_located_runtime_error_in_the_jit() {
    let src = "fn main() -> i64 {\n    let !d = map_new()\n    map_set(d, \"present\", 5)\n    \
               let v = map_get(d, \"missing\")\n    if v < 3 { 1 } else { 0 }\n}\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert_eq!(run.status.code(), Some(1), "dmc run: {}", stderr(&run));
    assert_eq!(stderr(&run), "runtime error: ordering operators not defined for nil");

    let jit = probe.run(&["jit"]);
    assert_eq!(jit.status.code(), Some(1), "dmc jit: {}", stderr(&jit));
    let jit_err = stderr(&jit);
    assert_eq!(jit_err, "runtime error at 5:8: ordering operators not defined for nil",
        "expected a LOCATED runtime error (#480), got: {}", jit_err);
}

// `+` against an absent key: same shape as the issue's own repro, kept as a
// regression pin for the interpreter-matching wording ("Add requires numeric
// operands, got nil and int") now that it's located in the JIT too.
#[test]
fn add_on_an_absent_key_is_a_located_runtime_error_matching_the_interpreters_wording() {
    let src = "fn main() -> i64 {\n    let !d = map_new()\n    map_set(d, \"present\", 5)\n    \
               let v = map_get(d, \"missing\")\n    v + 1\n}\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert_eq!(run.status.code(), Some(1), "dmc run: {}", stderr(&run));
    assert_eq!(stderr(&run), "runtime error: Add requires numeric operands, got nil and int");

    let jit = probe.run(&["jit"]);
    assert_eq!(jit.status.code(), Some(1), "dmc jit: {}", stderr(&jit));
    let jit_err = stderr(&jit);
    assert_eq!(jit_err, "runtime error at 5:5: Add requires numeric operands, got nil and int",
        "expected a LOCATED runtime error (#480), got: {}", jit_err);
}

// #865 REWORK of #767: both operands are `map_get` results (`OptI64`), only
// one absent. The located trap's message for the ABSENT side names the
// OTHER (present) operand's type — that name is a runtime question (the
// other operand's own present bit), not a static one, because the other
// side is itself an undecoded `OptI64` at the point the message is built.
// Before the fix, `jit.rs`'s `interp_scalar_type_name(OptI64)` unconditionally
// answered "nil" for that name regardless of the other operand's actual
// runtime presence, so `dmc jit` printed "got nil and nil" here where
// `dmc run` (and the fixed JIT) print "got nil and int" — the review's
// blocking finding on PR #865.
#[test]
fn both_map_get_lhs_absent_names_the_present_rhs_int_not_nil() {
    let src = "fn main() -> i64 {\n  let !d = map_new()  map_set(d, \"present\", 5)\n  \
               let a = map_get(d, \"missing\")\n  let b = map_get(d, \"present\")\n  a + b\n}\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert_eq!(run.status.code(), Some(1), "dmc run: {}", stderr(&run));
    assert_eq!(stderr(&run), "runtime error: Add requires numeric operands, got nil and int");

    let jit = probe.run(&["jit"]);
    assert_eq!(jit.status.code(), Some(1), "dmc jit: {}", stderr(&jit));
    let jit_err = stderr(&jit);
    assert_eq!(jit_err, "runtime error at 5:3: Add requires numeric operands, got nil and int",
        "expected the OTHER operand's name to reflect its actual runtime \
         presence (\"int\", it IS present), not a static \"nil\" guess — got: {}", jit_err);
}

// Mirror direction of the test above: the PRESENT operand is now the LHS,
// the ABSENT one the RHS. The doc comment on `scalar_binop`'s r-branch
// claims this direction was already correct before the fix (by the time the
// RHS's trap message is built, the LHS's own decode already proved it
// present, so its name is never in doubt) — pinned here so a future change
// can't quietly break it while "fixing" the LHS-absent direction.
#[test]
fn both_map_get_rhs_absent_names_the_present_lhs_int_not_nil() {
    let src = "fn main() -> i64 {\n  let !d = map_new()  map_set(d, \"present\", 5)\n  \
               let a = map_get(d, \"missing\")\n  let b = map_get(d, \"present\")\n  b + a\n}\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert_eq!(run.status.code(), Some(1), "dmc run: {}", stderr(&run));
    assert_eq!(stderr(&run), "runtime error: Add requires numeric operands, got int and nil");

    let jit = probe.run(&["jit"]);
    assert_eq!(jit.status.code(), Some(1), "dmc jit: {}", stderr(&jit));
    let jit_err = stderr(&jit);
    assert_eq!(jit_err, "runtime error at 5:3: Add requires numeric operands, got int and nil",
        "expected a LOCATED runtime error naming the present LHS \"int\", got: {}", jit_err);
}

// Both `map_get` operands absent: the runtime-selected name must resolve to
// "nil" for the other side too (not just "int") — the select isn't a one-way
// override, it genuinely reads the other operand's present bit.
#[test]
fn both_map_get_absent_names_nil_and_nil() {
    let src = "fn main() -> i64 {\n  let !d = map_new()\n  \
               let a = map_get(d, \"missing\")\n  let b = map_get(d, \"missing2\")\n  a + b\n}\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert_eq!(run.status.code(), Some(1), "dmc run: {}", stderr(&run));
    assert_eq!(stderr(&run), "runtime error: Add requires numeric operands, got nil and nil");

    let jit = probe.run(&["jit"]);
    assert_eq!(jit.status.code(), Some(1), "dmc jit: {}", stderr(&jit));
    let jit_err = stderr(&jit);
    assert_eq!(jit_err, "runtime error at 5:3: Add requires numeric operands, got nil and nil",
        "expected both sides named nil, got: {}", jit_err);
}

// #865 Fix 2 (#822 doctrine): `dmc jit --json` on a located binop-nil trap
// must emit a schema-1 diagnostic object plus a schema-1 summary object,
// exactly like every other JIT fatal routed through
// `diag::out_of_band_fatal` — not bare prose on stderr with no JSON at all
// (`dmc_binop_nil_trap` used to be a bare `eprintln!` + `exit(1)`).
// This crate has no JSON dep in tests (see `jit_map_len_751.rs`'s identical
// note on its own `kinds_and_codes` helper) — a flat substring walk over the
// JSON Lines stderr, not a real parse.
#[test]
fn binop_nil_trap_emits_a_json_diagnostic_and_summary() {
    let src = "fn main() -> i64 {\n    let !d = map_new()\n    map_set(d, \"present\", 5)\n    \
               let v = map_get(d, \"missing\")\n    v + 1\n}\n";
    let probe = Probe::new(src);

    let jit = probe.run(&["jit", "--json"]);
    assert_eq!(jit.status.code(), Some(1), "dmc jit --json: {}", stderr(&jit));
    let err = stderr(&jit);
    let jit_lines: Vec<&str> = err.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(jit_lines.len(), 2,
        "expected exactly one diagnostic line + one summary line, got: {:#?}", jit_lines);

    let diag = jit_lines[0];
    assert!(diag.contains("\"schema\":1"), "diagnostic missing schema 1: {}", diag);
    assert!(diag.contains("\"kind\":\"unstructured\""),
        "expected DIAGNOSTICS.md §5.1's `unstructured` interim kind (`runtime` is reserved, \
         not yet implemented), got: {}", diag);
    assert!(diag.contains("Add requires numeric operands, got nil and int"),
        "diagnostic message missing expected wording: {}", diag);
    assert!(diag.contains("5:5"), "diagnostic message missing the located span: {}", diag);

    let summary = jit_lines[1];
    assert!(summary.contains("\"schema\":1"), "summary missing schema 1: {}", summary);
    assert!(summary.contains("\"kind\":\"summary\""), "expected a summary object: {}", summary);
    assert!(summary.contains("\"status\":\"failed\""), "expected a failed summary: {}", summary);
    assert!(summary.contains("\"exit\":1"), "expected exit 1 in the summary: {}", summary);
}
