//! #781 (PR #790 reviewer rework, defect 1): a `!` tensor parameter written
//! before its frame leaves through a `?` early return writes back on BOTH
//! backends.
//!
//! `?` desugars to `return (default[T], e)` (SPEC.md's Error propagation `?`
//! subsection) — still a return, so #650's copy-in/copy-out
//! convention owes the caller the write exactly as it does when the callee
//! runs to its last expression. The JIT has always emitted
//! `emit_mut_param_copy_out` on every path out of a callee and was never
//! wrong here.
//!
//! `dmc run` briefly was. #781's forward-only `@grad` meter rewind needs the
//! `!`-param writebacks captured BEFORE it moves `forge_used`, and an
//! intermediate pass of this PR put both under one `if result.is_ok()` gate.
//! The `?` path arrives at `call_fn_in_module` as an `Err` carrying the
//! propagation marker (it is rebuilt into a normal `(default[T], e)` return
//! a few lines further down, by `finish_query_propagation`), so that gate
//! silently dropped the writebacks: `dmc run` printed the caller's untouched
//! 0 where `dmc jit` printed 7. Only the meter rewind is gated now.
//!
//! Lives here rather than in `examples/` on purpose: the example corpus is
//! byte-pinned by `tools/golden/`, and a new passing `test_*` there moves a
//! baseline this branch is not re-capturing. The same shape is also a
//! `tools/jit_probes.py` row
//! (`mut_param_writeback_survives_query_early_return`, in MUST_LOWER), so
//! the parity battery scores the two backends against each other as well.

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

    fn run(&self, mode: &str) -> Output {
        Command::new(dmc_binary())
            .arg(mode)
            .arg(&self.path)
            .output()
            .expect("spawn dmc")
    }
}

/// Run `src` under both backends and assert both print `expect`.
fn assert_both_backends_print(src: &str, expect: &str) {
    let probe = Probe::new(src);
    for mode in ["run", "jit"] {
        let out = probe.run(mode);
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(out.status.success(), "dmc {} failed: {}", mode, stderr);
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(
            stdout.contains(expect),
            "dmc {}: expected `{}` in stdout, got: {}\nstderr: {}",
            mode, expect, stdout, stderr,
        );
    }
}

/// The bare defect: write, then leave through `?`. Both backends must show
/// the caller the 7.
#[test]
fn a_mut_tensor_param_written_before_a_query_early_return_writes_back() {
    assert_both_backends_print(
        r#"
fn always_fails() -> (i64, Err) {
    (0, "no")
}

fn write_then_bail(!a: Tensor[i64, [1]]) -> (i64, Err) {
    a[0] = 7
    let k = always_fails()?
    (k, nil)
}

fn main() -> nil {
    let !t = forge.zeros[i64, [1]]
    let (r, e) = write_then_bail(t)
    print(t[0])
    nil
}
"#,
        "7",
    );
}

/// The `?` really did fire — otherwise the test above would pass on a build
/// that never took the early-return path at all. `r` is `default[i64]` and
/// `e` carries the message, and the write still landed.
#[test]
fn the_early_return_still_reports_its_error_and_default() {
    assert_both_backends_print(
        r#"
fn always_fails() -> (i64, Err) {
    (0, "no")
}

fn write_then_bail(!a: Tensor[i64, [1]]) -> (i64, Err) {
    a[0] = 7
    let k = always_fails()?
    a[0] = 99
    (k, nil)
}

fn main() -> nil {
    let !t = forge.zeros[i64, [1]]
    let (r, e) = write_then_bail(t)
    # 7, not 99: the statement after the `?` never ran.
    print(t[0] + r)
    print(e)
    nil
}
"#,
        "7",
    );
}

/// The success path is unchanged — the same fn, driven so the `?` does not
/// fire, still writes back the value its last statement left.
#[test]
fn the_success_path_writes_back_the_final_value() {
    assert_both_backends_print(
        r#"
fn never_fails() -> (i64, Err) {
    (1, nil)
}

fn write_twice(!a: Tensor[i64, [1]]) -> (i64, Err) {
    a[0] = 7
    let k = never_fails()?
    a[0] = 99
    (k, nil)
}

fn main() -> nil {
    let !t = forge.zeros[i64, [1]]
    let (r, e) = write_twice(t)
    print(t[0])
    nil
}
"#,
        "99",
    );
}
