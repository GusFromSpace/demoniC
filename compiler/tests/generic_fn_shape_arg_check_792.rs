//! #792 — a model's shape arguments were never compared when both sides came
//! from a *free* function's explicit shape-bracket call. `fn mk[N]() ->
//! Buf[N]` has no value parameter to infer `N` from, so `mk[8]()` is the only
//! way to pin it; nothing consumed that bracket (only a bracketed *method*
//! call did, #474's `bracketed_method_ty`), so `mk[8]()` fell through the
//! general indexing arm and typed `Unknown` — compatible with everything.
//! `dmc --check` passed a program where a caller handed a `Buf[8]` to a
//! `Buf[4]`-shaped return, parameter, field, or `let` binding, and both
//! backends ran it (`#480` doctrine: an ill-typed program should be a check
//! error, not a runtime one).
//!
//! `check_tests.rs` pins every position `--check` itself now refuses
//! (return, parameter, field init, `let` annotation, model-array element,
//! arity, the lowercase-shape-param call form, and the #786 inference rule
//! that must keep working). This file is the differential half: the real
//! `dmc` binary, both backends, on the exact repro from the issue (refused)
//! and on a correctly-typed sibling that uses the very same explicit-bracket
//! call form (still runs, and agrees between `run` and `jit`).

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

    fn dmc(&self, args: &[&str]) -> Output {
        Command::new(dmc_binary())
            .args(args)
            .arg(&self.path)
            .output()
            .expect("spawn dmc")
    }

    fn text(&self, o: &Output) -> String {
        let mut s = String::from_utf8_lossy(&o.stdout).to_string();
        s.push_str(&String::from_utf8_lossy(&o.stderr));
        s.trim().to_string()
    }

    fn check_output(&self) -> (bool, String) {
        let out = self.dmc(&["--check"]);
        (out.status.success(), self.text(&out))
    }

    fn run_output(&self) -> String {
        let out = self.dmc(&["run"]);
        assert!(out.status.success(), "dmc run failed:\n{}", self.text(&out));
        self.text(&out)
    }

    fn jit_output(&self) -> String {
        let out = self.dmc(&["jit"]);
        assert!(out.status.success(), "dmc jit failed:\n{}", self.text(&out));
        self.text(&out)
    }
}

/// The issue's own repro: `--check` must now refuse it, naming both shapes at
/// both the return position and the `take(...)` call.
#[test]
fn the_issue_repro_is_refused_at_check_792() {
    let probe = Probe::new(
        "model Buf[N] {\n\
         \x20   t: Tensor[i64, [N]]\n\
         }\n\
         fn mk[N]() -> Buf[N] { Buf[N] { t: forge.zeros[i64, [N]] } }\n\
         fn wrong() -> Buf[4] { mk[8]() }\n\
         fn take(b: Buf[4]) -> i64 { 4 }\n\
         fn main() -> i64 {\n\
         \x20   let b = wrong()\n\
         \x20   take(mk[8]())\n\
         }\n",
    );
    let (ok, out) = probe.check_output();
    assert!(!ok, "expected `--check` to refuse the shape mismatch, but it passed");
    // #792 rework: the repro carries the contradiction at TWO positions
    // (`wrong`'s return, and the `take(mk[8]())` call argument) — asserting
    // only that "Buf[4]" and "Buf[8]" appear *somewhere* in the output would
    // pass even if just one of those two sites were ever checked. Assert
    // both diagnostics, one per position, so a regression that silently
    // drops either check is caught here.
    assert!(out.contains("fn `wrong` returns `Buf[4]` but body produces `Buf[8]`"),
        "expected the return-position diagnostic, got:\n{out}");
    assert!(out.contains("arg 0: expected Buf[4], got Buf[8]"),
        "expected the call-argument-position diagnostic, got:\n{out}");
}

/// The differential half: a correctly-typed program using the very same
/// explicit-bracket call form (`mk[4]()`, not `mk[8]()`) must still check
/// clean and run identically on both backends — the fix must not make a
/// legitimate call newly refused or newly disagree between backends.
#[test]
fn a_correctly_typed_bracket_call_still_runs_on_both_backends_792() {
    let probe = Probe::new(
        "model Buf[N] {\n\
         \x20   t: Tensor[i64, [N]]\n\
         }\n\
         fn mk[N]() -> Buf[N] { Buf[N] { t: forge.zeros[i64, [N]] } }\n\
         fn ok() -> Buf[4] { mk[4]() }\n\
         fn take(b: Buf[4]) -> i64 { b.t[0] + 4 }\n\
         fn main() -> i64 {\n\
         \x20   let b = ok()\n\
         \x20   take(mk[4]())\n\
         }\n",
    );
    let (ok, out) = probe.check_output();
    assert!(ok, "expected the correctly-typed program to check clean, got:\n{out}");
    assert_eq!(probe.run_output(), "=> 4");
    assert_eq!(probe.jit_output(), "=> 4", "the JIT must agree with `dmc run`");
}

/// #786's inference rule must keep working when the call site writes no
/// bracket at all — the fix only changes what a *written* bracket does, and
/// must not require one where none was ever needed.
#[test]
fn implicit_shape_inference_still_runs_on_both_backends_786() {
    let probe = Probe::new(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         }\n\
         fn cap[N](b: Buf[N]) -> i64 { N }\n\
         fn main() -> i64 {\n\
         \x20   let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   cap(b)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 8");
    assert_eq!(probe.jit_output(), "=> 8", "the JIT must agree with `dmc run`");
}

/// The reviewer's gap: `infer_call_shape_bindings` (check.rs ~7090) had
/// same-kind arms only, so the Tensor-for-View widening this rework adds to
/// `compatible_with` was reachable through an explicit bracket (`total[4](t)`)
/// but not through ordinary inference (`total(t)`) — the very call form #786
/// exists to support. `check_tests.rs`'s
/// `generic_fn_unbracketed_call_infers_n_from_a_tensor_argument_for_a_view_
/// param_792` pins the check-only half; this is the differential half — `N`
/// must actually bind to a value the interpreter can run with, not merely
/// type-check.
///
/// #811 update: `dmc jit` is now asserted too — `unify_param_type` gained
/// `Type::View` arms (mirroring the `Tensor`/`KV` arms), and
/// `declare_template_signature`'s own parameter conversion now rewrites a
/// template's `View[T, S]` parameter to `Tensor[T, S]` before lowering it
/// (`view_param_ty_for_template` in `jit.rs` — the JIT has no runtime
/// representation for a View distinct from a contiguous Tensor), so
/// `total(t)` now lowers and agrees with `dmc run`. The argument is built
/// via `forge.zeros` rather than the array-literal form
/// (`[10u8, 20u8, 30u8, 40u8]`) the check-only original used: that literal
/// lowers to `Tensor[i64, [4]]` in the JIT regardless of its `u8` suffixes —
/// a separate, pre-existing element-type gap in array-literal lowering,
/// unrelated to and out of scope for #811 — so it would refuse on an
/// unrelated basis and defeat the point of asserting `dmc jit` here at all.
#[test]
fn unbracketed_call_infers_n_from_a_tensor_argument_792() {
    let probe = Probe::new(
        "fn total[N](v: View[u8, [N]]) -> i64 {\n\
         \x20   v[0] as i64 + N as i64\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !t = forge.zeros[u8, [4]]\n\
         \x20   t[0] = 10\n\
         \x20   t[1] = 20\n\
         \x20   t[2] = 30\n\
         \x20   t[3] = 40\n\
         \x20   total(t)\n\
         }\n",
    );
    let (ok, out) = probe.check_output();
    assert!(ok, "expected the unbracketed Tensor-for-View call to check clean, got:\n{out}");
    assert_eq!(probe.run_output(), "=> 14");
    assert_eq!(probe.jit_output(), "=> 14", "the JIT must agree with `dmc run` (#811)");
}

/// Same gap, `(View, KV)` side, for `--check` — `total(k)` must still check
/// clean, unbracketed, the way #786 inference requires.
///
/// #811 rework: this test used to also assert `dmc jit` LOWERED the call,
/// freezing the streaming axis at the KV constructor's compile-time `cap`
/// (filled to exactly capacity here). A reviewer pass found that unsound:
/// `cap` is only an upper bound, and the JIT cannot tell this KV apart, at
/// compile time, from one only partway filled — the normal state of a
/// streaming cache mid-fill, not an edge case — for which `cap` is silently
/// the WRONG length (`dmc run` reads the real runtime length; the old JIT
/// arm read the capacity; `jit_view_param_811.rs`'s
/// `total_from_a_half_filled_kv_argument_refuses_811` pins the resulting
/// divergence directly). The fix refuses this binding unconditionally
/// instead, so `dmc jit` is asserted as a REFUSAL below, not an answer — the
/// element type also moves off the check-only original's `u8`: the JIT's
/// `KvTy` is f32-only regardless of this issue, so a `u8`-element KV would
/// refuse for that separate, permanent reason instead of the one under test
/// here.
#[test]
fn unbracketed_call_infers_n_from_a_kv_argument_792() {
    let probe = Probe::new(
        "fn total[N](v: View[f32, [N]]) -> f32 {\n\
         \x20   v[0] + N as f32\n\
         }\n\
         fn main() -> f32 {\n\
         \x20   let !k = forge.kv[f32, [~]](capacity = 4)\n\
         \x20   let !one = forge.zeros[f32, [1]]\n\
         \x20   one[0] = 10.0\n\
         \x20   k <- one\n\
         \x20   one[0] = 20.0\n\
         \x20   k <- one\n\
         \x20   one[0] = 30.0\n\
         \x20   k <- one\n\
         \x20   one[0] = 40.0\n\
         \x20   k <- one\n\
         \x20   total(k)\n\
         }\n",
    );
    let (ok, out) = probe.check_output();
    assert!(ok, "expected the unbracketed KV-for-View call to check clean, got:\n{out}");
    assert_eq!(probe.run_output(), "=> 14", "dmc run has the real runtime length");
    let jit_out = probe.dmc(&["jit"]);
    assert!(!jit_out.status.success(),
        "expected dmc jit to refuse a View param's streaming-axis dimension bound to a \
         KV argument (#811 rework), got:\n{}", probe.text(&jit_out));
    let text = probe.text(&jit_out);
    assert!(text.contains("streaming-axis dimension bound to a `KV` argument"), "{text:?}");
}
