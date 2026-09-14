//! #798/#799: a top-level `let` named as an implicit dimension in a
//! model METHOD's parameter/return annotation.
//!
//! #798 — a top-level literal constant (`let N = 4`) used this way was
//! refused by `check_model`'s own #151 "unknown shape param" diagnostic,
//! unconditionally, even though SPEC.md §6.6 says a named dimension in ANY
//! type annotation that is not a declared shape parameter resolves against
//! a top-level literal `let`, and a free fn's parameter annotation already
//! did exactly that (#766/#793/#797). Fixed by making `check_model` consult
//! `const_dims` the way `check_fn` does — this file's `folds_n_to_4_and_
//! agrees_on_both_backends` test is the differential half of that fix: both
//! `dmc run` and `dmc jit` must not just check clean, but actually fold `N`
//! to `4` and agree, since the JIT lowers a method through its own template
//! that must see the SAME fold `dmc run`'s interpreter does.
//!
//! #799 — a MUTABLE top-level `let !N = 4` used the same way was silently a
//! fresh generic (no diagnostic at all, on either backend) instead of the
//! located "a mutable binding is never a constant dimension (SPEC.md §6.6)"
//! error the type-alias path already gave. `mutable_top_level_let_refuses_
//! at_check_before_either_backend_runs` below is the check-time half of
//! that fix — a program with an unusable mutable-let dimension is refused
//! at `--check`, so it never reaches either backend at all.

use std::path::PathBuf;
use std::process::{Command, Output};

/// Path to the `dmc` binary built by cargo for this crate.
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

    fn check(&self) -> Output {
        self.dmc(&["--check"])
    }

    fn run_output(&self) -> String {
        let out = self.dmc(&["run"]);
        assert!(
            out.status.success(),
            "dmc run failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn jit_output(&self) -> String {
        let out = self.dmc(&["jit"]);
        assert!(
            out.status.success(),
            "dmc jit failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }
}

/// #798: `let N = 4` + a method annotated `Tensor[f32,[N]]` checks clean and
/// folds `N` to `4` — verified on BOTH backends. The JIT lowers a method
/// through its own per-instantiation template (distinct from a free fn's),
/// so this is not redundant with a free-fn differential test: it is the
/// probe that the fold actually reaches the JIT's method template, not just
/// the interpreter.
#[test]
fn folds_n_to_4_and_agrees_on_both_backends_798() {
    let src = "\
let N = 4
model M {
    fn m(self, x: Tensor[f32, [N]]) -> i64 { sum(x) as i64 }
}
fn main() -> i64 {
    let m = M {}
    m.m(forge.zeros[f32, [N]])
}
";
    let probe = Probe::new(src);

    let check = probe.check();
    assert!(
        check.status.success(),
        "expected --check to pass, got:\n{}{}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr),
    );

    let run = probe.run_output();
    let jit = probe.jit_output();
    assert_eq!(run, jit, "dmc run and dmc jit must agree on the folded method");
    assert_eq!(run, "=> 0", "sum of a zero tensor of the folded [4] shape");

    // The JIT proof that the fold actually reached the method template, not
    // just the interpreter: a method whose shape param is genuinely generic
    // (unfolded) refuses to lower without an explicit call-site bracket
    // (`.m[N](..)`, see `explicit_method_shape_param_still_shadows_the_
    // constant_on_both_backends_798` below, which needs exactly that
    // bracket). This call above has NO bracket — `m.m(forge.zeros[f32,
    // [N]])` — and still lowered, which is only possible because `N` was
    // folded to the concrete constant `4` before the JIT's method template
    // ever saw it.
}

/// #798 strengthening: the test above calls with a `[N]`-sized argument,
/// which is `[4]` either way `N` was resolved — it cannot tell "`N` folded
/// to the concrete constant `4`" apart from "`N` was left generic and
/// INFERRED as `4` from this one call site", since both readings produce
/// the identical accept-and-run-clean result. Calling with a MISMATCHED
/// size (`[8]`) tells them apart: a folded, concrete `4` makes `[8]` a
/// genuine shape mismatch against a fixed dimension, where a still-generic,
/// per-call-inferred `N` would happily accept `[8]` and bind `N = 8` for
/// that call (exactly what happens in `explicit_method_shape_param_still_
/// shadows_the_constant_on_both_backends_798` above, where `N` truly is
/// the method's own shape param). `dmc jit` refusing `[8]` here — the same
/// `` cannot convert `Tensor[f32, [8]]` to `Tensor[f32, [4]]` `` a free
/// fn's identical mismatch gets — is the proof the fold reached the
/// method template as a fixed `4`, not a generic.
#[test]
fn mismatched_call_proves_the_method_fold_is_concrete_not_inferred_798() {
    let src = "\
let N = 4
model M {
    fn m(self, x: Tensor[f32, [N]]) -> i64 { len(x) }
}
fn main() -> i64 {
    let m = M {}
    m.m(forge.zeros[f32, [8]])
}
";
    let probe = Probe::new(src);
    assert!(probe.check().status.success(), "--check must still pass a [8] argument here (see #827)");
    let jit = probe.dmc(&["jit"]);
    assert!(
        !jit.status.success(),
        "dmc jit must refuse the [8] argument against the folded [4] method param — \
         acceptance here would mean N was left generic and inferred as 8, not folded to 4",
    );
    let jit_stderr = String::from_utf8_lossy(&jit.stderr).to_string();
    assert!(
        jit_stderr.contains("Tensor[f32, [8]]") && jit_stderr.contains("Tensor[f32, [4]]"),
        "expected the JIT's concrete [8]->[4] shape-mismatch refusal, got: {jit_stderr}",
    );
}

/// #798/#799 bug-pin, filed as #827: `dmc run`
/// and `dmc jit` DISAGREE about whether a top-level constant used as a
/// method's dimension is enforced at the call, and disagree in OPPOSITE
/// directions depending on whether the dimension is the bare name or an
/// arithmetic expression over it. Pre-existing on `main` for a free `fn`
/// (verified identical there before this PR); this PR's method-side const-
/// dim folding (#798/#799) widens the exact same interpreter/JIT gap to a
/// model method, since a method's shape-carrying param now folds a
/// top-level constant the same way a fn's does. NOT fixed here — out of
/// scope for #798/#799 — this test pins the CURRENT divergent behaviour on
/// BOTH backends, labelled as such, so a future fix (tracked in #827)
/// changes this test deliberately instead of by surprise.
#[test]
fn interp_and_jit_diverge_on_a_folded_method_dim_mismatch_798_799_827() {
    // Bare `[N]`, folds to `[4]`, called with `[8]`: `dmc run` does NOT
    // enforce the fold — it runs off the argument's actual runtime shape
    // and answers `8`. `dmc jit` DOES enforce it and refuses (see
    // `mismatched_call_proves_the_method_fold_is_concrete_not_inferred_798`
    // above). Labelled divergent behaviour, not a spec: `run`'s answer of
    // `8` is the interpreter gap #827 tracks, not a semantics this test
    // endorses.
    let bare_n = "\
let N = 4
model M {
    fn m(self, x: Tensor[f32, [N]]) -> i64 { len(x) }
}
fn main() -> i64 {
    let m = M {}
    m.m(forge.zeros[f32, [8]])
}
";
    let probe = Probe::new(bare_n);
    assert!(probe.check().status.success());
    assert_eq!(
        probe.run_output(), "=> 8",
        "#827: dmc run currently does not enforce the [N]->[4] fold at the call; \
         if this now fails, the interpreter gap is fixed — update this pin and close #827",
    );
    let jit = probe.dmc(&["jit"]);
    assert!(
        !jit.status.success(),
        "#827: dmc jit currently refuses [8] against the folded [4] here (opposite of dmc run)",
    );

    // `[N * 2]`, folds to `[8]`, called with `[8]` (a MATCHING size this
    // time): the split runs the OTHER way. `dmc run` does not fold `N * 2`
    // against the top-level `let` at all here, and since nothing else pins
    // `N`, it reports the shape param as uninferable. `dmc jit` DOES fold
    // `N * 2` to `8` and accepts the matching argument.
    let arithmetic = "\
let N = 4
model M {
    fn m(self, x: Tensor[f32, [N * 2]]) -> i64 { len(x) }
}
fn main() -> i64 {
    let m = M {}
    m.m(forge.zeros[f32, [8]])
}
";
    let probe = Probe::new(arithmetic);
    assert!(probe.check().status.success());
    let run = probe.dmc(&["run"]);
    assert!(
        !run.status.success(),
        "#827: dmc run currently fails to infer N from an [N*2]-shaped param here \
         (opposite split from the bare-[N] case above)",
    );
    let run_stderr = String::from_utf8_lossy(&run.stderr).to_string();
    assert!(
        run_stderr.contains('N') && run_stderr.contains("cannot be inferred"),
        "expected #827's pinned \"N cannot be inferred\" runtime error, got: {run_stderr}",
    );
    assert_eq!(
        probe.jit_output(), "=> 8",
        "#827: dmc jit currently folds [N*2] to [8] and accepts the matching argument here",
    );
}

/// #798 control: an explicit `fn m[N](self, ...)` on the method still
/// shadows the top-level constant — `N` stays genuinely generic, so an `[8]`
/// argument is legal on both backends.
#[test]
fn explicit_method_shape_param_still_shadows_the_constant_on_both_backends_798() {
    let src = "\
let N = 4
model M {
    fn m[N](self, x: Tensor[f32, [N]]) -> i64 { sum(x) as i64 }
}
fn main() -> i64 {
    let m = M {}
    m.m[8](forge.zeros[f32, [8]])
}
";
    let probe = Probe::new(src);
    assert!(probe.check().status.success());
    let run = probe.run_output();
    let jit = probe.jit_output();
    assert_eq!(run, jit);
    assert_eq!(run, "=> 0");
}

/// #799: a MUTABLE top-level `let !N = 4` named as a method's implicit
/// dimension is refused at `--check`, before either backend ever runs —
/// the located "a mutable binding is never a constant dimension" error,
/// not silent, unbounded genericity.
#[test]
fn mutable_top_level_let_refuses_at_check_before_either_backend_runs_799() {
    let src = "\
let !N = 4
model M {
    fn m(self, x: Tensor[f32, [N]]) -> i64 { sum(x) as i64 }
}
fn main() -> i64 { 0 }
";
    let probe = Probe::new(src);
    let check = probe.check();
    assert!(!check.status.success(), "a mutable-let dimension in a method annotation must be a check error");
    let stderr = String::from_utf8_lossy(&check.stderr).to_string();
    assert!(
        stderr.contains('N') && stderr.contains("mutable") && stderr.contains("never a constant dimension"),
        "expected the located mutable-binding diagnostic, got: {stderr}",
    );
    assert!(
        !stderr.contains("unknown shape param"),
        "must not ALSO claim N is an unknown shape param — it is a known binding, just not a usable \
         constant, got: {stderr}",
    );
}

/// #799 sibling: the free-fn annotation path gets the identical diagnostic
/// (not just the method path) — pinned here alongside the method case so a
/// future change to one is not accidentally checked against the other only.
#[test]
fn mutable_top_level_let_refuses_at_check_for_a_free_fn_annotation_too_799() {
    let src = "\
let !N = 4
fn f(x: Tensor[f32, [N]]) -> i64 { sum(x) as i64 }
fn main() -> i64 { 0 }
";
    let probe = Probe::new(src);
    let check = probe.check();
    assert!(!check.status.success(), "a mutable-let dimension in a fn annotation must be a check error");
    let stderr = String::from_utf8_lossy(&check.stderr).to_string();
    assert!(
        stderr.contains('N') && stderr.contains("mutable") && stderr.contains("never a constant dimension"),
        "expected the located mutable-binding diagnostic, got: {stderr}",
    );
}
