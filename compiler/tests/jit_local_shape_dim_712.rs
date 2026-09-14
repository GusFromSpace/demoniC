//! #712: a LOCAL `let` bound to an integer literal (or to a compile-time
//! -foldable expression of literals and already-folded constants), used as a
//! tensor shape dimension, had no fold path in the JIT's shape resolver — only
//! a MODULE-level constant did (#620). `forge.zeros[i64,[H]]` for a local
//! `let H = 32` failed with the bare, uncoded `jit error: unbound shape
//! parameter "H"` (the #480 defect class), on a program `--check` accepts and
//! `dmc run` answers.
//!
//! The fix extends #620's `ConstDim` fold mechanism to the current fn's own
//! literal `let`s: `jit.rs`'s `Translator::local_const_dims` records each
//! non-mutating `let`'s fold verdict as it is lowered (`record_local_const_dim`,
//! called from `lower_let`), and `scope_const_dims` — the same function the
//! shape resolver already consulted for module constants — merges it in.
//! Every OTHER name currently in scope (a value parameter, a loop/`for`
//! index, an enum-payload binding, a captured lambda variable, or a `let`
//! whose initializer isn't foldable) now defaults to a CLASSED refusal
//! (`jit-ineligible` / `jit-dynamic-shape` as of the #718 rework below —
//! originally `jit-type-form`, reclassified because the fact is "not
//! statically known", the same fact the shape resolver's other refusals in
//! this family already carry that class for — naming the binding) instead of
//! falling through to the bare #480-class error. A `let !` (mutable) is
//! never foldable even when its initializer is a literal, for the same
//! reason a module constant fold never substitutes for a shadowing binding: JIT-
//! compiled code may still write through it before a later shape use.
//!
//! Each fixture drives the real `dmc` binary, matching `dmc run` exactly
//! where the program is expected to lower, and asserting the refusal's class
//! and message where it is not.

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
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// The issue's own repro (gh #712), minimally: `let H = 32` sizing a tensor
/// allocated in the same fn. Was a bare `jit error: unbound shape parameter
/// "H"`; must now lower and match `dmc run`.
#[test]
fn the_issue_712_repro_lowers_and_matches_run() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 32\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   t[0] = 1\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 1");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must now lower this: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), stdout(&run).trim());
}

/// `let W = H * 2` — a foldable ARITHMETIC expression of an already-folded
/// local, not only a bare literal.
#[test]
fn a_let_folded_from_another_local_lets_arithmetic_sizes_a_shape() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 32\n\
               \x20   let W = H * 2\n\
               \x20   let !t = forge.zeros[i64,[W]]\n\
               \x20   t[0] = 1\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 1");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must now lower this: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), stdout(&run).trim());
}

/// Two independent literal `let`s sizing a 2D allocation — the doom_fire.dmc
/// `HEIGHT`/`WIDTH` shape, minimally.
#[test]
fn two_lets_each_fold_as_their_own_shape_dimension() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 4\n\
               \x20   let W = 6\n\
               \x20   let !t = forge.zeros[i64,[H, W]]\n\
               \x20   t[0, 0] = 1\n\
               \x20   t[3, 5] = 1\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 2");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must now lower this: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), stdout(&run).trim());
}

/// A local `let` shadowing a MODULE constant of the same name folds to the
/// LOCAL's own value, not the module constant's — the shadowing rule #620
/// already enforces for a runtime local, now applying identically to a
/// foldable one.
#[test]
fn a_local_let_shadowing_a_module_constant_folds_to_its_own_value() {
    let src = "let N = 10\n\
               \n\
               fn main() -> i64 {\n\
               \x20   let N = 5\n\
               \x20   let !t = forge.zeros[i64,[N]]\n\
               \x20   t[0] = 1\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 1");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must now lower this: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), stdout(&run).trim(), "must fold to the LOCAL 5, never the module constant 10");
}

/// A `let` inside a block, used as a shape dimension OUTSIDE that block: the
/// name is simply out of scope there — a checker-level scope error, never
/// reaching the JIT (or `dmc run`) at all. Pinned here so a future change
/// doesn't quietly turn this into a JIT-only concern.
#[test]
fn a_let_inside_a_block_used_outside_it_is_a_check_time_scope_error() {
    let src = "fn main() -> i64 {\n\
               \x20   if true {\n\
               \x20       let H = 32\n\
               \x20   }\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   t[0] = 1\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let checked = probe.run(&["--check"]);
    assert!(!checked.status.success(), "expected a check-time scope error");
    assert!(stdout(&checked).contains("undefined identifier `H`")
        || stderr(&checked).contains("undefined identifier `H`"),
        "expected an undefined-identifier scope error, got stdout={:?} stderr={:?}",
        stdout(&checked), stderr(&checked));

    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "dmc run must refuse the same out-of-scope read");

    // Never reaches the JIT's shape resolver at all — it is refused before that.
    let jit = probe.run(&["jit"]);
    assert!(!jit.status.success());
}

/// A `let !` (mutable) bound to a literal is NEVER foldable, even though the
/// initializer is a plain literal — mutability alone disqualifies it. Must be
/// a classed, named `jit-dynamic-shape` refusal, never the bare "unbound
/// shape parameter" #480-class error.
///
/// #718 rework (R2): was `jit-type-form` — reclassified to
/// `jit-dynamic-shape`, DIAGNOSTICS.md §5.2's row for "the op needs a
/// statically known shape or rank", which is the actual fact here (a mutable
/// binding's value is never statically known to the JIT). `jit-type-form` is
/// reserved for a value that IS known and simply isn't an integer.
#[test]
fn a_mutable_let_bound_to_a_literal_refuses_named_and_classed() {
    let src = "fn main() -> i64 {\n\
               \x20   let !H = 32\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   t[0] = 1\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 1");

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"kind\":\"jit-ineligible\""), "must be classed, not a bare jit-error: {}", err_line);
    assert!(err_line.contains("\"code\":\"jit-dynamic-shape\""), "got: {}", err_line);
    assert!(err_line.contains("`H`") && err_line.contains("mutable"),
        "must name the binding and say it is mutable: {}", err_line);
    assert!(!err_line.contains("unbound shape parameter"),
        "must never fall back to the bare #480-class error: {}", err_line);
}

/// A value PARAMETER used as a shape dimension is not a `let` at all — it is
/// bound to a caller-supplied runtime value — so it stays refused, but named
/// and classed rather than the bare #480-class error.
#[test]
fn a_value_parameter_used_as_a_shape_dim_refuses_named_and_classed() {
    let src = "fn f(n: i64) -> i64 {\n\
               \x20   let !t = forge.zeros[i64,[n]]\n\
               \x20   t[0] = 1\n\
               \x20   sum(t)\n\
               }\n\
               \n\
               fn main() -> i64 { f(8) }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 1");

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"kind\":\"jit-ineligible\""), "must be classed, not a bare jit-error: {}", err_line);
    // #718 rework (R2): was `jit-type-form`; a parameter's runtime value is
    // not statically known, so `jit-dynamic-shape` is the accurate class.
    assert!(err_line.contains("\"code\":\"jit-dynamic-shape\""), "got: {}", err_line);
    assert!(err_line.contains("`n`") && err_line.contains("parameter"),
        "must name the binding and say it is a parameter: {}", err_line);
    assert!(!err_line.contains("unbound shape parameter"),
        "must never fall back to the bare #480-class error: {}", err_line);
}

/// A `for` loop index shadowing a foldable outer `let` of the same name is
/// NOT foldable inside the loop body — it is a fresh runtime binding, not the
/// outer `let`'s literal. Refused, classed, naming the binding — never a
/// silent fold to the OUTER `let`'s value (which would allocate the wrong
/// size) and never the bare #480-class error.
#[test]
fn a_loop_index_shadowing_a_foldable_let_refuses_inside_the_loop() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 4\n\
               \x20   let !total = 0\n\
               \x20   for H in 0..3 {\n\
               \x20       let !t = forge.zeros[i64,[H]]\n\
               \x20       total = total + H\n\
               \x20   }\n\
               \x20   total\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 3");

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"kind\":\"jit-ineligible\""), "must be classed, not a bare jit-error: {}", err_line);
    // #718 rework (R2): was `jit-type-form`; a loop index's runtime value is
    // not statically known, so `jit-dynamic-shape` is the accurate class.
    assert!(err_line.contains("\"code\":\"jit-dynamic-shape\""), "got: {}", err_line);
    assert!(err_line.contains("`H`"), "must name the binding: {}", err_line);
    assert!(!err_line.contains("unbound shape parameter"),
        "must never fall back to the bare #480-class error: {}", err_line);
}

// ─── #718 rework ────────────────────────────────────────────────────────────
//
// An adversarial review of this PR found five defects in the fold this file
// otherwise pins (the #712 rework's R1-R5, folded into the PR body): the fold
// ran every initializer in bare `i64` regardless of the binding's own integer
// WIDTH (R1), two of the three default-refusal arms above carried the wrong
// DIAGNOSTICS.md §5.2 class (R2, fixed by the assertion edits above), a
// float/bool literal initializer was misreported as "not known until the
// function runs" rather than "not an integer" (R3), a parenthesized literal
// expression (which parses as a 1-element tuple) wasn't folded at all (R4),
// and a lambda capturing a foldable local `let` refuses even though SPEC.md
// §3.5.1 says capturing closures are supported on both backends (R5, tracked
// in `tools/jit_probes.py`'s `GAP_ALLOWLIST`, not fixed here).

/// R1, row 1: `200u8 + 56u8` wraps in `u8` to 0 — `dmc run` evaluates `+` at
/// the binding's own width, not in `i64`. Folding in bare `i64` (as this PR
/// did before the rework) would give the JIT a `[256]` dimension where `run`
/// has a `[0]` one; proved here by writing `t[0]` on both backends and
/// checking BOTH report the tensor's size as 0 — the shapes agree without
/// going anywhere near the pre-existing, unrelated empty-`sum` print
/// divergence (`dmc run` prints `-0` for `sum` of an empty tensor where `dmc
/// jit` prints `0`; see the PR body).
#[test]
fn r1_row1_u8_add_wraps_to_zero_width_matches_run() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 200u8 + 56u8\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   t[0] = 1\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "dmc run must reject an OOB write into a [0] tensor");
    assert!(stderr(&run).contains("size 0"), "run: {}", stderr(&run));

    let jit = probe.run(&["jit"]);
    assert!(!jit.status.success(), "dmc jit must fold H to the SAME [0] dimension, not the unwrapped 256");
    assert!(stderr(&jit).contains("size 0"), "jit: {}", stderr(&jit));
}

/// R1, row 2: `let H: i32 = 2147483647 + 1` — an explicit `i32` annotation on
/// an untyped-literal sum, which must fold at `i32` (wrapping to
/// `i32::MIN`), not `i64`. The folded dimension is negative, which the JIT's
/// shape resolver has always refused outright (`resolve_shape_expr`, a rule
/// that predates and is out of scope for this PR) — proving the fold itself
/// landed on the width-correct, wrapped value (`-2147483648`, exactly what
/// `H as i64` gives on both backends) is the point of this row, not that the
/// two backends agree on what to do with a negative shape.
#[test]
fn r1_row2_explicit_i32_annotation_wraps_and_folds_at_that_width() {
    let src = "fn main() -> i64 {\n\
               \x20   let H: i32 = 2147483647 + 1\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"kind\":\"jit-ineligible\""), "must be classed, not a bare jit-error: {}", err_line);
    assert!(err_line.contains("\"code\":\"jit-type-form\""), "got: {}", err_line);
    assert!(err_line.contains("-2147483648"),
        "must fold at i32 width (wrapping to i32::MIN), not bare i64 (2147483648): {}", err_line);
}

/// R1, row 3: the same wraparound as row 2, but the width comes from the
/// LITERALS' own suffixes (`2147483647i32 + 1i32`) rather than an explicit
/// `let` annotation — `shape_dim_literal_width` must read that off the
/// initializer when there is no annotation to consult.
#[test]
fn r1_row3_literal_suffix_width_wraps_and_folds_at_that_width() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 2147483647i32 + 1i32\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"kind\":\"jit-ineligible\""), "must be classed, not a bare jit-error: {}", err_line);
    assert!(err_line.contains("\"code\":\"jit-type-form\""), "got: {}", err_line);
    assert!(err_line.contains("-2147483648"),
        "must fold at i32 width (wrapping to i32::MIN), not bare i64 (2147483648): {}", err_line);
}

/// R1, row 4 — the row the review named explicitly: `4294967295u32 + 7u32`
/// wraps to 6 in `u32`. Before the rework this was a BARE jit error (the
/// unwrapped i64 sum, 4294967302, read as a `[4294967302]` dimension the
/// allocator reports needing tens of GiB for) — must now fold to 6 and match
/// `dmc run` exactly, never a bare jit error again.
#[test]
fn r1_row4_u32_add_wraps_and_matches_run_never_a_bare_jit_error() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 4294967295u32 + 7u32\n\
               \x20   let !t = forge.ones[i64,[H]]\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 6");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must fold H to 6 (the u32-wrapped value), not error: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), "=> 6");
    assert_eq!(stdout(&jit).trim(), stdout(&run).trim());
}

/// #718 review, second edit: `fold_shape_expr`'s DEFAULT (untyped, `i64`)
/// arithmetic arm used bare `l + r` / `l * r` before this fix — a debug-build
/// overflow there panics the compiler instead of returning a clean refusal,
/// where `fold_shape_expr_at_width` (the narrower-width sibling this arm now
/// matches) already wraps via `interp::int_arith`. `9223372036854775807 + 1`
/// wraps to `i64::MIN` on both backends (`dmc run`'s `int_arith` uses
/// `wrapping_add`, #300's ruling); the JIT's shape resolver already refuses
/// ANY negative dimension (pre-existing, out of scope here), so the proof
/// this row cares about is that folding the wrapped value doesn't panic and
/// the refusal names the correctly-wrapped `-9223372036854775808`, not that
/// the two backends agree on what to do with a negative shape.
#[test]
fn overflowing_i64_shape_arithmetic_wraps_instead_of_panicking() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 9223372036854775807 + 1\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1), "must be a clean refusal, never a debug-build overflow panic: {}", stderr(&out));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"code\":\"jit-type-form\""), "got: {}", err_line);
    assert!(err_line.contains("-9223372036854775808"),
        "must fold via wrapping_add (i64::MIN), not panic on overflow: {}", err_line);
}

/// #718 review, second edit: the OLD `Div if r != 0` guard fell through to
/// the catch-all `_` arm for `r == 0`, refusing as an unfoldable operator
/// (`jit-dynamic-shape`, "the JIT folds `+ - * /` only") — different from
/// `int_arith`'s div-by-zero-is-0 rule (#208), which narrower-width lets
/// already got via `fold_shape_expr_at_width`. Routing this arm through
/// `int_arith` too makes the DEFAULT (untyped `i64`) width agree with a
/// narrow one: `4 / 0` must fold to 0 and lower cleanly, exactly like
/// `4u8 / 0u8` already does, not refuse.
#[test]
fn i64_shape_div_by_zero_folds_to_zero_like_a_narrow_width_does() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 4 / 0\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(),
        "dmc jit must fold H to 0 (int_arith's div-by-zero rule), not refuse: {}", stderr(&jit));

    let src_u8 = "fn main() -> i64 {\n\
               \x20   let H = 4u8 / 0u8\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   sum(t)\n\
               }\n";
    let probe_u8 = Probe::new(src_u8);
    let jit_u8 = probe_u8.run(&["jit"]);
    assert!(jit_u8.status.success(), "narrow-width div-by-zero must already fold: {}", stderr(&jit_u8));
    assert_eq!(stdout(&jit).trim(), stdout(&jit_u8).trim(),
        "the default i64 width and an explicit narrow width must agree on div-by-zero");
}

/// R3: a float literal initializer is known at fold time and known NOT to be
/// an integer — a `NotAnInt` verdict, not the generic `LocalUnfoldable`
/// "not known until the function runs" message, which was wrong on both
/// counts for `let H = 4.0`. Also worded origin-neutrally (never "the module
/// constant `H`" for a LOCAL binding).
#[test]
fn r3_float_literal_initializer_is_not_an_int_not_unfoldable() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 4.0\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"code\":\"jit-type-form\""), "got: {}", err_line);
    assert!(err_line.contains("`H`") && err_line.contains("not an integer"),
        "must say H is known and simply not an integer: {}", err_line);
    assert!(!err_line.contains("not known until the function runs"),
        "a float literal IS known at fold time: {}", err_line);
    assert!(!err_line.contains("module constant"),
        "H is a LOCAL, not a module constant: {}", err_line);
}

/// R3's bool twin: `let H = true` is the same "known, not an integer" verdict.
#[test]
fn r3_bool_literal_initializer_is_not_an_int_not_unfoldable() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = true\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"code\":\"jit-type-form\""), "got: {}", err_line);
    assert!(err_line.contains("`H`") && err_line.contains("not an integer"), "got: {}", err_line);
    assert!(!err_line.contains("not known until the function runs"), "got: {}", err_line);
}

/// #718 review: R3's `NotAnInt` verdict only fired for a BARE float/bool
/// literal initializer — `let H = 4.0 + 1.0` (`dmc run` gives 5.0) is still
/// known-not-an-integer at fold time, but the initializer is a `BinOp`, not a
/// `Literal`, so it fell through to `LocalUnfoldable`'s "not known until the
/// function runs" message, which is wrong: the value is known, it just isn't
/// an integer. `is_float_or_bool_leaved` extends the bare-literal check to an
/// arithmetic initializer whose leaves are all float (or bool) literals.
#[test]
fn float_leaved_arithmetic_initializer_is_not_an_int_not_unfoldable() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 4.0 + 1.0\n\
               \x20   let !t = forge.zeros[i64,[H]]\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"code\":\"jit-type-form\""), "got: {}", err_line);
    assert!(err_line.contains("`H`") && err_line.contains("not an integer"),
        "must say H is known and simply not an integer: {}", err_line);
    assert!(!err_line.contains("not known until the function runs"),
        "a float-leaved arithmetic initializer IS known at fold time: {}", err_line);
    assert!(!err_line.contains("module constant"),
        "H is a LOCAL, not a module constant: {}", err_line);
}

/// R4: `(2 + 3) * 4` — a parenthesized expression parses as a 1-element
/// tuple (the same surface `interp.rs`'s `eval_expr` unwraps transparently),
/// so `fold_shape_expr`'s catch-all refused it even though the PR's
/// "foldable expressions of literals" claim says it shouldn't be. Must fold
/// to 20 and match `dmc run`.
#[test]
fn r4_parenthesized_literal_expression_folds() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = (2 + 3) * 4\n\
               \x20   let !t = forge.ones[i64,[H]]\n\
               \x20   sum(t)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 20");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must fold the parenthesized expression, not refuse it: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), "=> 20");
    assert_eq!(stdout(&jit).trim(), stdout(&run).trim());
}

/// R5: a foldable local `let` CAPTURED by a lambda and used as a shape
/// dimension inside its body refuses — `jit-dynamic-shape`, named and
/// classed, never a silent divergence — even though SPEC.md §3.5.1's JIT
/// note says capturing closures are supported on both backends. `dmc run`
/// closes over `H` normally; the JIT's per-fn `local_const_dims` fold table
/// isn't visible inside a deferred lambda body. Tracked as a gap
/// (`tools/jit_probes.py`'s `GAP_ALLOWLIST`,
/// `shape_local_let_capture_in_lambda_body_718`), not fixed by this rework.
#[test]
fn r5_lambda_capture_of_a_foldable_local_let_refuses_named_and_classed() {
    let src = "fn main() -> i64 {\n\
               \x20   let H = 4\n\
               \x20   let f = fn() -> i64 {\n\
               \x20       let !t = forge.zeros[i64,[H]]\n\
               \x20       t[0] = 1\n\
               \x20       sum(t)\n\
               \x20   }\n\
               \x20   f()\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 1");

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"kind\":\"jit-ineligible\""), "must be classed, not a bare jit-error: {}", err_line);
    assert!(err_line.contains("\"code\":\"jit-dynamic-shape\""), "got: {}", err_line);
    assert!(err_line.contains("`H`"), "must name the binding: {}", err_line);
    assert!(!err_line.contains("unbound shape parameter"),
        "must never fall back to the bare #480-class error: {}", err_line);
}
