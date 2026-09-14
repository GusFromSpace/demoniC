//! #810 — a **bare** model parameter (`fn g(a: Buf, b: Buf)`) forwarded to an
//! explicitly bracketed call (`f[8](a, b)`) binds nothing and checks nothing.
//!
//! `unify_param_type`'s model arm (`jit.rs` ~6925, opened by #732/#733's
//! `jit_shape_arg_inference_732_733.rs`) binds a template's shape parameter
//! from a model ARGUMENT's own shape arguments — `mdims.get(i)`. A bare
//! parameter (`a: Buf`, no bracket) lowers to `TyKind::Model("Buf", [])`
//! *inside `g`'s body*: `g`'s own parameter carries no shape, regardless of
//! what concrete instance the caller actually passed. The old comment at that
//! site assumed this was safe because `instantiate_template`'s "every
//! declared shape_param must end up bound" sweep would refuse by name a
//! moment later — true only while the shape parameter is still unbound when
//! that sweep runs. An explicit bracket (`f[8](a, b)`) seeds `bindings["N"] =
//! 8` BEFORE the model arm ever runs, so the sweep finds `N` already bound and
//! never fires, and nothing ever compared the bracket's `8` against `a`'s
//! real shape. `g(a, b)` called with `a` a 4-element `Buf` and `b` an
//! 8-element one, `dmc jit` read `a.t[7]` — three i64s past the end of a
//! 4-element tensor — and answered a number; `dmc run`'s runtime contradiction
//! check ("shape param `N` would need to be both 8 and 4") refuses.
//!
//! **Rework (reviewer pass on this PR):** the original fix only refused when
//! the shape parameter's name happened to be already bound AND uppercase by
//! the time this argument was visited — so the very same unsound call still
//! read out of bounds if the bare argument came first (`fn g(b: Buf, a:
//! Buf[8]) { f(b, a) }`), if the binding came from a later TENSOR argument
//! rather than a bracket (`fn f[N](x: Buf[N], k: Tensor[i64,[N]])` called
//! bare), or if the template's shape parameter was spelled lowercase (`fn
//! f[n]`, which `bind_or_check_shape_param`'s inference-only uppercase
//! convention (#733) does not gate an *explicit* bracket binding on). The
//! rework (`bind_or_refuse_model_shape_args`, shared by the scalar `Model`
//! arm and the `ModelArray` element arm below) refuses UNCONDITIONALLY
//! whenever the argument's own dims are empty at that position, regardless
//! of ordering, binding source, or case — mirroring `jit.rs` ~12431's "a
//! receiver whose type carries no arguments is refused, not inferred": a
//! model value at run time is only the i64 pointer to its field arena, no
//! header, so there is never a runtime shape to fall back on checking
//! instead. The refusal is now anchored at the CALL span (not the callee's
//! own parameter declaration) and worded like `check.rs`'s bare-generic-model
//! hint: "declare the shape parameter on the function: `fn g[N](a:
//! Buf[N])`". #480 doctrine: a wrong answer is the worst outcome, and an
//! unlowerable form is a located refusal, as `jit-type-form`
//! (`docs/DIAGNOSTICS.md §5.2`: "a type in a signature, annotation, or model
//! field the JIT cannot build").
//!
//! **#812 folded in:** the same unchecked-bare-model gap existed in the
//! `ModelArray` arm (`[ModelName; N]` parameters) — it checked only that the
//! element model's NAME matched, never its shape, so a bare model-array
//! element (`x: [Buf; 2]`) forwarded to a shape-parameterized element (`arr:
//! [Buf[N]; 2]`) read out of bounds exactly as a bare scalar model argument
//! did. Same fix, same helper, same wording.
//!
//! Each fixture drives the real `dmc` binary and asserts both backends, same
//! harness as `jit_shape_arg_inference_732_733.rs`.

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

    fn text(&self, o: &Output) -> String {
        let mut s = String::from_utf8_lossy(&o.stdout).to_string();
        s.push_str(&String::from_utf8_lossy(&o.stderr));
        s.trim().to_string()
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

    /// `(succeeded, combined output)` — for the cases where the point is that
    /// a backend REFUSES.
    fn try_run(&self) -> (bool, String) {
        let out = self.dmc(&["run"]);
        (out.status.success(), self.text(&out))
    }

    fn try_jit(&self) -> (bool, String) {
        let out = self.dmc(&["jit"]);
        (out.status.success(), self.text(&out))
    }

    /// stdout alone, trimmed — for a fixture that also emits a `#728`
    /// bare-generic-model WARNING (on stderr) alongside a successful answer,
    /// where `run_output`/`jit_output`'s combined stdout+stderr text would no
    /// longer equal just the answer.
    fn stdout_only(&self, out: &Output) -> String {
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }
}

/// **The issue's own program.** `f[8](a, b)` inside `g`, whose parameters are
/// bare (`a: Buf`, `b: Buf`). `a` is a 4-element `Buf`, `b` an 8-element one.
/// Before the fix, `dmc jit` answered `100`, having read `a.t[7]` off the end
/// of `a`'s 4-element tensor. `dmc run` has always refused this contradiction.
#[test]
fn the_issue_810_program_refuses_on_the_jit_with_runs_refusal() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn f[N](x: Buf[N], y: Buf[N]) -> i64 { x.t[N - 1] + y.t[N - 1] }\n\
         fn g(a: Buf, b: Buf) -> i64 { f[8](a, b) }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   a.t[3] = 5\n\
         \x20   b.t[7] = 100\n\
         \x20   g(a, b)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "the interpreter has always refused this contradiction: {run_out}");
    assert!(
        run_out.contains("shape param `N` would need to be both 8 and 4"),
        "interpreter wording: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(
        !jit_ok,
        "#810: `dmc jit` must not answer `100` here, reading `a.t[7]` out of bounds: {jit_out}",
    );
    assert!(
        jit_out.contains("jit unsupported"),
        "a located refusal, not an uncoded `jit error`: {jit_out}",
    );
    assert!(
        jit_out.contains("carries no shape arguments") && jit_out.contains("shape parameter `N`"),
        "must name the unresolved shape parameter: {jit_out}",
    );
    assert!(
        // This fixture's enclosing fn and bare parameter both happen to be
        // named `g`/`a` — the exact strings the hint used to hardcode — so
        // this assertion alone cannot tell a correctly-threaded hint from
        // the old literal placeholder. `a_bare_model_arg_names_the_real_fn_
        // and_param_810` below uses different names (`caller`/`first`)
        // specifically to catch that.
        jit_out.contains("fn g[N](a: Buf[N])"),
        "must give the check.rs-style declare-it-on-the-function hint: {jit_out}",
    );
}

/// **Reviewer pass: the hint must name the REAL enclosing fn and parameter,**
/// not the literal placeholder strings `g`/`a` the old hint hardcoded
/// (`the_issue_810_program_refuses_on_the_jit_with_runs_refusal` above
/// couldn't catch this — its own fn and parameter are coincidentally named
/// `g`/`a` too). Here the enclosing fn is `caller` and its bare parameter is
/// `first`; before the fix this still printed `fn g[N](a: Buf[N])` regardless.
#[test]
fn a_bare_model_arg_names_the_real_fn_and_param_810() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn f[N](x: Buf[N], y: Buf[N]) -> i64 { x.t[N - 1] + y.t[N - 1] }\n\
         fn caller(first: Buf, b: Buf) -> i64 { f[8](first, b) }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   a.t[3] = 5\n\
         \x20   b.t[7] = 100\n\
         \x20   caller(a, b)\n\
         }\n",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(
        !jit_ok,
        "#810: `dmc jit` must not answer here, reading `first.t[7]` out of bounds: {jit_out}",
    );
    assert!(jit_out.contains("jit unsupported"), "a located refusal: {jit_out}");
    assert!(
        jit_out.contains("carries no shape arguments") && jit_out.contains("shape parameter `N`"),
        "must name the unresolved shape parameter: {jit_out}",
    );
    assert!(
        jit_out.contains("fn caller[N](first: Buf[N])"),
        "must name the real enclosing fn (`caller`) and its real bare parameter \
         (`first`), not the hardcoded `fn g[N](a: Buf[N])` placeholder: {jit_out}",
    );
}

/// **The `f[4]` variant.** Same forwarding, but the bracket matches `a`'s own
/// shape (4) instead of `b`'s. Before the fix this answered `105` — `a.t[3]`
/// (5) plus `b.t[3]` (100, well inside `b`'s 8 elements) — a value that LOOKS
/// plausible precisely because it isn't out of bounds. But `a` is `Buf[4]`
/// and `b` is `Buf[8]`: the interpreter refuses this call too (the contract is
/// "every argument's shape agrees with the bracket", not merely "this read
/// happened not to fault"), so the JIT must refuse it as well rather than
/// answer a number that is coincidentally in-bounds.
#[test]
fn the_f4_variant_also_refuses_even_though_the_old_read_was_in_bounds() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn f[N](x: Buf[N], y: Buf[N]) -> i64 { x.t[N - 1] + y.t[N - 1] }\n\
         fn g(a: Buf, b: Buf) -> i64 { f[4](a, b) }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   a.t[3] = 5\n\
         \x20   b.t[3] = 100\n\
         \x20   g(a, b)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "the interpreter refuses `f[4]` too: {run_out}");
    assert!(
        run_out.contains("shape param `N` would need to be both 4 and 8"),
        "interpreter wording: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(
        !jit_ok,
        "the JIT must not answer `105` here either: {jit_out}",
    );
    assert!(jit_out.contains("jit unsupported"), "a located refusal: {jit_out}");
    assert!(
        jit_out.contains("carries no shape arguments") && jit_out.contains("shape parameter `N`"),
        "must name the unresolved shape parameter: {jit_out}",
    );
}

/// **The control.** A correctly matched bracketed call through the same
/// forwarding shape must still lower — the fix must not refuse an honest
/// agreement, only a contradiction it can no longer verify.
#[test]
fn a_correctly_matched_bracketed_call_still_lowers() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn f[N](x: Buf[N], y: Buf[N]) -> i64 { x.t[N - 1] + y.t[N - 1] }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !b = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   a.t[3] = 5\n\
         \x20   b.t[3] = 7\n\
         \x20   f[4](a, b)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 12");
    assert_eq!(probe.jit_output(), "=> 12");
}

/// **Reversed parameter order.** The same forwarding as the issue's own
/// program, but `g`'s bare parameter comes FIRST and the shape-carrying one
/// SECOND (`fn g(b: Buf, a: Buf[8]) { f(b, a) }`, no explicit bracket at
/// all — `N` is meant to be inferred from `a`). Before the rework, the old
/// guard only fired when `N` was ALREADY in `bindings` by the time the bare
/// argument was visited; visited first, it wasn't, so this order slipped
/// through and read `b.t[7]` off the end of `b`'s real 4-element tensor.
/// The unconditional refusal catches it regardless of visit order.
#[test]
fn the_reversed_parameter_order_refuses_too() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn f[N](x: Buf[N], y: Buf[N]) -> i64 { x.t[N - 1] + y.t[N - 1] }\n\
         fn g(b: Buf, a: Buf[8]) -> i64 { f(b, a) }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   let !b = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   a.t[7] = 100\n\
         \x20   b.t[3] = 5\n\
         \x20   g(b, a)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "the interpreter refuses this reordering too: {run_out}");
    assert!(
        run_out.contains("shape param `N` would need to be both 4 and 8"),
        "interpreter wording: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(
        !jit_ok,
        "the JIT must refuse `b` first, before `a` ever binds `N`: {jit_out}",
    );
    assert!(jit_out.contains("jit unsupported"), "a located refusal: {jit_out}");
    assert!(
        jit_out.contains("carries no shape arguments") && jit_out.contains("shape parameter `N`"),
        "must name the unresolved shape parameter: {jit_out}",
    );
}

/// **A tensor argument binds the shape parameter, not a bracket.** No
/// explicit bracket at all: `N` is meant to be inferred from `k`'s tensor
/// shape, and `k` is visited AFTER the bare model argument. Before the
/// rework, the old guard's "already bound" condition was false when the bare
/// `x: Buf[N]` argument was visited (nothing had bound `N` yet), so it fell
/// through, and `N` was only bound moments later from `k`'s real shape (8) —
/// too late to have checked `x` against it. `x.t[N - 1]` then read `a.t[7]`
/// off the end of `a`'s real 4-element tensor.
#[test]
fn a_bare_model_forwarded_beside_a_shape_binding_tensor_refuses() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn f[N](x: Buf[N], k: Tensor[i64, [N]]) -> i64 { x.t[N - 1] + k[N - 1] }\n\
         fn g(a: Buf, k: Tensor[i64, [8]]) -> i64 { f(a, k) }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !k = forge.zeros[i64, [8]]\n\
         \x20   a.t[3] = 5\n\
         \x20   k[7] = 100\n\
         \x20   g(a, k)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "the interpreter refuses this contradiction too: {run_out}");
    assert!(
        run_out.contains("shape param `N` would need to be both 4 and 8"),
        "interpreter wording: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(
        !jit_ok,
        "the JIT must refuse `x: Buf[N]` before `k` ever binds `N`: {jit_out}",
    );
    assert!(jit_out.contains("jit unsupported"), "a located refusal: {jit_out}");
    assert!(
        jit_out.contains("carries no shape arguments") && jit_out.contains("shape parameter `N`"),
        "must name the unresolved shape parameter: {jit_out}",
    );
}

/// **A lowercase shape parameter.** `fn f[n]` — `bind_or_check_shape_param`'s
/// uppercase-only convention (#733) is about free INFERENCE, but an explicit
/// bracket (`f[8](a, b)`) is inserted into `bindings` without that filter
/// (`instantiate_template`'s explicit-binding loop), so `bindings` held a
/// lowercase key the old guard's `pn.chars().next().is_some_and(is_uppercase)`
/// conjunct could never match — the refusal never fired for any lowercase
/// shape parameter at all, bound or not. Dropping that conjunct closes it.
#[test]
fn a_lowercase_shape_param_still_refuses() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn f[n](x: Buf[n], y: Buf[n]) -> i64 { x.t[n - 1] + y.t[n - 1] }\n\
         fn g(a: Buf, b: Buf) -> i64 { f[8](a, b) }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   a.t[3] = 5\n\
         \x20   b.t[7] = 100\n\
         \x20   g(a, b)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "the interpreter refuses this out-of-bounds read too: {run_out}");

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(
        !jit_ok,
        "a lowercase shape parameter must refuse exactly like an uppercase one: {jit_out}",
    );
    assert!(jit_out.contains("jit unsupported"), "a located refusal: {jit_out}");
    assert!(
        jit_out.contains("shape parameter `n`") && jit_out.contains("fn g[n](a: Buf[n])"),
        "must name the lowercase shape parameter: {jit_out}",
    );
}

/// **#812: a bare model-array element.** `[ModelName; N]` template
/// parameters went through a separate arm of `unify_param_type` that checked
/// only the element model's NAME, never its shape — so a bare model-array
/// argument (`x: [Buf; 2]`) forwarded to a shape-parameterized element
/// (`arr: [Buf[N]; 2]`) bound and checked nothing, the same gap #810 fixed
/// for a scalar model parameter. `arr[0].t[N - 1]` on a 4-element `Buf`
/// packed alongside an 8-element sibling read `a.t[7]` out of bounds.
#[test]
fn a_bare_model_array_element_refuses_812() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn f[N](arr: [Buf[N]; 2]) -> i64 { arr[0].t[N - 1] + arr[1].t[N - 1] }\n\
         fn g(x: [Buf; 2]) -> i64 { f[8](x) }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   a.t[3] = 5\n\
         \x20   b.t[7] = 100\n\
         \x20   let !x = forge.uninit[Buf, [2]]\n\
         \x20   x[0] = a\n\
         \x20   x[1] = b\n\
         \x20   g(x)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "the interpreter refuses this out-of-bounds read too: {run_out}");

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(
        !jit_ok,
        "#812: `dmc jit` must not read `arr[0].t[7]` off a 4-element `Buf`: {jit_out}",
    );
    assert!(jit_out.contains("jit unsupported"), "a located refusal: {jit_out}");
    assert!(
        jit_out.contains("carries no shape arguments") && jit_out.contains("shape parameter `N`"),
        "must name the unresolved shape parameter: {jit_out}",
    );
    assert!(
        // Reviewer pass: the hint used to hardcode a scalar `(a: Buf[N])`
        // parameter here too, even though `g`'s real parameter is `x` and its
        // real type is the ARRAY `[Buf; 2]` — a fix a reader couldn't apply
        // as written. It must name `g`'s own parameter (`x`) and render the
        // array form (`[Buf[N]; 2]`), not a bare scalar.
        jit_out.contains("fn g[N](x: [Buf[N]; 2])"),
        "must give the check.rs-style declare-it-on-the-function hint, naming the \
         real parameter and its array type: {jit_out}",
    );
}

/// **A non-parameterized callee still lowers.** `h`'s own parameter (`x:
/// Buf`) is bare, but `h` is a plain function — not a template with a shape
/// parameter to check `x` against — so `unify_param_type` never runs for
/// this call at all. The fix must not touch calls that were never the bug.
#[test]
fn a_non_generic_callee_still_lowers() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn h(x: Buf) -> i64 { 7 }\n\
         fn g(a: Buf) -> i64 { h(a) }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   g(a)\n\
         }\n",
    );

    let run_out = probe.dmc(&["run"]);
    assert!(run_out.status.success(), "dmc run failed:\n{}", probe.text(&run_out));
    assert_eq!(probe.stdout_only(&run_out), "=> 7");

    let jit_out = probe.dmc(&["jit"]);
    assert!(jit_out.status.success(), "dmc jit failed:\n{}", probe.text(&jit_out));
    assert_eq!(probe.stdout_only(&jit_out), "=> 7");
}

/// **A model with zero shape parameters, passed bare, still lowers.**
/// `Other` declares no shape parameters at all, so its own model argument
/// carries an empty type-arg list on BOTH sides (the parameter's annotation
/// and the argument's runtime dims) — there is nothing for the new refusal
/// to ever compare, and the loop it guards never iterates. The template's
/// actual shape parameter `M` is bound from the ordinary `Tensor` argument
/// beside it, untouched by this fix.
#[test]
fn a_zero_shape_param_model_passed_bare_still_lowers() {
    let probe = Probe::new(
        "model Other { !v: Tensor[i64, [1]] }\n\
         fn f[M](x: Other, t: Tensor[f32, [M]]) -> f32 { t[M - 1] }\n\
         fn g(x: Other, t: Tensor[f32, [4]]) -> f32 { f[4](x, t) }\n\
         fn main() -> f32 {\n\
         \x20   let !o = Other { v: forge.zeros[i64, [1]] }\n\
         \x20   let !t = forge.zeros[f32, [4]]\n\
         \x20   t[3] = 2.5\n\
         \x20   g(o, t)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 2.5");
    assert_eq!(probe.jit_output(), "=> 2.5");
}
