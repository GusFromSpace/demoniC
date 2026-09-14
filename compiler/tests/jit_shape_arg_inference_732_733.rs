//! The generic-model-params design's slice 0 — the JIT binds a
//! template's shape parameters from a **model argument's own shape
//! arguments**, refuses two arguments that disagree, and infers
//! uppercase-initial names only.
//!
//! `SPEC.md §3.10` (line 453): "A shape-parameterized model type is legal as a
//! field type (`!surface: Surface[H, W]`) and as a parameter type (`!b: Box[H,
//! W]`) … an argument's own shape arguments bind the parameter's shape names",
//! and (line 454) "inference binds uppercase-initial names only, for models and
//! tensors alike". The interpreter implements both — `harvest_shape_params`
//! reads a model instance's hidden `__shape_P__` fields, `bind_shape_name`
//! gates on the initial and raises on a contradiction — and the JIT implemented
//! neither, for two independent reasons in one `match` arm.
//!
//! **The inference hole.** `unify_param_type`'s model arm has bound ident type
//! args positionally since #273, but never fired for the spelling anyone
//! writes: `N` in a TYPE position (`b: Buf[N]`) parses as
//! `TypeArg::Type(Type::Named { name: "N", args: [] })`, not
//! `TypeArg::Expr(Expr::Ident("N"))`, and the arm's `TypeArg::Type(_) =>
//! continue` dropped it. So `fn cap[N](b: Buf[N])` called `cap(a)` on a `Buf[4]`
//! was `jit error: could not infer shape parameter `N` of `cap` from call args`
//! while `dmc run` answered — the design's g15/g23/g24. The forward direction
//! already read that exact spelling (`type_named_dims` resolves a bare
//! type-name arg through the shape env), which is why only this half was
//! missing: the JIT could build `Buf[N]` from an env but not an env from a
//! `Buf[4]`.
//!
//! **#732 — the arm did not compare.** It bound with
//! `bindings.entry(pn).or_insert(actual)`: first argument wins, later ones
//! silently discarded. The "would bind to N but already bound to M" sentence
//! lived only in the `Tensor`/`KV` arms, so `fn f[N](x: Buf[N], y: Buf[N])`
//! called `f[4](a, b)` on a `Buf[4]` and a `Buf[8]` was refused by `dmc run`
//! ("would need to be both 4 and 8") and answered `105` by `dmc jit`, which
//! pinned `N = 4` and read `y.t[3]` off the larger buffer. With the two buffers
//! the other way round that is an out-of-bounds read. Reachable today through
//! the explicit bracket, so it is a live divergence and not merely hardening
//! for the inference this slice opens.
//!
//! **#733 — inference was not gated on the initial.** The `Tensor` arm bound
//! any ident, so `fn head[n](x: Tensor[i64,[n]])` called `head(a)` answered `5`
//! under `dmc jit` against a `dmc run` refusal. The model spelling refused on
//! both only because the arm never fired, so opening it without the gate would
//! have handed the model arm the same defect.
//!
//! Both fixes are one helper, `bind_or_check_shape_param`, that every arm —
//! `Tensor`, `KV`, `KV`-array, model — now routes through, the JIT-side twin of
//! `interp.rs`'s `bind_shape_name`, matched down to its rule order: the
//! uppercase gate runs before the already-bound check, not after. A lowercase
//! name is never *bound* by inference, and — because that gate runs first — a
//! lowercase name the CALL SITE spelled is never *checked* against the
//! argument either; it is taken as written, same as the interpreter.
//!
//! Each fixture drives the real `dmc` binary and asserts both backends.

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

    /// `(succeeded, combined output)` — for the cases where the point is that a
    /// backend REFUSES.
    fn try_run(&self) -> (bool, String) {
        let out = self.dmc(&["run"]);
        (out.status.success(), self.text(&out))
    }

    fn try_jit(&self) -> (bool, String) {
        let out = self.dmc(&["jit"]);
        (out.status.success(), self.text(&out))
    }

    /// `(passed, combined output)` for `--check` alone — used where the point
    /// is that the checker does NOT refuse (`model_arg_binds_at_runtime`'s
    /// escape), so the contradiction is left for a backend's own runtime arm.
    fn try_check(&self) -> (bool, String) {
        let out = self.dmc(&["--check"]);
        (out.status.success(), self.text(&out))
    }

    /// Every `--json --all` diagnostic, the summary excluded. `--all` reports
    /// every refusal rather than stopping at the first, which is what makes one
    /// refusal *per declared symbol* visible.
    fn jit_json_all(&self) -> Vec<String> {
        let out = self.dmc(&["jit", "--json", "--all"]);
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .filter(|l| l.starts_with('{') && !l.contains("\"kind\":\"summary\""))
            .map(|l| l.to_string())
            .collect()
    }
}

/// A `--json` line's string field. The messages here contain no escapes, so a
/// scan to the next `"` is enough and pulls in no JSON dependency.
fn field_of(line: &str, key: &str) -> String {
    let needle = format!("\"{key}\":\"");
    let start = line.find(&needle).unwrap_or_else(|| panic!("no `{key}` in {line}"))
        + needle.len();
    let rest = &line[start..];
    let end = rest.find('"').expect("unterminated field");
    rest[..end].to_string()
}

// ─── what slice 0 lowers ─────────────────────────────────────────────────────

/// **g15** — the design's headline row. `fn cap[N](b: Buf[N])` called `cap(a)`
/// with no bracket, at two instantiations in one program. `dmc run` has always
/// answered `12`, `24`; `dmc jit` was `jit error: could not infer shape
/// parameter `N` of `cap` from call args`.
///
/// The body names `N`, so this also asserts the inferred dim reaches the mono
/// fn's `shape_env` and not merely the parameter's field types.
#[test]
fn a_shape_param_is_inferred_from_a_model_arguments_own_args_g15() {
    let probe = Probe::new(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   n: i64\n\
         }\n\
         fn cap[N](b: Buf[N]) -> i64 { b.t[0] + b.n + N }\n\
         fn main() -> nil {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]], n: 3 }\n\
         \x20   let !c = Buf[8] { t: forge.zeros[i64, [8]], n: 9 }\n\
         \x20   a.t[0] = 5\n\
         \x20   c.t[0] = 7\n\
         \x20   print(to_str(cap(a)))\n\
         \x20   print(to_str(cap(c)))\n\
         \x20   nil\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "12\n24");
    assert_eq!(probe.jit_output(), "12\n24", "the JIT must agree with `dmc run`");
}

/// **g23** — the argument written as the literal itself rather than through a
/// binding. `lower_struct_lit` computes `type_args` from the written bracket,
/// so the `TyKind::Model` reaching `unify_param_type` carries its dims either
/// way; this pins that the call-argument path and the `let`-bound path agree.
#[test]
fn the_argument_may_be_the_model_literal_itself_g23() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn cap[N](b: Buf[N]) -> i64 { b.t[0] + N }\n\
         fn main() -> i64 {\n\
         \x20   cap(Buf[4] { t: forge.zeros[i64, [4]] })\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 4");
    assert_eq!(probe.jit_output(), "=> 4");
}

/// **g24** — the minimal failure: one call, one instantiation, through a `let`.
#[test]
fn a_single_inferred_call_lowers_g24() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn cap[N](b: Buf[N]) -> i64 { b.t[0] + N }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   cap(a)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 4");
    assert_eq!(probe.jit_output(), "=> 4");
}

/// **g16 — the control that must not move.** The same callee with the bracket
/// spelled at the call lowered before this slice and must lower after it: the
/// explicit bracket seeds `bindings` before unification, and the model arm now
/// *checks* against that seed rather than ignoring it, so a regression here
/// would mean the check is rejecting an agreement.
#[test]
fn an_explicit_bracket_still_lowers_unchanged_g16() {
    let probe = Probe::new(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   n: i64\n\
         }\n\
         fn cap[N](b: Buf[N]) -> i64 { b.t[0] + b.n + N }\n\
         fn main() -> nil {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]], n: 3 }\n\
         \x20   let !c = Buf[8] { t: forge.zeros[i64, [8]], n: 9 }\n\
         \x20   a.t[0] = 5\n\
         \x20   c.t[0] = 7\n\
         \x20   print(to_str(cap[4](a)))\n\
         \x20   print(to_str(cap[8](c)))\n\
         \x20   nil\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "12\n24");
    assert_eq!(probe.jit_output(), "12\n24");
}

/// **The #474 regression gate for the inferred spelling.** Two instantiations
/// reached with no bracket at either call must be **two symbols**, not one body
/// serving both — #474 and #475 were both silent wrong answers, and a shared
/// body here answers `9` twice instead of `9` and `15`.
///
/// The behaviour is asserted first, then the mechanism: the second program adds
/// a construct the JIT does not lower to the shared body (`t[i] += v`, the
/// tracked #693 gap), which makes each declared symbol name observable —
/// `--json --all` reports every refusing item by name — and it reports `cap$N4`
/// and `cap$N8`, `mangle_mono_name`'s form, not one name twice.
#[test]
fn two_instantiations_reached_by_inference_are_two_symbols() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn cap[N](b: Buf[N]) -> i64 { b.t[0] + N }\n\
         fn main() -> nil {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !c = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   a.t[0] = 5\n\
         \x20   c.t[0] = 7\n\
         \x20   print(to_str(cap(a)))\n\
         \x20   print(to_str(cap(c)))\n\
         \x20   nil\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "9\n15");
    assert_eq!(probe.jit_output(), "9\n15", "not `9` and `9` — one body per instantiation");

    let observable = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn cap[N](!b: Buf[N]) -> i64 {\n\
         \x20   b.t[0] += 1\n\
         \x20   b.t[0] + N\n\
         }\n\
         fn main() -> nil {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !c = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   print(to_str(cap(a)))\n\
         \x20   print(to_str(cap(c)))\n\
         \x20   nil\n\
         }\n",
    );
    let items: Vec<String> = observable
        .jit_json_all()
        .iter()
        .filter(|l| l.contains("\"kind\":\"jit-ineligible\""))
        .map(|l| field_of(l, "item"))
        .collect();
    assert!(items.iter().any(|i| i == "cap$N4"), "declared at [4]: {items:?}");
    assert!(items.iter().any(|i| i == "cap$N8"), "declared at [8]: {items:?}");
}

/// **#717's shadow rule survives the new binding source.** A method may declare
/// a shape parameter shadowing its model's, which is why the receiver's dims
/// are bound under `receiver_shape_key`'s unspellable `Buf$$N`. A free function
/// inferring its OWN `N` from a model argument whose model also declares `N` is
/// a third `N` in the same program, and the three must not meet: `cap`'s `N` is
/// the instance's `4`, `mix`'s `N` is the bracket's `2`, and the model's own `N`
/// still types `self.t`.
///
/// `9` = `a.t[0]` (5) + `cap`'s inferred `N` (4); `52` = `self.t[0]` (5) × 10 +
/// `mix`'s explicit `N` (2).
#[test]
fn the_717_shadow_rule_holds_when_a_fns_own_n_is_inferred_from_a_model_arg() {
    let probe = Probe::new(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn mix[N](self, x: Tensor[i64, [N]]) -> i64 { self.t[0] * 10 + N }\n\
         }\n\
         fn cap[N](b: Buf[N]) -> i64 { b.t[0] + N }\n\
         fn main() -> nil {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   a.t[0] = 5\n\
         \x20   let !x = forge.zeros[i64, [2]]\n\
         \x20   print(to_str(cap(a)))\n\
         \x20   print(to_str(a.mix[2](x)))\n\
         \x20   nil\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "9\n52");
    assert_eq!(probe.jit_output(), "9\n52", "the three `N`s must stay separate");
}

// ─── #732: two model arguments that disagree ─────────────────────────────────

/// **#732, closed; superseded by #792.** `f[4](a, b)` on a `Buf[4]` and a
/// `Buf[8]`. `dmc run` used to refuse only at the interpreter's own harvest
/// ("would need to be both 4 and 8"); `dmc jit` used to answer `105`, having
/// pinned `N = 4` from `x` and read `y.t[3]` off the larger buffer. #732
/// closed the divergence by teaching the JIT's model arm the `Tensor`/`KV`
/// arms' own runtime contradiction check, so both backends refused — just at
/// two different, backend-specific runtime checks, neither one reaching the
/// checker (which, before #792, never compared a model's shape arguments at
/// all).
///
/// #792 gives the checker that comparison: an explicit bracket
/// (`fn_bracket_ty`) substitutes `f[4]`'s own `N` into both `x` and `y`'s
/// declared `Buf[N]`, so `y`'s actual `Buf[8]` now mismatches `Buf[4]` at
/// `--check` — before either backend's runtime harvest ever runs. Both
/// backends still refuse, in agreement, just earlier and with one shared
/// wording instead of two backend-specific ones (`#480` doctrine: an
/// ill-typed program is a check error, not a runtime one).
#[test]
fn two_model_args_disagreeing_on_one_shape_param_refuse_on_both_backends_732() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn f[N](x: Buf[N], y: Buf[N]) -> i64 { x.t[N - 1] + y.t[N - 1] }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   a.t[3] = 5\n\
         \x20   b.t[3] = 100\n\
         \x20   f[4](a, b)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "`dmc run` must refuse the contradiction: {run_out}");
    assert!(
        run_out.contains("arg 1: expected Buf[4], got Buf[8]"),
        "#792: this is now a `--check`-time type error, not the interpreter's \
         runtime harvest wording: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(!jit_ok, "#732/#792: `dmc jit` must not answer `105` here: {jit_out}");
    assert!(
        jit_out.contains("arg 1: expected Buf[4], got Buf[8]"),
        "#792: both backends share the checker's gate now, so they share its \
         wording too: {jit_out}",
    );
}

/// #792 rework: the test above (and `an_explicit_bracket_contradicted_by_
/// the_model_argument_refuses_732` below) both now refuse at `--check`,
/// before either backend's own MODEL-arm runtime contradiction check
/// (`jit.rs`'s `~6709`, `interp.rs`'s `~8626`) ever runs — so neither test
/// exercises that arm's own code path any more, only `--check`'s. The arm
/// is not dead: `compatible_with`'s `Named` comparison (`types.rs`) reads a
/// model argument whose checker-tracked type carries no shape args at all as
/// "no claim" (`model_arg_binds_at_runtime`, `check.rs` ~7080) — the escape a
/// bare, unparameterized `Buf` annotation takes (SPEC.md §3.10) — so the
/// contradiction is still only visible at runtime for a program that goes
/// through it. This probe takes that escape (`let !b: Buf = Buf[8] { .. }`,
/// a bare annotation on a variable actually holding a `Buf[8]`) so the
/// runtime arm stays pinned by something, rather than only by inference.
#[test]
fn a_bare_annotated_model_argument_escapes_check_and_is_caught_at_runtime_732() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn f[N](x: Buf[N], y: Buf[N]) -> i64 { x.t[N - 1] + y.t[N - 1] }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !b: Buf = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   a.t[3] = 5\n\
         \x20   b.t[3] = 100\n\
         \x20   f[4](a, b)\n\
         }\n",
    );

    // The bare `Buf` annotation on `b` erases its shape args from the
    // checker's view — `--check` has no contradiction left to compare, so it
    // passes (module a `bare-generic-model` warning, unrelated to this).
    let (check_ok, check_out) = probe.try_check();
    assert!(check_ok, "expected `--check` to let the bare-annotated escape \
        through (nothing left to compare): {check_out}");

    // Both backends still refuse the real contradiction at runtime — through
    // their own MODEL-arm check, not the checker's gate.
    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "the interpreter's own MODEL-arm check must still catch \
        this: {run_out}");
    assert!(
        run_out.contains("shape param `N` would need to be both 4 and 8"),
        "interpreter wording: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(!jit_ok, "#732: the JIT must not silently answer `105` here \
        either: {jit_out}");
}

/// The `Tensor` arm's own contradiction, asserted unchanged. It is the sentence
/// #732's fix copies, so a later edit to either must move both together.
#[test]
fn the_tensor_arms_contradiction_wording_is_unchanged() {
    let probe = Probe::new(
        "fn f[N](x: Tensor[i64, [N]], y: Tensor[i64, [N]]) -> i64 { x[0] + y[0] }\n\
         fn main() -> i64 {\n\
         \x20   let !a = forge.zeros[i64, [4]]\n\
         \x20   let !b = forge.zeros[i64, [8]]\n\
         \x20   f(a, b)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "`dmc run` must refuse: {run_out}");
    assert!(
        run_out.contains("shape param `N` would need to be both 4 and 8"),
        "interpreter wording: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(!jit_ok, "`dmc jit` must refuse: {jit_out}");
    assert!(
        jit_out.contains("shape parameter `N` would bind to 8 but already bound to 4"),
        "JIT wording: {jit_out}",
    );
}

/// The contradiction is caught against an **explicitly spelled** bracket too,
/// not only between two inferred arguments — which is what makes #732 a live
/// divergence rather than hardening for the inference this slice opens: `f[8]`
/// with a single `Buf[4]` argument was silently accepted by the old
/// `or_insert`, since the explicit seed was already there to keep.
///
/// Superseded by #792 the same way as the test above: `cap[8]`'s bracket
/// substitutes `N = 8` into `cap`'s own `Buf[N]` parameter type
/// (`fn_bracket_ty`), so `--check` now refuses `cap[8](a)` on a `Buf[4]`
/// before either backend's runtime harvest runs — one shared wording instead
/// of the interpreter's and the JIT's own two.
#[test]
fn an_explicit_bracket_contradicted_by_the_model_argument_refuses_732() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn cap[N](b: Buf[N]) -> i64 { b.t[0] + N }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   cap[8](a)\n\
         }\n",
    );
    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "the interpreter refuses this contradiction too: {run_out}");
    assert!(
        run_out.contains("arg 0: expected Buf[8], got Buf[4]"),
        "#792: this is now a `--check`-time type error: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(!jit_ok, "the bracket says 8, the instance says 4: {jit_out}");
    assert!(
        jit_out.contains("arg 0: expected Buf[8], got Buf[4]"),
        "#792: both backends share the checker's gate now: {jit_out}",
    );
}

// ─── #733: inference binds uppercase-initial names only ──────────────────────

/// **#733, closed.** `fn head[n](x: Tensor[i64,[n]])` called `head(a)`. `dmc
/// run` refuses — `bind_shape_name` never binds a lowercase-initial name, so
/// `n` reaches the "cannot be inferred" check unbound — and `dmc jit` answered
/// `5`, having inferred it. `SPEC.md §3.10`: "inference binds uppercase-initial
/// names only, for models and tensors alike."
#[test]
fn a_lowercase_shape_param_is_never_inferred_from_a_tensor_argument_733() {
    let probe = Probe::new(
        "fn head[n](x: Tensor[i64, [n]]) -> i64 { x[0] }\n\
         fn main() -> i64 {\n\
         \x20   let !a = forge.zeros[i64, [4]]\n\
         \x20   a[0] = 5\n\
         \x20   head(a)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "`dmc run` must refuse to infer `n`: {run_out}");
    assert!(
        run_out.contains("shape param `n` cannot be inferred from any argument"),
        "interpreter wording: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(!jit_ok, "#733: `dmc jit` must not answer `5` here: {jit_out}");
    assert!(
        jit_out.contains("could not infer shape parameter `n` of `head` from call args"),
        "the JIT must refuse by name: {jit_out}",
    );
}

/// **g30 — the model spelling, and the reason the gate is not dead code.**
/// `fn cap[n](b: Buf[n])` called `cap(a)` refused on both backends before this
/// slice, but only by accident: the model arm never fired at all. Opening the
/// arm without the uppercase gate would have made this answer `5` under `dmc
/// jit` — #733's defect, one construct over — so this case is what keeps the
/// gate honest once the inference works.
#[test]
fn a_lowercase_shape_param_is_never_inferred_from_a_model_argument_g30() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn cap[n](b: Buf[n]) -> i64 { b.t[0] }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   a.t[0] = 5\n\
         \x20   cap(a)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "`dmc run` must refuse to infer `n`: {run_out}");
    assert!(
        run_out.contains("shape param `n` cannot be inferred from any argument"),
        "interpreter wording: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(!jit_ok, "the uppercase gate must hold for models too: {jit_out}");
    assert!(
        jit_out.contains("could not infer shape parameter `n` of `cap` from call args"),
        "the JIT must refuse by name: {jit_out}",
    );
}

/// The uppercase gate is on **inference**, not on the name. A lowercase shape
/// parameter spelled at the call is bound from the bracket and lowers — the
/// spelling `SPEC.md §3.10` leaves available — and, matching the interpreter,
/// its agreement with the argument is never checked either: the contradicted
/// twin below answers rather than refusing.
#[test]
fn a_lowercase_shape_param_spelled_at_the_call_still_lowers() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn cap[n](b: Buf[n]) -> i64 { b.t[0] + n }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   a.t[0] = 5\n\
         \x20   cap[4](a)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 9");
    assert_eq!(probe.jit_output(), "=> 9");
}

/// The uppercase gate runs BEFORE the already-bound check, on both backends —
/// `interp.rs`'s `bind_shape_name` returns on a lowercase name before ever
/// consulting `inferred`, and `bind_or_check_shape_param` now matches that
/// order. So a lowercase name the call site spelled is not merely exempt from
/// *inference*; it is never *checked* against the argument either. `cap[8](a)`
/// on a `Buf[4]` takes the bracket's `8` as written and answers `13` on both
/// backends, in flat contradiction of the instance — the mismatch an
/// uppercase `N` in the same position refuses (the test above). Getting the
/// two rules in the wrong relative order was itself a new divergence: with the
/// gate consulted only after the bound check, this exact program answered `13`
/// under `dmc run` and raised an uncoded `jit error: n would bind to 4 but
/// already bound to 8` under `dmc jit`.
#[test]
fn a_lowercase_shape_param_spelled_at_the_call_is_never_checked_against_the_argument() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn cap[n](b: Buf[n]) -> i64 { b.t[0] + n }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   a.t[0] = 5\n\
         \x20   cap[8](a)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 13");
    assert_eq!(probe.jit_output(), "=> 13");
}

// ─── the gap slice 0 does NOT close ──────────────────────────────────────────

/// **A model ARRAY's element shape argument is still not inferable, on either
/// backend.** `fn get[N](ss: [Buf[N]; 4])` handed a `[Buf[4]; 4]` refuses under
/// `dmc run` — `harvest_shape_params`' `Type::Array` arm binds only the array's
/// LENGTH (#459), never the element type's own shape args, while the very
/// diagnostic it then raises offers "pass a tensor, model array or model whose
/// declared type uses `N`" — and under `dmc jit`, whose
/// `(Type::Array, TyKind::ModelArray)` arm checks the element's model name and
/// binds nothing.
///
/// A **different arm** from the one slice 0 fixes, and closing the JIT's half
/// alone would manufacture a divergence out of an agreement, so it stays a
/// tracked gap — filed as **#739** — and this case pins that the two backends
/// keep agreeing meanwhile. The interpreter's half is one category of its own
/// (`AGENTS.md §3`) and belongs to its own PR.
#[test]
fn a_model_arrays_element_shape_arg_is_inferable_on_neither_backend() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn get[N](ss: [Buf[N]; 4]) -> i64 { ss[0].t[0] + N }\n\
         fn main() -> i64 {\n\
         \x20   let !ss = forge.uninit[Buf[4], [4]]\n\
         \x20   ss[0] = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   ss[0].t[0] = 5\n\
         \x20   get(ss)\n\
         }\n",
    );

    let (run_ok, run_out) = probe.try_run();
    assert!(!run_ok, "the interpreter does not bind from an element type: {run_out}");
    assert!(
        run_out.contains("shape param `N` cannot be inferred from any argument"),
        "interpreter wording: {run_out}",
    );

    let (jit_ok, jit_out) = probe.try_jit();
    assert!(!jit_ok, "the JIT must not answer where `dmc run` refuses: {jit_out}");
    assert!(
        jit_out.contains("could not infer shape parameter `N` of `get` from call args"),
        "the JIT refuses by name: {jit_out}",
    );
}

/// The control for the case above: the same model array with the bracket
/// spelled at the call lowers on both backends, so the gap is inference from
/// the element type, not model arrays as parameters.
#[test]
fn the_same_model_array_lowers_when_the_bracket_is_spelled() {
    let probe = Probe::new(
        "model Buf[N] { !t: Tensor[i64, [N]] }\n\
         fn get[N](ss: [Buf[N]; 4]) -> i64 { ss[0].t[0] + N }\n\
         fn main() -> i64 {\n\
         \x20   let !ss = forge.uninit[Buf[4], [4]]\n\
         \x20   ss[0] = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   ss[0].t[0] = 5\n\
         \x20   get[4](ss)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 9");
    assert_eq!(probe.jit_output(), "=> 9");
}
