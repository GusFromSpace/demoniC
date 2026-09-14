//! Storing a whole **model value** into a model-typed field — `m.f = v` where
//! `v`'s type is itself `TyKind::Model(..)`, not a scalar. `lower_expr_stmt`'s
//! field-store arm (`jit.rs`, split by #635/#683) lowered a scalar (and
//! enum-typed) field's whole-value store but refused every `TyKind::Model`
//! target outright, citing #474 — "assignment to field `{}` — a nested-model
//! field is not shape-checked yet (#474)". #474 is closed (the ghost-method
//! bug, unrelated); the citation was stale.
//!
//! The downstream OS corpus's golden-frame path hits this once, through
//! `Compositor.set_mode!`: a non-generic `DisplayMode` model of scalar fields,
//! stored into `Compositor`'s `mode` field both directly (`c.mode = m`) and
//! through a `!self` method.
//!
//! **Layout.** A model value already IS its i64 field-arena pointer —
//! `lower_struct_lit`'s own field-init store treats a `TyKind::Model` field
//! the same as any other non-scalar (`_ => val`, the raw pointer, no memcpy),
//! and the field READ (`lower_postfix`'s `Field` arm) loads it back the same
//! way ("nested model pointers (i64)"). The store this file adds for a
//! **non-generic** nested model is the identical raw-pointer write those two
//! sites already perform — no new representation.
//!
//! **Aliasing, measured.** `interp.rs`'s `(AssignOp::Eq, Postfix{Field})` arm
//! does `*slot = rval` where `rval` is the same `Rc` clone a `let` binding
//! already shares (`SPEC.md §3.4`, the model-methods design's q11 property):
//! `self.mode = m` makes the field alias `m`'s storage, it does not copy it.
//! Mutating `m` after the store is visible through `self.mode` on `dmc run`;
//! the JIT's raw-pointer store matches, pinned below.
//!
//! **What stays refused.** A field whose nested model IS shape-generic keeps
//! refusing `jit-assign-form`: the field READ re-resolves the field's type
//! from the *receiver's* `type_args` (the model-methods design's own probe
//! q12), so an unchecked store could make a later read misinterpret the
//! stored buffer at the wrong stride. The refusal's citation is corrected to
//! #800, the issue this gap was filed under (#474's old citation named the
//! wrong bug).

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
    String::from_utf8_lossy(&out.stderr).to_string()
}

// ─── the reduced repro ───────────────────────────────────────────────────────

/// A direct field store, `c.mode = m`, with `DisplayMode` a non-generic model
/// of two scalar fields — the reduction of the corpus's `Compositor.mode`.
/// Was refused `jit-assign-form` on main; must now lower and match `dmc run`.
#[test]
fn a_direct_store_of_a_non_generic_model_value_lowers_and_matches_run() {
    let src = "model DisplayMode {\n\
               \x20   w: i64\n\
               \x20   h: i64\n\
               }\n\
               model Compositor {\n\
               \x20   !mode: DisplayMode\n\
               \x20   !mode_changed: bool\n\
               }\n\
               fn main() -> i64 {\n\
               \x20   let !c = Compositor { mode: DisplayMode { w: 1, h: 1 }, mode_changed: false }\n\
               \x20   let !m = DisplayMode { w: 800, h: 600 }\n\
               \x20   c.mode = m\n\
               \x20   c.mode.w + c.mode.h\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 1400");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must now lower this: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), stdout(&run).trim());
}

/// The same store reached through a `!self` mutating method — the exact shape
/// the downstream corpus's golden-frame path uses (`Compositor.set_mode!`,
/// which reads `self.mode` for a comparison before writing it). Exercises the
/// field READ that must already work (`self.mode.w`) alongside the new store,
/// and a second field (`mode_changed`) set unconditionally alongside it.
#[test]
fn a_store_through_a_mut_self_method_lowers_and_matches_run() {
    let src = "model DisplayMode {\n\
               \x20   w: i64\n\
               \x20   h: i64\n\
               }\n\
               fn mode_eq(a: DisplayMode, b: DisplayMode) -> bool {\n\
               \x20   a.w == b.w && a.h == b.h\n\
               }\n\
               model Compositor {\n\
               \x20   !mode: DisplayMode\n\
               \x20   !mode_changed: bool\n\
               \x20   fn set_mode!(self, m: DisplayMode) -> nil {\n\
               \x20       if mode_eq(m, self.mode) == false {\n\
               \x20           self.mode = m\n\
               \x20           self.mode_changed = true\n\
               \x20       }\n\
               \x20   }\n\
               }\n\
               fn main() -> i64 {\n\
               \x20   let !c = Compositor { mode: DisplayMode { w: 1, h: 1 }, mode_changed: false }\n\
               \x20   let !m = DisplayMode { w: 1024, h: 768 }\n\
               \x20   c.set_mode!(m)\n\
               \x20   let changed = if c.mode_changed { 1 } else { 0 }\n\
               \x20   c.mode.w + c.mode.h + changed * 1000000\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 1001792");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must now lower this: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), stdout(&run).trim());
}

/// The `set_mode!` call site where the *stored* mode already equals the
/// incoming one: the body's `if` guard must skip the store and
/// `mode_changed` must stay `false` — pins that the new lowering did not
/// widen the `if` or turn a conditional store into an unconditional one.
#[test]
fn set_mode_is_a_noop_when_the_incoming_mode_already_matches() {
    let src = "model DisplayMode {\n\
               \x20   w: i64\n\
               \x20   h: i64\n\
               }\n\
               fn mode_eq(a: DisplayMode, b: DisplayMode) -> bool {\n\
               \x20   a.w == b.w && a.h == b.h\n\
               }\n\
               model Compositor {\n\
               \x20   !mode: DisplayMode\n\
               \x20   !mode_changed: bool\n\
               \x20   fn set_mode!(self, m: DisplayMode) -> nil {\n\
               \x20       if mode_eq(m, self.mode) == false {\n\
               \x20           self.mode = m\n\
               \x20           self.mode_changed = true\n\
               \x20       }\n\
               \x20   }\n\
               }\n\
               fn main() -> bool {\n\
               \x20   let !c = Compositor { mode: DisplayMode { w: 800, h: 600 }, mode_changed: false }\n\
               \x20   let !m = DisplayMode { w: 800, h: 600 }\n\
               \x20   c.set_mode!(m)\n\
               \x20   c.mode_changed\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> false");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must now lower this: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), "=> false");
}

// ─── the aliasing pin ────────────────────────────────────────────────────────

/// After `c.mode = m`, mutating `m` must be visible through `c.mode` — a
/// model-field store ALIASES, it does not copy (`SPEC.md §3.4`'s "Bindings
/// from model instances" rule extended to a field target, measured against
/// `interp.rs`'s `*slot = rval` where `rval` is an `Rc` clone). The JIT's
/// store is a raw pointer write for exactly this reason; a lowering that
/// instead snapshotted the fields would answer `800`, not `999`, and would be
/// a silent divergence from `dmc run`.
#[test]
fn a_field_store_aliases_the_stored_instance_matches_run() {
    let src = "model DisplayMode {\n\
               \x20   w: i64\n\
               \x20   h: i64\n\
               }\n\
               model Compositor {\n\
               \x20   !mode: DisplayMode\n\
               \x20   !mode_changed: bool\n\
               }\n\
               fn main() -> i64 {\n\
               \x20   let !c = Compositor { mode: DisplayMode { w: 1, h: 1 }, mode_changed: false }\n\
               \x20   let !m = DisplayMode { w: 800, h: 600 }\n\
               \x20   c.mode = m\n\
               \x20   m.w = 999\n\
               \x20   c.mode.w\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 999", "the interpreter aliases the field, it does not copy it");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must now lower this: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), "=> 999", "the JIT's raw-pointer store must alias exactly like `dmc run`");
}

// ─── a nested model with a tensor field ─────────────────────────────────────

/// The same aliasing property holds when the stored model has a TENSOR field:
/// storing the whole `Palette` never copies its `colors` tensor either, on
/// either backend, because a model-field store is one pointer write
/// regardless of what the model's own fields are made of — `copy_tensor_value`
/// (the function `lower_struct_lit` uses to snapshot a place-expression into
/// a *tensor* field) is identity for a `TyKind::Model` value, so a nested
/// model is never partially copied one field at a time.
#[test]
fn a_nested_model_with_a_tensor_field_aliases_the_whole_instance() {
    let src = "model Palette {\n\
               \x20   w: i64\n\
               \x20   colors: Tensor[f32, [3]]\n\
               }\n\
               model Compositor2 {\n\
               \x20   !pal: Palette\n\
               }\n\
               fn main() -> f32 {\n\
               \x20   let !c = Compositor2 { pal: Palette { w: 1, colors: forge.zeros[f32, [3]] } }\n\
               \x20   let !p = Palette { w: 800, colors: forge.zeros[f32, [3]] }\n\
               \x20   p.colors[0] = 5.0\n\
               \x20   c.pal = p\n\
               \x20   p.colors[0] = 9.0\n\
               \x20   c.pal.colors[0]\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", stderr(&run));
    assert_eq!(stdout(&run).trim(), "=> 9", "the stored model's tensor field aliases too, not just its scalars");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit must now lower this: {}", stderr(&jit));
    assert_eq!(stdout(&jit).trim(), "=> 9");
}

// ─── the shape-generic case: still refused, with the corrected citation ─────

/// The model-methods design's probe q12, re-measured: storing an `Inner[9,9]`
/// into an `Outer[2,2]`'s `surf` field. `--check` passes and `dmc run`
/// accepts it (the interpreter has no shape check on a field store at all).
/// The JIT must keep refusing — `jit-assign-form`, naming the field and the
/// nested model, citing #800 rather than the closed #474 the message named
/// before this change.
#[test]
fn a_shape_generic_nested_model_field_store_still_refuses_named_and_classed() {
    let src = "model Inner[H, W] {\n\
               \x20   t: Tensor[f32, [H, W]]\n\
               }\n\
               model Outer[H, W] {\n\
               \x20   surf: Inner[H, W]\n\
               }\n\
               fn main() -> nil {\n\
               \x20   let !o = Outer[2, 2] { surf: Inner[2, 2] { t: forge.zeros[f32, [2, 2]] } }\n\
               \x20   o.surf = Inner[9, 9] { t: forge.zeros[f32, [9, 9]] }\n\
               \x20   print(o.surf.t[8, 8])\n\
               }\n";
    let probe = Probe::new(src);

    let checked = probe.run(&["--check"]);
    assert!(checked.status.success(), "--check must still pass: {}", stderr(&checked));

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run accepts an unchecked reshape on a field store: {}", stderr(&run));

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1), "the JIT must keep refusing the shape-generic nested case");
    let err_line = stderr(&out);
    assert!(err_line.contains("\"kind\":\"jit-ineligible\""), "must be classed, not a bare jit-error: {}", err_line);
    assert!(err_line.contains("\"code\":\"jit-assign-form\""), "got: {}", err_line);
    assert!(err_line.contains("`surf`"), "must name the field: {}", err_line);
    assert!(err_line.contains("Inner"), "must name the shape-generic model: {}", err_line);
    assert!(err_line.contains("#800"), "must cite the corrected issue, not the stale #474: {}", err_line);
    assert!(!err_line.contains("#474"), "the old citation named the wrong (closed, unrelated) bug: {}", err_line);
}

/// A plain non-generic field store must not, in the process of fixing the
/// case above, start touching the shape-generic refusal's message for an
/// *unrelated* field on the same model — a regression guard on the guard
/// itself: `Outer`'s `surf` field still refuses even when nothing about
/// `DisplayMode`-shaped code is anywhere in the same file.
#[test]
fn the_shape_generic_refusal_is_unconditional_on_the_nested_models_own_params() {
    let src = "model Leaf[N] {\n\
               \x20   v: Tensor[i64, [N]]\n\
               }\n\
               model Holder[N] {\n\
               \x20   leaf: Leaf[N]\n\
               }\n\
               fn main() -> nil {\n\
               \x20   let !h = Holder[4] { leaf: Leaf[4] { v: forge.zeros[i64, [4]] } }\n\
               \x20   h.leaf = Leaf[4] { v: forge.zeros[i64, [4]] }\n\
               \x20   print(h.leaf.v[0])\n\
               }\n";
    let probe = Probe::new(src);

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let err_line = stderr(&out);
    assert!(err_line.contains("\"code\":\"jit-assign-form\""), "got: {}", err_line);
    assert!(err_line.contains("#800"), "got: {}", err_line);
}
