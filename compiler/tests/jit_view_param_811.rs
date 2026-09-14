//! #811 — `jit.rs`'s `unify_param_type` had no `Type::View` arm at all, so a
//! shape-generic `fn total[N](v: View[u8, [N]])` could never bind `N` under
//! `dmc jit`, regardless of the argument kind or whether the call spelled an
//! explicit bracket (`total[4](t)` hit the same catch-all as `total(t)` —
//! `unify_param_type` runs on every param either way). `#808`/`#792` already
//! made the CHECKER accept a `Tensor` or `KV` argument for a `View`-typed
//! parameter (`docs/SPEC.md` ~308: a `Tensor[T, S]` is a `View` with
//! contiguous strides; ~446: a `KV[T, S]` is convertible to a `View[T, S]`
//! for any read-only operation) and infer `N` from it — this file is the
//! JIT half.
//!
//! Fixing only `unify_param_type` turned out to be moot on its own:
//! `declare_template_signature` (reached right after unification succeeds,
//! to actually build the callee's specialized signature) converts every
//! param type through the same `ty_from_ast` that a NON-generic function's
//! `View` param already refuses at, permanently, in `jit_model_methods.rs`'s
//! `#747`/`#755` series ("is a `View` param itself lowerable on a free fn? —
//! settled: no"). So this PR also gives `declare_template_signature`'s own
//! parameter conversion — ONLY that one, via `jit.rs`'s
//! `view_param_ty_for_template` — a `View[T, S]` → `Tensor[T, S]` rewrite:
//! the JIT has no runtime representation for a View distinct from a
//! contiguous `Tensor` (every slice it ever produces, contiguous or not, is
//! copied into one — see the non-contiguous-slice test below), so a
//! shape-generic function's `View` parameter binds and lowers exactly like a
//! `Tensor` parameter of the same shape. An ordinary (non-generic)
//! function's or method's `View` parameter is UNCHANGED by this PR and still
//! refuses exactly as `#747`/`#755` pinned it — only the template path
//! (`declare_fn` never calls the rewrite) is affected.

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

    /// The JIT must REFUSE (not crash, not silently answer). Returns the
    /// combined stdout+stderr text for the caller to search.
    fn jit_refusal_text(&self) -> String {
        let out = self.dmc(&["jit"]);
        assert!(!out.status.success(), "expected dmc jit to refuse, got:\n{}", self.text(&out));
        self.text(&out)
    }

    /// The `--json` refusal's `code` field, e.g. `"jit-dynamic-shape"`.
    fn jit_json_code(&self) -> String {
        let out = self.dmc(&["jit", "--json"]);
        assert!(!out.status.success());
        let text = String::from_utf8_lossy(&out.stderr).to_string();
        let line = text.lines().find(|l| l.contains("\"kind\":\"jit-ineligible\""))
            .unwrap_or_else(|| panic!("no jit-ineligible line in:\n{text}"));
        let key = "\"code\":\"";
        let start = line.find(key).unwrap_or_else(|| panic!("no code field in: {line}")) + key.len();
        let end = line[start..].find('"').unwrap() + start;
        line[start..end].to_string()
    }
}

const TOTAL_U8: &str = "fn total[N](v: View[u8, [N]]) -> i64 {\n\
     \x20   let !s = 0\n\
     \x20   let !i = 0\n\
     \x20   while i < N {\n\
     \x20       s = s + (v[i] as i64)\n\
     \x20       i = i + 1\n\
     \x20   }\n\
     \x20   s\n\
     }\n";

/// `total(t)` — unbracketed, `N` inferred from a `Tensor[u8, [4]]` argument.
/// The issue's headline repro: before this PR, `unify_param_type` had no
/// `Type::View` arm at all and every call refused with "a template parameter
/// outside scalar / Tensor / KV", regardless of argument kind or bracket.
#[test]
fn total_infers_n_from_a_tensor_argument_and_lowers_811() {
    let probe = Probe::new(&format!(
        "{TOTAL_U8}\
         fn main() -> i64 {{\n\
         \x20   let !t = forge.zeros[u8, [4]]\n\
         \x20   t[0] = 1\n\
         \x20   t[1] = 2\n\
         \x20   t[2] = 3\n\
         \x20   t[3] = 4\n\
         \x20   total(t)\n\
         }}\n",
    ));
    assert_eq!(probe.run_output(), "=> 10");
    assert_eq!(probe.jit_output(), "=> 10", "the JIT must agree with `dmc run` (#811)");
}

/// `total[4](t)` — the SAME callee, called with an explicit bracket.
/// `instantiate_template` runs `unify_param_type` on every param regardless
/// of whether its shape var was already bound explicitly, so this hit the
/// identical pre-fix refusal as the unbracketed call above — pinned
/// separately so a future change that special-cases inference can't silently
/// leave the explicit-bracket form behind.
#[test]
fn total_explicit_bracket_call_lowers_811() {
    let probe = Probe::new(&format!(
        "{TOTAL_U8}\
         fn main() -> i64 {{\n\
         \x20   let !t = forge.zeros[u8, [4]]\n\
         \x20   t[0] = 1\n\
         \x20   t[1] = 2\n\
         \x20   t[2] = 3\n\
         \x20   t[3] = 4\n\
         \x20   total[4](t)\n\
         }}\n",
    ));
    assert_eq!(probe.run_output(), "=> 10");
    assert_eq!(probe.jit_output(), "=> 10", "the JIT must agree with `dmc run` (#811)");
}

/// `total(t[0:2])` — a CONTIGUOUS slice (a `Tensor` sliced along its only
/// axis) as the argument, `N` inferred as 2. SPEC.md §3.3: "Slicing a
/// `Tensor` yields a `View`" — at the JIT-value level this is still just a
/// `TyKind::Tensor` (a fresh pointer, the same static-shape representation
/// every Tensor value has), which is exactly why the `View` arm mirrors the
/// `Tensor` arm rather than needing a distinct runtime case.
#[test]
fn total_infers_n_from_a_contiguous_slice_811() {
    let probe = Probe::new(&format!(
        "{TOTAL_U8}\
         fn main() -> i64 {{\n\
         \x20   let !t = forge.zeros[u8, [4]]\n\
         \x20   t[0] = 1\n\
         \x20   t[1] = 2\n\
         \x20   t[2] = 3\n\
         \x20   t[3] = 4\n\
         \x20   total(t[0:2])\n\
         }}\n",
    ));
    assert_eq!(probe.run_output(), "=> 3");
    assert_eq!(probe.jit_output(), "=> 3", "the JIT must agree with `dmc run` (#811)");
}

/// `total(m[0:2, 1])` — a genuinely NON-contiguous slice (one column out of
/// a `[2, 3]` tensor) as the argument. The JIT has no runtime representation
/// for a non-contiguous view at all: `lower_tensor_slice_read_copy` always
/// materializes a fresh, contiguous `TensorTy` for a slice shaped like this
/// one (a scalar index after a kept axis) before it ever becomes a value —
/// so by the time this call's argument reaches `unify_param_type`, it is an
/// ordinary contiguous `TyKind::Tensor`, same as every other case here. This
/// is the empirical basis for the View arms' doc comment: there is no
/// "refuse a non-contiguous View" case to add, because the JIT can never
/// produce one to refuse.
#[test]
fn total_infers_n_from_a_noncontiguous_slice_811() {
    let probe = Probe::new(&format!(
        "{TOTAL_U8}\
         fn main() -> i64 {{\n\
         \x20   let !m = forge.zeros[u8, [2, 3]]\n\
         \x20   m[0, 0] = 1\n\
         \x20   m[0, 1] = 2\n\
         \x20   m[0, 2] = 3\n\
         \x20   m[1, 0] = 4\n\
         \x20   m[1, 1] = 5\n\
         \x20   m[1, 2] = 6\n\
         \x20   total(m[0:2, 1])\n\
         }}\n",
    ));
    assert_eq!(probe.run_output(), "=> 7");
    assert_eq!(probe.jit_output(), "=> 7", "the JIT must agree with `dmc run` (#811)");
}

/// `total(k)` — `N` inferred from a `KV[f32, [~]]` argument, per SPEC.md
/// §3.6 ("A `KV[T, S]` is convertible to `View[T, S]` ... for any read-only
/// operation"). The KV here is filled to EXACTLY its declared capacity.
///
/// #811 rework: an earlier version of this test pinned the JIT LOWERING this
/// call, by freezing the streaming axis at the constructor's compile-time
/// `cap`. A reviewer pass found that reasoning unsound: `cap` is only an
/// upper bound, and the JIT has no way to tell THIS KV (filled to exactly
/// `cap`) apart from one only half full at the point it is compiled — the
/// normal case for a streaming cache mid-fill, not an edge case. Freezing at
/// `cap` regardless answers a plausible but silently WRONG number the
/// moment a KV isn't full (`kv_half.dmc`-shaped probes: `dmc run` reads the
/// real runtime length, the old arm read the capacity, and the two
/// disagreed on both the length and every stale-zero element past the fill
/// point). The fix refuses this binding UNCONDITIONALLY — known `cap` or
/// not, full or not — with `Refusal::DynamicShape`, the same class
/// `lower_index`'s KV arm already uses for a `KV` of compile-time-unknown
/// capacity. This probe (filled to capacity) is kept, now pinning the
/// refusal instead of a lowering: it is the one case where a reader might
/// expect `cap` to be safe, and it still is not, because the JIT cannot
/// verify that from the type alone.
#[test]
fn total_from_a_kv_argument_now_refuses_honestly_811() {
    let probe = Probe::new(
        "fn total[N](v: View[f32, [N]]) -> f32 {\n\
         \x20   let !s = 0.0\n\
         \x20   let !i = 0\n\
         \x20   while i < N {\n\
         \x20       s = s + (v[i] as f64)\n\
         \x20       i = i + 1\n\
         \x20   }\n\
         \x20   s as f32\n\
         }\n\
         fn main() -> f32 {\n\
         \x20   let !k = forge.kv[f32, [~]](capacity = 4)\n\
         \x20   let !one = forge.zeros[f32, [1]]\n\
         \x20   one[0] = 1.0\n\
         \x20   k <- one\n\
         \x20   one[0] = 2.0\n\
         \x20   k <- one\n\
         \x20   one[0] = 3.0\n\
         \x20   k <- one\n\
         \x20   one[0] = 4.0\n\
         \x20   k <- one\n\
         \x20   total(k)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 10", "dmc run has the real runtime length");
    let text = probe.jit_refusal_text();
    assert!(text.contains("streaming-axis dimension bound to a `KV` argument"), "{text:?}");
    assert_eq!(probe.jit_json_code(), "jit-dynamic-shape");
}

/// The case the reviewer's fix is actually about: a `KV` filled to LESS than
/// its declared capacity — the normal state of a streaming cache mid-fill,
/// not an edge case. Before the #811 rework, this bound the streaming axis
/// at the constructor's `cap` (4) regardless of the 2 real appends, so
/// `dmc jit` silently answered a different number than `dmc run`: `lenof(k)`
/// read 2 under `dmc run` and 4 under the old JIT arm, and `dump(k)` printed
/// the 2 real elements padded with 2 stale zeros. Now both refuse honestly
/// the same way the filled-to-capacity case above does.
#[test]
fn total_from_a_half_filled_kv_argument_refuses_811() {
    let probe = Probe::new(
        "fn lenof[N](v: View[f32, [N]]) -> f32 {\n\
         \x20   N as f32\n\
         }\n\
         fn main() -> f32 {\n\
         \x20   let !k = forge.kv[f32, [~]](capacity = 4)\n\
         \x20   let !one = forge.zeros[f32, [1]]\n\
         \x20   one[0] = 1.0\n\
         \x20   k <- one\n\
         \x20   one[0] = 2.0\n\
         \x20   k <- one\n\
         \x20   lenof(k)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 2", "dmc run's real runtime length, not the capacity of 4");
    let text = probe.jit_refusal_text();
    assert!(text.contains("streaming-axis dimension bound to a `KV` argument"), "{text:?}");
    assert_eq!(probe.jit_json_code(), "jit-dynamic-shape");
}

/// What #811 does NOT lower: a bare `KV[f32, [~]]` PARAMETER (no
/// `forge.kv[...]` constructor visible in this scope, so its capacity is a
/// runtime-only header field, `kv.cap == None`) forwarded into `total`.
/// `dmc run` answers (a gap, not a divergence — the interpreter always has
/// the real runtime length); the JIT refuses honestly, located at `total`'s
/// own `v` parameter, as `jit-dynamic-shape` — the same class `lower_index`
/// already uses for the identical "no compile-time capacity" limitation on
/// reading `k[i]` directly.
///
/// #811 rework: before, this was the ONE case the streaming-axis arm
/// refused — `kv.cap == None` specifically — while a visible constructor
/// (`kv.cap == Some(_)`) bound the axis to that capacity and lowered. Now
/// every case refuses the same way regardless of `cap`, so this test's
/// distinguishing feature (no visible constructor) no longer changes the
/// class or wording at all — kept anyway as a second constructor shape
/// alongside `total_from_a_kv_argument_now_refuses_honestly_811`'s and
/// `total_from_a_half_filled_kv_argument_refuses_811`'s visible ones.
#[test]
fn a_bare_kv_param_of_unknown_capacity_still_refuses_honestly_811() {
    let probe = Probe::new(
        "fn total[N](v: View[f32, [N]]) -> f32 {\n\
         \x20   v[0] + N as f32\n\
         }\n\
         fn forward(k: KV[f32, [~]]) -> f32 {\n\
         \x20   total(k)\n\
         }\n\
         fn main() -> f32 {\n\
         \x20   let !k = forge.kv[f32, [~]](capacity = 4)\n\
         \x20   let !one = forge.zeros[f32, [1]]\n\
         \x20   one[0] = 10.0\n\
         \x20   k <- one\n\
         \x20   forward(k)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 11", "dmc run has the real runtime length; this is a gap, not a divergence");
    let text = probe.jit_refusal_text();
    assert!(text.contains("streaming-axis dimension bound to a `KV` argument"), "{text:?}");
    assert_eq!(probe.jit_json_code(), "jit-dynamic-shape");
}

/// An ordinary (non-generic) function's `View` parameter is UNCHANGED by
/// this PR — `declare_fn` never calls `view_param_ty_for_template`, only
/// `declare_template_signature` does. Pinned here as the negative control
/// alongside `jit_model_methods.rs`'s `#747`/`#755` series (which this PR
/// leaves passing unmodified): the moment ANY eager `View` param starts
/// lowering, that entire series stops meaning what it says.
#[test]
fn an_eager_non_generic_view_param_still_refuses_811() {
    let probe = Probe::new(
        "fn total(v: View[u8, [4]]) -> i64 {\n\
         \x20   v[0] as i64\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !t = forge.zeros[u8, [4]]\n\
         \x20   t[0] = 9\n\
         \x20   total(t)\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 9");
    let text = probe.jit_refusal_text();
    assert!(text.contains("this type in a signature"), "{text:?}");
}

// ─── #811 rework: a MUTATING (`!`) View-spelled template parameter ────────
//
// `view_param_ty_for_template`'s rewrite (`View[T,S]` → `Tensor[T,S]`,
// applied only inside `declare_template_signature`) is sound for a READ —
// the JIT has no runtime representation for a View distinct from a
// contiguous Tensor, so `v[i]` lowers identically either way — but SPEC.md
// §3.6 promises the KV→View conversion "for any read-only operation" only,
// and the rewrite does not stop at reads: once a `!v: View[T,S]` parameter
// is rewritten to `Tensor[T,S]`, `check_mut_param_ty` grants it ordinary
// Tensor `!` copy-in/copy-out semantics (SPEC.md §3.4.1) rather than a write
// straight through the descriptor the caller asked for. For a slice or
// model-field-alias argument those two disagree with `dmc run`'s own
// answer. Before #811, EVERY `View` param (mutating or not) refused via
// `ty_from_ast`'s catch-all; #811 made the read-only case lower — this
// rework keeps the mutating case refusing, honestly, at the same class.

/// `bump(t[0:2])` — the reviewer's first quoted repro. `dmc run` treats the
/// slice as an isolated place (the same answer `mut_tensor_slice_811` below
/// gets for the identical body with `Tensor` in place of `View`): the write
/// through `v` never reaches `t`. Before this fix, the View→Tensor rewrite
/// made this lower anyway and silently disagreed (`dmc jit` used to answer
/// `99` for `t[0]`, where `dmc run` answers `1`).
#[test]
fn a_mutating_view_param_from_a_slice_arg_refuses_811() {
    let probe = Probe::new(
        "fn bump[N](!v: View[f32, [N]]) -> f32 {\n\
         \x20   v[0] = 99.0\n\
         \x20   v[0]\n\
         }\n\
         fn main() -> f32 {\n\
         \x20   let !t = forge.zeros[f32, [4]]\n\
         \x20   t[0] = 1.0\n\
         \x20   t[1] = 2.0\n\
         \x20   let r = bump(t[0:2])\n\
         \x20   r + t[0]\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 100", "dmc run: r=99, t[0] untouched at 1 (99+1)");
    let text = probe.jit_refusal_text();
    assert!(text.contains("this type in a signature"), "{text:?}");
}

/// `bump(y)` with `let !y = g.heat` — the reviewer's second quoted repro, a
/// model TENSOR-field alias (SPEC.md §3.4.1: `let !x = m.field` binds a live
/// alias for a tensor field). Before this fix, the rewrite's Tensor
/// copy-in/copy-out silently disagreed here too (`dmc jit` used to answer
/// `99` for `g.heat[0]`, where `dmc run` answers `1`).
#[test]
fn a_mutating_view_param_from_a_model_field_alias_refuses_811() {
    let probe = Probe::new(
        "model G {\n\
         \x20   !heat: Tensor[f32, [4]]\n\
         }\n\
         fn bump[N](!v: View[f32, [N]]) -> f32 {\n\
         \x20   v[0] = 99.0\n\
         \x20   v[0]\n\
         }\n\
         fn main() -> f32 {\n\
         \x20   let !g = G { heat: forge.zeros[f32, [4]] }\n\
         \x20   g.heat[0] = 1.0\n\
         \x20   let !y = g.heat\n\
         \x20   let r = bump(y)\n\
         \x20   r + g.heat[0]\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 100", "dmc run: r=99, g.heat[0] untouched at 1 (99+1)");
    let text = probe.jit_refusal_text();
    assert!(text.contains("this type in a signature"), "{text:?}");
}

/// The whole-argument case too, even though it happens to be the one shape
/// where the pre-fix rewrite's answer did NOT diverge from `dmc run` (a
/// tensor `!` parameter's copy-out reaches the caller's whole place exactly
/// when the argument IS that whole place). `declare_template_signature`
/// only sees the PARAMETER's declared type, never the call site's argument
/// expression, so it has no way to special-case "this call happens to be
/// safe" — refusing categorically is the only honest option, and this test
/// pins that the fix does so even on the shape that was never wrong.
#[test]
fn a_mutating_view_param_from_a_whole_tensor_arg_refuses_811() {
    let probe = Probe::new(
        "fn bump[N](!v: View[f32, [N]]) -> f32 {\n\
         \x20   v[0] = 99.0\n\
         \x20   v[0]\n\
         }\n\
         fn main() -> f32 {\n\
         \x20   let !t = forge.zeros[f32, [4]]\n\
         \x20   t[0] = 1.0\n\
         \x20   t[1] = 2.0\n\
         \x20   let r = bump(t)\n\
         \x20   r + t[0]\n\
         }\n",
    );
    assert_eq!(probe.run_output(), "=> 198", "dmc run: r=99, t[0]=99 too (whole-place copy-out): 99+99");
    let text = probe.jit_refusal_text();
    assert!(text.contains("this type in a signature"), "{text:?}");
}

// The negative control for the three refusal tests above — a NON-mutating
// (read-only) View param on the identical slice shape must keep lowering,
// since the fix targets `!` specifically, not `View` params in general —
// already exists above as `total_infers_n_from_a_contiguous_slice_811`
// (`total(t[0:2])`, read-only, asserts `=> 3` on both backends).

// ─── Non-contiguous View-param slices, pinned element-for-element ─────────
//
// `total_infers_n_from_a_noncontiguous_slice_811` above pins only the SUM
// `dump` returns, which cannot distinguish a correct element-for-element
// read from a coincidentally-equal-sum permutation. These two pin individual
// elements of the two non-contiguous column-slice shapes the reviewer
// verified by hand against `total_infers_n_from_a_noncontiguous_slice_811`'s
// underlying `m[0:2, 1]` shape and its `m[1:4, 2]` sibling — no summation.

const AT0: &str = "fn at0[N](v: View[f32, [N]]) -> f32 { v[0] }\n";
const AT1: &str = "fn at1[N](v: View[f32, [N]]) -> f32 { v[1] }\n";
const AT2: &str = "fn at2[N](v: View[f32, [N]]) -> f32 { v[2] }\n";

const MATRIX_4X4_BUILD: &str =
    "\x20   let !m = forge.zeros[f32, [4, 4]]\n\
     \x20   let !i = 0\n\
     \x20   while i < 4 {\n\
     \x20       let !j = 0\n\
     \x20       while j < 4 {\n\
     \x20           m[i, j] = (10 * i + j) as f32\n\
     \x20           j = j + 1\n\
     \x20       }\n\
     \x20       i = i + 1\n\
     \x20   }\n";

/// `m[0:2, 1]` — column 1, rows 0..2, of the `m[i,j] = 10*i+j` matrix:
/// elements `[1, 11]`. Non-contiguous (stride 4 between elements).
#[test]
fn a_noncontiguous_column_slice_two_rows_lowers_element_for_element_811() {
    let at0 = Probe::new(&format!(
        "{AT0}fn main() -> f32 {{\n{MATRIX_4X4_BUILD}\x20   at0(m[0:2, 1])\n}}\n"
    ));
    assert_eq!(at0.run_output(), "=> 1");
    assert_eq!(at0.jit_output(), "=> 1", "v[0] of m[0:2,1]");

    let at1 = Probe::new(&format!(
        "{AT1}fn main() -> f32 {{\n{MATRIX_4X4_BUILD}\x20   at1(m[0:2, 1])\n}}\n"
    ));
    assert_eq!(at1.run_output(), "=> 11");
    assert_eq!(at1.jit_output(), "=> 11", "v[1] of m[0:2,1]");
}

// ─── #811 reviewer rework (final pass): the non-generic KV→Tensor coercion ─
//
// Every test above calls a `View[T,[N]]`-typed TEMPLATE parameter, so the
// argument always passes through `unify_param_type`'s streaming-axis check
// first. `coerce_to`'s `(TyKind::KV, TyKind::Tensor)` arm — reached instead
// from the ORDINARY call-argument coercion path a non-generic, non-`View`
// `Tensor[T,S]` parameter takes — was left in the codebase as "inert" on the
// theory that no `(from, to)` pair it could match ever reaches codegen any
// more, `unify_param_type` having already refused first. That reasoning only
// covers the template path: a plain `fn total(v: Tensor[f32,[4]])` call never
// goes through `unify_param_type` at all, so the arm was live and reachable
// the whole time.

/// `total(k)` — a NON-generic `fn total(v: Tensor[f32, [4]])`, not a `View`
/// template param, called with a capacity-4 `KV` holding exactly ONE append.
/// `--check` is clean (a `KV[f32,[~]]` argument against a `Tensor[f32,[4]]`
/// param type-checks under the same `View`-convertibility rule SPEC.md §3.6
/// grants a read). Before this fix, the removed `coerce_to` arm skipped the
/// KV's 16-byte `[len, cap]` header and handed over the data region as an
/// ordinary 4-element `Tensor[f32,[4]]` pointer — the one real append reads
/// back correctly, and the three never-written slots past the fill point
/// read as stale zero, so `dmc jit` silently answered `7` where `dmc run`'s
/// real runtime length (1, not the constructor's `capacity` of 4) faults
/// with an out-of-bounds index. A silent divergence, and a regression from
/// main, which refused this call outright. The fix deletes the arm, so the
/// pair now falls through `coerce_to`'s catch-all and refuses with the exact
/// message main used to give.
#[test]
fn a_non_generic_tensor_param_from_a_kv_argument_refuses_honestly_811() {
    let probe = Probe::new(
        "fn total(v: Tensor[f32, [4]]) -> f32 {\n\
         \x20   let !s = 0.0\n\
         \x20   let !i = 0\n\
         \x20   while i < 4 {\n\
         \x20       s = s + (v[i] as f64)\n\
         \x20       i = i + 1\n\
         \x20   }\n\
         \x20   s as f32\n\
         }\n\
         fn main() -> f32 {\n\
         \x20   let !k = forge.kv[f32, [~]](capacity = 4)\n\
         \x20   let !one = forge.zeros[f32, [1]]\n\
         \x20   one[0] = 7.0\n\
         \x20   k <- one\n\
         \x20   total(k)\n\
         }\n",
    );
    let run_out = probe.dmc(&["run"]);
    assert!(!run_out.status.success(), "expected dmc run to fault, got:\n{}", probe.text(&run_out));
    let run_text = probe.text(&run_out);
    assert!(
        run_text.contains("index") && run_text.contains("out of bounds"),
        "expected an out-of-bounds fault from the real 1-element runtime length: {run_text:?}"
    );
    let jit_text = probe.jit_refusal_text();
    assert!(
        jit_text.contains("cannot convert `KV[f32, [~]]` to `Tensor[f32, [4]]`; \
                            use `dmc run` for full semantics"),
        "expected main's refusal message verbatim, got: {jit_text:?}"
    );
}

/// `m[1:4, 2]` — column 2, rows 1..4: elements `[12, 22, 32]`. A different
/// row range and column from the pair above, so a fix that special-cases
/// one starting offset can't hide behind this file alone.
#[test]
fn a_noncontiguous_column_slice_three_rows_lowers_element_for_element_811() {
    let at0 = Probe::new(&format!(
        "{AT0}fn main() -> f32 {{\n{MATRIX_4X4_BUILD}\x20   at0(m[1:4, 2])\n}}\n"
    ));
    assert_eq!(at0.run_output(), "=> 12");
    assert_eq!(at0.jit_output(), "=> 12", "v[0] of m[1:4,2]");

    let at1 = Probe::new(&format!(
        "{AT1}fn main() -> f32 {{\n{MATRIX_4X4_BUILD}\x20   at1(m[1:4, 2])\n}}\n"
    ));
    assert_eq!(at1.run_output(), "=> 22");
    assert_eq!(at1.jit_output(), "=> 22", "v[1] of m[1:4,2]");

    let at2 = Probe::new(&format!(
        "{AT2}fn main() -> f32 {{\n{MATRIX_4X4_BUILD}\x20   at2(m[1:4, 2])\n}}\n"
    ));
    assert_eq!(at2.run_output(), "=> 32");
    assert_eq!(at2.jit_output(), "=> 32", "v[2] of m[1:4,2]");
}
