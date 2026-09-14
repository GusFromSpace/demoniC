//! #674 slice 4 — the acceptance program's cross-backend half, per the
//! arena-budget design's §4 and §5 ("Slice 4").
//!
//! `examples/tenant_budgets.dmc` carries #674's five invariants, the
//! reap-order criterion and §4's (a)–(c), and `make conformance` runs it on
//! both backends — but unflagged, so what it asserts is that each section
//! FITS, not what it costs. `tools/jit_probes.py` carries the other half, one
//! byte under each section's bisected `--forge` minimum, so both backends
//! refuse. Neither of those is the differential §5 asks for: that `dmc run`
//! and `dmc jit` **bisect to the same ceiling**, which needs both sides of
//! the bracket on both backends in one place. That is this file.
//!
//! Each case below brackets one section: at the recorded minimum the backend
//! runs the program to completion, and at one byte less it refuses. Two
//! bracketed minimums that match ARE a shared ceiling — a bisection asserted
//! in four runs instead of forty-eight.
//!
//! Six of the nine sections bisect to the same figure on both backends. The
//! other three do not, and the reason is the same in all three: `dmc jit`
//! materializes a `Tenant` model, and a propagated `(i64, Err)` tuple, in the
//! Forge arena, where `dmc run` builds both outside any arena — so the JIT is
//! charged one 64-byte aligned slot per such value and the interpreter is
//! charged nothing. Those three assert the exact pair rather than equality,
//! and they assert the INEQUALITY too: a JIT that stopped charging for them
//! would turn the case red, which is the point — the figure would have moved
//! and this file, `tools/jit_probes.py`'s flags and the example's own comment
//! would all have to move with it.
//!
//! The sources are transcriptions of the example's sections as `fn main`
//! programs (a test may not read `../examples`). They are byte-identical to
//! the nine `tenant_budgets_*_674` probe sources in `tools/jit_probes.py`.

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

fn run_at(src: &str, mode: &str, budget: u64) -> Output {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("section.dmc");
    std::fs::write(&path, src).expect("write section");
    Command::new(dmc_binary())
        .arg(mode)
        .arg(&path)
        .arg(format!("--forge={budget}"))
        .output()
        .expect("spawn dmc")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// Assert that `mode`'s bisected minimum for `src` is exactly `min`: the
/// program completes at `min` and refuses at `min - 1` with `tag`.
fn assert_ceiling(src: &str, mode: &str, min: u64, tag: &str, section: &str) {
    let at = run_at(src, mode, min);
    assert!(
        at.status.success(),
        "{section}: `dmc {mode}` did not complete at its recorded minimum \
         --forge={min}:\n{}",
        stderr(&at),
    );
    let under = run_at(src, mode, min - 1);
    assert!(
        !under.status.success(),
        "{section}: `dmc {mode}` completed one byte under its recorded \
         minimum (--forge={}), so {min} is not a ceiling",
        min - 1,
    );
    let err = stderr(&under);
    assert!(
        err.contains(tag),
        "{section}: `dmc {mode}` refused at --forge={} for the wrong reason \
         — expected `{tag}`:\n{err}",
        min - 1,
    );
}

/// The six sections whose two backends bisect to the SAME ceiling.
fn assert_same_ceiling(src: &str, min: u64, tag: &str, section: &str) {
    assert_ceiling(src, "run", min, tag, section);
    assert_ceiling(src, "jit", min, tag, section);
}

/// The three whose ceilings differ by whole 64-byte slots the JIT spends in
/// Forge on values the interpreter never puts there. Both figures are
/// bracketed, and the gap itself is pinned.
fn assert_split_ceiling(
    src: &str,
    min_run: u64,
    tag_run: &str,
    min_jit: u64,
    tag_jit: &str,
    section: &str,
) {
    assert_ceiling(src, "run", min_run, tag_run, section);
    assert_ceiling(src, "jit", min_jit, tag_jit, section);
    // Documents, does not assert: both operands are literals passed by the
    // caller, so this can only ever fail at the call site. What turns red
    // when the JIT stops materializing the model/tuple in Forge is the
    // `assert_ceiling` for "jit" above (its minimum drops to the
    // interpreter's). Re-bisect and update this file, tools/jit_probes.py
    // and the example's own comment together when that happens.
    debug_assert!(
        min_jit > min_run && (min_jit - min_run) % 64 == 0,
        "{section}: the split rows are a whole number of 64-byte Forge \
         slots by construction ({min_run} vs {min_jit})",
    );
}

const SPEND_ALL: &str = "fn spend_all(b: Budget) -> f32 {
    let !t = forge.zeros[f32, [16384]] in b
    t[0] = 1.0
    sum(t)
}
";

const SPEND_HALF: &str = "fn spend_half(b: Budget) -> f32 {
    let !t = forge.zeros[f32, [8192]] in b
    t[0] = 1.0
    sum(t)
}
";

const SPEND_QUARTER: &str = "fn spend_quarter(b: Budget) -> f32 {
    let !t = forge.zeros[f32, [4096]] in b
    t[0] = 1.0
    sum(t)
}
";

const TICK: &str = "fn tick() -> f32 {
    let !t = forge.zeros[f32, [4096]]
    t[0] = 1.0
    sum(t)
}
";

const TENANT: &str = "model Tenant { budget: Budget, id: i64 }
fn carve_tenant(parent: Budget, id: i64, bytes: i64) -> Tenant {
    Tenant { budget: forge.carve(parent, bytes), id: id }
}
fn drop_tenant(r: Tenant) -> nil {
    forge.reap(r.budget)
    nil
}
";

const EDGES: &str = "fn leave_normally(b: Budget) -> nil {
    forge.enter(b) {
        let u = forge.zeros[f32, [16]]
    }
    nil
}
fn leave_by_return(b: Budget) -> i64 {
    forge.enter(b) {
        let u = forge.zeros[f32, [16]]
        return 7
    }
    3
}
fn leave_by_break(b: Budget) -> i64 {
    let !n = 0
    for i in 0..3 {
        n = n + 1
        forge.enter(b) {
            let u = forge.zeros[f32, [16]]
            break
        }
        n = n + 10
    }
    n
}
fn leave_by_continue(b: Budget) -> i64 {
    let !n = 0
    for i in 0..3 {
        n = n + 1
        forge.enter(b) {
            let u = forge.zeros[f32, [16]]
            continue
        }
        n = n + 10
    }
    n
}
fn fail_with(msg: Err) -> (i64, Err) {
    (0, msg)
}
fn leave_by_query(b: Budget, msg: Err) -> (i64, Err) {
    forge.enter(b) {
        let u = forge.zeros[f32, [16]]
        let y = fail_with(msg)?
        let v = forge.zeros[f32, [16]]
    }
    (5, nil)
}
";

const AMBIENT: &str = "fn hop(n: i64) -> f32 {
    deeper(n)
}
fn deeper(n: i64) -> f32 {
    let !t = forge.zeros[f32, [8192]]
    t[0] = 1.0
    sum(t)
}
";

// (INV) — a parent's capacity is its children's whole extents, spent and
// unspent alike. The root has to hold the 262 144 B parent whatever the four
// tenants under it go on to allocate.
#[test]
fn inv_bisects_to_the_same_ceiling_on_both_backends() {
    let src = format!(
        "{SPEND_ALL}fn main() -> i64 {{
    let p = forge.carve(forge.root, 262144)
    let b1 = forge.carve(p, 65536)
    let b2 = forge.carve(p, 65536)
    let b3 = forge.carve(p, 65536)
    let b4 = forge.carve(p, 65536)
    let s = spend_all(b1) + spend_all(b2) + spend_all(b3)
    forge.reap(b1)
    forge.reap(b2)
    forge.reap(b3)
    forge.reap(b4)
    forge.reap(p)
    s as i64
}}"
    );
    assert_same_ceiling(&src, 262144, "budget-carve", "(INV)");
}

// (LOC) — a reap credits the whole extent, not the unspent remainder.
#[test]
fn loc_bisects_to_the_same_ceiling_on_both_backends() {
    let src = format!(
        "{SPEND_ALL}{SPEND_QUARTER}fn main() -> i64 {{
    let p = forge.carve(forge.root, 65536)
    let c = forge.carve(p, 65536)
    let s = spend_quarter(c)
    forge.reap(c)
    let d = forge.carve(p, 65536)
    let u = spend_all(d)
    forge.reap(d)
    forge.reap(p)
    (s + u) as i64
}}"
    );
    assert_same_ceiling(&src, 65536, "budget-carve", "(LOC)");
}

// (PRE) — a carve is bounded by the parent's free capacity, at the carve.
#[test]
fn pre_bisects_to_the_same_ceiling_on_both_backends() {
    let src = format!(
        "{SPEND_HALF}fn main() -> i64 {{
    let p = forge.carve(forge.root, 65536)
    let c1 = forge.carve(p, 32768)
    let c2 = forge.carve(p, 32768)
    let s = spend_half(c1) + spend_half(c2)
    forge.reap(c1)
    forge.reap(c2)
    forge.reap(p)
    s as i64
}}"
    );
    assert_same_ceiling(&src, 65536, "budget-carve", "(PRE)");
}

// (PRE-A) — two tenants meter separately, each at its own ceiling. The two
// `Tenant` models are the whole of the 128 B the JIT wants beyond the
// interpreter's figure.
#[test]
fn prea_bisects_to_a_ceiling_that_differs_by_two_tenant_models() {
    let src = format!(
        "{TENANT}{SPEND_ALL}{TICK}fn step_tenant(r: Tenant) -> f32 {{
    let !s: f32 = 0.0
    forge.enter(r.budget) {{
        s = tick() + tick() + tick() + tick()
    }}
    s
}}
fn main() -> i64 {{
    let a = carve_tenant(forge.root, 1, 65536)
    let b = carve_tenant(forge.root, 2, 65536)
    let sa = step_tenant(a)
    let sb = spend_all(b.budget)
    drop_tenant(a)
    drop_tenant(b)
    (sa + sb) as i64
}}"
    );
    assert_split_ceiling(
        &src,
        131072,
        "budget-carve",
        131200,
        "budget-carve",
        "(PRE-A)",
    );
}

// (NN) — free and used never go negative: 64 reap-then-recarve cycles leave
// the parent's capacity exactly where it started.
#[test]
fn nn_bisects_to_the_same_ceiling_on_both_backends() {
    let src = format!(
        "{SPEND_ALL}fn main() -> i64 {{
    let p = forge.carve(forge.root, 131072)
    for i in 0..64 {{
        let c = forge.carve(p, 131072)
        forge.reap(c)
    }}
    let d = forge.carve(p, 131072)
    let s = spend_all(d)
    forge.reap(d)
    forge.reap(p)
    s as i64
}}"
    );
    assert_same_ceiling(&src, 131072, "budget-carve", "(NN)");
}

// Criterion 4's complement — the reap order a host must follow: children
// first, and the parent's whole extent then comes back.
#[test]
fn the_reap_order_bisects_to_the_same_ceiling_on_both_backends() {
    let src = format!(
        "{SPEND_ALL}fn main() -> i64 {{
    let host = forge.carve(forge.root, 131072)
    let p = forge.carve(host, 131072)
    let c1 = forge.carve(p, 65536)
    let c2 = forge.carve(p, 65536)
    let s = spend_all(c1) + spend_all(c2)
    forge.reap(c1)
    forge.reap(c2)
    forge.reap(p)
    let again = forge.carve(host, 131072)
    forge.reap(again)
    forge.reap(host)
    s as i64
}}"
    );
    assert_same_ceiling(&src, 131072, "budget-carve", "criterion 4");
}

// §4 (a) — `forge.enter` pops on every exit edge. The propagated
// `(i64, Err)` tuple of the `?` edge is the 64 B the JIT wants beyond the
// interpreter's figure.
#[test]
fn the_exit_edges_bisect_to_a_ceiling_that_differs_by_one_tuple() {
    let src = format!(
        "{TICK}{EDGES}fn main() -> i64 {{
    let msg = \"the query edge\"
    let b = forge.carve(forge.root, 512)
    let o = forge.carve(forge.root, 65536)
    let q = forge.carve(forge.root, 16448)
    let !acc: f32 = 0.0
    let !k = 0
    let !nb = 0
    let !nc = 0
    let !qv = 0
    forge.enter(o) {{
        leave_normally(b)
        acc = acc + tick()
        k = leave_by_return(b)
        acc = acc + tick()
        nb = leave_by_break(b)
        acc = acc + tick()
        nc = leave_by_continue(b)
        acc = acc + tick()
    }}
    forge.enter(q) {{
        let (v, e) = leave_by_query(b, msg)
        qv = v
        acc = acc + tick()
    }}
    forge.reap(q)
    forge.reap(o)
    forge.reap(b)
    (acc as i64) + k + nb + nc + qv
}}"
    );
    assert_split_ceiling(&src, 82496, "budget-carve", 82560, "budget-carve", "(a)");
}

// §4 (b) — a budget carved in one fn, stored in a model field and reaped in
// another names the same budget. The single `Tenant` model is the 64 B the JIT
// wants beyond the interpreter's figure.
#[test]
fn the_model_field_reap_bisects_to_a_ceiling_that_differs_by_one_tenant() {
    let src = format!(
        "{TENANT}{SPEND_ALL}fn main() -> i64 {{
    let p = forge.carve(forge.root, 65536)
    let r = carve_tenant(p, 3, 65536)
    let s = spend_all(r.budget)
    drop_tenant(r)
    let again = forge.carve(p, 65536)
    forge.reap(again)
    forge.reap(p)
    (s as i64) + r.id
}}"
    );
    // The JIT's own below-minimum refusal here is the `Tenant` model's own
    // 64 B allocation failing against the root, not a carve — the carve of
    // `p` still fits at 65 599.
    assert_split_ceiling(
        &src,
        65536,
        "budget-carve",
        65600,
        "forge arena exhausted",
        "(b)",
    );
}

// §4 (c) — the entered-budget register is ambient across an ordinary demoniC
// call. Not criterion 5, which needs a host boundary neither backend has.
#[test]
fn the_ambient_call_bisects_to_the_same_ceiling_on_both_backends() {
    let src = format!(
        "{AMBIENT}fn main() -> i64 {{
    let b = forge.carve(forge.root, 65536)
    let !s: f32 = 0.0
    forge.enter(b) {{
        s = s + hop(1)
        s = s + hop(2)
    }}
    forge.reap(b)
    s as i64
}}"
    );
    assert_same_ceiling(&src, 65536, "budget-carve", "(c)");
}
