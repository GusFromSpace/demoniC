//! #612 — `vault { }` / `forge { }` / `stream { }`, and a valueless block, in
//! the JIT (slice 1 of the arena-blocks design).
//!
//! The construct the JIT refused was doing nothing at run time on either
//! backend. Arena selection is a **check-time** property: `check.rs`'s
//! `arena_stack` is the only consumer of the block's kind, and it gates the
//! cross-arena write error (`MEMORY.md §3.1`, #442) for `dmc run` and `dmc jit`
//! alike. Nothing rewinds at the exit — #400 records the interpreter trying and
//! reverting exactly that, because it made `forge { … }` a one-token bypass of
//! `--forge`. So the block is a lexical scope, and this file pins the
//! consequences that a future rewinding lowering would break:
//!
//!   * a tensor allocated inside a block is readable after it, **and writable**
//!     — a rewind would put that store into reclaimed bytes;
//!   * the block's tail is its value, in expression position;
//!   * a block in *statement* position runs for effect and needs no tail
//!     (the #612 defect proper, which bit plain blocks before any arena work);
//!   * nesting resolves innermost-first, as `MEMORY.md §3.1` specifies.
//!
//! And the two boundaries slice 1 deliberately does not cross: a top-level
//! `Item::Arena` still refuses with `jit-item` (#611 is an open spec question),
//! and a block on a `@grad` gradient path stays refused rather than
//! miscompiled (#615).
//!
//! These drive the actual `dmc` binary under BOTH backends, because a backend
//! divergence is invisible to an in-process test, and each case asserts the
//! literal expected output rather than "the two agree" — two backends can agree
//! on the wrong number.

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
        Command::new(dmc_binary()).args(args).arg(&self.path).output().expect("spawn dmc")
    }
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// Run `src` under `dmc run` and `dmc jit`; both must exit 0 and print exactly
/// `want`.
fn assert_both_print(src: &str, want: &str) {
    let probe = Probe::new(src);
    for mode in ["run", "jit"] {
        let out = probe.run(&[mode]);
        assert!(
            out.status.success(),
            "[{mode}] exited {:?}\nstderr: {}", out.status.code(), stderr(&out),
        );
        assert_eq!(stdout(&out), want, "[{mode}] stdout (stderr: {})", stderr(&out));
    }
}

// ─── The four #639 programs ─────────────────────────────────────────────────
//
// `return` / `break` / `continue` crossing a block boundary. This was the one
// shape in this file where the two backends printed different numbers: the JIT
// propagated the control flow and the interpreter swallowed it (`interp.rs`'s
// `Expr::Block` and `Expr::ArenaBlock` mapped `Flow::Return(v)` to the block's
// *value* and `Flow::Break` / `Flow::Continue` to `Value::Nil`).
//
// #648 fixed the interpreter and closed #639, so every one of the four now
// answers the JIT's number on both backends. They were pinned here in two
// separate `#[test]`s — one asserting 42 / 3 / 13 / 9 under `jit`, one
// asserting 7 / 10 / 15 / 4 under `run` with every line commented "flips when
// #639 lands" — precisely so that this merge would be a one-line-per-program
// edit and not a rediscovery. The two are now one `assert_both_print` test,
// which is the stronger assertion: a single-backend expectation cannot catch a
// divergence, and there is no longer a reason to keep one.

/// `return` out of an arena block: the function returns 42, the trailing `7` is
/// never reached.
const CF_RETURN: &str = r#"fn f() -> i64 {
    forge { return 42 }
    7
}

fn main() -> nil {
    print(f())
    nil
}
"#;

/// `break` out of an arena block inside a `for`: 3 (0+1+2, stops at i==3). It
/// was 10 under `dmc run` (0+1+2+3+4 — the loop never broke) before #648.
const CF_BREAK: &str = r#"fn main() -> nil {
    let !s = 0
    for i in 0..5 {
        forge { if i == 3 { break } }
        s = s + i
    }
    print(s)
    nil
}
"#;

/// `continue` out of an arena block inside a `for`: 13 (0..5 without 2). It was
/// 15 under `dmc run` (0..5 with it — the `continue` was discarded) before #648.
const CF_CONTINUE: &str = r#"fn main() -> nil {
    let !s = 0
    for i in 0..6 {
        forge { if i == 2 { continue } }
        s = s + i
    }
    print(s)
    nil
}
"#;

/// The same shape with no arena keyword at all: a plain block. This is what
/// made #639 an interpreter bug rather than anything slice 1 introduced — it
/// reproduced on `main`, through `Expr::Block`, on syntax that predates the
/// arena work, and #648 fixed both arms together. 9; it was 4 under `dmc run`.
const CF_PLAIN_RETURN: &str = r#"fn g() -> i64 {
    { return 9 }
    4
}

fn main() -> nil {
    print(g())
    nil
}
"#;

/// #639, closed by #648. Control flow crossing a block boundary does what the
/// language says on **both** backends now: `return` returns from the function,
/// `break` and `continue` reach the enclosing loop.
///
/// The JIT column never moved — 42 / 3 / 13 / 9 is what it answered the day
/// slice 1 landed, and it is what the interpreter converged on. The four
/// programs are also probes in `tools/jit_probes.py` now (`block_return`,
/// `block_break`, `block_continue`, `plain_block_return`); they were kept out
/// of the battery only because a run/jit disagreement scores there as a
/// divergence and would have reddened every branch until #639 landed.
#[test]
fn control_flow_crosses_a_block_boundary_on_both_backends() {
    assert_both_print(CF_RETURN, "42\n");
    assert_both_print(CF_BREAK, "3\n");
    assert_both_print(CF_CONTINUE, "13\n");
    assert_both_print(CF_PLAIN_RETURN, "9\n");
}

// ─── The block is a scope, and its allocations outlive it ───────────────────

/// #630, closed by #642, in its **arena-block** spelling. The issue's repro is
/// a plain block and `compiler/tests/block_scope_630.rs` owns that family; this
/// is the one line of it slice 1 is responsible for, because before slice 1 the
/// arena spelling could not reach the defect at all — the JIT refused it with
/// `jit-construct` first.
///
/// `let a = 1  let b = forge { let a = 2  a + 10 }  a + b` is 13: the inner `a`
/// is 2 inside the block (so `b` is 12) and the outer `a` is 1 again after it.
/// A JIT that kept `locals` as one flat map for the whole function answered 14,
/// exiting 0 — which is why this is asserted on both backends rather than
/// merely "the JIT doesn't crash". The arena block gets its scope because all
/// three of its lowering routes go through `lower_block_value`, the wrapper,
/// and not through `lower_block_body`.
#[test]
fn a_shadowing_let_in_an_arena_block_dies_with_the_block() {
    assert_both_print(
        r#"fn probe() -> i64 {
    let a = 1
    let b = forge { let a = 2
        a + 10 }
    a + b
}

fn main() -> nil {
    print(probe())
    nil
}
"#,
        "13\n",
    );
}

/// The sharpest case (design probe e1). A rewinding lowering passes every read
/// test by luck — the bytes are still there until something else allocates —
/// and fails this one, because the store lands in reclaimed memory.
#[test]
fn a_tensor_allocated_in_a_block_is_writable_after_it() {
    assert_both_print(
        r#"fn main() -> nil {
    let !t = forge { forge.zeros[f32, [4]] }
    t[0] = 7.0
    print(t[0] as i64)
    nil
}
"#,
        "7\n",
    );
}

/// The read half (design probes p5 and r1), for both arena names. `vault.*` and
/// `forge.*` lower into the same one Forge region today; that is a slice-2
/// limitation, not something this test may assume away.
#[test]
fn a_tensor_allocated_in_a_block_is_readable_after_it() {
    assert_both_print(
        r#"fn main() -> nil {
    let v = vault { vault.ones[f32, [4]] }
    let f = forge { forge.ones[f32, [4]] }
    print(sum(v) as i64 + sum(f) as i64)
    nil
}
"#,
        "8\n",
    );
}

/// Expression position: the block's tail is its value, for all three arena
/// names (design probes p1, p8) and through nesting (p4).
#[test]
fn an_arena_block_in_expression_position_is_its_tail_value() {
    assert_both_print(
        r#"fn main() -> nil {
    let a = vault { let x = 2 + 3
        x }
    let b = stream { let y = 9
        y }
    let c = forge { vault { 2 + 2 } }
    print(a * 100 + b * 10 + c)
    nil
}
"#,
        "594\n",
    );
}

// ─── Statement position — the #612 defect ───────────────────────────────────

/// The issue's own repro: a **plain** block with no tail value, as a statement.
/// It failed with `jit error: block expression has no tail value` before slice
/// 1, i.e. a defect rather than a refusal — the JIT claiming a bug where the
/// program was fine.
#[test]
fn a_valueless_plain_block_in_statement_position_runs_for_effect() {
    assert_both_print(
        r#"fn main() -> nil {
    let !s = 0
    { s = 7 }
    print(s)
    nil
}
"#,
        "7\n",
    );
}

/// The same shape with an arena keyword in front (design probe p2), which is
/// how the corpus actually writes it — `examples/matmul.dmc` and
/// `examples/train_step.dmc` are whole `main` bodies wrapped this way.
#[test]
fn an_arena_block_in_statement_position_runs_for_effect() {
    assert_both_print(
        r#"fn main() -> nil {
    let !s = 0
    forge { s = 7 }
    vault { s = s + 1 }
    print(s)
    nil
}
"#,
        "8\n",
    );
}

/// A block in statement position that *does* have a tail value discards it,
/// rather than becoming the enclosing block's value.
#[test]
fn a_statement_block_with_a_tail_discards_it() {
    assert_both_print(
        r#"fn main() -> nil {
    let !s = 1
    forge { s = 2
        99 }
    print(s)
    nil
}
"#,
        "2\n",
    );
}

/// The other trailing statement `lower_block_value` has to yield: a
/// semicolon-terminated tail. `5;` is a `Stmt::Expr` rather than a `tail_expr`,
/// exactly as an arena block is, and the interpreter's `deferred_last` treats
/// the two identically — it defers *every* trailing assignment-free
/// `Stmt::Expr`. The JIT's arm was `ArenaBlock`-only, so this shape answered 5
/// under `dmc run` and `jit error: block expression has no tail value` under
/// `dmc jit`: a pre-existing defect, unrelated to arena syntax, that the
/// widened arm closes as a side effect of making the comment above it true.
#[test]
fn a_semicolon_terminated_tail_is_still_the_blocks_value() {
    assert_both_print(
        r#"fn main() -> nil {
    let x = { let z = 1
        5; }
    print(x)
    nil
}
"#,
        "5\n",
    );
}

// ─── The cross-arena write rule is the checker's, and both backends get it ──

/// Design probes p6 and e2. `vault { }` is what makes a write to Vault-resident
/// data legal (#442); nesting resolves innermost-first, so `forge { vault { … } }`
/// is legal for the same reason `vault { forge { … } }` is not.
#[test]
fn a_vault_block_legalises_a_write_to_vault_data() {
    assert_both_print(
        r#"fn main() -> nil {
    let !w = vault.zeros[f32, [4]]
    vault { w[0] = 5.0 }
    forge { vault { w[1] = 3.0 } }
    print(w[0] as i64 * 10 + w[1] as i64)
    nil
}
"#,
        "53\n",
    );
}

/// The mirror image (design probe e2, second half): the innermost block is
/// Forge, so the write is the #442 error — at CHECK time, before either backend
/// runs, which is why slice 1 leaves the JIT nothing to re-check.
#[test]
fn the_innermost_block_decides_and_forge_inside_vault_is_still_an_error() {
    let probe = Probe::new(
        r#"fn main() -> nil {
    let !w = vault.zeros[f32, [4]]
    vault { forge { w[0] = 3.0 } }
    nil
}
"#,
    );
    for mode in ["run", "jit"] {
        let out = probe.run(&[mode]);
        assert!(!out.status.success(), "[{mode}] accepted a cross-arena write");
        assert!(
            stderr(&out).contains("cross-arena write"),
            "[{mode}] wrong diagnostic: {}", stderr(&out),
        );
    }
}

// ─── The two boundaries slice 1 does not cross ──────────────────────────────

/// #611 is open: a top-level `vault { … }` item type-checks and is never
/// executed by the interpreter (its bindings do not even become module-level).
/// Until that is answered the JIT must keep refusing it — and refusing it as a
/// *gap*, `jit-item`, not as a defect.
#[test]
fn a_top_level_arena_item_is_still_refused_with_jit_item() {
    let probe = Probe::new(
        r#"vault {
    let g = 11
}

fn main() -> nil {
    print(1)
    nil
}
"#,
    );
    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1), "the JIT accepted a top-level arena item");
    let first = stderr(&out).lines().next().unwrap_or_default().to_string();
    assert!(
        first.contains("\"kind\":\"jit-ineligible\"") && first.contains("\"code\":\"jit-item\""),
        "expected a jit-item refusal, got:\n{}", first,
    );
}

/// #615: a block on a `@grad` gradient path is a **runtime** error under
/// `dmc run` ("must produce a scalar derived from a differentiable input").
/// Slice 1 makes arena blocks lower everywhere else, so the load-bearing
/// assertion is that it did not make this shape *compile*: the JIT's existing
/// `@grad` refusal must still catch it, and a refusal is the one honest answer
/// while `dmc run` throws. Both backends reject; neither returns a number.
///
/// The shapes are concrete on purpose. A shape-generic `@grad fn` is refused
/// earlier and for an unrelated reason ("monomorphizing the backward is a later
/// slice"), so a `[D]` spelling here would pass without ever reaching the block.
#[test]
fn an_arena_block_on_a_grad_path_is_rejected_not_miscompiled() {
    let src = |block_head: &str| {
        format!(
            r#"@grad fn loss(!w: Tensor[f32, [4]], x: Tensor[f32, [4]]) -> f32 {{
    {block_head} {{
        let d = w .* x
        sum(d)
    }}
}}

fn main() -> nil {{
    let !w = forge.ones[f32, [4]]
    let x = forge.ones[f32, [4]]
    let (l, g) = loss.fwd_bwd(w, x)
    print(l as i64)
    nil
}}
"#
        )
    };
    // The arena form and the plain form must answer identically: slice 1 gave
    // the JIT a lowering for the arena block, and that lowering must not reach
    // the backward pass by a route the plain block does not.
    for head in ["forge", ""] {
        let probe = Probe::new(&src(head));

        let run = probe.run(&["run"]);
        assert!(!run.status.success(), "[run {head}] returned a gradient: {}", stdout(&run));
        assert!(
            stderr(&run).contains("must produce a scalar derived from a differentiable input"),
            "[run {head}] wrong error: {}", stderr(&run),
        );

        let jit = probe.run(&["jit"]);
        assert!(!jit.status.success(), "[jit {head}] compiled it: {}", stdout(&jit));
        // A refusal with a class, not a `jit error` defect (#480) — and
        // `jit-grad`, because the backward pass is what cannot take it.
        let json = probe.run(&["jit", "--json"]);
        let first = stderr(&json).lines().next().unwrap_or_default().to_string();
        assert!(
            first.contains("\"kind\":\"jit-ineligible\"") && first.contains("\"code\":\"jit-grad\""),
            "[jit {head}] expected a jit-grad refusal, got:\n{}", first,
        );
    }
}
