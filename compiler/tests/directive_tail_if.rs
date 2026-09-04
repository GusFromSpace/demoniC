//! A trailing `@directive { … }` whose body ends in an `if` / `match`
//! statement yields that statement's value, on both backends.
//!
//! `fn main() -> i64 { @deterministic { if c { 10 } else { 20 } } }` parses the
//! directive block as a `Stmt::DirectiveBlock` (not a tail expression), and its
//! body's `if` as a keyword-led statement — so the body has no `tail_expr`. The
//! checker and the interpreter both read only that `tail_expr` when taking the
//! block's value, typed the fn body as nil, and `dmc --check` refused it. Bound
//! to a `let` first, the identical block took the `Expr::DirectiveBlock` path,
//! which evaluates the body as an ordinary block, and was right all along. The
//! JIT already lowered the trailing form through its ordinary block path.
//!
//! These drive the actual `dmc` binary under BOTH backends: `dmc run` and
//! `dmc jit` each run the checker first, so a checker regression shows up as a
//! refusal, and a backend divergence is invisible to an in-process test. Each
//! case asserts the literal expected output rather than "the two agree".

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

fn run_src(name: &str, mode: &str, src: &str) -> Output {
    let tmp = std::env::temp_dir().join(format!("dmc_directive_tail_{}_{}.dmc", name, mode));
    std::fs::write(&tmp, src).expect("write temp probe");
    let out = Command::new(dmc_binary())
        .args([mode, tmp.to_str().unwrap()])
        .output()
        .expect("invoke dmc");
    let _ = std::fs::remove_file(&tmp);
    out
}

/// Run `src` under `dmc run` and `dmc jit`; both must exit 0 and print exactly
/// `want`.
fn assert_both_print(name: &str, src: &str, want: &str) {
    for mode in ["run", "jit"] {
        let out = run_src(name, mode, src);
        let so = String::from_utf8_lossy(&out.stdout).to_string();
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(out.status.success(), "[{mode} {name}] exited {:?}: {se}", out.status.code());
        assert_eq!(so, want, "[{mode} {name}] stdout (stderr: {se})");
    }
}

/// The repro: a trailing `@deterministic` block ending in an `if` statement.
#[test]
fn trailing_deterministic_block_ending_in_if_yields_on_both_backends() {
    let src = r#"fn pick(c: bool) -> i64 { @deterministic { if c { 10 } else { 20 } } }

fn main() -> nil { print(pick(true))  print(pick(false))  nil }
"#;
    assert_both_print("if", src, "10\n20\n");
}

/// The `match` statement form takes the same arm.
#[test]
fn trailing_deterministic_block_ending_in_match_yields_on_both_backends() {
    let src = r#"fn pick(n: i64) -> i64 { @deterministic { match n { 3 => 10, _ => 20 } } }

fn main() -> nil { print(pick(3))  print(pick(4))  nil }
"#;
    assert_both_print("match", src, "10\n20\n");
}

/// A `let` inside the block is visible to the trailing `if`, and the value
/// still comes out — the block is walked as a block, not as tail-expr-only.
#[test]
fn trailing_block_inner_let_feeds_its_if_on_both_backends() {
    let src = r#"fn pick(n: i64) -> i64 { @deterministic { let k = n * 2  if k > 5 { k } else { 0 - k } } }

fn main() -> nil { print(pick(3))  print(pick(1))  nil }
"#;
    assert_both_print("inner_let", src, "6\n-2\n");
}
