//! #630: a `let` inside a block belongs to that block, and must not survive it.
//! Both backends must produce the same number.
//!
//! This is the #574 family's last member. The JIT keeps `locals` as one flat
//! map for the whole function, and the shadow/unshadow bookkeeping #574 added
//! covered the regions it knew about — match arms and a `for` index — but not
//! the block itself. So `let a = 1  let b = { let a = 2  a + 10 }  a + b`
//! answered 13 under `dmc run` and 14 under `dmc jit`: the inner `a` overwrote
//! the outer one for the rest of the function. Both backends exited 0. A
//! silent wrong answer, not a refusal.
//!
//! Every construct whose body is a block had it: a block in expression
//! position, a block in statement position, an `if` branch, a `while` / `loop`
//! / `for` body, a directive block, a block used as a match arm. Each is
//! covered below, because they reach the JIT through four different lowering
//! paths and a fix to one of them is not a fix to the others.
//!
//! The two directions that must NOT regress are here as well: the binding has
//! to work *inside* its own block (a fix that scoped the name by never binding
//! it would pass every leak test), and a write to an outer `let !` from inside
//! a block has to keep writing through to the outer binding.
//!
//! These drive the actual `dmc` binary under BOTH backends, because a backend
//! divergence is invisible to an in-process interpreter test. Each case asserts
//! the literal expected output rather than "the two agree" — two backends can
//! agree on the wrong number, and a test that only compares them would have
//! passed happily on the day the JIT was wrong.

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
    let tmp = std::env::temp_dir().join(format!("dmc_block_scope_{}_{}.dmc", name, mode));
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

/// The issue's repro, verbatim.
#[test]
fn the_630_repro_scopes_the_inner_let_to_its_block() {
    let src = r#"fn probe() -> i64 {
    let a = 1
    let b = { let a = 2  a + 10 }
    a + b
}

fn main() -> nil { print(probe())  nil }
"#;
    // The block yields 12; afterwards `a` is 1 again.
    assert_both_print("repro", src, "13\n");
}

/// The other direction: the shadowing binding has to actually be in effect
/// inside its own block. A fix that scoped the name by never binding it would
/// pass every leak test here and fail this one.
#[test]
fn the_shadowing_let_is_in_effect_inside_its_own_block() {
    let src = r#"fn probe() -> i64 {
    let a = 1
    let b = { let a = 2  a * 10 }
    a * 1000 + b
}

fn main() -> nil { print(probe())  nil }
"#;
    // Inside the block a is 2 (b = 20); outside it is 1 again.
    assert_both_print("inner_visible", src, "1020\n");
}

/// Nested blocks restore to the *enclosing block's* binding, not the
/// function's — the block counterpart of #574's nested-arm case.
#[test]
fn nested_blocks_restore_to_the_enclosing_block_not_the_function() {
    let src = r#"fn probe() -> i64 {
    let a = 1
    let b = { let a = 2  let c = { let a = 3  a * 100 }  c + a }
    a + b
}

fn main() -> nil { print(probe())  nil }
"#;
    // inner block: a = 3 -> 300; back in the outer block a is 2 -> 302;
    // back in the function a is 1 -> 303.
    assert_both_print("nested", src, "303\n");
}

/// A block in *statement* position reaches the JIT through a different arm
/// than a block in value position, and is a scope just the same.
#[test]
fn a_block_in_statement_position_is_still_a_scope() {
    let src = r#"fn probe() -> i64 {
    let a = 1
    { let a = 2  a }
    a
}

fn main() -> nil { print(probe())  nil }
"#;
    assert_both_print("stmt_position", src, "1\n");
}

/// An `if` branch is a block, and its `let` dies with the branch — including
/// when the branch is the one that runs.
#[test]
fn a_shadowing_let_in_an_if_branch_dies_with_the_branch() {
    let src = r#"fn probe(c: bool) -> i64 {
    let a = 1
    if c { let a = 2 } else { let a = 3 }
    a
}

fn main() -> nil { print(probe(true) * 10 + probe(false))  nil }
"#;
    assert_both_print("if_branch", src, "11\n");
}

/// A `while` body is a block. The JIT lowered its statements straight into the
/// enclosing scope, so a shadowing `let` in the body outlived the loop.
#[test]
fn a_shadowing_let_in_a_while_body_dies_with_the_loop() {
    let src = r#"fn probe() -> i64 {
    let a = 1
    let !i = 0
    while i < 3 { let a = 2  i = i + a }
    a + i
}

fn main() -> nil { print(probe())  nil }
"#;
    // The body sees a = 2 (so i reaches 4); afterwards a is 1 again.
    assert_both_print("while_body", src, "5\n");
}

/// Same for `loop`.
#[test]
fn a_shadowing_let_in_a_loop_body_dies_with_the_loop() {
    let src = r#"fn probe() -> i64 {
    let a = 1
    let !i = 0
    loop { let a = 2  i = i + a  if i >= 4 { break } }
    a + i
}

fn main() -> nil { print(probe())  nil }
"#;
    assert_both_print("loop_body", src, "5\n");
}

/// And for a `for` body — #574 scoped the loop *index*, but not the `let`s the
/// body introduces alongside it.
#[test]
fn a_shadowing_let_in_a_for_body_dies_with_the_loop() {
    let src = r#"fn probe() -> i64 {
    let a = 1
    let !s = 0
    for i in 0..3 { let a = 2  s = s + a }
    a + s
}

fn main() -> nil { print(probe())  nil }
"#;
    // The body sees a = 2 three times (s = 6); afterwards a is 1 again.
    assert_both_print("for_body", src, "7\n");
}

/// A directive block is a block. `@deterministic` is a pass-through in both
/// backends, which makes it the cheapest way to reach that lowering arm.
#[test]
fn a_shadowing_let_in_a_directive_block_dies_with_the_block() {
    let src = r#"fn probe() -> i64 {
    let a = 1
    @deterministic { let a = 2 }
    a
}

fn main() -> nil { print(probe())  nil }
"#;
    assert_both_print("directive_block", src, "1\n");
}

/// A block used as a match arm body: the arm's own bindings were already
/// scoped by #574, the block's `let`s were not.
#[test]
fn a_shadowing_let_in_a_match_arm_block_dies_with_the_arm() {
    let src = r#"fn probe(k: i64) -> i64 {
    let a = 1
    let b = match k {
        1 => { let a = 9  a + 1 },
        _ => 0,
    }
    a * 100 + b
}

fn main() -> nil { print(probe(1))  nil }
"#;
    assert_both_print("match_arm_block", src, "110\n");
}

/// Restoring the outer binding must restore the *binding*, not a stale value:
/// a write to the outer name after the block has to see what the function last
/// wrote, and the shadow must not have consumed the write.
#[test]
fn a_shadow_is_undone_before_a_later_write_to_the_outer_binding() {
    let src = r#"fn probe() -> i64 {
    let !x = 1
    let b = { let x = 100  x + 1 }
    x = x + 2
    x * 1000 + b
}

fn main() -> nil { print(probe())  nil }
"#;
    // The block's x is 100 (b = 101); the outer x is 1, then 3.
    assert_both_print("shadow_then_write", src, "3101\n");
}

/// The must-not-regress direction for mutation: a block that does NOT shadow
/// writes through to the outer `let !` binding, and the write survives the
/// block.
#[test]
fn a_write_to_an_outer_mut_binding_from_inside_a_block_writes_through() {
    let src = r#"fn probe() -> i64 {
    let !x = 1
    let b = { x = x + 4  x * 2 }
    x * 100 + b
}

fn main() -> nil { print(probe())  nil }
"#;
    // The block sets x to 5 and yields 10; x is still 5 afterwards.
    assert_both_print("write_through", src, "510\n");
}
