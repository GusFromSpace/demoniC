//! #574: a binding introduced by a match arm or a loop belongs to that region,
//! and must not survive it. Both backends must produce the same number.
//!
//! The JIT keeps `locals` as one flat map for the whole function. That is right
//! for `let` — a shadow really does run to the end of the function — and wrong
//! for every construct whose binding belongs to a region. Those wrote into the
//! map and never put back what they displaced, so an outer local of the same
//! name was destroyed for the remainder of the function.
//!
//! The sharp edge is that the map is consulted while lowering, which knows
//! nothing about which branch runs: an arm that never executes still clobbered
//! the outer name, and the clobbered local then read as 0. `dmc run` printed
//! 2.5 and `dmc jit` printed 0.5 for the issue's repro, both exiting 0. A
//! silent wrong answer, not a refusal.
//!
//! The issue reports the enum-payload case. Two more sites had the identical
//! defect and are covered here: a catch-all arm that binds the scrutinee, and a
//! `for` loop's index variable. All three are one root cause and one fix.
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
    let tmp = std::env::temp_dir().join(format!("dmc_bind_scope_{}_{}.dmc", name, mode));
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

/// The issue's repro. `Shape.Empty` is the scrutinee, so the `Circle(r)` arm
/// never runs — and the outer `r` must still be 2.0 afterwards.
#[test]
fn the_574_repro_keeps_the_outer_local_on_both_backends() {
    let src = r#"enum Shape { Circle(f32), Empty }

fn probe(s: Shape) -> f32 {
    let r = 2.0f32
    let a = match s {
        Circle(r) => r * r,
        Empty     => 0.0f32,
    }
    r + 0.5f32
}

fn main() -> nil { print(probe(Shape.Empty))  nil }
"#;
    assert_both_print("repro", src, "2.5\n");
}

/// The taken-arm direction: the payload binding has to actually work inside its
/// own arm. A fix that scoped the name by never binding it would pass the test
/// above and fail this one.
#[test]
fn the_payload_binding_still_works_inside_its_own_arm() {
    let src = r#"enum Shape { Circle(f32), Empty }

fn probe(s: Shape) -> f32 {
    let r = 2.0f32
    let a = match s {
        Circle(r) => r * r,
        Empty     => 0.0f32,
    }
    a + r
}

fn main() -> nil { print(probe(Shape.Circle(3.0f32)))  nil }
"#;
    // 3*3 from the payload r, plus the outer r restored to 2.0.
    assert_both_print("taken_arm", src, "11\n");
}

/// A catch-all arm binding the scrutinee is the same defect: `n` is rebound by
/// an arm that does not run when the literal arm matches first.
#[test]
fn a_catch_all_arm_binding_does_not_clobber_an_outer_local() {
    let src = r#"fn probe(s: i64) -> i64 {
    let n = 7
    let a = match s {
        1 => 100,
        n => n * 2,
    }
    n + a
}

fn main() -> nil { print(probe(1))  nil }
"#;
    assert_both_print("catch_all", src, "107\n");
}

/// And when the catch-all *is* the arm taken, the binding must still be the
/// scrutinee inside it, with the outer name back afterwards.
#[test]
fn a_taken_catch_all_binds_the_scrutinee_then_restores() {
    let src = r#"fn probe(s: i64) -> i64 {
    let n = 7
    let a = match s {
        1 => 100,
        n => n * 2,
    }
    n + a
}

fn main() -> nil { print(probe(5))  nil }
"#;
    // The arm sees n = 5 (so a = 10); afterwards the outer n is 7 again.
    assert_both_print("catch_all_taken", src, "17\n");
}

/// A `for` loop's index does not outlive the loop. The JIT left it bound to the
/// last value the loop reached, so the outer `i` read as 2 instead of 42.
#[test]
fn a_for_loop_index_does_not_outlive_the_loop() {
    let src = r#"fn probe() -> i64 {
    let i = 42
    for i in 0..3 { }
    i
}

fn main() -> nil { print(probe())  nil }
"#;
    assert_both_print("for_index", src, "42\n");
}

/// The loop body must still see the index while the loop is running.
#[test]
fn the_for_loop_index_is_visible_inside_the_body() {
    let src = r#"fn probe() -> i64 {
    let i = 42
    let !acc = 0
    for i in 0..4 { acc = acc + i }
    acc + i
}

fn main() -> nil { print(probe())  nil }
"#;
    // 0+1+2+3 = 6, plus the outer i restored to 42.
    assert_both_print("for_body", src, "48\n");
}

/// Nested regions restore in the right order: the inner arm's binding must not
/// put back the *function's* value over the outer arm's.
#[test]
fn nested_bindings_restore_to_the_enclosing_region_not_the_function() {
    let src = r#"fn probe(a: i64, b: i64) -> i64 {
    let n = 1
    let outer = match a {
        0 => 0,
        n => {
            let inner = match b {
                0 => 0,
                n => n * 10,
            }
            inner + n
        },
    }
    outer + n
}

fn main() -> nil { print(probe(5, 7))  nil }
"#;
    // inner arm: n = 7 -> 70; back in the outer arm n is 5 -> 75;
    // back in the function n is 1 -> 76.
    assert_both_print("nested", src, "76\n");
}
