//! #567: `\<x` and the stdlib `gelu(x)` are one function with one number.
//!
//! `OPERATORS.md` presents `\<` as "identical to the stdlib `gelu(x)`". The JIT
//! compiles both through a single `emit_gelu_f32`, so it held there. The
//! interpreter ran two kernels: the builtin computed the tanh approximation and
//! rounded the result through f32 (the #241/#473 convention that keeps `dmc run`
//! on the JIT's numbers), while `\<` kept the unrounded f64. The two answers
//! differed by ~1.6e-8, so `\<(x) == gelu(x)` was false under `dmc run` and true
//! under `dmc jit` — a program branching on it took different paths per backend.
//!
//! `\<` now calls the builtin's kernel, including its rounding, which is the
//! side the JIT is on: converging on the unrounded f64 instead would have made
//! the interpreter self-consistent and moved the divergence to the backends.
//!
//! Every test asserts on BOTH backends, since GeLU is fully JIT-supported.

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

fn run_src(name: &str, mode: &str, src: &str) -> Output {
    let tmp = std::env::temp_dir().join(format!("dmc_gelu_{}_{}.dmc", name, mode));
    std::fs::write(&tmp, src).expect("write temp probe");
    let out = Command::new(dmc_binary())
        .args([mode, tmp.to_str().unwrap()])
        .output()
        .expect("invoke dmc");
    let _ = std::fs::remove_file(&tmp);
    out
}

fn assert_both_print(name: &str, src: &str, want: &str) {
    for mode in ["run", "jit"] {
        let out = run_src(name, mode, src);
        let so = String::from_utf8_lossy(&out.stdout).to_string();
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(out.status.success(), "[{mode} {name}] exited {:?}: {se}", out.status.code());
        assert_eq!(so, want, "[{mode} {name}] stdout (stderr: {se})");
    }
}

/// The issue's repro, on both backends: equal, and the difference is exactly 0.
#[test]
fn the_567_repro_holds_on_both_backends() {
    let src = r#"fn main() -> nil {
    let x: f64 = 1.0
    print(\<(x) == gelu(x))
    print(\<(x) - gelu(x))
    nil
}
"#;
    assert_both_print("repro567", src, "true\n0\n");
}

/// Not just equal to each other — equal at the value the JIT computes, over a
/// spread of arguments including both signs and zero. A shared-but-wrong kernel
/// would pass an equality-only test.
#[test]
fn both_spellings_print_the_jit_value() {
    let src = r#"fn main() -> nil {
    let x: f64 = 1.0
    print(\<(x))
    print(gelu(x))
    let z: f64 = 0.0
    print(\<(z))
    print(gelu(z))
    nil
}
"#;
    assert_both_print("geluvalue", src,
        "0.8411920070648193\n0.8411920070648193\n0\n0\n");
}

/// The identity holds for an `f32` argument too, where the operand width is the
/// one the JIT computes in natively.
#[test]
fn the_two_spellings_agree_at_f32() {
    let src = r#"fn main() -> nil {
    let x: f32 = 0.7
    print(\<(x) == gelu(x))
    print(\<(x))
    nil
}
"#;
    assert_both_print("gelu32", src, "true\n0.5305701494216919\n");
}

/// And elementwise over a tensor, which is a separate lowering on both sides:
/// `\<T` walks its own loop while `gelu(T)` goes through the activation path.
///
/// This one asserts the two spellings agree WITHIN each backend rather than
/// pinning digits across both: at an argument like -1.0 the interpreter's
/// f64 `tanh` rounded to f32 still lands an ulp off the JIT's native f32
/// `tanh`. That residual is #241/#473's, it predates this fix, and it now moves
/// the two spellings identically — which is the whole of what #567 asked for.
#[test]
fn the_two_spellings_agree_over_a_tensor() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[f32, [3]]
    t[0] = -1.0
    t[1] = 0.0
    t[2] = 2.0
    let a = \<(t)
    let b = gelu(t)
    print(sum(a .- b))
    print(a[0] == b[0])
    print(a[2] == b[2])
    nil
}
"#;
    assert_both_print("gelutensor", src, "0\ntrue\ntrue\n");
}
