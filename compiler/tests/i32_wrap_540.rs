//! #540: arithmetic on a fixed-width integer type wraps at that width, and
//! both backends must produce the same number.
//!
//! The two backends used to hold an `i32` differently: the interpreter held it
//! as an i64 and narrowed only at an explicit `as i32`, the JIT held it as a
//! `cl::I32` and narrowed everywhere. `let c: i32 = 2147483647 + 1` was
//! `Check OK` and printed 2147483648 under `dmc run`, -2147483648 under
//! `dmc jit`. SPEC §3.1 now names the rule and the interpreter implements it.
//!
//! These drive the actual `dmc` binary under BOTH backends because the class of
//! bug being tested is a backend divergence — an in-process interpreter test
//! cannot see it. The interpreter-side unit tests live in `src/interp_tests.rs`.
//!
//! `/`, `%` and `<<` are covered here too, which needs #541: before it, an i32
//! `/`, `%` or `<<` built its guard constants at i64 and reached a Cranelift
//! verifier error, and an untyped literal never adopted an `i32` operand's
//! width. Both halves of the neighbourhood have to be in place for these to
//! mean anything.

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

fn run_file(mode: &str, path: &PathBuf) -> Output {
    Command::new(dmc_binary())
        .args([mode, path.to_str().unwrap()])
        .output()
        .expect("invoke dmc")
}

fn run_src(name: &str, mode: &str, src: &str) -> Output {
    let tmp = std::env::temp_dir().join(format!("dmc_i32_wrap_{}_{}.dmc", name, mode));
    std::fs::write(&tmp, src).expect("write temp probe");
    let out = run_file(mode, &tmp);
    let _ = std::fs::remove_file(&tmp);
    out
}

/// Run `src` under `dmc run` and `dmc jit`; both must exit 0 and print exactly
/// `want`. Asserting the literal expected text (not just "the two agree") is
/// what makes this a rule test rather than a tautology — two backends can agree
/// on the wrong number.
fn assert_both_print(name: &str, src: &str, want: &str) {
    for mode in ["run", "jit"] {
        let out = run_src(name, mode, src);
        let so = String::from_utf8_lossy(&out.stdout).to_string();
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(out.status.success(), "[{mode} {name}] exited {:?}: {se}", out.status.code());
        assert_eq!(so, want, "[{mode} {name}] stdout (stderr: {se})");
    }
}

/// The issue's repro, literal-free: two `i32` locals and an add.
#[test]
fn the_repro_wraps_identically_on_both_backends() {
    let src = r#"fn main() -> nil {
    let a: i32 = 2147483647
    let b: i32 = 1
    let c: i32 = a + b
    print(c as i64)
    nil
}
"#;
    assert_both_print("repro", src, "-2147483648\n");
}

/// Add and subtract at both ends of the range, and multiply past it.
#[test]
fn add_sub_and_mul_wrap_at_both_boundaries() {
    let src = r#"fn main() -> nil {
    let max: i32 = 2147483647
    let min: i32 = -2147483648
    let one: i32 = 1
    let big: i32 = 100000
    print((max + one) as i64)
    print((min - one) as i64)
    print((big * big) as i64)
    print((min * min) as i64)
    nil
}
"#;
    assert_both_print("boundaries", src, "-2147483648\n2147483647\n1410065408\n0\n");
}

/// Negation is `0 - n`, so it wraps too: `-i32::MIN` is `i32::MIN`.
#[test]
fn negating_the_minimum_is_the_minimum() {
    let src = r#"fn main() -> nil {
    let min:  i32 = -2147483648
    let zero: i32 = 0
    print((zero - min) as i64)
    nil
}
"#;
    assert_both_print("neg_min", src, "-2147483648\n");
}

/// The wrapped value is what a comparison sees — the case that a fix narrowing
/// only at the `let` annotation would still have got wrong.
#[test]
fn a_comparison_sees_the_wrapped_value() {
    let src = r#"fn main() -> nil {
    let max:  i32 = 2147483647
    let one:  i32 = 1
    let zero: i32 = 0
    print((max + one) < zero)
    print((max + one) == (zero - max - one))
    nil
}
"#;
    assert_both_print("compare", src, "true\ntrue\n");
}

/// The explicit `as i32` cast is UNCHANGED by #540: it narrowed on both
/// backends before the rule and narrows the same way after it. This is the case
/// that already agreed, and its agreement is what made the arithmetic case easy
/// to miss — so it is pinned.
#[test]
fn the_explicit_cast_keeps_its_existing_behavior() {
    let src = r#"fn main() -> nil {
    let wide: i64 = 5000000000
    let neg:  i64 = -5000000000
    print((wide as i32) as i64)
    print((neg as i32) as i64)
    print((wide as i32) as i64)
    nil
}
"#;
    assert_both_print("cast", src, "705032704\n-705032704\n705032704\n");
}

/// …and the cast's result is an i32, so the arithmetic after it wraps.
#[test]
fn the_cast_result_is_itself_an_i32() {
    let src = r#"fn main() -> nil {
    let wide: i64 = 2147483647
    let one:  i32 = 1
    print(((wide as i32) + one) as i64)
    nil
}
"#;
    assert_both_print("cast_then_add", src, "-2147483648\n");
}

/// i64 is untouched: 2^31 fits, so the same expression one width up does not
/// wrap. This is the control — the rule is about the DECLARED width, not about
/// the magnitude 2^31.
#[test]
fn the_same_expression_at_i64_does_not_wrap() {
    let src = r#"fn main() -> nil {
    let a: i64 = 2147483647
    let b: i64 = 1
    print(a + b)
    nil
}
"#;
    assert_both_print("i64_control", src, "2147483648\n");
}

/// An untyped integer literal adopts the `i32` operand's width (§3.1) on both
/// backends: the JIT decides it at lowering time (#541's
/// `adopt_int_literal_kind`), the interpreter reads it off the operand's
/// runtime tag. Both operand orders, so neither side is order-dependent.
#[test]
fn an_untyped_literal_adopts_the_i32_width_on_both_backends() {
    let src = r#"fn main() -> nil {
    let max: i32 = 2147483647
    print((max + 1) as i64)
    print((1 + max) as i64)
    print((max * 2) as i64)
    print(max + 1 < 0)
    nil
}
"#;
    assert_both_print("literal_adopt", src, "-2147483648\n-2147483648\n-2\ntrue\n");
}

/// `/` and `%` at i32 width: truncating, sign follows the dividend, and
/// division by zero stays total at 0 (#208) rather than trapping.
#[test]
fn division_and_modulo_run_at_i32_width() {
    let src = r#"fn main() -> nil {
    let max: i32 = 2147483647
    let min: i32 = -2147483648
    let z:   i32 = 0
    print((max / 2) as i64)
    print((min / 2) as i64)
    print((max % 7) as i64)
    print((min % 7) as i64)
    print((max / z) as i64)
    print((max % z) as i64)
    nil
}
"#;
    assert_both_print("divmod", src,
        "1073741823\n-1073741824\n1\n-2\n0\n0\n");
}

/// `<<` shifts at i32 width: 1 << 31 is `i32::MIN`, not 2^31.
#[test]
fn shifts_run_at_i32_width() {
    let src = r#"fn main() -> nil {
    let one: i32 = 1
    let max: i32 = 2147483647
    let m1:  i32 = -1
    print((one << 30) as i64)
    print((one << 31) as i64)
    print((max << 1) as i64)
    print((m1 << 1) as i64)
    nil
}
"#;
    assert_both_print("shift", src, "1073741824\n-2147483648\n-2\n-2\n");
}

/// `MIN / -1` is the one division with no wrapped value. Both backends abort
/// with the SAME message, naming the width that actually overflowed — the
/// interpreter from `scalar_arith`, the JIT from `__dmc_div_overflow_trap`.
#[test]
fn min_over_minus_one_aborts_identically_naming_i32() {
    let src = r#"fn main() -> nil {
    let min: i32 = -2147483648
    let m1:  i32 = -1
    print((min / m1) as i64)
    nil
}
"#;
    let want = "runtime error: integer overflow: -2147483648 / -1 exceeds the i32 range\n";
    for mode in ["run", "jit"] {
        let out = run_src("div_overflow", mode, src);
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(!out.status.success(), "[{mode}] expected a nonzero exit");
        assert!(se.ends_with(want), "[{mode}] stderr was: {se}");
    }
}

/// The width survives every container whose element type is DECLARED — a
/// tuple, an enum payload, a model field, a closure parameter.
#[test]
fn the_width_survives_the_typed_containers() {
    let src = r#"enum Opt { Some(i32), None }
model Cell { n: i32 }
fn pair(a: i32, b: i32) -> (i32, i32) { (a, b) }
fn main() -> nil {
    let a:   i32 = 2147483647
    let one: i32 = 1
    let o = Opt.Some(a)
    print(match o { Some(n) => (n + one) as i64, None => 0 })
    let (p, q) = pair(a, one)
    print((p + q) as i64)
    let c = Cell { n: a }
    print((c.n + one) as i64)
    let f = fn(x: i32) -> i32 { x + one }
    print(f(a) as i64)
    nil
}
"#;
    assert_both_print("containers", src,
        "-2147483648\n-2147483648\n-2147483648\n-2147483648\n");
}

/// …and does NOT survive a map, whose value type is `any` (§3.1) and whose JIT
/// backing is a raw i64 store. Both backends read it back as an i64, so
/// `map_get(m, k) + 1` does not wrap. Pinned because the interpreter's map
/// holds real `Value`s and would otherwise have kept the width.
#[test]
fn the_width_does_not_survive_an_any_typed_map() {
    let src = r#"fn main() -> nil {
    let a: i32 = 2147483647
    let m = map_new()
    map_set(m, "k", a)
    print(map_get(m, "k") + 1)
    nil
}
"#;
    assert_both_print("map", src, "2147483648\n");
}

/// The `spec_probes/` probe for §3.1, run under both backends. The probe runner
/// (`spec_probes.rs`) only exercises the interpreter, so the parity half of that
/// probe is asserted here.
#[test]
fn the_spec_probe_agrees_across_backends() {
    let probe = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests").join("spec_probes").join("p57_i32_arith_wraps.dmc");
    assert!(probe.is_file(), "probe missing: {}", probe.display());

    let want = "-2147483648\n2147483647\n1410065408\n-2147483648\ntrue\n\
                -2147483648\n-2147483648\n-2147483648\n1073741823\n0\n\
                705032704\n705032705\n2147483648\n";

    let jit = run_file("jit", &probe);
    let jit_out = String::from_utf8_lossy(&jit.stdout).to_string();
    assert!(jit.status.success(), "jit failed: {}", String::from_utf8_lossy(&jit.stderr));
    assert_eq!(jit_out, want, "jit output");

    let run = run_file("run", &probe);
    let run_out = String::from_utf8_lossy(&run.stdout).to_string();
    assert!(run.status.success(), "run failed: {}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(run_out, want, "run output");

    // The bare invocation is what `spec_probes.rs` uses: it prints the pipeline
    // banner around the same body and must end in "✅ Run OK".
    let bare = Command::new(dmc_binary())
        .arg(probe.to_str().unwrap())
        .output()
        .expect("invoke dmc");
    let bare_out = String::from_utf8_lossy(&bare.stdout).to_string();
    assert!(bare.status.success(), "bare failed: {}", String::from_utf8_lossy(&bare.stderr));
    assert!(bare_out.contains(want), "bare output missing probe body:\n{bare_out}");
    assert!(bare_out.contains("Run OK"), "bare output missing Run OK:\n{bare_out}");
}
