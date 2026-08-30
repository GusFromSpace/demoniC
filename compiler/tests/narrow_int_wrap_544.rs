//! #544 / #545: EVERY fixed-width integer type wraps at its declared width, on
//! both backends, and an integer literal's type suffix is one of the places
//! that width originates.
//!
//! #540 did `i32` alone, because `i32` was the only narrow integer
//! `ScalarKind` the JIT had. The remaining kinds were i64-backed in the
//! interpreter (so `i16` arithmetic did not wrap) and outright REFUSED by the
//! JIT (`scalar type `I16` not yet supported`), so the rule §3.1 states was
//! unenforced at seven widths. #544 gave the JIT masked i64-backed kinds for
//! `i8 i16 u8 u16 u32 int4 int8` — the value lives in an i64 register holding
//! the wrapped result sign- or zero-extended, and every arithmetic op re-masks
//! — which is what the interpreter's `IW` already did.
//!
//! #545 is the other half: the JIT dropped a suffixed literal's `i32`, so
//! `2147483647i32 + 1i32` did not wrap. Both backends now originate a width
//! from the suffix.
//!
//! These drive the real `dmc` binary under BOTH backends, because the class of
//! bug is a backend divergence an in-process test cannot see. The
//! interpreter-side unit tests live in `src/interp_tests.rs`; the `i32`
//! neighbourhood is `tests/i32_wrap_540.rs`.

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

fn run_file(mode: &str, path: &PathBuf) -> Output {
    Command::new(dmc_binary())
        .args([mode, path.to_str().unwrap()])
        .output()
        .expect("invoke dmc")
}

fn run_src(name: &str, mode: &str, src: &str) -> Output {
    let tmp = std::env::temp_dir().join(format!("dmc_narrow_wrap_{}_{}.dmc", name, mode));
    std::fs::write(&tmp, src).expect("write temp probe");
    let out = Command::new(dmc_binary())
        .args([mode, tmp.to_str().unwrap()])
        .output()
        .expect("invoke dmc");
    let _ = std::fs::remove_file(&tmp);
    out
}

/// Run `src` under `dmc run` and `dmc jit`; both must exit 0 and print exactly
/// `want`. Asserting the literal expected text (not just "the two agree") is
/// what makes this a rule test rather than a tautology — two backends can agree
/// on the wrong number, which is exactly the state #544 and #545 found.
fn assert_both_print(name: &str, src: &str, want: &str) {
    for mode in ["run", "jit"] {
        let out = run_src(name, mode, src);
        let so = String::from_utf8_lossy(&out.stdout).to_string();
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(out.status.success(), "[{mode} {name}] exited {:?}: {se}", out.status.code());
        assert_eq!(so, want, "[{mode} {name}] stdout (stderr: {se})");
    }
}

/// Both backends must fail the same way, with a message containing `needle`.
fn assert_both_fail(name: &str, src: &str, needle: &str) {
    for mode in ["run", "jit"] {
        let out = run_src(name, mode, src);
        let all = format!("{}{}",
            String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(!out.status.success(), "[{mode} {name}] unexpectedly succeeded: {all}");
        assert!(all.contains(needle), "[{mode} {name}] wanted {needle:?}, got: {all}");
    }
}

/// The issue's repro: `i16` MAX + 1 is MIN, not 32768.
#[test]
fn the_544_repro_wraps_identically_on_both_backends() {
    let src = r#"fn main() -> nil {
    let a: i16 = 32767
    let b: i16 = 1
    print((a + b) as i64)
    nil
}
"#;
    assert_both_print("repro544", src, "-32768\n");
}

/// The issue's #545 repro: the SUFFIX types the literal, so the add wraps.
#[test]
fn the_545_repro_wraps_identically_on_both_backends() {
    let src = r#"fn main() -> nil {
    let a = 2147483647i32
    let b = 1i32
    print((a + b) as i64)
    nil
}
"#;
    assert_both_print("repro545", src, "-2147483648\n");
}

/// Every signed width: MAX + 1 is MIN, MIN - 1 is MAX.
#[test]
fn signed_widths_wrap_at_both_boundaries() {
    let src = r#"fn main() -> nil {
    let a8: i8 = 127
    let n8: i8 = -128
    let o8: i8 = 1
    print((a8 + o8) as i64)
    print((n8 - o8) as i64)
    let a16: i16 = 32767
    let n16: i16 = -32768
    let o16: i16 = 1
    print((a16 + o16) as i64)
    print((n16 - o16) as i64)
    let a32: i32 = 2147483647
    let n32: i32 = -2147483648
    let o32: i32 = 1
    print((a32 + o32) as i64)
    print((n32 - o32) as i64)
    nil
}
"#;
    assert_both_print("signed_bounds", src,
        "-128\n127\n-32768\n32767\n-2147483648\n2147483647\n");
}

/// Unsigned wrap is UNSIGNED: `0u32 - 1u32` is 4294967295, held zero-extended,
/// and that zero-extension is what makes the later `/`, `>>` and `>` unsigned
/// too — a non-negative i64 divides, shifts and compares as the unsigned value.
#[test]
fn unsigned_widths_wrap_unsigned_and_stay_unsigned() {
    let src = r#"fn main() -> nil {
    let z8: u8 = 0
    let z16: u16 = 0
    let z32: u32 = 0
    let o8: u8 = 1
    let o16: u16 = 1
    let o32: u32 = 1
    print((z8 - o8) as i64)
    print((z16 - o16) as i64)
    print((z32 - o32) as i64)
    let two: u32 = 2
    print(((z32 - o32) / two) as i64)
    print(((z32 - o32) >> two) as i64)
    print((z32 - o32) > o32)
    nil
}
"#;
    assert_both_print("unsigned", src,
        "255\n65535\n4294967295\n2147483647\n1073741823\ntrue\n");
}

/// `u64` is the one width an i64-backed value cannot fully hold, so it carries
/// the 64-bit two's-complement PATTERN. Wrapping at 64 bits is bit-identical to
/// `i64`'s — which is all §3.1's rule asks of it — but a value above 2^63
/// renders signed. Pinned so the limitation is on record on both backends.
#[test]
fn u64_wraps_at_64_bits_as_a_signed_pattern() {
    let src = r#"fn main() -> nil {
    let z: u64 = 0
    let o: u64 = 1
    print((z - o) as i64)
    nil
}
"#;
    assert_both_print("u64", src, "-1\n");
}

/// Multiplication past the width, and the wrapped value seen by a comparison —
/// the case a fix that narrowed only at the `let` annotation would still miss.
#[test]
fn multiplication_wraps_and_the_comparison_sees_it() {
    let src = r#"fn main() -> nil {
    let big: i16 = 300
    print((big * big) as i64)
    let max: i16 = 32767
    let one: i16 = 1
    print((max + one) < 0)
    nil
}
"#;
    assert_both_print("mul_cmp", src, "24464\ntrue\n");
}

/// `~` complements the operand's WIDTH (`~0u8` is 255), and negating MIN is
/// MIN at every signed width.
#[test]
fn bitwise_not_and_negation_follow_the_width() {
    let src = r#"fn main() -> nil {
    let z: u8 = 0
    print((~z) as i64)
    let min: i8 = -128
    let zero: i8 = 0
    print((zero - min) as i64)
    nil
}
"#;
    assert_both_print("not_neg", src, "255\n-128\n");
}

/// `<<` is bounded by the SHIFTED VALUE's width — an `i8` shift tops out at 7 —
/// and the result wraps at that width. Out of range is an error on both
/// backends, naming the same range.
#[test]
fn shifts_are_bounded_by_the_operand_width() {
    let ok = r#"fn main() -> nil {
    let a: i8 = 1
    let k: i8 = 7
    print((a << k) as i64)
    nil
}
"#;
    assert_both_print("shl_ok", ok, "-128\n");
    // A runtime count, so the guard is the emitted range check rather than the
    // compile-time constant rejection.
    let bad = r#"fn main() -> nil {
    let a: i8 = 1
    let zero: i8 = 0
    let k: i8 = 8 + zero
    print((a << k) as i64)
    nil
}
"#;
    assert_both_fail("shl_bad", bad, "expected 0..=7");
}

/// `MIN / -1` has no wrapped value at any width and is an error naming the
/// width that overflowed; `/` and `%` by zero stay total (#208) at every width.
#[test]
fn division_edge_cases_agree_at_every_width() {
    let bad = r#"fn main() -> nil {
    let a: i16 = -32768
    let b: i16 = -1
    print((a / b) as i64)
    nil
}
"#;
    assert_both_fail("divmin", bad, "exceeds the i16 range");
    let zero = r#"fn main() -> nil {
    let a: i8 = 127
    let z: i8 = 0
    print(((a / z) + (a % z)) as i64)
    let u: u16 = 65535
    let uz: u16 = 0
    print(((u / uz) + (u % uz)) as i64)
    nil
}
"#;
    assert_both_print("divzero", zero, "0\n0\n");
}

/// An `as` cast to a narrow type is a width origin as well as a narrowing, so
/// what follows the cast wraps — §3.1's "the wrapped value is the value".
#[test]
fn a_narrow_as_cast_carries_its_width_forward() {
    let src = r#"fn main() -> nil {
    let wide: i64 = 100000
    let one: i16 = 1
    print(((wide as i16) + one) as i64)
    let full: i64 = 255
    let uone: u8 = 1
    print(((full as u8) + uone) as i64)
    nil
}
"#;
    assert_both_print("cast_width", src, "-31071\n0\n");
}

/// A parameter and a declared return are width origins too, at every kind.
#[test]
fn a_parameter_and_a_return_originate_the_width() {
    let src = r#"fn bump(x: u8) -> u8 { x + 1 }
fn wrapped() -> i8 { 127 + 1 }
fn main() -> nil {
    let a: u8 = 255
    print(bump(a) as i64)
    print(wrapped() as i64)
    nil
}
"#;
    assert_both_print("param_ret", src, "0\n-128\n");
}

/// The packed `int4` / `int8` kinds of §3.1 are signed 4-/8-bit types: they
/// wrap at those widths and an `as` cast to them narrows.
#[test]
fn the_packed_int4_and_int8_kinds_wrap() {
    let src = r#"fn main() -> nil {
    let a: int8 = 127
    let b: int8 = 1
    print((a + b) as i64)
    let p: int4 = 7
    let q: int4 = 1
    print((p + q) as i64)
    let wide: i64 = 9
    print((wide as int4) as i64)
    nil
}
"#;
    assert_both_print("packed", src, "-128\n-8\n-7\n");
}

/// An untyped literal adopts a narrow operand's width (§3.1's untyped-literal
/// rule, the JIT's `adopt_int_literal_kind`), and a suffix names it outright.
#[test]
fn literals_adopt_or_name_the_narrow_width() {
    let src = r#"fn main() -> nil {
    let a: i8 = 127
    print((a + 1) as i64)
    let b = 127i8
    print((b + 1i8) as i64)
    let c = 0u16
    print((c - 1u16) as i64)
    nil
}
"#;
    assert_both_print("literals", src, "-128\n-128\n65535\n");
}

/// The width survives the typed containers — a model field, an enum payload
/// and a tuple are all declared narrow on both backends, the #540 test's shape
/// one width down. (The JIT's enum-payload allowlist was widened to the
/// fixed-width kinds by #544; before that an `i8` payload routed the program
/// out of the JIT subset rather than diverging.)
#[test]
fn the_width_survives_typed_containers() {
    let src = r#"model Counter {
    lo: u8
    hi: i16
}

enum Opt { Some(i8), None }

fn pair(a: i8, b: i8) -> (i8, i8) { (a, b) }

fn main() -> nil {
    let c = Counter { lo: 255, hi: 32767 }
    let one8: u8 = 1
    let one16: i16 = 1
    print((c.lo + one8) as i64)
    print((c.hi + one16) as i64)
    let a: i8 = 127
    let one: i8 = 1
    let o = Opt.Some(a)
    let x = match o { Some(n) => n + one, None => one }
    print(x as i64)
    let (p, q) = pair(a, one)
    print((p + q) as i64)
    nil
}
"#;
    assert_both_print("containers", src, "0\n-32768\n-128\n-128\n");
}

/// The `spec_probes/` probe for §3.1's narrow widths, run under both backends.
/// The probe runner (`spec_probes.rs`) only exercises the interpreter, so the
/// parity half of that probe is asserted here — the same split
/// `i32_wrap_540.rs` uses for probe 57.
#[test]
fn the_spec_probe_agrees_across_backends() {
    let probe = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests").join("spec_probes").join("p58_narrow_int_arith_wraps.dmc");
    assert!(probe.is_file(), "probe missing: {}", probe.display());

    let want = "-128\n127\n-32768\n24464\n255\n255\n\
                4294967295\n2147483647\n1073741823\ntrue\n-1\n-128\n\
                -31072\n-31071\n-2147483648\n-128\n-128\n-8\n";

    for mode in ["run", "jit"] {
        let out = run_file(mode, &probe);
        let so = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(out.status.success(), "[{mode}] probe failed: {}",
            String::from_utf8_lossy(&out.stderr));
        assert_eq!(so, want, "[{mode}] probe output");
    }

    // The bare invocation is what `spec_probes.rs` uses: it prints the pipeline
    // banner around the same body and must end in "✅ Run OK".
    let bare = Command::new(dmc_binary())
        .arg(probe.to_str().unwrap())
        .output()
        .expect("invoke dmc");
    let bare_out = String::from_utf8_lossy(&bare.stdout).to_string();
    assert!(bare.status.success(), "bare pipeline failed: {}",
        String::from_utf8_lossy(&bare.stderr));
    assert!(bare_out.contains("Run OK"), "bare pipeline output: {bare_out}");
}
