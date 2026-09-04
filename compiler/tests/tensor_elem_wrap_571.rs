//! #571: a dotted elementwise op on a narrow-integer tensor wraps at the
//! ELEMENT width, and so does the element load that feeds it.
//!
//! SPEC §3.1 makes a fixed-width integer's width real at run time, and #540 /
//! #544 enforced that for scalars on both backends. A tensor element's width is
//! part of its type just as much — but the interpreter's tensor dtype tag was
//! width-less (`DType::Int`), so `a .+ b` on two `Tensor[i32, …]` was computed
//! in the f64 backing lanes: it did not wrap (`2147483647 .+ 1` answered
//! 2147483648, which is not an i32), `./` was float division rather than
//! integer division, and a division by zero was ∞ rather than §3.1's 0. The tag
//! now carries the width (`DType::Int(IW)`) and the elementwise loop calls the
//! same `int_arith` the scalar path does.
//!
//! The JIT refuses integer elementwise ops outright ("elementwise ops are
//! f32-only"), which is the only reason this was not a two-answer divergence.
//! `the_jit_still_refuses_integer_elementwise` pins that refusal, so the day it
//! is lifted, whoever lifts it has a correct interpreter to match — and these
//! expectations to match it against. Every case asserts the literal number, not
//! merely that the backends agree: two backends can agree on a wrong answer.
//!
//! Tensor element types the JIT can allocate are `f32 f64 i32 i64 bool`, so the
//! narrower widths are asserted under `dmc run` alone; everything the JIT can
//! run (element loads, the i64 control) is asserted under both.

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
    let tmp = std::env::temp_dir().join(format!("dmc_elem_wrap_{}_{}.dmc", name, mode));
    std::fs::write(&tmp, src).expect("write temp probe");
    let out = Command::new(dmc_binary())
        .args([mode, tmp.to_str().unwrap()])
        .output()
        .expect("invoke dmc");
    let _ = std::fs::remove_file(&tmp);
    out
}

/// Run `src` under one backend; it must exit 0 and print exactly `want`.
fn assert_prints(name: &str, mode: &str, src: &str, want: &str) {
    let out = run_src(name, mode, src);
    let so = String::from_utf8_lossy(&out.stdout).to_string();
    let se = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "[{mode} {name}] exited {:?}: {se}", out.status.code());
    assert_eq!(so, want, "[{mode} {name}] stdout (stderr: {se})");
}

/// The interpreter is the only backend that runs integer elementwise ops.
fn assert_run_prints(name: &str, src: &str, want: &str) {
    assert_prints(name, "run", src, want);
}

/// Both backends run it, and both print the same literal answer.
fn assert_both_print(name: &str, src: &str, want: &str) {
    for mode in ["run", "jit"] {
        assert_prints(name, mode, src, want);
    }
}

/// The program fails under `dmc run` with a message containing `needle`.
fn assert_run_fails(name: &str, src: &str, needle: &str) {
    let out = run_src(name, "run", src);
    let all = format!("{}{}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(!out.status.success(), "[run {name}] unexpectedly succeeded: {all}");
    assert!(all.contains(needle), "[run {name}] wanted {needle:?}, got: {all}");
}

/// The issue's repro: the scalar path and the elementwise path over the same
/// two values now answer the same number, and it is the one that fits an i32.
#[test]
fn the_571_repro_agrees_with_the_scalar_path() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[i32, [2]]
    a[0] = 2147483647
    let !b = forge.zeros[i32, [2]]
    b[0] = 1
    let sa: i32 = a[0]
    let sb: i32 = b[0]
    print((sa + sb) as i64)
    let c = a .+ b
    print(c[0] as i64)
    nil
}
"#;
    assert_run_prints("repro571", src, "-2147483648\n-2147483648\n");
}

/// The refusal that is currently hiding the divergence. If the JIT gains
/// integer elementwise lowering, this test fails — deliberately: whoever adds
/// it must come here, delete this test, and gate the rest of this file on both
/// backends instead.
#[test]
fn the_jit_still_refuses_integer_elementwise() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[i32, [2]]
    let !b = forge.zeros[i32, [2]]
    let c = a .+ b
    print(c[0] as i64)
    nil
}
"#;
    let out = run_src("jitrefuses", "jit", src);
    let all = format!("{}{}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(!out.status.success(), "[jit] unexpectedly compiled an int elementwise op: {all}");
    assert!(all.contains("elementwise ops are f32-only"),
            "[jit] refusal changed shape: {all}");
}

/// Every dotted arithmetic op at `i32`, at the boundary each one can cross.
/// `.*` is the one that also pins PRECISION: (2^31-1)^2 is 4611686014132420609,
/// which f64 cannot represent, so the old float path answered ...608 — the
/// wrapped i32 answer is 1.
#[test]
fn every_dotted_arith_op_wraps_at_i32() {
    let src = r#"fn main() -> nil {
    let !hi = forge.zeros[i32, [1]]
    hi[0] = 2147483647
    let !lo = forge.zeros[i32, [1]]
    lo[0] = -2147483648
    let !one = forge.zeros[i32, [1]]
    one[0] = 1
    let !two = forge.zeros[i32, [1]]
    two[0] = 2
    let add = hi .+ one
    let sub = lo .- one
    let mul = hi .* hi
    let div = hi ./ two
    let pow = two .^ two
    print(add[0] as i64)
    print(sub[0] as i64)
    print(mul[0] as i64)
    print(div[0] as i64)
    print(pow[0] as i64)
    nil
}
"#;
    assert_run_prints("i32ops", src, "-2147483648\n2147483647\n1\n1073741823\n4\n");
}

/// Every signed width, both boundaries, including the packed `int4`/`int8`.
#[test]
fn every_signed_width_wraps_elementwise_at_both_boundaries() {
    let src = r#"fn main() -> nil {
    let !a8 = forge.zeros[i8, [2]]
    a8[0] = 127     a8[1] = -128
    let !o8 = forge.zeros[i8, [2]]
    o8[0] = 1       o8[1] = 1
    print((a8 .+ o8)[0] as i64)
    print((a8 .- o8)[1] as i64)

    let !a16 = forge.zeros[i16, [2]]
    a16[0] = 32767  a16[1] = -32768
    let !o16 = forge.zeros[i16, [2]]
    o16[0] = 1      o16[1] = 1
    print((a16 .+ o16)[0] as i64)
    print((a16 .- o16)[1] as i64)

    let !p4 = forge.zeros[int4, [2]]
    p4[0] = 7       p4[1] = -8
    let !o4 = forge.zeros[int4, [2]]
    o4[0] = 1       o4[1] = 1
    print((p4 .+ o4)[0] as i64)
    print((p4 .- o4)[1] as i64)

    let !p8 = forge.zeros[int8, [2]]
    p8[0] = 127     p8[1] = -128
    let !q8 = forge.zeros[int8, [2]]
    q8[0] = 1       q8[1] = 1
    print((p8 .+ q8)[0] as i64)
    print((p8 .- q8)[1] as i64)
    nil
}
"#;
    assert_run_prints("signedwidths", src,
        "-128\n127\n-32768\n32767\n-8\n7\n-128\n127\n");
}

/// Unsigned wraps UNSIGNED (§3.1): `255u8 .+ 1` is 0, `0u8 .- 1` is 255 — never
/// -1, at any of the three unsigned widths the interpreter models.
#[test]
fn every_unsigned_width_wraps_unsigned_elementwise() {
    let src = r#"fn main() -> nil {
    let !a8 = forge.zeros[u8, [2]]
    a8[0] = 255     a8[1] = 0
    let !o8 = forge.zeros[u8, [2]]
    o8[0] = 1       o8[1] = 1
    print((a8 .+ o8)[0] as i64)
    print((a8 .- o8)[1] as i64)

    let !a16 = forge.zeros[u16, [2]]
    a16[0] = 65535  a16[1] = 0
    let !o16 = forge.zeros[u16, [2]]
    o16[0] = 1      o16[1] = 1
    print((a16 .+ o16)[0] as i64)
    print((a16 .- o16)[1] as i64)

    let !a32 = forge.zeros[u32, [2]]
    a32[0] = 4294967295   a32[1] = 0
    let !o32 = forge.zeros[u32, [2]]
    o32[0] = 1            o32[1] = 1
    print((a32 .+ o32)[0] as i64)
    print((a32 .- o32)[1] as i64)
    nil
}
"#;
    assert_run_prints("unsignedwidths", src, "0\n255\n0\n65535\n0\n4294967295\n");
}

/// `./` on integer elements is INTEGER division at the element width, with
/// §3.1's two defined edge cases: division by zero is 0 (never ∞), and unsigned
/// division stays unsigned (4294967295 / 2 is 2147483647, not -1 / 2).
#[test]
fn elementwise_division_follows_the_integer_rule() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[i32, [3]]
    a[0] = 7        a[1] = -7       a[2] = 5
    let !b = forge.zeros[i32, [3]]
    b[0] = 2        b[1] = 2        b[2] = 0
    let d = a ./ b
    print(d[0] as i64)
    print(d[1] as i64)
    print(d[2] as i64)

    let !u = forge.zeros[u32, [1]]
    u[0] = 4294967295
    let !t = forge.zeros[u32, [1]]
    t[0] = 2
    print((u ./ t)[0] as i64)
    nil
}
"#;
    assert_run_prints("intdiv", src, "3\n-3\n0\n2147483647\n");
}

/// The one integer case with no wrapped value: `MIN / -1` is a runtime error
/// naming the width that overflowed — the same error, from the same code, as
/// the scalar `/` (§3.1).
#[test]
fn elementwise_min_over_minus_one_names_the_width() {
    let src = r#"fn main() -> nil {
    let !m = forge.zeros[i32, [1]]
    m[0] = -2147483648
    let !n = forge.zeros[i32, [1]]
    n[0] = -1
    print((m ./ n)[0] as i64)
    nil
}
"#;
    assert_run_fails("mindiv", src, "exceeds the i32 range");
}

/// A scalar operand adopts the tensor's element width, the way an untyped
/// literal adopts the other operand's width in scalar arithmetic (§3.1).
#[test]
fn a_scalar_operand_wraps_at_the_tensor_element_width() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[i32, [2]]
    a[0] = 2147483647
    a[1] = -2147483648
    let up = a .+ 1
    let dn = a .- 1
    print(up[0] as i64)
    print(dn[1] as i64)
    nil
}
"#;
    assert_run_prints("scalarrhs", src, "-2147483648\n2147483647\n");
}

/// Broadcasting does not lose the width: a [1] row against a [2, 1] column
/// wraps in every lane the broadcast produces.
#[test]
fn a_broadcast_elementwise_op_keeps_the_width() {
    let src = r#"fn main() -> nil {
    let !m = forge.zeros[i32, [2, 2]]
    m[0, 0] = 2147483647   m[0, 1] = 2147483647
    m[1, 0] = 5            m[1, 1] = 5
    let !r = forge.zeros[i32, [2]]
    r[0] = 1               r[1] = 2
    let s = m .+ r
    print(s[0, 0] as i64)
    print(s[0, 1] as i64)
    print(s[1, 1] as i64)
    nil
}
"#;
    assert_run_prints("broadcast", src, "-2147483648\n-2147483647\n7\n");
}

/// The width survives the round trip through a tensor: the elementwise result
/// is stored, read back, and compared as an `i32`, so a later `<` sees the
/// wrapped (negative) value rather than the mathematical one.
#[test]
fn the_wrapped_element_is_what_is_stored_and_compared() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[i32, [1]]
    a[0] = 2147483647
    let c = a .+ a
    let !dst = forge.zeros[i32, [1]]
    dst[0] = c[0]
    let seen: i32 = dst[0]
    print((seen as i64))
    print(seen < 0)
    nil
}
"#;
    assert_run_prints("roundtrip", src, "-2\ntrue\n");
}

/// An element LOAD carries the element width too — the divergence that made
/// this reachable under both backends. The JIT loads an `i32` lane as an `I32`,
/// so `t[0] + 1` wrapped there and not in the interpreter; now both wrap.
#[test]
fn an_element_load_carries_its_width_on_both_backends() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[i32, [1]]
    a[0] = 2147483647
    let v = a[0]
    print((v + 1) as i64)
    nil
}
"#;
    assert_both_print("loadwidth", src, "-2147483648\n");
}

/// The control, on both backends: the rule is about the DECLARED element
/// width, not about the magnitude 2^31. An `i64` element has room for the sum
/// and does not wrap — and `.+` on an `i64` tensor is still exact.
#[test]
fn an_i64_element_does_not_wrap_at_32_bits() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[i64, [1]]
    t[0] = 2147483647
    let v = t[0]
    print((v + 1) as i64)
    nil
}
"#;
    assert_both_print("i64control", src, "2147483648\n");
}

/// The i64-width elementwise control: the same `.+` that wraps at 32 bits does
/// not wrap here, because the element type is what decides.
///
/// The representational limit stays where it was: the interpreter's tensor
/// lanes are f64, so an i64 RESULT above 2^53 is rounded on the way back into
/// the lane (`2147483647 .* 2147483647` reads back ...608, not ...609). #571 is
/// about the width the arithmetic is performed at, which is now the element's;
/// widening the lanes is a different, larger change.
#[test]
fn an_i64_element_tensor_does_not_wrap_narrow() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[i64, [1]]
    t[0] = 2147483647
    let up = t .+ t
    print(up[0] as i64)
    nil
}
"#;
    assert_run_prints("i64add", src, "4294967294\n");
}

/// A cast that names an integer element type carries that width into the
/// result, so the elementwise op after it wraps at the cast's width and not at
/// the source's.
#[test]
fn a_tensor_cast_names_the_element_width() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[i64, [1]]
    t[0] = 127
    let narrow = t as i8
    let bumped = narrow .+ 1
    print(bumped[0] as i64)
    nil
}
"#;
    assert_run_prints("castwidth", src, "-128\n");
}
