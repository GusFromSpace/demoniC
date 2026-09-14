//! #597: `u64` division and comparison were signed, and a 64-bit hex/binary
//! mask with a `u64` suffix did not lex when bound to a declared `u64`
//! annotation.
//!
//! Two separate defects, both rooted in the same representational gap: the
//! interpreter's `IW` and the JIT's `ScalarKind` both mapped `u64` onto the
//! same tag as `i64` (there being no wider machine word to hold the top half
//! of the unsigned range), so a `u64` value at or above 2^63 — a negative
//! `i64` bit pattern — divided, shifted and compared as the signed pattern
//! instead of the unsigned magnitude, and a checker helper that folded a
//! literal's value down to that same signed pattern rejected any mask whose
//! top bit was set.
//!
//! The fix gives `u64` its own tag on both backends (`interp::IW::U64`,
//! `jit::ScalarKind::U64`) — still one i64-backed register, since there is no
//! wider one, but the tag now tells `/ % >>` and the ordering comparisons to
//! read it unsigned. `+ - * << & | ^` are untouched: their result is the same
//! bits whether read signed or unsigned, which is why #544's `u64` coverage of
//! those already worked.
//!
//! Modelled on `tests/narrow_int_wrap_544.rs`: these drive the real `dmc`
//! binary under both backends, because a backend divergence is exactly the
//! class of bug an in-process test cannot see. The checker-level regression
//! (a declared `u64` annotation, `check_int_literal_range`'s own bug) lives in
//! `src/check_tests.rs::u64_mask_literal_with_declared_annotation_597`.

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
    let tmp = std::env::temp_dir().join(format!("dmc_u64_597_{}_{}.dmc", name, mode));
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
/// what makes this a rule test rather than a tautology.
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
/// Also exercises a check/lex-time failure: both `run` and `jit` go through
/// the same lexer/checker before either backend ever executes anything.
fn assert_both_fail(name: &str, src: &str, needle: &str) {
    for mode in ["run", "jit"] {
        let out = run_src(name, mode, src);
        let all = format!("{}{}",
            String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(!out.status.success(), "[{mode} {name}] unexpectedly succeeded: {all}");
        assert!(all.contains(needle), "[{mode} {name}] wanted {needle:?}, got: {all}");
    }
}

/// The issue's first repro: a 64-bit hex mask with a `u64` suffix, bound to a
/// declared `u64` local, must lex and check clean and print its unsigned
/// magnitude — not the `-1` (or other negative) signed reinterpretation of the
/// same bit pattern. Binary masks are the same lexical path (`lexer.rs`'s `0b`
/// arm mirrors its `0x` arm), and demoniC has no octal literal, so hex,
/// binary and DECIMAL are the three spellings a `u64` value has.
///
/// Decimal was the one that didn't work: the `0x`/`0b` arms have parsed a
/// literal as a 64-bit bit pattern since #282, but the decimal arm parsed into
/// an `i64`, so `18446744073709551615u64` — the spelling anyone reaches for
/// first — died at LEX time with "integer literal out of range" while its hex
/// twin compiled. The lexer now takes the unsigned range for a decimal literal
/// carrying an explicit `u64` suffix.
#[test]
fn all_ones_and_top_bit_masks_lex_and_print_unsigned() {
    let src = r#"fn main() -> nil {
    let a: u64 = 0xffffffffffffffffu64
    let b: u64 = 0x8000000000000000u64
    let c: u64 = 0b1111111111111111111111111111111111111111111111111111111111111111u64
    print(a)
    print(b)
    print(c)
    nil
}
"#;
    assert_both_print("masks", src,
        "18446744073709551615\n9223372036854775808\n18446744073709551615\n");
}

/// The decimal spelling of the same three values, which #597's first round
/// left rejected at lex time. Written out in full, with the digit-group
/// underscores the grammar allows, because the whole point is that the exact
/// text a person would write now compiles.
#[test]
fn decimal_u64_literals_above_i64_max_lex_and_print_unsigned() {
    let src = r#"fn main() -> nil {
    let a: u64 = 18446744073709551615u64
    let b: u64 = 9223372036854775808u64
    let c: u64 = 18_446_744_073_709_551_615u64
    print(a)
    print(b)
    print(c)
    print(a == 0xffffffffffffffffu64)
    nil
}
"#;
    assert_both_print("decimal_masks", src,
        "18446744073709551615\n9223372036854775808\n18446744073709551615\ntrue\n");
}

/// The suffix is what unlocks the unsigned parse, and it has to stay that way:
/// the lexer cannot see the binding's annotation, and `check_int_literal_range`
/// treats any BARE literal at a 64-bit target as a bit pattern, so widening
/// unsuffixed decimals would silently turn `let x: i64 = 18446744073709551615`
/// into `-1` instead of the error it is. Both halves pinned here — the
/// unsuffixed decimal still fails, and a `u64`-suffixed one at an `i64`
/// annotation is a type error, not a silent reinterpretation.
#[test]
fn the_u64_suffix_is_what_widens_the_decimal_parse() {
    assert_both_fail("unsuffixed_over_i64", r#"fn main() -> nil {
    let x: i64 = 18446744073709551615
    print(x)
    nil
}
"#, "integer literal out of range");
    assert_both_fail("u64_suffix_at_i64_annotation", r#"fn main() -> nil {
    let x: i64 = 18446744073709551615u64
    print(x)
    nil
}
"#, "let binding has type i64 but value has type u64");
}

/// `dmc fmt` has to reproduce a program that still compiles. The formatter
/// printed an `IntLit` as a signed `i64`, so every `u64` mask — hex, binary,
/// and now decimal — came back out as `-1u64`, which the checker REJECTS as an
/// explicit negative for an unsigned type. Formatting is idempotent here and
/// the formatted text is accepted.
#[test]
fn fmt_round_trips_a_u64_mask_literal() {
    let src = r#"fn main() -> nil {
    let a: u64 = 0xffffffffffffffffu64
    let b: u64 = 18446744073709551615u64
    print(a + b)
    nil
}
"#;
    let out = run_src("fmt", "fmt", src);
    assert!(out.status.success(), "fmt exited {:?}", out.status.code());
    let formatted = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        formatted.contains("18446744073709551615u64"),
        "fmt should print the unsigned magnitude, got:\n{formatted}"
    );
    assert!(
        !formatted.contains("-1u64"),
        "fmt must not emit a negative literal for an unsigned type:\n{formatted}"
    );
    // And the formatted text is itself a valid program (fmt is idempotent).
    let again = run_src("fmt_again", "fmt", &formatted);
    assert!(again.status.success(), "re-formatting failed: {:?}", again.status.code());
    assert_eq!(String::from_utf8_lossy(&again.stdout), formatted, "fmt is not idempotent");
}

/// The issue's second repro, `/`: an all-ones `u64` divided by 2 is the top of
/// the range shifted down by one bit (u64::MAX / 2), not 0 (what `-1 / 2`
/// gives as a signed i64 division). `%` at the same width, alongside it.
#[test]
fn division_and_modulo_are_unsigned() {
    let src = r#"fn main() -> nil {
    let x: u64 = 0xffffffffffffffffu64
    let two: u64 = 2
    print(x / two)
    print(x % two)
    nil
}
"#;
    assert_both_print("div", src, "9223372036854775807\n1\n");
}

/// The issue's second repro, comparison: `(-1 as u64) > (1 as u64)` must be
/// true (u64::MAX is bigger than 1), across all four ordering operators —
/// signed comparison would get every one of these backwards.
#[test]
fn ordering_comparisons_are_unsigned() {
    let src = r#"fn main() -> nil {
    let x: u64 = 0xffffffffffffffffu64
    let one: u64 = 1
    print(x > one)
    print(x >= one)
    print(one < x)
    print(one <= x)
    print(x == 0xffffffffffffffffu64)
    print(x != one)
    nil
}
"#;
    assert_both_print("cmp", src, "true\ntrue\ntrue\ntrue\ntrue\ntrue\n");
}

/// `docs/OPERATORS.md §8a`: `>>` is arithmetic for a signed operand and
/// logical for an unsigned one. A top-bit-set `u64` shifted right by one must
/// zero-fill (0x4000... not 0xc000...); an `i64` alongside it must still
/// sign-extend, unaffected by the `u64` fix.
#[test]
fn shift_right_is_logical_for_u64_and_still_arithmetic_for_i64() {
    let src = r#"fn main() -> nil {
    let x: u64 = 0x8000000000000000u64
    let one: u64 = 1
    print(x >> one)
    let n: i64 = -8
    print(n >> 1)
    nil
}
"#;
    assert_both_print("shr", src, "4611686018427387904\n-4\n");
}

/// `/` and `%` by zero stay TOTAL (0, never a trap) at every width (#208) —
/// unsigned included, and unsigned has no `MIN / -1` case to guard at all
/// (there is no most-negative unsigned value), so a divisor that IS the
/// all-ones pattern must not spuriously trip that guard.
#[test]
fn division_edge_cases_hold_for_u64() {
    let zero = r#"fn main() -> nil {
    let x: u64 = 0xffffffffffffffffu64
    let z: u64 = 0
    print((x / z) + (x % z))
    nil
}
"#;
    assert_both_print("div_zero", zero, "0\n");
    let by_max = r#"fn main() -> nil {
    let max_pattern: u64 = 0u64 - 1u64
    let x: u64 = 10
    print(x / max_pattern)
    nil
}
"#;
    assert_both_print("div_by_max", by_max, "0\n");
}

/// `+ - * << & | ^` are unaffected by #597 — the result is the same bits
/// whichever way the pattern is read, which is why #544 already covered these
/// for `u64` (`unsigned_widths_wrap_unsigned_and_stay_unsigned`-adjacent, one
/// width up). Pinned here so a future change to the arithmetic paths can't
/// silently start treating them as needing the same unsigned-instruction
/// split `/ % >>` and the comparisons did.
#[test]
fn wrapping_ops_are_the_same_bits_either_way() {
    let src = r#"fn main() -> nil {
    let z: u64 = 0
    let o: u64 = 1
    print((z - o) as i64)
    let x: u64 = 0xffffffffffffffffu64
    print(x << o)
    let y: u64 = 0x00000000ffffffffu64
    print(y & 0x0000ffff0000ffffu64)
    nil
}
"#;
    assert_both_print("wrap_ops", src, "-1\n18446744073709551614\n65535\n");
}

/// A value outside the u64 range at all — one bit too many for even a mask —
/// is still rejected, in both the hex and the decimal spelling. This is the
/// LEXER's refusal ("integer literal out of range", no type named), which is
/// why it is asserted apart from the checker's below: matching only the shared
/// substring "out of range" would have let the checker path rot unnoticed.
#[test]
fn literals_wider_than_64_bits_still_fail_in_the_lexer() {
    assert_both_fail("over_range_hex", r#"fn main() -> nil {
    let x: u64 = 0x10000000000000000u64
    print(x)
    nil
}
"#, "hex literal out of range");
    assert_both_fail("over_range_decimal", r#"fn main() -> nil {
    let x: u64 = 18446744073709551616u64
    print(x)
    nil
}
"#, "integer literal out of range");
}

/// An explicit negative written for an unsigned type stays an error, and it is
/// the CHECKER that says so — `-1u64` is `Neg` applied to a literal, not a bare
/// literal, so it falls past `check_int_literal_range`'s bit-pattern carve-out
/// and gets range-checked against `0..=u64::MAX`. Asserted on the checker's own
/// wording (it names the type and the valid range) so this cannot silently
/// start being caught by the lexer instead — which would mean the carve-out
/// had swallowed the case.
#[test]
fn explicit_negative_literal_for_u64_is_a_checker_error() {
    assert_both_fail("neg_literal", r#"fn main() -> nil {
    let x: u64 = -1u64
    print(x)
    nil
}
"#, "integer literal -1 out of range for `u64` (valid range 0..=18446744073709551615)");
}

/// `u64 as f64` reads the UNSIGNED magnitude. Both backends converted with a
/// signed instruction (`fcvt_from_sint` in the JIT, `*n as f64` in the
/// interpreter's `as_float`), so `u64::MAX as f64` printed `-1` — the pattern,
/// not the value. The reverse direction already saturated unsigned
/// (`fcvt_to_uint_sat`) and is pinned alongside so the pair stays symmetric.
#[test]
fn u64_to_float_is_unsigned_on_both_backends() {
    let to_float = r#"fn main() -> nil {
    let x: u64 = 0xffffffffffffffffu64
    print(x as f64)
    let h: u64 = 0x8000000000000000u64
    print(h as f64)
    let s: u64 = 5
    print(s as f64)
    nil
}
"#;
    assert_both_print("u64_to_f64", to_float,
        "18446744073709552000\n9223372036854776000\n5\n");
    let from_float = r#"fn main() -> nil {
    let big: f64 = 1.0e30
    print(big as u64)
    let neg: f64 = -5.0
    print(neg as u64)
    nil
}
"#;
    assert_both_print("f64_to_u64", from_float, "18446744073709551615\n0\n");
    // `u64 as i64` is a REINTERPRETATION, not a value conversion: the pattern
    // is unchanged, so u64::MAX becomes -1. Pinned because the `as_float` fix
    // above would otherwise have routed this through f64 and saturated it to
    // i64::MAX instead.
    let reinterpret = r#"fn main() -> nil {
    let x: u64 = 0xffffffffffffffffu64
    print(x as i64)
    nil
}
"#;
    assert_both_print("u64_to_i64", reinterpret, "-1\n");
}

/// `abs` of a `u64` is the identity — an unsigned type has no negative values
/// — and it must stay an integer at that width. Both backends read the pattern
/// signed instead (`i64::abs` in the interpreter, `fabs` on a signed float
/// conversion in the JIT), so `abs(u64::MAX)` was 1 on both.
#[test]
fn abs_of_a_u64_is_the_identity_on_both_backends() {
    let src = r#"fn main() -> nil {
    let x: u64 = 0xffffffffffffffffu64
    print(abs(x))
    let h: u64 = 0x8000000000000000u64
    print(abs(h))
    let s: u64 = 7
    print(abs(s))
    nil
}
"#;
    assert_both_print("abs_u64", src,
        "18446744073709551615\n9223372036854775808\n7\n");
}

/// The interpreter's scalar `min`/`max`/`clamp` compared through an f64 view of
/// their operands, which read a `u64` pattern at or above 2^63 as a negative
/// number: `min(u64::MAX, 1)` was -1, `max` was 1, `clamp(u64::MAX, 0, 10)` was
/// 0 — every one of them backwards. An all-integer call with a `u64` operand
/// now selects in the unsigned domain and never touches f64 (u64::MAX is not
/// exactly f64-representable either). Interpreter-only: the JIT refuses the
/// scalar forms entirely, which the next test pins.
#[test]
fn scalar_min_max_clamp_read_u64_unsigned() {
    let src = r#"fn main() -> nil {
    let x: u64 = 0xffffffffffffffffu64
    let one: u64 = 1
    let lo: u64 = 0
    let hi: u64 = 10
    print(min(x, one))
    print(max(x, one))
    print(clamp(x, lo, hi))
    print(clamp(one, lo, hi))
    print(min(x, x))
    nil
}
"#;
    let out = run_src("minmax", "run", src);
    let se = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "[run minmax] exited {:?}: {se}", out.status.code());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1\n18446744073709551615\n10\n1\n18446744073709551615\n",
        "[run minmax] stdout (stderr: {se})",
    );
}

/// The JIT does not lower the scalar/variadic `min`/`max` at all, and has no
/// `clamp`. That REFUSAL is the correct behaviour — a clean "use `dmc run`"
/// diagnostic, not a wrong answer — and it is what keeps the unsigned fix above
/// interpreter-only without the backends diverging. Pinned so a later JIT
/// implementation cannot land without also getting the `u64` reading right.
#[test]
fn the_jit_still_refuses_the_scalar_min_max_and_clamp_forms() {
    for (name, call, needle) in [
        ("jit_min", "min(x, one)", "scalar/variadic `min`"),
        ("jit_max", "max(x, one)", "scalar/variadic `max`"),
        ("jit_clamp", "clamp(x, one, one)", "unknown function `clamp`"),
    ] {
        let src = format!(r#"fn main() -> nil {{
    let x: u64 = 0xffffffffffffffffu64
    let one: u64 = 1
    print({call})
    nil
}}
"#);
        let out = run_src(name, "jit", &src);
        let all = format!("{}{}",
            String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(!out.status.success(), "[{name}] unexpectedly succeeded: {all}");
        assert!(all.contains(needle), "[{name}] wanted {needle:?}, got: {all}");
    }
}

// ─── Round 3: the untyped-literal adoption path, and the two casts out ───────

/// **The regression this round exists for.** SPEC §3.1: an untyped integer
/// literal adopts the width of the other operand. In the JIT that adoption is
/// `adopt_int_literal_kind`, which gated on "`I32`, or a masked narrow kind".
/// `u64` used to arrive there tagged `I64` — the two shared a tag — so
/// `a + 1` at a `u64` never needed adopting: both sides were already `I64`.
/// Giving `u64` its own `ScalarKind::U64` (round 1) made it a kind that is
/// neither `I32` nor `is_masked_int`, and it fell out of the path entirely:
/// EVERY binary operator with an untyped literal on either side started
/// hard-erroring under `dmc jit` with "operands disagree: `u64` vs `i64`",
/// while `dmc run` computed it and every other width — `u8` through `i64` —
/// lowered fine.
///
/// One test per operator would hide which ones regressed, so this walks the
/// whole operator set with the literal on the LEFT and on the RIGHT, and
/// asserts the value, not just that the backends agree.
#[test]
fn an_untyped_literal_adopts_u64_from_the_other_operand() {
    for (name, expr, want) in [
        ("add_r", "a + 1", "11"),
        ("add_l", "1 + a", "11"),
        ("sub_r", "a - 1", "9"),
        ("sub_l", "12 - a", "2"),
        ("mul_r", "a * 2", "20"),
        ("mul_l", "2 * a", "20"),
        ("and_r", "a & 1", "0"),
        ("and_l", "1 & a", "0"),
        ("or_r",  "a | 1", "11"),
        ("or_l",  "1 | a", "11"),
        ("xor_r", "a ^ 1", "11"),
        ("xor_l", "1 ^ a", "11"),
        ("shl_r", "a << 1", "20"),
        ("shl_l", "1 << a", "1024"),
        ("shr_r", "a >> 1", "5"),
        ("shr_l", "1024 >> a", "1"),
        ("div_r", "a / 2", "5"),
        ("div_l", "100 / a", "10"),
        ("rem_r", "a % 3", "1"),
        ("rem_l", "23 % a", "3"),
        ("eq",    "a == 1", "false"),
        ("eq_l",  "10 == a", "true"),
        ("ne",    "a != 1", "true"),
        ("gt",    "a > 1", "true"),
        ("gt_l",  "1 > a", "false"),
        ("lt",    "a < 1", "false"),
        ("lt_l",  "1 < a", "true"),
        ("ge",    "a >= 1", "true"),
        ("le",    "a <= 1", "false"),
    ] {
        let src = format!("fn main() -> nil {{\n    let a: u64 = 10u64\n    print({expr})\n    nil\n}}\n");
        assert_both_print(&format!("adopt_{name}"), &src, &format!("{want}\n"));
    }
}

/// Adoption has to carry the UNSIGNED reading with it, not just stop erroring:
/// a literal meeting a `u64` whose top bit is set must divide, shift and order
/// against the magnitude. The literal side of `adopt_int_literal_kind` is also
/// the one place a 64-bit MASK can appear as the untyped operand — `0xff..ff`
/// is a bit pattern per #282, so its `i64` reading is negative, and the range
/// check the other widths run (`0 ..= (1 << w) - 1`) both overflows at 64 bits
/// and would reject it. `u64` takes its own arm: the whole slot is in range.
#[test]
fn untyped_literal_adoption_at_u64_stays_unsigned() {
    for (name, expr, want) in [
        ("div", "m / 2", "9223372036854775807"),
        ("rem", "m % 10", "5"),
        ("shr", "m >> 1", "9223372036854775807"),
        ("gt", "m > 1", "true"),
        ("lt_l", "1 < m", "true"),
        ("mask_add", "m + 0xffffffffffffffff", "18446744073709551614"),
        ("mask_cmp", "m > 0x8000000000000000", "true"),
        ("mask_div", "m / 0xffffffffffffffffu64", "1"),
        ("decimal_add", "m + 18446744073709551615u64", "18446744073709551614"),
    ] {
        let src = format!(
            "fn main() -> nil {{\n    let m: u64 = 0xffffffffffffffffu64\n    print({expr})\n    nil\n}}\n");
        assert_both_print(&format!("adopt_max_{name}"), &src, &format!("{want}\n"));
    }
}

/// A suffix still types the literal concretely (§3.1), so a suffix naming a
/// DIFFERENT width keeps the refusal rather than being silently retagged —
/// the same rule every other adopting width follows. `u64` must not have
/// bought its adoption by loosening that.
#[test]
fn a_conflicting_suffix_does_not_adopt_u64() {
    let src = r#"fn main() -> nil {
    let a: u64 = 10u64
    print(a + 1i32)
    nil
}
"#;
    let out = run_src("adopt_conflict", "jit", src);
    let all = format!("{}{}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(!out.status.success(), "[adopt_conflict] unexpectedly succeeded: {all}");
}

/// The other half of the same regression, in `coerce_scalar`: `u64 as i32` and
/// `u64 as bool` lowered through the `i64` arms while the two kinds shared a
/// tag, and splitting `U64` off dropped both into the JIT's "outside the cast
/// set" refusal — a `dmc jit` failure on a program `dmc run` (and main)
/// computes. Neither reads the sign: `as i32` keeps the low 32 bits (§3.1's
/// two's-complement narrowing) and `as bool` tests the pattern against zero.
#[test]
fn u64_casts_to_i32_and_bool_lower_on_both_backends() {
    let src = r#"fn main() -> nil {
    let m: u64 = 0xffffffffffffffffu64
    let h: u64 = 0x8000000000000000u64
    let z: u64 = 0
    print(m as i32)
    print(h as i32)
    print(m as bool)
    print(h as bool)
    print(z as bool)
    nil
}
"#;
    assert_both_print("cast_i32_bool", src, "-1\n0\ntrue\ntrue\nfalse\n");
}

/// `print_u64` is the JIT's runtime formatter for a `u64`, reached only by
/// `print(x)`'s internal dispatch — it is deliberately NOT a builtin a program
/// may call, so this bug fix adds no undocumented name to STDLIB.md's builtin
/// list. Pinned in both directions: the name is undefined, and `print` on a
/// `u64` still renders the unsigned magnitude on both backends.
#[test]
fn print_u64_is_internal_and_print_still_renders_u64_unsigned() {
    assert_both_fail("print_u64_internal", r#"fn main() -> nil {
    let a: u64 = 7
    print_u64(a)
    nil
}
"#, "undefined identifier `print_u64`");

    assert_both_print("print_u64_via_print", r#"fn main() -> nil {
    let z: u64 = 0
    let o: u64 = 1
    print(z - o)
    nil
}
"#, "18446744073709551615\n");
}

/// `dmc fmt` printed an `IntLit` in decimal, and `i64::MIN` has no decimal
/// spelling the LEXER accepts: `-9223372036854775808` scans as a unary minus
/// applied to `9223372036854775808`, which is out of `i64` range. The only way
/// `i64::MIN` gets into an `IntLit` at all is #282's 64-bit mask rule
/// (`0x8000000000000000`), which this branch blesses for `u64`, so the
/// formatter now prints that mask back rather than a decimal that does not
/// re-lex. Asserted the way it matters: the formatted text COMPILES, and a
/// second format is a fixed point.
#[test]
fn fmt_keeps_i64_min_re_lexable() {
    let src = r#"fn main() -> nil {
    let x: i64 = 0x8000000000000000i64
    let y: i64 = 0x8000000000000000
    print(x)
    print(y)
    nil
}
"#;
    let out = run_src("fmt_i64_min", "fmt", src);
    assert!(out.status.success(), "fmt exited {:?}", out.status.code());
    let formatted = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        !formatted.contains("-9223372036854775808"),
        "fmt must not emit a decimal `i64::MIN`, which does not re-lex:\n{formatted}"
    );
    // The formatted program still compiles and computes the same value.
    assert_both_print("fmt_i64_min_runs", &formatted,
        "-9223372036854775808\n-9223372036854775808\n");
    // ... and formatting is a fixed point.
    let again = run_src("fmt_i64_min_again", "fmt", &formatted);
    assert!(again.status.success(), "re-formatting failed: {:?}", again.status.code());
    assert_eq!(String::from_utf8_lossy(&again.stdout), formatted, "fmt is not idempotent");
}
