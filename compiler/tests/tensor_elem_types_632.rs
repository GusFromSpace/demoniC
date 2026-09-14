//! #632: the JIT lowers a tensor element type at its OWN storage width.
//!
//! Before this, `Tensor[u8, …]` and every other byte-aligned integer element
//! type was refused as `jit-element-type` ("element type `u8` is not
//! lowered"), and the two element types that WERE lowered had a hidden hole:
//! `u32` was mapped to the signed `i32` kind, so `0u32 - 1` on a `u32` element
//! would have wrapped to -1 rather than 4294967295 the moment a constructor let
//! one be built.
//!
//! The fix splits two widths that had been one number. A kind's MACHINE width
//! is the register the value computes in (64 bits for every masked narrow
//! kind); its STORAGE width is the lane the element occupies in the arena (one
//! byte for `u8`). `elem_bytes`, the element load, the element store and every
//! copy loop now use the second. Conflating them is #552's bug one size class
//! down: a load at the machine width over a one-byte buffer reads its
//! neighbours, and a store at it overwrites them.
//!
//! Every expectation below is the literal number BOTH backends must print, and
//! the interpreter is the reference: its `int_type_name_width` table and this
//! file's `tensor_elem_kind` are the same table, which is what makes `u8`
//! wrap at 256 in the same place on both sides.

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
    let tmp = std::env::temp_dir().join(format!("dmc_elem_ty_{}_{}.dmc", name, mode));
    std::fs::write(&tmp, src).expect("write temp probe");
    let out = Command::new(dmc_binary())
        .args([mode, tmp.to_str().unwrap()])
        .output()
        .expect("invoke dmc");
    let _ = std::fs::remove_file(&tmp);
    out
}

/// Both backends run it, both exit 0, and both print exactly `want`.
fn both_print(name: &str, src: &str, want: &str) {
    for mode in ["run", "jit"] {
        let out = run_src(name, mode, src);
        let so = String::from_utf8_lossy(&out.stdout).to_string();
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(out.status.success(), "[{mode} {name}] exited {:?}: {se}", out.status.code());
        assert_eq!(so, want, "[{mode} {name}] stdout (stderr: {se})");
    }
}

/// The JIT declines to lower it, with `needle` in the diagnostic.
fn jit_refuses(name: &str, src: &str, needle: &str) {
    let out = run_src(name, "jit", src);
    let all = format!("{}{}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(!out.status.success(), "[jit {name}] unexpectedly compiled: {all}");
    assert!(all.contains(needle), "[jit {name}] wanted {needle:?}, got: {all}");
}

// ─── allocation ──────────────────────────────────────────────────────────────

/// `zeros` and `ones` fill at the element's own width. `ones` is the case the
/// old 32-bit fill could not express at all: a one-byte lane needs a one-byte
/// 1, not an f32 1.0 every fourth byte.
#[test]
fn zeros_and_ones_fill_every_lane_at_its_own_width() {
    let src = r#"fn main() -> nil {
    let a = forge.zeros[u8, [5]]
    let b = forge.ones[u8, [5]]
    for i in 0..5 { print(a[i]) }
    for i in 0..5 { print(b[i]) }
    nil
}
"#;
    both_print("fill_u8", src, "0\n0\n0\n0\n0\n1\n1\n1\n1\n1\n");
}

/// A length whose byte count is NOT a multiple of four — the case the wide
/// zero-fill cannot take, so it must fall to the element-wise one and must not
/// write past the last element.
#[test]
fn a_ragged_byte_length_zero_fills_without_overrunning() {
    let src = r#"fn main() -> nil {
    let !a = forge.ones[u8, [3]]
    let z = forge.zeros[u8, [3]]
    a[2] = 9
    for i in 0..3 { print(z[i]) }
    for i in 0..3 { print(a[i]) }
    nil
}
"#;
    both_print("ragged_u8", src, "0\n0\n0\n1\n1\n9\n");
}

/// The Vault allocates and fills the same way the Forge does. Read-only: a
/// write to a Vault tensor belongs in a `vault { … }` block, and the JIT does
/// not lower arena blocks yet — a separate gap, tracked on its own.
#[test]
fn the_vault_allocates_narrow_elements_too() {
    let src = r#"fn main() -> nil {
    let a = vault.ones[u32, [4]]
    let b = vault.zeros[u8, [3]]
    print(a[0])
    print(a[3])
    print(b[2])
    nil
}
"#;
    both_print("vault_narrow", src, "1\n1\n0\n");
}

/// `uninit` was already width-agnostic; keep it so, and pin that a value
/// written into an uninitialised lane reads back unchanged.
#[test]
fn uninit_lanes_round_trip_at_every_new_width() {
    let src = r#"fn main() -> nil {
    let !a = forge.uninit[u8, [2]]
    let !b = forge.uninit[u32, [2]]
    let !c = forge.uninit[u64, [2]]
    a[0] = 255      b[0] = 4294967295      c[0] = 9007199254740992
    print(a[0])     print(b[0])            print(c[0])
    nil
}
"#;
    both_print("uninit_widths", src, "255\n4294967295\n9007199254740992\n");
}

// ─── storage width ───────────────────────────────────────────────────────────

/// The lane really is one byte wide: writing element 0 of a `u8` tensor must
/// not disturb elements 1..3, which an 8-byte store at a 1-byte lane would
/// flatten. This is the test that fails loudest if `elem_bytes` ever goes back
/// to reporting the machine width.
#[test]
fn a_u8_write_touches_exactly_one_byte() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [8]]
    for i in 0..8 { a[i] = (i + 1) as u8 }
    a[0] = 200
    for i in 0..8 { print(a[i]) }
    nil
}
"#;
    both_print("u8_lane", src, "200\n2\n3\n4\n5\n6\n7\n8\n");
}

/// The same, one width up, with a value that fills all four bytes.
#[test]
fn a_u32_write_touches_exactly_four_bytes() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u32, [4]]
    a[0] = 4294967295
    a[1] = 7
    a[2] = 4294967295
    print(a[0])  print(a[1])  print(a[2])  print(a[3])
    nil
}
"#;
    both_print("u32_lane", src, "4294967295\n7\n4294967295\n0\n");
}

// ─── wrap at the element width ───────────────────────────────────────────────

/// A `u8` element read carries its width, so arithmetic on it wraps at 256 —
/// the headline case from the corpus (byte buffers).
#[test]
fn a_u8_element_wraps_at_256() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [2]]
    a[0] = 250
    a[1] = 10
    print(a[0] + a[1])
    print(a[0] * a[1])
    print(a[1] - a[0])
    nil
}
"#;
    both_print("u8_wrap", src, "4\n196\n16\n");
}

/// `u32` wraps UNSIGNED, which is the semantic the old `u32 -> i32` mapping
/// would have got wrong: `0u32 - 1` is 4294967295, never -1.
#[test]
fn a_u32_element_wraps_unsigned_not_signed() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u32, [2]]
    a[0] = 0
    a[1] = 4294967295
    print(a[0] - 1)
    print(a[1] + 1)
    print(a[1] / 2)
    nil
}
"#;
    both_print("u32_unsigned", src, "4294967295\n0\n2147483647\n");
}

/// A store wraps into the lane before it lands, so what is read back is the
/// wrapped value and not the value the program wrote down. 260 written into a
/// `u8` element is 4, on both backends, at the store.
#[test]
fn an_out_of_range_store_wraps_into_the_lane() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [2]]
    let !w = forge.zeros[i64, [1]]
    w[0] = 260
    a[0] = w[0] as u8
    print(a[0])
    nil
}
"#;
    both_print("u8_store_wrap", src, "4\n");
}

/// A `u64` element holds the full 64-bit pattern AND reads back unsigned.
///
/// This test used to expect `-1` in the middle slot, because `u64` shared the
/// signed `I64` kind on both backends and a value at or above 2^63 rendered as
/// its signed pattern (SPEC §3.1's old status note). #643 ended that: it gave
/// `u64` its own tag on both sides (`interp::IW::U64`, `jit::ScalarKind::U64`),
/// so the element is still one i64-backed slot — there is no wider machine
/// word — but the tag now says to read the pattern unsigned. The element type
/// table here follows, or a `Tensor[u64, …]` element would be the one `u64` in
/// the language still read as signed.
#[test]
fn a_u64_element_carries_the_full_64_bit_pattern() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u64, [3]]
    a[0] = 9007199254740991
    a[1] = 0
    a[1] = a[1] - 1
    a[2] = 4611686018427387904
    print(a[0])
    print(a[1])
    print(a[2])
    nil
}
"#;
    both_print("u64_pattern", src,
        "9007199254740991\n18446744073709551615\n4611686018427387904\n");
}

/// The #643 operators, applied to a `u64` TENSOR ELEMENT rather than a scalar.
///
/// #643 made `/ % >>` and the ordering comparisons read a `u64` unsigned, but
/// it could only reach scalars: the JIT refused to build a `Tensor[u64, …]` at
/// all, so the element path had no coverage and could have kept the signed
/// reading indefinitely without any test noticing. That is the exact shape of
/// the `u32`-mapped-to-`i32` hole this file exists to close, one width up.
///
/// Every expectation is a number that SEPARATES the two readings, so a signed
/// element load fails here rather than producing a plausible answer:
/// `0xffff_ffff_ffff_ffff / 2` is 9223372036854775807 unsigned and 0 signed
/// (-1 / 2 truncates to zero); `% 10` is 5 unsigned and -1 signed; `>> 4` is a
/// logical shift to 2^60-1 unsigned and stays -1 signed; and the ordering
/// comparison against a small value is `true` unsigned, `false` signed.
#[test]
fn u64_elements_above_2_63_divide_shift_and_compare_unsigned() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[u64, [2]]
    t[0] = 0xffffffffffffffffu64
    t[1] = 10u64
    let big: u64 = t[0]
    let small: u64 = t[1]
    print(big / 2u64)
    print(big % 10u64)
    print(big >> 4u64)
    print(big > small)
    print(big < small)
    nil
}
"#;
    both_print("u64_elem_unsigned_ops", src,
        "9223372036854775807\n5\n1152921504606846975\ntrue\nfalse\n");
}

/// The companion limit, pinned so it is a documented shared boundary rather
/// than an undiscovered divergence: `sum` over a `u64` tensor reduces through
/// an f64 accumulator on BOTH backends, and the interpreter's tensor backing
/// lane stores the element as its signed pattern (it must, or the element
/// could not round-trip — `-1.0` is exact in f64 while 1.8446744073709552e19
/// is not). So a `u64` element at or above 2^63 sums to a NEGATIVE f64 on both
/// backends.
///
/// That is three orders of magnitude outside f64's exact-integer range, which
/// is the range `NUMERICS.md §1.2` bounds backend parity by — below 2^63 the
/// signed and unsigned readings of a `u64` are the same number, which the
/// first two lines pin. Making the JIT's reduction convert unsigned instead
/// would be more faithful to `u64` in isolation and would DIVERGE from
/// `dmc run`, so `fused_sum_int` deliberately converts signed; see its
/// doc comment. The agreement is the property under test.
#[test]
fn a_u64_sum_agrees_on_both_backends_including_past_2_63() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u64, [3]]
    a[0] = 1000000u64
    a[1] = 2000000u64
    a[2] = 3000000u64
    print(sum(a))
    let !b = forge.zeros[u64, [1]]
    b[0] = 0x8000000000000000u64
    print(sum(b))
    nil
}
"#;
    both_print("u64_sum_parity", src, "6000000\n-9223372036854776000\n");
}

// ─── indexing, assignment, aliasing ──────────────────────────────────────────

/// Negative (Python-style) indices, at a narrow width, on both read and write.
#[test]
fn negative_indices_resolve_at_a_narrow_width() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [4]]
    a[-1] = 200
    a[-4] = 1
    print(a[3])
    print(a[0])
    print(a[-1])
    nil
}
"#;
    both_print("u8_negidx", src, "200\n1\n200\n");
}

/// A bound copy is a snapshot, not an alias — the copy loop has to walk the
/// source at the element's storage stride, or a `u8` copy reads eight lanes per
/// step and runs off the end.
#[test]
fn binding_a_narrow_tensor_copies_it() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [6]]
    for i in 0..6 { a[i] = (i * 40) as u8 }
    let b = a
    a[0] = 7
    for i in 0..6 { print(b[i]) }
    print(a[0])
    nil
}
"#;
    both_print("u8_copy", src, "0\n40\n80\n120\n160\n200\n7\n");
}

/// A row slice of a narrow 2-D tensor is a view at the right offset and width.
#[test]
fn a_row_slice_of_a_narrow_tensor_reads_the_right_lanes() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [3, 4]]
    for i in 0..3 { for j in 0..4 { a[i, j] = (i * 4 + j) as u8 } }
    let r = a[2, ..]
    print(len(r))
    for j in 0..4 { print(r[j]) }
    nil
}
"#;
    both_print("u8_slice", src, "4\n8\n9\n10\n11\n");
}

/// A slice STORE writes through to the source at the narrow stride.
#[test]
fn a_slice_store_writes_narrow_lanes_through() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [2, 3]]
    let !r = forge.ones[u8, [3]]
    r[1] = 200
    a[1, ..] = r
    for j in 0..3 { print(a[1, j]) }
    print(a[0, 0])
    nil
}
"#;
    both_print("u8_slice_store", src, "1\n200\n1\n0\n");
}

// ─── signatures, fields, templates ───────────────────────────────────────────

/// A narrow element type in a function SIGNATURE binds the argument the caller
/// built, which is only true if the annotation and the constructor agree about
/// the element kind — the drift the shared table exists to prevent. The sum
/// wraps because both element reads are `u8`: 200 + 100 is 44.
#[test]
fn a_narrow_element_type_binds_across_a_call() {
    let src = r#"fn total(x: Tensor[u8, [4]]) -> u8 { x[0] + x[3] }
fn main() -> nil {
    let !a = forge.zeros[u8, [4]]
    a[0] = 200
    a[3] = 100
    print(total(a))
    nil
}
"#;
    both_print("u8_param", src, "44\n");
}

/// A shape-parameterised signature binds the same way.
#[test]
fn a_narrow_element_binds_through_a_shape_parameter() {
    let src = r#"fn first[N](x: Tensor[u32, [N]]) -> u32 { x[0] }
fn main() -> nil {
    let !a = forge.zeros[u32, [3]]
    a[0] = 4294967295
    print(first(a))
    nil
}
"#;
    both_print("u32_shapeparam", src, "4294967295\n");
}

/// A narrow tensor as a model FIELD: the field's declared element type sizes
/// the same lanes the initialiser allocated.
#[test]
fn a_narrow_tensor_model_field_round_trips() {
    let src = r#"model Buf {
    !px: Tensor[u8, [4]]
}
fn main() -> nil {
    let !m = Buf { px: forge.zeros[u8, [4]] }
    m.px[1] = 250
    m.px[2] = 6
    print(m.px[1] + m.px[2])
    print(m.px[0])
    nil
}
"#;
    both_print("u8_field", src, "0\n0\n");
}

// ─── elementwise and reductions ──────────────────────────────────────────────

/// `.+ .- .* ./` over narrow integer elements is integer arithmetic at the
/// element width (`NUMERICS.md §1.3`), so the products wrap rather than
/// widening into a float lane.
#[test]
fn dotted_arithmetic_wraps_at_the_narrow_element_width() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [3]]
    let !b = forge.zeros[u8, [3]]
    a[0] = 200   a[1] = 250   a[2] = 7
    b[0] = 100   b[1] = 10    b[2] = 2
    let mul = a .* b
    let add = a .+ b
    let div = a ./ b
    print(mul[0])
    print(add[1])
    print(div[2])
    nil
}
"#;
    both_print("u8_dotted", src, "32\n4\n3\n");
}

/// An integer tensor literal is `i64`-typed; joined against a narrow tensor it
/// yields the NARROW width, exactly as `IW::join` does in the interpreter.
#[test]
fn a_literal_operand_adopts_the_narrow_element_width() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [3]]
    a[0] = 255   a[1] = 250   a[2] = 1
    let up = a .+ 1
    let dn = a .- 2
    print(up[0])
    print(up[1])
    print(dn[2])
    nil
}
"#;
    both_print("u8_literal_join", src, "0\n251\n255\n");
}

/// `elem_kind_join`'s comment used to claim the checker forbids `u8 OP i16` —
/// it does not. Two DIFFERENTLY-NAMED narrow integer kinds meeting is a real,
/// `--check`-clean program, and the fall-through picks the NARROWER width
/// regardless of which side it is on: `u8 .+ i16` and `i16 .+ u8` both wrap at
/// 8 bits. 200 + 100 is 300, which wraps to 44 at 8 bits either way.
#[test]
fn two_different_narrow_kinds_join_at_the_narrower_width_either_order() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [2]]
    let !b = forge.zeros[i16, [2]]
    a[0] = 200
    b[0] = 100
    print((a .+ b)[0])
    nil
}
"#;
    both_print("u8_plus_i16", src, "44\n");

    let src = r#"fn main() -> nil {
    let !a = forge.zeros[i16, [2]]
    let !b = forge.zeros[u8, [2]]
    a[0] = 100
    b[0] = 200
    print((a .+ b)[0])
    nil
}
"#;
    both_print("i16_plus_u8", src, "44\n");
}

/// Same rule at a wider width gap: `u8` is still narrower than `u32`, so the
/// join is `u8` and the result wraps at 8 bits no matter which operand is on
/// the left. 200 + 4000000100 wraps (mod 256) to 44 either way.
#[test]
fn u8_and_u32_join_at_u8_either_order() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [2]]
    let !b = forge.zeros[u32, [2]]
    a[0] = 200
    b[0] = 4000000100
    print((a .+ b)[0])
    nil
}
"#;
    both_print("u8_plus_u32", src, "44\n");

    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u32, [2]]
    let !b = forge.zeros[u8, [2]]
    a[0] = 4000000100
    b[0] = 200
    print((a .+ b)[0])
    nil
}
"#;
    both_print("u32_plus_u8", src, "44\n");
}

/// `u32` and `i32` are the SAME width (32 bits), so `elem_kind_join` cannot
/// pick the narrower one — it ties to the LEFT operand's kind, which is the
/// order-dependence #641's review caught: `u32 .+ i32` computes as `u32`
/// (unsigned, no wrap: 4000000000 + 100 fits) and `i32 .+ u32` computes as
/// `i32` (signed, wraps negative: the same bit pattern read as a negative
/// `i32`). Both are what the interpreter's `IW::join` does too.
#[test]
fn equal_width_narrow_kinds_join_to_the_left_operand_either_order() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u32, [2]]
    let !b = forge.zeros[i32, [2]]
    a[0] = 4000000000
    b[0] = 100
    print((a .+ b)[0])
    nil
}
"#;
    both_print("u32_plus_i32", src, "4000000100\n");

    let src = r#"fn main() -> nil {
    let !a = forge.zeros[i32, [2]]
    let !b = forge.zeros[u32, [2]]
    a[0] = 100
    b[0] = 4000000000
    print((a .+ b)[0])
    nil
}
"#;
    both_print("i32_plus_u32", src, "-294967196\n");
}

/// `sum` over an integer tensor accumulates in f64 and yields an f64 scalar —
/// it does NOT wrap at the element width, because the interpreter sums its f64
/// backing lanes and tags the result f64. 200+250+200 is 650, not 138.
#[test]
fn sum_over_narrow_elements_does_not_wrap() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u8, [3]]
    a[0] = 200   a[1] = 250   a[2] = 200
    print(sum(a))
    print(mean(a))
    nil
}
"#;
    both_print("u8_sum", src, "650\n216.66666666666666\n");
}

/// A `u32` sum stays inside f64's exact range for values a 32-bit lane can
/// hold, so the two backends agree bit for bit (`NUMERICS.md §1.2`'s bound).
#[test]
fn sum_over_u32_elements_stays_exact() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[u32, [3]]
    a[0] = 4294967295   a[1] = 4294967295   a[2] = 1
    print(sum(a))
    nil
}
"#;
    both_print("u32_sum", src, "8589934591\n");
}

/// `sum` over a bool tensor counts the set lanes.
#[test]
fn sum_over_a_bool_tensor_counts_true_lanes() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[bool, [4]]
    a[0] = true
    a[3] = true
    print(sum(a))
    nil
}
"#;
    both_print("bool_sum", src, "2\n");
}

// ─── bool ────────────────────────────────────────────────────────────────────

/// A bool tensor is one byte per lane, and `zeros`/`ones` now fill it.
#[test]
fn a_bool_tensor_allocates_indexes_and_assigns() {
    let src = r#"fn main() -> nil {
    let !a = forge.zeros[bool, [4]]
    let b = forge.ones[bool, [4]]
    a[2] = true
    print(a[0])
    print(a[2])
    print(b[3])
    nil
}
"#;
    both_print("bool_tensor", src, "false\ntrue\ntrue\n");
}

// ─── what is still refused ───────────────────────────────────────────────────

/// `int4` is the one integer element type with no byte-aligned lane, so it
/// stays refused rather than being silently widened to a byte — the refusal
/// keeps its `jit-element-type` class.
#[test]
fn the_sub_byte_int4_element_is_still_refused() {
    jit_refuses("int4_refused",
        "fn main() -> nil { let t = forge.zeros[int4, [2]]  print(t[0])  nil }",
        "element type `int4` is not lowered");
}

/// The same refusal from the other direction: `int4` in a SIGNATURE, where the
/// element kind comes from `ty_from_ast` rather than the constructor. Both read
/// the same table, so both decline the same set.
#[test]
fn a_sub_byte_element_in_a_signature_is_still_refused() {
    jit_refuses("int4_sig_refused",
        "fn f(x: Tensor[int4, [2]]) -> i64 { x[0] as i64 }\nfn main() -> nil { print(0)  nil }",
        "is not lowered");
}
