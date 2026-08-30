//! #529: a leading `::` in an index element (start AND stop both omitted,
//! step present) never reached the start-omitted colon-slice branch in
//! `parse_index_elem_or_arg` — it matched on `TokenKind::Colon` but not the
//! single `TokenKind::ColonColon` token, so `x[::-1]` died with "expected
//! expression, found ColonColon" under every command.
//!
//! Second symptom closed by the same fix: `IndexElem::Slice { start: None,
//! end: None, step: Some(_) }` was constructible only via the space-separated
//! `a[: :2]` spelling (`Colon`, `Colon` as two tokens); `dmc fmt` printed it
//! back as `a[::2]`, which then failed to re-parse. These tests cover both.

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

fn run_mode(name: &str, mode: &str, src: &str) -> Output {
    let bin = dmc_binary();
    let tmp = std::env::temp_dir().join(format!("dmc_leading_cc_529_{}.dmc", name));
    std::fs::write(&tmp, src).expect("write temp probe");
    let out = Command::new(&bin)
        .args([mode, tmp.to_str().unwrap()])
        .output()
        .expect("invoke dmc");
    let _ = std::fs::remove_file(&tmp);
    out
}

/// Strided slices (any `a:b:c` form, with or without an omitted start) are
/// not JIT-lowered at all yet — `jit.rs` refuses them with a dedicated
/// "need slice-5 support" message, pre-existing and independent of #529
/// (confirmed below: the already-working `x[3::-1]` hits the identical
/// refusal). So only the interpreter (`run`) is exercised for numeric
/// correctness here; `dmc --check` covers the parse side on both backends'
/// shared front end, and a separate test pins the JIT's refusal message so
/// a future JIT slice-5 implementation is expected to change it.
fn assert_run_prints(name: &str, src: &str, want: &str) {
    let out = run_mode(name, "run", src);
    let so = String::from_utf8_lossy(&out.stdout).to_string();
    let se = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "[run {}] exited {:?}: {}", name, out.status.code(), se);
    assert_eq!(so, want, "[run {}] stdout (stderr: {})", name, se);
}

const REPRO: &str = r#"fn main() -> nil {
    let !x = forge.zeros[f32, [4]]
    for i in 0..4 { x[i] = (i * 10) as f32 }
    let r = x[::-1]
    print(r[0])
    print(r[1])
    print(r[2])
    print(r[3])
    nil
}
"#;

/// The issue's exact repro: `x[::-1]` on `[0, 10, 20, 30]` is the full-axis
/// reverse, `[30, 20, 10, 0]`. Must parse (both backends share one front
/// end) and give the right answer under the interpreter.
#[test]
fn issue_529_leading_colon_colon_reverses_full_axis() {
    assert_run_prints("repro", REPRO, "30\n20\n10\n0\n");
}

/// `--check` alone (the exact command the issue's repro was run through)
/// must succeed — no diagnostics, exit 0.
#[test]
fn issue_529_check_accepts_leading_colon_colon() {
    let out = run_mode("check", "--check", REPRO);
    let se = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "--check failed: {}", se);
}

/// `x[3::-1]` (start present) already worked before #529; confirm the fix
/// didn't disturb it, and cross-check against the new start-omitted form on
/// the same data — both must reverse the full axis identically.
#[test]
fn issue_529_start_present_negative_stride_unaffected() {
    let src = r#"fn main() -> nil {
    let !x = forge.zeros[f32, [4]]
    for i in 0..4 { x[i] = (i * 10) as f32 }
    let r = x[3::-1]
    print(r[0])
    print(r[1])
    print(r[2])
    print(r[3])
    nil
}
"#;
    assert_run_prints("start_present", src, "30\n20\n10\n0\n");
}

/// The JIT does not lower ANY strided slice yet (`jit.rs`: "strided slices
/// (`a:b:c`) need slice-5 support") — confirmed pre-existing and orthogonal
/// to #529 by checking it fires identically for the already-working
/// start-present form and the newly-parseable start-omitted form. Both get
/// PAST parsing (the #529 fix) and fail at JIT lowering with the same
/// message, not a parse error — proof the parser fix didn't paper over a
/// still-broken JIT path, and that the JIT gap isn't new.
#[test]
fn issue_529_jit_strided_slice_gap_is_preexisting_not_a_parse_error() {
    let leading = run_mode("jit_leading", "jit", REPRO);
    let start_present = run_mode(
        "jit_start_present",
        "jit",
        r#"fn main() -> nil {
    let !x = forge.zeros[f32, [4]]
    let r = x[3::-1]
    print(r[0])
    nil
}
"#,
    );
    for (label, out) in [("x[::-1]", &leading), ("x[3::-1]", &start_present)] {
        assert!(!out.status.success(), "{}: expected jit to refuse, it didn't", label);
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(
            se.contains("need slice-5 support"),
            "{}: expected the pre-existing slice-5 refusal, got: {}",
            label, se
        );
        assert!(
            !se.contains("expected expression"),
            "{}: jit hit a PARSE error, not the slice-5 refusal: {}",
            label, se
        );
    }
}

/// Second symptom: `a[: :2]` (two separate `Colon` tokens, space-separated)
/// lexes to the same AST shape `dmc fmt` emits as `a[::2]`. Round-trip:
/// format the space-separated spelling, then re-parse and run the formatted
/// output — it must still work, and must literally print `x[::2]` in the
/// formatted source.
#[test]
fn issue_529_fmt_round_trips_space_separated_colon_colon_shape() {
    let src = r#"fn main() -> nil {
    let !x = forge.zeros[f32, [4]]
    for i in 0..4 { x[i] = (i * 10) as f32 }
    let r = x[: :2]
    print(r[0])
    print(r[1])
    nil
}
"#;
    let fmt_out = run_mode("fmt_src", "fmt", src);
    assert!(
        fmt_out.status.success(),
        "dmc fmt failed on space-separated `: :2`: {}",
        String::from_utf8_lossy(&fmt_out.stderr)
    );
    let formatted = String::from_utf8_lossy(&fmt_out.stdout).to_string();
    assert!(
        formatted.contains("x[::2]"),
        "expected fmt to print the canonical `x[::2]`, got:\n{}",
        formatted
    );

    // Re-parse and run the FORMATTED source — this is the half that was
    // broken pre-#529 (fmt produced `a[::2]`, but the parser couldn't read
    // it back).
    let check = run_mode("fmt_reparse_check", "--check", &formatted);
    assert!(
        check.status.success(),
        "reformatted source failed --check: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    assert_run_prints("fmt_reparse_run", &formatted, "0\n20\n");

    // A second fmt pass over the already-canonical source must be a no-op
    // (idempotence) — this is what "round-trips" means.
    let fmt_again = run_mode("fmt_twice", "fmt", &formatted);
    assert!(fmt_again.status.success());
    let formatted_again = String::from_utf8_lossy(&fmt_again.stdout).to_string();
    assert_eq!(formatted, formatted_again, "fmt is not idempotent on `x[::2]`");
}

/// `a[::]` (start, stop, AND step all omitted) stays a parse error — the
/// `::` forms require a step expression right after the token, same as the
/// pre-existing `a[0::]`. #529 only adds the start-omitted case with a step
/// present; it does not make bare `::` a spelling of `:`.
#[test]
fn issue_529_bare_double_colon_with_no_step_still_errors() {
    let src = r#"fn main() -> nil {
    let !x = forge.zeros[f32, [4]]
    let r = x[::]
    print(r[0])
    nil
}
"#;
    let out = run_mode("bare_cc", "--check", src);
    assert!(!out.status.success(), "expected `x[::]` to still be a parse error");
    let se = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(se.contains("parse error"), "unexpected stderr: {}", se);
}
