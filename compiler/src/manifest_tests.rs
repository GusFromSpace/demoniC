/// `demoni.json` reader tests (#463) — the dial extraction and the
/// nearest-manifest resolution walk. The reader must never invent a
/// threshold: anything but a well-formed positive integer at
/// `lints.max_file_lines` in the nearest manifest is `None`.

use super::manifest::{max_file_lines_for, max_file_lines_in};

// ─── Extraction ──────────────────────────────────────────────────────────────

#[test]
fn dial_read_from_lints_object() {
    let text = r#"{
      "schema": "demoni.package.v0",
      "name": "demonios",
      "lints": { "max_file_lines": 2000 }
    }"#;
    assert_eq!(max_file_lines_in(text), Some(2000));
}

#[test]
fn absent_lints_or_dial_is_none() {
    assert_eq!(max_file_lines_in(r#"{ "name": "p" }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": {} }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": { "other_dial": 3 } }"#), None);
}

#[test]
fn malformed_or_wrong_typed_dial_is_none() {
    // The validator owns manifest diagnostics; the reader just declines.
    assert_eq!(max_file_lines_in("not json"), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": "2000" } }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": 0 } }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": -5 } }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": 12.5 } }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": true } }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": [2000] }"#), None);
    // Trailing garbage after the root value is not json.
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": 9 } } x"#), None);
}

/// #518 (post-merge nit from #514): before this, `1e1`/`10.0` armed a
/// 10-line dial the same as a plain `10` — the reader read straight through
/// to the numeric VALUE. `tools/validate_manifest.py`'s
/// `isinstance(value, int)` check rejects both (Python's `json.loads` keeps
/// `10` an `int` but decodes `1e1`/`10.0` as `float`), so a manifest using
/// either failed `make check` while still silently working under `dmc` —
/// the exact divergence the issue named. Tightened to the plain-integer
/// TOKEN, not the value: none of these arm the dial now, even though every
/// one is numerically 10.
#[test]
fn a_float_form_token_does_not_arm_the_dial_even_at_an_integral_value() {
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": 1e1 } }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": 1E1 } }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": 10.0 } }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": 1.0e1 } }"#), None);
}

/// #518/#514: "leading-zero forms lex too" — `007` is not a JSON number at
/// all (the grammar is `INT = "0" | [1-9] DIGIT*`), and the old number
/// scanner's `[0-9.eE+-]` character class read straight through it as 7.
/// `tools/validate_manifest.py` does not parse it as JSON either
/// (`json.loads` raises on a leading zero), so `None` here — the manifest
/// fails to parse at all, and this module's own stated design is that a
/// manifest that does not parse is silently ignored — now matches, where
/// the old scanner instead found a number and armed the lint from it.
#[test]
fn a_leading_zero_form_does_not_parse_as_a_number() {
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": 007 } }"#), None);
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": -007 } }"#), None);
}

/// A plain integer token is still exactly what arms the dial — the
/// tightening narrows accepted *forms*, not the ordinary case.
#[test]
fn a_plain_integer_token_still_arms_the_dial() {
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": 10 } }"#), Some(10));
    assert_eq!(max_file_lines_in(r#"{ "lints": { "max_file_lines": 1 } }"#), Some(1));
}

#[test]
fn dial_not_confused_by_strings_or_nesting() {
    // The key must be found structurally, not textually: a string value
    // containing the spelling, or the key nested under the wrong parent,
    // must not read as the dial.
    let decoy = r#"{
      "name": "p",
      "entry": "lints/max_file_lines.dmc",
      "modules": { "m": "a \"lints\": {\"max_file_lines\": 7} b.dmc" }
    }"#;
    assert_eq!(max_file_lines_in(decoy), None);
    let nested = r#"{ "modules": { "lints": { "max_file_lines": 7 } } }"#;
    assert_eq!(max_file_lines_in(nested), None);
}

// ─── Resolution walk ─────────────────────────────────────────────────────────

fn write(path: &std::path::Path, text: &str) {
    std::fs::write(path, text).expect("write test file");
}

#[test]
fn nearest_manifest_governs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    let pkg = root.join("pkg");
    let src_dir = pkg.join("src");
    std::fs::create_dir_all(&src_dir).expect("mkdirs");
    write(&root.join("demoni.json"), r#"{ "lints": { "max_file_lines": 100 } }"#);
    let src = src_dir.join("a.dmc");
    write(&src, "fn main() -> i64 { 0 }");

    // No nearer manifest: the root's dial applies, from any depth.
    assert_eq!(max_file_lines_for(&src), Some(100));

    // A nearer manifest shadows the outer one even when it has no dial —
    // nearest wins, the walk does not keep looking past it.
    write(&pkg.join("demoni.json"), r#"{ "name": "inner" }"#);
    assert_eq!(max_file_lines_for(&src), None);

    // And when the nearer manifest sets its own dial, that dial governs.
    write(&pkg.join("demoni.json"), r#"{ "lints": { "max_file_lines": 55 } }"#);
    assert_eq!(max_file_lines_for(&src), Some(55));
}

#[test]
fn no_manifest_anywhere_is_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("a.dmc");
    write(&src, "fn main() -> i64 { 0 }");
    // A tempdir can still sit under some ancestor with a manifest in
    // pathological environments; this repo's CI and dev machines do not.
    assert_eq!(max_file_lines_for(&src), None);
}
