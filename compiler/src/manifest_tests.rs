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
