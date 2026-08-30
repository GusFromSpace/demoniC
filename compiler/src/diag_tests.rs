//! Unit tests for the `--json` encoder and the refusal-code vocabulary (#485).
//!
//! The end-to-end behaviour is `tests/json_diagnostics.rs`, which drives the
//! real binary. These cover the two things a subprocess test cannot see: the
//! escaping edge cases, and the completeness of `jit::Refusal`.

use crate::diag::{
    shape_array, tag_of, Diagnostic, JitVerdict, Kind, Severity, Summary, TestResult, SCHEMA,
};
use crate::jit::Refusal;
use crate::shape::SymDim;

fn err(msg: &str) -> Diagnostic {
    Diagnostic::new(Kind::Check, Severity::Error, msg)
}

#[test]
fn the_schema_version_is_one() {
    // Not ceremony: a bump is a breaking change to every consumer, so it should
    // never happen by accident, and a test is the cheapest tripwire.
    assert_eq!(SCHEMA, 1);
}

#[test]
fn a_minimal_diagnostic_omits_every_absent_field() {
    assert_eq!(
        err("nope").to_json(),
        r#"{"schema":1,"kind":"check","severity":"error","message":"nope"}"#,
    );
}

#[test]
fn field_order_is_fixed() {
    // Consumers may parse, but the tests in this repo assert whole lines, and a
    // reordering would be a silent diff-storm. Order is part of the format.
    let d = err("m")
        .code("c")
        .file(std::path::Path::new("/f.dmc"))
        .at(3, 4)
        .bytes(10, 20)
        .hint(Some("h"));
    assert_eq!(
        d.to_json(),
        r#"{"schema":1,"kind":"check","severity":"error","code":"c","message":"m",\
"file":"/f.dmc","line":3,"col":4,"start":10,"end":20,"hint":"h"}"#
            .replace("\\\n", ""),
    );
}

#[test]
fn quotes_backslashes_and_newlines_survive_the_encoding() {
    // Diagnostics quote user source. A `"` in an identifier, a Windows path in
    // a `use`, or a multi-line hint must not be able to break the line format.
    let d = err("bad \"tok\" in C:\\x\nsecond line\ttabbed");
    assert_eq!(
        d.to_json(),
        r#"{"schema":1,"kind":"check","severity":"error",\
"message":"bad \"tok\" in C:\\x\nsecond line\ttabbed"}"#
            .replace("\\\n", "")
            // The raw string above holds literal backslash-n / backslash-t
            // already, so nothing else to do — this is the escaped form.
    );
    // And the encoded line really is one line.
    assert_eq!(d.to_json().lines().count(), 1);
}

#[test]
fn control_characters_become_short_or_u_escapes() {
    let d = err("a\u{0}b\u{8}c\u{c}d\u{1f}e");
    assert_eq!(
        d.to_json(),
        "{\"schema\":1,\"kind\":\"check\",\"severity\":\"error\",\
         \"message\":\"a\\u0000b\\bc\\fd\\u001fe\"}",
    );
}

#[test]
fn non_ascii_diagnostic_text_is_emitted_as_utf8() {
    // The JIT's messages carry `≥` and `…`; JSON's default encoding is UTF-8,
    // so escaping them would only make the output harder to read.
    let d = err("needs rank ≥ 2 — use `dmc run` …");
    assert!(d.to_json().contains("needs rank ≥ 2 — use `dmc run` …"), "{}", d.to_json());
}

#[test]
fn the_summary_reports_the_exit_code_it_is_about_to_use() {
    let s = Summary {
        command: "--check".into(),
        errors: 2,
        warnings: 1,
        passed: None,
        failed: None,
        jit_ran: None,
        jit_skipped: None,
        items: Some(7),
        exit: 1,
    };
    assert_eq!(
        s.to_json(),
        r#"{"schema":1,"kind":"summary","command":"--check","status":"failed",\
"errors":2,"warnings":1,"items":7,"exit":1}"#
            .replace("\\\n", ""),
    );
}

#[test]
fn a_summary_with_no_item_count_omits_it() {
    let s = Summary {
        command: "jit".into(),
        errors: 0,
        warnings: 0,
        passed: None,
        failed: None,
        jit_ran: None,
        jit_skipped: None,
        items: None,
        exit: 0,
    };
    assert_eq!(
        s.to_json(),
        r#"{"schema":1,"kind":"summary","command":"jit","status":"ok",\
"errors":0,"warnings":0,"exit":0}"#
            .replace("\\\n", ""),
    );
}

#[test]
fn every_kind_has_a_distinct_wire_name() {
    let all = [
        Kind::Lex, Kind::Parse, Kind::Check, Kind::JitIneligible,
        Kind::JitError, Kind::Cli, Kind::Unstructured,
    ];
    let mut names: Vec<&str> = all.iter().map(|k| k.as_str()).collect();
    names.sort_unstable();
    let n = names.len();
    names.dedup();
    assert_eq!(names.len(), n, "two kinds share a wire name");
    // `summary` is written by `Summary`, not `Kind`, and must not collide.
    assert!(!names.contains(&"summary"), "a diagnostic kind shadows the envelope");
}

// ── Structured shape data ────────────────────────────────────────────────────

#[test]
fn shape_dims_encode_constants_as_numbers_and_symbols_as_strings() {
    // The point of the payload: a consumer indexes dims as data. A constant is
    // a number it can compare; anything symbolic is its §8 rendering, quoted.
    assert_eq!(shape_array(&[]), "[]");
    assert_eq!(shape_array(&[SymDim::Const(2), SymDim::Const(3)]), "[2,3]");
    assert_eq!(
        shape_array(&[SymDim::Const(4), SymDim::Var("n".into())]),
        r#"[4,"n"]"#,
    );
    // A dynamic dim goes back out as `?` — the spelling the source used and,
    // since #501 (S3), the only one that re-parses. An unanalyzable dim is a
    // distinct state and must not borrow that spelling.
    assert_eq!(
        shape_array(&[SymDim::Streaming, SymDim::Wildcard]),
        r#"["~","?"]"#,
    );
    assert_eq!(
        shape_array(&[SymDim::Unknown, SymDim::Wildcard]),
        r#"["??","?"]"#,
    );
    assert_eq!(
        shape_array(&[SymDim::Mul(
            Box::new(SymDim::Var("h".into())),
            Box::new(SymDim::Const(2)),
        )]),
        r#"["(h*2)"]"#,
    );
}

#[test]
fn a_shape_error_carries_expected_and_actual_between_message_and_file() {
    let d = err("shape off")
        .shapes(&[SymDim::Const(2), SymDim::Const(3)], &[SymDim::Const(2), SymDim::Var("n".into())])
        .file(std::path::Path::new("/f.dmc"))
        .at(1, 2);
    assert_eq!(
        d.to_json(),
        r#"{"schema":1,"kind":"check","severity":"error","message":"shape off",\
"expected":[2,3],"actual":[2,"n"],"file":"/f.dmc","line":1,"col":2}"#
            .replace("\\\n", ""),
    );
}

// ── `test` results ───────────────────────────────────────────────────────────

#[test]
fn a_passing_test_is_status_name_file_and_nothing_else() {
    let t = TestResult {
        name: "test_add".into(),
        file: "/t.dmc".into(),
        pass: true,
        message: None,
        jit: None,
        jit_message: None,
    };
    assert_eq!(
        t.to_json(),
        r#"{"schema":1,"kind":"test","status":"pass","name":"test_add","file":"/t.dmc"}"#,
    );
}

#[test]
fn a_failing_test_carries_its_reason_and_the_parity_verdict_rides_along() {
    let t = TestResult {
        name: "test_bad".into(),
        file: "/t.dmc".into(),
        pass: false,
        message: Some("returned false".into()),
        jit: Some(JitVerdict::Fail),
        jit_message: Some("returned false (diverges from interp)".into()),
    };
    assert_eq!(
        t.to_json(),
        r#"{"schema":1,"kind":"test","status":"fail","name":"test_bad","file":"/t.dmc",\
"message":"returned false","jit":"fail",\
"jit_message":"returned false (diverges from interp)"}"#
            .replace("\\\n", ""),
    );
}

#[test]
fn the_three_parity_verdicts_have_distinct_wire_names() {
    let names = [JitVerdict::Pass, JitVerdict::Fail, JitVerdict::Skip]
        .map(JitVerdict::as_str);
    assert_eq!(names, ["pass", "fail", "skip"]);
}

#[test]
fn a_test_summary_reports_the_tallies_the_human_lines_print() {
    let s = Summary {
        command: "test".into(),
        errors: 0,
        warnings: 0,
        passed: Some(2),
        failed: Some(1),
        jit_ran: Some(1),
        jit_skipped: Some(2),
        items: None,
        exit: 1,
    };
    assert_eq!(
        s.to_json(),
        r#"{"schema":1,"kind":"summary","command":"test","status":"failed",\
"errors":0,"warnings":0,"passed":2,"failed":1,"jit_ran":1,"jit_skipped":2,"exit":1}"#
            .replace("\\\n", ""),
    );
}

// ── Tag extraction ───────────────────────────────────────────────────────────

#[test]
fn a_leading_kebab_tag_becomes_the_code() {
    assert_eq!(tag_of("port-forbidden: `port_open` is illegal"), Some("port-forbidden"));
    assert_eq!(tag_of("decode-type: expected i64, got f64"), Some("decode-type"));
    assert_eq!(tag_of("comptime-non-static: `n` is not comptime"), Some("comptime-non-static"));
    assert_eq!(tag_of("fuse-infeasible: cannot collapse `@`"), Some("fuse-infeasible"));
}

#[test]
fn ordinary_prose_with_a_colon_is_not_a_tag() {
    // The load-bearing negative: most checker messages contain a colon
    // somewhere, and mistaking one for a code would put garbage in the field
    // that consumers are meant to switch on.
    assert_eq!(tag_of("expected `i64`, found `str`"), None);
    assert_eq!(tag_of("let binding has type Str but value has type {integer}"), None);
    assert_eq!(tag_of("cannot infer: try an annotation"), None, "no hyphen — not a tag");
    assert_eq!(tag_of("Port-Forbidden: shouty"), None, "tags are lowercase");
    assert_eq!(tag_of("arg -1: out of range"), None, "a space disqualifies it");
    assert_eq!(tag_of("-leading: hyphen"), None);
    assert_eq!(tag_of("trailing-: hyphen"), None);
    assert_eq!(tag_of("no colon here"), None);
    assert_eq!(tag_of("tight-tag:no space"), None, "the separator is `: `");
}

// ── Refusal classes ──────────────────────────────────────────────────────────

/// Fails to compile when a `Refusal` variant is added, which is the point:
/// whoever adds one is forced to also add it to `Refusal::ALL` below.
#[allow(dead_code)]
fn every_variant_is_accounted_for(r: Refusal) {
    match r {
        Refusal::Item | Refusal::Import | Refusal::Extern | Refusal::Signature
        | Refusal::Directive | Refusal::Construct | Refusal::Pattern
        | Refusal::ArgForm | Refusal::AssignForm | Refusal::LoopForm
        | Refusal::IndirectCall | Refusal::UnknownFn | Refusal::OperandType
        | Refusal::BranchType | Refusal::F32Only | Refusal::DynamicShape
        | Refusal::Axis | Refusal::Grad | Refusal::SecondOrder => {}
    }
}

#[test]
fn every_refusal_class_is_listed() {
    assert_eq!(
        Refusal::ALL.len(),
        19,
        "a class was added or removed — update the roster and this count",
    );
}

#[test]
fn refusal_codes_are_unique_namespaced_and_kebab_case() {
    let mut seen: Vec<&str> = Vec::new();
    for r in Refusal::ALL {
        let c = r.code();
        assert!(c.starts_with("jit-"), "{:?} -> {} is not namespaced", r, c);
        assert!(
            c.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-'),
            "{:?} -> {} is not kebab-case", r, c,
        );
        assert!(!seen.contains(&c), "duplicate refusal code {}", c);
        seen.push(c);
    }
    assert_eq!(seen.len(), Refusal::ALL.len());
}

#[test]
fn a_refusal_code_is_a_valid_diagnostic_code() {
    // The codes have to survive the same tag rules the checker's do, so a
    // consumer can treat `code` as one flat vocabulary.
    for r in Refusal::ALL {
        let msg = format!("{}: something", r.code());
        assert_eq!(tag_of(&msg), Some(r.code()), "{:?}", r);
    }
}

#[test]
fn the_roster_is_sorted_by_code() {
    let codes: Vec<&str> = Refusal::ALL.iter().map(|r| r.code()).collect();
    let mut sorted = codes.clone();
    sorted.sort_unstable();
    assert_eq!(codes, sorted, "keep Refusal::ALL in code order");
}
