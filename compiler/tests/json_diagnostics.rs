//! End-to-end coverage of `--json` structured diagnostics (#485), schema 1.
//!
//! The unit tests in `src/diag_tests.rs` cover the encoder and the refusal
//! vocabulary; this file drives the actual `dmc` binary, because the thing
//! being tested is a command line: what a consumer reads off the pipe, on which
//! stream, with which exit code — and, just as load-bearing, that the human
//! rendering is untouched when the flag is absent.
//!
//! Assertions here are deliberately whole-line rather than `contains`: a schema
//! is a promise about the exact bytes, and a substring test would pass through
//! a renamed field, a reordered object, or a dropped `schema`.

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

struct Probe {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Probe {
    fn new(src: &str) -> Probe {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("probe.dmc");
        std::fs::write(&path, src).expect("write probe");
        Probe { _dir: dir, path }
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(dmc_binary())
            .args(args)
            .arg(&self.path)
            .output()
            .expect("spawn dmc")
    }

    /// `run`, but the probe path goes where `{}` appears rather than last.
    fn run_with_path_at(&self, args: &[&str]) -> Output {
        let mut cmd = Command::new(dmc_binary());
        for a in args {
            if *a == "{}" { cmd.arg(&self.path); } else { cmd.arg(a); }
        }
        cmd.output().expect("spawn dmc")
    }

    /// The path exactly as the diagnostics report it: `dmc` canonicalizes the
    /// input before anything else, so that is what lands in `"file"`.
    fn reported(&self) -> String {
        self.path.canonicalize().expect("canonicalize probe").display().to_string()
    }
}

fn stderr(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).expect("stderr is utf-8")
}

fn stdout(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout is utf-8")
}

/// The JSON Lines on stderr, split into lines.
fn lines(out: &Output) -> Vec<String> {
    let e = stderr(out);
    if e.is_empty() { return Vec::new(); }
    e.trim_end_matches('\n').split('\n').map(|s| s.to_string()).collect()
}

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// Type-checks and runs on both backends.
const CLEAN: &str = "fn main() -> i64 {\n    let x: i64 = 41\n    x + 1\n}\n";

/// One type error, at a location the tests pin exactly.
const TYPE_ERROR: &str = "fn main() -> i64 {\n    let z: str = 5\n    1\n}\n";

/// `PORTS.md §5` — a tagged diagnostic, so `code` is populated from the tag.
const PORT_FORBIDDEN: &str = "@grad fn loss(!w: Tensor[f32, [4]]) -> f32 {\n    \
                              let (p, e) = port_open(\"python\")\n    \
                              sum(w .* w)\n}\n";

/// Type-checks clean; the JIT declines to lower a range expression.
const JIT_REFUSED: &str = "fn main() -> i64 {\n    let r = 0..3\n    0\n}\n";

/// A non-fatal safe-mode lint (#232) — a warning, not an error.
const WARNS: &str = "fn main() -> i64 {\n    let x: i64 = 1\n    let x = x\n    x\n}\n";

/// A shape error: the declared and actual shapes disagree, so the diagnostic
/// carries them as `expected`/`actual` arrays.
const SHAPE_ERROR: &str =
    "fn main() -> nil {\n    let a: Tensor[f32, [2]] = [1.0, 2.0, 3.0]\n    nil\n}\n";

/// Two zero-arg tests, one passing and one failing — scalar only, so the file
/// is inside the JIT subset and `--jit` runs both.
const TESTS_MIXED: &str = "fn test_add() -> bool { 1 + 1 == 2 }\nfn test_bad() -> bool { false }\n";

/// One passing test in a file the JIT declines (a range expression), so the
/// `--jit` parity verdict is `skip`, not a failure.
const TESTS_OUTSIDE_JIT: &str = "fn test_range() -> bool {\n    let r = 0..3\n    true\n}\n";

/// An element-type mismatch over EQUAL shapes — DIAGNOSTICS.md §4: "An
/// element-type mismatch over equal shapes carries neither field [expected,
/// actual] — the payload appears only when the shapes are the problem."
/// `make()`'s declared return carries a real `Tensor[i64, [2]]` (unlike a bare
/// `forge.zeros[i64, [2]]`, which types `Unknown` and would not mismatch at
/// all — `let a: Tensor[f32, [2]] = forge.zeros[i64, [2]]` passes `--check`
/// clean, so the function-return detour is what actually surfaces this
/// diagnostic to test).
const ELEM_TYPE_MISMATCH_EQUAL_SHAPES: &str =
    "fn make() -> Tensor[i64, [2]] { forge.zeros[i64, [2]] }\n\
     fn main() -> nil {\n    let a: Tensor[f32, [2]] = make()\n    nil\n}\n";

// ── The envelope ─────────────────────────────────────────────────────────────

#[test]
fn a_clean_check_emits_nothing_but_the_envelope() {
    let probe = Probe::new(CLEAN);
    let out = probe.run(&["--check", "--json"]);
    assert!(out.status.success(), "clean program failed:\n{}", stderr(&out));
    assert_eq!(
        stderr(&out),
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"--check\",\"status\":\"ok\",\
         \"errors\":0,\"warnings\":0,\"items\":1,\"exit\":0}\n",
    );
    // The human `✅ Check OK` line is the summary's job now; stdout stays clean
    // for whatever the *program* writes.
    assert_eq!(stdout(&out), "");
}

#[test]
fn a_clean_jit_run_keeps_program_output_on_stdout_and_json_on_stderr() {
    let probe = Probe::new(CLEAN);
    let out = probe.run(&["jit", "--json"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(stdout(&out), "=> 42\n", "program output moved or changed");
    assert_eq!(
        stderr(&out),
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"ok\",\
         \"errors\":0,\"warnings\":0,\"exit\":0}\n",
    );
}

#[test]
fn every_run_ends_with_exactly_one_summary() {
    // The envelope is what makes a truncated stream detectable. One per run —
    // on the clean path, the error path, the refusal path, and the usage-error
    // path alike.
    for (args, src) in [
        (vec!["--check", "--json"], CLEAN),
        (vec!["--check", "--json"], TYPE_ERROR),
        (vec!["--check", "--json"], WARNS),
        (vec!["--check", "--json"], SHAPE_ERROR),
        (vec!["jit", "--json"], CLEAN),
        (vec!["jit", "--json"], JIT_REFUSED),
        (vec!["run", "--json"], CLEAN),
        (vec!["test", "--json"], CLEAN),
        (vec!["test", "--json"], TESTS_MIXED),
        (vec!["test", "--json", "--jit"], TESTS_MIXED),
    ] {
        let probe = Probe::new(src);
        let out = probe.run(&args);
        let ls = lines(&out);
        let summaries = ls.iter().filter(|l| l.contains("\"kind\":\"summary\"")).count();
        assert_eq!(summaries, 1, "{:?} emitted {} summaries:\n{:#?}", args, summaries, ls);
        assert!(
            ls.last().unwrap().contains("\"kind\":\"summary\""),
            "{:?}: the summary is not last:\n{:#?}", args, ls,
        );
    }
}

#[test]
fn every_line_is_a_standalone_object_carrying_the_schema() {
    // A line torn out of a log has to be interpretable on its own, and nothing
    // that is not JSON may reach the stream — no `⚠`, no `✅`, no bare prose.
    for (args, src) in [
        (vec!["--check", "--json"], TYPE_ERROR),
        (vec!["--check", "--json"], PORT_FORBIDDEN),
        (vec!["--check", "--json"], WARNS),
        (vec!["--check", "--json"], SHAPE_ERROR),
        (vec!["jit", "--json"], JIT_REFUSED),
        (vec!["run", "--json"], CLEAN),
        (vec!["test", "--json"], TESTS_MIXED),
        (vec!["test", "--json", "--jit"], TESTS_MIXED),
    ] {
        let probe = Probe::new(src);
        let out = probe.run(&args);
        let ls = lines(&out);
        assert!(!ls.is_empty(), "{:?} emitted nothing", args);
        for l in &ls {
            assert!(l.starts_with('{') && l.ends_with('}'), "{:?}: not an object: {}", args, l);
            assert!(l.starts_with("{\"schema\":1,\"kind\":\""), "{:?}: {}", args, l);
            assert!(!l.contains('⚠') && !l.contains('✅'), "{:?}: prose leaked: {}", args, l);
        }
    }
}

// ── Check errors ─────────────────────────────────────────────────────────────

#[test]
fn a_check_error_is_one_whole_json_object() {
    let probe = Probe::new(TYPE_ERROR);
    let out = probe.run(&["--check", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        lines(&out),
        vec![
            format!(
                "{{\"schema\":1,\"kind\":\"check\",\"severity\":\"error\",\
                 \"message\":\"let binding has type Str but value has type {{integer}}\",\
                 \"file\":\"{}\",\"line\":2,\"col\":5,\"start\":23,\"end\":37}}",
                probe.reported(),
            ),
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"--check\",\"status\":\"failed\",\
             \"errors\":1,\"warnings\":0,\"items\":1,\"exit\":1}".to_string(),
        ],
    );
    assert_eq!(stdout(&out), "");
}

#[test]
fn a_tagged_check_error_lifts_its_tag_into_the_code() {
    // `PORTS.md §5` names `port-forbidden`; `--json` puts it where a consumer
    // can switch on it instead of splitting the sentence. The whole object,
    // because the point is that nothing else about the diagnostic moved.
    let probe = Probe::new(PORT_FORBIDDEN);
    let out = probe.run(&["--check", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        lines(&out)[0],
        format!(
            "{{\"schema\":1,\"kind\":\"check\",\"severity\":\"error\",\
             \"code\":\"port-forbidden\",\
             \"message\":\"port-forbidden: `port_open` is illegal inside a `@grad fn` \
             — a port call is an effect boundary the gradient cannot cross (PORTS.md §5)\",\
             \"file\":\"{}\",\"line\":2,\"col\":18,\"start\":62,\"end\":81}}",
            probe.reported(),
        ),
    );
}

#[test]
fn a_warning_is_a_severity_not_a_failure() {
    // A lint must not change the verdict: `severity` says warning, the summary
    // counts it separately, and the exit code is still 0.
    let probe = Probe::new(WARNS);
    let out = probe.run(&["--check", "--json"]);
    assert!(out.status.success(), "a lint failed the check:\n{}", stderr(&out));
    assert_eq!(
        lines(&out),
        vec![
            format!(
                "{{\"schema\":1,\"kind\":\"check\",\"severity\":\"warning\",\
                 \"message\":\"identity rebind `let x = x` is a redundant self-copy\",\
                 \"file\":\"{}\",\"line\":3,\"col\":5,\"start\":42,\"end\":51,\
                 \"hint\":\"drop the rebind — dead code \
                 (also the signature of degenerate codegen)\"}}",
                probe.reported(),
            ),
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"--check\",\"status\":\"ok\",\
             \"errors\":0,\"warnings\":1,\"items\":1,\"exit\":0}".to_string(),
        ],
    );
}

#[test]
fn a_shape_error_carries_expected_and_actual_as_data() {
    // The issue's own proposal: the two shapes as arrays a consumer indexes,
    // not a rendered `Tensor[F32, [2]]` it re-parses out of the message. The
    // message itself is unchanged — this is a second encoding, not a rewording.
    let probe = Probe::new(SHAPE_ERROR);
    let out = probe.run(&["--check", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        lines(&out),
        vec![
            format!(
                "{{\"schema\":1,\"kind\":\"check\",\"severity\":\"error\",\
                 \"message\":\"let binding has type Tensor[F32, [2]] but value \
                 has type Tensor[F32, [3]]\",\
                 \"expected\":[2],\"actual\":[3],\
                 \"file\":\"{}\",\"line\":2,\"col\":5,\"start\":23,\"end\":64}}",
                probe.reported(),
            ),
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"--check\",\"status\":\"failed\",\
             \"errors\":1,\"warnings\":0,\"items\":1,\"exit\":1}".to_string(),
        ],
    );
}

/// #518 (post-merge nit from #516): the companion case DIAGNOSTICS.md §4
/// states in prose but nothing pinned — an element-type mismatch over EQUAL
/// shapes carries the message and nothing else. Same shape `[2]` on both
/// sides, only the element type differs, so `expected`/`actual` must be
/// absent entirely (never present-but-equal, never null — §4's "optional
/// fields are omitted").
#[test]
fn an_element_type_mismatch_over_equal_shapes_carries_no_shape_payload() {
    let probe = Probe::new(ELEM_TYPE_MISMATCH_EQUAL_SHAPES);
    let out = probe.run(&["--check", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(
        ls[0],
        format!(
            "{{\"schema\":1,\"kind\":\"check\",\"severity\":\"error\",\
             \"message\":\"let binding has type Tensor[F32, [2]] but value \
             has type Tensor[I64, [2]]\",\
             \"file\":\"{}\",\"line\":3,\"col\":5,\"start\":79,\"end\":111}}",
            probe.reported(),
        ),
    );
    assert!(!ls[0].contains("\"expected\"") && !ls[0].contains("\"actual\""),
        "an equal-shape mismatch must carry no shape payload: {}", ls[0]);
}

// ── JIT ineligibility ────────────────────────────────────────────────────────

#[test]
fn a_jit_refusal_is_one_whole_json_object_with_a_class_code() {
    let probe = Probe::new(JIT_REFUSED);
    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        lines(&out),
        vec![
            format!(
                "{{\"schema\":1,\"kind\":\"jit-ineligible\",\"severity\":\"error\",\
                 \"code\":\"jit-construct\",\
                 \"message\":\"range expressions; use `dmc run` for full semantics\",\
                 \"file\":\"{}\",\"line\":2,\"col\":13}}",
                probe.reported(),
            ),
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"failed\",\
             \"errors\":1,\"warnings\":0,\"exit\":1}".to_string(),
        ],
    );
    assert_eq!(stdout(&out), "", "a refused program produced output");
}

#[test]
fn distinct_refusal_classes_get_distinct_codes_at_real_locations() {
    // The claim roadmap §2 rests on: the support table can be *generated*,
    // because the code separates classes of refusal that prose alone would
    // conflate. Each program below type-checks and is refused by the JIT for a
    // different reason; each must report its own code, at its own span.
    let cases: [(&str, &str, usize, usize); 7] = [
        // (source, expected code, line, col)
        (JIT_REFUSED, "jit-construct", 2, 13),
        ("type Foo = i64\nfn main() -> i64 { 0 }\n", "jit-item", 1, 1),
        (
            "fn helper(a: i64, b: i64) -> i64 { a + b }\n\
             fn main() -> i64 { helper(a = 1, b = 2) }\n",
            "jit-arg-form", 2, 27,
        ),
        (
            "fn main() -> i64 {\n    let t = (1, (2, 3))\n    let (a, (b, c)) = t\n    0\n}\n",
            "jit-pattern", 3, 5,
        ),
        (
            "fn main() -> i64 {\n    let xs = [1, 2, 3]\n    for x in xs { print(x) }\n    0\n}\n",
            "jit-loop-form", 3, 5,
        ),
        (
            "fn main() -> i64 {\n    @deterministic\n    let x = 1\n    0\n}\n",
            "jit-directive", 2, 5,
        ),
        (
            "fn main() -> i64 {\n    let s = \"hi\"\n    let n = -s\n    0\n}\n",
            "jit-operand-type", 3, 13,
        ),
    ];

    let mut seen: Vec<&str> = Vec::new();
    for (src, code, line, col) in cases {
        let probe = Probe::new(src);
        // Precondition: the checker accepts it, so the refusal really is the
        // JIT's subset boundary and not a type error wearing a jit hat.
        let checked = probe.run(&["--check", "--json"]);
        assert!(
            checked.status.success(),
            "fixture for {} does not type-check:\n{}", code, stderr(&checked),
        );

        let out = probe.run(&["jit", "--json"]);
        assert_eq!(out.status.code(), Some(1), "{} was accepted by the JIT", code);
        let first = &lines(&out)[0];
        assert!(
            first.starts_with(&format!(
                "{{\"schema\":1,\"kind\":\"jit-ineligible\",\"severity\":\"error\",\"code\":\"{}\",",
                code,
            )),
            "expected {} first on the line, got:\n{}", code, first,
        );
        assert!(
            first.ends_with(&format!(
                "\"file\":\"{}\",\"line\":{},\"col\":{}}}", probe.reported(), line, col,
            )),
            "expected {}:{} at the end of the line, got:\n{}", line, col, first,
        );
        assert!(!seen.contains(&code), "two fixtures share the code {}", code);
        seen.push(code);
    }
    assert_eq!(seen.len(), 7);
}

#[test]
fn a_jit_refusal_and_a_jit_defect_are_different_kinds() {
    // #480's distinction, in the schema rather than in English: a refusal is
    // `jit-ineligible` and carries a class; anything else is `jit-error` and
    // carries none. A consumer allowlists gaps by kind, never by grepping the
    // message — which is exactly the bug #480 reports.
    let probe = Probe::new(JIT_REFUSED);
    let out = probe.run(&["jit", "--json"]);
    let first = &lines(&out)[0];
    assert!(first.contains("\"kind\":\"jit-ineligible\""), "{}", first);
    assert!(first.contains("\"code\":\"jit-"), "a refusal with no class code: {}", first);

    // The human encoding of the same distinction is unchanged.
    let human = stderr(&probe.run(&["jit"]));
    assert!(human.starts_with("jit unsupported at "), "{}", human);
}

// ── `dmc test` results ───────────────────────────────────────────────────────

#[test]
fn each_test_gets_one_object_and_the_summary_carries_the_tally() {
    let probe = Probe::new(TESTS_MIXED);
    let out = probe.run(&["test", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        lines(&out),
        vec![
            format!(
                "{{\"schema\":1,\"kind\":\"test\",\"status\":\"pass\",\
                 \"name\":\"test_add\",\"file\":\"{}\"}}",
                probe.reported(),
            ),
            format!(
                "{{\"schema\":1,\"kind\":\"test\",\"status\":\"fail\",\
                 \"name\":\"test_bad\",\"file\":\"{}\",\
                 \"message\":\"returned false\"}}",
                probe.reported(),
            ),
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"test\",\"status\":\"failed\",\
             \"errors\":0,\"warnings\":0,\"passed\":1,\"failed\":1,\"exit\":1}".to_string(),
        ],
    );
    // The `ok`/`FAIL` harness lines are the objects' job now; stdout stays
    // clean for whatever the tests themselves print.
    assert_eq!(stdout(&out), "");
}

#[test]
fn under_jit_the_parity_verdict_rides_on_each_test_object() {
    // A failed test is not a *diagnostic*: `errors` stays 0, and the verdict
    // lives in `passed`/`failed` — the same numbers, and the same
    // double-counting of a jit divergence, as the human `test result:` line.
    let probe = Probe::new(TESTS_MIXED);
    let out = probe.run(&["test", "--json", "--jit"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        lines(&out),
        vec![
            format!(
                "{{\"schema\":1,\"kind\":\"test\",\"status\":\"pass\",\
                 \"name\":\"test_add\",\"file\":\"{}\",\"jit\":\"pass\"}}",
                probe.reported(),
            ),
            format!(
                "{{\"schema\":1,\"kind\":\"test\",\"status\":\"fail\",\
                 \"name\":\"test_bad\",\"file\":\"{}\",\
                 \"message\":\"returned false\",\"jit\":\"fail\",\
                 \"jit_message\":\"returned false (diverges from interp)\"}}",
                probe.reported(),
            ),
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"test\",\"status\":\"failed\",\
             \"errors\":0,\"warnings\":0,\"passed\":0,\"failed\":2,\
             \"jit_ran\":2,\"jit_skipped\":0,\"exit\":1}".to_string(),
        ],
    );
}

#[test]
fn a_file_outside_the_jit_subset_is_skipped_not_failed() {
    // #480's allowlisting need, on the test path: a real gap in the JIT subset
    // is `"jit":"skip"` — machine-readable, and not a failure. The exit code
    // and the parity counters agree with the human note.
    let probe = Probe::new(TESTS_OUTSIDE_JIT);
    let out = probe.run(&["test", "--json", "--jit"]);
    assert!(out.status.success(), "a skipped file failed the run:\n{}", stderr(&out));
    assert_eq!(
        lines(&out),
        vec![
            format!(
                "{{\"schema\":1,\"kind\":\"test\",\"status\":\"pass\",\
                 \"name\":\"test_range\",\"file\":\"{}\",\"jit\":\"skip\"}}",
                probe.reported(),
            ),
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"test\",\"status\":\"ok\",\
             \"errors\":0,\"warnings\":0,\"passed\":1,\"failed\":0,\
             \"jit_ran\":0,\"jit_skipped\":1,\"exit\":0}".to_string(),
        ],
    );
}

#[test]
fn a_test_file_with_no_tests_still_gets_the_envelope() {
    // The human path prints `0 tests` and exits 0; the JSON path says the same
    // thing as a summary, because a stream without an envelope is truncated.
    let probe = Probe::new(CLEAN);
    let out = probe.run(&["test", "--json"]);
    assert!(out.status.success());
    assert_eq!(
        stderr(&out),
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"test\",\"status\":\"ok\",\
         \"errors\":0,\"warnings\":0,\"passed\":0,\"failed\":0,\"exit\":0}\n",
    );
    assert_eq!(stdout(&out), "");
}

/// #518 (post-merge nit from #516): DIAGNOSTICS.md §7 states this path in
/// prose — "Under `dmc test`, a file-level failure (`FAIL <file>: resolution
/// failed …`) likewise arrives as `unstructured` — it is not a *test*, so it
/// gets no `test` object, but it still counts into the summary's `failed`"
/// — but nothing committed exercised it. A `use` of a file that does not
/// exist fails `Resolver::resolve_all` before any test runs, which is a
/// different path from every other test in this file (all of which fail
/// inside a real test run, and so DO get a `test` object).
#[test]
fn a_file_level_test_failure_is_unstructured_with_no_test_object() {
    let probe = Probe::new("use \"does_not_exist_518.dmc\" as m\n\
                             fn test_never_runs() -> bool { true }\n");
    let out = probe.run(&["test", "--json"]);
    // Disclosed quirk (DIAGNOSTICS.md §4): a run that found no *tests* (this
    // one never got that far) reports `exit:0` regardless — the exit code
    // does not depend on `--json`, so neither do the numbers explaining it.
    assert!(out.status.success(), "{}", stderr(&out));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "expected exactly the failure line + summary:\n{:#?}", ls);
    assert!(ls[0].starts_with("{\"schema\":1,\"kind\":\"unstructured\",\"severity\":\"error\",\
                                \"message\":\"FAIL "),
        "not unstructured: {}", ls[0]);
    assert!(ls[0].contains("resolution failed"), "{}", ls[0]);
    // No `test` object exists for this failure — `kind` is never `"test"`.
    assert!(!ls[0].contains("\"kind\":\"test\""), "{}", ls[0]);
    assert_eq!(
        ls[1],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"test\",\"status\":\"ok\",\
         \"errors\":1,\"warnings\":0,\"passed\":0,\"failed\":1,\"exit\":0}",
    );
}

// ── Reserved categories ──────────────────────────────────────────────────────

#[test]
fn a_category_with_no_schema_yet_still_reaches_the_stream() {
    // A trap in compiled code is a *runtime* diagnostic — a later slice. Schema
    // 1 must not drop it, and must not emit prose beside the JSON either: it
    // comes through as `unstructured`, text intact, stream still parseable.
    let probe = Probe::new(
        "fn main() -> i64 {\n    let t = forge.zeros[f32, [512, 512]]\n    0\n}\n",
    );
    let out = probe.run(&["jit", "--json", "--forge=1K"]);
    assert_eq!(out.status.code(), Some(1), "a 1 KiB forge allocated 1 MiB");
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "{:#?}", ls);
    assert!(
        ls[0].starts_with("{\"schema\":1,\"kind\":\"unstructured\",\"severity\":\"error\",\"message\":\""),
        "{}", ls[0],
    );
    assert_eq!(
        ls[1],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"failed\",\
         \"errors\":1,\"warnings\":0,\"exit\":1}",
    );
    // The prose the human renderer would have printed is carried verbatim.
    let human = stderr(&probe.run(&["jit", "--forge=1K"]));
    let msg = human.trim_end().replace('\\', "\\\\").replace('"', "\\\"");
    assert!(!msg.is_empty(), "the human path printed nothing to compare against");
    assert!(ls[0].contains(&format!("\"message\":\"{}\"", msg)), "{}\nvs\n{}", ls[0], msg);
}

#[test]
fn the_flag_is_refused_where_schema_1_has_no_answer() {
    // Accepting `--json` on `dmc run` and emitting only prose would be
    // indistinguishable, to a consumer, from a compiler that produced no
    // diagnostics. Refuse in JSON, so even the refusal is machine-readable.
    for (args, cmd) in [
        (vec!["run", "--json"], "run"),
        (vec!["fmt", "--json"], "fmt"),
        (vec!["--parse", "--json"], "--parse"),
        (vec!["--lex", "--json"], "--lex"),
        // A bare file argument is the implicit full pipeline. The summary must
        // name the command, not echo the path back as if it were one.
        (vec!["--json"], "pipeline"),
    ] {
        let probe = Probe::new(CLEAN);
        let out = probe.run(&args);
        assert_eq!(out.status.code(), Some(1), "`dmc {} --json` was accepted", cmd);
        assert_eq!(
            lines(&out),
            vec![
                format!(
                    "{{\"schema\":1,\"kind\":\"cli\",\"severity\":\"error\",\
                     \"code\":\"json-unsupported-command\",\
                     \"message\":\"`--json` is wired for `dmc --check`, `dmc jit`, \
                     and `dmc test` in schema 1; `{}` still reports in the human format\"}}",
                    cmd,
                ),
                format!(
                    "{{\"schema\":1,\"kind\":\"summary\",\"command\":\"{}\",\
                     \"status\":\"failed\",\"errors\":1,\"warnings\":0,\"exit\":1}}",
                    cmd,
                ),
            ],
        );
    }
}

// ── The additive promise ─────────────────────────────────────────────────────

#[test]
fn human_output_is_byte_identical_without_the_flag() {
    // The whole bytes, pinned. `--json` is a second encoding; if adding it
    // moved so much as a space in the default rendering, that is a regression
    // for every human and every tool that already reads this.
    let clean = Probe::new(CLEAN);
    let out = clean.run(&["--check"]);
    assert!(out.status.success());
    assert_eq!(stdout(&out), "✅ Check OK — 1 top-level items, no type errors\n");
    assert_eq!(stderr(&out), "");

    let bad = Probe::new(TYPE_ERROR);
    let out = bad.run(&["--check"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(stdout(&out), "");
    assert_eq!(
        stderr(&out),
        "type error at 2:5: let binding has type Str but value has type {integer}\n\
         \n1 type error(s)\n",
    );

    let warns = Probe::new(WARNS);
    let out = warns.run(&["--check"]);
    assert!(out.status.success());
    assert_eq!(stdout(&out), "✅ Check OK — 1 top-level items, no type errors\n");
    assert_eq!(
        stderr(&out),
        "warning at 3:5: identity rebind `let x = x` is a redundant self-copy\n  \
         hint: drop the rebind — dead code (also the signature of degenerate codegen)\n",
    );

    let refused = Probe::new(JIT_REFUSED);
    let out = refused.run(&["jit"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(stdout(&out), "");
    assert_eq!(
        stderr(&out),
        "jit unsupported at 2:13: range expressions; use `dmc run` for full semantics\n",
    );

    let ok = Probe::new(CLEAN);
    let out = ok.run(&["jit"]);
    assert!(out.status.success());
    assert_eq!(stdout(&out), "=> 42\n");
    assert_eq!(stderr(&out), "");

    // `dmc test`: the `ok`/`FAIL` lines, their streams, and the summary line
    // are all pinned — with and without `--jit`. The label spells the path as
    // given; only the JSON encoding canonicalizes.
    let tests = Probe::new(TESTS_MIXED);
    let label = tests.path.display().to_string();
    let out = tests.run(&["test"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(stdout(&out), format!("ok   {}::test_add\n", label));
    assert_eq!(
        stderr(&out),
        format!(
            "FAIL {}::test_bad: returned false\ntest result: FAILED. 1 passed; 1 failed\n",
            label,
        ),
    );

    let out = tests.run(&["test", "--jit"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(stdout(&out), format!("ok   {}::test_add\n", label));
    assert_eq!(
        stderr(&out),
        format!(
            "FAIL {}::test_bad: returned false\n\
             FAIL {}::test_bad [jit]: returned false (diverges from interp)\n\
             test result: FAILED. 0 passed; 2 failed | jit parity: 2 ran, 0 skipped \
             (outside JIT subset)\n",
            label, label,
        ),
    );
}

#[test]
fn the_flag_changes_the_format_and_nothing_else() {
    // Same programs, same commands, with and without `--json`: the exit code
    // must be identical every time, and the program's own stdout untouched.
    for (cmd, src) in [
        ("--check", CLEAN),
        ("--check", TYPE_ERROR),
        ("--check", PORT_FORBIDDEN),
        ("--check", WARNS),
        ("--check", SHAPE_ERROR),
        ("jit", CLEAN),
        ("jit", JIT_REFUSED),
        ("test", CLEAN),
        ("test", TESTS_MIXED),
    ] {
        let probe = Probe::new(src);
        let human = probe.run(&[cmd]);
        let json = probe.run(&[cmd, "--json"]);
        assert_eq!(
            human.status.code(), json.status.code(),
            "`dmc {}` changed its exit code under --json", cmd,
        );
        // `dmc --check`'s stdout *is* its report, and so are `dmc test`'s
        // `ok` lines — both move into the envelope; a program's own output
        // does not.
        if cmd == "jit" {
            assert_eq!(stdout(&human), stdout(&json), "`dmc jit` program output moved");
        } else {
            assert_eq!(stdout(&json), "", "`{} --json` wrote to stdout", cmd);
        }
    }
}

#[test]
fn the_flag_is_stripped_wherever_it_appears() {
    // `--json` is a `dmc` flag, stripped in the same pass as `--profile` and
    // the arena flags — so it must not be mistaken for the input path, and it
    // must work after the path as well as before it.
    let probe = Probe::new(TYPE_ERROR);
    let before = probe.run_with_path_at(&["--check", "--json", "{}"]);
    let after = probe.run_with_path_at(&["--check", "{}", "--json"]);
    assert_eq!(before.status.code(), Some(1));
    assert_eq!(stderr(&before), stderr(&after), "flag position changed the output");
    assert!(stderr(&after).starts_with("{\"schema\":1,"), "{}", stderr(&after));
}

// ── An independent decoder ───────────────────────────────────────────────────

/// Decode one flat JSON object into ordered `(key, value)` pairs. Deliberately
/// *not* the compiler's writer run backwards: an assertion built out of the
/// encoder's own assumptions cannot catch the encoder being wrong. Strict —
/// anything it does not recognise is a panic, so malformed output fails loudly.
fn decode(line: &str) -> Vec<(String, String)> {
    let b: Vec<char> = line.chars().collect();
    let mut i = 0;
    assert_eq!(b[i], '{', "not an object: {}", line);
    i += 1;
    let mut out = Vec::new();
    if b[i] == '}' { return out; }
    loop {
        let key = decode_string(&b, &mut i, line);
        assert_eq!(b[i], ':', "expected `:` at {} in {}", i, line);
        i += 1;
        let value = if b[i] == '"' {
            decode_string(&b, &mut i, line)
        } else if b[i] == '[' {
            // A flat array of numbers and strings — `expected`/`actual`. The
            // value is kept as its raw text; assertions on it are whole-value.
            let start = i;
            i += 1;
            if b[i] == ']' {
                i += 1;
            } else {
                loop {
                    if b[i] == '"' {
                        decode_string(&b, &mut i, line);
                    } else {
                        let s = i;
                        while b[i].is_ascii_digit() || b[i] == '-' { i += 1; }
                        assert!(i > s, "expected an array element at {} in {}", i, line);
                    }
                    match b[i] {
                        ',' => i += 1,
                        ']' => { i += 1; break; }
                        c => panic!("expected `,` or `]`, found {:?} at {} in {}", c, i, line),
                    }
                }
            }
            b[start..i].iter().collect()
        } else {
            let start = i;
            while b[i].is_ascii_digit() { i += 1; }
            assert!(i > start, "expected a string or a number at {} in {}", i, line);
            b[start..i].iter().collect()
        };
        out.push((key, value));
        match b[i] {
            ',' => i += 1,
            '}' => { i += 1; break; }
            c => panic!("expected `,` or `}}`, found {:?} at {} in {}", c, i, line),
        }
    }
    assert_eq!(i, b.len(), "trailing bytes after the object: {}", line);
    out
}

fn decode_string(b: &[char], i: &mut usize, line: &str) -> String {
    assert_eq!(b[*i], '"', "expected a string at {} in {}", i, line);
    *i += 1;
    let mut s = String::new();
    loop {
        let c = b[*i];
        *i += 1;
        match c {
            '"' => return s,
            '\\' => {
                let e = b[*i];
                *i += 1;
                match e {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    '/' => s.push('/'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    'b' => s.push('\u{8}'),
                    'f' => s.push('\u{c}'),
                    'u' => {
                        let hex: String = b[*i..*i + 4].iter().collect();
                        *i += 4;
                        let n = u32::from_str_radix(&hex, 16).expect("bad \\u escape");
                        s.push(char::from_u32(n).expect("bad code point"));
                    }
                    other => panic!("bad escape \\{} in {}", other, line),
                }
            }
            c => {
                assert!(!c.is_control(), "raw control character in {}", line);
                s.push(c);
            }
        }
    }
}

fn field<'a>(pairs: &'a [(String, String)], key: &str) -> &'a str {
    pairs
        .iter()
        .find(|(k, _)| k == key)
        .unwrap_or_else(|| panic!("no `{}` in {:?}", key, pairs))
        .1
        .as_str()
}

#[test]
fn the_decoder_round_trips_a_message_full_of_punctuation() {
    // A parser diagnostic that quotes a token puts `"` inside the message, so
    // the escaping is exercised by a real compile, not a synthetic string. An
    // independent decoder must get the original text back, byte for byte.
    const QUOTED: &str = "expected string literal for import path, found Ident(\"foo\")";
    for cmd in ["--check", "jit"] {
        let probe = Probe::new("use foo\nfn main() -> i64 { 0 }\n");
        let out = probe.run(&[cmd, "--json"]);
        assert_eq!(out.status.code(), Some(1));
        let ls = lines(&out);
        let d = decode(&ls[0]);
        assert_eq!(field(&d, "schema"), "1");
        assert_eq!(field(&d, "kind"), "parse", "`dmc {}`", cmd);
        assert_eq!(field(&d, "severity"), "error");
        assert_eq!(field(&d, "message"), QUOTED, "`dmc {}`", cmd);
        assert_eq!(field(&d, "file"), probe.reported());
        assert_eq!(field(&d, "line"), "1");
        assert_eq!(field(&d, "col"), "5");
        // Both commands reach the same encoding of the same failure.
        assert_eq!(d.len(), 7, "unexpected fields: {:?}", d);
    }
}

#[test]
fn every_emitted_line_decodes() {
    // The whole corpus of shapes this slice can produce, through a decoder that
    // knows nothing about how they were written.
    for (args, src) in [
        (vec!["--check", "--json"], CLEAN),
        (vec!["--check", "--json"], TYPE_ERROR),
        (vec!["--check", "--json"], PORT_FORBIDDEN),
        (vec!["--check", "--json"], WARNS),
        (vec!["--check", "--json"], SHAPE_ERROR),
        (vec!["jit", "--json"], CLEAN),
        (vec!["jit", "--json"], JIT_REFUSED),
        (vec!["run", "--json"], CLEAN),
        (vec!["test", "--json"], CLEAN),
        (vec!["test", "--json"], TESTS_MIXED),
        (vec!["test", "--json", "--jit"], TESTS_MIXED),
        (vec!["test", "--json", "--jit"], TESTS_OUTSIDE_JIT),
    ] {
        let probe = Probe::new(src);
        let out = probe.run(&args);
        for l in lines(&out) {
            let d = decode(&l);
            assert_eq!(field(&d, "schema"), "1", "{:?}: {}", args, l);
            assert_eq!(d[0].0, "schema", "{:?}: schema is not first: {}", args, l);
            assert_eq!(d[1].0, "kind", "{:?}: kind is not second: {}", args, l);
            let mut keys: Vec<&str> = d.iter().map(|(k, _)| k.as_str()).collect();
            let n = keys.len();
            keys.sort_unstable();
            keys.dedup();
            assert_eq!(keys.len(), n, "{:?}: duplicate key in {}", args, l);
        }
    }
}

#[test]
fn a_lex_error_reports_the_same_way_from_both_commands() {
    // A front-end failure has its own kind, so a consumer can tell "this never
    // became an AST" from "this failed the type checker". `--check` reaches the
    // lexer through the module resolver and `dmc jit` reaches it directly; the
    // encoding must not depend on which.
    for cmd in ["--check", "jit"] {
        let probe = Probe::new("fn main() -> i64 {\n    let s = \"unterminated\n}\n");
        let out = probe.run(&[cmd, "--json"]);
        assert_eq!(out.status.code(), Some(1), "`dmc {}` accepted it", cmd);
        assert_eq!(
            lines(&out)[0],
            format!(
                "{{\"schema\":1,\"kind\":\"lex\",\"severity\":\"error\",\
                 \"message\":\"unterminated string literal\",\
                 \"file\":\"{}\",\"line\":3,\"col\":1}}",
                probe.reported(),
            ),
            "`dmc {}`", cmd,
        );
    }
}

