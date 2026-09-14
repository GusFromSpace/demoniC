//! End-to-end coverage of `dmc test --fail-empty` (#599).
//!
//! Before this flag, `dmc test` on a path with zero discovered `test_*`
//! functions printed `0 tests` and exited 0 — indistinguishable, by exit
//! status alone, from a suite that ran and passed. `--fail-empty` makes that
//! case exit 1 instead; without the flag, behavior is unchanged.

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

/// A program with no `test_*` functions at all.
const NO_TESTS: &str = "fn main() -> i64 {\n    let x: i64 = 41\n    x + 1\n}\n";

/// A program with one passing `test_*`.
const ONE_TEST: &str = "fn test_add() -> bool {\n    1 + 1 == 2\n}\n";

/// A program that fails to parse — a genuine file-level failure, not an
/// absence of tests.
const PARSE_FAILURE: &str = "fn test_broken( -> bool {\n    true\n}\n";

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
            .arg("test")
            .args(args)
            .arg(&self.path)
            .output()
            .expect("spawn dmc")
    }
}

/// A directory of `.dmc` files, for the mixed-suite case.
struct Suite {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Suite {
    /// `files` is `(basename, source)`. `dmc test` walks the directory in
    /// sorted order.
    fn new(files: &[(&str, &str)]) -> Suite {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, src) in files {
            std::fs::write(dir.path().join(name), src).expect("write suite file");
        }
        let path = dir.path().to_path_buf();
        Suite { _dir: dir, path }
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(dmc_binary())
            .arg("test")
            .args(args)
            .arg(&self.path)
            .output()
            .expect("spawn dmc")
    }
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

// ── Human path ───────────────────────────────────────────────────────────────

#[test]
fn without_the_flag_a_file_with_no_tests_exits_0() {
    let probe = Probe::new(NO_TESTS);
    let out = probe.run(&[]);
    assert!(out.status.success(), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    assert_eq!(stdout(&out), "0 tests\n");
    assert!(
        !stderr(&out).contains("--fail-empty"),
        "the FAILED line appeared without the flag: {}", stderr(&out),
    );
}

#[test]
fn fail_empty_exits_1_on_a_file_with_no_tests() {
    let probe = Probe::new(NO_TESTS);
    let out = probe.run(&["--fail-empty"]);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    // The existing `0 tests` line is unchanged, plus the new FAILED line.
    assert_eq!(stdout(&out), "0 tests\n");
    assert_eq!(stderr(&out), "test result: FAILED. 0 tests ran (--fail-empty)\n");
}

#[test]
fn fail_empty_does_not_blame_the_flag_for_a_parse_failure() {
    // A file that fails to parse is a genuine file-level failure — `total`
    // stays 0 (no `test_*` ever ran) but `failed` is nonzero. The run must
    // still exit 1, on the strength of the real `FAIL …` line alone, without
    // also printing the `--fail-empty` line: that would misattribute the
    // exit to the flag when a parse error already earned it.
    let probe = Probe::new(PARSE_FAILURE);
    let out = probe.run(&["--fail-empty"]);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    assert_eq!(stdout(&out), "0 tests\n");
    let err = stderr(&out);
    assert!(err.contains("FAIL") && err.contains("resolution failed"), "{}", err);
    assert!(
        !err.contains("--fail-empty"),
        "the flag line must not appear alongside a real file-level failure: {}", err,
    );
}

#[test]
fn fail_empty_does_not_trigger_when_the_path_has_at_least_one_test() {
    let probe = Probe::new(ONE_TEST);
    let out = probe.run(&["--fail-empty"]);
    assert!(out.status.success(), "a nonempty suite failed under --fail-empty:\n{}", stderr(&out));
    assert!(stdout(&out).contains("test result: ok. 1 passed"), "{}", stdout(&out));
    assert!(!stderr(&out).contains("--fail-empty"), "{}", stderr(&out));
}

#[test]
fn fail_empty_does_not_trigger_on_a_directory_where_only_some_files_have_tests() {
    // AGENTS.md's directory-sweep case: one file test-free, one with a real
    // test. The total across the directory is nonzero, so --fail-empty must
    // not fire even though one file individually contributed zero tests.
    let suite = Suite::new(&[("no_tests.dmc", NO_TESTS), ("has_tests.dmc", ONE_TEST)]);
    let out = suite.run(&["--fail-empty"]);
    assert!(out.status.success(), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    assert!(stdout(&out).contains("test_add"), "{}", stdout(&out));
    assert!(stdout(&out).contains("test result: ok. 1 passed"), "{}", stdout(&out));
    assert!(!stderr(&out).contains("--fail-empty"), "{}", stderr(&out));
}

#[test]
fn fail_empty_also_exits_1_on_an_empty_directory_sweep() {
    // A directory where every file is test-free: the total is still zero.
    let suite = Suite::new(&[("a.dmc", NO_TESTS), ("b.dmc", NO_TESTS)]);
    let out = suite.run(&["--fail-empty"]);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    assert_eq!(stdout(&out), "0 tests\n");
    assert_eq!(stderr(&out), "test result: FAILED. 0 tests ran (--fail-empty)\n");
}

#[test]
fn fail_empty_composes_with_jit() {
    let probe = Probe::new(NO_TESTS);
    let out = probe.run(&["--fail-empty", "--jit"]);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    assert_eq!(stdout(&out), "0 tests\n");
}

// ── `--json` ─────────────────────────────────────────────────────────────────

#[test]
fn fail_empty_with_json_reports_tests_ran_zero_and_exit_1() {
    let probe = Probe::new(NO_TESTS);
    let out = probe.run(&["--fail-empty", "--json"]);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    assert_eq!(stdout(&out), "", "the human line must not leak into --json mode");
    assert_eq!(
        stderr(&out),
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"test\",\"status\":\"failed\",\
         \"errors\":0,\"warnings\":0,\"passed\":0,\"failed\":0,\"tests_ran\":0,\"exit\":1}\n",
    );
}

#[test]
fn without_fail_empty_json_still_reports_tests_ran_zero_but_exit_0() {
    // The flag changes only the exit code the zero-tests branch reports —
    // `tests_ran` itself is unconditional, so a consumer can already see the
    // count either way; only the CI-relevant field (`exit`) moves.
    let probe = Probe::new(NO_TESTS);
    let out = probe.run(&["--json"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(
        stderr(&out),
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"test\",\"status\":\"ok\",\
         \"errors\":0,\"warnings\":0,\"passed\":0,\"failed\":0,\"tests_ran\":0,\"exit\":0}\n",
    );
}

#[test]
fn fail_empty_with_json_on_a_nonempty_suite_is_a_well_formed_summary_that_still_exits_0() {
    let probe = Probe::new(ONE_TEST);
    let out = probe.run(&["--fail-empty", "--json"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    let last_line = err.trim_end().lines().last().expect("at least one line").to_string();
    assert!(last_line.starts_with("{\"schema\":1,\"kind\":\"summary\","), "{}", last_line);
    assert!(last_line.contains("\"tests_ran\":1"), "{}", last_line);
    assert!(last_line.contains("\"exit\":0"), "{}", last_line);
}
