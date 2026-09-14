//! `dmc test <dir>`: a file-level failure must not zero out a different
//! file's passing tests (#854).
//!
//! Before this fix, a directory-level `total`/`failed` tally shared one
//! `failed` counter between per-test failures (which are counted into
//! `total`) and file-level failures (a parse/resolve/canonicalization/
//! dependency-check error, which never reach `total` at all — the file's
//! tests are never even collected). Because the summary's "passed" number
//! was computed as `total - failed`, one bad file in the directory silently
//! zeroed out every `ok` a *different*, successfully-resolved file had
//! printed: `0 passed; 1 failed` even though a test elsewhere printed `ok`.
//!
//! The fix tracks file-level failures separately and folds them back in only
//! for the printed/exit-code "failed" count, never for the subtraction that
//! yields "passed".

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

/// A program with one passing `test_*`.
const GOOD: &str = "fn test_ok() -> bool {\n    true\n}\n";

/// A program that fails to parse — a genuine file-level failure.
const BAD: &str = "fn broken(-> bool {\n    true\n}\n";

/// A program with no `test_*` functions at all.
const NO_TESTS: &str = "fn main() -> i64 {\n    let x: i64 = 41\n    x + 1\n}\n";

/// A directory of `.dmc` files. `dmc test` walks it in sorted order.
struct Suite {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Suite {
    /// `files` is `(basename, source)`.
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

#[test]
fn a_parse_failure_in_one_file_does_not_zero_a_passing_test_in_another() {
    let suite = Suite::new(&[("bad.dmc", BAD), ("good.dmc", GOOD)]);
    let out = suite.run(&[]);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    assert!(stdout(&out).contains("ok   "), "the passing test's `ok` line went missing:\n{}", stdout(&out));
    assert!(stdout(&out).contains("::test_ok"), "{}", stdout(&out));
    let err = stderr(&out);
    assert!(err.contains("FAIL") && err.contains("resolution failed"), "{}", err);
    assert!(
        err.contains("test result: FAILED. 1 passed; 1 failed"),
        "expected the good file's test to count as passed:\n{}", err,
    );
}

#[test]
fn the_jit_run_keeps_the_same_tally() {
    let suite = Suite::new(&[("bad.dmc", BAD), ("good.dmc", GOOD)]);
    let out = suite.run(&["--jit"]);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    assert!(stdout(&out).contains("ok   "), "{}", stdout(&out));
    let err = stderr(&out);
    assert!(
        err.contains("test result: FAILED. 1 passed; 1 failed"),
        "expected the same 1 passed; 1 failed tally under --jit:\n{}", err,
    );
}

#[test]
fn an_all_good_directory_is_unaffected() {
    // No file-level failure anywhere: the summary is byte-identical to
    // before this fix — this is the case `tools/golden` pins.
    let suite = Suite::new(&[("good.dmc", GOOD)]);
    let out = suite.run(&[]);
    assert!(out.status.success(), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    assert_eq!(stderr(&out), "");
    let label = suite.path.join("good.dmc").display().to_string();
    assert_eq!(
        stdout(&out),
        format!("ok   {}::test_ok\ntest result: ok. 1 passed; 0 failed\n", label),
    );
}

#[test]
fn an_empty_suite_still_fails_under_fail_empty_854() {
    // #599: a directory where every file is test-free still counts as zero
    // tests overall, and `--fail-empty` still turns that into a failure —
    // untouched by separating the file-level-failure counter out.
    let suite = Suite::new(&[("a.dmc", NO_TESTS), ("b.dmc", NO_TESTS)]);
    let out = suite.run(&["--fail-empty"]);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    assert_eq!(stdout(&out), "0 tests\n");
    assert_eq!(stderr(&out), "test result: FAILED. 0 tests ran (--fail-empty)\n");
}

#[test]
fn json_output_carries_the_same_counts() {
    let suite = Suite::new(&[("bad.dmc", BAD), ("good.dmc", GOOD)]);
    let out = suite.run(&["--json"]);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}\nstderr:\n{}", stdout(&out), stderr(&out));
    let err = stderr(&out);
    let last_line = err.trim_end().lines().last().expect("at least one line").to_string();
    assert!(last_line.starts_with("{\"schema\":1,\"kind\":\"summary\","), "{}", last_line);
    assert!(last_line.contains("\"passed\":1"), "{}", last_line);
    assert!(last_line.contains("\"failed\":1"), "{}", last_line);
    assert!(last_line.contains("\"tests_ran\":1"), "{}", last_line);
    assert!(last_line.contains("\"exit\":1"), "{}", last_line);
}
