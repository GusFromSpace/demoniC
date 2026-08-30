//! End-to-end coverage of the arena sizing flags (`MEMORY.md §1.1`, #400).
//!
//! The unit tests in `src/arena_tests.rs` cover parsing and `src/interp_tests.rs`
//! the metering; this file drives the actual `dmc` binary, because the thing
//! being tested is a command line: a size reaches the arena, a bad size is
//! rejected before the program starts, and a tiny budget changes what the
//! program does.

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

/// A program that makes one 2 MiB Forge allocation (512×512 f64 in the
/// interpreter, f32 under the JIT) and prints a marker if it survives.
const ALLOCATES: &str = r#"
fn main() -> i64 {
    let t = forge.zeros[f32, [512, 512]]
    print("allocated")
    0
}
"#;

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
        let mut cmd = Command::new(dmc_binary());
        cmd.arg(args[0]).arg(&self.path).args(&args[1..]);
        cmd.output().expect("spawn dmc")
    }
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

#[test]
fn the_flags_are_accepted() {
    let probe = Probe::new(ALLOCATES);
    let out = probe.run(&["run", "--vault=16G", "--forge=2G"]);
    assert!(out.status.success(), "flags rejected:\n{}", stderr(&out));
    assert!(stdout(&out).contains("allocated"), "{}", stdout(&out));
}

#[test]
fn a_separated_value_is_accepted_too() {
    let probe = Probe::new(ALLOCATES);
    let out = probe.run(&["run", "--forge", "2G"]);
    assert!(out.status.success(), "`--forge 2G` rejected:\n{}", stderr(&out));
    assert!(stdout(&out).contains("allocated"));
}

#[test]
fn a_tiny_forge_budget_changes_behavior() {
    // The load-bearing end-to-end check: the same program that succeeds under
    // a 2 GiB budget fails under a 1 KiB one, with the documented
    // arena-exhaustion error and a nonzero exit.
    let probe = Probe::new(ALLOCATES);
    let out = probe.run(&["run", "--forge=1K"]);
    assert!(!out.status.success(), "a 1 KiB forge allocated 2 MiB");
    let err = stderr(&out);
    assert!(err.contains("forge arena exhausted"), "{}", err);
    assert!(err.contains("--forge"), "{}", err);
    assert!(!stdout(&out).contains("allocated"), "the program ran on past the error");
}

#[test]
fn the_jit_honors_the_forge_budget() {
    let probe = Probe::new(ALLOCATES);
    let ok = probe.run(&["jit", "--forge=64M"]);
    assert!(ok.status.success(), "jit rejected a sufficient budget:\n{}", stderr(&ok));

    let out = probe.run(&["jit", "--forge=1K"]);
    assert!(!out.status.success(), "a 1 KiB forge allocated a 1 MiB tensor under the JIT");
    // The whole message, not a substring: the numbers have to add up, and the
    // wording has to be the interpreter's (`MEMORY.md §1.1`). The JIT stores
    // this tensor as f32, so it needs 1 MiB where `dmc run` would need 2 MiB.
    assert_eq!(
        stderr(&out).trim(),
        "runtime error: forge arena exhausted: this allocation needs 1 MiB, \
         but only 1 KiB of the 1 KiB --forge budget is free",
    );
}

#[test]
fn the_two_backends_word_exhaustion_the_same_way() {
    // `MEMORY.md §1.1` promises the same diagnostic from both backends, with
    // the interpreter adding the source location the JIT's allocation callback
    // does not have. Anything less makes the shipped doc a lie.
    let probe = Probe::new(ALLOCATES);
    let interp = stderr(&probe.run(&["run", "--forge=1K"]));
    let jit = stderr(&probe.run(&["jit", "--forge=1K"]));

    let body = |line: &str| -> String {
        let line = line.trim();
        // Strip `runtime error at L:C: ` / `runtime error: `.
        match line.find(": ") {
            Some(i) => line[i + 2..].to_string(),
            None => line.to_string(),
        }
    };
    assert!(interp.trim().starts_with("runtime error at "), "{}", interp);
    assert!(jit.trim().starts_with("runtime error: "), "{}", jit);
    // Same sentence modulo the element width each backend actually stores.
    assert_eq!(
        body(&interp).replace("needs 2 MiB", "needs N"),
        body(&jit).replace("needs 1 MiB", "needs N"),
        "interp: {}\njit: {}", interp, jit,
    );
}

/// A directory of `dmc test` files, written for `dmc test --jit`.
struct Suite {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Suite {
    /// `files` is `(basename, source)`. `dmc test` walks the directory in
    /// sorted order, so the names fix the run order.
    fn new(files: &[(&str, String)]) -> Suite {
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
            .arg(&self.path)
            .args(args)
            .output()
            .expect("spawn dmc")
    }
}

/// A test file whose single test allocates `side`² f32 in Forge.
fn alloc_test(n: usize, side: usize) -> (String, String) {
    (
        format!("t{}.dmc", n),
        format!(
            "fn test_alloc_{n}() -> bool {{\n    \
             let t = forge.zeros[f32, [{side}, {side}]]\n    \
             true\n}}\n"
        ),
    )
}

#[test]
fn the_forge_budget_is_per_file_under_dmc_test_jit() {
    // #400: the JIT's arena is a thread-local created once per process, so
    // `committed` used to climb across the whole suite. Four files that each
    // fit the budget on their own were run as if they shared it: two passed,
    // the third exited the process from inside the allocation callback, and
    // the fourth never ran. Each file must get the whole budget, exactly as
    // the interpreter does (`dmc test` builds a fresh `Interpreter` per test).
    let files: Vec<(String, String)> = (1..=4).map(|n| alloc_test(n, 1024)).collect();
    let refs: Vec<(&str, String)> =
        files.iter().map(|(n, s)| (n.as_str(), s.clone())).collect();
    let suite = Suite::new(&refs);

    let out = suite.run(&["--jit", "--forge=12M"]);
    let (o, e) = (stdout(&out), stderr(&out));
    assert!(out.status.success(), "stdout:\n{}\nstderr:\n{}", o, e);
    for n in 1..=4 {
        assert!(o.contains(&format!("test_alloc_{}", n)), "file {} did not run:\n{}", n, o);
    }
    assert!(o.contains("test result: ok. 4 passed"), "{}", o);
    assert!(!e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn one_over_budget_file_fails_without_killing_the_run() {
    // The other half: a file that genuinely does not fit must be *that file's*
    // FAIL — with the diagnostic, a printed summary, and a nonzero exit — and
    // the files after it must still run. Previously this was
    // `std::process::exit(1)` from the allocator: no FAIL line, no summary,
    // and everything downstream silently skipped.
    let big = (
        "t2_big.dmc".to_string(),
        "fn test_too_big() -> bool {\n    \
         let t = forge.zeros[f32, [2048, 2048]]\n    \
         true\n}\n"
            .to_string(),
    );
    let files = [alloc_test(1, 1024), big, alloc_test(3, 1024)];
    let refs: Vec<(&str, String)> =
        files.iter().map(|(n, s)| (n.as_str(), s.clone())).collect();
    let suite = Suite::new(&refs);

    let out = suite.run(&["--jit", "--forge=12M"]);
    let (o, e) = (stdout(&out), stderr(&out));

    assert!(!out.status.success(), "an over-budget file exited 0:\n{}\n{}", o, e);
    assert!(
        e.contains("FAIL") && e.contains("test_too_big") && e.contains("[jit]"),
        "no jit FAIL line for the over-budget file:\n{}", e,
    );
    assert!(e.contains("forge arena exhausted"), "the FAIL carried no diagnostic:\n{}", e);
    assert!(e.contains("test result: FAILED"), "no summary was printed:\n{}", e);
    // The file *after* the failure still ran, and the one before it still passed.
    assert!(o.contains("test_alloc_1"), "the file before the failure did not run:\n{}", o);
    assert!(o.contains("test_alloc_3"), "the file after the failure did not run:\n{}", o);
}

#[test]
fn the_jit_refuses_a_vault_budget_rather_than_ignoring_it() {
    // The JIT lowers `vault.*` into the same Forge arena, so there is no Vault
    // to size. That must be a diagnostic, never a silent no-op.
    let probe = Probe::new(ALLOCATES);
    let out = probe.run(&["jit", "--vault=1G"]);
    assert!(!out.status.success(), "`--vault` was accepted under the JIT");
    let err = stderr(&out);
    assert!(err.contains("--vault"), "{}", err);
    assert!(err.contains("JIT"), "{}", err);
}

#[test]
fn a_vault_budget_bounds_vault_allocations_under_run() {
    let probe = Probe::new(r#"
fn main() -> i64 {
    let t = vault.zeros[f32, [512, 512]]
    print("allocated")
    0
}
"#);
    let ok = probe.run(&["run", "--vault=64M"]);
    assert!(ok.status.success(), "{}", stderr(&ok));

    let out = probe.run(&["run", "--vault=1K"]);
    assert!(!out.status.success(), "a 1 KiB vault allocated 2 MiB");
    assert!(stderr(&out).contains("vault arena exhausted"), "{}", stderr(&out));
}

#[test]
fn invalid_sizes_are_rejected_before_the_program_runs() {
    let probe = Probe::new(ALLOCATES);
    for (flag, needle) in [
        ("--forge=0", "zero"),
        ("--vault=0", "zero"),
        ("--forge=banana", "not a size"),
        ("--forge=", "expected a size"),
        ("--forge=-1", "not a size"),
        ("--forge=1.5G", "not a size unit"),
        ("--forge=16Q", "not a size unit"),
        ("--forge=16777216T", "overflows"),
    ] {
        let out = probe.run(&["run", flag]);
        assert!(!out.status.success(), "`{}` was accepted", flag);
        let err = stderr(&out);
        assert!(err.contains(needle), "`{}` said: {}", flag, err);
        assert!(
            !stdout(&out).contains("allocated"),
            "`{}` let the program run", flag
        );
    }
}

#[test]
fn a_flag_with_no_value_says_so() {
    let probe = Probe::new(ALLOCATES);
    let out = probe.run(&["run", "--forge"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("needs a size"), "{}", stderr(&out));
}

#[test]
fn the_flags_do_not_leak_into_the_program_argv() {
    // Pre-#400 these were swallowed into the program's `argv()`. They are
    // `dmc` flags; the program must not see them.
    let probe = Probe::new(r#"
fn main() -> i64 {
    print(list_len(argv()))
    0
}
"#);
    let bare = probe.run(&["run"]);
    let sized = probe.run(&["run", "--forge=2G", "--vault=4G"]);
    assert!(sized.status.success(), "{}", stderr(&sized));
    assert_eq!(stdout(&bare), stdout(&sized), "sizing flags reached argv()");
}
