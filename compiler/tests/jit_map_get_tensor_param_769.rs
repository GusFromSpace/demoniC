//! #769: `dmc jit` on a `map_get` result passed to a `Tensor`-typed
//! parameter segfaulted (exit 139, no diagnostic) instead of refusing.
//! Found by the #764 edit pass (2026-09-06), reduced on the #764 head
//! (781d6bf).
//!
//! Root cause: `coerce_to` (`src/jit.rs`) unconditionally decoded an
//! `OptI64` source (a `map_get` result — always opaque `i64` in the JIT,
//! #721) and recursed with a plain `i64` source. For a genuinely scalar
//! target that recursion is correct (`coerce_scalar` never treats the
//! payload as anything but a number), but for `Tensor` it lands on
//! `(Scalar(I64), Tensor(_)) => Ok(v)` — a zero-cost bitcast kept for
//! legitimate raw-pointer model-field initialization (`field: 0`). Handed
//! a map's plain `i64` payload instead, that arm turns it into a "tensor"
//! whose data pointer is whatever small integer the map held; `sum` then
//! dereferences it.
//!
//! Fix: `coerce_to` now refuses, via the same `jit-conversion`-tagged
//! `opt_i64_refusal` helper #760/#764 built for this exact
//! "unverifiable past the opaque payload" shape, before ever recursing
//! into a pointer-carrying target (`Tensor`, `Bf16Tensor`, `KV`, `KvArray`,
//! `DynTensor`, `TritTensor`, `Model`, `ModelArray`, `Str`, `Port`, `Map`,
//! `Tuple`, and a payload-carrying `Enum`) — everything except the
//! documented `Fn` exception (an opaque i64 is deliberately accepted there
//! as a code pointer) and `Budget` (documented as never dereferenced).
//!
//! `compiler/src/jit.rs`'s own unit tests (`jit_769_*`) cover the
//! conversion directly against several of those target shapes; this file
//! drives the real `dmc` binary end to end, the way `jit_map_get_695.rs`
//! does for its own issue, to pin the actual process exit code (139 before
//! the fix, not merely "some Err returned somewhere").

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
}

fn lines(out: &Output) -> Vec<String> {
    let e = String::from_utf8_lossy(&out.stderr).to_string();
    if e.is_empty() { return Vec::new(); }
    e.trim_end_matches('\n').split('\n').map(|s| s.to_string()).collect()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn extract_field(line: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", field);
    let start = line.find(&needle)? + needle.len();
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

fn kinds_and_codes(out: &Output) -> Vec<(String, Option<String>)> {
    lines(out).iter().filter_map(|l| {
        let kind = extract_field(l, "kind")?;
        if kind == "summary" { return None; }
        let code = extract_field(l, "code");
        Some((kind, code))
    }).collect()
}

// The issue's own repro, verbatim.
const REPRO: &str = "fn f(t: Tensor[f32, [2]]) -> f32 { sum(t) }\n\
                      fn main() -> f32 {\n\
                      \x20   let d = map()\n\
                      \x20   map_set(d, \"k\", 3)\n\
                      \x20   let x = map_get(d, \"k\")\n\
                      \x20   f(x)\n\
                      }\n";

#[test]
fn dmc_run_answers_the_scalar_map_get_wrote() {
    // `dmc --check`: OK (`map_get`'s result is `Unknown`, so the `Tensor`
    // param accepts it — #721's typed-map-values gap; unrelated to this fix
    // and not touched here).
    let probe = Probe::new(REPRO);
    let chk = probe.run(&["--check"]);
    assert!(chk.status.success(), "--check should still pass: {}", String::from_utf8_lossy(&chk.stderr));

    // `dmc run`: the interpreter hands the plain int straight to `sum`,
    // which answers it back — its own oddity, out of scope for this issue
    // (filed separately as #860).
    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(stdout(&run), "\n=> 3\n");
}

#[test]
fn dmc_jit_no_longer_segfaults_it_refuses_with_a_coded_conversion() {
    let probe = Probe::new(REPRO);
    let jit = probe.run(&["jit"]);

    // Before the fix: exit 139 (SIGSEGV). A signal-killed process reports
    // no exit code on unix (`status.code()` is `None`); assert the process
    // terminated normally with a nonzero *code* instead, which a segfault
    // can never produce.
    assert!(!jit.status.success(), "expected dmc jit to refuse, not succeed");
    assert_eq!(jit.status.code(), Some(1),
        "expected an ordinary refusal exit code, not a signal death (139 pre-fix): status={:?}, stderr={}",
        jit.status, String::from_utf8_lossy(&jit.stderr));

    let stderr = String::from_utf8_lossy(&jit.stderr);
    assert!(stderr.contains("cannot convert `i64?` to `Tensor[f32, [2]]`"), "{}", stderr);
    assert!(stderr.contains("#721"), "{}", stderr);
    // Located: a `line:col` in the diagnostic.
    assert!(stderr.contains(":5:") || stderr.contains("6:5"),
        "expected a located line:col in the refusal: {}", stderr);
}

#[test]
fn dmc_jit_json_reports_one_classed_jit_conversion_refusal() {
    let probe = Probe::new(REPRO);
    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "expected exactly one classed jit-conversion refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));
}
