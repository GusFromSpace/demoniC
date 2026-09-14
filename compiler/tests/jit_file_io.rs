//! `read_file`/`write_file`/`append_file`/`file_exists` now lower under `dmc
//! jit` as host calls into the same `std::fs` operations the interpreter
//! uses (`interp.rs` ~10098-10168, `jit.rs`'s `lower_builtin_{read,write,
//! append}_file`/`lower_builtin_file_exists` and their `__dmc_*` runtime
//! functions). This file pins backend parity: identical stdout, identical
//! on-disk bytes, and identical `err` text for a missing file, on both
//! `dmc run` and `dmc jit`.
//!
//! `read_bytes` is deliberately NOT covered here as a lowering — its
//! `Tensor[i64,[N]]` result needs a runtime-extent tensor, and the JIT has
//! one (`TyKind::DynTensor`), but it's reachable only from KV reads and
//! isn't plumbed for an i64-element host-call result.
//! `read_bytes_still_refuses_under_jit` below is a regression guard: it must
//! keep refusing with the existing, located `jit-unknown-fn` class, never
//! fall silently quiet nor regress to an uncoded `jit-error`.
//!
//! All file targets live under `std::env::temp_dir()`, named with the test's
//! own identity plus the process id so parallel `cargo test` runs (this
//! binary's tests execute concurrently by default) never collide.

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

/// The printed output only, with `dmc run`'s extra blank line before the
/// final `=> <value>` summary (a pre-existing CLI-driver quirk, unrelated to
/// file I/O — see `jit_map_get_695.rs`'s `answer()`) normalized away, so a
/// run-vs-jit comparison is about the program's own output, not that quirk.
fn printed(out: &str) -> &str {
    let idx = out.rfind("=> ").expect("no `=> ` in output");
    out[..idx].trim_end_matches('\n')
}

fn kinds_and_codes(out: &Output) -> Vec<(String, Option<String>)> {
    lines(out).iter().filter_map(|l| {
        let kind = extract_field(l, "kind")?;
        if kind == "summary" { return None; }
        let code = extract_field(l, "code");
        Some((kind, code))
    }).collect()
}

fn extract_field(line: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", field);
    let start = line.find(&needle)? + needle.len();
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

/// A scratch file path under `std::env::temp_dir()`, unique to this test
/// process and this call site — `dmc run`/`dmc jit` each write real bytes to
/// disk here, so a shared path across parallel tests would race.
fn scratch_path(tag: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "dmc_jit_file_io_{}_{}_{}.txt", std::process::id(), tag, n))
}

/// Escape a path for embedding in a demoniC string literal — the paths this
/// file builds never contain a `"` or `\`, but escaping defensively costs
/// nothing and keeps the source valid if that ever changes.
fn dmc_str_lit(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

// ─── write_file / read_file round trip ──────────────────────────────────────

#[test]
fn write_then_read_round_trips_identically_on_both_backends() {
    let run_path = scratch_path("roundtrip_run");
    let jit_path = scratch_path("roundtrip_jit");
    let _ = std::fs::remove_file(&run_path);
    let _ = std::fs::remove_file(&jit_path);

    let src = |path: &PathBuf| format!(
        "fn main() -> i64 {{\n\
        \x20   let (wok, werr) = write_file({path}, \"hello demoniC\\n\")\n\
        \x20   let (content, rerr) = read_file({path})\n\
        \x20   print(content)\n\
        \x20   print(wok)\n\
        \x20   print(werr)\n\
        \x20   print(rerr)\n\
        \x20   0\n\
        }}\n",
        path = dmc_str_lit(&path.to_string_lossy()),
    );

    let run_probe = Probe::new(&src(&run_path));
    let run = run_probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let jit_probe = Probe::new(&src(&jit_path));
    let jit = jit_probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit.stderr));

    // Both backends print the same thing: the content, then `nil` three
    // times (wok/werr/rerr are all nil on the success path — write_file
    // never returns a bool, per interp.rs:10140).
    assert_eq!(printed(&stdout(&run)), printed(&stdout(&jit)));
    assert_eq!(printed(&stdout(&run)), "hello demoniC\n\nnil\nnil\nnil");

    // And the bytes actually landed on disk, identically, under each path.
    let run_bytes = std::fs::read(&run_path).expect("run_path exists");
    let jit_bytes = std::fs::read(&jit_path).expect("jit_path exists");
    assert_eq!(run_bytes, jit_bytes);
    assert_eq!(run_bytes, b"hello demoniC\n");

    let _ = std::fs::remove_file(&run_path);
    let _ = std::fs::remove_file(&jit_path);
}

// ─── append_file ─────────────────────────────────────────────────────────────

#[test]
fn append_file_round_trips_identically_on_both_backends() {
    let run_path = scratch_path("append_run");
    let jit_path = scratch_path("append_jit");
    let _ = std::fs::remove_file(&run_path);
    let _ = std::fs::remove_file(&jit_path);

    let src = |path: &PathBuf| format!(
        "fn main() -> i64 {{\n\
        \x20   let (_a, aerr1) = append_file({path}, \"one\\n\")\n\
        \x20   let (_b, aerr2) = append_file({path}, \"two\\n\")\n\
        \x20   let (content, rerr) = read_file({path})\n\
        \x20   print(content)\n\
        \x20   print(aerr1)\n\
        \x20   print(aerr2)\n\
        \x20   print(rerr)\n\
        \x20   0\n\
        }}\n",
        path = dmc_str_lit(&path.to_string_lossy()),
    );

    let run_probe = Probe::new(&src(&run_path));
    let run = run_probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let jit_probe = Probe::new(&src(&jit_path));
    let jit = jit_probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit.stderr));

    assert_eq!(printed(&stdout(&run)), printed(&stdout(&jit)));
    assert_eq!(printed(&stdout(&run)), "one\ntwo\n\nnil\nnil\nnil");

    let run_bytes = std::fs::read(&run_path).expect("run_path exists");
    let jit_bytes = std::fs::read(&jit_path).expect("jit_path exists");
    assert_eq!(run_bytes, jit_bytes);
    assert_eq!(run_bytes, b"one\ntwo\n");

    // append_file also CREATES the file when it does not exist yet
    // (interp.rs:10154's `.create(true)`) — pin that on the JIT side too,
    // on a path that has never been written.
    let create_path = scratch_path("append_creates");
    let _ = std::fs::remove_file(&create_path);
    let src2 = format!(
        "fn main() -> i64 {{\n\
        \x20   let (_a, err) = append_file({path}, \"fresh\")\n\
        \x20   print(err)\n\
        \x20   file_exists({path}) as i64\n\
        }}\n",
        path = dmc_str_lit(&create_path.to_string_lossy()),
    );
    let probe2 = Probe::new(&src2);
    let jit2 = probe2.run(&["jit"]);
    assert!(jit2.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit2.stderr));
    assert_eq!(stdout(&jit2), "nil\n=> 1\n");
    assert_eq!(std::fs::read(&create_path).expect("created"), b"fresh");

    let _ = std::fs::remove_file(&run_path);
    let _ = std::fs::remove_file(&jit_path);
    let _ = std::fs::remove_file(&create_path);
}

// ─── missing-file error parity ───────────────────────────────────────────────

#[test]
fn read_file_missing_path_gives_identical_err_text_on_both_backends() {
    let missing = scratch_path("definitely_missing");
    let _ = std::fs::remove_file(&missing); // must not exist

    let src = format!(
        "fn main() -> i64 {{\n\
        \x20   let (content, err) = read_file({path})\n\
        \x20   print(content)\n\
        \x20   print(err)\n\
        \x20   0\n\
        }}\n",
        path = dmc_str_lit(&missing.to_string_lossy()),
    );

    let run_probe = Probe::new(&src);
    let run = run_probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let jit_probe = Probe::new(&src);
    let jit = jit_probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit.stderr));

    assert_eq!(printed(&stdout(&run)), printed(&stdout(&jit)),
        "read_file's err text on a missing path must be byte-identical on both backends");
    // `content` is nil (line 1), and line 2 is `e.to_string()` for a
    // `std::io::Error` — non-empty, and NOT the literal word "nil".
    let out = stdout(&run);
    let mut ls = out.lines();
    assert_eq!(ls.next(), Some("nil"));
    let err_line = ls.next().expect("err line");
    assert!(!err_line.is_empty() && err_line != "nil",
        "expected a real io error message, got: {:?}", err_line);
}

#[test]
fn write_file_unwritable_path_gives_identical_err_text_on_both_backends() {
    // A path inside a directory that does not exist — ENOENT on the parent,
    // not a permissions question, so it is portable across CI machines.
    let bogus = std::env::temp_dir()
        .join(format!("dmc_jit_file_io_{}_no_such_dir", std::process::id()))
        .join("out.txt");
    let src = format!(
        "fn main() -> i64 {{\n\
        \x20   let (ok, err) = write_file({path}, \"x\")\n\
        \x20   print(ok)\n\
        \x20   print(err)\n\
        \x20   0\n\
        }}\n",
        path = dmc_str_lit(&bogus.to_string_lossy()),
    );

    let run_probe = Probe::new(&src);
    let run = run_probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let jit_probe = Probe::new(&src);
    let jit = jit_probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit.stderr));

    assert_eq!(printed(&stdout(&run)), printed(&stdout(&jit)),
        "write_file's err text on an unwritable path must be byte-identical on both backends");
    let out = stdout(&run);
    let mut ls = out.lines();
    assert_eq!(ls.next(), Some("nil"), "the ok half is always nil, success or failure");
    let err_line = ls.next().expect("err line");
    assert!(!err_line.is_empty() && err_line != "nil",
        "expected a real io error message, got: {:?}", err_line);
}

// ─── file_exists ──────────────────────────────────────────────────────────────

#[test]
fn file_exists_agrees_on_both_backends_present_and_absent() {
    let present = scratch_path("exists_present");
    std::fs::write(&present, b"x").expect("seed file");
    let absent = scratch_path("exists_absent");
    let _ = std::fs::remove_file(&absent);

    let src = format!(
        "fn main() -> i64 {{\n\
        \x20   print(file_exists({present}))\n\
        \x20   print(file_exists({absent}))\n\
        \x20   0\n\
        }}\n",
        present = dmc_str_lit(&present.to_string_lossy()),
        absent = dmc_str_lit(&absent.to_string_lossy()),
    );

    let run_probe = Probe::new(&src);
    let run = run_probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let jit_probe = Probe::new(&src);
    let jit = jit_probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit.stderr));

    assert_eq!(printed(&stdout(&run)), printed(&stdout(&jit)));
    assert_eq!(printed(&stdout(&run)), "true\nfalse");

    let _ = std::fs::remove_file(&present);
}

#[test]
fn file_exists_used_as_a_branch_condition_under_jit() {
    // Exercises the Cranelift-side `icmp`-narrowed bool (ScalarKind::Bool is
    // cl::I8, not the runtime call's raw i64 — see lower_builtin_file_exists)
    // in an actual conditional, not just a `print`.
    let present = scratch_path("exists_branch");
    std::fs::write(&present, b"x").expect("seed file");

    let src = format!(
        "fn main() -> i64 {{\n\
        \x20   if file_exists({present}) {{ 1 }} else {{ 0 }}\n\
        }}\n",
        present = dmc_str_lit(&present.to_string_lossy()),
    );
    let probe = Probe::new(&src);
    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit.stderr));
    assert_eq!(stdout(&jit), "=> 1\n");

    let _ = std::fs::remove_file(&present);
}

// ─── read_bytes: regression guard, stays refused ─────────────────────────────

#[test]
fn read_bytes_still_refuses_under_jit_with_the_existing_class() {
    // read_bytes's Tensor[i64,[N]] result would need TyKind::DynTensor, which
    // exists but is reachable only from KV reads and isn't plumbed for an
    // i64-element host-call result — it must keep refusing as the existing
    // `jit-unknown-fn` class, not fall through to an uncoded `jit-error`
    // and not silently start "working" with a wrong shape.
    let src = "fn main() -> i64 {\n\
               \x20   let (b, err) = read_bytes(\"whatever\")\n\
               \x20   0\n\
               }\n";
    let probe = Probe::new(src);

    let jit = probe.run(&["jit"]);
    assert!(!jit.status.success(), "read_bytes must still refuse under dmc jit");

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-unknown-fn".to_string()))],
        "expected the existing located jit-unknown-fn refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));

    // dmc run still has the full semantics.
    let run = probe.run(&["run"]);
    // `whatever` doesn't exist as a file relative to the probe's tempdir, so
    // this is expected to succeed and bind an err — it must NOT be refused
    // the way the JIT refuses it.
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
}
