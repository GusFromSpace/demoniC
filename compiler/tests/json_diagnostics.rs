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

    /// A probe with a sibling module beside it in the same directory — the only
    /// shape a `use` can take, since import paths are relative to the importing
    /// file (`SPEC.md §6.7`).
    fn with_module(src: &str, module_name: &str, module_src: &str) -> Probe {
        let probe = Probe::new(src);
        std::fs::write(probe._dir.path().join(module_name), module_src).expect("write module");
        probe
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(dmc_binary())
            .args(args)
            .arg(&self.path)
            .output()
            .expect("spawn dmc")
    }

    /// `run`, with the probe's *directory* as the argument. `dmc test` takes a
    /// tree, and a multi-file program is only ever a tree.
    fn run_on_dir(&self, args: &[&str]) -> Output {
        Command::new(dmc_binary())
            .args(args)
            .arg(self._dir.path())
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

    /// Write a second file beside `probe.dmc`, for `use` fixtures. Returns
    /// its canonical path, as the diagnostics will report it.
    fn sibling(&self, name: &str, src: &str) -> PathBuf {
        let path = self._dir.path().join(name);
        std::fs::write(&path, src).expect("write sibling");
        path.canonicalize().expect("canonicalize sibling")
    }

    /// The OS's own text for a missing file — `No such file or directory (os
    /// error 2)` on Unix — as it lands verbatim in the resolver's sentence.
    fn missing_file_error(&self, name: &str) -> String {
        std::fs::metadata(self._dir.path().join(name))
            .expect_err("the file must not exist")
            .to_string()
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

/// The `"message"` field's value out of one `--json` diagnostic line,
/// unescaped. Hand-rolled rather than pulled in with a JSON crate: this crate
/// has no JSON dependency, and one field of one well-known shape does not earn
/// one (mirrors `jit_multifile_merge.rs`'s own `message_of` — each integration
/// test file is its own crate, so the two cannot share one definition).
fn message_of(line: &str) -> String {
    let key = "\"message\":\"";
    let Some(start) = line.find(key) else { return String::new() };
    let rest = &line[start + key.len()..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(esc) = chars.next() {
                    out.push(esc);
                }
            }
            '"' => break,
            _ => out.push(c),
        }
    }
    out
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

/// A module beside the probe, exporting one `pub fn` that is squarely inside
/// the JIT's scalar subset — so what the JIT refuses about the importing file
/// is the import itself, not anything the module contains.
const MODULE: &str = "pub fn add(a: i64, b: i64) -> i64 { a + b }\n";

/// Imports that module unqualified. Type-checks clean (the resolver finds the
/// module, the checker threads its `pub` items in), and since the multi-file
/// merge (slice 3) the JIT lowers it too — the two files are merged
/// into one item list before `compile_program` sees them.
const IMPORTS_MODULE: &str = "use \"module.dmc\"\n\nfn main() -> i64 {\n    add(20, 22)\n}\n";

/// The same import carrying one passing test, so `dmc test --jit` has a parity
/// verdict to report on a multi-file file.
const IMPORTS_MODULE_TEST: &str =
    "use \"module.dmc\"\n\nfn test_add() -> bool { add(20, 22) == 42 }\n";

/// A `use` of a file that does not exist, on line 2 so the location is
/// demonstrably the item's and not a default. Resolution fails before any
/// test runs.
const MISSING_IMPORT: &str = "fn helper() -> i64 { 1 }\n\
                              use \"does_not_exist_518.dmc\" as m\n\
                              fn test_never_runs() -> bool { true }\n";

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

/// #684 acceptance (b): three top-level fns, each refused for a different,
/// independent reason — none calls another, so this is exactly three sites,
/// not a cascade. `blocker_a`'s `any` parameter is `jit-type-form` (Pass 1,
/// the signature); `blocker_b`'s range expression is `jit-construct` (Pass 2,
/// the body); `blocker_c`'s named call argument to `helper` is `jit-arg-form`
/// (Pass 2). `helper` itself lowers cleanly, so it contributes no site of its
/// own. Type-checks clean (a `--check` precondition every case below asserts).
const THREE_BLOCKERS: &str = "fn main() -> i64 { 0 }\n\
    fn helper(a: i64, b: i64) -> i64 { a + b }\n\
    fn blocker_a(x: any) -> i64 { 0 }\n\
    fn blocker_b() -> i64 {\n    let r = 0..3\n    0\n}\n\
    fn blocker_c() -> i64 { helper(a = 1, b = 2) }\n";

/// The same program with `blocker_b` (the range expression) removed — the
/// N-1 half of acceptance (b): depth must drop by exactly one, with no other
/// change (same two classes, same two items, in the same order).
const TWO_BLOCKERS: &str = "fn main() -> i64 { 0 }\n\
    fn helper(a: i64, b: i64) -> i64 { a + b }\n\
    fn blocker_a(x: any) -> i64 { 0 }\n\
    fn blocker_c() -> i64 { helper(a = 1, b = 2) }\n";

/// #684 R1's own repro (reviewer's shape, verbatim): `g[N]` is a shape-generic
/// template whose OWN body has an unsupported construct (a range expression);
/// `caller1` calls `g` and ALSO has its own, unrelated blocker (a second range
/// expression) after the call; `caller2` only calls `g`, cleanly.
///
/// Before the R1 fix: `instantiate_template` inserts `g`'s mangled name into
/// `self.fns` (so a caller can resolve the call) BEFORE queuing its body in
/// `self.pending`. When `caller1` is compiled first and its OWN body later
/// fails (the `0..3`), the old code did `self.pending.truncate(pending_before)`
/// — dropping `g`'s just-queued monomorphization while its `self.fns` entry
/// survives. `caller2`, compiled next, resolves its call to `g` against that
/// surviving (but now permanently un-queued) entry and never re-enqueues the
/// body, so `g`'s own blocker is never reported: 1 site instead of 2. Declared
/// in the other order, `caller2` (which pushes the pending monomorphization
/// and never itself fails) runs first, so the truncation in `caller1`'s later
/// failure is a no-op and both sites survive: 2 sites. The two orders must
/// report the SAME site set either way; the fix removes the truncation instead
/// of relying on which order happens to dodge it.
const ORDER_CALLER1_FIRST: &str = "fn main() -> i64 { 0 }\n\
    fn g[N](x: Tensor[f32, [N]]) -> f32 {\n    let z = 0..3\n    sum(x)\n}\n\
    fn caller1() -> f32 {\n    \
        let t = forge.zeros[f32, [3]]\n    \
        let v = g(t)\n    \
        let r = 0..3\n    \
        v\n\
    }\n\
    fn caller2() -> f32 {\n    \
        let t = forge.zeros[f32, [3]]\n    \
        g(t)\n\
    }\n";

/// `ORDER_CALLER1_FIRST` with `caller1` and `caller2` swapped — otherwise
/// byte-identical.
const ORDER_CALLER2_FIRST: &str = "fn main() -> i64 { 0 }\n\
    fn g[N](x: Tensor[f32, [N]]) -> f32 {\n    let z = 0..3\n    sum(x)\n}\n\
    fn caller2() -> f32 {\n    \
        let t = forge.zeros[f32, [3]]\n    \
        g(t)\n\
    }\n\
    fn caller1() -> f32 {\n    \
        let t = forge.zeros[f32, [3]]\n    \
        let v = g(t)\n    \
        let r = 0..3\n    \
        v\n\
    }\n";

/// #684 R2(a): `any_type.dmc`'s own shape — one root refusal on a fn's
/// SIGNATURE (`broken`'s `any` parameter, `jit-type-form`, refused at Pass 1
/// before `self.fns` ever gets an entry for it) with two callers that each
/// reference it by name and so each independently refuse `jit-unknown-fn`
/// ("unknown function `broken`"). Those two are cascades of the one root, not
/// two more root blockers — R2's fix for the "147/258 corpus-wide
/// `jit-unknown-fn` is partly artifact" finding.
const ONE_ROOT_TWO_UNKNOWN_FN_CASCADES: &str = "fn main() -> i64 { 0 }\n\
    fn broken(x: any) -> i64 { 0 }\n\
    fn caller_a() -> i64 { broken(1) }\n\
    fn caller_b() -> i64 { broken(2) }\n";

/// #684 R2(b): a shape-generic `@grad fn` is refused (`jit-grad`,
/// `Unsupported` — `declare_grad_fn`'s shape-params check), so neither its
/// forward nor `$fwd_bwd` entry is ever declared. `caller`'s
/// `lossy.fwd_bwd(...)` then hits `lower_fwd_bwd_destructure`'s
/// `` `lossy` is not a `@grad fn` (no fwd_bwd entry) `` — a `jit-error`
/// (`JitErrorKind::Error`, #480's defect class) manufactured purely because
/// collect mode skipped `lossy`, not a real defect in the program. Before the
/// fix this ABORTED the file right there (28 of 94 corpus files did); after
/// it, it is recognized as a cascade of `lossy`'s root and collection
/// continues past it.
const GRAD_FN_ROOT_AND_FWD_BWD_CASCADE: &str = "fn main() -> i64 { 0 }\n\
    @grad fn lossy[N](!w: Tensor[f32, [N]]) -> f32 { sum(w .* w) }\n\
    fn caller() -> f32 {\n    \
        let !ww = forge.zeros[f32, [3]]\n    \
        let (l, g) = lossy.fwd_bwd(ww)\n    \
        l\n\
    }\n";

/// #684 fixup, F1's own repro: two calls to two DIFFERENT missing builtins,
/// each an independent gap — `isclose` and `clamp` share no root, so they
/// must count as two blockers. Before the fix, `blocker_counts` blanked every
/// backtick-quoted identifier out of the message before comparing (`` unknown
/// function `_` `` for both), so this exact shape landed on `blockers: 1` —
/// the #664 failure mode this fixup exists to close.
const TWO_INDEPENDENT_UNKNOWN_FN_BLOCKERS: &str = "fn main() -> i64 { 0 }\n\
    fn a() -> i64 {\n    \
        isclose(1.0, 1.0)\n    \
        0\n\
    }\n\
    fn b() -> i64 {\n    \
        clamp(1.0, 0.0, 2.0)\n    \
        0\n\
    }\n";

/// #684 fixup, F2's own repro: `F.Named(str)`'s `str` payload refuses the
/// whole enum as `jit-item` (root, item `F`) — `build_enum_payload_layout`
/// never inserts `F` into `enum_payload_layouts`. `fa`/`fb` each go on to
/// construct `F.Named(...)`, which — finding no layout for `F` — falls
/// through the qualified-construction intercept into generic call lowering
/// and reports the callee as the unresolved `` `.Named()` `` method (jit.rs's
/// generic Field+Call fallback, worded for the unrelated `.starts_with` str
/// method it was written for). Before the fix, `cascade_candidate_name`
/// extracted `.Named()` and looked THAT UP in `refused_items` — which holds
/// item names (`F`), never a variant/method name — so it never matched, and
/// `fa`/`fb` each counted as their own root blocker instead of a cascade of
/// `F`'s one root: `blockers: 2` for what is really one gap.
const ENUM_STR_PAYLOAD_ROOT_AND_METHOD_CASCADES: &str = "fn main() -> i64 { 0 }\n\
    enum F {\n    \
        Named(str),\n\
    }\n\
    fn fa() -> i64 {\n    \
        let v = F.Named(\"x\")\n    \
        0\n\
    }\n\
    fn fb() -> i64 {\n    \
        let v = F.Named(\"y\")\n    \
        0\n\
    }\n";

/// #684 fixup, F3's own repro: `run_ret[P](code: Tensor[u8, [P]])` is a
/// shape-generic template; `caller` instantiates it at `run_ret[3](prog)`
/// with a `[u8; 3]` literal, which trips a genuine JIT defect (#480) —
/// `unify_param_type`'s element-type check (jit.rs, `"element type mismatch
/// in template instantiation"`) rejects a match that should hold, a real bug
/// in the compiler, not a program outside the lowered subset (`--check`
/// passes clean). This is `JitErrorKind::Error`, not a cascade (nothing
/// upstream of `caller` refused), so it aborts collection right there —
/// `never_reached`, declared after `caller`, is never even attempted, and its
/// own independent blocker (a different missing builtin) must never appear
/// in the collected sites. Reduced straight out of `examples/demonivm.dmc`'s
/// `test_push_halt` (`dmc test --jit --json --all`, `jit_sites: 1` — the
/// exact "several at sites=1" floor #684's own issue text describes), not
/// something this PR is fixing — only the `truncated` metric around it.
const ROOT_JIT_ERROR_ABORTS_AND_TRUNCATES: &str = "fn main() -> i64 { 0 }\n\
    fn run_ret[P](code: Tensor[u8, [P]]) -> i64 {\n    \
        code[0] as i64\n\
    }\n\
    fn caller() -> i64 {\n    \
        let prog = [1u8, 42u8, 0u8]\n    \
        let got = run_ret[3](prog)\n    \
        got\n\
    }\n\
    fn never_reached() -> i64 {\n    \
        isclose(1.0, 1.0)\n    \
        0\n\
    }\n";

/// #684 rework 3: `ROOT_JIT_ERROR_ABORTS_AND_TRUNCATES` with `never_reached`
/// dropped — the same root `jit-error` (`caller`'s template instantiation),
/// but now on the LAST top-level item, with nothing left in `metas`,
/// `self.pending`, or `self.pending_lambdas` when it strikes: every item in
/// the program was in fact walked. The bug this const exists to catch: an
/// earlier revision inferred `truncated` from "the last recorded site is a
/// non-cascade `jit-error`" alone, which is true here exactly as it is for
/// `ROOT_JIT_ERROR_ABORTS_AND_TRUNCATES` above — and reported `truncated:
/// true` regardless, even though nothing was left uncollected.
const ROOT_JIT_ERROR_ON_LAST_ITEM_IS_NOT_TRUNCATED: &str = "fn main() -> i64 { 0 }\n\
    fn run_ret[P](code: Tensor[u8, [P]]) -> i64 {\n    \
        code[0] as i64\n\
    }\n\
    fn caller() -> i64 {\n    \
        let prog = [1u8, 42u8, 0u8]\n    \
        let got = run_ret[3](prog)\n    \
        got\n\
    }\n";

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
                 \"message\":\"let binding has type str but value has type {{integer}}\",\
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
    // not a rendered `Tensor[f32, [2]]` it re-parses out of the message. The
    // message itself is unchanged — this is a second encoding, not a rewording.
    let probe = Probe::new(SHAPE_ERROR);
    let out = probe.run(&["--check", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        lines(&out),
        vec![
            format!(
                "{{\"schema\":1,\"kind\":\"check\",\"severity\":\"error\",\
                 \"message\":\"let binding has type Tensor[f32, [2]] but value \
                 has type Tensor[f32, [3]]\",\
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
             \"message\":\"let binding has type Tensor[f32, [2]] but value \
             has type Tensor[i64, [2]]\",\
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
        // ROADMAP.md §2: a top-level `type` alias used to be this fixture's
        // `jit-item` case, but it's now transparent to the JIT (accepted,
        // resolved through at every type-building site) rather than refused
        // — an arena block as a top-level item is still refused instead.
        ("vault { let x = 1 }\nfn main() -> i64 { 0 }\n", "jit-item", 1, 1),
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
fn a_multi_file_program_lowers_through_the_merge_and_emits_no_diagnostic() {
    // The multi-file merge (slice 3). `dmc jit` used to lex and parse only the
    // file named on the command line and check it with an empty module table,
    // so an importing program died at `{"kind":"check", … "imported module not
    // found / checked yet"}`; #624 made it share the front end, and the refusal
    // became the backend's own `jit-import`. It is now neither: the whole
    // import DAG is merged into one item list (`merge.rs`) and handed to
    // `compile_program`, so the program runs and the stream carries nothing but
    // the envelope.
    let probe = Probe::with_module(IMPORTS_MODULE, "module.dmc", MODULE);

    // Precondition, and the premise of the whole path: the program is well
    // typed, import and all.
    let checked = probe.run(&["--check", "--json"]);
    assert!(
        checked.status.success(),
        "the multi-file probe does not type-check:\n{}", stderr(&checked),
    );

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert_eq!(
        lines(&out),
        vec![
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"ok\",\
             \"errors\":0,\"warnings\":0,\"exit\":0}".to_string(),
        ],
    );
    // The imported `add` really ran: the answer is the program's, not a stub.
    assert_eq!(stdout(&out), "=> 42\n");

    // The human rendering agrees, with the program's output on stdout.
    let human = probe.run(&["jit"]);
    assert_eq!(out.status.code(), human.status.code());
    assert_eq!(stdout(&human), "=> 42\n");
    assert_eq!(stderr(&human), "");
}

#[test]
fn a_merge_that_cannot_be_built_is_jit_ineligible_under_jit_import() {
    // What is left of `jit-import` on the `dmc jit` path after slice 3, and why
    // the class keeps its meaning rather than being retired: it is still "a
    // `use` the JIT could not lower", raised one pass later — by the merge,
    // when two files' declarations cannot be given one name apiece in the
    // single flat symbol table the backend has.
    //
    // The shape here is an `extern fn` of one name declared with two different
    // signatures in two imported files. An extern's name *is* the foreign
    // symbol, so it is the one declaration the merge may not rename apart.
    // `--check` accepts the program and `dmc run` answers `3` — which is what
    // makes this a subset boundary and not a type error.
    let probe = Probe::new("use \"ea.dmc\"\nuse \"eb.dmc\"\nfn main() -> i64 { av() + bv() }\n");
    probe.sibling("ea.dmc", "extern fn dmc_probe_sym(x: i64) -> i64\npub fn av() -> i64 { 1 }\n");
    probe.sibling("eb.dmc", "extern fn dmc_probe_sym(x: f64) -> f64\npub fn bv() -> i64 { 2 }\n");
    let checked = probe.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));

    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "{:#?}", ls);
    assert!(ls[0].contains("\"kind\":\"jit-ineligible\""), "{}", ls[0]);
    assert!(ls[0].contains("\"code\":\"jit-import\""), "{}", ls[0]);
    // Never `jit-error`: the program is outside the lowered subset, and
    // `dmc run` has the full semantics for it (#480).
    assert!(!ls[0].contains("jit-error"), "{}", ls[0]);
    assert!(ls[0].contains("dmc_probe_sym"), "the message does not name the item: {}", ls[0]);
    assert_eq!(stdout(&out), "", "a refused program produced output");

    let human = probe.run(&["jit"]);
    assert_eq!(human.status.code(), Some(1));
    assert!(
        stderr(&human).starts_with("jit unsupported"),
        "a refusal must read as one to the tools that grep for it: {}", stderr(&human),
    );
}

#[test]
fn an_extern_mismatch_message_reads_relative_paths_and_source_type_spelling() {
    // #661 review finding 3. The extern-mismatch message is rendered by
    // `MergeError`'s `Display`, whose paths used to be `{:?}` (absolute,
    // quoted, machine-dependent) and whose `ty_key` rendered `ScalarType` with
    // `{:?}` too (`I64`, not the `i64` the user wrote) — and this is the first
    // place either reaches a `--json` consumer's `message` field.
    let probe = Probe::new("use \"ea.dmc\"\nuse \"eb.dmc\"\nfn main() -> i64 { av() + bv() }\n");
    probe.sibling("ea.dmc", "extern fn dmc_probe_sym(x: i64) -> i64\npub fn av() -> i64 { 1 }\n");
    let eb = probe.sibling(
        "eb.dmc", "extern fn dmc_probe_sym(x: f64) -> f64\npub fn bv() -> i64 { 2 }\n",
    );
    let checked = probe.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));

    let out = probe.run(&["jit", "--json"]);
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "{:#?}", ls);
    let msg = ls[0].clone();
    // The `"message"` field's own value, unescaped — not the whole JSON line,
    // which must keep an absolute `/` in its separate `"file"` field.
    let message = message_of(&msg);
    // The two files sit beside each other with nothing above them, so a
    // correctly-relative rendering has no `/` at all — `ea.dmc`/`eb.dmc`, not
    // `/tmp/.../ea.dmc`.
    assert!(
        !message.contains('/'),
        "the message still carries an absolute path: {}", message,
    );
    assert!(
        message.contains("ea.dmc") && message.contains("eb.dmc"),
        "the message dropped the filenames along with the path: {}", message,
    );
    // Source spelling, not `Debug`'s: `fn(i64) -> i64`, never `fn(I64) -> I64`.
    assert!(
        message.contains("fn(i64) -> i64") && message.contains("fn(f64) -> f64"),
        "the message did not use source type spelling: {}", message,
    );
    // `--json`'s own `file` field is unaffected: still the absolute, canonical
    // path (`merge_diag` reads it straight off `MergeError::file`, never off
    // the message this test just checked).
    assert!(
        msg.contains(&format!("\"file\":\"{}\"", eb.display())),
        "the json `file` field must stay absolute: {}", msg,
    );
}

#[test]
fn a_multi_file_test_now_runs_under_jit_parity_instead_of_skipping() {
    // The counterpart on the test path. A multi-file file used to be outside
    // the JIT subset by construction, so every test in it was a `skip`; slice 3
    // merges each entry's own closure before `compile_program`, so the ones
    // whose merged form lowers are now `jit_ran`. The accounting is unchanged
    // in shape — `jit_ran` + `jit_skipped`, a refusal is still a skip — only
    // which side of it these files land on.
    let probe = Probe::with_module(IMPORTS_MODULE_TEST, "module.dmc", MODULE);
    let out = probe.run_on_dir(&["test", "--json", "--jit"]);
    assert!(out.status.success(), "a multi-file parity run failed:\n{}", stderr(&out));
    assert_eq!(
        lines(&out),
        vec![
            format!(
                "{{\"schema\":1,\"kind\":\"test\",\"status\":\"pass\",\
                 \"name\":\"test_add\",\"file\":\"{}\",\"jit\":\"pass\"}}",
                probe.reported(),
            ),
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"test\",\"status\":\"ok\",\
             \"errors\":0,\"warnings\":0,\"passed\":1,\"failed\":0,\"tests_ran\":1,\
             \"jit_ran\":1,\"jit_skipped\":0,\"exit\":0}".to_string(),
        ],
    );
}

#[test]
fn a_type_error_in_an_imported_module_is_reported_against_that_module() {
    // Sharing the front end means `dmc jit` inherits its per-module
    // attribution too: an error in the imported file is reported with THAT
    // file in `file`, not the one on the command line. A single-file JIT front
    // end could not have said this at all — it never read the module.
    let probe = Probe::with_module(
        IMPORTS_MODULE,
        "module.dmc",
        "pub fn add(a: i64, b: i64) -> i64 {\n    let z: str = 5\n    a + b\n}\n",
    );
    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "{:#?}", ls);
    let module = probe._dir.path().join("module.dmc").canonicalize().expect("canonicalize module");
    assert!(
        ls[0].starts_with("{\"schema\":1,\"kind\":\"check\",\"severity\":\"error\","),
        "{}", ls[0],
    );
    assert!(
        ls[0].contains(&format!("\"file\":\"{}\"", module.display())),
        "the error was not attributed to the imported module: {}", ls[0],
    );
    // And it is the module's real type error, not the old "not found / checked
    // yet" placeholder the JIT path used to raise for every import.
    assert!(!ls[0].contains("imported module not found"), "{}", ls[0]);
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

// ── `--all` (#684) ───────────────────────────────────────────────────────────

#[test]
fn all_reports_every_refusal_and_removing_one_blocker_reports_one_fewer() {
    // Acceptance (b). Preconditions first: both fixtures type-check clean, so
    // every refusal below really is the JIT's subset boundary.
    let three = Probe::new(THREE_BLOCKERS);
    let checked = three.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));
    let two = Probe::new(TWO_BLOCKERS);
    let checked = two.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));

    let out = three.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 4, "3 refusals + 1 summary:\n{:#?}", ls);
    for (i, code, item) in [
        (0usize, "jit-type-form", "blocker_a"),
        (1, "jit-construct", "blocker_b"),
        (2, "jit-arg-form", "blocker_c"),
    ] {
        let d = decode(&ls[i]);
        assert_eq!(field(&d, "kind"), "jit-ineligible");
        assert_eq!(field(&d, "code"), code, "site {}: {}", i, ls[i]);
        assert_eq!(field(&d, "index"), i.to_string(), "site {}: {}", i, ls[i]);
        assert_eq!(field(&d, "item"), item, "site {}: {}", i, ls[i]);
    }
    let summary = decode(&ls[3]);
    assert_eq!(field(&summary, "refusals"), "3");
    assert_eq!(
        field(&summary, "classes"),
        r#"{"jit-arg-form":1,"jit-construct":1,"jit-type-form":1}"#,
    );

    // Remove `blocker_b` (the range expression, `jit-construct`): depth drops
    // by exactly one, and the two surviving sites are otherwise unchanged —
    // same classes, same items, same relative order.
    let out2 = two.run(&["jit", "--json", "--all"]);
    assert_eq!(out2.status.code(), Some(1));
    let ls2 = lines(&out2);
    assert_eq!(ls2.len(), 3, "2 refusals + 1 summary:\n{:#?}", ls2);
    for (i, code, item) in [(0usize, "jit-type-form", "blocker_a"), (1, "jit-arg-form", "blocker_c")] {
        let d = decode(&ls2[i]);
        assert_eq!(field(&d, "code"), code, "site {}: {}", i, ls2[i]);
        assert_eq!(field(&d, "index"), i.to_string(), "site {}: {}", i, ls2[i]);
        assert_eq!(field(&d, "item"), item, "site {}: {}", i, ls2[i]);
    }
    let summary2 = decode(&ls2[2]);
    assert_eq!(field(&summary2, "refusals"), "2");
    assert_eq!(field(&summary2, "classes"), r#"{"jit-arg-form":1,"jit-type-form":1}"#);
}

#[test]
fn all_reports_the_same_site_set_regardless_of_caller_declaration_order() {
    // #684 R1: `instantiate_template` records a template instantiation's
    // mangled name in `self.fns` before its body is queued in `self.pending`.
    // The old collect-mode code truncated `self.pending` on ANY later failure
    // in the SAME pass, which could silently un-queue that monomorphization
    // while leaving the `self.fns` entry in place — permanently orphaning it:
    // never defined, never re-enqueued (a sibling caller's
    // `instantiate_template` finds the `self.fns` entry and short-circuits),
    // and its own blocker never reported. Whether that happened depended on
    // caller declaration order, which is the bug: `g`'s own blocker (the
    // `0..3` in its body) must be reported in EITHER order.
    let order1 = Probe::new(ORDER_CALLER1_FIRST);
    let checked = order1.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));
    let order2 = Probe::new(ORDER_CALLER2_FIRST);
    let checked = order2.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));

    // One refusal for `caller1`'s own blocker, one for `g`'s own — never one
    // (the pre-fix bug) — in either declaration order.
    for probe in [&order1, &order2] {
        let out = probe.run(&["jit", "--json", "--all"]);
        assert_eq!(out.status.code(), Some(1));
        let ls = lines(&out);
        assert_eq!(ls.len(), 3, "2 refusals + 1 summary:\n{:#?}", ls);
        let items: Vec<String> = (0..2).map(|i| {
            let d = decode(&ls[i]);
            assert_eq!(field(&d, "code"), "jit-construct", "site {}: {}", i, ls[i]);
            assert!(!d.iter().any(|(k, _)| k == "cascade_of"), "site {}: {}", i, ls[i]);
            field(&d, "item").to_string()
        }).collect();
        // One site is `caller1` itself; the other is `g`'s own instantiation,
        // under its mangled name — never `caller1` a second time, and never
        // `caller2` (which calls `g` but has no blocker of its own).
        assert!(items.contains(&"caller1".to_string()), "{:?}", items);
        assert!(!items.contains(&"caller2".to_string()), "{:?}", items);
        assert!(items.iter().any(|it| it != "caller1"), "g's own blocker missing: {:?}", items);
        let summary = decode(&ls[2]);
        // Two raw SITES — the R1 fix's own claim, and the one that regressed
        // to 1 before it: `g`'s own blocker really is recovered in both
        // orders. `blockers` (R2's distinct-root dedup) is 1, not 2: both
        // sites are the identical unsupported construct (a bare range
        // expression) with no distinguishing identifier in the message, so
        // R2(c)'s "one gap, hit twice" collapse legitimately applies here too
        // — a different, correct mechanism from the R1 bug this test targets.
        assert_eq!(field(&summary, "refusals"), "2");
        assert_eq!(field(&summary, "blockers"), "1");
        assert_eq!(field(&summary, "cascades"), "0");
    }

    // The two orders' site sets are identical — not merely equal in count.
    let sites_of = |probe: &Probe| -> Vec<(String, String)> {
        let out = probe.run(&["jit", "--json", "--all"]);
        lines(&out)[..2].iter().map(|l| {
            let d = decode(l);
            (field(&d, "code").to_string(), field(&d, "item").to_string())
        }).collect()
    };
    let mut set1 = sites_of(&order1);
    let mut set2 = sites_of(&order2);
    set1.sort();
    set2.sort();
    assert_eq!(set1, set2, "order1 sites: {:?}, order2 sites: {:?}", set1, set2);
}

#[test]
fn unknown_fn_callers_of_a_refused_item_are_cascades_not_root_blockers() {
    // #684 R2(a): `any_type.dmc`'s shape — one root (`broken`'s signature) and
    // two `jit-unknown-fn` callers of it, which must be reported as cascades
    // of that one root rather than as two more root blockers.
    let probe = Probe::new(ONE_ROOT_TWO_UNKNOWN_FN_CASCADES);
    let checked = probe.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 4, "3 refusals + 1 summary:\n{:#?}", ls);

    let root = decode(&ls[0]);
    assert_eq!(field(&root, "code"), "jit-type-form");
    assert_eq!(field(&root, "item"), "broken");
    assert_eq!(field(&root, "index"), "0");
    assert!(!root.iter().any(|(k, _)| k == "cascade_of"), "{:?}", root);

    for i in [1usize, 2] {
        let d = decode(&ls[i]);
        assert_eq!(field(&d, "code"), "jit-unknown-fn", "site {}: {}", i, ls[i]);
        assert_eq!(field(&d, "message"), "unknown function `broken`", "site {}: {}", i, ls[i]);
        assert_eq!(field(&d, "cascade_of"), "0", "site {}: {}", i, ls[i]);
    }

    let summary = decode(&ls[3]);
    assert_eq!(field(&summary, "refusals"), "3");
    assert_eq!(field(&summary, "blockers"), "1");
    assert_eq!(field(&summary, "cascades"), "2");
    // `classes` counts only the non-cascade sites (R2): `broken`'s one
    // `jit-type-form`, not the two `jit-unknown-fn` cascades it caused.
    assert_eq!(field(&summary, "classes"), r#"{"jit-type-form":1}"#);
}

#[test]
fn a_fwd_bwd_missing_entry_defect_is_a_cascade_and_does_not_abort_collection() {
    // #684 R2(b): a shape-generic `@grad fn` refuses cleanly (`jit-grad`,
    // `Unsupported`); a caller destructuring its `.fwd_bwd(...)` then hits the
    // `jit-error` "not a `@grad fn` (no fwd_bwd entry)" — manufactured by
    // collect mode having skipped it, not a real defect. Before the fix this
    // was indistinguishable from a genuine compiler defect and ABORTED
    // collection right there; after it, it is a cascade and collection
    // continues (there is nothing left to find in this particular fixture,
    // but the exit code and site count prove collection did not stop early).
    let probe = Probe::new(GRAD_FN_ROOT_AND_FWD_BWD_CASCADE);
    let checked = probe.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 3, "2 refusals + 1 summary:\n{:#?}", ls);

    let root = decode(&ls[0]);
    assert_eq!(field(&root, "kind"), "jit-ineligible");
    assert_eq!(field(&root, "code"), "jit-grad");
    assert_eq!(field(&root, "item"), "lossy");
    assert!(!root.iter().any(|(k, _)| k == "cascade_of"), "{:?}", root);

    let cascade = decode(&ls[1]);
    assert_eq!(field(&cascade, "kind"), "jit-error");
    assert!(!cascade.iter().any(|(k, _)| k == "code"), "a jit-error carries no code: {:?}", cascade);
    assert_eq!(field(&cascade, "message"), "`lossy` is not a `@grad fn` (no fwd_bwd entry)");
    assert_eq!(field(&cascade, "cascade_of"), "0");

    let summary = decode(&ls[2]);
    assert_eq!(field(&summary, "refusals"), "2");
    assert_eq!(field(&summary, "blockers"), "1");
    assert_eq!(field(&summary, "cascades"), "1");
    // `classes` counts only non-cascade sites (R2): the `jit-error` cascade
    // never enters it at all — cascades are excluded outright, not bucketed
    // under a `jit-error` key. `jit::JIT_ERROR_CLASS_KEY` (R3) only matters
    // for a `jit-error` that is itself a ROOT (see `diag_tests.rs`), which
    // this fixture does not exercise.
    assert_eq!(field(&summary, "classes"), r#"{"jit-grad":1}"#);
}

#[test]
fn two_calls_to_different_missing_builtins_are_two_distinct_blockers() {
    // #684 fixup, F1: `blocker_counts` used to blank every backtick-quoted
    // identifier out of a message before deduplicating, so `` unknown
    // function `isclose` `` and `` unknown function `clamp` `` — two
    // independent gaps in two unrelated fns — collapsed onto the same
    // blanked message and counted as one blocker. Fixed by keying on the
    // message verbatim: two different missing builtins are two blockers.
    let probe = Probe::new(TWO_INDEPENDENT_UNKNOWN_FN_BLOCKERS);
    let checked = probe.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 3, "2 refusals + 1 summary:\n{:#?}", ls);

    for (i, item, name) in [(0usize, "a", "isclose"), (1, "b", "clamp")] {
        let d = decode(&ls[i]);
        assert_eq!(field(&d, "code"), "jit-unknown-fn", "site {}: {}", i, ls[i]);
        assert_eq!(field(&d, "item"), item, "site {}: {}", i, ls[i]);
        assert_eq!(
            field(&d, "message"), format!("unknown function `{}`", name),
            "site {}: {}", i, ls[i],
        );
        assert!(!d.iter().any(|(k, _)| k == "cascade_of"), "site {}: {}", i, ls[i]);
    }

    let summary = decode(&ls[2]);
    assert_eq!(field(&summary, "refusals"), "2");
    // The headline claim: TWO blockers, not one — the pre-fix bug's own
    // number for this exact shape.
    assert_eq!(field(&summary, "blockers"), "2");
    assert_eq!(field(&summary, "cascades"), "0");
    assert_eq!(field(&summary, "classes"), r#"{"jit-unknown-fn":2}"#);
}

#[test]
fn a_str_payload_enum_root_and_its_construction_callers_are_one_blocker_not_two() {
    // #684 fixup, F2: `F.Named(str)`'s `str` payload refuses the whole enum
    // as `jit-item` (root, item `F`) — `enum_payload_layouts` never gains an
    // entry for `F`. `fa`/`fb` each go on to construct `F.Named(...)`, which
    // — finding no layout — falls through the qualified-construction
    // intercept into generic call lowering and reports the callee as the
    // unresolved `.Named()` method. Before the fix, `cascade_candidate_name`
    // looked `.Named()` up in `refused_items` (item names only, never a
    // variant/method name) and never matched, so `fa`/`fb` each counted as
    // their own root blocker: `blockers: 2` for what is really one gap
    // (`an i64 payload lowers fine` — the defect is squarely `F`'s).
    let probe = Probe::new(ENUM_STR_PAYLOAD_ROOT_AND_METHOD_CASCADES);
    let checked = probe.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 4, "3 refusals + 1 summary:\n{:#?}", ls);

    let root = decode(&ls[0]);
    assert_eq!(field(&root, "code"), "jit-item");
    assert_eq!(field(&root, "item"), "F");
    assert_eq!(field(&root, "index"), "0");
    assert!(!root.iter().any(|(k, _)| k == "cascade_of"), "{:?}", root);

    for (i, item) in [(1usize, "fa"), (2, "fb")] {
        let d = decode(&ls[i]);
        assert_eq!(field(&d, "code"), "jit-unknown-fn", "site {}: {}", i, ls[i]);
        assert_eq!(field(&d, "item"), item, "site {}: {}", i, ls[i]);
        assert!(
            field(&d, "message").contains("`.Named()`"),
            "site {}: {}", i, ls[i],
        );
        assert_eq!(field(&d, "cascade_of"), "0", "site {}: {}", i, ls[i]);
    }

    let summary = decode(&ls[3]);
    assert_eq!(field(&summary, "refusals"), "3");
    // The headline claim: ONE blocker (`F`'s own), not two — `fa`/`fb` are
    // cascades of it, never root blockers of their own.
    assert_eq!(field(&summary, "blockers"), "1");
    assert_eq!(field(&summary, "cascades"), "2");
    assert_eq!(field(&summary, "classes"), r#"{"jit-item":1}"#);
}

#[test]
fn a_root_jit_error_truncates_collection_and_is_reported_as_such() {
    // #684 fixup, F3: a root `jit-error` (a genuine JIT defect, #480 — not a
    // cascade) still aborts collection where it strikes, exactly as before
    // this whole PR. What is new is that the abort is now REPORTED: the
    // summary's `truncated` says the one recorded site is a FLOOR, not the
    // file's true depth — `never_reached`, declared after the fn that trips
    // the defect, is never even attempted, and its own independent blocker
    // never appears.
    let probe = Probe::new(ROOT_JIT_ERROR_ABORTS_AND_TRUNCATES);
    let checked = probe.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "1 refusal + 1 summary — never_reached's own \
        blocker must not appear:\n{:#?}", ls);

    let root = decode(&ls[0]);
    assert_eq!(field(&root, "kind"), "jit-error");
    assert!(!root.iter().any(|(k, _)| k == "code"), "a jit-error carries no code: {:?}", root);
    assert_eq!(field(&root, "item"), "caller");
    assert!(!root.iter().any(|(k, _)| k == "cascade_of"), "{:?}", root);

    let summary = decode(&ls[1]);
    assert_eq!(field(&summary, "refusals"), "1");
    assert_eq!(field(&summary, "blockers"), "1");
    assert_eq!(field(&summary, "cascades"), "0");
    assert_eq!(field(&summary, "truncated"), "true");
}

#[test]
fn a_root_jit_error_on_the_last_item_is_not_truncated() {
    // #684 rework 3: the negative this PR adds alongside the test above —
    // the identical root `jit-error` (`caller`'s template instantiation
    // mismatch), but with `never_reached` removed, so `caller` is the LAST
    // top-level item and nothing is left in `metas`/`self.pending`/
    // `self.pending_lambdas` when it strikes. Every item was walked, so
    // `truncated` must be `false` — not `true`, which is what an earlier
    // revision reported here by inferring the flag from "the last recorded
    // site is a non-cascade `jit-error`" alone, a fact that holds for THIS
    // probe exactly as it does for the one above, regardless of whether any
    // work actually remained.
    let probe = Probe::new(ROOT_JIT_ERROR_ON_LAST_ITEM_IS_NOT_TRUNCATED);
    let checked = probe.run(&["--check", "--json"]);
    assert!(checked.status.success(), "{}", stderr(&checked));

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "1 refusal + 1 summary:\n{:#?}", ls);

    let root = decode(&ls[0]);
    assert_eq!(field(&root, "kind"), "jit-error");
    assert!(!root.iter().any(|(k, _)| k == "code"), "a jit-error carries no code: {:?}", root);
    assert_eq!(field(&root, "item"), "caller");
    assert!(!root.iter().any(|(k, _)| k == "cascade_of"), "{:?}", root);

    let summary = decode(&ls[1]);
    assert_eq!(field(&summary, "refusals"), "1");
    assert_eq!(field(&summary, "blockers"), "1");
    assert_eq!(field(&summary, "cascades"), "0");
    assert_eq!(field(&summary, "truncated"), "false");
}

#[test]
fn a_file_with_only_refusals_and_no_aborting_defect_is_not_truncated() {
    // #684 fixup, F3's negative: a file that collects to completion (every
    // item walked, nothing aborted it) reports `truncated: false` — never
    // omitted, never a bare absence a consumer could mistake for "not
    // measured". Reuses `THREE_BLOCKERS`, already proven elsewhere to be
    // three independent, non-aborting refusals.
    let probe = Probe::new(THREE_BLOCKERS);
    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    let summary = decode(&ls[ls.len() - 1]);
    assert_eq!(field(&summary, "truncated"), "false");
}

#[test]
fn all_is_diagnostic_only_and_never_runs_main() {
    // A clean program under `--all` reports zero refusals rather than running
    // it — collect mode never calls `finalize_definitions` or executes
    // anything, by design (it may leave a skipped item declared-but-undefined
    // in the same module a clean run would otherwise finalize and run).
    let probe = Probe::new(CLEAN);
    let out = probe.run(&["jit", "--json", "--all"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(stdout(&out), "", "`--all` ran main on a clean program");
    assert_eq!(
        lines(&out),
        vec![
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"ok\",\
             \"errors\":0,\"warnings\":0,\"refusals\":0,\"blockers\":0,\"cascades\":0,\
             \"classes\":{},\"truncated\":false,\"exit\":0}".to_string(),
        ],
    );
}

#[test]
fn without_all_jit_json_is_byte_identical_to_the_first_refusal_baseline() {
    // The other half of acceptance (b)/(c): `--all` is strictly additive.
    // Dropping it from the same command line reproduces today's exact
    // first-refusal encoding — the first of the three sites above, and
    // nothing else on the line.
    let probe = Probe::new(THREE_BLOCKERS);
    let out = probe.run(&["jit", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        lines(&out),
        vec![
            format!(
                "{{\"schema\":1,\"kind\":\"jit-ineligible\",\"severity\":\"error\",\
                 \"code\":\"jit-type-form\",\
                 \"message\":\"`any`-typed values (they are dynamically typed); \
                 use `dmc run` for full semantics\",\"file\":\"{}\",\"line\":3,\"col\":17}}",
                probe.reported(),
            ),
            "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"failed\",\
             \"errors\":1,\"warnings\":0,\"exit\":1}".to_string(),
        ],
    );
}

#[test]
fn all_requires_json_under_jit_but_not_under_test() {
    // `dmc jit --all` has no rendering but the JSON one, so argument parsing
    // refuses it without `--json`; `dmc test --jit --all` has a plain-text
    // blocker-depth line, so it needs no `--json` at all.
    let probe = Probe::new(THREE_BLOCKERS);
    let out = probe.run(&["jit", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    // No `--json` on this invocation, so `fail_cli` never builds an emitter —
    // the plain human sentence, not a JSON `cli`/`all-requires-json` object.
    assert_eq!(
        stderr(&out),
        "`dmc jit --all` only has a JSON rendering today — pass `--json` too, \
         or drop `--all` for the first-refusal behavior\n",
    );

    // `dmc test --jit --all` (no `--json`) needs no gate at all: it is
    // accepted and its exit code is exactly what the same run without
    // `--all` gives (`TESTS_MIXED` has one failing test, so this is a
    // non-zero exit — the point is that it is the *same* nonzero exit, not
    // the CLI-usage refusal `--all` gets under `jit`).
    let probe = Probe::new(TESTS_MIXED);
    let with_all = probe.run_on_dir(&["test", "--jit", "--all"]);
    let without_all = probe.run_on_dir(&["test", "--jit"]);
    assert_eq!(with_all.status.code(), without_all.status.code());
    // At least one test fails, so this run's summary line is the `FAILED`
    // branch, printed to stderr (`eprintln!`), not stdout.
    assert!(
        stderr(&with_all).contains("blocker depth: 0 files, 0 blockers"),
        "{}", stderr(&with_all),
    );
}

#[test]
fn all_is_unwired_for_commands_other_than_jit_and_test() {
    let probe = Probe::new(CLEAN);
    let out = probe.run(&["--check", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        stderr(&out),
        "`--all` reports every JIT refusal instead of the first; it is wired \
         for `dmc jit` and `dmc test --jit`, not `--check`\n",
    );
}

#[test]
fn test_jit_all_reports_per_file_blocker_depth() {
    // Requirement (3), reworked by #684 R2: `dmc test --jit --json --all`
    // gains `jit_blockers` / `jit_sites` / `jit_cascades` / `jit_classes` per
    // test, computed via collect mode — the same numbers `dmc jit --json
    // --all` reports for the file as a whole — and the human summary line
    // gains `| blocker depth: F files, D blockers`. `THREE_BLOCKERS`' three
    // sites are independent roots (none calls another), so blockers == sites
    // == 3 and cascades == 0 here; the cascade-specific split is covered by
    // `all_reports_cascades_distinctly_from_root_blockers` below.
    let probe = Probe::with_module(
        "fn test_ok() -> bool { true }\n",
        "unused.dmc",
        "fn spacer() -> i64 { 0 }\n",
    );
    // Overwrite the probe file with the three-blocker fixture plus one test,
    // so the file both has a `test_*` and is outside the JIT subset thrice.
    std::fs::write(
        &probe.path,
        format!("{}\nfn test_something() -> bool {{ true }}\n", THREE_BLOCKERS),
    ).expect("rewrite probe");

    let out = probe.run_on_dir(&["test", "--json", "--jit", "--all"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "{:#?}", ls);
    let d = decode(&ls[0]);
    assert_eq!(field(&d, "kind"), "test");
    assert_eq!(field(&d, "jit"), "skip");
    assert_eq!(field(&d, "jit_blockers"), "3");
    assert_eq!(field(&d, "jit_sites"), "3");
    assert_eq!(field(&d, "jit_cascades"), "0");
    assert_eq!(
        field(&d, "jit_classes"),
        r#"{"jit-arg-form":1,"jit-construct":1,"jit-type-form":1}"#,
    );
    // #684 fixup, F3: none of `THREE_BLOCKERS`' three sites is an aborting
    // `jit-error` — every item was walked, so the depth above is exact, not
    // a floor.
    assert_eq!(field(&d, "truncated"), "false");
    let summary = decode(&ls[1]);
    assert_eq!(field(&summary, "blocker_files"), "1");
    assert_eq!(field(&summary, "blocker_sites"), "3");

    let human = probe.run_on_dir(&["test", "--jit", "--all"]);
    assert!(human.status.success(), "{}", stderr(&human));
    // All tests pass, so the summary line prints to stdout (`println!`), not
    // stderr — same as the existing `| jit parity` note it extends.
    assert!(
        stdout(&human).contains("| blocker depth: 1 files, 3 blockers"),
        "{}", stdout(&human),
    );

    // Without `--all`, both the JSON and the human line are unchanged — no
    // `jit_blockers`/`jit_sites`/`jit_cascades`/`jit_classes`, no
    // `blocker_files`/`blocker_sites`, no `| blocker depth` suffix.
    let plain = probe.run_on_dir(&["test", "--json", "--jit"]);
    let plain_ls = lines(&plain);
    assert_eq!(plain_ls.len(), 2, "{:#?}", plain_ls);
    let plain_d = decode(&plain_ls[0]);
    assert!(!plain_d.iter().any(|(k, _)| k == "jit_blockers"), "{:?}", plain_d);
    assert!(!plain_d.iter().any(|(k, _)| k == "jit_sites"), "{:?}", plain_d);
    assert!(!plain_d.iter().any(|(k, _)| k == "jit_cascades"), "{:?}", plain_d);
    assert!(!plain_d.iter().any(|(k, _)| k == "jit_classes"), "{:?}", plain_d);
    assert!(!plain_d.iter().any(|(k, _)| k == "truncated"), "{:?}", plain_d);
    let plain_summary = decode(&plain_ls[1]);
    assert!(!plain_summary.iter().any(|(k, _)| k == "blocker_files"), "{:?}", plain_summary);
    let plain_human = probe.run_on_dir(&["test", "--jit"]);
    assert!(!stdout(&plain_human).contains("blocker depth"), "{}", stdout(&plain_human));
}

#[test]
fn a_truncated_file_appends_the_floor_note_to_the_human_depth_line() {
    // #684 rework 3: when at least one measured file's depth is a floor
    // (`truncated:true`), the human `| blocker depth` line names it —
    // `(N truncated — floor)`, appended after the existing "F files, D
    // blockers" text, never in place of it. `ROOT_JIT_ERROR_ABORTS_AND_TRUNCATES`
    // is the known truncated fixture (its own dedicated test above asserts
    // `truncated:true` on the JSON side); this test is the human-line half.
    let probe = Probe::new(&format!(
        "{}\nfn test_something() -> bool {{ true }}\n",
        ROOT_JIT_ERROR_ABORTS_AND_TRUNCATES,
    ));
    let human = probe.run_on_dir(&["test", "--jit", "--all"]);
    assert!(human.status.success(), "{}", stderr(&human));
    assert!(
        stdout(&human).contains("| blocker depth: 1 files, 1 blockers (1 truncated — floor)"),
        "{}", stdout(&human),
    );
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
             \"errors\":0,\"warnings\":0,\"passed\":1,\"failed\":1,\"tests_ran\":2,\"exit\":1}".to_string(),
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
             \"errors\":0,\"warnings\":0,\"passed\":0,\"failed\":2,\"tests_ran\":2,\
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
             \"errors\":0,\"warnings\":0,\"passed\":1,\"failed\":0,\"tests_ran\":1,\
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
         \"errors\":0,\"warnings\":0,\"passed\":0,\"failed\":0,\"tests_ran\":0,\"exit\":0}\n",
    );
    assert_eq!(stdout(&out), "");
}

/// DIAGNOSTICS.md §7: under `dmc test`, a file-level failure (`FAIL <file>:
/// resolution failed …`) is not a *test*, so it gets no `test` object, but
/// still counts into the summary's `failed`. When it is located — here, a
/// `use` of a missing file — it is the same `resolve` object `--check` emits
/// for that file. This is a different path from every other test in this
/// file (all of which fail inside a real test run, and so DO get a `test`
/// object).
#[test]
fn a_file_level_test_failure_is_the_located_resolve_object_with_no_test_object() {
    let probe = Probe::new(MISSING_IMPORT);
    let out = probe.run(&["test", "--json"]);
    // Disclosed quirk (DIAGNOSTICS.md §4): a run that found no *tests* (this
    // one never got that far) reports `exit:0` regardless — the exit code
    // does not depend on `--json`, so neither do the numbers explaining it.
    assert!(out.status.success(), "{}", stderr(&out));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "expected exactly the failure line + summary:\n{:#?}", ls);
    // Byte-identical to what `--check --json` emits for the same file: the
    // encoding of a resolution failure does not depend on which command hit it.
    let check = lines(&probe.run(&["--check", "--json"]));
    assert_eq!(ls[0], check[0]);
    assert!(ls[0].starts_with("{\"schema\":1,\"kind\":\"resolve\",\"severity\":\"error\",\
                                \"code\":\"import-unresolved\",\"message\":\""), "{}", ls[0]);
    // No `test` object exists for this failure — `kind` is never `"test"`.
    assert!(!ls[0].contains("\"kind\":\"test\""), "{}", ls[0]);
    assert_eq!(
        ls[1],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"test\",\"status\":\"ok\",\
         \"errors\":1,\"warnings\":0,\"passed\":0,\"failed\":1,\"tests_ran\":0,\"exit\":0}",
    );
}

#[test]
fn a_parse_error_in_a_test_file_is_parse_not_unstructured() {
    // The other located file-level failure `dmc test` can hit. Before the
    // resolver carried spans, this was `unstructured` prose with a `FAIL`
    // prefix; it is the `parse` object `--check` emits, file and all.
    let probe = Probe::new("use foo\nfn test_never_runs() -> bool { true }\n");
    let out = probe.run(&["test", "--json"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "{:#?}", ls);
    assert_eq!(
        ls[0],
        format!(
            "{{\"schema\":1,\"kind\":\"parse\",\"severity\":\"error\",\
             \"message\":\"expected string literal for import path, found Ident(\\\"foo\\\")\",\
             \"file\":\"{}\",\"line\":1,\"col\":5}}",
            probe.reported(),
        ),
    );
    assert_eq!(ls[0], lines(&probe.run(&["--check", "--json"]))[0]);
}

// ── Module resolution ────────────────────────────────────────────────────────

#[test]
fn a_missing_import_is_a_resolve_diagnostic_located_at_the_use() {
    // The resolver raises this while standing on the `use` item, so the span
    // is the item's: line 2 of the fixture, not a default 1:1. `file` is the
    // importing file — the failure is the `use`'s, not the missing file's.
    let probe = Probe::new(MISSING_IMPORT);
    let out = probe.run(&["--check", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "{:#?}", ls);
    let d = decode(&ls[0]);
    assert_eq!(field(&d, "schema"), "1");
    assert_eq!(field(&d, "kind"), "resolve");
    assert_eq!(field(&d, "severity"), "error");
    assert_eq!(field(&d, "code"), "import-unresolved");
    // The human sentence, unmodified — the resolver's own `Display`. It names
    // the import and the OS's reason but not the importing file (#855): the
    // span above already names it, and repeating it in the message body was
    // both redundant and an absolute path a consumer would have to re-parse
    // out of prose instead of reading off `file`.
    assert_eq!(
        field(&d, "message"),
        format!(
            "cannot resolve import {:?}: {}",
            "does_not_exist_518.dmc",
            probe.missing_file_error("does_not_exist_518.dmc"),
        ),
    );
    assert_eq!(field(&d, "file"), probe.reported());
    assert_eq!(field(&d, "line"), "2");
    assert_eq!(field(&d, "col"), "1");
    // The byte span covers the whole `use "..." as m` statement — the same
    // span `dmc jit --json` already reports for this `use` (there, via the
    // checker, since single-file JIT doesn't follow imports).
    assert_eq!(field(&d, "start"), "25");
    assert_eq!(field(&d, "end"), "58");
    assert_eq!(d.len(), 10, "unexpected fields: {:?}", d);
    assert_eq!(
        ls[1],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"--check\",\"status\":\"failed\",\
         \"errors\":1,\"warnings\":0,\"exit\":1}",
    );
}

#[test]
fn an_import_cycle_is_a_resolve_diagnostic_located_at_the_use_that_closes_it() {
    // probe.dmc → ring.dmc → probe.dmc. The cycle is detected at ring.dmc's
    // `use`, so that is the file and line the diagnostic points at, and the
    // message names the whole cycle from the entry file round to itself.
    let probe = Probe::new("use \"ring.dmc\" as r\nfn main() -> i64 { 0 }\n");
    let ring = probe.sibling("ring.dmc", "fn spacer() -> i64 { 1 }\nuse \"probe.dmc\" as p\n");
    let out = probe.run(&["--check", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "{:#?}", ls);
    let d = decode(&ls[0]);
    assert_eq!(field(&d, "kind"), "resolve");
    assert_eq!(field(&d, "severity"), "error");
    assert_eq!(field(&d, "code"), "import-cycle");
    let entry = probe.path.canonicalize().unwrap();
    assert_eq!(
        field(&d, "message"),
        format!("circular import detected: {:?} -> {:?} -> {:?}", entry, ring, entry),
    );
    assert_eq!(field(&d, "file"), ring.display().to_string());
    assert_eq!(field(&d, "line"), "2");
    assert_eq!(field(&d, "col"), "1");
    // The byte span is `use "probe.dmc" as p`, within ring.dmc's own source.
    assert_eq!(field(&d, "start"), "25");
    assert_eq!(field(&d, "end"), "45");
    assert_eq!(d.len(), 10, "unexpected fields: {:?}", d);

    // A file that imports itself is the one-element cycle: same code, and
    // the file is the entry file, since its own `use` closes it.
    let selfie = Probe::new("use \"probe.dmc\" as me\nfn main() -> i64 { 0 }\n");
    let out = selfie.run(&["--check", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let d = decode(&lines(&out)[0]);
    assert_eq!(field(&d, "code"), "import-cycle");
    assert_eq!(field(&d, "file"), selfie.reported());
    assert_eq!(field(&d, "line"), "1");
    // Its `use "probe.dmc" as me` is the very first thing in the file.
    assert_eq!(field(&d, "start"), "0");
    assert_eq!(field(&d, "end"), "21");
    let me = selfie.path.canonicalize().unwrap();
    assert_eq!(
        field(&d, "message"),
        format!("circular import detected: {:?} -> {:?}", me, me),
    );
}

#[test]
fn a_failure_inside_an_imported_module_is_located_in_that_module() {
    // The import resolves; the imported file itself fails to parse. That is
    // the module's `parse` diagnostic, located in the module — not a `resolve`
    // diagnostic at the `use` — so a consumer is sent to the actual defect.
    let probe = Probe::new("use \"broken.dmc\" as b\nfn main() -> i64 { 0 }\n");
    let broken = probe.sibling("broken.dmc", "fn spacer() -> i64 { 1 }\nuse foo\n");
    let out = probe.run(&["--check", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let d = decode(&lines(&out)[0]);
    assert_eq!(field(&d, "kind"), "parse");
    assert_eq!(field(&d, "file"), broken.display().to_string());
    assert_eq!(field(&d, "line"), "2");
    assert_eq!(field(&d, "col"), "5");
}

#[test]
fn an_import_that_resolves_but_cannot_be_read_as_a_directory_is_still_a_resolve_diagnostic() {
    // resolver.rs ~107: `Err(ResolveError::Other(msg)) if origin.is_some()`.
    // `"sub"` names a real path — it canonicalizes fine, so this is NOT
    // `import-unresolved`'s canonicalize failure — but a directory can't be
    // read as a module, so `parse_file`'s `read_to_string` fails, and that
    // (span-less) `Other` is re-homed onto the importing `use`'s span, same
    // as any other import failure. The message shape differs from a
    // canonicalize failure's `cannot resolve import "...": ...`: it is
    // `parse_file`'s own `error reading "<imported>": ...`, naming only the
    // target, not the importer.
    let probe = Probe::new(
        "fn helper() -> i64 { 1 }\n\
         fn spacer() -> i64 { 2 }\n\
         use \"sub\" as m\n\
         fn test_never_runs() -> bool { true }\n",
    );
    let sub = probe._dir.path().join("sub");
    std::fs::create_dir(&sub).expect("mkdir sub");
    let sub = sub.canonicalize().expect("canonicalize sub");
    let out = probe.run(&["--check", "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let d = decode(&lines(&out)[0]);
    assert_eq!(field(&d, "kind"), "resolve");
    assert_eq!(field(&d, "code"), "import-unresolved");
    assert_eq!(field(&d, "file"), probe.reported());
    assert_eq!(field(&d, "line"), "3");
    assert_eq!(field(&d, "col"), "1");
    assert_eq!(field(&d, "start"), "50");
    assert_eq!(field(&d, "end"), "64");
    let read_err = std::fs::read_to_string(&sub).expect_err("a directory is not readable as a file");
    assert_eq!(field(&d, "message"), format!("error reading {:?}: {}", sub, read_err));
}

#[test]
#[cfg(unix)]
fn an_import_that_resolves_but_cannot_be_read_for_permissions_is_still_a_resolve_diagnostic() {
    // Same `Other` re-homing as the directory case above, for the other way
    // a resolvable path can fail to read: no read permission.
    use std::os::unix::fs::PermissionsExt;

    let probe = Probe::new(
        "fn helper() -> i64 { 1 }\n\
         fn spacer() -> i64 { 2 }\n\
         use \"locked.dmc\" as m\n\
         fn test_never_runs() -> bool { true }\n",
    );
    let locked = probe.sibling("locked.dmc", "fn main() -> i64 { 0 }\n");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000))
        .expect("chmod 000");
    let out = probe.run(&["--check", "--json"]);
    let read_err = std::fs::read_to_string(&locked).expect_err("mode 000 is not readable");
    // Restore permissions before any assertion can panic — `tempfile` needs
    // to be able to delete the directory on drop.
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644))
        .expect("restore permissions");
    assert_eq!(out.status.code(), Some(1));
    let d = decode(&lines(&out)[0]);
    assert_eq!(field(&d, "kind"), "resolve");
    assert_eq!(field(&d, "code"), "import-unresolved");
    assert_eq!(field(&d, "file"), probe.reported());
    assert_eq!(field(&d, "line"), "3");
    assert_eq!(field(&d, "col"), "1");
    assert_eq!(field(&d, "start"), "50");
    assert_eq!(field(&d, "end"), "71");
    assert_eq!(field(&d, "message"), format!("error reading {:?}: {}", locked, read_err));
}

#[test]
fn resolver_human_output_is_located_without_the_flag() {
    // #855: `--check`, `run` and `jit` used to print the resolver's bare,
    // unlocated `msg` — no `line:col` at all, and the importing file's
    // absolute path folded into the sentence instead. `Display` now prints
    // `resolve error at L:C: <msg>`, the same shape `parse error at L:C` /
    // `type error at L:C` already use, and the importing file's path is gone
    // from the message body (the span already names it) — only the import
    // string and the OS's own reason remain.
    let probe = Probe::new(MISSING_IMPORT);
    let sentence = format!(
        "resolve error at 2:1: cannot resolve import {:?}: {}\n",
        "does_not_exist_518.dmc",
        probe.missing_file_error("does_not_exist_518.dmc"),
    );
    for cmd in ["--check", "run", "jit"] {
        let out = probe.run(&[cmd]);
        assert_eq!(out.status.code(), Some(1), "`dmc {}`", cmd);
        assert_eq!(stdout(&out), "", "`dmc {}`", cmd);
        assert_eq!(stderr(&out), sentence, "`dmc {}`", cmd);
        // The importing file's absolute path must not appear anywhere in the
        // message body — the span (carried separately, under `--json`) is
        // where a consumer reads the file, not the prose.
        let dir_str = probe._dir.path().display().to_string();
        assert!(!stderr(&out).contains(&dir_str), "`dmc {}`: {}", cmd, stderr(&out));
    }
    // `dmc test`: the `FAIL <file>:` prefix on stderr, and — since no test
    // ever ran — the `0 tests` line on stdout with exit 0 (DIAGNOSTICS.md §4).
    let out = probe.run(&["test"]);
    assert!(out.status.success());
    assert_eq!(stdout(&out), "0 tests\n");
    assert_eq!(
        stderr(&out),
        format!("FAIL {}: resolution failed: {}", probe.path.display(), sentence),
    );

    // The cyclic case: located the same way, on all three modes. `msg` still
    // names every file in the cycle (DIAGNOSTICS.md §5.4: "message names the
    // whole cycle") — that content is not the importing file's redundant
    // path, so it stays.
    let probe = Probe::new("use \"ring.dmc\" as r\nfn main() -> i64 { 0 }\n");
    let ring = probe.sibling("ring.dmc", "use \"probe.dmc\" as p\n");
    let entry = probe.path.canonicalize().unwrap();
    let cycle_sentence = format!(
        "resolve error at 1:1: circular import detected: {:?} -> {:?} -> {:?}\n",
        entry, ring, entry,
    );
    for cmd in ["--check", "run", "jit"] {
        let out = probe.run(&[cmd]);
        assert_eq!(out.status.code(), Some(1), "`dmc {}`", cmd);
        assert_eq!(stderr(&out), cycle_sentence, "`dmc {}`", cmd);
    }
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
        "type error at 2:5: let binding has type str but value has type {integer}\n\
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
        } else if b[i] == '{' {
            // A flat object of string keys to non-negative integers — `classes`
            // / `jit_classes` (#684). One level deep only (a refusal-class
            // multiset never nests), kept as raw text like the array case.
            let start = i;
            i += 1;
            if b[i] == '}' {
                i += 1;
            } else {
                loop {
                    decode_string(&b, &mut i, line);
                    assert_eq!(b[i], ':', "expected `:` at {} in {}", i, line);
                    i += 1;
                    let s = i;
                    while b[i].is_ascii_digit() { i += 1; }
                    assert!(i > s, "expected a count at {} in {}", i, line);
                    match b[i] {
                        ',' => i += 1,
                        '}' => { i += 1; break; }
                        c => panic!("expected `,` or `}}`, found {:?} at {} in {}", c, i, line),
                    }
                }
            }
            b[start..i].iter().collect()
        } else if b[i] == 't' || b[i] == 'f' || b[i] == 'n' {
            // A bare literal — `true`/`false` (#684 fixup, F3's `truncated`)
            // or `null` (R3's tri-state marker) — kept as its own word so
            // `field()` compares it the same way it compares any other
            // decoded value.
            let start = i;
            while b[i].is_ascii_alphabetic() { i += 1; }
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
        (vec!["--check", "--json"], MISSING_IMPORT),
        (vec!["test", "--json"], MISSING_IMPORT),
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

#[test]
fn a_broken_import_reports_the_same_way_from_both_commands() {
    // This PR's actual property: `dmc jit` shares `resolve_and_check`'s single
    // resolution path with `--check` and `run`, so a broken import can no
    // longer be encoded differently depending on which command hit it, or
    // located inside the checker rather than at the `use`. Since #633, that
    // shared failure is `resolve`/`import-unresolved`, spanned at the `use` —
    // the same object `a_missing_import_is_a_resolve_diagnostic_located_at_the_use`
    // pins for `--check` alone. Here the point is parity: `--check` and `jit`
    // must produce it byte-for-byte identically.
    for cmd in ["--check", "jit"] {
        let probe = Probe::new("use \"missing.dmc\"\nfn main() -> i64 { 0 }\n");
        let out = probe.run(&[cmd, "--json"]);
        assert_eq!(out.status.code(), Some(1), "`dmc {}` accepted a broken import", cmd);
        let ls = lines(&out);
        let d = decode(&ls[0]);
        assert_eq!(field(&d, "schema"), "1");
        assert_eq!(field(&d, "kind"), "resolve", "`dmc {}`", cmd);
        assert_eq!(field(&d, "severity"), "error");
        assert_eq!(field(&d, "code"), "import-unresolved", "`dmc {}`", cmd);
        // #855: the message names the import string but not the importing
        // file — the span (`file`/`line`/`col` below) is where that lives.
        let msg = field(&d, "message");
        assert!(
            msg.contains("missing.dmc") && !msg.contains(&probe.reported()),
            "`dmc {}`: message {:?} should name the import but not the importing file",
            cmd, msg,
        );
        assert_eq!(field(&d, "file"), probe.reported(), "`dmc {}`", cmd);
        assert_eq!(field(&d, "line"), "1", "`dmc {}`", cmd);
        assert_eq!(field(&d, "col"), "1", "`dmc {}`", cmd);
        // The byte span covers the whole `use "missing.dmc"` item.
        assert_eq!(field(&d, "start"), "0", "`dmc {}`", cmd);
        assert_eq!(field(&d, "end"), "17", "`dmc {}`", cmd);
        assert_eq!(d.len(), 10, "unexpected fields: {:?}", d);
    }

    // Same failure, same encoding, regardless of which command hit it — the
    // exact parity claim, checked line-for-line rather than shape-only. Both
    // probes live in their own tempdir at the same relative layout, so the
    // only difference the raw lines could carry is the tempdir path itself,
    // and there is none: canonicalizing to the same probe name makes the two
    // command's lines byte-identical outright.
    let check_probe = Probe::new("use \"missing.dmc\"\nfn main() -> i64 { 0 }\n");
    let check_out = check_probe.run(&["--check", "--json"]);
    let jit_probe = Probe::new("use \"missing.dmc\"\nfn main() -> i64 { 0 }\n");
    let jit_out = jit_probe.run(&["jit", "--json"]);
    let check_line = lines(&check_out)[0].replace(&check_probe.reported(), "<path>");
    let jit_line = lines(&jit_out)[0].replace(&jit_probe.reported(), "<path>");
    assert_eq!(
        check_line, jit_line,
        "`--check` and `jit` disagree on a broken import beyond the tempdir path",
    );
}

