//! #751: `len(m)` on a map — found by `dmc test --jit --all` on the
//! downstream corpus right after #743 landed. `--check` accepts it, `dmc
//! run` answers the entry count (`interp.rs`'s `"len"` arm has a `Value::Map`
//! case), and `dmc jit` refused with an uncoded `jit error: len requires a
//! tensor or str argument` — #480's exact defect class, since `lower_
//! builtin_len` only recognized `str` and `Tensor`.
//!
//! Fix: `lower_builtin_len` now also recognizes `TyKind::Map`, calling a new
//! `__dmc_map_len` runtime helper that returns the backing `HashMap`'s entry
//! count — the same `HashMap<String, i64>` `dmc run`'s `Value::Map` reads,
//! so the two backends agree without any separate bookkeeping (`HashMap::
//! insert` overwrites an existing key's slot rather than adding a second
//! one, so re-setting a key already present never double-counts on either
//! backend).
//!
//! The corpus site was `len(map_get(d, "fns"))` — `len` of a nested map
//! value, i.e. a `map_get` result (always `TyKind::Scalar(OptI64)` in the
//! JIT, #742/#721) fed to `len`. That's the exact same opaque-i64 shape
//! `map_slot_type_refusal` (#743) already classes for `map_get`/`map_set`/
//! `map_contains` — reused here for `len`'s argument too: cites #721. Any
//! other non-tensor/str/Map/tuple type (an `i64`/`bool`/`f64` literal, a
//! `fn`, a `Port`, a real model instance, …) is a real bug in the caller's
//! program `dmc run` also rejects at runtime, and gets the same located
//! `jit-conversion` refusal (`len_type_refusal`, a PR #754 review follow-up
//! to this fix), never the uncoded `jit error` this replaces — citing #758,
//! not #745: #745's fix is one concrete parameter type per position
//! (sufficient for the map builtins' single-typed arguments), which cannot
//! express `len`'s polymorphic domain (tensor, str, Map, or tuple).
//!
//! A tuple is NOT in that "real bug" bucket — `len((1,2,3))` is `3` on
//! `dmc run` (`Value::Tuple(vs) => vs.len()`), and the same PR #754 review
//! gave it a static `iconst(arity)` lowering alongside this file's fixes
//! (see `jit_len_of_tuple_is_arity` in `jit.rs`'s own test module, not
//! duplicated here).

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

/// Every JSON Lines diagnostic parsed out of a `--json` run's stderr — see
/// `jit_map_nested_742.rs`'s identical helper for why this is a flat
/// substring walk rather than a serde_json parse (this crate has no JSON dep
/// in tests).
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

fn message_of(out: &Output) -> String {
    lines(out).iter()
        .find_map(|l| extract_field(l, "message"))
        .unwrap_or_default()
}

/// Run `src` under both backends and assert they agree on the printed i64.
fn assert_run_and_jit_agree_on_i64(src: &str, expect: i64) {
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    let run_out = String::from_utf8_lossy(&run.stdout).to_string();
    assert!(run_out.contains(&expect.to_string()),
        "dmc run: expected `{}` in stdout, got: {}", expect, run_out);

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit.stderr));
    let jit_out = String::from_utf8_lossy(&jit.stdout).to_string();
    assert!(jit_out.contains(&expect.to_string()),
        "dmc jit: expected `{}` in stdout, got: {}", expect, jit_out);

    let json = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&json);
    assert!(kc.is_empty(), "expected `len` to lower cleanly, got refusals: {:#?}\nraw: {:#?}",
        kc, lines(&json));
}

// ─── len(m) lowers to the entry count, agreeing with `dmc run` ────────────

#[test]
fn len_of_empty_map_is_zero() {
    let src = "fn main() -> i64 {\n\
               \x20   let !m = map_new()\n\
               \x20   print(len(m))\n\
               \x20   0\n\
               }\n";
    assert_run_and_jit_agree_on_i64(src, 0);
}

#[test]
fn len_of_map_after_two_sets_is_two() {
    let src = "fn main() -> i64 {\n\
               \x20   let !m = map_new()\n\
               \x20   map_set(m, \"a\", 1)\n\
               \x20   map_set(m, \"b\", 2)\n\
               \x20   print(len(m))\n\
               \x20   0\n\
               }\n";
    assert_run_and_jit_agree_on_i64(src, 2);
}

#[test]
fn len_of_map_after_reset_of_existing_key_does_not_double_count() {
    // Re-setting "a" must not grow the entry count — a re-set key overwrites
    // its existing slot on both backends (`HashMap::insert`), it does not
    // add a second one.
    let src = "fn main() -> i64 {\n\
               \x20   let !m = map_new()\n\
               \x20   map_set(m, \"a\", 1)\n\
               \x20   map_set(m, \"b\", 2)\n\
               \x20   map_set(m, \"a\", 99)\n\
               \x20   print(len(m))\n\
               \x20   0\n\
               }\n";
    assert_run_and_jit_agree_on_i64(src, 2);
}

// This is the exact minimal repro from #751.
#[test]
fn len_of_map_after_one_set_matches_issue_repro() {
    let src = "fn main() -> i64 {\n\
               \x20   let !m = map_new()\n\
               \x20   map_set(m, \"a\", 1)\n\
               \x20   len(m)\n\
               }\n";
    assert_run_and_jit_agree_on_i64(src, 1);
}

// ─── len(map_get(...)) — the OptI64 shape of #742/#743 — refuses cleanly ──

#[test]
fn len_of_map_get_result_refuses_as_located_jit_conversion_citing_721() {
    // The corpus site was `len(map_get(d, "fns"))` reading a nested map
    // value; a `map_get` result is `TyKind::Scalar(OptI64)` in the JIT
    // regardless of what the interpreter's map actually stored there
    // (#742/#721), so `len` cannot verify it as tensor/str/Map here even
    // though `d`'s "fns" value is a plain i64 below (isolating this test to
    // the `len` check alone, the same way `jit_map_nested_742.rs`'s tests
    // isolate the key/value checks from each other).
    let src = "fn main() -> i64 {\n\
               \x20   let !d = map_new()\n\
               \x20   map_set(d, \"fns\", 1)\n\
               \x20   len(map_get(d, \"fns\"))\n\
               }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(chk.status.success(), "--check failed: {}", String::from_utf8_lossy(&chk.stderr));

    // dmc run: the value stored really isn't a tensor/tuple/str/list/map at
    // runtime, so a runtime error here is expected and fine — the point is
    // it is NOT a compiler crash, and the JIT below must not report the
    // uncoded `jit error` #751 reported regardless of what `dmc run` does.
    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "expected a runtime type error, got success");
    assert!(String::from_utf8_lossy(&run.stderr).contains("len:"),
        "expected the interpreter's own len-domain check: {}", String::from_utf8_lossy(&run.stderr));

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "expected exactly one classified jit-conversion refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));

    let msg = message_of(&out);
    assert!(msg.contains("len") && msg.contains("argument"), "should name the site: {}", msg);
    assert!(msg.contains("i64?"), "should name the opaque JIT type: {}", msg);
    assert!(msg.contains("#721"), "should cite #721: {}", msg);
    assert!(!lines(&out).join("\n").contains("\"kind\":\"jit-error\""),
        "must not be the uncoded jit-error #751 reported: {:#?}", lines(&out));
}

// ─── len of any other non-tensor/str/Map type: located, never uncoded ─────

#[test]
fn len_of_bool_is_now_a_check_error_758() {
    // `dmc run` used to refuse this exact program too (`interp.rs`'s own
    // "len: requires tensor, tuple, str, list, or map"), so per #480 this
    // was a real bug in the caller's program, not a #721 representation gap —
    // but `--check` did not catch it. #758 closed that gap: `types.rs`'s new
    // `TyType::OneOf` names `len`'s admissible domain (tensor, str, Map, or
    // tuple) as a kind set, and the existing general call-arg mechanism now
    // reports `len(true)` as a located `--check`-time type error, the same
    // #745 precedent set for a map builtin's non-str key
    // (`jit_map_nested_742.rs`'s `assert_literal_key_is_now_a_check_error`).
    // Both `dmc run` and `dmc jit` type-check first (`resolve_and_check` in
    // `main.rs`), so neither the interpreter's own runtime message nor the
    // JIT's classed `jit-conversion` refusal (`len_type_refusal`) is
    // reachable from any real CLI path for this shape any more; the JIT's
    // classing code stays in place as defense in depth (the same reasoning
    // #745's own review gave for the map-key case).
    let src = "fn main() -> i64 {\n\
               \x20   len(true)\n\
               }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(!chk.status.success(),
        "--check should now refuse a concretely-typed non-tensor/str/Map/tuple \
         argument (#758): {}", String::from_utf8_lossy(&chk.stderr));
    let chk_msg = String::from_utf8_lossy(&chk.stderr);
    assert!(chk_msg.contains("arg 0")
        && chk_msg.contains("expected tensor, str, Map, or tuple")
        && chk_msg.contains("bool"),
        "expected the #758 domain check naming arg 0 and the offending type: {}", chk_msg);

    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "expected dmc run to also refuse, got success");
    assert!(String::from_utf8_lossy(&run.stderr).contains("expected tensor, str, Map, or tuple"),
        "dmc run type-checks first too, so it should surface the same #758 \
         check error rather than reaching the interpreter's own runtime len \
         check: {}", String::from_utf8_lossy(&run.stderr));

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("check".to_string(), None)],
        "dmc jit type-checks first (`resolve_and_check`), so this must surface \
         as the check error, never reach the backend's own jit-conversion \
         classing, got: {:#?}\nraw: {:#?}", kc, lines(&out));
    assert!(!lines(&out).join("\n").contains("\"kind\":\"jit-error\""),
        "must not be the uncoded jit-error #751 reported: {:#?}", lines(&out));
}

// ─── regression: len on tensor/str is unchanged ────────────────────────────

#[test]
fn len_of_str_is_unchanged() {
    let src = "fn main() -> i64 {\n\
               \x20   len(\"hello\")\n\
               }\n";
    assert_run_and_jit_agree_on_i64(src, 5);
}

#[test]
fn len_of_tensor_is_unchanged() {
    let src = "fn main() -> i64 {\n\
               \x20   let t = forge.zeros[f32, [4, 3]]\n\
               \x20   len(t)\n\
               }\n";
    assert_run_and_jit_agree_on_i64(src, 4);
}
