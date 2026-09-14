//! #742: `map_set(defs, map_get(rec, "name"), rec)` — found by `dmc test
//! --jit --all` on the downstream corpus right after #720 landed — refused
//! with an uncoded `jit error: map_set arg 2 must be str, got i64?`, #480's
//! exact defect class. Two things meet here that the JIT's map model (i64
//! values only, #721) cannot express: a `map_get` result (always
//! `TyKind::Scalar(OptI64)` in the JIT — opaque i64 plus a maybe-nil present
//! bit, regardless of what the interpreter's map actually stored) used as a
//! `map_set`/`map_get`/`map_contains` KEY, and a map used as a `map_set`
//! VALUE (storing one map inside another). `--check` accepts both shapes
//! (the checker's map model is dynamically typed, #723's match-anything
//! `Named` resolution among the reasons why) and `dmc run` executes them
//! with full interpreter semantics. Until #721's typed map values land,
//! both must be located `jit-conversion` refusals naming the opaque/nested
//! type and citing #721 — never the uncoded `jit error`.
//!
//! The VALUE half was already covered by #720's F4 fix
//! (`jit_map_get_695.rs`'s `map_set_refuses_a_{bool,float,tensor}_value_*`):
//! `map_set`'s value check refuses anything that isn't a plain integer or a
//! `fn`, and `TyKind::Map` was already caught by that same arm (`as_scalar()`
//! is `None` for `Map`, so it never needed a dedicated case) — confirmed
//! below, not assumed. What was still missing, both uncoded `err()` calls
//! before this fix:
//!
//! 1. `map_set`/`map_get`/`map_contains`'s arg-2 (key) check, `!kty.is_str()`
//!    — any non-str key fell through to a bare, unclassified `err()`. A key
//!    whose JIT type is `OptI64` (it traces back through a `map_get`) gets
//!    the located `jit-conversion`, citing #721.
//! 2. `map_get`/`map_set`/`map_contains`'s arg-1 (map) check, `!mty.is_map()`
//!    — reachable with `mty == OptI64` only through a NESTED map read
//!    (`map_get(map_get(outer, "k"), "j")`): the outer call's own arg 1 is
//!    itself a `map_get` result. Same reasoning, same fix.
//!
//! **Review D1 (2026-09-06):** the original fix left a THIRD shape — a key
//! or map argument whose JIT-known type is definitively, statically wrong
//! (an `i64`/`bool`/`f64` literal, a tensor) — falling through to the plain,
//! unclassified `err()`, on the reasoning that `dmc run` refuses it too, so
//! it must be a real bug rather than a #721 gap. That is true, but per #480
//! a real bug in an ill-typed program is exactly what must be a CHECK
//! error, never an uncoded jit error — and until the checker catches it
//! (`--check` currently accepts all of these; filed as #745, since the map
//! builtins' signatures carry no parameter types), the JIT is the only
//! thing standing between the caller and a silent divergence. So this
//! shape now gets the same located `jit-conversion` refusal as the `OptI64`
//! shape, citing #745 instead of #721 to keep the two gaps distinguishable.
//! Both checks share one helper, `Jit::map_slot_type_refusal`, so all three
//! map builtins (`map_get`, `map_set`, `map_contains`) get the same
//! treatment at both argument positions, for every offending type.

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
/// `jit_map_get_695.rs`'s identical helper for why this is a flat substring
/// walk rather than a serde_json parse (this crate has no JSON dep in tests).
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

// ─── key = a `map_get` result (#742's original repro shape) ────────────────

#[test]
fn map_set_key_from_map_get_result_refuses_as_located_jit_conversion() {
    // `rec`'s field is a plain i64 (not str) so the EARLIER `map_set(rec,
    // "code", 1)` lowers cleanly under the JIT — isolating this test to the
    // KEY check alone, rather than also tripping the (separately covered,
    // pre-existing) value-must-be-int check. `defs`'s value is a plain i64
    // for the same reason.
    let src = "fn main() -> i64 {\n\
               \x20   let !rec = map_new()\n\
               \x20   map_set(rec, \"code\", 1)\n\
               \x20   let !defs = map_new()\n\
               \x20   map_set(defs, map_get(rec, \"code\"), 99)\n\
               \x20   0\n\
               }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(chk.status.success(), "--check failed: {}", String::from_utf8_lossy(&chk.stderr));

    // dmc run: full interpreter semantics — a runtime error here (the key
    // really is an int, not a str, at runtime too) is expected and fine;
    // the point is it is NOT a compiler crash, and the JIT below must not
    // report an uncoded `jit error` regardless of what `dmc run` does with
    // the value.
    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "expected a runtime type error, got success");
    assert!(String::from_utf8_lossy(&run.stderr).contains("key must be str"),
        "expected the interpreter's own key-type check: {}", String::from_utf8_lossy(&run.stderr));

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "expected exactly one classified jit-conversion refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));

    let msg = message_of(&out);
    assert!(msg.contains("map_set") && msg.contains("arg 2"), "should name the site: {}", msg);
    assert!(msg.contains("i64?"), "should name the opaque JIT type: {}", msg);
    assert!(msg.contains("#721"), "should cite #721: {}", msg);
    assert!(!lines(&out).join("\n").contains("\"kind\":\"jit-error\""),
        "must not be the uncoded jit-error #742 reported: {:#?}", lines(&out));
}

// ─── key = a definitively-wrong static type (review D2: #745 landed) ──────
//
// An `i64`/`bool`/`f64`/tensor LITERAL key is not ambiguous the way a
// `map_get` result is: the JIT knows statically it is not `str`, and `dmc
// run` refuses the exact same program too (`interp.rs`'s own "key must be
// str" check) — so this really is a bug in the caller's program, not a
// #721 representation gap. Per #480 that is exactly what must be a CHECK
// error, not an uncoded jit error — and Review D1 (2026-09-06, above) noted
// `--check` did not catch it yet and filed it as #745, so at the time the
// JIT's own located `jit-conversion` refusal was the only thing standing
// between the caller and a silent divergence.
//
// #745 has since landed (`types.rs::builtin_sig`'s keyed map builtins now
// carry a concrete `str` param type for arg 1): a genuinely concrete,
// non-map-annotated, non-`Unit` key type — which an `i64`/`bool`/`f64`
// LITERAL is, but a `forge.zeros[..]` tensor CONSTRUCTOR is not (arena
// constructors type `Unknown` regardless of annotation, #248) — is now a
// `--check` error in its own right. And `dmc jit` always type-checks first
// (`resolve_and_check` in `main.rs`), so for the three literal shapes below
// the located `jit-conversion` refusal this test used to see is now
// unreachable from any real CLI path: the program never gets past `--check`
// to reach the backend at all. The tensor shape is unaffected — it still
// reaches the backend, still gets the classed refusal, unchanged from D1.
//
// The JIT's own per-argument classing (`Jit::map_slot_type_refusal`) is left
// in place rather than deleted: it is still correct if anything ever lowers
// this shape without checking first (a future direct-to-backend entry
// point, a test harness that skips `resolve_and_check`), so it stands as
// defense in depth even though the ordinary `dmc jit`/`dmc test --jit`
// paths can no longer reach it for these three shapes.

fn assert_key_of_wrong_type_is_classed_jit_conversion(key_expr: &str, expect_ty_substr: &str) {
    let src = format!(
        "fn main() -> i64 {{\n\
         \x20   let !m = map_new()\n\
         \x20   map_set(m, {}, 1)\n\
         \x20   0\n\
         }}\n",
        key_expr);
    let probe = Probe::new(&src);

    let chk = probe.run(&["--check"]);
    assert!(chk.status.success(),
        "--check should still accept this (arg stays Unknown, #248): {}",
        String::from_utf8_lossy(&chk.stderr));

    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "expected a runtime type error, got success");
    assert!(String::from_utf8_lossy(&run.stderr).contains("key must be str"),
        "expected the interpreter's own key-type check: {}", String::from_utf8_lossy(&run.stderr));

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "a statically-wrong-type key that `--check` still can't see (arg stays \
         Unknown, #248) must still be a located jit-conversion, never the \
         uncoded jit-error, got: {:#?}\nraw: {:#?}", kc, lines(&out));
    let msg = message_of(&out);
    assert!(msg.contains("map_set") && msg.contains("arg 2"), "should name the site: {}", msg);
    assert!(msg.contains("must be str") && msg.contains(expect_ty_substr),
        "should name the offending type: {}", msg);
    assert!(msg.contains("#745"), "should cite #745, not #721: {}", msg);
    assert!(!msg.contains("#721"), "not the #721 gap: {}", msg);
    assert!(!lines(&out).join("\n").contains("\"kind\":\"jit-error\""),
        "must not be the uncoded jit-error #742 reported: {:#?}", lines(&out));
}

/// A key whose type `--check` now pins down concretely (#745: arg 1 is `str`
/// in `types.rs::builtin_sig`) is a `--check`-time error, full stop — `dmc
/// jit` type-checks first and never reaches the backend, so the located
/// `jit-conversion` refusal this used to exercise (review D1, above) is
/// unreachable from source for these three shapes. The JIT's own classing
/// stays as defense in depth (see the D2 note above `assert_key_of_wrong_
/// type_is_classed_jit_conversion`) — this just confirms `--check` is now
/// the layer that actually catches it, and names the same slot (`arg 1`,
/// 0-indexed, vs. the JIT's own `arg 2`, 1-indexed — a pre-existing
/// numbering difference between the two backends' conventions).
fn assert_literal_key_is_now_a_check_error(key_expr: &str, expect_ty_substr: &str) {
    let src = format!(
        "fn main() -> i64 {{\n\
         \x20   let !m = map_new()\n\
         \x20   map_set(m, {}, 1)\n\
         \x20   0\n\
         }}\n",
        key_expr);
    let probe = Probe::new(&src);

    let chk = probe.run(&["--check"]);
    assert!(!chk.status.success(),
        "--check should now refuse a concretely-typed non-str key (#745): {}",
        String::from_utf8_lossy(&chk.stderr));
    let chk_msg = String::from_utf8_lossy(&chk.stderr);
    assert!(chk_msg.contains("arg 1") && chk_msg.contains("str") && chk_msg.contains(expect_ty_substr),
        "expected the #745 str-key check naming arg 1 and the offending type: {}", chk_msg);

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("check".to_string(), None)],
        "dmc jit type-checks first (`resolve_and_check`), so this must surface \
         as the check error, never reach the backend's own jit-conversion \
         classing, got: {:#?}\nraw: {:#?}", kc, lines(&out));
}

#[test]
fn map_set_key_i64_literal_is_now_a_check_error_745() {
    assert_literal_key_is_now_a_check_error("5", "integer");
}

#[test]
fn map_set_key_bool_literal_is_now_a_check_error_745() {
    assert_literal_key_is_now_a_check_error("true", "bool");
}

#[test]
fn map_set_key_f64_literal_is_now_a_check_error_745() {
    assert_literal_key_is_now_a_check_error("1.5", "float");
}

#[test]
fn map_set_key_tensor_is_a_real_bug_but_still_a_classed_refusal() {
    assert_key_of_wrong_type_is_classed_jit_conversion("forge.zeros[f32, [2]]", "Tensor");
}

// ─── key = a str variable (must still lower — no regression) ──────────────

#[test]
fn map_set_key_str_variable_still_lowers() {
    let src = "fn main() -> i64 {\n\
               \x20   let !m = map_new()\n\
               \x20   let k = \"name\"\n\
               \x20   map_set(m, k, 42)\n\
               \x20   map_get(m, k)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert!(kc.is_empty(), "expected zero refusals, got: {:#?}\nraw: {:#?}", kc, lines(&out));

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "jit failed: {}", String::from_utf8_lossy(&jit.stderr));
}

// ─── value = a map (nested maps) ────────────────────────────────────────────

#[test]
fn map_set_value_map_from_map_new_refuses_as_located_jit_conversion() {
    let src = "fn main() -> i64 {\n\
               \x20   let !rec = map_new()\n\
               \x20   map_set(rec, \"a\", 1)\n\
               \x20   let !defs = map_new()\n\
               \x20   map_set(defs, \"rec\", rec)\n\
               \x20   0\n\
               }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(chk.status.success(), "--check failed: {}", String::from_utf8_lossy(&chk.stderr));

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "expected exactly one classified jit-conversion refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));
    let msg = message_of(&out);
    assert!(msg.contains("Map"), "should name the value's type: {}", msg);
    assert!(msg.contains("#721"), "should cite #721: {}", msg);
}

#[test]
fn map_set_value_map_from_bare_ctor_inline_refuses_as_located_jit_conversion() {
    // The bare `map()` constructor (SPEC.md's Map-type alias for `map_new()`,
    // #695) fed directly as `map_set`'s value argument, with no intervening
    // `let` — a distinct shape from the named-variable case above (a fresh
    // call expression, not a local read).
    let src = "fn main() -> i64 {\n\
               \x20   let !defs = map_new()\n\
               \x20   map_set(defs, \"rec\", map())\n\
               \x20   0\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "expected exactly one classified jit-conversion refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));
    let msg = message_of(&out);
    assert!(msg.contains("Map"), "should name the value's type: {}", msg);
    assert!(msg.contains("#721"), "should cite #721: {}", msg);
}

// ─── a nested read: map_get(map_get(outer, "k"), "j") ──────────────────────

#[test]
fn nested_map_get_read_refuses_cleanly_not_as_uncoded_jit_error() {
    // `outer`'s "k" field is a plain i64 so the setup lowers under the JIT;
    // the outer `map_get`'s arg 1 is itself a `map_get` result (always
    // `OptI64` in the JIT), which is exactly the nested-map shape #742
    // names. `dmc run` genuinely fails here too (`outer["k"]` really is an
    // int at runtime, not a map — the interpreter's own "first arg must be
    // map" check fires) — same reasoning as the i64-literal-key case: this
    // is what the type system can prove, not what it can't, but the JIT
    // must still never answer with the uncoded `jit error` #742 reported.
    let src = "fn main() -> i64 {\n\
               \x20   let !outer = map_new()\n\
               \x20   map_set(outer, \"k\", 7)\n\
               \x20   let v = map_get(map_get(outer, \"k\"), \"j\")\n\
               \x20   print(v)\n\
               \x20   0\n\
               }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(chk.status.success(), "--check failed: {}", String::from_utf8_lossy(&chk.stderr));

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "expected exactly one classified jit-conversion refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));
    let msg = message_of(&out);
    assert!(msg.contains("map_get") && msg.contains("arg 1"), "should name the site: {}", msg);
    assert!(msg.contains("i64?"), "should name the opaque JIT type: {}", msg);
    assert!(msg.contains("#721"), "should cite #721: {}", msg);
    assert!(!lines(&out).join("\n").contains("\"kind\":\"jit-error\""),
        "must not be the uncoded jit-error #742 reported: {:#?}", lines(&out));
}

// ─── map_contains gets the same treatment (both argument positions) ───────

#[test]
fn map_contains_key_from_map_get_result_refuses_as_located_jit_conversion() {
    let src = "fn main() -> i64 {\n\
               \x20   let !rec = map_new()\n\
               \x20   map_set(rec, \"code\", 1)\n\
               \x20   let !defs = map_new()\n\
               \x20   if map_contains(defs, map_get(rec, \"code\")) { 1 } else { 0 }\n\
               }\n";
    let probe = Probe::new(src);

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "expected exactly one classified jit-conversion refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));
    let msg = message_of(&out);
    assert!(msg.contains("map_contains") && msg.contains("arg 2"), "should name the site: {}", msg);
    assert!(msg.contains("#721"), "should cite #721: {}", msg);
}
