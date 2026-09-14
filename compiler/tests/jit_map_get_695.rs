//! #695: `dmc jit`'s `map_get` lowering rejected values `dmc run` accepts,
//! with a bare uncoded `jit error` (#480's defect class) rather than a
//! located, classified refusal — found by the #684 blocker-depth review of
//! PR #690 (`dmc jit --json --all` over `examples/`) on three sites:
//! `examples/demonic_lexer.dmc:128:30` and `examples/lispy_vm.dmc:29:21`
//! (`` `map_get` arg 1 must be a Map, got `map` ``), and
//! `examples/package_manifest.dmc:36:5` (`` ...got `nil` ``).
//!
//! Two independent gaps, both in `jit.rs`'s Map handling, produced the three
//! messages:
//!
//! 1. **`ty_from_ast`** recognized only `Type::Named { name: "Map", .. }`
//!    (capitalized) as the Map type. But `types.rs` — the checker `dmc
//!    --check`/`dmc run` actually type-check against — parses only
//!    lowercase `"map"` into `TyType::Map` (and its own `Display` renders
//!    that variant back as `"map"`, lowercase). A signature written `t: map`
//!    (the only spelling `--check` admits) fell into the generic `Named`
//!    arm and became `TyKind::Model("map", [])` — an unknown model, not a
//!    Map at all — so `map_get`'s `is_map()` check failed with exactly
//!    "must be a Map, got `map`". This is sites 1 and 2.
//!
//! 2. **The bare `map()` constructor** — SPEC.md's Map-type paragraph names
//!    it as `map_new()`'s alias — was entirely absent from the JIT's builtin
//!    dispatch (only `map_new` was wired). A program using `map()` refused
//!    as `jit-unknown-fn` before ever reaching `map_get`
//!    (`examples/package_manifest.dmc`'s `dependency`). Once that's fixed,
//!    a SEPARATE, deeper gap surfaces: `dependency` has no `-> T` return
//!    type, and `declare_fn`'s signature builder assumes an omitted return
//!    type means "returns nothing" — correct for a call made only for
//!    effect, wrong the moment a caller keeps the result. Before this fix,
//!    `let dep = dependency(..)` silently bound `dep` to a bare `nil`
//!    regardless of what `dependency`'s body actually produced, and the
//!    mistype surfaced three lines away as the uncoded `map_get arg 1 must
//!    be a Map, got nil` (site 3). This shape is genuinely outside the
//!    JIT's lowered subset (it has no return-type inference), so it is now
//!    a located `jit-signature` refusal, naming the callee, raised before
//!    `map_get` (or anything else) ever sees the bogus value.
//!
//! Each fixture below drives the real `dmc` binary, mirroring
//! `jit_global_const_696.rs`'s harness.
//!
//! #720 (adversarial review of this fix, 2026-09-06, REWORK) found the first
//! cut of gap 2 wrong in four ways, all addressed below:
//!
//! * **F1** — the doc comment (and this file's own header, above) claimed
//!   `Map` (capitalized) was never accepted by `--check`-admitted source.
//!   #723 (filed by the review) found `check.rs`'s `resolve_type` leaves ANY
//!   undeclared `Named` type unresolved and accepted — `t: Map`, `t: map`,
//!   and `t: Zorkmid` all pass `--check` alike. The JIT still recognizes
//!   both `map` and `Map` (see `jit.rs`'s `ty_from_ast` comment) — that part
//!   was never wrong — only the claim about *why* was.
//! * **F2** — the `"map" | "map_new"` builtin-dispatch arm shadowed a
//!   user-declared top-level fn of the same name: `fn map(f: fn(i64)->i64,
//!   x: i64) -> i64 { f(x) }` refused `map_new takes no arguments` instead
//!   of calling the user's `map`. Fixed: a declared `map`/`map_new` wins.
//! * **F3** — the omitted-return guard lived in `lower_let` and refused
//!   EVERY such binding unconditionally, which is wrong the moment the
//!   callee genuinely returns nothing (`fn side(v: i64) { print(v) }`,
//!   legitimately `nil` on both backends). It was also scope-blind (a LOCAL
//!   binding sharing a top-level fn's name got refused too) and missed a
//!   model method's mangled name entirely, and it never caught the mistyped
//!   value in ARGUMENT position (`map_get(dep(7), "v")`). Fixed: `lower_let`
//!   only records provenance now; `map_get`/`map_set`/`map_contains` are
//!   where the refusal actually fires, exactly when a value that traces to
//!   an omitted-return call is asked to be a `Map` and isn't.
//! * **F4** — `map_set` coerced ANY value to `i64`, silently: `true` stored
//!   as `1`, `1.5` stored as `1`, a tensor stored as its raw pointer (only
//!   `str` refused). Fixed: `map_set` now refuses every non-integer value
//!   with a located `jit-conversion` (#721 tracks the eventual typed-value
//!   representation that would let this lower for real).
//!
//! #720 review of PR #720 itself (2026-09-06, this fix) found two more holes
//! in F3's guard, both closed below:
//!
//! * **Block-scoped shadow** — `omitted_ret_provenance` is keyed by bare
//!   source name, and `lower_let` unconditionally drops any existing entry
//!   for that name before deciding whether to add its own. A nested block's
//!   `let` of the SAME name — `let d = dep(7)` outside, `if true { let d = 5
//!   .. }` inside — reached `lower_let` for the inner `d` too, which dropped
//!   the outer `d`'s provenance and never restored it once the block ended
//!   (`lower_let` has no notion of "this drop is temporary"). `map_get(d,
//!   ..)` after the block then saw no provenance at all and fell to the
//!   uncoded `jit-error` #695 reported in the first place — the same defect
//!   class, one line away. Fixed by giving `omitted_ret_provenance` the same
//!   lexical-scope treatment `locals`/`local_tys`/`field_aliases` already
//!   get: `LocalScope` (`scope_enter`/`scope_exit`) now saves and restores it
//!   too, so the inner `let d` can never outlive its block's provenance.
//! * **Shape-generic model method (#717)** — `resolve_direct_call_name`
//!   deliberately answered `None` for a shape-generic model's method,
//!   leaving `let d = s.build(9)` (for `model B[N] { fn build(self, k) {
//!   ..map.. } }`, no `-> T`) with no provenance and the same uncoded
//!   `jit-error` at `map_get`. Closed for the narrow, common slice: a
//!   bracket-less method call whose OWN shape params are empty resolves
//!   under the identical mangled name `lower_model_method_call` instantiates
//!   (`mangle_mono_method_name` with an empty method-side binding), computed
//!   from the receiver's own `type_args` alone — nothing is lowered twice. A
//!   second, independent gap surfaced closing this one: `omitted_ret_fns` was
//!   only ever populated by `declare_fn`'s EAGER pass-1 declaration; a
//!   template method is declared LAZILY, per instantiation, via
//!   `declare_template_signature` inside `lower_model_method_call`, which
//!   never recorded an omitted `-> T` there at all. Fixed by recording it at
//!   that same site, and by moving `lower_let`'s provenance check to AFTER
//!   `l.value` is lowered (previously before, which never mattered while
//!   every entry was eager) so a just-lazily-instantiated method's entry is
//!   already in `omitted_ret_fns` by the time the check runs.

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

/// Every JSON Lines diagnostic parsed out of a `--json` run's stderr, as
/// serde_json-free flat string checks (this crate has no JSON dep in tests;
/// substring assertions on the fixed wire-order fields are exact enough).
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

// ─── site 1 / 2 shape: a `map`-typed (lowercase) parameter ──────────────────

#[test]
fn map_typed_parameter_lowers_instead_of_jit_error() {
    // examples/demonic_lexer.dmc:128 (`fn tok_kind(t: map) -> str`) and
    // lispy_vm.dmc:28 (`fn alloc_num(heap: map, ...)`) shape: a parameter
    // written `t: map`. Before #695 this refused with the exact wording
    // #695 reported: `map_get` arg 1 must be a Map, got `map`, and it was an
    // uncoded `jit error`, not a classified refusal.
    let src = "fn get_k(t: map) -> i64 { map_get(t, \"k\") }\n\
               fn main() -> i64 {\n\
               \x20   let !m = map_new()\n\
               \x20   map_set(m, \"k\", 42)\n\
               \x20   get_k(m)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(stdout(&run), "\n=> 42\n");

    let chk = probe.run(&["--check"]);
    assert!(chk.status.success(), "--check failed: {}", String::from_utf8_lossy(&chk.stderr));

    // `dmc jit` (no --json) must produce the ANSWER, not a refusal at all.
    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "jit failed: {}", String::from_utf8_lossy(&jit.stderr));
    assert_eq!(stdout(&jit), "=> 42\n");

    // And under --json --all, no diagnostic at all — fully lowered.
    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert!(kc.is_empty(), "expected zero refusals, got: {:#?}\nraw: {:#?}", kc, lines(&out));
}

// ─── site 3 shape: `map()`, then a `let` binding an omitted-return-type call ─

#[test]
fn bare_map_ctor_lowers() {
    // SPEC.md's Map-type paragraph: `map_new()` (alias: `map()`). Before
    // #695 the bare spelling was absent from the JIT's dispatch entirely —
    // `jit-unknown-fn: unknown function 'map'` — which is what blocked
    // examples/package_manifest.dmc's `dependency` from ever reaching its
    // own `map_get`/`map_set` calls.
    let src = "fn main() -> i64 { let !m = map()  map_set(m, \"k\", 7)  map_get(m, \"k\") }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert!(kc.is_empty(), "expected zero refusals, got: {:#?}\nraw: {:#?}", kc, lines(&out));

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "jit failed: {}", String::from_utf8_lossy(&jit.stderr));
    assert_eq!(stdout(&jit), "=> 7\n");
}

#[test]
fn omitted_return_type_let_bound_call_now_refuses_at_check_753() {
    // examples/package_manifest.dmc:35 shape: `dependency` has no `-> T`.
    // This test used to document `dmc run`/`--check` passing this program
    // clean — the interpreter doesn't care about declared return types, and
    // `--check`'s own return-type comparison (`declared_ret.compatible_with
    // (&effective_body_ty)`, the same check that makes `fn make() { 7 }`
    // error) never even ran the `Unit`-vs-`Map` comparison: `compatible_with`'s
    // old `TyType::Map` arm (#204) returned `true` for ANY `Unit`-vs-`Map`
    // pair. So the JIT's own dedicated `declare_fn` signature-builder gap
    // (assuming an omitted return type means "returns nothing") was, for this
    // exact shape, the ONLY thing catching the mistype — hence the
    // `jit-signature` refusal this test used to assert.
    //
    // #753 closes the checker-level hole instead: `compatible_with` now
    // refuses the `(Unit, Map)` pair specifically (types.rs ~146), so
    // `dependency`'s own declaration is a check error, and `--check`/`dmc
    // run`/`dmc jit` (all three call `resolve_and_check` first) now refuse
    // this program before the JIT's own `map_get`-argument provenance check
    // ever runs. That JIT logic is unreachable from any real CLI path for
    // this repro now, the same way #752 left its map-key-literal JIT classing
    // in place as defense in depth rather than removing it — out of scope
    // here (a JIT change, not a checker one).
    let src = "fn dependency(v: i64) {\n\
               \x20   let !m = map()\n\
               \x20   map_set(m, \"v\", v)\n\
               \x20   m\n\
               }\n\
               fn main() -> i64 {\n\
               \x20   let dep = dependency(7)\n\
               \x20   map_get(dep, \"v\")\n\
               }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(!chk.status.success(),
        "--check should now refuse a map-tailed omitted-return fn (#753): {}",
        String::from_utf8_lossy(&chk.stderr));
    let chk_msg = String::from_utf8_lossy(&chk.stderr);
    assert!(chk_msg.contains("dependency") && chk_msg.contains("returns `nil`")
                && chk_msg.contains("body produces `map`"),
        "expected the #753 return-type mismatch naming `dependency`: {}", chk_msg);

    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "dmc run also type-checks first and must refuse");

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    // The mistyped value now trips more than one check (the fn's own
    // return-type mismatch, plus the map-builtin's dedicated arg-0 check AND
    // the general per-arg mechanism — both now fire for a `nil` arg 0, since
    // `compatible_with` no longer wildcards that specific pair either) — all
    // of them "check" kind, none of them the JIT's own classing.
    assert!(!kc.is_empty() && kc.iter().all(|(k, _)| k == "check"),
        "dmc jit type-checks first (`resolve_and_check`), so this must surface \
         only as check error(s), never reach the JIT's own signature classing, \
         got: {:#?}\nraw: {:#?}", kc, lines(&out));
}

// ─── #720 review fixes ───────────────────────────────────────────────────────

/// Pull the value after the last `"=> "` in `dmc run`/`dmc jit`'s stdout, so a
/// comparison between the two doesn't depend on the leading-newline quirk
/// `dmc run` has and `dmc jit` doesn't (see the asymmetric literals above).
fn answer(out: &str) -> &str {
    let idx = out.rfind("=> ").expect("no `=> ` in output");
    out[idx + 3..].trim_end_matches('\n')
}

// F2: a user-declared top-level fn named `map` must win over the `map_new`
// alias — before the fix, `map(square, 3)` refused `map_new takes no
// arguments` instead of calling the program's own `map`.
#[test]
fn user_declared_fn_named_map_wins_over_the_builtin_alias() {
    let src = "fn square(x: i64) -> i64 { x * x }\n\
               fn map(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }\n\
               fn main() -> i64 { map(square, 3) }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "jit failed (still shadowed by the `map_new` alias?): {}",
        String::from_utf8_lossy(&jit.stderr));
    assert_eq!(answer(&stdout(&jit)), answer(&stdout(&run)));
    assert_eq!(answer(&stdout(&jit)), "9");
}

// F3a: an omitted-return-type fn whose body genuinely produces nothing.
// Binding its call's result is not a bug — `dmc run` agrees the value is
// `nil` — so the JIT must not refuse the `let` just because the callee has
// no `-> T`.
#[test]
fn omitted_return_binding_a_genuinely_nil_call_is_not_refused() {
    let src = "fn side(v: i64) { print(v) }\n\
               fn main() -> i64 { let z = side(3)  7 }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(),
        "jit refused a binding whose callee is genuinely nil-returning: {}",
        String::from_utf8_lossy(&jit.stderr));
    assert_eq!(answer(&stdout(&jit)), answer(&stdout(&run)));
}

// F3b: a LOCAL binding that shadows a top-level omitted-return-type fn of
// the same name must resolve to the local, never to the top-level fn's
// `omitted_ret_fns` entry.
#[test]
fn local_binding_shadowing_an_omitted_return_top_level_fn_is_not_confused_for_it() {
    let src = "fn dep(v: i64) { print(v) }\n\
               fn main() -> i64 {\n\
               \x20   let dep = fn(x: i64) -> i64 { x * 2 }\n\
               \x20   let y = dep(21)\n\
               \x20   y\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(),
        "jit refused a local closure because a same-named top-level fn omits `-> T`: {}",
        String::from_utf8_lossy(&jit.stderr));
    assert_eq!(answer(&stdout(&jit)), answer(&stdout(&run)));
    assert_eq!(answer(&stdout(&jit)), "42");
}

// F3c: a model METHOD with no declared return type, consumed through
// `map_get`. `omitted_ret_fns` keys a method by its mangled `M$$method`
// name; before the fix nothing resolved a method-call initializer back to
// that key at all, so this shape never got the located refusal — it hit
// the old uncoded `jit error` (#480's class) instead.
#[test]
fn omitted_return_method_consumed_via_map_get_now_refuses_at_check_753() {
    // Same #753 shape as the plain-fn case above, on a model method instead:
    // `build` has no `-> T` and tails a real map, so `--check` now refuses at
    // the method's own declaration before the JIT's method-specific
    // provenance handling (the reason this test originally existed) is ever
    // reached.
    let src = "model Store {\n\
               \x20   tag: i64\n\
               \x20   fn build(self, v: i64) {\n\
               \x20       let !m = map()\n\
               \x20       map_set(m, \"v\", v)\n\
               \x20       m\n\
               \x20   }\n\
               }\n\
               fn main() -> i64 {\n\
               \x20   let s = Store { tag: 1 }\n\
               \x20   let b = s.build(9)\n\
               \x20   map_get(b, \"v\")\n\
               }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(!chk.status.success(),
        "--check should now refuse a map-tailed omitted-return method (#753): {}",
        String::from_utf8_lossy(&chk.stderr));
    let chk_msg = String::from_utf8_lossy(&chk.stderr);
    assert!(chk_msg.contains("build") && chk_msg.contains("returns `nil`")
                && chk_msg.contains("body produces `map`"),
        "expected the #753 return-type mismatch naming `build`: {}", chk_msg);

    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "dmc run also type-checks first and must refuse");

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    // The mistyped value now trips more than one check (the fn's own
    // return-type mismatch, plus the map-builtin's dedicated arg-0 check AND
    // the general per-arg mechanism — both now fire for a `nil` arg 0, since
    // `compatible_with` no longer wildcards that specific pair either) — all
    // of them "check" kind, none of them the JIT's own classing.
    assert!(!kc.is_empty() && kc.iter().all(|(k, _)| k == "check"),
        "dmc jit type-checks first (`resolve_and_check`), so this must surface \
         only as check error(s), never reach the JIT's own signature classing, \
         got: {:#?}\nraw: {:#?}", kc, lines(&out));
}

// This PR's fix, gap 1: a nested block's `let` of the SAME name as an outer
// omitted-return-provenance binding used to wipe that outer entry for good —
// `omitted_ret_provenance` is keyed by bare name, not by Cranelift `Variable`,
// and `lower_let` drops any existing entry for a name before deciding whether
// to add its own, with no notion of "this drop is scoped to a block". `map_get
// (d, ..)` after the block then saw no provenance and fell all the way back
// to the uncoded `jit-error` #695 reported in the first place — one line
// away from #695's exact defect class. Fixed by giving `omitted_ret_provenance`
// the same `LocalScope` save/restore treatment `locals`/`local_tys`/
// `field_aliases` already get.
#[test]
fn omitted_return_provenance_survives_a_block_scoped_shadow_of_the_same_name() {
    // Same #753 shape once more (`dep` has no `-> T` and tails a real map):
    // `--check` now refuses at `dep`'s own declaration, before ever reaching
    // the JIT's block-scoped-shadow provenance handling this test was
    // originally written to exercise.
    let src = "fn dep(v: i64) {\n\
               \x20   let !m = map()\n\
               \x20   map_set(m, \"v\", v)\n\
               \x20   m\n\
               }\n\
               fn main() -> i64 {\n\
               \x20   let d = dep(7)\n\
               \x20   if true {\n\
               \x20       let d = 5\n\
               \x20       print(d)\n\
               \x20   }\n\
               \x20   map_get(d, \"v\")\n\
               }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(!chk.status.success(),
        "--check should now refuse a map-tailed omitted-return fn (#753): {}",
        String::from_utf8_lossy(&chk.stderr));
    let chk_msg = String::from_utf8_lossy(&chk.stderr);
    assert!(chk_msg.contains("dep") && chk_msg.contains("returns `nil`")
                && chk_msg.contains("body produces `map`"),
        "expected the #753 return-type mismatch naming `dep`: {}", chk_msg);

    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "dmc run also type-checks first and must refuse");

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    // The mistyped value now trips more than one check (the fn's own
    // return-type mismatch, plus the map-builtin's dedicated arg-0 check AND
    // the general per-arg mechanism — both now fire for a `nil` arg 0, since
    // `compatible_with` no longer wildcards that specific pair either) — all
    // of them "check" kind, none of them the JIT's own classing.
    assert!(!kc.is_empty() && kc.iter().all(|(k, _)| k == "check"),
        "dmc jit type-checks first (`resolve_and_check`), so this must surface \
         only as check error(s), never reach the JIT's own signature classing, \
         got: {:#?}\nraw: {:#?}", kc, lines(&out));
}

// This PR's fix, gap 2 (#717 follow-up): a shape-generic model's method with
// no declared return type, consumed through `map_get` — the same shape as
// the F3c test above, but on a model that is itself shape-generic
// (`model B[N]`) rather than plain. `resolve_direct_call_name` used to
// answer `None` unconditionally for ANY shape-generic model's method,
// deliberately (see its old doc comment), which left this shape with no
// provenance and the same uncoded `jit-error` as every other case in this
// file started from. Closed for the narrow slice that needs no argument- or
// bracket-lowering to resolve: a bracket-less call whose method declares no
// shape parameter of its own (`build` here declares only ordinary value
// params) resolves under the exact mangled name `lower_model_method_call`
// instantiates, computed from the receiver's own `type_args` alone.
#[test]
fn omitted_return_shape_generic_model_method_consumed_via_map_get_now_refuses_at_check_753() {
    // Same #753 shape on a shape-generic model's method: `build` has no
    // `-> T` and tails a real map, so `--check` now refuses at `build`'s own
    // declaration, before the JIT's shape-generic-method resolution (this
    // test's original subject) is ever reached.
    let src = "model B[N] {\n\
               \x20   tag: i64\n\
               \x20   fn build(self, k: i64) {\n\
               \x20       let !m = map()\n\
               \x20       map_set(m, \"v\", k)\n\
               \x20       m\n\
               \x20   }\n\
               }\n\
               fn main() -> i64 {\n\
               \x20   let s = B[9] { tag: 1 }\n\
               \x20   let d = s.build(9)\n\
               \x20   map_get(d, \"v\")\n\
               }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(!chk.status.success(),
        "--check should now refuse a map-tailed omitted-return method (#753): {}",
        String::from_utf8_lossy(&chk.stderr));
    let chk_msg = String::from_utf8_lossy(&chk.stderr);
    assert!(chk_msg.contains("build") && chk_msg.contains("returns `nil`")
                && chk_msg.contains("body produces `map`"),
        "expected the #753 return-type mismatch naming `build`: {}", chk_msg);

    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "dmc run also type-checks first and must refuse");

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    // The mistyped value now trips more than one check (the fn's own
    // return-type mismatch, plus the map-builtin's dedicated arg-0 check AND
    // the general per-arg mechanism — both now fire for a `nil` arg 0, since
    // `compatible_with` no longer wildcards that specific pair either) — all
    // of them "check" kind, none of them the JIT's own classing.
    assert!(!kc.is_empty() && kc.iter().all(|(k, _)| k == "check"),
        "dmc jit type-checks first (`resolve_and_check`), so this must surface \
         only as check error(s), never reach the JIT's own signature classing, \
         got: {:#?}\nraw: {:#?}", kc, lines(&out));
}

// F3d: the mistyped value reached `map_get`/`map_contains` in ARGUMENT
// position — never bound to a name at all — which the original `lower_let`-
// only guard could not see.
#[test]
fn omitted_return_call_as_a_direct_map_get_argument_now_refuses_at_check_753() {
    // Same #753 shape, with the mistyped call inline in argument position
    // (never bound to a name) rather than via a `let` — `--check` refuses at
    // `dependency`'s own declaration regardless, before the JIT's argument-
    // position provenance handling (this test's original subject) is ever
    // reached.
    let src = "fn dependency(v: i64) {\n\
               \x20   let !m = map()\n\
               \x20   map_set(m, \"v\", v)\n\
               \x20   m\n\
               }\n\
               fn main() -> i64 { map_get(dependency(7), \"v\") }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(!chk.status.success(),
        "--check should now refuse a map-tailed omitted-return fn (#753): {}",
        String::from_utf8_lossy(&chk.stderr));
    let chk_msg = String::from_utf8_lossy(&chk.stderr);
    assert!(chk_msg.contains("dependency") && chk_msg.contains("returns `nil`")
                && chk_msg.contains("body produces `map`"),
        "expected the #753 return-type mismatch naming `dependency`: {}", chk_msg);

    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "dmc run also type-checks first and must refuse");

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    // The mistyped value now trips more than one check (the fn's own
    // return-type mismatch, plus the map-builtin's dedicated arg-0 check AND
    // the general per-arg mechanism — both now fire for a `nil` arg 0, since
    // `compatible_with` no longer wildcards that specific pair either) — all
    // of them "check" kind, none of them the JIT's own classing.
    assert!(!kc.is_empty() && kc.iter().all(|(k, _)| k == "check"),
        "dmc jit type-checks first (`resolve_and_check`), so this must surface \
         only as check error(s), never reach the JIT's own signature classing, \
         got: {:#?}\nraw: {:#?}", kc, lines(&out));
}

#[test]
fn omitted_return_call_as_a_map_contains_condition_argument_now_refuses_at_check_753() {
    // Same #753 shape once more, this time consumed via `map_contains`'s
    // condition-position argument — `--check` refuses at `dependency`'s own
    // declaration regardless of where the mistyped value is eventually used.
    let src = "fn dependency(v: i64) {\n\
               \x20   let !m = map()\n\
               \x20   map_set(m, \"v\", v)\n\
               \x20   m\n\
               }\n\
               fn main() -> i64 {\n\
               \x20   if map_contains(dependency(1), \"v\") { 1 } else { 0 }\n\
               }\n";
    let probe = Probe::new(src);

    let chk = probe.run(&["--check"]);
    assert!(!chk.status.success(),
        "--check should now refuse a map-tailed omitted-return fn (#753): {}",
        String::from_utf8_lossy(&chk.stderr));
    let chk_msg = String::from_utf8_lossy(&chk.stderr);
    assert!(chk_msg.contains("dependency") && chk_msg.contains("returns `nil`")
                && chk_msg.contains("body produces `map`"),
        "expected the #753 return-type mismatch naming `dependency`: {}", chk_msg);

    let run = probe.run(&["run"]);
    assert!(!run.status.success(), "dmc run also type-checks first and must refuse");

    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    // The mistyped value now trips more than one check (the fn's own
    // return-type mismatch, plus the map-builtin's dedicated arg-0 check AND
    // the general per-arg mechanism — both now fire for a `nil` arg 0, since
    // `compatible_with` no longer wildcards that specific pair either) — all
    // of them "check" kind, none of them the JIT's own classing.
    assert!(!kc.is_empty() && kc.iter().all(|(k, _)| k == "check"),
        "dmc jit type-checks first (`resolve_and_check`), so this must surface \
         only as check error(s), never reach the JIT's own signature classing, \
         got: {:#?}\nraw: {:#?}", kc, lines(&out));
}

// F4: `map_set` must REFUSE a non-integer value (`jit-conversion`) rather
// than silently coercing it — a `bool` collapsing to 0/1, a float
// truncating, and a tensor's raw pointer being stored as a number are all
// silently wrong, not merely unsupported.
#[test]
fn map_set_refuses_a_bool_value_instead_of_coercing_it() {
    let src = "fn main() -> i64 {\n\
               \x20   let !m = map()\n\
               \x20   map_set(m, \"k\", true)\n\
               \x20   0\n\
               }\n";
    let probe = Probe::new(src);
    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "expected a classified jit-conversion refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));
}

#[test]
fn map_set_refuses_a_float_value_instead_of_truncating_it() {
    let src = "fn main() -> i64 {\n\
               \x20   let !m = map()\n\
               \x20   map_set(m, \"k\", 1.5)\n\
               \x20   0\n\
               }\n";
    let probe = Probe::new(src);
    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "expected a classified jit-conversion refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));
}

#[test]
fn map_set_refuses_a_tensor_value_instead_of_storing_its_raw_pointer() {
    let src = "fn main() -> i64 {\n\
               \x20   let !m = map()\n\
               \x20   let t = forge.zeros[f32, [2]]\n\
               \x20   map_set(m, \"k\", t)\n\
               \x20   0\n\
               }\n";
    let probe = Probe::new(src);
    let out = probe.run(&["jit", "--json", "--all"]);
    let kc = kinds_and_codes(&out);
    assert_eq!(kc, vec![("jit-ineligible".to_string(), Some("jit-conversion".to_string()))],
        "expected a classified jit-conversion refusal, got: {:#?}\nraw: {:#?}",
        kc, lines(&out));
}

// map_set with a real `i64` value must keep working (no regression from the
// tighter F4 guard).
#[test]
fn map_set_still_accepts_a_plain_i64_value() {
    let src = "fn main() -> i64 {\n\
               \x20   let !m = map()\n\
               \x20   map_set(m, \"k\", 42)\n\
               \x20   map_get(m, \"k\")\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "jit failed: {}", String::from_utf8_lossy(&jit.stderr));
    assert_eq!(answer(&stdout(&jit)), answer(&stdout(&run)));
    assert_eq!(answer(&stdout(&jit)), "42");
}

// ─── the three shipped example sites, inlined ───────────────────────────────
//
// #720 review of THIS fix (standalone-crate CI, 2026-09-06): the "Standalone
// crate (compiler/ alone)" job builds `compiler/` with no sibling `examples/`
// directory at all, so the three tests that used to read
// `examples/demonic_lexer.dmc`, `examples/lispy_vm.dmc`, and
// `examples/package_manifest.dmc` directly (via `assert_file_has_no_jit_error`
// below) panicked on `path.is_file()` in that job every time — a corpus fixed
// only ever existing beside the full workspace checkout, never beside the
// crate alone. Fixed by inlining a minimal program of each site's own shape
// here (self-contained, no `../examples` needed) and moving the full-corpus
// read into a separate test, below, that SKIPS (printing why) rather than
// fails when `../examples` isn't there.

fn assert_no_jit_error(src: &str, what: &str) {
    let probe = Probe::new(src);
    let out = probe.run(&["jit", "--json", "--all"]);
    let ls = lines(&out);
    for l in &ls {
        assert!(
            !l.contains("\"kind\":\"jit-error\""),
            "{}: still emits an uncoded jit-error (#695): {}\nfull run: {:#?}",
            what, l, ls,
        );
    }
}

// examples/demonic_lexer.dmc:128:30's shape: a lowercase `map`-typed
// parameter read straight back out with `map_get` (`fn tok_kind(t: map) ->
// str`).
#[test]
fn demonic_lexer_shape_map_typed_parameter_has_no_jit_error() {
    let src = "fn tok_kind(t: map) -> str { map_get(t, \"k\") }\n\
               fn main() -> str {\n\
               \x20   let !t = map_new()\n\
               \x20   map_set(t, \"k\", \"ident\")\n\
               \x20   tok_kind(t)\n\
               }\n";
    assert_no_jit_error(src, "demonic_lexer.dmc shape (tok_kind)");
}

// examples/lispy_vm.dmc:29:21's shape: the same lowercase `map`-typed
// parameter gap, on a function that also writes through it (`fn
// alloc_num(heap: map, v: f64) -> i64`).
#[test]
fn lispy_vm_shape_map_typed_parameter_has_no_jit_error() {
    let src = "fn alloc_num(heap: map, v: f64) -> i64 {\n\
               \x20   let id = to_int(map_get(heap, \"n\"))\n\
               \x20   map_set(heap, \"n\", id + 1)\n\
               \x20   id\n\
               }\n\
               fn main() -> i64 {\n\
               \x20   let !heap = map_new()\n\
               \x20   map_set(heap, \"n\", 0)\n\
               \x20   alloc_num(heap, 3.5)\n\
               }\n";
    assert_no_jit_error(src, "lispy_vm.dmc shape (alloc_num)");
}

// examples/package_manifest.dmc:4-8 / :36's shape: the bare `map()` ctor
// (`dependency`'s body) feeding an omitted-`-> T` fn whose result is then
// read with `map_get` at the call site.
#[test]
fn package_manifest_shape_bare_ctor_and_omitted_return_has_no_jit_error() {
    let src = "fn dependency(name: str, version: str) {\n\
               \x20   let dep = map()\n\
               \x20   let dep = map_set(dep, \"name\", name)\n\
               \x20   let dep = map_set(dep, \"version\", version)\n\
               \x20   dep\n\
               }\n\
               fn main() -> str {\n\
               \x20   let dep = dependency(\"stdlib.fs\", \"0.0.1\")\n\
               \x20   map_get(dep, \"name\")\n\
               }\n";
    assert_no_jit_error(src, "package_manifest.dmc shape (dependency)");
}

// ─── the full example corpus, when it's actually there ──────────────────────
//
// A broader sanity sweep over the real, larger example files — skipped, not
// failed, when `../examples` isn't beside `compiler/` (the standalone-crate
// build above all three inlined tests exist to cover).

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("examples")
}

// #753's own fix required `examples/package_manifest.dmc`'s `dependency` to
// declare `-> map` (a genuine missing annotation the corpus fix caught, not
// a #695 regression). That lets `dmc jit --json --all` get past the
// omitted-return-type gate for `dependency`'s callers, which uncovers a
// SEPARATE, pre-existing, unrelated JIT gap one function over:
// `test_dependency_name`/`test_dependency_version` compare a `map_get`
// result to a `str` literal with `==`, and the JIT's map slots are untyped
// `i64` words (#721) — `map_get`'s JIT-side return type is generically
// `i64?` regardless of what's actually stored, so the `Eq` type-check falls
// through to the generic "no such operator" message instead of a classified
// refusal. Filed as #759 (a JIT-only fix — different category from this
// checker fix per AGENTS.md §3, not addressed here). `allow` lets exactly
// that one known, cited substring through per file; anything else still
// fails the assertion.
fn assert_file_has_no_jit_error(rel: &str, allow: &[&str]) {
    let path = examples_dir().join(rel);
    assert!(path.is_file(), "missing example: {:?}", path);
    let out = Command::new(dmc_binary())
        .args(["jit", "--json", "--all"])
        .arg(&path)
        .output()
        .expect("spawn dmc");
    let ls = lines(&out);
    for l in &ls {
        if l.contains("\"kind\":\"jit-error\"") && allow.iter().any(|a| l.contains(a)) {
            continue;
        }
        assert!(
            !l.contains("\"kind\":\"jit-error\""),
            "{}: still emits an uncoded jit-error (#695): {}\nfull run: {:#?}",
            rel, l, ls,
        );
    }
}

#[test]
fn example_corpus_files_no_longer_emit_a_jit_error() {
    if !examples_dir().is_dir() {
        println!(
            "skipping: no ../examples corpus beside compiler/ (standalone \
             crate build — see the three inlined shape tests above instead)"
        );
        return;
    }
    for rel in ["demonic_lexer.dmc", "lispy_vm.dmc"] {
        assert_file_has_no_jit_error(rel, &[]);
    }
    // #759: `test_dependency_name`/`test_dependency_version`'s `map_get(..) ==
    // "str literal"` — see the comment on `assert_file_has_no_jit_error`.
    assert_file_has_no_jit_error("package_manifest.dmc",
        &["operator `Eq` is not defined for types `i64?` and `str`"]);
}
