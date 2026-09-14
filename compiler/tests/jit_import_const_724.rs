//! #724: an importing file's read of an imported module's PRIVATE top-level
//! `let` — `use "util.dmc"` then a bare `CAP` in the importer, where
//! `util.dmc` declares `let CAP: i64 = 8` with no `pub` — type-checks clean
//! and `dmc run` answers it (a checker leak against SPEC.md §6.7: "Only
//! `pub`-annotated items (`pub fn`, `pub model`, `pub type`) are exported ...
//! Non-`pub` items are file-private and invisible to importers"; fixing the
//! checker side of this is #724's other half, tracked separately). The JIT
//! used to fail it with an uncoded `jit error: unknown binding CAP (JIT has
//! no globals)` (the #480 defect class), aborting `--all` collection before
//! anything past the read site was walked.
//!
//! **Measurement, from `merge.rs` and `jit.rs` (not guessed):** the merge
//! renames `util.dmc`'s private `let CAP` to `util$CAP` in the merged item
//! list — every non-`pub` declaration is prefixed, unconditionally
//! (`merge.rs`'s rename rule). But `merge.rs`'s `FileScope` — "what one
//! file's references resolve against" — is built from a file's own
//! declarations plus only the **`pub`** declarations of what it imports
//! (`merge_program`'s scope-building pass skips every `!d.is_pub` decl of an
//! imported file entirely). `CAP` is not `pub`, so it is never entered into
//! the importer's scope, and the importer's reference to it is therefore
//! **never rewritten** — it reaches `Jit::lower_ident` exactly as written,
//! bare `CAP`. `Jit::global_consts` and `Jit::top_level_let_names` are both
//! keyed by the MERGED name (`util$CAP`), so neither a `global_consts` hit
//! nor the #696/#711 `top_level_let_names.contains(name)` check can ever
//! match a bare `CAP` — the old code fell all the way to the uncoded
//! `err(...)` at the bottom of `lower_ident`.
//!
//! Two honest outcomes, and this is the spec-correct one: (A) the constant
//! IS a compile-time literal, so the JIT COULD lower it exactly the way
//! `util.dmc`'s own methods do (#620) — but doing so would make the JIT
//! MORE permissive than the spec and silently endorse the leak SPEC.md §6.7
//! already forbids. So (B): a NEW check, `imported_private_let_module`,
//! recognizes a bare unresolved name that matches the un-prefixed SOURCE
//! spelling of some OTHER file's top-level `let` (recovered by splitting a
//! merged name on its LAST `$` — a top-level `let` is never `pub` and never
//! given a further mangling suffix, so this is exact, no `MergedProgram`
//! needed) and refuses it as `jit-item` — the same class #696/#711 already
//! use for a top-level `let` the JIT does not lower — with a message naming
//! the binding as a private top-level `let` of an imported module and
//! citing the `pub fn`-accessor rule, rather than the uncoded `jit error`.
//!
//! The module's OWN fns/methods reading `CAP` are UNAFFECTED: a same-file
//! reference is always rewritten by `merge.rs`'s own-declarations scope pass
//! (unconditional, before the import pass runs), so it already reads
//! `util$CAP` and hits the existing `global_consts` literal-fold path
//! (#268/#620) — measured below to still lower, and to still agree with
//! `dmc run` byte-for-byte.
//!
//! **Update — #724's checker half has since landed** (`import_private_let_724.rs`):
//! `check_expr`'s `Expr::Range` arm now recurses into `start`/`end` at every
//! position a range can occupy, not only inside a `for` loop's own iterable,
//! so EVERY repro in this file — all of which read the leaked constant
//! through exactly that `for k in 0..(CAP + 1)` shape — is now refused by the
//! CHECKER itself, before `dmc jit` (which type-checks first, same as `dmc
//! run`) ever reaches the JIT-specific `imported_private_let_module` workaround
//! this file was written to pin. The tests below are updated accordingly:
//! `dmc jit` on each of these programs now surfaces the checker's own
//! `import-let` diagnostic (`"kind":"check"`), not `"kind":"jit-ineligible"`/
//! `"code":"jit-item"` — a strictly earlier and more precisely located
//! refusal than the JIT-side workaround gave. That workaround stays coded
//! (defense in depth, and #724's checker fix does not cover every import
//! surface — e.g. it is scoped to unaliased `use` chains, `Range` positions,
//! and a handful of others enumerated in `check.rs`), it is simply
//! unreachable through any of THESE particular repros now.

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

/// A directory of `.dmc` files; the first is the entry. A name may itself
/// contain `/` (`"a-b/util.dmc"`) to put a file in a subdirectory — the
/// collision test below needs two same-named files in two different
/// directories, so parent directories are created as needed.
struct Tree {
    dir: tempfile::TempDir,
    entry: String,
}

impl Tree {
    fn new(files: &[(&str, &str)]) -> Tree {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, src) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("mkdir");
            }
            std::fs::write(path, src).expect("write");
        }
        Tree { dir, entry: files[0].0.to_string() }
    }

    fn dmc(&self, args: &[&str]) -> Output {
        Command::new(dmc_binary())
            .args(args)
            .arg(self.dir.path().join(&self.entry))
            .output()
            .expect("spawn dmc")
    }

    fn run_output(&self) -> String {
        let out = self.dmc(&["run"]);
        assert!(
            out.status.success(),
            "dmc run failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn jit_output(&self) -> String {
        let out = self.dmc(&["jit"]);
        assert!(
            out.status.success(),
            "dmc jit failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Every `--json` diagnostic line `dmc jit` emits, the summary excluded.
    fn jit_json(&self) -> Vec<String> {
        let out = self.dmc(&["jit", "--json"]);
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .filter(|l| l.starts_with('{') && !l.contains("\"kind\":\"summary\""))
            .map(|l| l.to_string())
            .collect()
    }

    /// Every line `dmc jit --json --all` writes to stderr, summary included.
    fn jit_json_all(&self) -> Vec<String> {
        let out = self.dmc(&["jit", "--json", "--all"]);
        String::from_utf8_lossy(&out.stderr)
            .trim_end_matches('\n')
            .lines()
            .map(|l| l.to_string())
            .collect()
    }

    fn path_of(&self, name: &str) -> std::path::PathBuf {
        self.dir.path().join(name)
    }
}

/// Issue #724's own two-file repro, verbatim.
const UTIL: &str = "\
let CAP: i64 = 8
pub model Buf[N] { !t: Tensor[i64,[N]]  fn bump!(!self) -> i64 { self.t[0] = self.t[0] + CAP  self.t[0] } }
";
const MAIN: &str = "\
use \"util.dmc\"
fn main() -> i64 { let !s = 0  for k in 0..(CAP + 1) { s = s + k }  s }
";

/// The measurement, updated: `dmc jit` used to fail with the uncoded
/// `jit error: unknown binding CAP (JIT has no globals)` when the checker
/// missed this leak. The checker no longer misses it — `--check` and
/// `dmc run` now both REFUSE (see `import_private_let_724.rs`) — so `dmc
/// jit`, which type-checks before it ever lowers anything, surfaces that
/// same checker-level `import-let` refusal instead of reaching its own
/// `jit-item` workaround. Still a located diagnostic naming the binding and
/// the accessor fix, just one step earlier and one layer more precise than
/// the JIT-specific message this test used to pin.
#[test]
fn importer_read_of_a_private_imported_let_is_caught_by_the_checker_before_jit() {
    let t = Tree::new(&[("main.dmc", MAIN), ("util.dmc", UTIL)]);

    assert!(!t.dmc(&["--check"]).status.success(), "the checker now refuses this leak directly");
    assert!(!t.dmc(&["run"]).status.success(), "dmc run refuses too — it type-checks first");

    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{:#?}", json);
    let main_dmc = t.path_of("main.dmc").canonicalize().expect("canonicalize main.dmc");
    assert_eq!(
        json[0],
        format!(
            "{{\"schema\":1,\"kind\":\"check\",\"severity\":\"error\",\"code\":\"import-let\",\
             \"message\":\"import-let: `CAP` is a top-level `let` of `util.dmc`; a constant \
             crosses a module boundary as a `pub fn` accessor, not a bare reference \
             (SPEC.md §6.7)\",\"file\":\"{}\",\"line\":2,\"col\":45,\"start\":59,\"end\":62}}",
            main_dmc.display(),
        ),
        "must be the checker's own located `import-let` refusal, naming the binding, \
         the declaring module, and the accessor fix — never the uncoded `jit error`, \
         and never even reaching the JIT's own `jit-item` workaround",
    );
}

/// `--json --all` must not truncate collection (the defect this issue's
/// title complains about — an uncoded `jit-error` used to abort `--all`
/// before anything past the read site was walked). Now that the checker
/// refuses this leak up front, `dmc jit` never even reaches JIT-item
/// collection at all — it stops at the checker's own `import-let` error,
/// same as `--check` alone, one classed error and no truncation.
#[test]
fn importer_read_of_a_private_imported_let_does_not_truncate_all_collection() {
    let t = Tree::new(&[("main.dmc", MAIN), ("util.dmc", UTIL)]);

    let lines = t.jit_json_all();
    assert_eq!(lines.len(), 2, "refusal + summary:\n{:#?}", lines);
    assert!(lines[0].contains("\"kind\":\"check\""), "{}", lines[0]);
    assert!(lines[0].contains("\"code\":\"import-let\""), "{}", lines[0]);
    assert!(!lines[0].contains("jit-error"), "{}", lines[0]);
    assert_eq!(
        lines[1],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"failed\",\
         \"errors\":1,\"warnings\":0,\"exit\":1}",
        "one checker error, not truncated — never a jit-error abort",
    );
}

/// The module's OWN method reading its OWN private `CAP` is unaffected: the
/// self-reference is rewritten to `util$CAP` by `merge.rs`'s own-declarations
/// scope pass (unconditional, unlike the import pass, which is `pub`-only),
/// so it hits the ordinary #268/#620 literal-fold path and lowers — agreeing
/// with `dmc run` byte-for-byte, not refusing at all.
#[test]
fn the_modules_own_method_reading_its_private_let_still_lowers_and_agrees_with_run() {
    let entry = "\
use \"util.dmc\"
fn main() -> i64 {
    let !b = Buf[3] { t: forge.zeros[i64, [3]] }
    b.bump!()
    b.bump!()
}
";
    let t = Tree::new(&[("main.dmc", entry), ("util.dmc", UTIL)]);
    assert!(t.dmc(&["--check"]).status.success());
    let expected = t.run_output();
    assert_eq!(expected, "=> 16", "8 + 8, two `bump!()` calls");
    assert_eq!(t.jit_output(), expected, "the JIT must lower the module's own read of CAP identically");
}

/// #724 fixup: two DIFFERENT modules each privately declaring a `let CAP`,
/// both imported (unqualified) by one caller that reads bare `CAP` — two
/// equally-plausible declaring files. This used to be the JIT's own
/// ambiguity to resolve (and it used to get it wrong — see the history in
/// this comment before the checker fix landed); now the CHECKER refuses the
/// read outright before `dmc run` or `dmc jit` ever resolves either module's
/// value, so there is no live, divergent "actual" answer left to be wrong
/// about. `Checker::import_private_lets` is a plain first-write-wins map
/// keyed by name, so it names whichever `use` was processed first in
/// `main.dmc` — `alp.dmc`, textually first — deterministically, for a
/// diagnostic that (unlike the old JIT one) is never checked against a
/// runtime answer because nothing runs.
const ALP: &str = "\
let CAP: i64 = 3
pub fn alp_noop() -> i64 { CAP }
";
const ZED: &str = "\
let CAP: i64 = 8
pub fn zed_noop() -> i64 { CAP }
";
const MAIN_TWO_PRIVATE_CAPS: &str = "\
use \"alp.dmc\"
use \"zed.dmc\"
fn main() -> i64 { let !s = 0  for k in 0..(CAP + 1) { s = s + k }  s }
";

#[test]
fn two_imported_private_lets_sharing_a_name_names_the_first_processed_use() {
    let t = Tree::new(&[("main.dmc", MAIN_TWO_PRIVATE_CAPS), ("alp.dmc", ALP), ("zed.dmc", ZED)]);

    assert!(!t.dmc(&["--check"]).status.success(), "the checker now refuses this read outright");
    assert!(!t.dmc(&["run"]).status.success(), "dmc run never resolves either module's CAP now");

    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{:#?}", json);
    let main_dmc = t.path_of("main.dmc").canonicalize().expect("canonicalize main.dmc");
    assert_eq!(
        json[0],
        format!(
            "{{\"schema\":1,\"kind\":\"check\",\"severity\":\"error\",\"code\":\"import-let\",\
             \"message\":\"import-let: `CAP` is a top-level `let` of `alp.dmc`; a constant \
             crosses a module boundary as a `pub fn` accessor, not a bare reference \
             (SPEC.md §6.7)\",\"file\":\"{}\",\"line\":3,\"col\":45,\"start\":72,\"end\":75}}",
            main_dmc.display(),
        ),
        "the checker's diagnostic names alp.dmc (the first `use` processed) — a defensible, \
         deterministic choice now that nothing ever runs to prove it right or wrong",
    );
}

/// #724 fixup: a merge-prefix collision. Two files both literally named
/// `util.dmc`, in sibling directories whose names ALSO sanitize alike (`-`
/// and `+` both fold to `_`, `merge.rs`'s `sanitize_ident`) — so `util.dmc`'s
/// stem collides, and only the JIT's own `$<index>$` disambiguator
/// (`file_prefixes`) told the two apart in the merge-lowering path this test
/// used to pin. The checker never merges by that scheme at all — it tracks
/// imports by canonical file path — so it was never at risk of leaking an
/// internal key like `a_b$util$0` in the first place; it names the real
/// import path (`a-b/util.dmc`, exactly as written in the `use`) directly.
/// Only `a-b/util.dmc` declares `CAP`; `a+b/util.dmc` declares something
/// unrelated, so there is exactly one real candidate here.
const COLLIDING_UTIL: &str = "\
let CAP: i64 = 8
pub fn noop() -> i64 { CAP }
";
const COLLIDING_OTHER: &str = "pub fn unrelated() -> i64 { 1 }\n";
const MAIN_COLLIDING_STEMS: &str = "\
use \"a-b/util.dmc\"
use \"a+b/util.dmc\"
fn main() -> i64 { let !s = 0  for k in 0..(CAP + 1) { s = s + k }  s }
";

#[test]
fn a_merge_prefix_collision_names_the_real_file_not_the_internal_key() {
    let t = Tree::new(&[
        ("main.dmc", MAIN_COLLIDING_STEMS),
        ("a-b/util.dmc", COLLIDING_UTIL),
        ("a+b/util.dmc", COLLIDING_OTHER),
    ]);

    assert!(!t.dmc(&["--check"]).status.success(), "the checker refuses this read outright");
    assert!(!t.dmc(&["run"]).status.success());

    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{:#?}", json);
    assert!(json[0].contains("\"kind\":\"check\""), "{}", json[0]);
    assert!(json[0].contains("\"code\":\"import-let\""), "{}", json[0]);
    assert!(
        json[0].contains("of `a-b/util.dmc`"),
        "must name the real import path exactly as written in the `use`: {}",
        json[0],
    );
    assert!(
        !json[0].contains('$'),
        "must never leak a merge-internal key (`a_b$util$0`, `file_prefixes`'s \
         `$<index>$` collision disambiguator) into the message: {}",
        json[0],
    );
}

/// #724 fixup: N reads of the SAME leaked private import. This used to be
/// the JIT's own root-blocker/cascade classification to get right (one root
/// plus N−1 cascades, never N independent root blockers — the #696 R2
/// shape). Now that the checker catches the leak directly, EACH read is its
/// own independent `check_expr` call and earns its own `import-let` error —
/// there is no blocker/cascade classification at the checker level at all,
/// so two reads simply means two errors, both located, neither collapsed nor
/// truncated.
const MAIN_TWO_READS_ONE_LEAK: &str = "\
use \"util.dmc\"
fn read_a() -> i64 { let !s = 0  for k in 0..(CAP + 1) { s = s + k }  s }
fn read_b() -> i64 { let !s = 0  for k in 0..(CAP + 1) { s = s + k }  s }
fn main() -> i64 { read_a() + read_b() }
";

#[test]
fn two_reads_of_one_leaked_import_are_two_independent_checker_errors() {
    let t = Tree::new(&[("main.dmc", MAIN_TWO_READS_ONE_LEAK), ("util.dmc", UTIL)]);

    let lines = t.jit_json_all();
    assert_eq!(lines.len(), 3, "expected 2 read refusals + summary:\n{:#?}", lines);
    assert!(lines[0].contains("\"kind\":\"check\"") && lines[0].contains("\"code\":\"import-let\""), "{}", lines[0]);
    assert!(lines[1].contains("\"kind\":\"check\"") && lines[1].contains("\"code\":\"import-let\""), "{}", lines[1]);
    assert_eq!(
        lines[2],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"failed\",\
         \"errors\":2,\"warnings\":0,\"exit\":1}",
        "two independent checker errors, neither collapsed nor truncated: {:#?}",
        lines,
    );
}
