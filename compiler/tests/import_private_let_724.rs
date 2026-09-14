//! #724 (checker half): an importer's read of an imported module's private
//! top-level `let` used to pass `--check` and run under `dmc run` — `SPEC.md
//! §6.7` exports only `pub fn`/`pub model`/`pub type`, so a top-level `let`,
//! `pub` or not, never crosses a module boundary; a constant does that as a
//! `pub fn` accessor.
//!
//! The leak had two independent causes, both fixed here:
//!
//! 1. `check.rs`'s `process_imports` never merged a top-level `let` into the
//!    importing file's scope (as the spec requires), but nothing *named* that
//!    fact — an importer's stray reference just fell through to a generic
//!    `undefined identifier`, or (see 2) was never checked at all. `import-let`
//!    is the tagged diagnostic that replaces the generic message with one that
//!    names the binding and the module it belongs to.
//! 2. `check_expr`'s `Expr::Range` arm returned `Unknown` without recursing
//!    into `start`/`end` (the same gap #562 worked around at `t[a..b]`
//!    indexing), so a range's bounds went type-checked only where some OTHER
//!    call site happened to check them explicitly first. The first pass at
//!    this fix patched `Stmt::For` alone (checking a `for` loop's iterable
//!    bounds by hand before falling into `check_expr`), which closed the
//!    issue's own repro (`for k in 0..(CAP + 1)`) but left every other
//!    position a `Range` can occupy open — a value-position range
//!    (`let r = 0..CAP`), a call argument (`f(0..CAP)`), anywhere `check_expr`
//!    reaches a `Range` without going through `Stmt::For` first. Fixed at the
//!    root instead: the `Range` arm itself now recurses into `start`/`end`,
//!    so every position closes at once, and the `Stmt::For`-specific
//!    workaround was removed (keeping it would have double-reported the same
//!    error once from each call).
//!
//! `dmc run` agrees with `--check` by construction: `main.rs` runs the checker
//! before the interpreter and refuses to execute a program with type errors,
//! so a rejected `--check` is a rejected `dmc run` with no separate interpreter
//! fix needed.

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

/// A directory of `.dmc` files; the first is the entry.
struct Tree {
    dir: tempfile::TempDir,
    entry: String,
}

impl Tree {
    fn new(files: &[(&str, &str)]) -> Tree {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, src) in files {
            std::fs::write(dir.path().join(name), src).expect("write");
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

    fn check_stderr(&self) -> String {
        let out = self.dmc(&["--check"]);
        String::from_utf8_lossy(&out.stderr).to_string()
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
}

const UTIL_WITH_MODEL: &str = "\
let CAP: i64 = 8
pub model Buf[N] { !t: Tensor[i64,[N]]  fn bump!(!self) -> i64 { self.t[0] = self.t[0] + CAP  self.t[0] } }
";

/// The issue's own minimal repro. An importer's bare read of an imported
/// module's private top-level `let`, reached through a `for` loop's range
/// bound — `--check` used to pass and `dmc run` used to answer `36`.
#[test]
fn rejects_importer_read_of_private_let_through_a_range_bound() {
    let t = Tree::new(&[
        (
            "main.dmc",
            "use \"util.dmc\"\n\
             fn main() -> i64 { let !s = 0  for k in 0..(CAP + 1) { s = s + k }  s }\n",
        ),
        ("util.dmc", UTIL_WITH_MODEL),
    ]);
    let stderr = t.check_stderr();
    assert!(
        stderr.contains("import-let"),
        "expected the `import-let` tag, got:\n{stderr}"
    );
    assert!(
        stderr.contains("`CAP` is a top-level `let` of `util.dmc`"),
        "expected the binding and module named, got:\n{stderr}"
    );
    assert!(
        stderr.contains("pub fn"),
        "expected the accessor-rule fix named, got:\n{stderr}"
    );
    // `dmc run` agrees: `main.rs` runs the checker first and refuses to
    // execute a program with type errors, so it never reaches the old `36`.
    let run_out = t.dmc(&["run"]);
    assert!(
        !run_out.status.success(),
        "dmc run should refuse a program `--check` rejects"
    );
    let run_stderr = String::from_utf8_lossy(&run_out.stderr);
    assert!(
        run_stderr.contains("import-let"),
        "expected `dmc run` to surface the same diagnostic, got:\n{run_stderr}"
    );
}

/// The same read, without the `for`-loop range indirection — a direct
/// expression-position read (`CAP + 1`) was already rejected before this PR
/// (the leak was specific to the range-bound gap); this pins that it stays
/// rejected, now with the specific diagnostic rather than a bare `undefined
/// identifier`.
#[test]
fn rejects_a_direct_importer_read_of_private_let() {
    let t = Tree::new(&[
        ("main.dmc", "use \"util.dmc\"\nfn main() -> i64 { CAP + 1 }\n"),
        ("util.dmc", "let CAP: i64 = 8\n"),
    ]);
    let stderr = t.check_stderr();
    assert!(
        stderr.contains("import-let") && stderr.contains("`CAP` is a top-level `let` of `util.dmc`"),
        "got:\n{stderr}"
    );
}

/// A qualified read (`use "..." as alias` then `alias.CAP`) earns the same
/// diagnostic at the qualified-field-access site.
#[test]
fn rejects_a_qualified_importer_read_of_private_let() {
    let t = Tree::new(&[
        (
            "main.dmc",
            "use \"util.dmc\" as util\nfn main() -> i64 { util.CAP + 1 }\n",
        ),
        ("util.dmc", "let CAP: i64 = 8\n"),
    ]);
    let stderr = t.check_stderr();
    assert!(
        stderr.contains("import-let") && stderr.contains("`CAP` is a top-level `let` of `util.dmc`"),
        "got:\n{stderr}"
    );
}

/// The spec-correct crossing: a `pub fn` accessor. Works on both backends,
/// and the module's own model method keeps reading the private `let`
/// directly — `import-let` only fires at an *importer's* reference.
#[test]
fn pub_fn_accessor_works_on_both_backends() {
    let util = format!(
        "let CAP: i64 = 8\n\
         pub fn cap() -> i64 {{ CAP }}\n\
         {}",
        "pub model Buf[N] { !t: Tensor[i64,[N]]  fn bump!(!self) -> i64 { self.t[0] = self.t[0] + CAP  self.t[0] } }\n"
    );
    let t = Tree::new(&[
        (
            "main.dmc",
            "use \"util.dmc\" as util\n\
             fn main() -> i64 { let !s = 0  for k in 0..(util.cap() + 1) { s = s + k }  s }\n",
        ),
        ("util.dmc", &util),
    ]);
    assert!(t.dmc(&["--check"]).status.success(), "the accessor program type-checks");
    assert_eq!(t.run_output(), "=> 36");
    assert_eq!(t.jit_output(), "=> 36");
}

/// An importer declaring its own top-level `let` of the same name shadows
/// nothing and conflicts with nothing — its own binding resolves through the
/// ordinary scope lookup before `import-let`'s check ever runs.
#[test]
fn importers_own_let_of_the_same_name_is_no_conflict() {
    let t = Tree::new(&[
        (
            "main.dmc",
            "use \"util.dmc\"\n\
             let CAP: i64 = 100\n\
             fn main() -> i64 { let !s = 0  for k in 0..(CAP + 1) { s = s + k }  s }\n",
        ),
        ("util.dmc", UTIL_WITH_MODEL),
    ]);
    assert!(t.dmc(&["--check"]).status.success(), "the importer's own `let CAP` is legal");
    // sum(0..=100) = 5050 — the importer's own binding, not util's `8`.
    assert_eq!(t.run_output(), "=> 5050");
}

/// The module's own fn/method bodies keep reading its private top-level
/// `let` exactly as before — `import-let` fires only at an importer's
/// reference, never at the declaring file's own use of its own binding.
#[test]
fn a_modules_own_method_still_reads_its_own_private_let() {
    let t = Tree::new(&[
        (
            "main.dmc",
            "use \"util.dmc\"\n\
             fn main() -> i64 { let !b = Buf[1] { t: forge.zeros[i64,[1]] }  b.bump!() }\n",
        ),
        ("util.dmc", UTIL_WITH_MODEL),
    ]);
    assert!(t.dmc(&["--check"]).status.success());
    assert_eq!(t.run_output(), "=> 8");
    assert_eq!(t.jit_output(), "=> 8");
}

/// General regression pin for the range-bound gap itself (#724 finding 2),
/// independent of the module system: a genuinely undefined identifier inside
/// a `for` loop's range bound, in a single file with no `use` at all, used to
/// pass `--check` silently because `check_expr` never recursed into the
/// range's `start`/`end`.
#[test]
fn undefined_identifier_in_a_for_range_bound_is_now_rejected() {
    let t = Tree::new(&[(
        "main.dmc",
        "fn main() -> i64 { let !s = 0  for k in 0..(NOPE + 1) { s = s + k }  s }\n",
    )]);
    let stderr = t.check_stderr();
    assert!(
        stderr.contains("undefined identifier `NOPE`"),
        "got:\n{stderr}"
    );
}

/// A `for` loop's range bound wrapped in an EXTRA pair of parens around the
/// whole range (`for k in (0..(NOPE + 1))`, not just around one bound) —
/// pins that the fix does not depend on the range sitting bare as `iter`;
/// grouping parens are transparent in this grammar (no `Paren` AST node), so
/// this is really the same `Expr::Range` either way, but it is worth a
/// dedicated regression given the first attempt at this fix lived in
/// `Stmt::For` specifically.
#[test]
fn rejects_parenthesised_for_range_bound_with_undefined_name() {
    let t = Tree::new(&[(
        "main.dmc",
        "fn main() -> i64 { let !s = 0  for k in (0..(NOPE + 1)) { s = s + k }  s }\n",
    )]);
    let stderr = t.check_stderr();
    assert!(stderr.contains("undefined identifier `NOPE`"), "got:\n{stderr}");
}

/// Same shape, reading an imported module's private `let` instead of a
/// genuinely undefined name — the issue's own leak (`dmc run` used to answer
/// `36`), reproduced with the extra parens intact.
#[test]
fn rejects_parenthesised_for_range_bound_reading_private_let() {
    let t = Tree::new(&[
        (
            "main.dmc",
            "use \"util.dmc\"\n\
             fn main() -> i64 { let !s = 0  for k in (0..(CAP + 1)) { s = s + k }  s }\n",
        ),
        ("util.dmc", "let CAP: i64 = 8\n"),
    ]);
    let stderr = t.check_stderr();
    assert!(
        stderr.contains("import-let") && stderr.contains("`CAP` is a top-level `let` of `util.dmc`"),
        "got:\n{stderr}"
    );
    let run_out = t.dmc(&["run"]);
    assert!(!run_out.status.success(), "dmc run should refuse a program `--check` rejects");
}

/// A range that never touches a `for` loop at all — bound to a `let` in
/// value position. This is the position the original `Stmt::For`-only patch
/// could not reach: `check_expr` recurses into a `let`'s RHS regardless of
/// what the RHS's outer shape is, so once the fix lives in the `Range` arm
/// itself, this closes for free.
#[test]
fn rejects_value_position_range_with_undefined_name() {
    let t = Tree::new(&[(
        "main.dmc",
        "fn main() -> i64 { let r = 0..(NOPE)  0 }\n",
    )]);
    let stderr = t.check_stderr();
    assert!(stderr.contains("undefined identifier `NOPE`"), "got:\n{stderr}");
}

#[test]
fn rejects_value_position_range_reading_private_let() {
    let t = Tree::new(&[
        (
            "main.dmc",
            "use \"util.dmc\"\nfn main() -> i64 { let r = 0..(CAP)  0 }\n",
        ),
        ("util.dmc", "let CAP: i64 = 8\n"),
    ]);
    let stderr = t.check_stderr();
    assert!(
        stderr.contains("import-let") && stderr.contains("`CAP` is a top-level `let` of `util.dmc`"),
        "got:\n{stderr}"
    );
}

/// A range passed directly as a call argument — another position
/// `Stmt::For`'s special case could never reach, since it never goes through
/// a `for` loop at all.
#[test]
fn rejects_call_argument_range_with_undefined_name() {
    let t = Tree::new(&[(
        "main.dmc",
        "fn f(r: bool) -> bool { r }\nfn main() -> bool { f(0..NOPE) }\n",
    )]);
    let stderr = t.check_stderr();
    assert!(stderr.contains("undefined identifier `NOPE`"), "got:\n{stderr}");
}

#[test]
fn rejects_call_argument_range_reading_private_let() {
    let t = Tree::new(&[
        (
            "main.dmc",
            "use \"util.dmc\"\nfn f(r: bool) -> bool { r }\nfn main() -> bool { f(0..CAP) }\n",
        ),
        ("util.dmc", "let CAP: i64 = 8\n"),
    ]);
    let stderr = t.check_stderr();
    assert!(
        stderr.contains("import-let") && stderr.contains("`CAP` is a top-level `let` of `util.dmc`"),
        "got:\n{stderr}"
    );
}

/// D4: a three-file chain — `main` `use`s `mid` (unaliased), `mid` `use`s
/// `base` (unaliased), `base` alone declares `let CAP`. `main`'s bare read of
/// `CAP` reaches it only transitively, through `mid`; before this fix,
/// `process_imports` only ever consulted the DIRECTLY-`use`d module's own
/// `top_level_let_names` (here, `mid`'s — empty, since `CAP` is `base`'s),
/// so the diagnostic fell back to a generic `undefined identifier `CAP``
/// instead of naming the module that actually declares it. Fixed by
/// threading `Checker::import_private_lets` out through `ModuleEnv` (as
/// `transitive_import_lets`) so an importer merges in whatever its own
/// import already resolved, transitively to any chain depth.
#[test]
fn three_file_chain_names_the_original_declaring_module() {
    let t = Tree::new(&[
        (
            "main.dmc",
            "use \"mid.dmc\"\nfn main() -> i64 { let r = 0..(CAP)  0 }\n",
        ),
        (
            "mid.dmc",
            "use \"base.dmc\"\npub fn mid_helper() -> i64 { 1 }\n",
        ),
        ("base.dmc", "let CAP: i64 = 8\n"),
    ]);
    let stderr = t.check_stderr();
    assert!(
        stderr.contains("import-let") && stderr.contains("`CAP` is a top-level `let` of `base.dmc`"),
        "expected the ORIGINAL declaring module (`base.dmc`) named, not a generic \
         `undefined identifier` or the intermediate `mid.dmc`, got:\n{stderr}"
    );
}

/// A name an intermediate module re-declares itself takes precedence over
/// the one it inherited transitively — `main` should see `mid.dmc` named,
/// not `base.dmc`, once `mid` shadows `CAP` with its own top-level `let`.
#[test]
fn three_file_chain_own_declaration_shadows_the_transitively_inherited_one() {
    let t = Tree::new(&[
        (
            "main.dmc",
            "use \"mid.dmc\"\nfn main() -> i64 { let r = 0..(CAP)  0 }\n",
        ),
        (
            "mid.dmc",
            "use \"base.dmc\"\nlet CAP: i64 = 99\npub fn mid_helper() -> i64 { 1 }\n",
        ),
        ("base.dmc", "let CAP: i64 = 8\n"),
    ]);
    let stderr = t.check_stderr();
    assert!(
        stderr.contains("import-let") && stderr.contains("`CAP` is a top-level `let` of `mid.dmc`"),
        "got:\n{stderr}"
    );
}
