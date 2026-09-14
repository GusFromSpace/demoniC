//! Tests for the whole-program merge.
//!
//! These drive the real pipeline in process — resolve, fold `@comptime`, check
//! leaf-first per file, merge, then `Jit::compile_program` and `run_main` — and
//! assert the **answer**, not that two backends agree. Two backends can agree on
//! the wrong number, and on the programs that matter here the interpreter is the
//! other half of the fact being pinned, not the oracle.
//!
//! Why in `src/` rather than the `compiler/tests/jit_multifile_merge.rs` the
//! design names: this crate has no library target, so an integration test can
//! only drive the `dmc` binary, and until slice 3 wires `merge_program` into the
//! `dmc jit` arm there is no command that reaches it. The binary-level half —
//! the interpreter's per-module answers on the same programs, and the refusal
//! slice 3 flips — is in `compiler/tests/jit_multifile_merge.rs`.
//!
//! Every program here returns `i64` from `main`, because `run_main` returns one:
//! the `1 1 2 2` the design's gate quotes is spelled `1122` by
//! `a1()*1000 + a2()*100 + b1()*10 + b2()`, which is the same four calls in the
//! same order.

use std::path::{Path, PathBuf};

use crate::ast::Item;
use crate::check::{self, Checker};
use crate::jit::Jit;
use crate::merge::{merge_program, MergeError, MergedProgram};
use crate::resolver::Resolver;

// ─── Harness ─────────────────────────────────────────────────────────────────

/// A directory of `.dmc` files. The entry file is always the first one written,
/// which also fixes the `use` order the interpreter's answer used to depend on.
struct Tree {
    dir: tempfile::TempDir,
}

impl Tree {
    fn new(files: &[(&str, &str)]) -> Tree {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, src) in files {
            std::fs::write(dir.path().join(name), src).expect("write");
        }
        Tree { dir }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name).canonicalize().expect("canonicalize")
    }
}

/// Resolve the DAG under `entry` and fold `@comptime` per file, exactly as
/// every `main.rs` entry point does before a checker runs.
fn resolved(entry: &Path) -> Resolver {
    let mut r = Resolver::new();
    r.resolve_all(entry).expect("resolve_all");
    for p in r.sorted_paths.clone() {
        if let Some(prog) = r.files.get_mut(&p) {
            let errs = crate::comptime::fold_program(prog);
            assert!(errs.is_empty(), "comptime fold failed in {:?}: {:?}", p, errs);
        }
    }
    r
}

/// The per-file, leaf-first check the merge runs after (design §4.1(d)).
/// Returns the checker errors of the first file that has any.
fn check_leaf_first(r: &Resolver) -> Result<(), String> {
    let mut checked = std::collections::HashMap::new();
    for p in &r.sorted_paths {
        let prog = r.files.get(p).unwrap();
        let mut checker = Checker::new();
        checker.checked_modules = checked.clone();
        checker.check_program(prog, Some(p));
        if !checker.errors.is_empty() {
            return Err(format!(
                "{:?}: {}",
                p.file_name().unwrap(),
                checker.errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ")
            ));
        }
        checked.insert(
            p.clone(),
            check::ModuleEnv {
                env: checker.env.clone(),
                aliases: checker.aliases.clone(),
                resolved_aliases: checker.resolved_aliases.clone(),
                public_items: crate::ast::collect_public_items(prog),
                top_level_let_names: crate::ast::collect_top_level_let_names(prog),
                transitive_import_lets: checker.import_private_lets.clone(),
            },
        );
    }
    Ok(())
}

/// The whole slice-1 path: resolve → fold → check → merge.
fn merged(entry: &Path) -> Result<MergedProgram, MergeError> {
    let r = resolved(entry);
    check_leaf_first(&r).expect("the program must type-check per module before it is merged");
    merge_program(&r, entry)
}

/// The gate: what the JIT answers for a merged multi-file program.
fn jit_answer(entry: &Path) -> Result<i64, String> {
    let m = merged(entry).map_err(|e| e.to_string())?;
    let mut jit = Jit::new().map_err(|e| e.to_string())?;
    jit.compile_program(&m.program).map_err(|e| e.to_string())?;
    jit.run_main().map_err(|e| e.to_string())
}

/// Every top-level declared name in a merged program, for the rename assertions.
fn declared_names(p: &MergedProgram) -> Vec<String> {
    fn walk(item: &Item, out: &mut Vec<String>) {
        match item {
            Item::Pub(inner) => walk(inner, out),
            Item::Directive { inner, .. } => walk(inner, out),
            Item::Fn(f) => out.push(f.name.clone()),
            Item::ExternFn(e) => out.push(e.name.clone()),
            Item::Model(m) => out.push(m.name.clone()),
            Item::Enum(e) => out.push(e.name.clone()),
            Item::TypeAlias(t) => out.push(t.name.clone()),
            Item::Let(l) => {
                if let crate::ast::Pattern::Ident(n, _) = &l.pattern {
                    out.push(n.clone());
                }
            }
            Item::Arena(_) | Item::Use(_) => {}
        }
    }
    let mut out = Vec::new();
    for item in &p.program.items {
        walk(item, &mut out);
    }
    out
}

// ─── The gate: private names are scoped per module ───────────────────────────

const HELPER_A: &str = "\
fn helper() -> i64 { 1 }
pub fn a1() -> i64 { helper() }
pub fn a2() -> i64 { helper() }
";
const HELPER_B: &str = "\
fn helper() -> i64 { 2 }
pub fn b1() -> i64 { helper() }
pub fn b2() -> i64 { helper() }
";

/// Slice 1's gate (design §4.1(g)). Two leaves each declare a *file-private*
/// `fn helper`, so no importer can see either; per-module semantics answers
/// `1 1 2 2` under both `use` orders, which is `1122` read off `main`'s return.
///
/// This is the program §5.1 is about: without the unconditional private-name
/// prefix, both `helper`s land on one key in `Jit::fns` and the merge is a
/// `duplicate function` error — or, for the model and const maps that do not
/// guard, a silent miscompile.
///
/// The gate leans on the entry-`main` carve-out: had the merge prefixed the
/// entry's `main`, `run_main`'s literal `"main"` lookup would find nothing and
/// this would fail as a missing entry point rather than as a wrong answer.
#[test]
fn jit_merge_scopes_private_fns_per_module() {
    for (label, entry_src) in [
        (
            "a then b",
            "use \"ha.dmc\"\nuse \"hb.dmc\"\nfn main() -> i64 { a1()*1000 + a2()*100 + b1()*10 + b2() }\n",
        ),
        (
            "b then a",
            "use \"hb.dmc\"\nuse \"ha.dmc\"\nfn main() -> i64 { a1()*1000 + a2()*100 + b1()*10 + b2() }\n",
        ),
    ] {
        let t = Tree::new(&[("e.dmc", entry_src), ("ha.dmc", HELPER_A), ("hb.dmc", HELPER_B)]);
        assert_eq!(
            jit_answer(&t.path("e.dmc")),
            Ok(1122),
            "the merged program must answer 1 1 2 2 with the `use`s in {label} order"
        );
    }
}

/// The same program, at the AST level: both `helper`s survive under distinct
/// prefixed names, the entry's `main` keeps its bare one, and no `use` is left.
#[test]
fn jit_merge_prefixes_every_private_name_and_drops_every_use() {
    let t = Tree::new(&[
        ("e.dmc", "use \"ha.dmc\"\nuse \"hb.dmc\"\nfn main() -> i64 { a1() + b1() }\n"),
        ("ha.dmc", HELPER_A),
        ("hb.dmc", HELPER_B),
    ]);
    let p = merged(&t.path("e.dmc")).expect("merge");
    let names = declared_names(&p);
    assert!(names.contains(&"ha$helper".to_string()), "{names:?}");
    assert!(names.contains(&"hb$helper".to_string()), "{names:?}");
    assert!(!names.contains(&"helper".to_string()), "no bare `helper` survives: {names:?}");
    // `pub` names are unique across the merge, so they keep the spelling the
    // importer wrote.
    for n in ["a1", "a2", "b1", "b2", "main"] {
        assert!(names.contains(&n.to_string()), "`{n}` must keep its bare name: {names:?}");
    }
    assert!(
        !p.program.items.iter().any(|i| matches!(i, Item::Use(_))),
        "every `Item::Use` is dropped before lowering"
    );
}

// ─── `pub` names, colliding and not ──────────────────────────────────────────

/// The `pub fn g` diamond (Appendix A, #613's second half): two leaves each
/// export `pub fn g`; two mid-level files import one each; the entry imports the
/// two mids, so no single file sees both `g`s and `ambiguous import` never
/// fires. Per-module semantics answers `12`, and since #652 so does `dmc run`.
///
/// This is the design's one internal contradiction, resolved in `merge.rs`'s
/// module docs: §3(a) would make this a hard merge error, §1 answers it by
/// prefixing. Erroring would refuse a program the interpreter runs correctly.
#[test]
fn jit_merge_prefixes_colliding_pub_names_rather_than_refusing() {
    let t = Tree::new(&[
        ("e.dmc", "use \"m1.dmc\"\nuse \"m2.dmc\"\nfn main() -> i64 { m1() * 10 + m2() }\n"),
        ("m1.dmc", "use \"g1.dmc\"\npub fn m1() -> i64 { g() }\n"),
        ("m2.dmc", "use \"g2.dmc\"\npub fn m2() -> i64 { g() }\n"),
        ("g1.dmc", "pub fn g() -> i64 { 1 }\n"),
        ("g2.dmc", "pub fn g() -> i64 { 2 }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(12));

    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"g1$g".to_string()), "{names:?}");
    assert!(names.contains(&"g2$g".to_string()), "{names:?}");
    assert!(!names.contains(&"g".to_string()), "neither `g` keeps the bare name: {names:?}");
}

/// The case prefixing cannot answer, and the design's hard error: one file's
/// scope binds one name to two different declarations. `check.rs` already rules
/// this `ambiguous import`; the merge asserts it again rather than picking a
/// winner, and names both files so a reader can get back to the declarations.
#[test]
fn jit_merge_refuses_a_pub_name_one_importer_sees_twice() {
    let t = Tree::new(&[
        ("e.dmc", "use \"g1.dmc\"\nuse \"g2.dmc\"\nfn main() -> i64 { g() }\n"),
        ("g1.dmc", "pub fn g() -> i64 { 1 }\n"),
        ("g2.dmc", "pub fn g() -> i64 { 2 }\n"),
    ]);
    // The checker rejects this program too, so the merge is reached only by
    // calling it directly — which is the point of the assertion: it does not
    // rely on the checker having run.
    let r = resolved(&t.path("e.dmc"));
    let err = merge_program(&r, &t.path("e.dmc")).expect_err("a name bound twice is refused");
    match &err {
        MergeError::Ambiguous { what, name, first, second, .. } => {
            assert_eq!(*what, "function");
            assert_eq!(name, "g");
            let both = format!("{:?} {:?}", first, second);
            assert!(both.contains("g1.dmc"), "{both}");
            assert!(both.contains("g2.dmc"), "{both}");
        }
        other => panic!("expected Ambiguous, got {other:?}"),
    }
    assert!(err.to_string().contains("g1.dmc") && err.to_string().contains("g2.dmc"));
}

// ─── Cross-file models ───────────────────────────────────────────────────────

/// Appendix A's diamond: an entry over two mid-level modules onto one shared
/// leaf, the leaf's `pub model Point` crossing two file boundaries as a field
/// type. `_flattened` — the same four files concatenated by hand — already
/// compiled and agreed with the interpreter, which is what made the merge a
/// delivery problem rather than a codegen one; this is the delivery.
///
/// Every file declares its own `fn main`; the dependencies' are prefixed and
/// the entry's is not, which is why `run_main` finds one entry point and not
/// four.
#[test]
fn jit_merge_lowers_a_cross_file_pub_model_diamond() {
    let t = Tree::new(&[
        (
            "diamond.dmc",
            "use \"boxed.dmc\"\nuse \"sum.dmc\"\n\
             fn main() -> i64 { let bx = boxed(4, 5)  sum_pt(bx.p) + bx.tag }\n",
        ),
        (
            "boxed.dmc",
            "use \"point.dmc\"\n\
             pub model Boxed { p: Point, tag: i64 }\n\
             pub fn boxed(x: i64, y: i64) -> Boxed { Boxed { p: mk(x, y), tag: 1 } }\n\
             fn main() -> nil { print(\"Run OK\") nil }\n",
        ),
        (
            "sum.dmc",
            "use \"point.dmc\"\n\
             pub fn sum_pt(p: Point) -> i64 { p.x + p.y }\n\
             fn main() -> nil { print(\"Run OK\") nil }\n",
        ),
        (
            "point.dmc",
            "pub model Point { x: i64, y: i64 }\n\
             pub fn mk(x: i64, y: i64) -> Point { Point { x: x, y: y } }\n\
             fn main() -> nil { print(\"Run OK\") nil }\n",
        ),
    ]);
    assert_eq!(jit_answer(&t.path("diamond.dmc")), Ok(10));

    let names = declared_names(&merged(&t.path("diamond.dmc")).expect("merge"));
    // The dependencies' `main`s are prefixed; only the entry's is bare.
    assert_eq!(names.iter().filter(|n| *n == "main").count(), 1, "{names:?}");
    for n in ["point$main", "boxed$main", "sum$main"] {
        assert!(names.contains(&n.to_string()), "{names:?}");
    }
}

/// Design §5.1, the reason private-name prefixing is unconditional. Two files
/// each declare a file-private `model Node` with *different* fields and each
/// constructs its own from a `pub fn`. `--check` passes, because type-checking
/// is per module, and the merged program is not re-checked — so nothing between
/// the checker and Cranelift would notice both `Node`s landing in one
/// `model_layouts` slot. The first file's field access would then lower against
/// the second's layout: wrong offsets, no diagnostic.
#[test]
fn jit_merge_keeps_two_private_models_apart() {
    let t = Tree::new(&[
        ("e.dmc", "use \"na.dmc\"\nuse \"nb.dmc\"\nfn main() -> i64 { na() + nb() }\n"),
        (
            "na.dmc",
            "model Node { v: i64 }\npub fn na() -> i64 { let n = Node { v: 2 }  n.v }\n",
        ),
        (
            "nb.dmc",
            "model Node { a: i64, b: i64 }\n\
             pub fn nb() -> i64 { let n = Node { a: 1, b: 2 }  n.a + n.b }\n",
        ),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(5));

    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"na$Node".to_string()), "{names:?}");
    assert!(names.contains(&"nb$Node".to_string()), "{names:?}");
}

// ─── Alias imports ───────────────────────────────────────────────────────────

/// `use "f.dmc" as a` then `a.g()`. The call arrives as a postfix chain —
/// `Field` then `Call` — and the merge collapses the `Field` onto the target's
/// merged name, so the backend sees an ordinary call and needs to learn nothing
/// about modules.
#[test]
fn jit_merge_rewrites_an_aliased_call() {
    let t = Tree::new(&[
        ("e.dmc", "use \"f.dmc\" as a\nfn main() -> i64 { a.g() + a.h(1) }\n"),
        ("f.dmc", "pub fn g() -> i64 { 7 }\npub fn h(n: i64) -> i64 { n + 2 }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(10));
}

/// The measured case §3(a) rests the "rewrite references, not declarations"
/// rule on: one `pub fn g` imported unqualified by one file and as `s` by
/// another, with an entry importing both. An either/or rename would need `g`
/// and `s.g` to name one function in one flat item list.
#[test]
fn jit_merge_handles_one_pub_fn_imported_both_ways() {
    let t = Tree::new(&[
        ("e.dmc", "use \"plain.dmc\"\nuse \"aliased.dmc\"\nfn main() -> i64 { p() + q() }\n"),
        ("plain.dmc", "use \"shared.dmc\"\npub fn p() -> i64 { g() }\n"),
        ("aliased.dmc", "use \"shared.dmc\" as s\npub fn q() -> i64 { s.g() }\n"),
        ("shared.dmc", "pub fn g() -> i64 { 5 }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(10));
}

/// An aliased *model* constructor. The parser folds `u.Pair { .. }` into one
/// dotted `StructLit` name, so the merge rewrites the name rather than a chain.
#[test]
fn jit_merge_rewrites_an_aliased_model_constructor() {
    let t = Tree::new(&[
        ("e.dmc", "use \"m.dmc\" as u\nfn main() -> i64 { let p = u.Pair { a: 4, b: 6 }  p.a + p.b }\n"),
        ("m.dmc", "pub model Pair { a: i64, b: i64 }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(10));
}

// ─── `extern fn` ─────────────────────────────────────────────────────────────

/// Carve-out 2: an `extern fn`'s declared name *is* the foreign symbol, so it is
/// never prefixed. Two files declaring the same one collapse to a single
/// declaration — a second would be an unguarded overwrite of `Jit::fns`.
#[test]
fn jit_merge_dedups_a_matching_extern_fn() {
    let t = Tree::new(&[
        ("e.dmc", "use \"xa.dmc\"\nuse \"xb.dmc\"\nfn main() -> i64 { xa() + xb() }\n"),
        ("xa.dmc", "extern fn abs(x: i32) -> i32\npub fn xa() -> i64 { 4 }\n"),
        ("xb.dmc", "extern fn abs(x: i32) -> i32\npub fn xb() -> i64 { 6 }\n"),
    ]);
    let p = merged(&t.path("e.dmc")).expect("matching externs collapse");
    let names = declared_names(&p);
    assert_eq!(names.iter().filter(|n| *n == "abs").count(), 1, "{names:?}");
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(10));
}

/// The other half of carve-out 2: two declarations of one foreign symbol that
/// disagree cannot be prefixed apart, so they are a hard error naming both
/// files and both signatures.
#[test]
fn jit_merge_refuses_two_extern_fns_that_disagree() {
    let t = Tree::new(&[
        ("e.dmc", "use \"xa.dmc\"\nuse \"xb.dmc\"\nfn main() -> i64 { xa() + xb() }\n"),
        ("xa.dmc", "extern fn abs(x: i32) -> i32\npub fn xa() -> i64 { 4 }\n"),
        ("xb.dmc", "extern fn abs(x: i64) -> i64\npub fn xb() -> i64 { 6 }\n"),
    ]);
    let r = resolved(&t.path("e.dmc"));
    let err = merge_program(&r, &t.path("e.dmc")).expect_err("mismatched externs are refused");
    match &err {
        MergeError::ExternMismatch { name, first, second, .. } => {
            assert_eq!(name, "abs");
            let both = format!("{:?} {:?}", first, second);
            assert!(both.contains("xa.dmc") && both.contains("xb.dmc"), "{both}");
        }
        other => panic!("expected ExternMismatch, got {other:?}"),
    }
}

// ─── `main` in a dependency ──────────────────────────────────────────────────

/// "Only the entry file may declare one" is false: a dependency declaring
/// `pub fn main` passes `--check`, and `dmc run` returns the entry's. Merged
/// unprefixed it would collide in `Jit::fns`, so slice 1's default — pending
/// the ruling in §6 — prefixes a non-entry `main` like any private item,
/// whatever its visibility, so it can never be the entry.
#[test]
fn jit_merge_prefixes_a_dependency_pub_fn_main() {
    let t = Tree::new(&[
        ("e.dmc", "use \"dep.dmc\"\nfn main() -> i64 { d() + 1 }\n"),
        ("dep.dmc", "pub fn main() -> i64 { 99 }\npub fn d() -> i64 { 1 }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(2));

    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"dep$main".to_string()), "{names:?}");
    assert_eq!(names.iter().filter(|n| *n == "main").count(), 1, "{names:?}");
}

// ─── Cross-file UFCS (#627) ──────────────────────────────────────────────────

/// `desugar_ufcs` runs inside `Parser::parse_program` and builds its function
/// set from the one file it is handed, so `v.twice()` onto an *imported*
/// `pub fn` is never rewritten. It arrives as a postfix `Field` then `Call` —
/// exactly the shape an aliased call has.
///
/// The merge must not mistake one for the other. It keys the alias rewrite on
/// "the base identifier is bound to a module by a `use ... as`", so this chain
/// is left exactly as written and the backend refuses it as the method call it
/// looks like. Refused, not miscompiled — the failure mode this test exists to
/// exclude is the merge rewriting it into a call on a module that does not
/// exist, or the program compiling and answering something other than 6.
#[test]
fn jit_merge_leaves_cross_file_ufcs_alone_627() {
    let t = Tree::new(&[
        ("e.dmc", "use \"lib.dmc\"\nfn main() -> i64 { let v = 3  v.twice() }\n"),
        ("lib.dmc", "pub fn twice(v: i64) -> i64 { v * 2 }\n"),
    ]);
    // The merge itself succeeds — it has nothing to say about this chain.
    let p = merged(&t.path("e.dmc")).expect("merge");
    assert!(declared_names(&p).contains(&"twice".to_string()));

    let answer = jit_answer(&t.path("e.dmc"));
    let msg = match &answer {
        Err(m) => m.clone(),
        Ok(v) => panic!("an un-desugared cross-file UFCS call must be refused, not answered: {v}"),
    };
    // `Refusal::UnknownFn` — the `jit-unknown-fn` class — naming the method
    // that was actually written. Not `jit-import`: every `use` was dropped, and
    // the residual is a call shape the backend has no lowering for.
    assert!(msg.contains(".twice()"), "the refusal must name the method: {msg}");
}

// ─── Enums, consts and type aliases across files ─────────────────────────────

/// `Item::Enum` takes the enum name only: variants are spelled `Enum.Variant`
/// and both JIT maps key on the enum name, so no variant is separately renamed.
/// Two files' private enums must still not share a `Jit::enums` slot.
#[test]
fn jit_merge_keeps_two_private_enums_apart() {
    let t = Tree::new(&[
        ("e.dmc", "use \"ea.dmc\"\nuse \"eb.dmc\"\nfn main() -> i64 { ea() * 10 + eb() }\n"),
        ("ea.dmc", "enum E { A, B }\npub fn ea() -> i64 { E.B as i64 }\n"),
        ("eb.dmc", "enum E { X, Y, Z }\npub fn eb() -> i64 { E.Z as i64 }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(12));

    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"ea$E".to_string()) && names.contains(&"eb$E".to_string()), "{names:?}");
}

/// A top-level `let` is what lands in `global_consts`, one of §5.1's unguarded
/// inserts. Two files' private constants of one name must not overwrite each
/// other, and a reference must follow the rename.
///
/// **This is the one shape where the merged JIT and `dmc run` disagree, and the
/// JIT is right.** #652 gave the interpreter per-module resolution for
/// functions, models, enums and aliases, but a top-level `let` still lands in
/// one flat global scope, so `dmc run` answers **22** on this exact program:
/// `cb.dmc`'s `let K = 2` overwrites `ca.dmc`'s, and both `ca()` and `cb()`
/// read the survivor. Per-module semantics — what this test asserts, and what
/// `--check` type-checks — is `12`. That gap is filed as **#657**; it is an
/// interpreter bug, not a merge bug, so the assertion below stays at `12`.
///
/// Consequence for slice 3: once `dmc test --jit` runs the merged program,
/// this shape will report a run-vs-jit **parity failure** until #657 lands. The
/// failure is the interpreter's; nothing here should be relaxed to hide it.
#[test]
fn jit_merge_keeps_two_private_consts_apart() {
    let t = Tree::new(&[
        ("e.dmc", "use \"ca.dmc\"\nuse \"cb.dmc\"\nfn main() -> i64 { ca() * 10 + cb() }\n"),
        ("ca.dmc", "let K = 1\npub fn ca() -> i64 { K }\n"),
        ("cb.dmc", "let K = 2\npub fn cb() -> i64 { K }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(12));

    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"ca$K".to_string()) && names.contains(&"cb$K".to_string()), "{names:?}");
}

/// After #622 an alias name is a type position exactly like a model name:
/// `resolve_alias_ty` keys on the bare name inside a `Type::Named`, so a
/// prefixed `ta$Role` declaration must be found by a rewritten `Role` in a
/// signature — and, worse, an unprefixed `Role` from a *different* file must
/// not be.
#[test]
fn jit_merge_renames_type_aliases_in_type_positions() {
    let t = Tree::new(&[
        ("e.dmc", "use \"ta.dmc\"\nuse \"tb.dmc\"\nfn main() -> i64 { ta(7) + tb(3) }\n"),
        ("ta.dmc", "type Role = i64\npub fn ta(r: Role) -> i64 { r }\n"),
        ("tb.dmc", "type Role = i32\npub fn tb(r: Role) -> i64 { r as i64 }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(10));

    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"ta$Role".to_string()) && names.contains(&"tb$Role".to_string()), "{names:?}");
}

// ─── Shadowing, and the single-file identity ─────────────────────────────────

/// A parameter or `let` binding shadows a top-level name, so an identifier
/// inside its scope is not a reference to the declaration the merge renamed.
/// Without the shadow stack this program's `helper` parameter would be rewritten
/// to the merged function name and the body would call the function instead of
/// reading the argument.
#[test]
fn jit_merge_does_not_rewrite_a_shadowed_identifier() {
    let t = Tree::new(&[
        ("e.dmc", "use \"lib.dmc\"\nfn main() -> i64 { shadow(4) + lib_val() }\n"),
        (
            "lib.dmc",
            "fn helper() -> i64 { 100 }\n\
             pub fn shadow(helper: i64) -> i64 { helper }\n\
             pub fn lib_val() -> i64 { let helper = 6  helper }\n",
        ),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(10));
}

/// A program with no `use` merges to itself, name for name — every kind of
/// declaration included. This is what keeps slice 3 from changing a single-file
/// program's behaviour at all: the corpus is overwhelmingly single-file, a `dmc
/// jit` refusal keeps naming the symbol the user wrote, and `dmc test --jit`
/// keeps finding `test_*` by the name it read off the file.
#[test]
fn jit_merge_of_a_single_file_renames_nothing() {
    let t = Tree::new(&[(
        "e.dmc",
        "model Node { v: i64 }\n\
         enum E { A, B }\n\
         type Role = i64\n\
         let K = 3\n\
         fn helper(r: Role) -> i64 { r }\n\
         fn test_thing() -> bool { true }\n\
         fn main() -> i64 { let n = Node { v: 7 }  n.v + helper(K) + (E.A as i64) }\n",
    )]);
    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert_eq!(names, vec!["Node", "E", "Role", "K", "helper", "test_thing", "main"]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(10));
}

/// The entry file's `test_*` functions keep their bare names, for the same
/// reason its `main` does: `Jit::run_test_fn` looks one up by the literal
/// string `dmc test --jit` read off the entry file's AST, so a prefixed
/// `test_x` would be a missing entry point rather than a wrong answer. A
/// dependency's are prefixed like any other private item.
#[test]
fn jit_merge_keeps_the_entry_files_test_fn_names() {
    let t = Tree::new(&[
        (
            "e.dmc",
            "use \"lib.dmc\"\nfn test_entry() -> bool { helped() == 1 }\nfn main() -> i64 { 0 }\n",
        ),
        ("lib.dmc", "fn test_dep() -> bool { true }\npub fn helped() -> i64 { 1 }\n"),
    ]);
    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"test_entry".to_string()), "{names:?}");
    assert!(names.contains(&"lib$test_dep".to_string()), "{names:?}");
}

/// Two files whose stems sanitize alike are told apart by their path relative
/// to the entry's directory — `one/util.dmc` gives `one$util$h` — so the prefix
/// is collision-free without depending on directory names being distinct after
/// sanitization, and *without* depending on the `sorted_paths` index.
#[test]
fn jit_merge_disambiguates_colliding_file_stems_by_relative_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir(dir.path().join("one")).unwrap();
    std::fs::create_dir(dir.path().join("two")).unwrap();
    std::fs::write(dir.path().join("one/util.dmc"), "fn h() -> i64 { 4 }\npub fn u1() -> i64 { h() }\n").unwrap();
    std::fs::write(dir.path().join("two/util.dmc"), "fn h() -> i64 { 6 }\npub fn u2() -> i64 { h() }\n").unwrap();
    std::fs::write(
        dir.path().join("e.dmc"),
        "use \"one/util.dmc\"\nuse \"two/util.dmc\"\nfn main() -> i64 { u1() + u2() }\n",
    )
    .unwrap();
    let entry = dir.path().join("e.dmc").canonicalize().unwrap();

    let names = declared_names(&merged(&entry).expect("merge"));
    assert!(names.contains(&"one$util$h".to_string()), "{names:?}");
    assert!(names.contains(&"two$util$h".to_string()), "{names:?}");
    // No index anywhere: the relative path already separated them.
    assert!(!names.iter().any(|n| n.contains("util$0$") || n.contains("util$1$")), "{names:?}");

    assert_eq!(jit_answer(&entry), Ok(10));
}

/// The disambiguator must be a function of the *file*, not of what else is in
/// the merge. Adding an unrelated third file renumbers `sorted_paths`, so an
/// index-derived prefix would silently rename `one/util.dmc`'s private `h` —
/// changing a name that appears in JIT diagnostics, and in any dump of the
/// merged unit, for an edit in a different directory. The relative-path key
/// does not move.
#[test]
fn jit_merge_stem_disambiguator_is_stable_under_an_unrelated_file() {
    fn names_of(with_extra: bool) -> Vec<String> {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("one")).unwrap();
        std::fs::create_dir(dir.path().join("two")).unwrap();
        std::fs::write(dir.path().join("one/util.dmc"), "fn h() -> i64 { 4 }\npub fn u1() -> i64 { h() }\n").unwrap();
        std::fs::write(dir.path().join("two/util.dmc"), "fn h() -> i64 { 6 }\npub fn u2() -> i64 { h() }\n").unwrap();
        // `aaa.dmc` sorts before both `util.dmc` paths, so it takes index 0 and
        // pushes each of them up by one.
        let (extra_use, extra_call) = if with_extra {
            std::fs::write(dir.path().join("aaa.dmc"), "pub fn z() -> i64 { 0 }\n").unwrap();
            ("use \"aaa.dmc\"\n", " + z()")
        } else {
            ("", "")
        };
        std::fs::write(
            dir.path().join("e.dmc"),
            format!(
                "{extra_use}use \"one/util.dmc\"\nuse \"two/util.dmc\"\n\
                 fn main() -> i64 {{ u1() + u2(){extra_call} }}\n"
            ),
        )
        .unwrap();
        let entry = dir.path().join("e.dmc").canonicalize().unwrap();
        assert_eq!(jit_answer(&entry), Ok(10));
        let mut n: Vec<String> = declared_names(&merged(&entry).expect("merge"))
            .into_iter()
            .filter(|n| n.ends_with("$h"))
            .collect();
        n.sort();
        n
    }
    assert_eq!(names_of(false), vec!["one$util$h".to_string(), "two$util$h".to_string()]);
    assert_eq!(names_of(true), names_of(false), "an unrelated file must not rename a private symbol");
}

/// `merge_program` refuses an entry the resolver never saw rather than silently
/// merging some other file's DAG.
#[test]
fn jit_merge_refuses_an_unresolved_entry() {
    let t = Tree::new(&[("e.dmc", "fn main() -> i64 { 1 }\n"), ("other.dmc", "fn main() -> i64 { 2 }\n")]);
    let r = resolved(&t.path("e.dmc"));
    assert!(matches!(
        merge_program(&r, &t.path("other.dmc")),
        Err(MergeError::EntryNotResolved { .. })
    ));
}

// ─── The name map back to the source ─────────────────────────────────────────

/// `merge_program` returns the way back from a merged name to the declaration
/// it came from, because `jit.rs` builds its diagnostics out of the AST it is
/// handed — `unknown function \`{}\`` and `duplicate function \`{}\`` would
/// otherwise print a `$`-prefixed name no user wrote. Slice 3 spends the map;
/// slice 1 owes it completeness, which is the half that has to be true at the
/// moment of renaming.
///
/// Every renamed item is in it, under its declaring file and source name; every
/// item the merge left bare is *not*, and `source_name` hands those back
/// unchanged. `$` cannot occur in a user identifier, so "contains `$`" and
/// "was renamed" are the same predicate and the check below is exhaustive.
#[test]
fn jit_merge_maps_every_renamed_name_back_to_its_source() {
    let t = Tree::new(&[
        ("e.dmc", "use \"ha.dmc\"\nuse \"hb.dmc\"\nfn main() -> i64 { a1() + b1() }\n"),
        ("ha.dmc", HELPER_A),
        ("hb.dmc", HELPER_B),
    ]);
    let m = merged(&t.path("e.dmc")).expect("merge");

    for (merged_name, file) in [("ha$helper", "ha.dmc"), ("hb$helper", "hb.dmc")] {
        assert_eq!(m.source_name(merged_name), "helper", "{merged_name}");
        assert!(
            m.declaring_file(merged_name).is_some_and(|p| p.ends_with(file)),
            "`{merged_name}` must name the file that declared it"
        );
    }

    for n in declared_names(&m) {
        if n.contains('$') {
            assert!(m.names.contains_key(&n), "renamed `{n}` is missing from the name map");
        } else {
            assert!(!m.names.contains_key(&n), "bare `{n}` must not be in the name map");
            assert_eq!(m.source_name(&n), n, "a bare name is its own source name");
        }
    }
    for bare in ["main", "a1", "a2", "b1", "b2"] {
        assert!(!m.names.contains_key(bare), "`{bare}` was never renamed");
    }

    // The single-file path renames nothing, so it carries no map at all.
    let s = Tree::new(&[(
        "e.dmc",
        "model Node { v: i64 }\nfn helper() -> i64 { 1 }\nfn main() -> i64 { helper() }\n",
    )]);
    assert!(merged(&s.path("e.dmc")).expect("merge").names.is_empty());
}

// ─── `test_*` outside the entry file ─────────────────────────────────────────

/// The carve-out is the **entry file's** `test_*`, not the name `test_*`. A
/// dependency's `pub fn test_x` is a legal program today — `--check` accepts it
/// and `dmc test dep.dmc` runs it — and if the carve-out were read as "any
/// `test_*` stays bare" it would land on the entry's `test_x` and the merge
/// would refuse, with `DuplicateMerged`, a pair both backends already run.
/// `test_*` is therefore treated exactly like `main` in both directions:
/// prefixed outside the entry whatever its visibility, bare only inside it.
#[test]
fn jit_merge_prefixes_a_dependencys_pub_test_fn() {
    let t = Tree::new(&[
        (
            "e.dmc",
            "use \"dep.dmc\"\nfn test_x() -> bool { d() == 1 }\nfn main() -> i64 { 0 }\n",
        ),
        ("dep.dmc", "pub fn test_x() -> bool { true }\npub fn d() -> i64 { 1 }\n"),
    ]);
    let m = merged(&t.path("e.dmc"))
        .expect("a dependency's `pub fn test_x` must merge, not collide with the entry's");
    let names = declared_names(&m);
    assert!(names.contains(&"dep$test_x".to_string()), "{names:?}");
    assert_eq!(
        names.iter().filter(|n| *n == "test_x").count(),
        1,
        "only the entry's `test_x` keeps the name `run_test_fn` looks up: {names:?}"
    );
    assert_eq!(m.source_name("dep$test_x"), "test_x");
}

// ─── Colliding `pub` names, the cases the rewrite has to reach ───────────────

/// A colliding `pub` name called from **its own declaring file**. Each file's
/// own declarations enter its scope before any import, so `c1.dmc`'s `own()`
/// must follow `g` to `c1$g` — the bare `g` it wrote exists nowhere in the
/// merged unit. Mid-level files keep the entry from importing both leaves, so
/// this is the prefix rule answering, not `Ambiguous`.
#[test]
fn jit_merge_rewrites_a_colliding_pub_fn_called_from_its_own_file() {
    let t = Tree::new(&[
        ("e.dmc", "use \"m1.dmc\"\nuse \"m2.dmc\"\nfn main() -> i64 { m1() * 10 + m2() }\n"),
        ("m1.dmc", "use \"c1.dmc\"\npub fn m1() -> i64 { own1() }\n"),
        ("m2.dmc", "use \"c2.dmc\"\npub fn m2() -> i64 { own2() }\n"),
        ("c1.dmc", "pub fn g() -> i64 { 1 }\npub fn own1() -> i64 { g() }\n"),
        ("c2.dmc", "pub fn g() -> i64 { 2 }\npub fn own2() -> i64 { g() }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(12));

    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"c1$g".to_string()) && names.contains(&"c2$g".to_string()), "{names:?}");
    assert!(!names.contains(&"g".to_string()), "no bare `g` survives: {names:?}");
}

/// A colliding `pub` **model** name in a type position — a `pub model` field
/// whose type is the other file's `Point`. This is the shape §5.1 is about with
/// `pub` visibility: `model_layouts` is a bare `insert`, so two `Point`s on one
/// key would give the second file's field offsets to the first file's accesses,
/// with no diagnostic anywhere. The answer is the check: `12` needs `ba`'s
/// one-field `Point` and `bb`'s two-field one to stay separate layouts.
#[test]
fn jit_merge_prefixes_a_colliding_pub_model_used_as_a_field_type() {
    let t = Tree::new(&[
        ("e.dmc", "use \"ba.dmc\"\nuse \"bb.dmc\"\nfn main() -> i64 { fa() * 10 + fb() }\n"),
        (
            "ba.dmc",
            "use \"pa.dmc\"\n\
             pub model BoxA { p: Point }\n\
             pub fn fa() -> i64 { let b = BoxA { p: Point { x: 1 } }  b.p.x }\n",
        ),
        (
            "bb.dmc",
            "use \"pb.dmc\"\n\
             pub model BoxB { p: Point }\n\
             pub fn fb() -> i64 { let b = BoxB { p: Point { x: 0, y: 2 } }  b.p.y }\n",
        ),
        ("pa.dmc", "pub model Point { x: i64 }\n"),
        ("pb.dmc", "pub model Point { x: i64, y: i64 }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(12));

    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"pa$Point".to_string()) && names.contains(&"pb$Point".to_string()), "{names:?}");
    assert!(!names.contains(&"Point".to_string()), "no bare `Point` survives: {names:?}");
}

/// One entry reaching **both halves of a colliding pair**, one bare and one
/// through an alias. The two arrive on different scope keys — `g` and `s.g` —
/// so nothing is ambiguous, and each has to land on a different merged name:
/// the bare reference on `c1$g`, the aliased one on `c2$g`.
#[test]
fn jit_merge_reaches_a_colliding_pub_pair_bare_and_by_alias() {
    let t = Tree::new(&[
        ("e.dmc", "use \"c1.dmc\"\nuse \"c2.dmc\" as s\nfn main() -> i64 { g() * 10 + s.g() }\n"),
        ("c1.dmc", "pub fn g() -> i64 { 1 }\n"),
        ("c2.dmc", "pub fn g() -> i64 { 2 }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(12));

    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"c1$g".to_string()) && names.contains(&"c2$g".to_string()), "{names:?}");
}

/// The entry importing **one** of a colliding pair directly, the other only
/// through a mid-level file. The entry's bare `g` must resolve to the leaf it
/// imported and not to the one that happens to be merged nearer it.
#[test]
fn jit_merge_lets_an_entry_import_one_of_a_colliding_pair_directly() {
    let t = Tree::new(&[
        ("e.dmc", "use \"c1.dmc\"\nuse \"m2.dmc\"\nfn main() -> i64 { g() * 10 + m2() }\n"),
        ("c1.dmc", "pub fn g() -> i64 { 1 }\n"),
        ("m2.dmc", "use \"c2.dmc\"\npub fn m2() -> i64 { g() }\n"),
        ("c2.dmc", "pub fn g() -> i64 { 2 }\n"),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(12));
}

// ─── `@grad` in a dependency (design §4.1(f)) ────────────────────────────────

/// A `@grad fn` declared in a **dependency**, called through `.fwd_bwd` from a
/// `pub fn` in that same file. `declare_grad_fn` registers three entries off
/// one `f.name` — the forward `f`, the mangled `f$fwd_bwd`, and the
/// `grad_schemas` entry under that same mangled key — and the call site looks
/// the schema up as `format!("{}$fwd_bwd", fname)` with `fname` read off the
/// rewritten call expression. Because the merge renames the *declaration* and
/// both sides derive the mangled name from it, `grad_schemas` inherits the
/// rename for free: the key is `dep$dot_loss$fwd_bwd`.
///
/// The run is the assertion. A schema registered under `dot_loss$fwd_bwd` while
/// the call site asks for `dep$dot_loss$fwd_bwd` is a `None` lookup, which is
/// the hard error "`...` has no mutable parameters" — so an answer of `10` can
/// only mean the two mangled names met.
#[test]
fn jit_merge_renames_a_dependencys_grad_fn_and_its_fwd_bwd_key() {
    let t = Tree::new(&[
        ("e.dmc", "use \"dep.dmc\"\nfn main() -> i64 { fit() }\n"),
        (
            "dep.dmc",
            "@grad fn dot_loss(!w: Tensor[f32, [4]], x: Tensor[f32, [4]]) -> f32 { sum(w .* x) }\n\
             pub fn fit() -> i64 {\n\
             \x20   let !w = forge.zeros[f32, [4]]\n\
             \x20   let !x = forge.zeros[f32, [4]]\n\
             \x20   x[0] = 10.0\n\
             \x20   let (_l, g) = dot_loss.fwd_bwd(w, x)\n\
             \x20   g.w[0] as i64\n\
             }\n",
        ),
    ]);
    assert_eq!(jit_answer(&t.path("e.dmc")), Ok(10));

    let names = declared_names(&merged(&t.path("e.dmc")).expect("merge"));
    assert!(names.contains(&"dep$dot_loss".to_string()), "{names:?}");
    assert!(!names.contains(&"dot_loss".to_string()), "{names:?}");
}
