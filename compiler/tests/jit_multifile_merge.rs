//! The whole-program merge, exercised at the command line (slice 1 of the
//! multi-file design).
//!
//! The merge pass itself is exercised end-to-end in `src/merge_tests.rs`: this
//! crate has no library target, so only an in-crate test can call
//! `merge_program` and hand the result to `Jit::compile_program`, and until
//! slice 3 wires it into the `dmc jit` arm no *command* reaches it.
//!
//! What belongs here is the half a command can see, and it is two things.
//!
//! First, the **interpreter's** answers on the programs the merge is measured
//! against. `SPEC.md §9` promises visibility is enforced across file boundaries;
//! before #652 the interpreter loaded every file into one flat name table and
//! the later file overwrote, so two files each declaring a file-private
//! `fn helper` answered `2 2 2 2` or `1 1 1 1` by `use` order where per-module
//! semantics answers `1 1 2 2`, and two leaves each exporting `pub fn g` behind
//! two mid-level files answered `22` where per-module semantics answers `12`
//! (#613). #652 fixed both. The merge is built to reproduce exactly these
//! answers, so if the interpreter ever drifts back the JIT gate's premise is
//! gone — these tests are what says so out loud rather than leaving
//! `src/merge_tests.rs` pinning one backend against nothing.
//!
//! Second, the **refusal slice 3 flipped**. A multi-file program used not to
//! compile under `dmc jit` — check errors before #624, `jit-import` after — and
//! now it does: the `jit` arm hands `compile_program` the merged program rather
//! than the entry file's own. What is left of `jit-import` on this path is the
//! merge itself refusing, which is the same fact ("a `use` the JIT could not
//! lower") arriving one pass later.
//!
//! Third, the **diagnostics that cross a file boundary**. A merged name is a
//! name no user wrote, so no message may quote one, and a refusal raised inside
//! an imported item must report the file that declared it rather than the file
//! typed on the command line (design §5.2, and §6's second open question —
//! answered here: the declaring file).

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

    /// `dmc jit`'s stdout, asserting it ran.
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

    fn path_of(&self, name: &str) -> std::path::PathBuf {
        self.dir.path().join(name)
    }
}

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
const PRINT_FOUR: &str = "fn main() -> nil { print(a1(), a2(), b1(), b2()) nil }\n";

/// #652: a body resolves its names against the file it was written in, so two
/// leaves' file-private `fn helper`s stay apart and the answer no longer depends
/// on the order the entry writes its `use`s. This is the per-module answer the
/// merged program must reproduce; the test writes both entry files itself, so
/// the `use` order is fixed here and is not a property of the two leaves.
///
/// Before #652 this printed `2 2 2 2` in one order and `1 1 1 1` in the other.
#[test]
fn interpreter_scopes_private_fns_per_module_652() {
    for (label, uses) in [
        ("a then b", "use \"ha.dmc\"\nuse \"hb.dmc\"\n"),
        ("b then a", "use \"hb.dmc\"\nuse \"ha.dmc\"\n"),
    ] {
        let entry = format!("{uses}{PRINT_FOUR}");
        let t = Tree::new(&[("e.dmc", &entry), ("ha.dmc", HELPER_A), ("hb.dmc", HELPER_B)]);
        assert_eq!(t.run_output(), "1 1 2 2", "with the `use`s in {label} order");
    }
}

/// #652, the half that is *not* confined to private names: two leaves each
/// export `pub fn g`, two mid-level files import one each, and the entry imports
/// the two mids — so no single file sees both and `ambiguous import` never
/// fires. `--check` passes. Per-module semantics answers `12`, and since #652 so
/// does the interpreter, where it used to answer `22`.
///
/// This is the program the design's §3(a) would have made a hard merge error and
/// its §1 answers by prefixing. Because `dmc run` gets it right, erroring would
/// refuse a program the interpreter runs correctly, so the merge prefixes —
/// see `merge.rs`'s module docs.
#[test]
fn interpreter_answers_the_pub_fn_g_diamond_per_module_652() {
    let t = Tree::new(&[
        ("e.dmc", "use \"m1.dmc\"\nuse \"m2.dmc\"\nfn main() -> i64 { m1() * 10 + m2() }\n"),
        ("m1.dmc", "use \"g1.dmc\"\npub fn m1() -> i64 { g() }\n"),
        ("m2.dmc", "use \"g2.dmc\"\npub fn m2() -> i64 { g() }\n"),
        ("g1.dmc", "pub fn g() -> i64 { 1 }\n"),
        ("g2.dmc", "pub fn g() -> i64 { 2 }\n"),
    ]);
    assert!(t.dmc(&["--check"]).status.success(), "the program type-checks");
    assert_eq!(t.run_output(), "=> 12");
}

/// A dependency declaring `pub fn main` passes `--check` and `dmc run` returns
/// the *entry's*. The merge must therefore not let a dependency's `main` reach
/// `Jit::fns` under that name, which is why it prefixes one whatever its
/// visibility (design §3(a), *`main` in a dependency*; the ruling in §6 may
/// later narrow this to a hard error).
#[test]
fn interpreter_runs_the_entrys_main_when_a_dependency_declares_one() {
    let t = Tree::new(&[
        ("e.dmc", "use \"dep.dmc\"\nfn main() -> i64 { d() + 1 }\n"),
        ("dep.dmc", "pub fn main() -> i64 { 99 }\npub fn d() -> i64 { 1 }\n"),
    ]);
    assert_eq!(t.run_output(), "=> 2");
}

/// The fact slice 3 turned over. A multi-file program used to run under
/// `dmc run` and not under `dmc jit` — the checker rejected it before #624, the
/// backend refused it as `jit-import` after — and now both backends answer.
#[test]
fn dmc_jit_runs_a_multi_file_program() {
    let t = Tree::new(&[
        ("e.dmc", "use \"lib.dmc\"\nfn main() -> i64 { lib_val() }\n"),
        ("lib.dmc", "pub fn lib_val() -> i64 { 10 }\n"),
    ]);
    assert_eq!(t.run_output(), "=> 10", "the interpreter runs it");
    assert_eq!(t.jit_output(), "=> 10", "and so does the JIT");
}

/// The design's own diamond (§0, Appendix A): an entry over two mid-level
/// modules onto one shared leaf, with the leaf's `pub model` crossing two file
/// boundaries as a field type. Every file declares its own `fn main`, so the
/// entry-`main` carve-out is load-bearing here — a prefixed entry `main` would
/// be a missing entry point rather than a wrong answer.
///
/// §0.2 measured the *hand-flattened* form as already compiling; this is the
/// same program delivered through the merge instead of by hand.
#[test]
fn dmc_jit_runs_the_diamond_through_the_merge() {
    let t = Tree::new(&[
        (
            "d.dmc",
            "use \"boxed.dmc\"\nuse \"sum.dmc\"\n\
             fn main() -> i64 { let bx = boxed(4, 5)  sum_pt(bx.p) + bx.tag }\n",
        ),
        (
            "boxed.dmc",
            "use \"point.dmc\"\npub model Boxed { p: Point, tag: i64 }\n\
             pub fn boxed(x: i64, y: i64) -> Boxed { Boxed { p: mk(x, y), tag: 1 } }\n\
             fn main() -> nil { print(\"dep\") nil }\n",
        ),
        (
            "sum.dmc",
            "use \"point.dmc\"\npub fn sum_pt(p: Point) -> i64 { p.x + p.y }\n\
             fn main() -> nil { print(\"dep\") nil }\n",
        ),
        (
            "point.dmc",
            "pub model Point { x: i64, y: i64 }\n\
             pub fn mk(x: i64, y: i64) -> Point { Point { x: x, y: y } }\n\
             fn main() -> nil { print(\"dep\") nil }\n",
        ),
    ]);
    assert_eq!(t.run_output(), "=> 10");
    assert_eq!(t.jit_output(), "=> 10");
}

/// Slice 2's rename, at the command line: an aliased import reaches the JIT as
/// a postfix `Field` then `Call` on an identifier bound to a module, and the
/// merge rewrites it to the `.`-joined key `check.rs` already uses.
#[test]
fn dmc_jit_runs_an_aliased_import() {
    let t = Tree::new(&[
        ("a.dmc", "use \"mod.dmc\" as u\nfn main() -> i64 { u.mk(1, 3).x + u.twice(4) }\n"),
        (
            "mod.dmc",
            "pub model P { x: i64 }\npub fn mk(a: i64, b: i64) -> P { P { x: a + b } }\n\
             pub fn twice(v: i64) -> i64 { v * 2 }\n",
        ),
    ]);
    assert_eq!(t.run_output(), "=> 12");
    assert_eq!(t.jit_output(), "=> 12");
}

/// `dmc test --jit` merges each entry's own closure, so a multi-file test file
/// is `jit_ran` rather than `jit_skipped`. The accounting keeps its shape — a
/// refusal is still a skip — and only which files land on which side moves.
#[test]
fn dmc_test_jit_runs_a_multi_file_entry_instead_of_skipping_it() {
    let t = Tree::new(&[
        ("t.dmc", "use \"lib.dmc\"\nfn test_add() -> bool { lib_val() == 10 }\n"),
        ("lib.dmc", "pub fn lib_val() -> i64 { 10 }\n"),
    ]);
    let out = Command::new(dmc_binary())
        .args(["test", "--jit"])
        .arg(t.dir.path())
        .output()
        .expect("spawn dmc");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "{}", text);
    assert!(text.contains("jit parity: 1 ran, 0 skipped"), "{}", text);
}

/// The `jit-import` class after slice 3: still "a `use` the JIT could not
/// lower", raised by the merge rather than by `collect_fns`. `DIAGNOSTICS.md
/// §6` makes refusal codes append-only, so the class is not retired for having
/// moved one pass later.
///
/// The shape is the one collision the merge genuinely cannot rename apart: an
/// `extern fn` name *is* the foreign symbol, so two files declaring one name
/// with two signatures have nothing to be told apart by. `--check` accepts it
/// and `dmc run` answers, which is what makes it a subset boundary rather than
/// a type error.
#[test]
fn a_merge_collision_is_refused_as_jit_ineligible_not_as_a_jit_error() {
    let t = Tree::new(&[
        ("e.dmc", "use \"xa.dmc\"\nuse \"xb.dmc\"\nfn main() -> i64 { av() + bv() }\n"),
        ("xa.dmc", "extern fn dmc_probe_sym(x: i64) -> i64\npub fn av() -> i64 { 1 }\n"),
        ("xb.dmc", "extern fn dmc_probe_sym(x: f64) -> f64\npub fn bv() -> i64 { 2 }\n"),
    ]);
    assert!(t.dmc(&["--check"]).status.success(), "the program type-checks");
    assert_eq!(t.run_output(), "=> 3", "and the interpreter answers");

    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{:#?}", json);
    let d = &json[0];
    assert!(d.contains("\"kind\":\"jit-ineligible\""), "{}", d);
    assert!(d.contains("\"code\":\"jit-import\""), "{}", d);
    assert!(d.contains("dmc_probe_sym"), "the refusal does not name the item: {}", d);
    // A refusal, never a defect (#480): `dmc run` has the full semantics.
    assert!(!d.contains("jit-error"), "{}", d);
}

/// The design's §5.2 risk and §6's second open question, answered: a refusal
/// raised inside an imported item reports **that** file, not the one typed on
/// the command line. Before the merge was wired in, `jit_diag` stamped the
/// command's path onto a `JitError` that carries no file of its own, so such a
/// refusal read as the entry file's path beside the imported file's line.
#[test]
fn a_refusal_inside_an_imported_item_reports_the_declaring_file() {
    let t = Tree::new(&[
        ("e.dmc", "use \"lib.dmc\"\nfn main() -> i64 { lib_val() }\n"),
        // `**` on ints is not lowered (`jit-operand-type`, #215), inside a
        // file-private fn of the dependency — so the refusal is raised under
        // the merged name `lib$helper`, in a file the command never named.
        (
            "lib.dmc",
            "fn helper(v: i64) -> i64 { v ** 2 }\npub fn lib_val() -> i64 { helper(10) }\n",
        ),
    ]);
    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{:#?}", json);
    let d = &json[0];
    let lib = t.path_of("lib.dmc").canonicalize().expect("canonicalize lib.dmc");
    assert!(
        d.contains(&format!("\"file\":\"{}\"", lib.display())),
        "the refusal was not attributed to the declaring file: {}", d,
    );
    assert!(d.contains("\"line\":1"), "{}", d);
}

/// #661 review finding 1. `declare_extern_fns` (JIT pass 0) never set
/// `current_decl_file` while walking its own items, so an `extern fn` refusal
/// raised there inherited whatever `current_decl_file` last held — here,
/// nothing (the merged program has no enum, so the pre-pass never touches
/// it), leaving the fallback to blame the entry file rather than the file the
/// `extern fn` was actually written in.
///
/// Both files declare the *same* extern signature, so the merge dedups them
/// (`extern_sig_key` agrees) rather than refusing — the refusal instead comes
/// from `declare_one_extern_fn`'s own dlsym pre-flight, once, for whichever
/// declaration the merge kept. That declaration is `xa.dmc`'s (leaf-first
/// merge order), which is the file this refusal must name.
#[test]
fn an_extern_refusal_is_attributed_to_its_own_declaring_file_not_the_entry() {
    let t = Tree::new(&[
        ("e.dmc", "use \"xa.dmc\"\nuse \"xb.dmc\"\nfn main() -> i64 { av() + bv() }\n"),
        ("xa.dmc", "extern fn dmc_probe_sym(x: i64) -> i64\npub fn av() -> i64 { 1 }\n"),
        ("xb.dmc", "extern fn dmc_probe_sym(x: i64) -> i64\npub fn bv() -> i64 { 2 }\n"),
    ]);
    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{:#?}", json);
    let d = &json[0];
    assert!(d.contains("\"code\":\"jit-extern\""), "{}", d);
    let xa = t.path_of("xa.dmc").canonicalize().expect("canonicalize xa.dmc");
    assert!(
        d.contains(&format!("\"file\":\"{}\"", xa.display())),
        "the refusal was attributed to the wrong file (expected xa.dmc): {}", d,
    );
}

/// #661 review finding 1's second shape: the enum-layout pre-pass (which runs
/// before `declare_extern_fns` and sets `current_decl_file` for every
/// payload-carrying enum it lays out) left the tracker pointed at `ea.dmc`
/// with nothing to reset it, so `declare_extern_fns` inherited that stale
/// value instead of setting its own — an `extern fn` in a *different* file
/// (`eb.dmc`) was blamed on `ea.dmc`, even though the line number (read
/// straight off the `extern fn`'s own span, never from the tracker) was
/// already correct. `eb.dmc`'s four leading comment lines put its `extern fn`
/// at line 5, past `ea.dmc`'s own 2 lines — so the bug could not be mistaken
/// for a coincidence of the two files' line counts.
#[test]
fn an_extern_refusal_is_not_attributed_to_an_unrelated_enums_file() {
    let t = Tree::new(&[
        ("e.dmc", "use \"ea.dmc\"\nuse \"eb.dmc\"\nfn main() -> i64 { ea_v() + eb_v() }\n"),
        ("ea.dmc", "enum E { A(i64), B }\npub fn ea_v() -> i64 { 1 }\n"),
        (
            "eb.dmc",
            "# line 1\n# line 2\n# line 3\n# line 4\n\
             extern fn dmc_probe_sym(x: i64) -> i64\npub fn eb_v() -> i64 { 2 }\n",
        ),
    ]);
    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{:#?}", json);
    let d = &json[0];
    assert!(d.contains("\"code\":\"jit-extern\""), "{}", d);
    let eb = t.path_of("eb.dmc").canonicalize().expect("canonicalize eb.dmc");
    assert!(
        d.contains(&format!("\"file\":\"{}\"", eb.display())),
        "the refusal was attributed to ea.dmc (the enum's file) instead of \
         eb.dmc (the extern's own file): {}", d,
    );
    assert!(d.contains("\"line\":5"), "{}", d);
}

/// A merged name is a name no user wrote, so none may reach a message. The
/// merge prefixes a dependency's private items `<stem>$<name>` to keep two
/// files' `fn helper`s off one key in the JIT's flat symbol table, and every
/// name-bearing diagnostic in `jit.rs` reads its name straight off the merged
/// AST — so the guarantee has to be bought at the exit, not per message.
///
/// `$` is the whole test: the lexer admits no `$` in an identifier, so a `$` in
/// a `--json` message can only be a mangling that escaped.
#[test]
fn no_merged_name_reaches_a_json_message() {
    // Each fixture is (files, what it refuses for), chosen so the refusal is
    // raised *inside* a dependency's private item — the only place a merged
    // name can come from.
    let fixtures: Vec<(Vec<(&str, &str)>, &str)> = vec![
        (
            vec![
                ("e.dmc", "use \"lib.dmc\"\nfn main() -> i64 { lib_val() }\n"),
                (
                    "lib.dmc",
                    "fn helper(v: i64) -> i64 { v ** 2 }\n\
                     pub fn lib_val() -> i64 { helper(10) }\n",
                ),
            ],
            "integer `**` in a private fn of a dependency",
        ),
        (
            vec![
                ("e.dmc", "use \"lib.dmc\"\nfn main() -> i64 { lib_val() }\n"),
                (
                    "lib.dmc",
                    "model Node { v: i64 }\nfn make() -> Node { Node { v: 3 } }\n\
                     pub fn lib_val() -> i64 { let n = make()  n.w }\n",
                ),
            ],
            "a field the private model does not have",
        ),
        (
            vec![
                ("e.dmc", "use \"xa.dmc\"\nuse \"xb.dmc\"\nfn main() -> i64 { av() + bv() }\n"),
                ("xa.dmc", "extern fn dmc_probe_sym(x: i64) -> i64\npub fn av() -> i64 { 1 }\n"),
                ("xb.dmc", "extern fn dmc_probe_sym(x: f64) -> f64\npub fn bv() -> i64 { 2 }\n"),
            ],
            "a merge collision",
        ),
    ];
    for (files, what) in fixtures {
        let t = Tree::new(&files);
        let json = t.jit_json();
        assert!(!json.is_empty(), "{what}: expected a refusal, got none");
        for d in &json {
            let message = message_of(d);
            assert!(
                !message.contains('$'),
                "{what}: a merged name reached the user in {message:?}",
            );
        }
    }
}

/// The positive half of the rule above: not merely "no `$` got out", but that
/// the name the user actually wrote got out in its place. Without the
/// merged→source map this exact diagnostic reads ``model `lib$Node` has no
/// field `w` `` — a model the user never declared, in a file they never named.
#[test]
fn a_diagnostic_quotes_the_source_name_of_a_renamed_item() {
    let t = Tree::new(&[
        ("e.dmc", "use \"lib.dmc\"\nfn main() -> i64 { lib_val() }\n"),
        (
            "lib.dmc",
            "model Node { v: i64 }\nfn make() -> Node { Node { v: 3 } }\n\
             pub fn lib_val() -> i64 { let n = make()  n.w }\n",
        ),
    ]);
    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{:#?}", json);
    assert_eq!(message_of(&json[0]), "model `Node` has no field `w`");
}

/// The `"message"` field of a `--json` diagnostic line, unescaped enough to
/// look for a `$` in. Hand-rolled rather than pulled in with a JSON crate: this
/// crate has no JSON dependency, and one field of one well-known shape does not
/// earn one.
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

/// The control the design rests §0.2 on, kept honest at the command line: the
/// diamond hand-flattened into one file compiles and agrees under both backends
/// today. If this ever stops holding, the merge's premise — that no new lowering
/// is needed — is gone.
#[test]
fn the_hand_flattened_diamond_agrees_on_both_backends() {
    let src = "\
model Point { x: i64, y: i64 }
fn mk(x: i64, y: i64) -> Point { Point { x: x, y: y } }
model Boxed { p: Point, tag: i64 }
fn boxed(x: i64, y: i64) -> Boxed { Boxed { p: mk(x, y), tag: 1 } }
fn sum_pt(p: Point) -> i64 { p.x + p.y }
fn main() -> i64 { let bx = boxed(4, 5)  sum_pt(bx.p) + bx.tag }
";
    let t = Tree::new(&[("flat.dmc", src)]);
    assert_eq!(t.run_output(), "=> 10");
    let jit = t.dmc(&["jit"]);
    assert!(
        jit.status.success(),
        "{}{}",
        String::from_utf8_lossy(&jit.stdout),
        String::from_utf8_lossy(&jit.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&jit.stdout).trim(), "=> 10");
}
