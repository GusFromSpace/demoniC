//! Whole-program merge for the JIT.
//!
//! The JIT's unit of compilation is one `Program` — one file's AST — inside one
//! `JITModule` with one flat symbol table (`jit.rs`, `Jit::fns`). It has no
//! module concept and nowhere to put a second file's functions, so
//! `collect_fns` refuses `Item::Use` outright (`Refusal::Import`, the
//! `jit-import` class). What it needs is not modules but **one item list no two
//! files can collide inside**.
//!
//! This pass builds that list: after the resolver has parsed and desugared the
//! whole import DAG, `merge_program` concatenates every file's items in
//! `sorted_paths` order (leaf-first, entry last), drops every `Item::Use`, and
//! renames as it goes so that each declaration has exactly one canonical merged
//! name and every reference names it.
//!
//! # The rename rule
//!
//! * **Every non-`pub` declaration takes its file's prefix, unconditionally** —
//!   not only where it would collide, and not only where the pair was noticed.
//!   Collision-freedom must not be a function of what else is in the merge: the
//!   JIT's `model_layouts`,
//!   `model_field_asts`, `model_shape_params`, `enums`, `enum_payload_layouts`
//!   and `global_consts` are bare `insert`s where the last writer wins, so a
//!   pair the rule failed to notice is a silent miscompile with no diagnostic
//!   (design §5.1), not a `duplicate function` error. The one exception is a
//!   merge of **one** file, which is the identity: it has no pairs, and never
//!   will, so prefixing there would rename every private symbol of every
//!   single-file program in the corpus and put a name no user wrote into its
//!   refusal diagnostics, for nothing.
//! * **A `pub` declaration keeps its bare name when no other file exports that
//!   name in the same key space.** It is the name importers wrote.
//! * **When two files export one `pub` name, both are prefixed** and each
//!   importer's references are rewritten to the one it imported. See
//!   *Deviation* below.
//! * **`extern fn` is never renamed.** Its declared name *is* the foreign
//!   symbol (`Linkage::Import`/dlsym) and `pub extern fn` is a compile-time
//!   error, so an `extern fn` is file-private yet unprefixable. Two files
//!   declaring the same extern collapse to one declaration when the signatures
//!   agree, and are a `MergeError` when they do not.
//! * **The entry file's `main` and its `test_*` functions keep their bare
//!   names.** This is the design's carve-out 1 and its stated reason —
//!   `Jit::run_main` looks up the literal string `"main"` — applied to the
//!   other literal-name entry point the JIT has: `Jit::run_test_fn(name)`,
//!   which `dmc test --jit` calls with a name read off the *entry* file's AST.
//!   A prefixed `test_x` would be a missing entry point, not a wrong answer.
//! * **A `main` or `test_*` declared in a dependency is always prefixed**,
//!   `pub` or not, so it can never be mistaken for the entry point (design
//!   §3(a), *`main` in a dependency*; the ruling in §6 may later narrow the
//!   `main` case to a hard error, which is strictly narrower acceptance and
//!   costs nothing here). Both names are treated the same way in both
//!   directions: a dependency's `pub fn test_x` is legal today — `--check`
//!   accepts it and `dmc test dep.dmc` runs it — so if it kept its bare name it
//!   would land on the entry's carved-out `test_x` and the merge would refuse a
//!   program both backends run.
//!
//! The prefix is `<stem>$<name>`; where two files' stems sanitize alike it is
//! the file's path **relative to the entry file's directory**, `$`-joined —
//! `sub/leaf.dmc` becomes `sub$leaf$<name>` — and only if two files still key
//! alike after that does the `sorted_paths` index break the tie. The relative
//! path is preferred to the index because the index is a function of *what else
//! is in the merge*: adding one unrelated file would otherwise renumber, and
//! so rename, a private symbol that had nothing to do with it. `$` is the
//! separator because the lexer admits
//! `[A-Za-z_\x80-\xFF][A-Za-z0-9_\x80-\xFF]*!?` and nothing else: **no user
//! identifier can contain `$`**, so the prefix is collision-free by
//! construction rather than by a check — where `util__helper` is a name a user
//! may write. It is also the escape the JIT already mangles with
//! (`format!("{}$fwd_bwd", ...)`), which is why `grad_schemas` inherits the fn
//! rename for free, and it is distinct from the `.`-joined alias key `check.rs`
//! uses, so the two never have to be told apart.
//!
//! # Deviation from the design doc, and why
//!
//! The original merge design contradicted itself on one point. One passage
//! rules that "two files exporting the same `pub` name is a hard merge
//! error"; another passage rules the same case — the `pub fn g` diamond,
//! where two leaves each export `pub fn g` and no single file imports both — one that
//! "prefixing, not this check, is what answers". The doc was measured before
//! #652, when the interpreter's flat name table answered `22` on that program
//! where per-module semantics answers `12`, and erroring was then the only way
//! not to reproduce a known-wrong answer.
//!
//! #652 landed. `dmc run` now answers `12`. Erroring would refuse a program the
//! interpreter runs correctly, which the slice gate forbids, so slice 1 follows
//! §1: **a colliding `pub` name is prefixed like a private one**, and the
//! merged program reproduces per-module semantics exactly. The hard error is
//! kept for the case prefixing genuinely cannot answer — one file's scope
//! binding one name to two different declarations, which is the shape
//! `check.rs` already rules `ambiguous import`.
//!
//! # Ordering, and what the merge is *not*
//!
//! `desugar_ufcs` runs inside `Parser::parse_program` and `comptime::fold_program`
//! runs per file, so the merge comes **last**: after parse-and-desugar, after
//! the fold, after the per-file leaf-first check. The price of being last is
//! that it sees AST `Type`s rather than checked types, which is why the type
//! rewrite here is its own thing and not `check.rs`'s `prefix_types_in_ty`.
//!
//! One consequence is load-bearing (#627): a UFCS call onto an **imported**
//! `pub fn` — `v.twice()` where `twice` came from a `use` — never desugared,
//! because `desugar_ufcs` builds its function set from the one file it is
//! handed. It arrives as a postfix `Field` then `Call`, the same shape an
//! aliased call has. The merge must not mistake one for the other, so the
//! alias rewrite below keys on **"the base identifier is bound to a module by a
//! `use ... as`"**, never on the chain's shape. A cross-file UFCS chain is
//! therefore left exactly as written and reaches the backend, which refuses it
//! as the method call it looks like. Refused, not miscompiled.
//!
//! The merged program is **not re-checked** (design §4.1(e)) — a second full
//! check would re-report every diagnostic under prefixed names no user wrote,
//! over a program that exists nowhere in the source. What it does instead is
//! assert, after renaming, that no two declarations landed on one key in a map
//! the JIT actually keys on. That is the property the unguarded `insert`s lose
//! silently, and it is bought in O(items) before Cranelift sees anything.
//!
//! # The way back to the source name
//!
//! A merged name is a name no user wrote, and `jit.rs` builds its diagnostics
//! out of the AST it is handed — `unknown function \`{}\``, `duplicate function
//! \`{}\`` and the rest read the merged spelling. So `merge_program` returns a
//! [`MergedProgram`], not a bare `Program`: alongside the item list it carries
//! `merged name → (declaring file, name as written)` for **every renamed
//! item**, and [`MergedProgram::source_name`] maps a merged name back.
//! [`MergedProgram::decl_files`] answers the other half of the same question —
//! *which file* a merged name came from — for **every** declaration and not
//! only the renamed ones, because a dependency's uncontested `pub fn` keeps its
//! bare name and a refusal inside it must still report that file rather than
//! the one typed on the command line (design §5.2).
//!
//! Slice 3 is what spends both: `Jit::set_merge` hands this value to the
//! backend, which routes every `JitError` on its way out through
//! `source_name`, and stamps the declaring file onto `--json`'s `file`. That is
//! why the map has to be complete at the moment of renaming — nothing later can
//! reconstruct it.
//!
//! # JIT-only
//!
//! `merge_program` is never called by the interpreter, which keeps per-module
//! semantics of its own (#652) and a per-module `@grad` epoch (#398). Hanging
//! this off `Resolver` would invite a caller to merge on a path where the
//! interpreter would then see prefixed names, so it is a free function taking
//! `&Resolver`, and nothing in `interp.rs` mentions it.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::ast::*;
use crate::lexer::Span;
use crate::resolver::Resolver;

// ─── Key spaces ──────────────────────────────────────────────────────────────

/// The JIT's own maps, as name spaces. Not one set over every declared name:
/// the backend's namespaces are separate maps that nothing cross-checks, so
/// `model Foo { x: i64 }` beside `fn Foo() -> i64 { 1 }` compiles today and the
/// merge must never be stricter than the backend it feeds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
enum Ns {
    /// `Jit::fns` + `Jit::templates`, one key space — both are tested before a
    /// concrete fn or a template is registered. `extern fn` names live here too.
    Fn,
    /// `model_layouts` / `model_field_asts` / `model_shape_params`, one key:
    /// all three are inserted together under the model name.
    Model,
    /// `enums` + `enum_payload_layouts`, one key: the enum name. Variants are
    /// spelled `Enum.Variant` and are not separately renamed.
    Enum,
    /// `global_consts` — the top-level `let` name.
    Const,
    /// `Jit::aliases` — the alias name (#622 made top-level `type` transparent
    /// rather than refused, so an alias name is a type position like a model
    /// name and must be rewritten wherever one is).
    Alias,
}

impl Ns {
    /// The human word for this key space, for a `MergeError`.
    fn what(self) -> &'static str {
        match self {
            Ns::Fn => "function",
            Ns::Model => "model",
            Ns::Enum => "enum",
            Ns::Const => "constant",
            Ns::Alias => "type alias",
        }
    }
}

/// Where an identifier in **expression** position may resolve, in priority
/// order. Callee position is by far the most common, so functions come first;
/// `Enum` is here because `Color.Red` reaches the merge as a postfix `Field` on
/// the bare identifier `Color`, and `Model` because `@cast(Model) { .. }` (#598)
/// names a model in a directive argument.
const EXPR_NS: [Ns; 5] = [Ns::Fn, Ns::Const, Ns::Enum, Ns::Model, Ns::Alias];

/// Where a name in **type** position may resolve.
const TYPE_NS: [Ns; 3] = [Ns::Model, Ns::Alias, Ns::Enum];

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Why a merge could not be built.
///
/// Every variant names the files involved, because the merged program exists
/// nowhere in the source: a reader given only a name has no way back to the
/// declaration that produced it.
#[derive(Debug)]
pub enum MergeError {
    /// `entry` is not among the resolver's files — `resolve_all` was not run,
    /// or was run on a different path.
    EntryNotResolved { entry: PathBuf },
    /// One file's scope binds one name to two different declarations, which is
    /// the shape `check.rs` rules `ambiguous import`. Prefixing cannot answer
    /// this: the importer wrote one name and two declarations answer to it.
    Ambiguous {
        what: &'static str,
        name: String,
        importer: PathBuf,
        first: PathBuf,
        second: PathBuf,
        span: Span,
        /// The entry file's directory, carried only so `Display` can render
        /// `first`/`second`/`importer` relative to it (#661 review finding
        /// 3) — never read by `file()`, which still hands back the absolute,
        /// canonical path for `--json`'s `file` field.
        entry_dir: PathBuf,
    },
    /// Two `extern fn` declarations of one name disagree. An extern is never
    /// renamed — its name *is* the foreign symbol — so there is nothing to
    /// prefix them apart with.
    ExternMismatch {
        name: String,
        first: PathBuf,
        first_sig: String,
        second: PathBuf,
        second_sig: String,
        span: Span,
        /// See `Ambiguous::entry_dir`.
        entry_dir: PathBuf,
    },
    /// Two declarations landed on one key in a map the JIT keys on. This is the
    /// assertion §4.1(e) buys in place of re-checking the merged program.
    DuplicateMerged {
        what: &'static str,
        merged: String,
        first: PathBuf,
        second: PathBuf,
        span: Span,
        /// See `Ambiguous::entry_dir`.
        entry_dir: PathBuf,
    },
}

/// A path for a `MergeError` message, relative to `entry_dir` when the two
/// share a root (the common case — every file the merge touches sits under
/// or beside the entry) and the absolute path otherwise. Cosmetic only: it is
/// used for `Display`'s prose, never for `MergeError::file`, which stays the
/// absolute, canonical path `--json`'s `file` field needs.
///
/// Mirrors `relative_key`'s common-prefix walk but keeps real path syntax
/// (`sub/leaf.dmc`, extension and all) instead of `relative_key`'s
/// sanitized, `$`-joined identifier fragment — the two serve different
/// readers, one a JIT symbol table and this one a person.
fn display_path(p: &Path, entry_dir: &Path) -> String {
    let file: Vec<_> = p.components().collect();
    let dir: Vec<_> = entry_dir.components().collect();
    let common = file.iter().zip(dir.iter()).take_while(|(a, b)| a == b).count();
    // The whole path is a prefix of itself only when `p` and `entry_dir` are
    // literally the same path — not a real case here, but falling back to the
    // absolute path rather than printing an empty string is the safe answer.
    if common == file.len() {
        return p.display().to_string();
    }
    let mut rel = PathBuf::new();
    for _ in common..dir.len() {
        rel.push("..");
    }
    for c in &file[common..] {
        rel.push(c.as_os_str());
    }
    rel.display().to_string()
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::EntryNotResolved { entry } => write!(
                f,
                "merge: entry file {:?} was not resolved — `resolve_all` must run first",
                entry
            ),
            MergeError::Ambiguous { what, name, importer, first, second, entry_dir, .. } => write!(
                f,
                "merge: the {} `{}` is declared by both {} and {}, and {} sees both under \
                 that one name; use an aliased import (`use \"...\" as alias`) to disambiguate",
                what, name,
                display_path(first, entry_dir), display_path(second, entry_dir),
                display_path(importer, entry_dir),
            ),
            MergeError::ExternMismatch { name, first, first_sig, second, second_sig, entry_dir, .. } => write!(
                f,
                "merge: `extern fn {}` is declared with different signatures in {} ({}) and \
                 {} ({}); an extern name is the foreign symbol and cannot be renamed apart",
                name,
                display_path(first, entry_dir), first_sig,
                display_path(second, entry_dir), second_sig,
            ),
            MergeError::DuplicateMerged { what, merged, first, second, entry_dir, .. } => write!(
                f,
                "merge: the {} `{}` is declared by both {} and {} and both merged to that one \
                 name",
                what, merged, display_path(first, entry_dir), display_path(second, entry_dir),
            ),
        }
    }
}

impl MergeError {
    /// The file the failure's span is measured in — a span is a byte offset
    /// into one file's text, so reporting one without the other is how a
    /// diagnostic points at a line that does not exist.
    ///
    /// For `Ambiguous` that is the *importing* file, whose `use` the span
    /// names; for the other two it is the second declaration, the one that
    /// could not be reconciled with what was already there.
    pub fn file(&self) -> Option<&Path> {
        match self {
            MergeError::EntryNotResolved { .. } => None,
            MergeError::Ambiguous { importer, .. } => Some(importer.as_path()),
            MergeError::ExternMismatch { second, .. }
            | MergeError::DuplicateMerged { second, .. } => Some(second.as_path()),
        }
    }

    /// The location the failure is anchored at, when there is one.
    pub fn span(&self) -> Option<&Span> {
        match self {
            MergeError::EntryNotResolved { .. } => None,
            MergeError::Ambiguous { span, .. }
            | MergeError::ExternMismatch { span, .. }
            | MergeError::DuplicateMerged { span, .. } => Some(span),
        }
    }
}

// ─── The pass ────────────────────────────────────────────────────────────────

/// One declaration, as collected from one file before any renaming.
struct Decl {
    ns: Ns,
    /// The name as written.
    name: String,
    /// Index into the merge's file list.
    file: usize,
    is_pub: bool,
    is_extern: bool,
    span: Span,
}

/// What one file's references resolve against: its own declarations, the `pub`
/// declarations of the files it imports unqualified, and `alias.name` keys for
/// the files it imports aliased.
#[derive(Default)]
struct FileScope {
    /// `(key space, name-as-written)` → merged name. `name-as-written` is the
    /// bare name for an unqualified import and `alias.name` for an aliased one.
    names: HashMap<(Ns, String), String>,
    /// Which file each entry came from, for the ambiguity diagnostic.
    from: HashMap<(Ns, String), usize>,
    /// Identifiers bound to a module by `use "..." as alias`. The alias rewrite
    /// keys on *this*, never on the shape of a postfix chain — see the module
    /// docs on #627.
    module_aliases: HashSet<String>,
}

impl FileScope {
    fn get(&self, order: &[Ns], name: &str) -> Option<&str> {
        order
            .iter()
            .find_map(|ns| self.names.get(&(*ns, name.to_string())))
            .map(|s| s.as_str())
    }
}

/// The result of a merge: one `Program`, and the way back from every merged
/// name to the declaration it came from.
#[derive(Debug)]
pub struct MergedProgram {
    /// The single unit the JIT compiles.
    pub program: Program,
    /// Merged name → (declaring file, name as written).
    ///
    /// **Renamed items only.** A declaration the merge left bare — every
    /// declaration of a single-file program, an uncontested `pub` name, an
    /// `extern fn`, the entry's `main` and `test_*` — is absent, because for
    /// those the merged name *is* the source name and an entry would say
    /// nothing. `source_name` therefore returns its argument unchanged for
    /// them, and the map is empty on the single-file path.
    ///
    /// Keyed by the merged name alone and not by `(namespace, merged name)`:
    /// a diagnostic has a string and no namespace. That is sound because a
    /// renamed name carries its declaring file's prefix, so only that file can
    /// produce it, and the prefix determines both halves of the value — two
    /// namespaces of one file colliding on a merged name (`model Foo` beside
    /// `fn Foo`, which the backend permits) agree on the file *and* on `Foo`.
    pub names: HashMap<String, (PathBuf, String)>,
    /// Merged name → declaring file, for **every** top-level declaration, not
    /// only the renamed ones.
    ///
    /// `names` answers "what did the user call this?", which only a renamed
    /// item can answer differently from its merged spelling. This answers
    /// "which file is this from?", which every merged item answers differently
    /// from the entry file — a dependency's uncontested `pub fn` keeps its bare
    /// name and is absent from `names`, yet a refusal inside it must still
    /// report *that* file and not the one on the command line (design §5.2,
    /// and §6's second open question, which slice 3 answers "the declaring
    /// file").
    ///
    /// First writer wins, which matters for the one item form the merge emits
    /// once from several declarations: two files declaring the same `extern fn`
    /// collapse to the first file's declaration, so the first file is also the
    /// one whose text a diagnostic quotes.
    pub decl_files: HashMap<String, PathBuf>,
}

impl MergedProgram {
    /// The name the user actually wrote, for a name read off the merged AST.
    /// Returns `merged` unchanged when the merge did not rename it.
    ///
    /// This is what keeps a `$`-prefixed name out of a user-facing diagnostic.
    /// Nothing in slice 1 renders one; `jit.rs`'s messages are slice 3's to
    /// route through here, together with the declaring file, since a bare
    /// source name is ambiguous across files in a way the merged one is not.
    pub fn source_name<'a>(&'a self, merged: &'a str) -> &'a str {
        self.names.get(merged).map(|(_, n)| n.as_str()).unwrap_or(merged)
    }

    /// The file that declared a merged name — for any top-level declaration of
    /// the merge, renamed or not. `None` only for a name the merge never
    /// declared (a JIT-internal mangling, a builtin, a typo in a diagnostic).
    ///
    /// This is what `dmc jit --json` reports as a refusal's `file` once slice 3
    /// hands the merged program to `compile_program`: without it every refusal
    /// raised inside an imported item would carry the entry file's path beside
    /// the imported file's line number (design §5.2).
    pub fn declaring_file(&self, merged: &str) -> Option<&Path> {
        self.decl_files.get(merged).map(|p| p.as_path())
    }
}

/// Merge the whole import DAG rooted at `entry` into one `Program` the JIT can
/// compile as a single unit.
///
/// `r` must already have been through `resolve_all` (and, on the paths that use
/// it, `comptime` folding and the per-file leaf-first check — the merged
/// program is not re-checked). Every `Item::Use` is dropped; every remaining
/// item is emitted once, in `sorted_paths` order, under its merged name.
pub fn merge_program(r: &Resolver, entry: &Path) -> Result<MergedProgram, MergeError> {
    // `resolve_all` canonicalizes before it stores, so the lookup must too. A
    // caller that already holds the canonical path is unaffected.
    let entry_path = entry.canonicalize().unwrap_or_else(|_| entry.to_path_buf());

    // `sorted_paths` is the topological order — leaf-first, entry last — and
    // the resolver's `visited` early return means each canonical path appears
    // exactly once, so no shared leaf is merged twice.
    let paths: Vec<PathBuf> = r.sorted_paths.clone();
    let entry_idx = match paths.iter().position(|p| *p == entry_path) {
        Some(i) => i,
        None => return Err(MergeError::EntryNotResolved { entry: entry_path }),
    };

    let entry_dir = entry_path.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
    let prefixes = file_prefixes(&paths, &entry_dir);

    // ── 1. Collect every declaration, per file. ──────────────────────────────
    let mut decls: Vec<Vec<Decl>> = Vec::with_capacity(paths.len());
    for (i, p) in paths.iter().enumerate() {
        let prog = match r.files.get(p) {
            Some(prog) => prog,
            // A path in `sorted_paths` with no program is not reachable through
            // `Resolver`'s own API; treat it as an empty file rather than
            // panicking on a caller that built one by hand.
            None => {
                decls.push(Vec::new());
                continue;
            }
        };
        let mut out = Vec::new();
        for item in &prog.items {
            collect_decls(item, false, i, &mut out);
        }
        decls.push(out);
    }

    // ── 2. Decide each declaration's canonical merged name. ──────────────────
    //
    // A `pub` name is bare only when no *other* file exports it in the same key
    // space; a `main` or `test_*` outside the entry file is prefixed whatever
    // its visibility; the entry file's `main` and `test_*` fns are bare because
    // the JIT looks them up by literal string.
    //
    // Only `Ns::Fn` and `Ns::Model` are reachable here in practice. `check.rs`
    // does not carry a `pub enum` across a `use` (#655) and a `pub let` is a
    // hard error (#616 / #604), so no program can today put an `Ns::Enum` or an
    // `Ns::Const` key in this map. The arms below are deliberately written for
    // all five key spaces anyway — the rename rule is about the JIT's maps, not
    // about what the checker currently exports, and the day either issue lands
    // this needs no edit — but nothing here is exercised for those two, so do
    // not read them as tested parity.
    let mut pub_exporters: HashMap<(Ns, String), Vec<usize>> = HashMap::new();
    for file_decls in &decls {
        for d in file_decls {
            if d.is_pub && !d.is_extern && !is_literal_entry_point(d.ns, &d.name) {
                let e = pub_exporters.entry((d.ns, d.name.clone())).or_default();
                if !e.contains(&d.file) {
                    e.push(d.file);
                }
            }
        }
    }

    let mut merged_of: Vec<HashMap<(Ns, String), String>> =
        vec![HashMap::new(); paths.len()];
    // A merge of one file is the identity. Prefixing there would rename every
    // private symbol of every single-file program in the corpus for no gain —
    // there is nothing to collide with, permanently, not just at this moment —
    // and would put a name no user wrote into every JIT refusal diagnostic. The
    // unconditionality §3(a) argues for is about *pairs* across files, and a
    // one-file merge has none.
    let single_file = paths.len() == 1;
    // Merged name → (declaring file, name as written), for the renamed items.
    let mut name_map: HashMap<String, (PathBuf, String)> = HashMap::new();
    // Merged name → declaring file, for every declaration. See `decl_files`.
    let mut file_map: HashMap<String, PathBuf> = HashMap::new();
    for file_decls in &decls {
        for d in file_decls {
            let keeps_bare_name = single_file
                || d.is_extern
                || (d.file == entry_idx && is_literal_entry_point(d.ns, &d.name))
                || (d.is_pub
                    && !is_literal_entry_point(d.ns, &d.name)
                    && pub_exporters
                        .get(&(d.ns, d.name.clone()))
                        .is_some_and(|v| v.len() == 1));
            let merged = if keeps_bare_name {
                d.name.clone()
            } else {
                let merged = format!("{}{}", prefixes[d.file], d.name);
                name_map.insert(merged.clone(), (paths[d.file].clone(), d.name.clone()));
                merged
            };
            file_map.entry(merged.clone()).or_insert_with(|| paths[d.file].clone());
            merged_of[d.file].insert((d.ns, d.name.clone()), merged);
        }
    }

    // ── 3. Build each file's reference scope. ────────────────────────────────
    let mut scopes: Vec<FileScope> = Vec::with_capacity(paths.len());
    for (i, p) in paths.iter().enumerate() {
        let mut scope = FileScope::default();
        // The file's own declarations first: they shadow an import of the same
        // name rather than colliding with it, matching `check.rs`, which
        // processes imports before collecting the file's own signatures and
        // lets the later collection win.
        for ((ns, name), merged) in &merged_of[i] {
            scope.names.insert((*ns, name.clone()), merged.clone());
            scope.from.insert((*ns, name.clone()), i);
        }
        let Some(prog) = r.files.get(p) else {
            scopes.push(scope);
            continue;
        };
        let parent = p.parent().unwrap_or_else(|| Path::new(""));
        for item in &prog.items {
            let Item::Use(us) = item else { continue };
            let target = match parent.join(&us.path).canonicalize() {
                Ok(t) => t,
                // The resolver already proved every `use` resolvable; a path
                // that stopped resolving between then and now contributes no
                // names rather than failing the merge.
                Err(_) => continue,
            };
            let Some(j) = paths.iter().position(|q| *q == target) else { continue };
            for d in &decls[j] {
                if !d.is_pub {
                    continue;
                }
                let Some(merged) = merged_of[j].get(&(d.ns, d.name.clone())) else { continue };
                match &us.alias {
                    Some(alias) => {
                        scope.module_aliases.insert(alias.clone());
                        let key = (d.ns, format!("{}.{}", alias, d.name));
                        scope.names.insert(key.clone(), merged.clone());
                        scope.from.insert(key, j);
                    }
                    None => {
                        let key = (d.ns, d.name.clone());
                        if let Some(existing) = scope.names.get(&key) {
                            if existing != merged {
                                let owner = scope.from.get(&key).copied().unwrap_or(j);
                                // The file's own declaration shadows; only two
                                // *imports* of one name are ambiguous.
                                if owner != i {
                                    return Err(MergeError::Ambiguous {
                                        what: d.ns.what(),
                                        name: d.name.clone(),
                                        importer: p.clone(),
                                        first: paths[owner].clone(),
                                        second: paths[j].clone(),
                                        span: us.span.clone(),
                                        entry_dir: entry_dir.clone(),
                                    });
                                }
                            }
                            continue;
                        }
                        scope.names.insert(key.clone(), merged.clone());
                        scope.from.insert(key, j);
                    }
                }
            }
        }
        scopes.push(scope);
    }

    // ── 4. Rewrite and concatenate. ──────────────────────────────────────────
    let mut items: Vec<Item> = Vec::new();
    // `extern fn` is never renamed, so one declaration of a name is emitted and
    // the rest are reconciled against it.
    let mut externs: HashMap<String, (usize, String)> = HashMap::new();
    for (i, p) in paths.iter().enumerate() {
        let Some(prog) = r.files.get(p) else { continue };
        let mut rw = Rewriter { scope: &scopes[i], merged: &merged_of[i], shadow: Vec::new() };
        for item in &prog.items {
            if matches!(item, Item::Use(_)) {
                continue;
            }
            if let Some(e) = extern_decl_of(item) {
                let sig = extern_sig_key(e);
                match externs.get(&e.name) {
                    Some((first, first_sig)) => {
                        if *first_sig != sig {
                            return Err(MergeError::ExternMismatch {
                                name: e.name.clone(),
                                first: paths[*first].clone(),
                                first_sig: first_sig.clone(),
                                second: p.clone(),
                                second_sig: sig,
                                span: e.span.clone(),
                                entry_dir: entry_dir.clone(),
                            });
                        }
                        // Same symbol, same signature: one declaration is
                        // enough, and a second would be an unguarded overwrite
                        // of `Jit::fns`.
                        continue;
                    }
                    None => {
                        externs.insert(e.name.clone(), (i, sig));
                    }
                }
            }
            let mut item = item.clone();
            rw.item(&mut item);
            items.push(item);
        }
    }

    // ── 5. Assert uniqueness per key space (design §4.1(e)). ─────────────────
    assert_unique(&paths, &decls, &merged_of, &entry_dir)?;

    let entry_prog = r.files.get(&entry_path);
    Ok(MergedProgram {
        program: Program {
            items,
            // The merged program exists in no file. It carries the entry's span
            // and line count so anything that reads them reads a real source
            // extent rather than a sum over files that were never one text;
            // nothing downstream re-checks it, so neither drives a diagnostic.
            span: entry_prog
                .map(|p| p.span.clone())
                .unwrap_or(Span { start: 0, end: 0, line: 1, col: 1 }),
            source_lines: entry_prog.map(|p| p.source_lines).unwrap_or(0),
        },
        names: name_map,
        decl_files: file_map,
    })
}

/// `main` and `test_*`: the names the JIT looks up as literal strings
/// (`Jit::run_main`, `Jit::run_test_fn`). On the **entry** file they are carved
/// out of the rename, because prefixing one would be a missing entry point
/// rather than a wrong answer. Everywhere else they are prefixed *whatever
/// their visibility*, which is the same predicate read the other way: a
/// dependency's `pub fn test_x` is a legal program today (`--check` accepts it,
/// `dmc test dep.dmc` runs it), so leaving it bare would put it on the entry's
/// carved-out `test_x` and turn a program both backends run into a
/// `DuplicateMerged`.
fn is_literal_entry_point(ns: Ns, name: &str) -> bool {
    ns == Ns::Fn && (name == "main" || name.starts_with("test_"))
}

/// `<stem>$` per file, and where two files' stems sanitize alike, a **stable**
/// disambiguator: the file's path relative to the entry file's directory,
/// `$`-joined (`sub/leaf.dmc` → `sub$leaf$`). Only when two files still key
/// alike after that — which takes a path that differs solely in characters the
/// sanitizer folds together — does the `sorted_paths` index break the tie.
///
/// The index is the last resort rather than the first because it is a function
/// of what else is in the merge: adding one unrelated file to the DAG would
/// renumber, and so rename, a private symbol in a file that had nothing to do
/// with it. Every merged name a diagnostic or a `--emit` dump can show is worth
/// keeping stable under an edit elsewhere in the program.
fn file_prefixes(paths: &[PathBuf], entry_dir: &Path) -> Vec<String> {
    let stems: Vec<String> = paths.iter().map(|p| sanitize_stem(p)).collect();
    let mut stem_counts: HashMap<&str, usize> = HashMap::new();
    for s in &stems {
        *stem_counts.entry(s.as_str()).or_insert(0) += 1;
    }

    // A unique stem is the whole key; a shared one falls back to the path.
    let keys: Vec<String> = paths
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if stem_counts[stems[i].as_str()] == 1 {
                stems[i].clone()
            } else {
                relative_key(p, entry_dir)
            }
        })
        .collect();

    let mut key_counts: HashMap<&str, usize> = HashMap::new();
    for k in &keys {
        *key_counts.entry(k.as_str()).or_insert(0) += 1;
    }
    keys.iter()
        .enumerate()
        .map(|(i, k)| {
            if key_counts[k.as_str()] > 1 {
                format!("{}${}$", k, i)
            } else {
                format!("{}$", k)
            }
        })
        .collect()
}

/// The file's path relative to the entry file's directory, sanitized component
/// by component, `$`-joined, extension dropped: `sub/leaf.dmc` under the entry's
/// directory is `sub$leaf`. A dependency *outside* that directory contributes
/// one `up` per level climbed, so `../shared/util.dmc` and `sibling/util.dmc`
/// cannot key alike. Both paths are canonical here, so they always share at
/// least a root component and the walk below never runs off either end.
fn relative_key(p: &Path, entry_dir: &Path) -> String {
    let file: Vec<_> = p.components().collect();
    let dir: Vec<_> = entry_dir.components().collect();
    let common = file.iter().zip(dir.iter()).take_while(|(a, b)| a == b).count();

    let mut parts: Vec<String> = Vec::new();
    for _ in common..dir.len() {
        parts.push("up".to_string());
    }
    let last = file.len().saturating_sub(1);
    for (k, c) in file.iter().enumerate().skip(common) {
        let raw = c.as_os_str().to_str().unwrap_or("mod");
        // The file name itself loses its extension; a directory keeps its name.
        let raw = if k == last {
            Path::new(raw).file_stem().and_then(|s| s.to_str()).unwrap_or(raw)
        } else {
            raw
        };
        parts.push(sanitize_ident(raw));
    }
    if parts.is_empty() {
        "mod".to_string()
    } else {
        parts.join("$")
    }
}

fn sanitize_stem(p: &Path) -> String {
    sanitize_ident(p.file_stem().and_then(|s| s.to_str()).unwrap_or("mod"))
}

/// One path component as an identifier fragment. Anything the lexer would not
/// admit becomes `_`; `$` is not in the mapped set, so a sanitized fragment can
/// never forge a separator.
fn sanitize_ident(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    if out.is_empty() {
        out.push_str("mod");
    }
    out
}

// ─── Declaration collection ──────────────────────────────────────────────────

fn collect_decls(item: &Item, is_pub: bool, file: usize, out: &mut Vec<Decl>) {
    match item {
        Item::Pub(inner) => collect_decls(inner, true, file, out),
        Item::Directive { inner, .. } => collect_decls(inner, is_pub, file, out),
        Item::Fn(f) => out.push(Decl {
            ns: Ns::Fn,
            name: f.name.clone(),
            file,
            is_pub,
            is_extern: false,
            span: f.span.clone(),
        }),
        Item::ExternFn(e) => out.push(Decl {
            ns: Ns::Fn,
            name: e.name.clone(),
            file,
            is_pub,
            is_extern: true,
            span: e.span.clone(),
        }),
        Item::Model(m) => out.push(Decl {
            ns: Ns::Model,
            name: m.name.clone(),
            file,
            is_pub,
            is_extern: false,
            span: m.span.clone(),
        }),
        Item::Enum(e) => out.push(Decl {
            ns: Ns::Enum,
            name: e.name.clone(),
            file,
            is_pub,
            is_extern: false,
            span: e.span.clone(),
        }),
        Item::TypeAlias(t) => out.push(Decl {
            ns: Ns::Alias,
            name: t.name.clone(),
            file,
            is_pub,
            is_extern: false,
            span: t.span.clone(),
        }),
        Item::Let(l) => {
            let mut names = Vec::new();
            pattern_names(&l.pattern, &mut names);
            for name in names {
                out.push(Decl {
                    ns: Ns::Const,
                    name,
                    file,
                    is_pub,
                    is_extern: false,
                    span: l.span.clone(),
                });
            }
        }
        Item::Arena(_) | Item::Use(_) => {}
    }
}

fn extern_decl_of(item: &Item) -> Option<&ExternFnDecl> {
    match item {
        Item::ExternFn(e) => Some(e),
        Item::Pub(inner) => extern_decl_of(inner),
        Item::Directive { inner, .. } => extern_decl_of(inner),
        _ => None,
    }
}

/// A canonical rendering of an `extern fn`'s ABI-visible signature. Parameter
/// *names* are deliberately not part of it: the C ABI does not carry them, so
/// two files spelling the same foreign symbol with different parameter names
/// declare the same thing.
fn extern_sig_key(e: &ExternFnDecl) -> String {
    let params: Vec<String> = e
        .params
        .iter()
        .map(|p| p.ty.as_ref().map(ty_key).unwrap_or_else(|| "?".to_string()))
        .collect();
    let ret = e.ret_type.as_ref().map(ty_key).unwrap_or_else(|| "nil".to_string());
    format!(
        "{}fn({}) -> {}",
        e.abi.as_ref().map(|a| format!("\"{}\" ", a)).unwrap_or_default(),
        params.join(", "),
        ret
    )
}

/// A structural key for a `Type`, span-free, so two written types can be
/// compared. `Debug` cannot be used for this: it embeds spans, which differ for
/// every occurrence.
fn ty_key(t: &Type) -> String {
    match t {
        // Source spelling (`i64`), not `Debug`'s `I64` — #661 review finding
        // 3: this key feeds `MergeError::ExternMismatch`'s message, the first
        // thing a `--json` consumer's `message` field shows a user, and that
        // reader wrote `i64`, never `I64`. Comparisons that use this key for
        // equality (not display) are unaffected: the mapping is still
        // injective, just spelled differently.
        Type::Scalar(s, _) => crate::fmt::scalar_type_str(s).to_string(),
        Type::Tensor(inner, sh, _) => format!("Tensor[{}, [{}]]", ty_key(inner), shape_key(sh)),
        Type::View(inner, sh, _) => format!("View[{}, [{}]]", ty_key(inner), shape_key(sh)),
        Type::KV(inner, sh, _) => format!("KV[{}, [{}]]", ty_key(inner), shape_key(sh)),
        Type::Mesh(axes, _) => format!(
            "Mesh[{}]",
            axes.iter().map(|a| format!("{}={}", a.name, expr_key(&a.size))).collect::<Vec<_>>().join(", ")
        ),
        Type::Fn(ps, ret, _) => format!(
            "fn({}) -> {}",
            ps.iter().map(ty_key).collect::<Vec<_>>().join(", "),
            ty_key(ret)
        ),
        Type::Tuple(es, _) => format!("({})", es.iter().map(ty_key).collect::<Vec<_>>().join(", ")),
        Type::Array(inner, n, _) => format!("[{}; {}]", ty_key(inner), expr_key(n)),
        Type::RawPtr(inner, _) => format!("*{}", ty_key(inner)),
        Type::Named { name, args, .. } => {
            if args.is_empty() {
                name.clone()
            } else {
                let a: Vec<String> = args
                    .iter()
                    .map(|a| match a {
                        TypeArg::Type(t) => ty_key(t),
                        TypeArg::Expr(e) => expr_key(e),
                        TypeArg::Named { name, value, .. } => format!("{}={}", name, expr_key(value)),
                    })
                    .collect();
                format!("{}[{}]", name, a.join(", "))
            }
        }
    }
}

fn shape_key(sh: &ShapeSpec) -> String {
    sh.elems
        .iter()
        .map(|e| match e {
            ShapeElem::Wildcard(_) => "_".to_string(),
            ShapeElem::Spread(_) => "..".to_string(),
            ShapeElem::Streaming(_) => "~".to_string(),
            ShapeElem::Expr(e) => expr_key(e),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// A coarse, span-free key for the expressions that can appear inside a type —
/// dimensions and array sizes. `extern fn` signatures are scalars, `*T` and
/// `nil` (SPEC §3.12), so this is only ever reached by the shape-carrying arms
/// above, where a literal or a name is the whole vocabulary in practice.
fn expr_key(e: &Expr) -> String {
    match e {
        Expr::Literal(Literal::Int(i, _), _) => i.to_string(),
        Expr::Literal(Literal::Float(x, _), _) => format!("{}", x),
        Expr::Literal(Literal::Str(s), _) => format!("{:?}", s),
        Expr::Literal(Literal::Bool(b), _) => b.to_string(),
        Expr::Literal(Literal::Char(c), _) => format!("c{:?}", c),
        Expr::Literal(Literal::Nil, _) | Expr::Nil(_) => "nil".to_string(),
        Expr::Ident(n, _) => n.clone(),
        Expr::BinOp { op, lhs, rhs, .. } => {
            format!("({} {:?} {})", expr_key(lhs), op, expr_key(rhs))
        }
        Expr::UnOp { op, operand, .. } => format!("({:?} {})", op, expr_key(operand)),
        Expr::Tuple(es, _) => format!("({})", es.iter().map(expr_key).collect::<Vec<_>>().join(", ")),
        // Anything richer than a dimension expression. The only caller is
        // `extern fn` signature reconciliation, and `check.rs` already confines
        // that boundary to scalars, `*T` and `nil` (`extern-boundary`,
        // SPEC §3.12), so no shape-carrying type — and therefore no expression
        // of this shape — reaches it.
        other => format!("<{:?}>", std::mem::discriminant(other)),
    }
}

fn pattern_names(p: &Pattern, out: &mut Vec<String>) {
    match p {
        Pattern::Ident(n, _) => out.push(n.clone()),
        Pattern::Tuple(ps, _) => ps.iter().for_each(|p| pattern_names(p, out)),
        Pattern::Bind(a, b, _) => {
            pattern_names(a, out);
            pattern_names(b, out);
        }
        Pattern::EnumVariant { bindings, .. } => bindings.iter().for_each(|p| pattern_names(p, out)),
        Pattern::Wildcard(_) | Pattern::Literal(..) | Pattern::Shape(..) | Pattern::Rest(_) => {}
    }
}

// ─── The uniqueness assertion (design §4.1(e)) ───────────────────────────────

/// After renaming, no two declarations may share a key in a map the JIT keys
/// on. This is what the merge buys instead of a second full check: the
/// backend's `model_layouts`, `enums`, `enum_payload_layouts` and
/// `global_consts` are bare `insert`s where the last writer wins, so a
/// collision there is wrong offsets and no diagnostic.
fn assert_unique(
    paths: &[PathBuf],
    decls: &[Vec<Decl>],
    merged_of: &[HashMap<(Ns, String), String>],
    entry_dir: &Path,
) -> Result<(), MergeError> {
    // `(key space, merged name)` → the file that first declared it, and whether
    // that declaration was an `extern fn`.
    let mut seen: HashMap<(Ns, String), (usize, bool)> = HashMap::new();
    for file_decls in decls {
        for d in file_decls {
            let Some(merged) = merged_of[d.file].get(&(d.ns, d.name.clone())) else { continue };
            let key = (d.ns, merged.clone());
            let Some(&(first, first_extern)) = seen.get(&key) else {
                seen.insert(key, (d.file, d.is_extern));
                continue;
            };
            // Two declarations in ONE file are the backend's business, not the
            // merge's: it must never be stricter than the backend it feeds on a
            // program that is already a single file.
            if first == d.file {
                continue;
            }
            // Two `extern fn`s of one name were reconciled while items were
            // emitted — identical signature, one declaration kept — or already
            // rejected there as a mismatch. An extern colliding with a
            // *non*-extern is an ordinary duplicate and falls through.
            if first_extern && d.is_extern {
                continue;
            }
            return Err(MergeError::DuplicateMerged {
                what: d.ns.what(),
                merged: merged.clone(),
                first: paths[first].clone(),
                second: paths[d.file].clone(),
                span: d.span.clone(),
                entry_dir: entry_dir.to_path_buf(),
            });
        }
    }
    Ok(())
}

// ─── The rewrite ─────────────────────────────────────────────────────────────

struct Rewriter<'a> {
    scope: &'a FileScope,
    merged: &'a HashMap<(Ns, String), String>,
    /// Lexical scopes of locally bound names. A parameter, `let`, loop or match
    /// binding shadows a top-level name, so an identifier inside one is not a
    /// reference to the declaration the merge renamed.
    shadow: Vec<HashSet<String>>,
}

impl<'a> Rewriter<'a> {
    fn shadowed(&self, name: &str) -> bool {
        self.shadow.iter().any(|s| s.contains(name))
    }

    fn push(&mut self, names: Vec<String>) {
        self.shadow.push(names.into_iter().collect());
    }

    fn pop(&mut self) {
        self.shadow.pop();
    }

    /// This file's own merged name for a declaration it declares.
    fn own(&self, ns: Ns, name: &str) -> Option<String> {
        self.merged.get(&(ns, name.to_string())).cloned()
    }

    // ── items ──

    fn item(&mut self, item: &mut Item) {
        match item {
            Item::Pub(inner) => self.item(inner),
            Item::Directive { directives, inner, .. } => {
                self.directives(directives);
                self.item(inner);
            }
            Item::Fn(f) => self.fn_decl(f, true),
            Item::ExternFn(e) => {
                // The name is the foreign symbol and is never renamed; the
                // written types still go through the rewrite so a `*T` naming a
                // merged model resolves.
                let shape: Vec<String> = e.shape_params.iter().map(|s| s.name.clone()).collect();
                self.push(shape);
                for p in &mut e.params {
                    if let Some(t) = &mut p.ty {
                        self.ty(t);
                    }
                }
                if let Some(t) = &mut e.ret_type {
                    self.ty(t);
                }
                self.pop();
            }
            Item::Model(m) => {
                if let Some(n) = self.own(Ns::Model, &m.name) {
                    m.name = n;
                }
                // A model's own shape params are not module constants — the JIT
                // is explicit that they stay opaque and are re-resolved per
                // instantiation — so they shadow inside the declaration.
                let shape: Vec<String> = m.shape_params.iter().map(|s| s.name.clone()).collect();
                self.push(shape);
                for member in &mut m.members {
                    match member {
                        ModelMember::Field { ty, .. } => self.ty(ty),
                        // A method takes no merged name of its own: the JIT
                        // registers only field layouts for a model and refuses
                        // a user method call outright, so there is no map for a
                        // method name to collide in. The day it lowers one,
                        // this is what has to change.
                        ModelMember::Method(f) => self.fn_decl(f, false),
                    }
                }
                self.pop();
            }
            Item::Enum(e) => {
                if let Some(n) = self.own(Ns::Enum, &e.name) {
                    e.name = n;
                }
                // Variants are spelled `Enum.Variant` and both JIT maps key on
                // the enum name, so no variant is renamed — only its payload
                // types, which may name a merged model.
                for v in &mut e.variants {
                    for t in &mut v.fields {
                        self.ty(t);
                    }
                }
            }
            Item::TypeAlias(t) => {
                if let Some(n) = self.own(Ns::Alias, &t.name) {
                    t.name = n;
                }
                let shape: Vec<String> = t.shape_params.iter().map(|s| s.name.clone()).collect();
                self.push(shape);
                self.ty(&mut t.ty);
                self.pop();
            }
            Item::Let(l) => {
                self.expr(&mut l.value);
                if let Some(t) = &mut l.ty {
                    self.ty(t);
                }
                self.rename_const_pattern(&mut l.pattern);
            }
            Item::Arena(a) => {
                self.push(Vec::new());
                self.block(&mut a.body);
                self.pop();
            }
            Item::Use(_) => {}
        }
    }

    /// Rename the binding names of a *top-level* `let`, which is what lands in
    /// `global_consts`.
    fn rename_const_pattern(&mut self, p: &mut Pattern) {
        match p {
            Pattern::Ident(n, _) => {
                if let Some(m) = self.own(Ns::Const, n) {
                    *n = m;
                }
            }
            Pattern::Tuple(ps, _) => ps.iter_mut().for_each(|p| self.rename_const_pattern(p)),
            Pattern::Bind(a, b, _) => {
                self.rename_const_pattern(a);
                self.rename_const_pattern(b);
            }
            _ => {}
        }
    }

    fn fn_decl(&mut self, f: &mut FnDecl, rename: bool) {
        if rename {
            if let Some(n) = self.own(Ns::Fn, &f.name) {
                f.name = n;
            }
        }
        self.directives(&mut f.directives);
        let mut bound: Vec<String> = f.shape_params.iter().map(|s| s.name.clone()).collect();
        bound.extend(f.params.iter().map(|p| p.name.clone()));
        self.push(bound);
        for p in &mut f.params {
            if let Some(t) = &mut p.ty {
                self.ty(t);
            }
        }
        if let Some(t) = &mut f.ret_type {
            self.ty(t);
        }
        self.block(&mut f.body);
        self.pop();
    }

    fn directives(&mut self, ds: &mut [Directive]) {
        for d in ds {
            for a in &mut d.args {
                match a {
                    DArg::Positional(e) => self.expr(e),
                    DArg::Named { value, .. } => self.expr(value),
                }
            }
        }
    }

    // ── blocks and statements ──

    fn block(&mut self, b: &mut Block) {
        self.push(Vec::new());
        for s in &mut b.stmts {
            self.stmt(s);
        }
        if let Some(t) = &mut b.tail_expr {
            self.expr(t);
        }
        self.pop();
    }

    /// Bind names into the *innermost* scope, which a `block` has just pushed.
    /// A `let`'s own right-hand side is rewritten before this runs, so
    /// `let helper = helper()` still calls the merged function.
    fn bind(&mut self, names: Vec<String>) {
        if let Some(top) = self.shadow.last_mut() {
            top.extend(names);
        }
    }

    fn stmt(&mut self, s: &mut Stmt) {
        match s {
            Stmt::Let(l) => {
                self.expr(&mut l.value);
                if let Some(t) = &mut l.ty {
                    self.ty(t);
                }
                self.pattern(&mut l.pattern);
                let mut names = Vec::new();
                pattern_names(&l.pattern, &mut names);
                self.bind(names);
            }
            Stmt::Expr { lhs, assign, .. } => {
                self.expr(lhs);
                if let Some((_, rhs)) = assign {
                    self.expr(rhs);
                }
            }
            Stmt::If(i) => self.if_expr(i),
            Stmt::Match(m) => self.match_expr(m),
            Stmt::For { pattern, iter, body, .. } => {
                self.expr(iter);
                self.pattern(pattern);
                let mut names = Vec::new();
                pattern_names(pattern, &mut names);
                self.push(names);
                self.block(body);
                self.pop();
            }
            Stmt::While { cond, body, .. } => {
                self.expr(cond);
                self.block(body);
            }
            Stmt::Loop { body, .. } => self.block(body),
            Stmt::Stage { body, .. } => self.expr(body),
            Stmt::Directive { directives, inner, .. } => {
                self.directives(directives);
                self.stmt(inner);
            }
            Stmt::DirectiveBlock { directives, body, .. } => {
                self.directives(directives);
                self.block(body);
            }
            Stmt::Break(_) | Stmt::Continue(_) => {}
            Stmt::Return { value, .. } => {
                if let Some(v) = value {
                    self.expr(v);
                }
            }
        }
    }

    fn if_expr(&mut self, i: &mut IfExpr) {
        self.expr(&mut i.cond);
        self.block(&mut i.then_branch);
        match &mut i.else_branch {
            Some(ElseBranch::Block(b)) => self.block(b),
            Some(ElseBranch::If(inner)) => self.if_expr(inner),
            None => {}
        }
    }

    fn match_expr(&mut self, m: &mut MatchExpr) {
        self.expr(&mut m.scrutinee);
        for arm in &mut m.arms {
            self.pattern(&mut arm.pattern);
            let mut names = Vec::new();
            pattern_names(&arm.pattern, &mut names);
            self.push(names);
            if let Some(g) = &mut arm.guard {
                self.expr(g);
            }
            self.expr(&mut arm.body);
            self.pop();
        }
    }

    /// Only the qualifier of a `Enum.Variant` pattern is a renameable name; a
    /// bare variant resolves against the scrutinee's enum and is untouched.
    fn pattern(&mut self, p: &mut Pattern) {
        match p {
            Pattern::EnumVariant { enum_name, bindings, .. } => {
                if !enum_name.is_empty() {
                    if let Some(n) = self.scope.get(&TYPE_NS, enum_name) {
                        *enum_name = n.to_string();
                    }
                }
                bindings.iter_mut().for_each(|b| self.pattern(b));
            }
            Pattern::Tuple(ps, _) => ps.iter_mut().for_each(|p| self.pattern(p)),
            Pattern::Bind(a, b, _) => {
                self.pattern(a);
                self.pattern(b);
            }
            _ => {}
        }
    }

    // ── expressions ──

    fn expr(&mut self, e: &mut Expr) {
        match e {
            Expr::Ident(name, _) => {
                if !self.shadowed(name) {
                    if let Some(n) = self.scope.get(&EXPR_NS, name) {
                        *name = n.to_string();
                    }
                }
            }
            Expr::Postfix { expr, op, span } => {
                // `alias.name` — and only that. The base identifier must be
                // bound to a module by a `use ... as`, never merely be the head
                // of a `Field`-then-`Call` chain: a cross-file UFCS call
                // (#627) has exactly that shape and is not a module reference.
                if let (Expr::Ident(base, _), PostfixOp::Field(field)) = (&**expr, &*op) {
                    if self.scope.module_aliases.contains(base) && !self.shadowed(base) {
                        let key = format!("{}.{}", base, field);
                        if let Some(n) = self.scope.get(&EXPR_NS, &key) {
                            *e = Expr::Ident(n.to_string(), span.clone());
                            return;
                        }
                    }
                }
                self.expr(expr);
                self.postfix_op(op);
            }
            Expr::StructLit { name, type_args, fields, .. } => {
                self.rename_type_name(name);
                for a in type_args {
                    self.expr(a);
                }
                for (_, v) in fields {
                    self.expr(v);
                }
            }
            Expr::Tuple(es, _) | Expr::TensorLit(es, _) => es.iter_mut().for_each(|x| self.expr(x)),
            Expr::Block(b) => self.block(b),
            Expr::If(i) => self.if_expr(i),
            Expr::Match(m) => self.match_expr(m),
            Expr::FnLit(l) => {
                let mut bound: Vec<String> = l.shape_params.iter().map(|s| s.name.clone()).collect();
                bound.extend(l.params.iter().map(|p| p.name.clone()));
                self.push(bound);
                for p in &mut l.params {
                    if let Some(t) = &mut p.ty {
                        self.ty(t);
                    }
                }
                if let Some(t) = &mut l.ret_type {
                    self.ty(t);
                }
                self.block(&mut l.body);
                self.pop();
            }
            Expr::ArenaBlock(a) => self.block(&mut a.body),
            // #674 slice 3a: structural recursion, same shape as `ArenaBlock`
            // — either sub-expression may reference a cross-file name this
            // pass needs to rewrite.
            Expr::BudgetBlock(bb) => { self.expr(&mut bb.operand); self.block(&mut bb.body); }
            Expr::AllocIn { expr, budget, .. } => { self.expr(expr); self.expr(budget); }
            Expr::DirectiveBlock { directives, body, .. } => {
                self.directives(directives);
                self.block(body);
            }
            Expr::BinOp { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            Expr::UnOp { operand, .. } => self.expr(operand),
            Expr::Cast { expr, ty, .. } => {
                self.expr(expr);
                self.ty(ty);
            }
            Expr::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.expr(s);
                }
                if let Some(en) = end {
                    self.expr(en);
                }
            }
            Expr::Literal(..) | Expr::Underscore(_) | Expr::Spread(_) | Expr::Nil(_) => {}
        }
    }

    fn postfix_op(&mut self, op: &mut PostfixOp) {
        match op {
            PostfixOp::Index(elems) => {
                for el in elems {
                    match el {
                        IndexElem::Expr(e) => self.expr(e),
                        IndexElem::Slice { start, end, step, .. } => {
                            for o in [start, end, step].into_iter().flatten() {
                                self.expr(o);
                            }
                        }
                        IndexElem::FullSlice(_) => {}
                    }
                }
            }
            PostfixOp::Call(args) | PostfixOp::BracketArgs(args) => {
                for a in args {
                    match a {
                        CallArg::Positional(e) => self.expr(e),
                        CallArg::Named { value, .. } => self.expr(value),
                        CallArg::Spread(_) => {}
                    }
                }
            }
            PostfixOp::Constructor(fields) => {
                for (_, v) in fields {
                    self.expr(v);
                }
            }
            PostfixOp::Transpose | PostfixOp::Query | PostfixOp::Field(_) => {}
        }
    }

    // ── types ──

    /// A name in type position: a model, a `type` alias (#622 made those
    /// transparent, and `resolve_alias_ty` keys on the bare name inside a
    /// `Type::Named`, so an alias name is a type position exactly like a model
    /// name), or an enum. Also reached for `alias.Model { .. }`, which the
    /// parser folds into one dotted `StructLit` name.
    fn rename_type_name(&mut self, name: &mut String) {
        if let Some((base, rest)) = name.split_once('.') {
            if self.scope.module_aliases.contains(base) {
                if let Some(n) = self.scope.get(&TYPE_NS, &format!("{}.{}", base, rest)) {
                    *name = n.to_string();
                }
                return;
            }
        }
        if let Some(n) = self.scope.get(&TYPE_NS, name) {
            *name = n.to_string();
        }
    }

    fn ty(&mut self, t: &mut Type) {
        match t {
            Type::Tensor(inner, sh, _) | Type::View(inner, sh, _) | Type::KV(inner, sh, _) => {
                self.ty(inner);
                self.shape(sh);
            }
            Type::Mesh(axes, _) => {
                for a in axes {
                    self.expr(&mut a.size);
                }
            }
            Type::Fn(ps, ret, _) => {
                ps.iter_mut().for_each(|p| self.ty(p));
                self.ty(ret);
            }
            Type::Tuple(es, _) => es.iter_mut().for_each(|e| self.ty(e)),
            Type::Array(inner, n, _) => {
                self.ty(inner);
                self.expr(n);
            }
            Type::RawPtr(inner, _) => self.ty(inner),
            Type::Named { name, args, .. } => {
                self.rename_type_name(name);
                for a in args {
                    match a {
                        TypeArg::Type(t) => self.ty(t),
                        TypeArg::Expr(e) => self.expr(e),
                        TypeArg::Named { value, .. } => self.expr(value),
                    }
                }
            }
            Type::Scalar(..) => {}
        }
    }

    fn shape(&mut self, sh: &mut ShapeSpec) {
        for el in &mut sh.elems {
            if let ShapeElem::Expr(e) = el {
                self.expr(e);
            }
        }
    }
}
