/// demoniC compiler entry point — bootstrap phase
///
/// Phase 1: Lexer ✓
/// Phase 2: Parser / AST ✓
/// Phase 3: Type-checker ✓ (pre-alpha)
/// Phase 3.5: Tree-walking interpreter ✓ (pre-alpha)
/// Phase 4: Metal backend (MLX where applicable)

mod lexer;
mod ast;
// #485: `--json` structured diagnostics, schema 1.
mod diag;
mod parser;
mod desugar;
// #505: `@comptime` folding — the reference interpreter at compile time.
// Runs after parse and before check, so both backends lower the same tree.
mod comptime;
mod shape;
mod types;
mod check;
mod interp;
mod ports;
mod fmt;
mod resolver;
// #15: fold a resolved import DAG into
// one `Program` the JIT can compile as a single unit. JIT-only — the
// interpreter keeps per-module semantics (#652) and never sees a merged name.
mod merge;
mod jit;
mod selftest;
mod assimilate;
// #400: arena sizing flags (`--vault=` / `--forge=`), shared by both backends.
mod arena;
// #463: `demoni.json` reader — advisory lint dials only, never semantics.
mod manifest;
// #326 GPU/Metal backend — compiled only on macOS with `--features gpu`.
#[cfg(all(target_os = "macos", feature = "gpu"))]
mod gpu;
#[cfg(test)]
#[path = "arena_tests.rs"]
mod arena_tests;
#[cfg(test)]
#[path = "lexer_tests.rs"]
mod lexer_tests;
#[cfg(test)]
#[path = "parser_tests.rs"]
mod parser_tests;
#[cfg(test)]
#[path = "check_tests.rs"]
mod check_tests;
#[cfg(test)]
#[path = "comptime_tests.rs"]
mod comptime_tests;
#[cfg(test)]
#[path = "interp_tests.rs"]
mod interp_tests;
#[cfg(test)]
#[path = "fmt_tests.rs"]
mod fmt_tests;
#[cfg(test)]
#[path = "diag_tests.rs"]
mod diag_tests;
#[cfg(test)]
#[path = "manifest_tests.rs"]
mod manifest_tests;
#[cfg(test)]
#[path = "usage_tests.rs"]
mod usage_tests;
#[cfg(test)]
#[path = "merge_tests.rs"]
mod merge_tests;

use std::path::{Path, PathBuf};
use lexer::{Lexer, TokenKind};
use parser::Parser;
use check::Checker;
use interp::Interpreter;
use fmt::pretty_print_program;
use jit::Jit;

/// The usage banner, one entry per line.
///
/// A `const` rather than a run of `eprintln!`s so `usage_tests.rs` can hold
/// exactly what the user sees against the flags the argument parser actually
/// accepts. The banner had drifted: `--gpu` and `--blas` were parsed, wired
/// and named in the README while `dmc` with no arguments never mentioned
/// them (#534 — a rule whose behavior was never demonstrated is not a rule).
const USAGE_LINES: &[&str] = &[
    "usage:",
    "  dmc <file.dmc>                    run the full pipeline (lex + parse + check + execute)",
    "  dmc --lex <file.dmc>              dump the token stream",
    "  dmc --parse <file.dmc>            dump the AST tree",
    "  dmc --check <file.dmc>            type-check, report diagnostics",
    "  dmc run <file.dmc>                execute the program (tree-walking interpreter)",
    "  dmc jit <file.dmc>                JIT (Cranelift, scalars + control flow)",
    "  dmc test <file-or-dir>            run zero-arg test_* functions (interpreter)",
    "  dmc test --jit <file-or-dir>      also run each test_* under the JIT (parity gate; skips files outside the JIT subset)",
    "  dmc test --fail-empty <file-or-dir>",
    "                                    exit 1 (instead of 0) when zero tests ran across the given path",
    "  dmc selftest [opts]               interp-vs-JIT differential fuzzer over generated scalar programs",
    "                                    opts: --iters N --seed S --repro SEED --floats / --no-floats --meta-test --verbose --timeout SECS",
    "  dmc fmt <file.dmc>                pretty-print with canonical formatting (to stdout)",
    "  dmc assimilate <desc.json> [-o f] generate port-wrapper bindings from a descriptor (ASSIMILATE.md)",
    "  dmc assimilate python:<module>    introspect a live runtime into a draft descriptor (add --bindings for wrappers)",
    "  dmc --profile <file.dmc>          run with op-count profiling (emits summary to stderr)",
    "  dmc --profile run <file.dmc>      run mode with profiling",
    "  dmc --demon <file.dmc>            demon mode: release the safe-mode lints (raw, no guardrails)",
    "  dmc jit --gpu <file.dmc>          experimental Metal backend (`dmc jit` only); honored only on a macOS",
    "                                    `--features gpu` build, ignored (`gpu-unavailable`) anywhere else.",
    "  dmc jit --blas <file.dmc>         route a large f32 matmul to the host BLAS `cblas_sgemm` (`dmc jit`",
    "                                    only). Opt-in: BLAS accumulates in its own blocked order, so the",
    "                                    last bits can differ from the Cranelift kernel. Ignored",
    "                                    (`blas-unavailable`) where `cblas_sgemm` does not resolve in this",
    "                                    process, and never selected inside `@deterministic`.",
    "  dmc run <file.dmc> --vault=16G --forge=2G",
    "                                    arena byte budgets (MEMORY.md §1.1); B/K/M/G/T suffixes,",
    "                                    exhausting one is a runtime error. `--vault` is `dmc run` only.",
    "  dmc --check --json <file.dmc>     diagnostics as JSON Lines on stderr (schema 1)",
    "  dmc jit --json <file.dmc>         same, including machine-coded JIT-ineligibility refusals",
    "  dmc test --json <file-or-dir>     same, streaming one `test` object per test (with `--jit`, the parity verdict rides along)",
    "  dmc jit --json --all <file.dmc>   report every refusal in the program instead of stopping at the first",
    "  dmc test --jit --all <file-or-dir>",
    "                                    per-file blocker-depth count (`| blocker depth: F files, D blockers`); JSON adds it per test too",
];

fn print_usage() {
    for line in USAGE_LINES {
        eprintln!("{}", line);
    }
}

/// A checker diagnostic in `--json` form. The `code` is the kebab-case tag the
/// message leads with where the docs define one (`port-forbidden`, …); a
/// message with no tag simply has no `code`.
fn check_diag(
    e: &check::TypeError,
    file: &Path,
    severity: diag::Severity,
) -> diag::Diagnostic {
    let mut d = diag::Diagnostic::new(diag::Kind::Check, severity, e.msg.clone())
        .file(file)
        .at(e.span.line, e.span.col)
        .bytes(e.span.start, e.span.end)
        .hint(e.hint.as_deref());
    if let Some(tag) = diag::tag_of(&e.msg) {
        d = d.code(tag);
    }
    // A shape error carries the two shapes as data (#485): `expected`/`actual`
    // arrays of dims, so a consumer diffs them instead of re-parsing `message`.
    if let Some((exp, act)) = &e.shapes {
        d = d.shapes(&exp.dims, &act.dims);
    }
    d
}

/// #505: fold every `@comptime` block in a resolved program set, before any
/// checker runs.
///
/// The pass rewrites the AST in place — a closed block becomes an integer or
/// boolean literal — so every consumer downstream of here, both backends
/// included, lowers the same tree. Its diagnostics are `check::TypeError`s,
/// returned per file and seeded into that file's `Checker` before
/// `check_program` runs, which is why the human renderer, the exit code and
/// the `--json` stream need no case for them: they are checker errors by the
/// time anything reports them.
///
/// `dmc fmt` deliberately does not call this, so `@comptime` source
/// round-trips through the formatter unfolded.
fn fold_comptime_all(
    r: &mut resolver::Resolver,
) -> std::collections::HashMap<std::path::PathBuf, Vec<check::TypeError>> {
    let mut out = std::collections::HashMap::new();
    for p in r.sorted_paths.clone() {
        if let Some(prog) = r.files.get_mut(&p) {
            let errs = comptime::fold_program(prog);
            if !errs.is_empty() {
                out.insert(p, errs);
            }
        }
    }
    out
}

/// One module of a resolved program, after the checker has run on it.
///
/// Kept per file rather than merged, so a caller attributes a diagnostic to the
/// module it came from and not to the file named on the command line.
struct CheckedModule {
    path: std::path::PathBuf,
    errors: Vec<check::TypeError>,
    warnings: Vec<check::TypeError>,
    items: usize,
}

/// Type-check every module of an already-resolved, already-folded import DAG,
/// leaf-first, threading each module's public environment into the files that
/// import it (`SPEC.md §6.7`: only `pub` items are exported, and files are
/// checked leaf-first over the import DAG).
///
/// That threading is the whole of what makes a `use` check: `check.rs` reports
/// `imported module not found / checked yet` for any import whose `ModuleEnv`
/// is absent from `checked_modules` when the importing file is checked, and
/// `Resolver::sorted_paths` is the topological order that guarantees it is
/// present.
fn check_resolved(
    r: &resolver::Resolver,
    folded: &std::collections::HashMap<std::path::PathBuf, Vec<check::TypeError>>,
    demon: bool,
) -> Vec<CheckedModule> {
    let mut checked_modules = std::collections::HashMap::new();
    let mut out = Vec::with_capacity(r.sorted_paths.len());
    for p in &r.sorted_paths {
        let prog = r.files.get(p).unwrap();
        let mut checker = Checker::new();
        checker.errors = folded.get(p).cloned().unwrap_or_default();
        checker.demon = demon;
        checker.checked_modules = checked_modules.clone();
        checker.check_program(prog, Some(p));
        let mod_env = check::ModuleEnv {
            env: checker.env.clone(),
            aliases: checker.aliases.clone(),
            resolved_aliases: checker.resolved_aliases.clone(),
            public_items: ast::collect_public_items(prog),
            top_level_let_names: ast::collect_top_level_let_names(prog),
            transitive_import_lets: checker.import_private_lets.clone(),
        };
        checked_modules.insert(p.clone(), mod_env);
        out.push(CheckedModule {
            path: p.clone(),
            errors: checker.errors,
            warnings: checker.warnings,
            items: prog.items.len(),
        });
    }
    out
}

/// Resolve the import DAG rooted at `path`, fold `@comptime` across all of it,
/// and type-check it. The single front end `dmc --check`, `dmc run` and
/// `dmc jit` share.
///
/// `dmc jit` used to lex and parse only the file named on the command
/// line and hand that one `Program` to a `Checker` with an empty module table.
/// A program with a `use` therefore died at `imported module not found /
/// checked yet` — a `check` diagnostic, and a false one, since `--check` and
/// `run` resolved the very same import — and the JIT's own `jit-import`
/// refusal (`jit.rs`, the `Item::Use` arm of `collect_fns`) was unreachable:
/// nothing ever got as far as asking the backend whether it could lower the
/// program. Sharing this path means a multi-file program is refused by the
/// backend, with the `use`'s span and the `jit-import` class, or lowered —
/// never rejected by an artifact of which subcommand was typed.
fn resolve_and_check(
    path: &Path,
    demon: bool,
) -> Result<(resolver::Resolver, Vec<CheckedModule>), resolver::ResolveError> {
    let mut r = resolver::Resolver::new();
    r.resolve_all(path)?;
    let folded = fold_comptime_all(&mut r);
    let modules = check_resolved(&r, &folded, demon);
    Ok((r, modules))
}

/// A `JitError` in `--json` form. A refusal (`JitErrorKind::Unsupported`) is
/// `jit-ineligible` and always carries its class code (#485); a defect is
/// `jit-error` and carries none — that distinction is #480's, and this is the
/// encoding of it that does not go through English.
fn jit_diag(e: &jit::JitError, file: &Path) -> diag::Diagnostic {
    let kind = match e.kind {
        jit::JitErrorKind::Unsupported => diag::Kind::JitIneligible,
        jit::JitErrorKind::Error => diag::Kind::JitError,
    };
    let mut d = diag::Diagnostic::new(kind, diag::Severity::Error, e.msg.clone()).file(file);
    // `line: 0` is the JIT's "no source location" sentinel (module-level
    // failures, `no fn main`). Omit the location rather than claim line 0.
    if e.line != 0 {
        d = d.at(e.line, e.col);
    }
    if let Some(r) = e.refusal {
        d = d.code(r.code());
    }
    d
}

/// A failed whole-program merge in `--json` form.
///
/// A `MergeError` is the JIT declining to lower a *multi-file* program — two
/// files whose declarations cannot be given one name apiece in the single flat
/// symbol table the backend has. That is exactly what the `jit-import` class
/// has always meant, "a `use` the JIT could not lower", so the merge's failures
/// are reported under it rather than under a new code: `DIAGNOSTICS.md §6`
/// makes refusal codes append-only, and this is the same fact arriving one pass
/// later. It is `jit-ineligible`, never `jit-error` — the program is outside
/// the lowered subset, and `dmc run` has the full semantics for it.
///
/// The span is the `use` that pulled the second declaration in, or the
/// declaration that could not be reconciled; either way it is measured in the
/// file `MergeError::file` names, which is why the two are read together.
///
/// Of `MergeError`'s three non-`EntryNotResolved` variants (`EntryNotResolved`
/// itself cannot reach here either — `resolve_and_check`, called just above in
/// the `jit` arm, is what populates the resolver `merge_program` reads),
/// `ExternMismatch` is the only one a real program can actually surface
/// through this function. `Ambiguous` is the shape `check.rs` independently
/// rejects as `ambiguous import` — and `resolve_and_check` runs before
/// `merge_program` on every path that calls `merge_diag` — so a program that
/// would trigger it never gets this far. `DuplicateMerged` is design §4.1(e)'s
/// backstop assertion, not a diagnosable program shape: if the rename rule
/// above it is correct, no merge ever reaches it. Both stay reachable from
/// `merge_program` directly (see `merge_tests.rs`, which calls it unchecked)
/// precisely so that assertion has a test.
fn merge_diag(e: &merge::MergeError, fallback: &Path) -> diag::Diagnostic {
    let mut d = diag::Diagnostic::new(
        diag::Kind::JitIneligible,
        diag::Severity::Error,
        format!("{}; use `dmc run` for full semantics", e),
    )
    .code(jit::Refusal::Import.code())
    .file(e.file().unwrap_or(fallback));
    if let Some(s) = e.span() {
        d = d.at(s.line, s.col).bytes(s.start, s.end);
    }
    d
}

/// A module-resolution failure in `--json` form, shared by `--check`, `run`
/// and `jit` alike. `Lex` and `Parse` carry a span and use the same `lex` /
/// `parse` kinds a diagnostic from lexing or parsing the entry file directly
/// would use. A `use` that cannot be resolved, or that closes a cycle, is
/// `resolve`, located at the `use` in the importing file (with its byte span)
/// and classed by its `code` (`DIAGNOSTICS.md §5.4`) — the class comes from
/// the variant, not from the prose, so the human sentence is untouched. Only
/// a per-file I/O failure under `dmc test`, with no importing `use` to blame
/// it on, has no span and stays `unstructured`; an I/O failure on the actual
/// entry file is caught earlier, in `main`, before the resolver ever runs,
/// and is `cli`.
fn resolve_diag(e: &resolver::ResolveError) -> diag::Diagnostic {
    match e {
        resolver::ResolveError::Lex { file, msg, line, col } => {
            diag::Diagnostic::new(diag::Kind::Lex, diag::Severity::Error, msg.clone())
                .file(file)
                .at(*line, *col)
        }
        resolver::ResolveError::Parse { file, msg, line, col } => {
            diag::Diagnostic::new(diag::Kind::Parse, diag::Severity::Error, msg.clone())
                .file(file)
                .at(*line, *col)
        }
        resolver::ResolveError::Import { file, msg, line, col, start, end } => {
            diag::Diagnostic::new(diag::Kind::Resolve, diag::Severity::Error, msg.clone())
                .code("import-unresolved")
                .file(file)
                .at(*line, *col)
                .bytes(*start, *end)
        }
        resolver::ResolveError::Cycle { file, msg, line, col, start, end } => {
            diag::Diagnostic::new(diag::Kind::Resolve, diag::Severity::Error, msg.clone())
                .code("import-cycle")
                .file(file)
                .at(*line, *col)
                .bytes(*start, *end)
        }
        resolver::ResolveError::Other(s) => diag::Diagnostic::unstructured(s.clone()),
    }
}

/// Report a diagnostic whose category schema 1 does not model yet — a JIT
/// runtime trap — and exit 1. Without `--json` this is the
/// unchanged human line.
fn fail_unstructured(em: &mut Option<diag::Emitter>, msg: impl std::fmt::Display) -> ! {
    match em.take() {
        Some(mut e) => {
            e.emit(&diag::Diagnostic::unstructured(msg.to_string()));
            std::process::exit(e.finish(1));
        }
        None => {
            eprintln!("{}", msg);
            std::process::exit(1);
        }
    }
}

/// Report a command-line usage error and exit 1.
fn fail_cli(em: &mut Option<diag::Emitter>, code: &str, msg: impl std::fmt::Display) -> ! {
    match em.take() {
        Some(mut e) => {
            let d = diag::Diagnostic::new(
                diag::Kind::Cli,
                diag::Severity::Error,
                msg.to_string(),
            )
            .code(code);
            e.emit(&d);
            std::process::exit(e.finish(1));
        }
        None => {
            eprintln!("{}", msg);
            std::process::exit(1);
        }
    }
}

fn print_profile(p: &interp::OpProfile) {
    eprintln!("── demoniC profile ──────────────────────────────");
    eprintln!("tensor ops       : {:>11}", p.tensor_ops);
    eprintln!("tensor elements  : {:>11}", p.tensor_elements);
    eprintln!("scalar ops       : {:>11}", p.scalar_ops);
    eprintln!("fn calls         : {:>11}", p.fn_calls);
    eprintln!("allocs           : {:>11}", p.allocs);
    eprintln!("─────────────────────────────────────────────────");
}

/// Print non-fatal lint diagnostics. Unlike type errors these never block
/// execution; they nudge the author toward correct code.
fn print_warnings(warnings: &[check::TypeError]) {
    for w in warnings {
        eprint!("warning at {}:{}: {}", w.span.line, w.span.col, w.msg);
        if let Some(h) = &w.hint {
            eprint!("\n  hint: {}", h);
        }
        eprintln!();
    }
}

/// `dmc assimilate <descriptor.json> [-o <out.dmc>]` — generate the port-wrapper
/// module and write it to stdout (or `-o` file). Returns a process exit code.
fn run_assimilate(args: &[String]) -> i32 {
    // Target is either a descriptor file, or a `runtime:module` introspection
    // target (e.g. `python:math`) that assimilate reads from the live runtime.
    let mut target: Option<&str> = None;
    let mut out_path: Option<&str> = None;
    let mut to_bindings = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                match args.get(i) {
                    Some(p) => out_path = Some(p),
                    None => { eprintln!("assimilate: -o needs a file path"); return 1; }
                }
            }
            // For an introspection target, emit wrappers instead of the draft
            // descriptor (fns whose types were all inferred; the rest reported).
            "--bindings" => to_bindings = true,
            other if target.is_none() => target = Some(other),
            other => { eprintln!("assimilate: unexpected argument `{}`", other); return 1; }
        }
        i += 1;
    }
    let Some(target) = target else {
        eprintln!("assimilate: missing target (usage: dmc assimilate <desc.json | python:module> [--bindings] [-o out])");
        return 1;
    };

    // `runtime:module` (an introspection target) vs. a descriptor path. A target
    // whose scheme is a bare identifier and which is not an existing file is an
    // introspection target — so `lua:math` reaches introspect()'s "only wired
    // for python" diagnostic instead of a misleading "no such file".
    let is_scheme = |t: &str| matches!(t.split_once(':'),
        Some((rt, _)) if !rt.is_empty()
            && rt.chars().next().map(|c| c.is_ascii_alphabetic() || c == '_').unwrap_or(false)
            && rt.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
    let introspected: Option<String> = if is_scheme(target) && !std::path::Path::new(target).exists() {
        let (rt, module) = target.split_once(':').unwrap();
        match assimilate::introspect(rt, module) {
            Ok(desc) => Some(desc),
            Err(e) => { eprintln!("assimilate: {}", e); return 1; }
        }
    } else {
        None
    };

    let output = match introspected {
        // Introspection: default output is the reviewable draft descriptor;
        // --bindings pipes it straight through the generator.
        Some(desc) if to_bindings => match assimilate::generate(&desc) {
            Ok(m) => m,
            Err(e) => { eprintln!("assimilate: {}", e); return 1; }
        },
        Some(desc) => desc + "\n",
        // A descriptor file → wrappers.
        None => {
            if to_bindings {
                eprintln!("assimilate: --bindings applies to an introspection target; a \
                           descriptor file already generates bindings");
                return 1;
            }
            let src = match std::fs::read_to_string(target) {
                Ok(s) => s,
                Err(e) => { eprintln!("assimilate: error reading {}: {}", target, e); return 1; }
            };
            match assimilate::generate(&src) {
                Ok(m) => m,
                Err(e) => { eprintln!("assimilate: {}", e); return 1; }
            }
        }
    };

    match out_path {
        Some(p) => match std::fs::write(p, &output) {
            Ok(()) => 0,
            Err(e) => { eprintln!("assimilate: error writing {}: {}", p, e); 1 }
        },
        None => { print!("{}", output); 0 }
    }
}

/// Parse `dmc selftest` flags and run the differential fuzzer (#408).
fn run_selftest(args: &[String]) -> i32 {
    fn val(args: &[String], i: &mut usize, flag: &str) -> String {
        *i += 1;
        if *i >= args.len() {
            eprintln!("selftest: {} expects a value", flag);
            std::process::exit(1);
        }
        args[*i].clone()
    }
    fn bad<T>(flag: &str, v: &str) -> T {
        eprintln!("selftest: bad {} {:?}", flag, v);
        std::process::exit(1);
    }

    let mut cfg = selftest::Config::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                let v = val(args, &mut i, "--iters");
                cfg.iters = v.parse().unwrap_or_else(|_| bad("--iters", &v));
            }
            "--seed" => {
                let v = val(args, &mut i, "--seed");
                cfg.seed = v.parse().unwrap_or_else(|_| bad("--seed", &v));
            }
            "--repro" => {
                let v = val(args, &mut i, "--repro");
                cfg.repro = Some(v.parse().unwrap_or_else(|_| bad("--repro", &v)));
            }
            "--timeout" => {
                let v = val(args, &mut i, "--timeout");
                let secs: f64 = v.parse().unwrap_or_else(|_| bad("--timeout", &v));
                cfg.timeout = std::time::Duration::from_secs_f64(secs);
            }
            "--floats" => cfg.floats = true,
            "--no-floats" => cfg.floats = false,
            "--meta-test" => cfg.meta_test = true,
            "--verbose" => cfg.verbose = true,
            other => {
                eprintln!("selftest: unknown option {:?}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    selftest::run(&cfg)
}

fn collect_test_fns_from_item(item: &ast::Item, out: &mut Vec<String>) {
    match item {
        ast::Item::Fn(f) if f.name.starts_with("test_") => out.push(f.name.clone()),
        ast::Item::Directive { inner, .. } => collect_test_fns_from_item(inner, out),
        ast::Item::Pub(inner) => collect_test_fns_from_item(inner, out),
        _ => {}
    }
}

fn find_fn_decl<'a>(item: &'a ast::Item, name: &str) -> Option<&'a ast::FnDecl> {
    match item {
        ast::Item::Fn(f) if f.name == name => Some(f),
        ast::Item::Directive { inner, .. } => find_fn_decl(inner, name),
        ast::Item::Pub(inner) => find_fn_decl(inner, name),
        _ => None,
    }
}

fn collect_test_fns(program: &ast::Program) -> Vec<String> {
    let mut tests = Vec::new();
    for item in &program.items {
        collect_test_fns_from_item(item, &mut tests);
    }
    tests
}

fn collect_dmc_files(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("dmc") {
            out.push(path.to_path_buf());
        }
        return Ok(());
    }
    if path.is_dir() {
        let entries = std::fs::read_dir(path)
            .map_err(|e| format!("error reading directory {:?}: {}", path, e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("error reading directory entry: {}", e))?;
            collect_dmc_files(&entry.path(), out)?;
        }
        return Ok(());
    }
    Err(format!("test path does not exist: {:?}", path))
}

fn run_tests(
    path: &Path,
    profile_mode: bool,
    jit_test: bool,
    fail_empty: bool,
    all_mode: bool,
    arena_limits: arena::ArenaLimits,
    mut em: Option<diag::Emitter>,
) -> i32 {
    // A fatal before any test ran: one diagnostic, then the envelope. Without
    // `--json`, the unchanged human line.
    fn fatal(em: Option<diag::Emitter>, msg: String) -> i32 {
        match em {
            Some(mut e) => {
                e.emit(&diag::Diagnostic::unstructured(msg));
                e.finish(1)
            }
            None => {
                eprintln!("{}", msg);
                1
            }
        }
    }
    // A per-file failure (`FAIL <file>: …`) — not a *test* result, so it has no
    // `test` object; under `--json` the prose is carried as `unstructured` (a
    // located resolution failure goes through `resolve_diag` instead), and it
    // still counts into the summary's `failed`, exactly as the human tally.
    fn fail_line(em: &mut Option<diag::Emitter>, msg: String) {
        match em {
            Some(e) => e.emit(&diag::Diagnostic::unstructured(msg)),
            None => eprintln!("{}", msg),
        }
    }

    let mut files = Vec::new();
    if let Err(e) = collect_dmc_files(path, &mut files) {
        return fatal(em, e);
    }
    files.sort();
    if files.is_empty() {
        return fatal(em, format!("no .dmc files found under {:?}", path));
    }

    let mut total = 0usize;
    let mut failed = 0usize;
    // #854: a failure before any test in the file is even collected — a
    // canonicalization/parse/resolve/dependency-check error — is not a test,
    // so it never touches `total`. It used to share `failed` with the
    // per-test tally below, and since the summary's "passed" is derived as
    // `total - failed`, one such file-level failure silently zeroed out every
    // `ok` a *different*, successfully-resolved file in the same run had
    // printed. Kept separate here and folded back in only for the printed/
    // exit-code "failed" count, never for the subtraction that yields
    // "passed".
    let mut file_failures = 0usize;
    // #: `--jit` parity counters. Each test_* is additionally run under the JIT;
    // a file the JIT can't compile is skipped (outside its subset), not failed.
    let mut jit_ran = 0usize;
    let mut jit_skipped = 0usize;
    // #684 R2: `--all`'s blocker-depth aggregate — how many files have at
    // least one recorded root blocker, and the total DISTINCT root-blocker
    // count across every measured file (not the raw refusal-site count —
    // `jit::BlockerCounts::blockers`, not `::sites`). `--jit`-only; stays `0`
    // (and unprinted) otherwise. Kept the name `blocker_sites` from before R2
    // split the metric in three, to hold the diff small; it now sums
    // `jit_blockers`.
    let mut blocker_files = 0usize;
    let mut blocker_sites = 0usize;
    // #684 rework 3: how many measured files had `truncated:true` (a root
    // `jit-error` cut collection short) — feeds the human line's appended
    // "(N truncated — floor)", omitted entirely when this stays `0`.
    let mut blocker_truncated_files = 0usize;
    // #400: a `--forge` budget must mean the same thing here as under
    // `dmc run` — the whole budget, per test. The interpreter gets that for
    // free (a fresh `Interpreter`, hence a fresh meter, is built below for
    // every test); the JIT's arena is a thread-local that outlives the whole
    // suite, so it is explicitly reset before each JIT run. And exhaustion
    // must be reportable: the arena's default is to exit(1) from the
    // allocation callback, which would kill the harness mid-run with no FAIL
    // line, no summary, and every later file unrun.
    if jit_test {
        jit::set_forge_exhaustion_recoverable(true);
    }
    for file in files {
        let canonical_file = match file.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                file_failures += 1;
                fail_line(&mut em, format!("FAIL {}: canonicalization failed: {}", file.display(), e));
                continue;
            }
        };
        let mut resolver = resolver::Resolver::new();
        if let Err(e) = resolver.resolve_all(&canonical_file) {
            file_failures += 1;
            // Under `--json` a located failure is the same `lex` / `parse` /
            // `resolve` object `--check` would emit for this file — it carries
            // the file itself, so the `FAIL <file>:` prefix has nothing to add.
            // The span-less remainder keeps the prefixed line, where the file
            // name is only in the prose.
            match &mut em {
                Some(emj) if e.is_located() => emj.emit(&resolve_diag(&e)),
                _ => fail_line(&mut em, format!("FAIL {}: resolution failed: {}", file.display(), e)),
            }
            continue;
        }

        let folded = fold_comptime_all(&mut resolver);

        let program = match resolver.files.get(&canonical_file) {
            Some(prog) => prog.clone(),
            None => {
                file_failures += 1;
                fail_line(&mut em, format!("FAIL {}: program not found in resolved files", file.display()));
                continue;
            }
        };
        let tests = collect_test_fns(&program);
        if tests.is_empty() {
            continue;
        }

        // The same leaf-first check `dmc --check` and `dmc jit` run. `demon` is
        // off here: the harness has never enabled safe mode for the files it
        // runs, and turning it on would be a change to what `dmc test` accepts,
        // not to how it resolves.
        //
        // `check_resolved` walks the whole DAG unconditionally, inserting even
        // a failing module's `ModuleEnv` for the files that import it — unlike
        // the inline loop this replaced, which stopped at the first failing
        // module and never built or inserted an env past that point. Output is
        // unchanged: the loop below still `break`s on the first module with
        // errors, and since `sorted_paths` is leaf-first that is the same
        // root-cause module the old code reported, so a downstream importer's
        // errors — even ones seeded by a failing module's env — are never
        // reached. The only cost is checking modules past the first failure,
        // work the old code skipped and this one now discards after the fact.
        let mut check_failed = false;
        for m in &check_resolved(&resolver, &folded, false) {
            if m.errors.is_empty() {
                continue;
            }
            file_failures += 1;
            fail_line(&mut em, format!(
                "FAIL {}: type check failed in dependency {:?}", file.display(), m.path));
            match &mut em {
                // The checker errors themselves are `check` diagnostics with
                // spans — the same encoding `dmc --check` gives them.
                Some(e) => {
                    for err in &m.errors {
                        e.emit(&check_diag(err, &m.path, diag::Severity::Error));
                    }
                }
                None => {
                    for err in &m.errors {
                        eprintln!("{}", err);
                    }
                }
            }
            check_failed = true;
            break;
        }
        if check_failed {
            continue;
        }

        // The whole-program merge, the second of the two `compile_program`
        // call sites: each entry file merges its own import closure, once per
        // file rather than once per test — the merge is a function of the file
        // set, not of which `test_*` is being run.
        //
        // A merge that cannot be built is scored exactly as a `compile_program`
        // refusal already is: the file is outside the JIT's subset, so every
        // test in it is `skip`. That is the verdict the merge design leaves
        // alone — what changes is only *which* files stop skipping, and a
        // multi-file entry that lowers now counts as ran.
        let merged = if jit_test {
            Some(merge::merge_program(&resolver, &canonical_file).map(std::sync::Arc::new))
        } else {
            None
        };

        // #684: the per-file blocker-depth counts, via collect mode — a
        // property of the file (the whole program it merges into), computed
        // once, not once per `test_*` the way `jit_verdict` below is.
        //
        // `None` outside `--jit --all`. `Some(None)` (R3) when the merge
        // built but the JIT engine itself failed to initialize for this file
        // — distinct from `Some(Some(counts))` with `sites: 0`, a file that
        // is genuinely fully inside the lowered subset; the old code
        // collapsed the two onto the same `0`. The merge itself refusing is
        // not that case — it is already the file's one `jit-import`-class
        // root blocker, so it is `Some(Some(..))` with one site, like any
        // other collected root, and collect mode has nothing further to walk.
        let file_blockers: Option<Option<jit::BlockerCounts>> =
            if jit_test && all_mode {
                match merged.as_ref().and_then(|m| m.as_ref().ok()) {
                    Some(m) => {
                        jit::reset_forge_arena();
                        match jit::Jit::new() {
                            Ok(mut j) => {
                                j.set_merge(m.clone());
                                let (refusals, truncated) = j.compile_program_collect(&m.program);
                                Some(Some(jit::blocker_counts(&refusals, truncated)))
                            }
                            Err(_) => Some(None),
                        }
                    }
                    // The merge itself refused — a real, complete root
                    // blocker (`jit-import`), not a mid-collection abort:
                    // collect mode never even started, so there is nothing
                    // past it to have truncated.
                    None => Some(Some(jit::BlockerCounts {
                        sites: 1, blockers: 1, cascades: 0,
                        classes: std::collections::BTreeMap::from([("jit-import", 1u64)]),
                        truncated: false,
                    })),
                }
            } else {
                None
            };
        if let Some(Some(counts)) = &file_blockers {
            if counts.blockers > 0 {
                blocker_files += 1;
                blocker_sites += counts.blockers;
            }
            // `truncated` implies at least one recorded root blocker (the
            // aborting `jit-error` itself), so this never fires for a file
            // that skipped the `blocker_files`/`blocker_sites` tally above.
            if counts.truncated {
                blocker_truncated_files += 1;
            }
        }
        let file_jit_classes = file_blockers.as_ref()
            .and_then(|f| f.as_ref())
            .map(|c| diag::classes_object(&c.classes));
        let file_jit_blockers: Option<Option<usize>> =
            file_blockers.as_ref().map(|f| f.as_ref().map(|c| c.blockers));
        let file_jit_sites: Option<Option<usize>> =
            file_blockers.as_ref().map(|f| f.as_ref().map(|c| c.sites));
        let file_jit_cascades: Option<Option<usize>> =
            file_blockers.as_ref().map(|f| f.as_ref().map(|c| c.cascades));
        // #684 fixup (F3): whether a root `jit-error` aborted collection
        // before every item was walked, making the three counts above a
        // FLOOR rather than the file's true depth. Same `Some(None)`-is-
        // `null` tri-state as `jit_blockers` et al. (R3) — a file whose JIT
        // engine failed to initialize has no truncation verdict either.
        let file_jit_truncated: Option<Option<bool>> =
            file_blockers.as_ref().map(|f| f.as_ref().map(|c| c.truncated));

        for name in tests {
            total += 1;
            let label = format!("{}::{}", file.display(), name);
            // Under `--json`, one `test` object per test — the interpreter
            // verdict and (with `--jit`) the parity verdict on the same line,
            // emitted after both halves have run. `mk_test` binds the identity
            // once so every exit path below reports the same name and file.
            let mk_test = |pass: bool, message: Option<String>| diag::TestResult {
                name: name.clone(),
                file: canonical_file.display().to_string(),
                pass,
                message,
                jit: None,
                jit_message: None,
                jit_blockers: file_jit_blockers,
                jit_sites: file_jit_sites,
                jit_cascades: file_jit_cascades,
                jit_classes: file_jit_classes.clone(),
                jit_truncated: file_jit_truncated,
            };
            let fn_decl = program.items.iter().find_map(|item| find_fn_decl(item, &name));
            if let Some(f) = fn_decl {
                if !f.params.is_empty() {
                    failed += 1;
                    match &mut em {
                        Some(e) => e.emit_test(&mk_test(
                            false, Some("test function must take zero args".to_string()),
                        )),
                        None => eprintln!("FAIL {}: test function must take zero args", label),
                    }
                    continue;
                }
            }

            let mut interp = Interpreter::new();
            interp.set_arena_limits(arena_limits);
            if profile_mode { interp.enable_profile(); }

            let mut interp_failed = false;
            for p in &resolver.sorted_paths {
                let prog = resolver.files.get(p).unwrap();
                if let Err(e) = interp.load_program(prog, Some(p)) {
                    failed += 1;
                    let msg = format!("interpreter load failed in dependency {:?}: {}", p, e);
                    match &mut em {
                        Some(emj) => emj.emit_test(&mk_test(false, Some(msg))),
                        None => eprintln!("FAIL {}: {}", file.display(), msg),
                    }
                    interp_failed = true;
                    break;
                }
                let mod_env = interp.get_module_env();
                interp.interp_modules.insert(p.clone(), mod_env);
            }
            if interp_failed {
                continue;
            }

            let result = interp.call_named_fn(&name, Vec::new());
            let (pass, message): (bool, Option<String>) = match result {
                Ok(interp::Value::Bool(false)) => {
                    failed += 1;
                    if em.is_none() { eprintln!("FAIL {}: returned false", label); }
                    (false, Some("returned false".to_string()))
                }
                Ok(interp::Value::Bool(true)) | Ok(interp::Value::Nil) => {
                    if em.is_none() {
                        println!("ok   {}", label);
                        if profile_mode {
                            if let Some(p) = &interp.profile { print_profile(p); }
                        }
                    }
                    (true, None)
                }
                Ok(v) => {
                    failed += 1;
                    let msg = format!("returned {:?}; tests must return bool or nil", v);
                    if em.is_none() { eprintln!("FAIL {}: {}", label, msg); }
                    (false, Some(msg))
                }
                Err(e) => {
                    failed += 1;
                    let msg = e.to_string();
                    if em.is_none() { eprintln!("FAIL {}: {}", label, msg); }
                    (false, Some(msg))
                }
            };

            // `--jit`: run the same test under the JIT for parity. A compile
            // error means the program is outside the JIT's subset → skip (the
            // JIT can't run it). A test that compiles but returns false or errors
            // is a real interp/JIT divergence → fail.
            let mut jit_verdict: Option<diag::JitVerdict> = None;
            let mut jit_message: Option<String> = None;
            // `merged` is computed exactly when `jit_test` is set (just above
            // the loop), so matching on it here — rather than branching on
            // `jit_test` and separately asserting `merged` is `Some` — makes
            // that pairing structural: there is no `merged.expect(...)` left
            // to panic if the two ever drift apart (#661 review finding 7).
            if let Some(merge_result) = merged.as_ref() {
                // Fresh arena per test, taken before anything JIT-side exists,
                // so no pointer handed out by this test's compile or run can
                // outlive the reset.
                jit::reset_forge_arena();
                match jit::Jit::new() {
                    Err(e) => {
                        failed += 1;
                        jit_verdict = Some(diag::JitVerdict::Fail);
                        jit_message = Some(format!("jit init: {}", e.msg));
                        if em.is_none() { eprintln!("FAIL {} [jit]: jit init: {}", label, e.msg); }
                    }
                    Ok(mut j) => {
                        // The merged program, not the entry file's own — the
                        // second of slice 3's two `compile_program` sites. The
                        // two ways a file can turn out to be outside the JIT's
                        // subset (the merge cannot be built; the merged program
                        // does not lower) are collapsed here because they earn
                        // the same verdict, `skip`.
                        let compiled = match merge_result {
                            Err(_) => Err(()),
                            Ok(m) => {
                                j.set_merge(m.clone());
                                j.compile_program(&m.program).map_err(|_| ())
                            }
                        };
                        match compiled {
                        Err(()) => {
                            jit_skipped += 1;
                            jit_verdict = Some(diag::JitVerdict::Skip);
                        }
                        Ok(()) => {
                            jit_ran += 1;
                            let outcome = j.run_test_fn(&name);
                            // An exhausted budget outranks whatever the test
                            // went on to return: under the recoverable policy
                            // the over-budget allocation was still handed out,
                            // so a test that blew `--forge` can perfectly well
                            // return true.
                            if let Some(diag) = jit::take_forge_exhaustion() {
                                failed += 1;
                                if em.is_none() { eprintln!("FAIL {} [jit]: {}", label, diag); }
                                jit_verdict = Some(diag::JitVerdict::Fail);
                                jit_message = Some(diag);
                            } else {
                                match outcome {
                                    Ok(true) => {
                                        jit_verdict = Some(diag::JitVerdict::Pass);
                                    }
                                    Ok(false) => {
                                        failed += 1;
                                        jit_verdict = Some(diag::JitVerdict::Fail);
                                        jit_message =
                                            Some("returned false (diverges from interp)".to_string());
                                        if em.is_none() {
                                            eprintln!("FAIL {} [jit]: returned false (diverges from interp)", label);
                                        }
                                    }
                                    Err(e) => {
                                        failed += 1;
                                        jit_verdict = Some(diag::JitVerdict::Fail);
                                        jit_message = Some(e.msg.clone());
                                        if em.is_none() { eprintln!("FAIL {} [jit]: {}", label, e.msg); }
                                    }
                                }
                            }
                        }
                        }
                    }
                }
            }

            if let Some(e) = &mut em {
                let mut t = mk_test(pass, message);
                t.jit = jit_verdict;
                t.jit_message = jit_message;
                e.emit_test(&t);
            }
        }
    }

    // The summary. Under `--json` the tallies ride in the terminal object —
    // the same numbers the human lines print, including the exit-code quirks
    // (`0 tests` exits 0 even after file-level failures), because the exit
    // code must not depend on the flag.
    if total == 0 {
        // #599: with `--fail-empty`, zero tests ran across the whole given
        // path is itself a failure — the exit code no longer says "0 tests"
        // and "everything passed" the same way. Without the flag this branch
        // is unchanged: `0 tests`, exit 0, even after file-level failures.
        let exit = if fail_empty { 1 } else { 0 };
        return match em {
            Some(mut e) => {
                // `failed` (the per-test tally) is always 0 here — every path
                // that increments it also increments `total` first — so
                // `file_failures` alone is the whole count.
                e.set_test_tally(0, file_failures);
                e.set_tests_ran(0);
                if jit_test { e.set_jit_parity(jit_ran, jit_skipped); }
                if jit_test && all_mode { e.set_blocker_depth(blocker_files, blocker_sites); }
                e.finish(exit)
            }
            None => {
                println!("0 tests");
                // A genuine file-level failure (a `FAIL …` line already printed
                // above) already earns its own exit; blaming that on the flag
                // here would be wrong, so the flag's own line is only for the
                // case it actually explains: zero tests and nothing failed.
                if fail_empty && file_failures == 0 {
                    eprintln!("test result: FAILED. 0 tests ran (--fail-empty)");
                }
                exit
            }
        };
    }
    // #854: `total_failed` is what gets printed and what decides the exit
    // code — every failure, file-level or per-test, counts as one failure.
    // `passed` stays `total - failed` (the per-test tally only): a file-level
    // failure was never part of `total` to begin with, so it must never be
    // subtracted back out of it, or a red run from one bad file would zero
    // out every `ok` a different, successfully-resolved file printed.
    let total_failed = failed + file_failures;
    let exit = if total_failed == 0 { 0 } else { 1 };
    match em {
        Some(mut e) => {
            e.set_test_tally(total.saturating_sub(failed), total_failed);
            e.set_tests_ran(total);
            if jit_test { e.set_jit_parity(jit_ran, jit_skipped); }
            if jit_test && all_mode { e.set_blocker_depth(blocker_files, blocker_sites); }
            e.finish(exit)
        }
        None => {
            // #684: `--all` appends the blocker-depth count to the existing
            // parity note rather than replacing it — the golden gate parses
            // the line this is built from, and without `--all` this string is
            // built exactly as it always was (`all_mode` is false, so this
            // whole branch is dead code for that string).
            let jit_note = if jit_test {
                let depth = if all_mode {
                    // #684 R2: the human line reports DISTINCT ROOT BLOCKERS
                    // (`blocker_sites` sums each file's `jit_blockers`, not its
                    // raw `jit_sites` — see the doc comment on `blocker_sites`
                    // in `diag.rs`), not raw refusal-site count.
                    //
                    // #684 rework 3: when at least one measured file's depth
                    // is a floor (`truncated:true`), name that here too —
                    // omitted entirely when it's zero, so a corpus with no
                    // truncated files reproduces the line exactly as before
                    // this rework. Appended, never inserted mid-line: the
                    // golden gate (`tools/example_runner.py`'s `JIT_RE`)
                    // matches `jit parity: (\d+) ran, (\d+) skipped` with no
                    // trailing anchor, so text appended after it is safe.
                    let floor_note = if blocker_truncated_files > 0 {
                        format!(" ({} truncated — floor)", blocker_truncated_files)
                    } else {
                        String::new()
                    };
                    format!(" | blocker depth: {} files, {} blockers{}",
                        blocker_files, blocker_sites, floor_note)
                } else {
                    String::new()
                };
                format!(" | jit parity: {} ran, {} skipped (outside JIT subset){}",
                    jit_ran, jit_skipped, depth)
            } else {
                String::new()
            };
            if total_failed == 0 {
                println!("test result: ok. {} passed; 0 failed{}", total, jit_note);
            } else {
                eprintln!(
                    "test result: FAILED. {} passed; {} failed{}",
                    total.saturating_sub(failed),
                    total_failed,
                    jit_note,
                );
            }
            exit
        }
    }
}

/// Native stack for the interpreter thread. The tree-walking interpreter spends
/// ~100 KB of stack per demoniC call; the default 8 MB main-thread stack caps
/// recursion at ~80 deep. Running the whole CLI on a dedicated 256 MB thread
/// (the same trick `rustc` uses for itself) lifts that to a few thousand, and the
/// `MAX_CALL_DEPTH` guard in the interpreter trips below the native ceiling so deep
/// recursion is a catchable error rather than a SIGABRT.
const INTERP_STACK_SIZE: usize = 256 * 1024 * 1024;

fn main() {
    let child = std::thread::Builder::new()
        .name("dmc-main".into())
        .stack_size(INTERP_STACK_SIZE)
        .spawn(real_main)
        .expect("failed to spawn dmc main thread");
    // `real_main` drives the whole CLI and calls `std::process::exit` on every
    // terminal path; if it instead panics, surface that as a process failure.
    if child.join().is_err() {
        std::process::exit(101);
    }
}

fn real_main() {
    let raw_args: Vec<String> = std::env::args().collect();
    if raw_args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    // Strip --profile / --demon flags from args; pass the rest through normal
    // parsing. --demon releases the Control Art Restriction: the safe-mode lint
    // family is suppressed (#196) — raw, full speed, no guardrails.
    let mut profile_mode = false;
    let mut demon_mode = false;
    let mut jit_test = false;
    // #599: `dmc test --fail-empty` — a suite that discovered zero `test_*`
    // functions across the given file or directory exits 1 instead of 0, so a
    // CI gate that only checks the exit status cannot be fooled by a suite
    // that verified nothing. Without the flag, behavior is unchanged: `0
    // tests` still exits 0.
    let mut fail_empty = false;
    let mut gpu_mode = false;
    // #578: `--blas` routes a large f32 matmul to the host BLAS
    // (`cblas_sgemm`, Accelerate/AMX on macOS) instead of the Cranelift
    // kernel. Off by default — BLAS accumulates in its own blocked order, so
    // the numbers move (`NUMERICS.md §2.2`); the flag is where that trade is
    // stated. Inert on a host with no BLAS in its process image, and never
    // selected inside `@deterministic`.
    let mut blas_mode = false;
    // #485: `--json` re-encodes the diagnostics as JSON Lines on stderr. It
    // changes nothing else — not the set of diagnostics, not the exit code.
    let mut json_mode = false;
    // #684: `--all` reports every JIT refusal in a program instead of just
    // the first (`dmc jit`), and gives `dmc test --jit` a per-file
    // blocker-depth count. Without it, `dmc jit` and `dmc test --jit` output
    // is unchanged in every particular — the same first-refusal behavior
    // `tools/jit_probes.py` and the golden gate already parse.
    let mut all_mode = false;
    // #400: `--vault=<size>` / `--forge=<size>` size the arenas (`MEMORY.md
    // §1.1`). Both spellings are accepted — `--forge=2G` and `--forge 2G`.
    let mut arena_limits = arena::ArenaLimits::default();
    let mut args: Vec<String> = Vec::with_capacity(raw_args.len());
    args.push(raw_args[0].clone());
    let mut i = 1;
    while i < raw_args.len() {
        let a = raw_args[i].as_str();
        if a == "--profile" { profile_mode = true; }
        else if a == "--demon" { demon_mode = true; }
        else if a == "--jit" { jit_test = true; }
        else if a == "--fail-empty" { fail_empty = true; }
        else if a == "--gpu" { gpu_mode = true; }
        else if a == "--blas" { blas_mode = true; }
        else if a == "--json" { json_mode = true; }
        else if a == "--all" { all_mode = true; }
        else if let Some(which) = arena::flag_arena(a) {
            let value = match a.split_once('=') {
                Some((_, v)) => v.to_string(),
                None => match raw_args.get(i + 1) {
                    Some(v) => { i += 1; v.clone() }
                    None => {
                        eprintln!("error: `{}` needs a size, e.g. `{}=2G`", a, which.flag());
                        std::process::exit(1);
                    }
                },
            };
            match arena::parse_size(&value) {
                Ok(bytes) => arena_limits.set(which, bytes),
                Err(e) => {
                    eprintln!("error: {}: {}", which.flag(), e);
                    std::process::exit(1);
                }
            }
        }
        else { args.push(raw_args[i].clone()); }
        i += 1;
    }

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    // #400: `--forge` is a real ceiling on the JIT's Forge arena, installed on
    // this thread before any compiled code runs. `--vault` is not: the JIT
    // lowers `vault.*` and `forge.*` constructors into that same single arena
    // (there is no separate Vault region yet), so honoring the flag would mean
    // metering an arena that does not exist. Refuse it instead of accepting it
    // and quietly doing nothing.
    if arena_limits.vault.is_some() && (jit_test || args[1] == "jit") {
        eprintln!(
            "error: `--vault` is not honored under the JIT — it lowers `vault.*` and \
             `forge.*` into one Forge arena, so there is no Vault to size. Use the \
             interpreter (`dmc run`, or `dmc test` without `--jit`) for a Vault \
             budget, or drop the flag — `--forge` works here."
        );
        std::process::exit(1);
    }
    jit::set_forge_limit(arena_limits.forge);

    // #485 wires `--json` to the three consumer-ready commands: check errors
    // (`dmc --check`), JIT-ineligibility refusals (`dmc jit`), and per-test
    // results (`dmc test`). Runtime errors are a later slice with a reserved
    // kind; accepting `--json` elsewhere and emitting nothing structured would
    // be worse than refusing, because a consumer cannot tell the two apart.
    // The subcommand as spelled, for the summary's `command` field. A bare
    // file argument is the implicit full pipeline, not a command name — say so
    // rather than reporting the path as if it were one.
    let command = match args[1].as_str() {
        c if c.starts_with("--") => c.to_string(),
        c @ ("run" | "jit" | "test" | "selftest" | "assimilate" | "fmt") => c.to_string(),
        _ => "pipeline".to_string(),
    };
    let mut emitter = if json_mode {
        // Arm the out-of-band path too: the JIT's arena-exhaustion abort
        // exits from inside compiled code and never returns here.
        diag::arm_out_of_band(&command);
        Some(diag::Emitter::new(&command))
    } else {
        None
    };
    if json_mode && command != "--check" && command != "jit" && command != "test" {
        fail_cli(
            &mut emitter,
            "json-unsupported-command",
            format!(
                "`--json` is wired for `dmc --check`, `dmc jit`, and `dmc test` \
                 in schema {}; `{}` still reports in the human format",
                diag::SCHEMA, command,
            ),
        );
    }
    // #684: `--all` is wired for `dmc jit` and `dmc test --jit` only. Under
    // `dmc jit` it has no rendering but the JSON one (requirement 2 never
    // describes a multi-refusal human line), so it additionally needs
    // `--json`; under `dmc test` the blocker-depth count is a plain-text
    // summary-line addition too, so `--json` is not required there.
    if all_mode && command != "jit" && command != "test" {
        fail_cli(
            &mut emitter,
            "all-unsupported-command",
            format!(
                "`--all` reports every JIT refusal instead of the first; it is wired \
                 for `dmc jit` and `dmc test --jit`, not `{}`",
                command,
            ),
        );
    }
    if all_mode && command == "jit" && !json_mode {
        fail_cli(
            &mut emitter,
            "all-requires-json",
            "`dmc jit --all` only has a JSON rendering today — pass `--json` too, \
             or drop `--all` for the first-refusal behavior",
        );
    }

    let (mode, path) = match args[1].as_str() {
        "--lex" => {
            if args.len() < 3 { eprintln!("missing file"); std::process::exit(1); }
            ("lex", PathBuf::from(&args[2]))
        }
        "--parse" => {
            if args.len() < 3 { eprintln!("missing file"); std::process::exit(1); }
            ("parse", PathBuf::from(&args[2]))
        }
        "--check" => {
            if args.len() < 3 { fail_cli(&mut emitter, "missing-input", "missing file"); }
            ("check", PathBuf::from(&args[2]))
        }
        "run" => {
            if args.len() < 3 { eprintln!("missing file"); std::process::exit(1); }
            ("run", PathBuf::from(&args[2]))
        }
        "jit" => {
            if args.len() < 3 { fail_cli(&mut emitter, "missing-input", "missing file"); }
            ("jit", PathBuf::from(&args[2]))
        }
        "test" => {
            if args.len() < 3 {
                fail_cli(&mut emitter, "missing-input", "missing file or directory");
            }
            let code = run_tests(
                Path::new(&args[2]), profile_mode, jit_test, fail_empty, all_mode, arena_limits, emitter.take(),
            );
            std::process::exit(code);
        }
        "selftest" => {
            // In-process interp-vs-JIT differential fuzzer (#408). Generates
            // well-typed scalar programs and asserts the two backends agree.
            let code = run_selftest(&args[2..]);
            std::process::exit(code);
        }
        "assimilate" => {
            // ASSIMILATE.md — generate port-wrapper bindings from a JSON
            // descriptor. `assimilate <descriptor.json> [-o <out.dmc>]`.
            std::process::exit(run_assimilate(&args[2..]));
        }
        "fmt" => {
            if args.len() < 3 { eprintln!("missing file"); std::process::exit(1); }
            ("fmt", PathBuf::from(&args[2]))
        }
        flag if flag.starts_with("--") => {
            eprintln!("unknown flag: {}", flag);
            print_usage();
            std::process::exit(1);
        }
        _ => ("pipeline", PathBuf::from(&args[1])),
    };

    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => fail_cli(
            &mut emitter,
            "input-unreadable",
            format_args!("error reading {:?}: {}", path, e),
        ),
    };

    let path = match path.canonicalize() {
        Ok(p) => p,
        Err(e) => fail_cli(
            &mut emitter,
            "input-unresolvable",
            format_args!("error resolving path {:?}: {}", path, e),
        ),
    };

    let tokens = match Lexer::new(&src).tokenize() {
        Ok(t) => t,
        Err(e) => {
            match emitter.take() {
                Some(mut em) => {
                    let d = diag::Diagnostic::new(
                        diag::Kind::Lex,
                        diag::Severity::Error,
                        e.msg.clone(),
                    )
                    .file(&path)
                    .at(e.line, e.col);
                    em.emit(&d);
                    std::process::exit(em.finish(1));
                }
                None => {
                    eprintln!("lex error in {:?}: {}", path, e);
                    std::process::exit(1);
                }
            }
        }
    };

    match mode {
        "lex" => {
            for tok in &tokens {
                if tok.kind == TokenKind::Eof { break; }
                println!("{:4}:{:3}  {:30}  {:?}",
                    tok.span.line, tok.span.col,
                    format!("{:?}", tok.kind),
                    tok.raw
                );
            }
            println!("\n{} tokens total", tokens.len());
        }
        "parse" => {
            let mut parser = Parser::new(tokens);
            match parser.parse_program() {
                Ok(program) => {
                    println!("{:#?}", program);
                    println!("\n✅ Parse OK — {} top-level items", program.items.len());
                }
                Err(e) => { eprintln!("{}", e); std::process::exit(1); }
            }
        }
        "check" => {
            let (_resolver, modules) = match resolve_and_check(&path, demon_mode) {
                Ok(v) => v,
                Err(e) => match emitter.take() {
                    Some(mut em) => {
                        em.emit(&resolve_diag(&e));
                        std::process::exit(em.finish(1));
                    }
                    None => { eprintln!("{}", e); std::process::exit(1); }
                },
            };
            let mut total_errors = 0;
            let mut items_count = 0;
            for m in &modules {
                items_count += m.items;
                total_errors += m.errors.len();
                match &mut emitter {
                    Some(em) => {
                        for err in &m.errors {
                            em.emit(&check_diag(err, &m.path, diag::Severity::Error));
                        }
                        for w in &m.warnings {
                            em.emit(&check_diag(w, &m.path, diag::Severity::Warning));
                        }
                    }
                    None => {
                        for err in &m.errors {
                            eprintln!("{}", err);
                        }
                        print_warnings(&m.warnings);
                    }
                }
            }
            if let Some(mut em) = emitter.take() {
                // The `✅ Check OK — N top-level items` line and the
                // `N type error(s)` tally are the same two facts the summary
                // object carries; they are not repeated in prose.
                em.set_items(items_count);
                std::process::exit(em.finish(if total_errors == 0 { 0 } else { 1 }));
            }
            if total_errors == 0 {
                println!("✅ Check OK — {} top-level items, no type errors", items_count);
            } else {
                eprintln!("\n{} type error(s)", total_errors);
                std::process::exit(1);
            }
        }
        "run" => {
            let (resolver, modules) = match resolve_and_check(&path, demon_mode) {
                Ok(v) => v,
                Err(e) => { eprintln!("{}", e); std::process::exit(1); }
            };
            let mut total_errors = 0;
            for m in &modules {
                for err in &m.errors {
                    eprintln!("{}", err);
                }
                total_errors += m.errors.len();
                print_warnings(&m.warnings);
            }
            if total_errors > 0 {
                eprintln!("\n⚠ {} type error(s) — refusing to run", total_errors);
                std::process::exit(1);
            }
            let mut interp = Interpreter::new();
            interp.set_arena_limits(arena_limits);
            if profile_mode { interp.enable_profile(); }
            interp.set_argv(args[3..].to_vec());
            for p in &resolver.sorted_paths {
                // The main file's own top-level items are loaded and evaluated
                // exactly once, below, by `interp.run` — not here too (#794).
                // Every dependency still gets this pass (`interp.run` only
                // loads the one path it is handed), and each still runs its
                // own top-level `let`s exactly once, here.
                if p == &path {
                    continue;
                }
                let prog = resolver.files.get(p).unwrap();
                if let Err(e) = interp.load_program(prog, Some(p)) {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
                let mod_env = interp.get_module_env();
                interp.interp_modules.insert(p.clone(), mod_env);
            }
            let main_prog = match resolver.files.get(&path) {
                Some(prog) => prog,
                None => {
                    eprintln!("error: main program {:?} not found in resolved files", path);
                    std::process::exit(1);
                }
            };
            match interp.run(main_prog, Some(&path)) {
                Ok(v) => {
                    // Print final value if non-nil (REPL-style feedback).
                    if !matches!(v, interp::Value::Nil) {
                        println!("\n=> {:?}", v);
                    }
                    if profile_mode {
                        if let Some(p) = &interp.profile { print_profile(p); }
                    }
                }
                Err(e) => { eprintln!("{}", e); std::process::exit(1); }
            }
        }
        "fmt" => {
            let mut parser = Parser::new(tokens);
            let program = match parser.parse_program() {
                Ok(p) => p,
                Err(e) => { eprintln!("{}", e); std::process::exit(1); }
            };
            print!("{}", pretty_print_program(&program));
        }
        "jit" => {
            // As of #661, this arm lowers a
            // multi-file program by merging its whole import DAG into one
            // item list (`merge.rs`, slice 1) and handing that to the single
            // `JITModule` backend — so a `use` is no longer refused just for
            // existing. `jit-import` still fires, but only when the merge
            // itself cannot be built (a shape the JIT's subset does not
            // cover, e.g. `SPEC.md`'s ambiguous-import case); see the
            // `merge::merge_program` call below. Before slice 3 this arm
            // lexed and parsed only the one named file and gave it to a
            // `Checker` with no module table, so an importing program died at
            // `imported module not found / checked yet` — a check error for
            // an import `--check` and `run` both resolve — and never reached
            // the backend that would have refused it properly. The
            // resolution and the checking are now `resolve_and_check`, the
            // same call those two make.
            //
            // #505 (fold `@comptime` before the checker) and #478 step 1
            // (`dmc jit` type-checks at all, so `--check` is sufficient in both
            // directions) both live inside that call now, and so hold across
            // every module rather than only the entry file.
            let (resolver, modules) = match resolve_and_check(&path, demon_mode) {
                Ok(v) => v,
                Err(e) => match emitter.take() {
                    Some(mut em) => {
                        em.emit(&resolve_diag(&e));
                        std::process::exit(em.finish(1));
                    }
                    None => { eprintln!("{}", e); std::process::exit(1); }
                },
            };
            let total_errors: usize = modules.iter().map(|m| m.errors.len()).sum();
            if total_errors > 0 {
                if let Some(mut em) = emitter.take() {
                    for m in &modules {
                        for err in &m.errors {
                            em.emit(&check_diag(err, &m.path, diag::Severity::Error));
                        }
                    }
                    std::process::exit(em.finish(1));
                }
                for m in &modules {
                    for err in &m.errors {
                        eprintln!("{}", err);
                    }
                }
                eprintln!("\n⚠ {} type error(s) — refusing to run", total_errors);
                std::process::exit(1);
            }
            match &mut emitter {
                Some(em) => {
                    for m in &modules {
                        for w in &m.warnings {
                            em.emit(&check_diag(w, &m.path, diag::Severity::Warning));
                        }
                    }
                }
                None => {
                    for m in &modules {
                        print_warnings(&m.warnings);
                    }
                }
            }
            let program = match resolver.files.get(&path) {
                Some(prog) => prog,
                None => fail_unstructured(
                    &mut emitter,
                    format!("error: main program {:?} not found in resolved files", path),
                ),
            };
            // The backend's unit of compilation is
            // one `Program` inside one `JITModule` with one flat symbol table,
            // and it has nowhere to put a second file's functions — so what it
            // gets is not the entry file's own `Program` but the whole import
            // DAG merged into one item list no two files can collide inside
            // (`merge.rs`, slice 1). On a single-file program the merge is the
            // identity and this is the path that always ran.
            //
            // A merge that cannot be built is a refusal, not a failure: the
            // program is outside the JIT's subset and `dmc run` has the full
            // semantics, which is `jit-ineligible` / `jit-import`, never
            // `jit-error`.
            let merged = match merge::merge_program(&resolver, &path).map(std::sync::Arc::new) {
                Ok(m) => m,
                Err(e) => match emitter.take() {
                    Some(mut em) => {
                        em.emit(&merge_diag(&e, &path));
                        std::process::exit(em.finish(1));
                    }
                    None => {
                        // The same shape `JitError`'s `Display` gives every
                        // other refusal — `jit unsupported at L:C: …` — because
                        // that leading phrase is what the differential tools
                        // match on to tell a refusal from a defect (#480).
                        match e.span() {
                            Some(s) => eprintln!(
                                "jit unsupported at {}:{}: {}; use `dmc run` for full semantics",
                                s.line, s.col, e,
                            ),
                            None => eprintln!(
                                "jit unsupported: {}; use `dmc run` for full semantics", e),
                        }
                        std::process::exit(1);
                    }
                },
            };
            let mut jit = match Jit::new() {
                Ok(j) => j,
                Err(e) => match emitter.take() {
                    Some(mut em) => {
                        em.emit(&jit_diag(&e, &path));
                        std::process::exit(em.finish(1));
                    }
                    None => { eprintln!("{}", e); std::process::exit(1); }
                },
            };
            if gpu_mode {
                // #326: honored only on a macOS `--features gpu` build; otherwise
                // the GPU symbol is absent and this stays on the CPU kernel.
                #[cfg(all(target_os = "macos", feature = "gpu"))]
                jit.set_gpu(true);
                #[cfg(not(all(target_os = "macos", feature = "gpu")))]
                {
                    const GPU_IGNORED: &str =
                        "--gpu ignored (build with `--features gpu` on macOS to enable)";
                    match &mut emitter {
                        Some(em) => em.emit(&diag::Diagnostic::new(
                            diag::Kind::Cli,
                            diag::Severity::Warning,
                            GPU_IGNORED,
                        ).code("gpu-unavailable")),
                        None => eprintln!("warning: {}", GPU_IGNORED),
                    }
                }
            }
            if blas_mode {
                // Honored only where `cblas_sgemm` resolves in this process.
                // Elsewhere the setter is inert and the Cranelift kernel runs,
                // so a build or a host without a BLAS needs no cfg — the
                // absence of the symbol is the whole mechanism.
                jit.set_blas(true);
                if !jit::blas_gemm_available() {
                    const BLAS_IGNORED: &str =
                        "--blas ignored (no `cblas_sgemm` in this process's dynamic symbol table)";
                    match &mut emitter {
                        Some(em) => em.emit(&diag::Diagnostic::new(
                            diag::Kind::Cli,
                            diag::Severity::Warning,
                            BLAS_IGNORED,
                        ).code("blas-unavailable")),
                        None => eprintln!("warning: {}", BLAS_IGNORED),
                    }
                }
            }
            // The merged→source map, so a refusal raised inside an imported
            // item quotes the name the user wrote and reports the file that
            // declared it — the design's §6 question, answered "the declaring
            // file". Without it the message would read `util$helper` and blame
            // whichever file was typed on the command line (§5.2).
            jit.set_merge(merged.clone());
            // #684: `--all` collects every refusal instead of stopping at the
            // first, and is diagnostic-only — it never runs `main`, even when
            // it finds nothing to report, so the collected count is never in
            // tension with "did the program actually execute". Argument
            // parsing already refused `--all` without `--json` for `jit`, so
            // `emitter` is `Some` here.
            if all_mode {
                let (refusals, truncated) = jit.compile_program_collect(&merged.program);
                let mut em = emitter.take().expect("--all requires --json (checked at arg parsing)");
                for (i, r) in refusals.iter().enumerate() {
                    let file = r.file.as_deref().unwrap_or(&path);
                    em.emit(&jit_diag(&r.error, file).index(i).item(r.item.clone()).cascade_of(r.cascade_of));
                }
                // #684 R2: the three headline numbers, computed once and
                // shared with `dmc test --jit --json --all`'s per-file fields
                // (`jit::blocker_counts`) — `sites` (raw, unchanged from
                // before R2), `blockers` (distinct roots, the new headline),
                // `cascades`, and `classes` now counted over non-cascade
                // sites only, keyed on `kind` rather than a fabricated `code`
                // for a `jit-error` (R3, `jit::JIT_ERROR_CLASS_KEY`). F3:
                // `truncated` says whether those three are a floor — `jit`
                // (unlike `test --jit`) always has a successfully constructed
                // `Jit` by this point (a `Jit::new()` failure exits above,
                // before `--all` collection ever runs), so this is a plain
                // `bool`, never the `--jit --json --all` per-file tri-state.
                // `compile_program_collect` returns it explicitly now
                // (rework 3) — it is no longer inferred from `refusals`.
                let counts = jit::blocker_counts(&refusals, truncated);
                em.set_jit_all(counts.sites, counts.blockers, counts.cascades,
                    diag::classes_object(&counts.classes), counts.truncated);
                std::process::exit(em.finish(if counts.sites == 0 { 0 } else { 1 }));
            }
            if let Err(e) = jit.compile_program(&merged.program) {
                // The payload #485 exists for: an agent iterating a program
                // toward JIT eligibility reads `code` and `line`/`col`, not an
                // English sentence.
                let file = jit.error_file().unwrap_or(&path).to_path_buf();
                if let Some(mut em) = emitter.take() {
                    em.emit(&jit_diag(&e, &file));
                    std::process::exit(em.finish(1));
                }
                eprintln!("{}", e);
                std::process::exit(1);
            }
            match jit.run_main() {
                Ok(rc) => {
                    // #271: print REPL feedback by main's *declared return type*,
                    // not by `rc != 0` — the old guard swallowed a real `0`/`false`
                    // return (run_main also returns 0 for a nil main). Float and
                    // str mains already print inside run_main (only it can read a
                    // forge string); nil prints nothing.
                    let ret = program.items.iter()
                        .find_map(|it| find_fn_decl(it, "main"))
                        .and_then(|f| f.ret_type.as_ref());
                    match ret {
                        None => {} // unannotated main = nil
                        Some(ast::Type::Scalar(ast::ScalarType::Nil, _))
                        | Some(ast::Type::Scalar(ast::ScalarType::F32, _))
                        | Some(ast::Type::Scalar(ast::ScalarType::F64, _))
                        | Some(ast::Type::Scalar(ast::ScalarType::Str, _)) => {}
                        Some(ast::Type::Scalar(ast::ScalarType::Bool, _)) => {
                            println!("=> {}", rc != 0);
                        }
                        // #597: `u64`'s register can hold a negative `i64`
                        // pattern (top bit set) naming a value at or above
                        // 2^63 — reinterpret before printing, matching
                        // `dmc run`'s interpreter path and the JIT's own
                        // `print(x)` / `to_str(x)` fix for the same type.
                        Some(ast::Type::Scalar(ast::ScalarType::U64, _)) => {
                            println!("=> {}", rc as u64);
                        }
                        Some(_) => println!("=> {}", rc),
                    }
                }
                // A trap inside compiled code is a *runtime* diagnostic — a
                // later slice's category. Schema 1 reserves `runtime` and
                // carries the prose through as `unstructured` so the stream
                // stays parseable and nothing is lost.
                Err(e) => fail_unstructured(&mut emitter, e),
            }
        }
        "pipeline" => {
            let mut resolver = resolver::Resolver::new();
            if let Err(e) = resolver.resolve_all(&path) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
            println!("✅ Lex OK");
            println!("✅ Parse OK");
            let folded = fold_comptime_all(&mut resolver);
            let mut total_errors = 0;
            for m in &check_resolved(&resolver, &folded, demon_mode) {
                for err in &m.errors {
                    eprintln!("{}", err);
                }
                total_errors += m.errors.len();
                print_warnings(&m.warnings);
            }
            if total_errors > 0 {
                eprintln!("\n⚠ {} type error(s) — pre-alpha checker", total_errors);
                std::process::exit(1);
            }
            println!("✅ Check OK — no type errors");
            let mut interp = Interpreter::new();
            interp.set_arena_limits(arena_limits);
            if profile_mode { interp.enable_profile(); }
            for p in &resolver.sorted_paths {
                // Same double-evaluation the `run` arm fixes (#794): the main
                // file loads and runs exactly once, below, via `interp.run`.
                if p == &path {
                    continue;
                }
                let prog = resolver.files.get(p).unwrap();
                if let Err(e) = interp.load_program(prog, Some(p)) {
                    eprintln!("⚠ runtime: {}", e);
                    std::process::exit(1);
                }
                let mod_env = interp.get_module_env();
                interp.interp_modules.insert(p.clone(), mod_env);
            }
            let main_prog = match resolver.files.get(&path) {
                Some(prog) => prog,
                None => {
                    eprintln!("error: main program {:?} not found in resolved files", path);
                    std::process::exit(1);
                }
            };
            match interp.run(main_prog, Some(&path)) {
                Ok(v) => {
                    println!("✅ Run OK");
                    if !matches!(v, interp::Value::Nil) {
                        println!("\n=> {:?}", v);
                    }
                    if profile_mode {
                        if let Some(p) = &interp.profile { print_profile(p); }
                    }
                }
                Err(e) => {
                    eprintln!("⚠ runtime: {}", e);
                    std::process::exit(1);
                }
            }
        }
        _ => unreachable!(),
    }

    // Every `--json` run ends with exactly one summary object. The failing
    // paths above emit theirs on the way out; this is the one for a run that
    // reached the end without exiting.
    if let Some(em) = emitter.take() {
        std::process::exit(em.finish(0));
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    fn parse(src: &str) -> ast::Program {
        let tokens = Lexer::new(src).tokenize().expect("lex failed");
        Parser::new(tokens).parse_program().expect("parse failed")
    }

    #[test]
    fn test_discovery_finds_only_test_prefix() {
        let program = parse(r#"
            fn helper() -> bool { true }
            fn test_add() -> bool { 1 + 1 == 2 }
            fn test_nil() -> nil { nil }
        "#);
        assert_eq!(collect_test_fns(&program), vec!["test_add", "test_nil"]);
    }

    #[test]
    fn test_discovery_looks_through_item_directive() {
        let program = parse(r#"
            @deterministic
            fn test_seeded() -> bool { true }
        "#);
        assert_eq!(collect_test_fns(&program), vec!["test_seeded"]);
    }
}
