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

use std::path::{Path, PathBuf};
use lexer::{Lexer, TokenKind};
use parser::Parser;
use check::Checker;
use interp::Interpreter;
use fmt::pretty_print_program;
use jit::Jit;

fn print_usage() {
    eprintln!("usage:");
    eprintln!("  dmc <file.dmc>                    run the full pipeline (lex + parse + check + execute)");
    eprintln!("  dmc --lex <file.dmc>              dump the token stream");
    eprintln!("  dmc --parse <file.dmc>            dump the AST tree");
    eprintln!("  dmc --check <file.dmc>            type-check, report diagnostics");
    eprintln!("  dmc run <file.dmc>                execute the program (tree-walking interpreter)");
    eprintln!("  dmc jit <file.dmc>                JIT (Cranelift, scalars + control flow)");
    eprintln!("  dmc test <file-or-dir>            run zero-arg test_* functions (interpreter)");
    eprintln!("  dmc test --jit <file-or-dir>      also run each test_* under the JIT (parity gate; skips files outside the JIT subset)");
    eprintln!("  dmc selftest [opts]               interp-vs-JIT differential fuzzer over generated scalar programs");
    eprintln!("                                    opts: --iters N --seed S --repro SEED --floats --meta-test --verbose --timeout SECS");
    eprintln!("  dmc fmt <file.dmc>                pretty-print with canonical formatting (to stdout)");
    eprintln!("  dmc assimilate <desc.json> [-o f] generate port-wrapper bindings from a descriptor (ASSIMILATE.md)");
    eprintln!("  dmc assimilate python:<module>    introspect a live runtime into a draft descriptor (add --bindings for wrappers)");
    eprintln!("  dmc --profile <file.dmc>          run with op-count profiling (emits summary to stderr)");
    eprintln!("  dmc --profile run <file.dmc>      run mode with profiling");
    eprintln!("  dmc --demon <file.dmc>            demon mode: release the safe-mode lints (raw, no guardrails)");
    eprintln!("  dmc run <file.dmc> --vault=16G --forge=2G");
    eprintln!("                                    arena byte budgets (MEMORY.md §1.1); B/K/M/G/T suffixes,");
    eprintln!("                                    exhausting one is a runtime error. `--vault` is `dmc run` only.");
    eprintln!("  dmc --check --json <file.dmc>     diagnostics as JSON Lines on stderr (schema 1)");
    eprintln!("  dmc jit --json <file.dmc>         same, including machine-coded JIT-ineligibility refusals");
    eprintln!("  dmc test --json <file-or-dir>     same, streaming one `test` object per test (with `--jit`, the parity verdict rides along)");
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
/// checker runs (`COMPTIME_V1.md §3`).
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
/// round-trips through the formatter unfolded (`COMPTIME_V1.md §7`).
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

/// A module-resolution failure in `--json` form. The located halves are the
/// same `lex` / `parse` diagnostics `dmc jit` reports; the rest (I/O, a bad
/// import path, a cycle) has no span and stays `unstructured`.
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
        resolver::ResolveError::Other(s) => diag::Diagnostic::unstructured(s.clone()),
    }
}

/// Report a diagnostic whose category schema 1 does not model yet — resolver
/// prose, a JIT runtime trap — and exit 1. Without `--json` this is the
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
    // `test` object; under `--json` the prose is carried as `unstructured`, and
    // it still counts into the summary's `failed`, exactly as the human tally.
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
    // #: `--jit` parity counters. Each test_* is additionally run under the JIT;
    // a file the JIT can't compile is skipped (outside its subset), not failed.
    let mut jit_ran = 0usize;
    let mut jit_skipped = 0usize;
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
                failed += 1;
                fail_line(&mut em, format!("FAIL {}: canonicalization failed: {}", file.display(), e));
                continue;
            }
        };
        let mut resolver = resolver::Resolver::new();
        if let Err(e) = resolver.resolve_all(&canonical_file) {
            failed += 1;
            fail_line(&mut em, format!("FAIL {}: resolution failed: {}", file.display(), e));
            continue;
        }

        let folded = fold_comptime_all(&mut resolver);

        let program = match resolver.files.get(&canonical_file) {
            Some(prog) => prog.clone(),
            None => {
                failed += 1;
                fail_line(&mut em, format!("FAIL {}: program not found in resolved files", file.display()));
                continue;
            }
        };
        let tests = collect_test_fns(&program);
        if tests.is_empty() {
            continue;
        }

        let mut checked_modules = std::collections::HashMap::new();
        let mut check_failed = false;
        for p in &resolver.sorted_paths {
            let prog = resolver.files.get(p).unwrap();
            let mut checker = Checker::new();
            checker.errors = folded.get(p).cloned().unwrap_or_default();
            checker.checked_modules = checked_modules.clone();
            checker.check_program(prog, Some(p));
            if !checker.errors.is_empty() {
                failed += 1;
                fail_line(&mut em, format!("FAIL {}: type check failed in dependency {:?}", file.display(), p));
                match &mut em {
                    // The checker errors themselves are `check` diagnostics with
                    // spans — the same encoding `dmc --check` gives them.
                    Some(e) => {
                        for err in &checker.errors {
                            e.emit(&check_diag(err, p, diag::Severity::Error));
                        }
                    }
                    None => {
                        for err in &checker.errors {
                            eprintln!("{}", err);
                        }
                    }
                }
                check_failed = true;
                break;
            }
            let mod_env = check::ModuleEnv {
                env: checker.env.clone(),
                aliases: checker.aliases.clone(),
                public_items: ast::collect_public_items(prog),
            };
            checked_modules.insert(p.clone(), mod_env);
        }
        if check_failed {
            continue;
        }

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
            if jit_test {
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
                    Ok(mut j) => match j.compile_program(&program) {
                        Err(_) => {
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
                    },
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
        return match em {
            Some(mut e) => {
                e.set_test_tally(0, failed);
                if jit_test { e.set_jit_parity(jit_ran, jit_skipped); }
                e.finish(0)
            }
            None => {
                println!("0 tests");
                0
            }
        };
    }
    let exit = if failed == 0 { 0 } else { 1 };
    match em {
        Some(mut e) => {
            e.set_test_tally(total.saturating_sub(failed), failed);
            if jit_test { e.set_jit_parity(jit_ran, jit_skipped); }
            e.finish(exit)
        }
        None => {
            let jit_note = if jit_test {
                format!(" | jit parity: {} ran, {} skipped (outside JIT subset)", jit_ran, jit_skipped)
            } else {
                String::new()
            };
            if failed == 0 {
                println!("test result: ok. {} passed; 0 failed{}", total, jit_note);
            } else {
                eprintln!(
                    "test result: FAILED. {} passed; {} failed{}",
                    total.saturating_sub(failed),
                    failed,
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
        else if a == "--gpu" { gpu_mode = true; }
        else if a == "--blas" { blas_mode = true; }
        else if a == "--json" { json_mode = true; }
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
                Path::new(&args[2]), profile_mode, jit_test, arena_limits, emitter.take(),
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
            let mut resolver = resolver::Resolver::new();
            if let Err(e) = resolver.resolve_all(&path) {
                match emitter.take() {
                    Some(mut em) => {
                        em.emit(&resolve_diag(&e));
                        std::process::exit(em.finish(1));
                    }
                    None => { eprintln!("{}", e); std::process::exit(1); }
                }
            }
            let folded = fold_comptime_all(&mut resolver);
            let mut checked_modules = std::collections::HashMap::new();
            let mut total_errors = 0;
            let mut items_count = 0;
            for p in &resolver.sorted_paths {
                let prog = resolver.files.get(p).unwrap();
                items_count += prog.items.len();
                let mut checker = Checker::new();
                checker.errors = folded.get(p).cloned().unwrap_or_default();
                checker.demon = demon_mode;
                checker.checked_modules = checked_modules.clone();
                checker.check_program(prog, Some(p));
                total_errors += checker.errors.len();
                match &mut emitter {
                    Some(em) => {
                        for err in &checker.errors {
                            em.emit(&check_diag(err, p, diag::Severity::Error));
                        }
                        for w in &checker.warnings {
                            em.emit(&check_diag(w, p, diag::Severity::Warning));
                        }
                    }
                    None => {
                        for err in &checker.errors {
                            eprintln!("{}", err);
                        }
                        print_warnings(&checker.warnings);
                    }
                }
                let mod_env = check::ModuleEnv {
                    env: checker.env.clone(),
                    aliases: checker.aliases.clone(),
                    public_items: ast::collect_public_items(prog),
                };
                checked_modules.insert(p.clone(), mod_env);
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
            let mut resolver = resolver::Resolver::new();
            if let Err(e) = resolver.resolve_all(&path) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
            let folded = fold_comptime_all(&mut resolver);
            let mut checked_modules = std::collections::HashMap::new();
            let mut total_errors = 0;
            for p in &resolver.sorted_paths {
                let prog = resolver.files.get(p).unwrap();
                let mut checker = Checker::new();
                checker.errors = folded.get(p).cloned().unwrap_or_default();
                checker.demon = demon_mode;
                checker.checked_modules = checked_modules.clone();
                checker.check_program(prog, Some(p));
                if !checker.errors.is_empty() {
                    for err in &checker.errors {
                        eprintln!("{}", err);
                    }
                    total_errors += checker.errors.len();
                }
                print_warnings(&checker.warnings);
                let mod_env = check::ModuleEnv {
                    env: checker.env.clone(),
                    aliases: checker.aliases.clone(),
                    public_items: ast::collect_public_items(prog),
                };
                checked_modules.insert(p.clone(), mod_env);
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
            // Single-file JIT path: lex, parse, TYPE-CHECK, lower to Cranelift,
            // invoke `main`. Multi-file `use` imports are still deferred (issue
            // #15 slice 1); only one example in the corpus imports at all, and
            // it is outside the JIT subset.
            let mut parser = Parser::new(tokens);
            let mut program = match parser.parse_program() {
                Ok(p) => p,
                Err(e) => match emitter.take() {
                    Some(mut em) => {
                        let d = diag::Diagnostic::new(
                            diag::Kind::Parse,
                            diag::Severity::Error,
                            e.msg.clone(),
                        )
                        .file(&path)
                        .at(e.line, e.col);
                        em.emit(&d);
                        std::process::exit(em.finish(1));
                    }
                    None => { eprintln!("{}", e); std::process::exit(1); }
                },
            };
            // #505: fold `@comptime` before the checker, on this path too. The
            // single-file JIT route does not go through the resolver, so it
            // needs its own call — without one, `dmc jit` and `dmc run` would
            // disagree about a folded program, which is the exact defect the
            // directive's v1 closes.
            let comptime_errors = comptime::fold_program(&mut program);
            // #478 step 1: `dmc jit` type-checks, like `dmc run` already did.
            // It used to lower raw AST and rely on the JIT's own per-function
            // validation, so it accepted programs the checker rejects —
            // `fn main() -> i64 { let z: str = 5  1 }` compiled and ran. That
            // made `--check` necessary-but-not-sufficient in BOTH directions
            // and left the JIT unable to trust any property the checker
            // establishes, which is what a type-directed fix for #478 needs.
            // Measured at f8affbe: every corpus example passes `--check`, so
            // this rejects nothing that used to work.
            let mut checker = Checker::new();
            checker.errors = comptime_errors;
            checker.demon = demon_mode;
            checker.check_program(&program, Some(&path));
            if !checker.errors.is_empty() {
                if let Some(mut em) = emitter.take() {
                    for err in &checker.errors {
                        em.emit(&check_diag(err, &path, diag::Severity::Error));
                    }
                    std::process::exit(em.finish(1));
                }
                for err in &checker.errors {
                    eprintln!("{}", err);
                }
                eprintln!("\n⚠ {} type error(s) — refusing to run", checker.errors.len());
                std::process::exit(1);
            }
            match &mut emitter {
                Some(em) => {
                    for w in &checker.warnings {
                        em.emit(&check_diag(w, &path, diag::Severity::Warning));
                    }
                }
                None => print_warnings(&checker.warnings),
            }
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
            if let Err(e) = jit.compile_program(&program) {
                // The payload #485 exists for: an agent iterating a program
                // toward JIT eligibility reads `code` and `line`/`col`, not an
                // English sentence.
                if let Some(mut em) = emitter.take() {
                    em.emit(&jit_diag(&e, &path));
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
            let mut checked_modules = std::collections::HashMap::new();
            let mut total_errors = 0;
            for p in &resolver.sorted_paths {
                let prog = resolver.files.get(p).unwrap();
                let mut checker = Checker::new();
                checker.errors = folded.get(p).cloned().unwrap_or_default();
                checker.demon = demon_mode;
                checker.checked_modules = checked_modules.clone();
                checker.check_program(prog, Some(p));
                if !checker.errors.is_empty() {
                    for err in &checker.errors {
                        eprintln!("{}", err);
                    }
                    total_errors += checker.errors.len();
                }
                print_warnings(&checker.warnings);
                let mod_env = check::ModuleEnv {
                    env: checker.env.clone(),
                    aliases: checker.aliases.clone(),
                    public_items: ast::collect_public_items(prog),
                };
                checked_modules.insert(p.clone(), mod_env);
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
