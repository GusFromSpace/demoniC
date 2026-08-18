/// demoniC compiler entry point — bootstrap phase
///
/// Phase 1: Lexer ✓
/// Phase 2: Parser / AST ✓
/// Phase 3: Type-checker ✓ (pre-alpha)
/// Phase 3.5: Tree-walking interpreter ✓ (pre-alpha)
/// Phase 4: Metal backend (MLX where applicable)

mod lexer;
mod ast;
mod parser;
mod desugar;
mod shape;
mod types;
mod check;
mod interp;
mod fmt;
mod resolver;
mod jit;
mod selftest;
// #326 GPU/Metal backend — compiled only on macOS with `--features gpu`.
#[cfg(all(target_os = "macos", feature = "gpu"))]
mod gpu;
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
#[path = "interp_tests.rs"]
mod interp_tests;
#[cfg(test)]
#[path = "fmt_tests.rs"]
mod fmt_tests;

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
    eprintln!("  dmc --profile <file.dmc>          run with op-count profiling (emits summary to stderr)");
    eprintln!("  dmc --profile run <file.dmc>      run mode with profiling");
    eprintln!("  dmc --demon <file.dmc>            demon mode: release the safe-mode lints (raw, no guardrails)");
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

fn run_tests(path: &Path, profile_mode: bool, jit_test: bool) -> i32 {
    let mut files = Vec::new();
    if let Err(e) = collect_dmc_files(path, &mut files) {
        eprintln!("{}", e);
        return 1;
    }
    files.sort();
    if files.is_empty() {
        eprintln!("no .dmc files found under {:?}", path);
        return 1;
    }

    let mut total = 0usize;
    let mut failed = 0usize;
    // #: `--jit` parity counters. Each test_* is additionally run under the JIT;
    // a file the JIT can't compile is skipped (outside its subset), not failed.
    let mut jit_ran = 0usize;
    let mut jit_skipped = 0usize;
    for file in files {
        let canonical_file = match file.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                failed += 1;
                eprintln!("FAIL {}: canonicalization failed: {}", file.display(), e);
                continue;
            }
        };
        let mut resolver = resolver::Resolver::new();
        if let Err(e) = resolver.resolve_all(&canonical_file) {
            failed += 1;
            eprintln!("FAIL {}: resolution failed: {}", file.display(), e);
            continue;
        }

        let program = match resolver.files.get(&canonical_file) {
            Some(prog) => prog.clone(),
            None => {
                failed += 1;
                eprintln!("FAIL {}: program not found in resolved files", file.display());
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
            checker.checked_modules = checked_modules.clone();
            checker.check_program(prog, Some(p));
            if !checker.errors.is_empty() {
                failed += 1;
                eprintln!("FAIL {}: type check failed in dependency {:?}", file.display(), p);
                for err in &checker.errors {
                    eprintln!("{}", err);
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
            let fn_decl = program.items.iter().find_map(|item| find_fn_decl(item, &name));
            if let Some(f) = fn_decl {
                if !f.params.is_empty() {
                    failed += 1;
                    eprintln!("FAIL {}: test function must take zero args", label);
                    continue;
                }
            }

            let mut interp = Interpreter::new();
            if profile_mode { interp.enable_profile(); }

            let mut interp_failed = false;
            for p in &resolver.sorted_paths {
                let prog = resolver.files.get(p).unwrap();
                if let Err(e) = interp.load_program(prog, Some(p)) {
                    failed += 1;
                    eprintln!("FAIL {}: interpreter load failed in dependency {:?}: {}", file.display(), p, e);
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
            match result {
                Ok(interp::Value::Bool(false)) => {
                    failed += 1;
                    eprintln!("FAIL {}: returned false", label);
                }
                Ok(interp::Value::Bool(true)) | Ok(interp::Value::Nil) => {
                    println!("ok   {}", label);
                    if profile_mode {
                        if let Some(p) = &interp.profile { print_profile(p); }
                    }
                }
                Ok(v) => {
                    failed += 1;
                    eprintln!(
                        "FAIL {}: returned {:?}; tests must return bool or nil",
                        label, v,
                    );
                }
                Err(e) => {
                    failed += 1;
                    eprintln!("FAIL {}: {}", label, e);
                }
            }

            // `--jit`: run the same test under the JIT for parity. A compile
            // error means the program is outside the JIT's subset → skip (the
            // JIT can't run it). A test that compiles but returns false or errors
            // is a real interp/JIT divergence → fail.
            if jit_test {
                match jit::Jit::new() {
                    Err(e) => { failed += 1; eprintln!("FAIL {} [jit]: jit init: {}", label, e.msg); }
                    Ok(mut j) => match j.compile_program(&program) {
                        Err(_) => { jit_skipped += 1; }
                        Ok(()) => {
                            jit_ran += 1;
                            match j.run_test_fn(&name) {
                                Ok(true) => {}
                                Ok(false) => {
                                    failed += 1;
                                    eprintln!("FAIL {} [jit]: returned false (diverges from interp)", label);
                                }
                                Err(e) => {
                                    failed += 1;
                                    eprintln!("FAIL {} [jit]: {}", label, e.msg);
                                }
                            }
                        }
                    },
                }
            }
        }
    }

    if total == 0 {
        println!("0 tests");
        return 0;
    }
    let jit_note = if jit_test {
        format!(" | jit parity: {} ran, {} skipped (outside JIT subset)", jit_ran, jit_skipped)
    } else {
        String::new()
    };
    if failed == 0 {
        println!("test result: ok. {} passed; 0 failed{}", total, jit_note);
        0
    } else {
        eprintln!(
            "test result: FAILED. {} passed; {} failed{}",
            total.saturating_sub(failed),
            failed,
            jit_note,
        );
        1
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
    let args: Vec<String> = raw_args.into_iter().enumerate()
        .filter_map(|(i, a)| {
            if i > 0 && a == "--profile" { profile_mode = true; None }
            else if i > 0 && a == "--demon" { demon_mode = true; None }
            else if i > 0 && a == "--jit" { jit_test = true; None }
            else if i > 0 && a == "--gpu" { gpu_mode = true; None }
            // #400: `--vault=`/`--forge=` are documented arena-sizing flags, but
            // sizing is unimplemented and they were silently forwarded into the
            // program's argv (swallowed AND inert). Reject them loudly rather than
            // leak a `dmc` flag into the program. (Arenas currently size
            // dynamically; drop the flag.)
            else if i > 0 && (a == "--vault" || a == "--forge"
                              || a.starts_with("--vault=") || a.starts_with("--forge=")) {
                eprintln!("error: `{}` — arena sizing flags are specified but not yet \
                           implemented; remove it (arenas size dynamically today)", a);
                std::process::exit(1);
            }
            else { Some(a) }
        })
        .collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
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
            if args.len() < 3 { eprintln!("missing file"); std::process::exit(1); }
            ("check", PathBuf::from(&args[2]))
        }
        "run" => {
            if args.len() < 3 { eprintln!("missing file"); std::process::exit(1); }
            ("run", PathBuf::from(&args[2]))
        }
        "jit" => {
            if args.len() < 3 { eprintln!("missing file"); std::process::exit(1); }
            ("jit", PathBuf::from(&args[2]))
        }
        "test" => {
            if args.len() < 3 { eprintln!("missing file or directory"); std::process::exit(1); }
            let code = run_tests(Path::new(&args[2]), profile_mode, jit_test);
            std::process::exit(code);
        }
        "selftest" => {
            // In-process interp-vs-JIT differential fuzzer (#408). Generates
            // well-typed scalar programs and asserts the two backends agree.
            let code = run_selftest(&args[2..]);
            std::process::exit(code);
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
        Err(e) => {
            eprintln!("error reading {:?}: {}", path, e);
            std::process::exit(1);
        }
    };

    let path = match path.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error resolving path {:?}: {}", path, e);
            std::process::exit(1);
        }
    };

    let tokens = match Lexer::new(&src).tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("lex error in {:?}: {}", path, e);
            std::process::exit(1);
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
                eprintln!("{}", e);
                std::process::exit(1);
            }
            let mut checked_modules = std::collections::HashMap::new();
            let mut total_errors = 0;
            let mut items_count = 0;
            for p in &resolver.sorted_paths {
                let prog = resolver.files.get(p).unwrap();
                items_count += prog.items.len();
                let mut checker = Checker::new();
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
            let mut checked_modules = std::collections::HashMap::new();
            let mut total_errors = 0;
            for p in &resolver.sorted_paths {
                let prog = resolver.files.get(p).unwrap();
                let mut checker = Checker::new();
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
            // Single-file JIT path: lex, parse, lower to Cranelift, invoke
            // `main`. Slice 1 of issue #15 — multi-file `use` imports and
            // pre-JIT type checking are deferred (the JIT does its own
            // per-fn validation at lowering time, with diagnostics).
            let mut parser = Parser::new(tokens);
            let program = match parser.parse_program() {
                Ok(p) => p,
                Err(e) => { eprintln!("{}", e); std::process::exit(1); }
            };
            let mut jit = match Jit::new() {
                Ok(j) => j,
                Err(e) => { eprintln!("{}", e); std::process::exit(1); }
            };
            if gpu_mode {
                // #326: honored only on a macOS `--features gpu` build; otherwise
                // the GPU symbol is absent and this stays on the CPU kernel.
                #[cfg(all(target_os = "macos", feature = "gpu"))]
                jit.set_gpu(true);
                #[cfg(not(all(target_os = "macos", feature = "gpu")))]
                eprintln!("warning: --gpu ignored (build with `--features gpu` on macOS to enable)");
            }
            if let Err(e) = jit.compile_program(&program) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
            match jit.run_main() {
                Ok(rc) => {
                    // #271: print REPL feedback by main's *declared return type*,
                    // not by `rc != 0` — the old guard swallowed a real `0`/`false`
                    // return (run_main also returns 0 for a nil main). Float mains
                    // already print inside run_main; nil prints nothing.
                    let ret = program.items.iter()
                        .find_map(|it| find_fn_decl(it, "main"))
                        .and_then(|f| f.ret_type.as_ref());
                    match ret {
                        None => {} // unannotated main = nil
                        Some(ast::Type::Scalar(ast::ScalarType::Nil, _))
                        | Some(ast::Type::Scalar(ast::ScalarType::F32, _))
                        | Some(ast::Type::Scalar(ast::ScalarType::F64, _)) => {}
                        Some(ast::Type::Scalar(ast::ScalarType::Bool, _)) => {
                            println!("=> {}", rc != 0);
                        }
                        Some(_) => println!("=> {}", rc),
                    }
                }
                Err(e) => { eprintln!("{}", e); std::process::exit(1); }
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
            let mut checked_modules = std::collections::HashMap::new();
            let mut total_errors = 0;
            for p in &resolver.sorted_paths {
                let prog = resolver.files.get(p).unwrap();
                let mut checker = Checker::new();
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
