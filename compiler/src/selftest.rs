//! `dmc selftest` — in-process interp-vs-JIT differential fuzzer (issue #408).
//!
//! A port of `tools/diff_fuzz.py` into the binary. Same job: generate well-typed
//! *scalar* demoniC programs (arithmetic, casts, control flow), run each through
//! the interpreter and the JIT, and assert the two backends agree. Any divergence
//! is a real bug (a miscompile, a missing guard, or a semantics mismatch).
//!
//! WHY IN-PROCESS. The Python tool shells out to `dmc run` and `dmc jit` — two
//! process spawns per program. This calls the interpreter (`Interpreter::run`)
//! and the JIT (`Jit::run_main_scalar`) directly, so a sweep is a single process
//! and floats come back structurally instead of being scraped from stdout.
//!
//! THE SCALAR REGIME. Generation stays bit-exact (small integers, nonzero
//! divisors, no `INT_MIN/-1`, additive loop accumulators) so interp and JIT must
//! agree *exactly* (#241). f32 generation is on by default since #473 made
//! scalar f32 real on both backends — that comparison is exact too, see
//! `floats_agree`. `--no-floats` restores the integer-only regime.
//!
//! VERDICT LATTICE (mirrors diff_fuzz.py / jit_probes.py):
//!   * both backends agree                         -> ok
//!   * run ok, jit emits a clean "not lowered"      -> gap  (SKIP)
//!   * run ok, jit ok, values DIFFER                -> DIVERGE (FAIL)
//!   * both agree it's an error                     -> both-fail (fine)
//!   * one side crashes/hangs/errs, the other is ok -> DIVERGE (FAIL)
//!
//! DETERMINISM. Every program is seeded from `--seed * 1_000_003 + i`; a failing
//! program prints its exact seed and source for a one-command repro
//! (`dmc selftest --repro <seed>`). `REGRESSION_SEEDS` pins the minimal historical
//! repros as literal source so a regression that reintroduces a fixed bug is
//! caught immediately (coverage-equivalence with the retired scalar gate).
//!
//! HANG SAFETY. A miscompile can turn a bounded loop into an infinite one (that
//! was a real #302 finding). Each program runs in a worker thread with a bounded
//! `recv_timeout`; a timeout is reported as a divergence. Everything non-`Send`
//! (the AST, the interpreter, the JIT) lives inside the worker — only the `Send`
//! verdict crosses back. A *native* JIT crash (SIGILL/SIGSEGV) still aborts the
//! process; that class is why `tools/diff_fuzz.py` (subprocess-isolated) is kept
//! alongside this — see `tools/diff_fuzz.py`.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crate::check::Checker;
use crate::interp::{Interpreter, Value};
use crate::jit::{Jit, ScalarRet};
use crate::lexer::Lexer;
use crate::parser::Parser;

// #480: a jit "gap" is a clean refusal to lower a feature, not a miscompile —
// and the JIT now SAYS which it is (`JitErrorKind`), so this no longer greps the
// message for English fragments. That guess was wrong in both directions: a
// clean refusal worded differently was scored as a divergence, and a genuine
// failure whose text happened to contain "unsupported" was downgraded to a gap.

/// Minimal historical repros — kept as literal source so a regression that
/// reintroduces a fixed bug is caught immediately. All are FIXED (interp == jit);
/// This is the coverage-equivalence anchor for retiring the scalar slice of
/// `dmc test examples`: every scalar bug the corpus once caught has a
/// pinned repro here.
pub const REGRESSION_SEEDS: &[(&str, &str)] = &[
    // --- manual-sweep parity bugs (control flow) ---
    (
        "continue_in_for (JIT hang)",
        "fn main() -> i64 {\n    let !s = 0\n    for i in 0..10 { if i == 3 { continue } s = s + i }\n    s\n}\n",
    ),
    (
        "non_exhaustive_match (JIT SIGILL)",
        "fn classify(x: i64) -> i64 {\n    match x { 0 => 10, 1 => 20, _ => 99 }\n}\nfn main() -> i64 { classify(1) + classify(7) }\n",
    ),
    (
        "match_dispatch (jump-table over a loop)",
        "fn main() -> i64 { let !s = 0  for k in 0..4 { s = s + match k { 0 => 10, 1 => 20, 2 => 30, _ => 40 } }  s }\n",
    ),
    // --- scalar corners the generator deliberately avoids (div/0, INT_MIN,
    // overflow, shifts): both backends must agree on the guarded/wrapping
    // result. These re-cover the pure-scalar slice of tools/jit_probes.py so
    // selftest stands in for that battery's scalar cases. ---
    (
        "int_div_round_toward_zero",
        "fn main() -> i64 { (0 - 7) / 2 }\n",
    ),
    (
        "int_mod_sign_follows_dividend",
        "fn main() -> i64 { (0 - 7) % 3 }\n",
    ),
    (
        "int_div_by_zero_guard (0, not SIGFPE)",
        "fn main() -> i64 { let !z = 0  10 / z }\n",
    ),
    (
        "int_shl_max_in_range",
        "fn main() -> i64 { let !n = 63  1 << n }\n",
    ),
    (
        "int_add_overflow_wraps",
        "fn main() -> i64 { let a: i64 = 9223372036854775807  a + 1 }\n",
    ),
    (
        "int_sub_overflow_wraps",
        "fn main() -> i64 { let a: i64 = -9223372036854775807 - 1  a - 1 }\n",
    ),
    (
        "cast_scalar_f32_rounds",
        "fn main() -> f64 { (0.1 as f32) as f64 }\n",
    ),
];

/// Configuration for a `dmc selftest` run (parsed from CLI flags in `main`).
pub struct Config {
    pub iters: usize,
    pub seed: u64,
    pub timeout: Duration,
    pub floats: bool,
    pub verbose: bool,
    pub repro: Option<u64>,
    pub meta_test: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            iters: 500,
            seed: 1,
            timeout: Duration::from_secs(10),
            // #473: f32 generation is ON by default. It was opt-in while scalar
            // f32 was known to diverge and could only be compared tolerantly;
            // now it is exact on both backends, so the default sweep should
            // gate it. `--no-floats` restores the integer-only regime.
            floats: true,
            verbose: false,
            repro: None,
            meta_test: false,
        }
    }
}

// ----------------------------------------------------------------------------
// Deterministic PRNG (SplitMix64). We do not — and need not — reproduce Python's
// Mersenne Twister sequence: determinism means "same seed → same program" within
// this binary, and the historical coverage is pinned by REGRESSION_SEEDS' literal
// source, not by seed-derived generation.
// ----------------------------------------------------------------------------
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Inclusive integer range [lo, hi].
    fn randint(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo + 1) as u64;
        lo + (self.next_u64() % span) as i64
    }

    /// Uniform float in [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.random()
    }

    fn choice<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[(self.next_u64() as usize) % xs.len()]
    }
}

// ----------------------------------------------------------------------------
// Type-directed generator — a direct port of diff_fuzz.py's `Gen`.
// ----------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq)]
enum Ty {
    I64,
    F32,
    Bool,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::I64 => "i64",
            Ty::F32 => "f32",
            Ty::Bool => "bool",
        }
    }
}

struct Gen {
    rng: Rng,
    floats: bool,
    vars: Vec<(String, Ty)>,
    n: u32,
}

impl Gen {
    fn new(seed: u64, floats: bool) -> Self {
        Gen { rng: Rng::new(seed), floats, vars: Vec::new(), n: 0 }
    }

    fn types(&self) -> &'static [Ty] {
        if self.floats {
            &[Ty::I64, Ty::F32, Ty::Bool]
        } else {
            &[Ty::I64, Ty::Bool]
        }
    }

    fn fresh(&mut self) -> String {
        self.n += 1;
        format!("v{}", self.n)
    }

    fn lit(&mut self, typ: Ty) -> String {
        match typ {
            Ty::I64 => self.rng.randint(-50, 50).to_string(),
            Ty::F32 => format!("{:.3}f32", self.rng.uniform(-10.0, 10.0)),
            Ty::Bool => (*self.rng.choice(&["true", "false"])).to_string(),
        }
    }

    fn var_of(&mut self, typ: Ty) -> Option<String> {
        let cand: Vec<&String> =
            self.vars.iter().filter(|(_, t)| *t == typ).map(|(n, _)| n).collect();
        if cand.is_empty() {
            None
        } else {
            Some((*self.rng.choice(&cand)).clone())
        }
    }

    fn expr(&mut self, typ: Ty, depth: i32) -> String {
        // Leaf: literal or in-scope variable.
        if depth <= 0 || self.rng.random() < 0.3 {
            if let Some(v) = self.var_of(typ) {
                if self.rng.random() < 0.5 {
                    return v;
                }
            }
            return self.lit(typ);
        }
        match typ {
            Ty::I64 => self.i64_expr(depth),
            Ty::F32 => self.f32_expr(depth),
            Ty::Bool => self.bool_expr(depth),
        }
    }

    fn i64_expr(&mut self, depth: i32) -> String {
        let k = self.rng.random();
        if k < 0.4 {
            let op = *self.rng.choice(&["+", "-", "*"]);
            return format!("({} {} {})", self.expr(Ty::I64, depth - 1), op, self.expr(Ty::I64, depth - 1));
        }
        if k < 0.55 {
            // divide / modulo by a nonzero positive constant (avoids /0 and the
            // INT_MIN/-1 SIGFPE, both tracked separately).
            let op = *self.rng.choice(&["/", "%"]);
            let d = self.rng.randint(1, 9);
            return format!("({} {} {})", self.expr(Ty::I64, depth - 1), op, d);
        }
        if k < 0.7 && self.floats {
            return format!("(({}) as i64)", self.expr(Ty::F32, depth - 1));
        }
        if k < 0.8 {
            return format!("(if {} {{ 1 }} else {{ 0 }})", self.expr(Ty::Bool, depth - 1));
        }
        format!(
            "(if {} {{ {} }} else {{ {} }})",
            self.expr(Ty::Bool, depth - 1),
            self.expr(Ty::I64, depth - 1),
            self.expr(Ty::I64, depth - 1)
        )
    }

    fn f32_expr(&mut self, depth: i32) -> String {
        let k = self.rng.random();
        if k < 0.45 {
            let op = *self.rng.choice(&["+", "-", "*"]);
            return format!("({} {} {})", self.expr(Ty::F32, depth - 1), op, self.expr(Ty::F32, depth - 1));
        }
        if k < 0.6 {
            let d = self.rng.uniform(1.0, 9.0);
            return format!("({} / {:.3}f32)", self.expr(Ty::F32, depth - 1), d);
        }
        if k < 0.75 {
            return format!("(({}) as f32)", self.expr(Ty::I64, depth - 1));
        }
        format!(
            "(if {} {{ {} }} else {{ {} }})",
            self.expr(Ty::Bool, depth - 1),
            self.expr(Ty::F32, depth - 1),
            self.expr(Ty::F32, depth - 1)
        )
    }

    fn bool_expr(&mut self, depth: i32) -> String {
        let k = self.rng.random();
        if k < 0.45 {
            let t = if self.floats {
                *self.rng.choice(&[Ty::I64, Ty::F32])
            } else {
                Ty::I64
            };
            // f32 uses only ordering comparisons; i64 uses all six.
            let op = if t == Ty::F32 {
                *self.rng.choice(&["<", "<=", ">", ">="])
            } else {
                *self.rng.choice(&["<", "<=", ">", ">=", "==", "!="])
            };
            return format!("({} {} {})", self.expr(t, depth - 1), op, self.expr(t, depth - 1));
        }
        if k < 0.7 {
            let op = *self.rng.choice(&["&&", "||"]);
            return format!("({} {} {})", self.expr(Ty::Bool, depth - 1), op, self.expr(Ty::Bool, depth - 1));
        }
        format!("(!{})", self.expr(Ty::Bool, depth - 1))
    }

    fn program(&mut self) -> String {
        let ret = *self.rng.choice(self.types());
        let mut lines: Vec<String> = Vec::new();
        // A few immutable let-bindings to give expressions some variables.
        let n_lets = self.rng.randint(0, 4);
        for _ in 0..n_lets {
            let t = *self.rng.choice(self.types());
            let name = self.fresh();
            let e = self.expr(t, 2);
            lines.push(format!("    let {} = {}", name, e));
            self.vars.push((name, t));
        }
        // Sometimes a bounded for-loop with an additive accumulator (control flow
        // + mutation, the class the continue-in-for hang lived in). i64/f32 only.
        let tail = if (ret == Ty::I64 || ret == Ty::F32) && self.rng.random() < 0.5 {
            let acc = self.fresh();
            let init = if ret == Ty::I64 { "0" } else { "0.0f32" };
            let n = self.rng.randint(1, 12);
            let body_term = self.expr(ret, 1);
            let idx_term = if ret == Ty::I64 { "i" } else { "(i as f32)" };
            lines.push(format!("    let !{} = {}", acc, init));
            lines.push(format!(
                "    for i in 0..{} {{ {} = {} + {} + {} }}",
                n, acc, acc, idx_term, body_term
            ));
            self.vars.push((acc.clone(), ret));
            acc
        } else {
            self.expr(ret, 3)
        };
        let body = lines.join("\n");
        let sep = if body.is_empty() { "" } else { "\n" };
        format!("fn main() -> {} {{\n{}{}    {}\n}}\n", ret.name(), body, sep, tail)
    }
}

// ----------------------------------------------------------------------------
// Differential classification.
// ----------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Ok,
    Gap,
    BothFail,
    Diverge,
}

impl Verdict {
    fn tag(self) -> &'static str {
        match self {
            Verdict::Ok => "ok",
            Verdict::Gap => "gap",
            Verdict::BothFail => "both-fail",
            Verdict::Diverge => "DIVERGE",
        }
    }
}

/// Interpreter outcome for one program.
enum Interp {
    Ok(Value),
    Err(String),
}

/// JIT outcome for one program.
enum JitOut {
    Ok(ScalarRet),
    Gap(String),
    Err(String),
}

/// Exact float agreement, NaN included (NaN != NaN under `==`, but the two
/// backends producing NaN *is* agreement).
///
/// #473 made this exact. It used to be a tolerant `|a-b| <= 1e-4 + 1e-3*|b|`
/// compare, on the reasoning that scalar f32 was computed at different widths
/// on the two backends and could only be expected to agree to ~7 digits. That
/// reasoning no longer holds: an `f32` scalar is f32 on both sides, and the
/// interpreter's round-through-f32 is bit-exact with the JIT's native f32 for
/// `+ - * /` (#241's double-rounding property). A tolerant compare here is
/// what let the #473 divergence — up to 1e-8 relative, three orders of
/// magnitude inside the old window — run green through 500 iterations a sweep.
fn floats_agree(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

fn agree(v: &Value, r: &ScalarRet) -> bool {
    match (v, r) {
        (Value::Int(a), ScalarRet::I64(b)) => a == b,
        (Value::Bool(a), ScalarRet::Bool(b)) => a == b,
        (Value::Nil, ScalarRet::Nil) => true,
        (Value::Float(a, _), ScalarRet::F32(b)) => floats_agree(*a, *b as f64),
        (Value::Float(a, _), ScalarRet::F64(b)) => floats_agree(*a, *b),
        _ => false,
    }
}

fn show_ret(r: &ScalarRet) -> String {
    match r {
        ScalarRet::Nil => "nil".into(),
        ScalarRet::I64(n) => n.to_string(),
        ScalarRet::Bool(b) => b.to_string(),
        ScalarRet::F32(x) => x.to_string(),
        ScalarRet::F64(x) => x.to_string(),
    }
}

fn panic_message(e: &(dyn Any + Send)) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Run the interpreter on a single self-contained program. Mirrors `dmc run`:
/// the type-checker runs first and a type error is a run-time failure. A Rust
/// panic (an interpreter bug) is caught and reported as an error outcome.
fn run_interp(program: &crate::ast::Program) -> Interp {
    let res = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut checker = Checker::new();
        checker.check_program(program, None);
        if !checker.errors.is_empty() {
            return Interp::Err(format!("check: {}", checker.errors[0].msg));
        }
        match Interpreter::new().run(program, None) {
            Ok(v) => Interp::Ok(v),
            Err(e) => Interp::Err(e.msg),
        }
    }));
    res.unwrap_or_else(|e| Interp::Err(format!("interp panic: {}", panic_message(&*e))))
}

/// JIT-compile and run a single program. Mirrors `dmc jit`: no pre-check, the
/// JIT does its own per-fn validation. A clean "not lowered" compile error is a
/// gap; a Rust panic is caught and reported as an error.
fn run_jit(program: &crate::ast::Program) -> JitOut {
    let res = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut jit = match Jit::new() {
            Ok(j) => j,
            Err(e) => return JitOut::Err(e.msg),
        };
        if let Err(e) = jit.compile_program(program) {
            let is_gap = e.kind == crate::jit::JitErrorKind::Unsupported;
            return if is_gap { JitOut::Gap(e.msg) } else { JitOut::Err(e.msg) };
        }
        match jit.run_main_scalar() {
            Ok(r) => JitOut::Ok(r),
            Err(e) => {
                let is_gap = e.kind == crate::jit::JitErrorKind::Unsupported;
                if is_gap { JitOut::Gap(e.msg) } else { JitOut::Err(e.msg) }
            }
        }
    }));
    res.unwrap_or_else(|e| JitOut::Err(format!("jit panic: {}", panic_message(&*e))))
}

/// Classify one program — parse, run both backends, combine per the lattice.
/// Runs entirely inside a worker thread (see `classify`).
fn eval_program(source: &str) -> (Verdict, String) {
    let tokens = match Lexer::new(source).tokenize() {
        Ok(t) => t,
        Err(e) => return (Verdict::BothFail, format!("lex: {}", e)),
    };
    let program = match Parser::new(tokens).parse_program() {
        Ok(p) => p,
        Err(e) => return (Verdict::BothFail, format!("parse: {}", e)),
    };

    let i = run_interp(&program);
    let j = run_jit(&program);

    match (i, j) {
        (Interp::Ok(v), JitOut::Ok(r)) => {
            if agree(&v, &r) {
                (Verdict::Ok, format!("{:?}", v))
            } else {
                (Verdict::Diverge, format!("run={:?} jit={}", v, show_ret(&r)))
            }
        }
        (Interp::Ok(_), JitOut::Gap(m)) => (Verdict::Gap, truncate(&m, 80)),
        (Interp::Ok(v), JitOut::Err(m)) => {
            (Verdict::Diverge, format!("one-sided: run=ok:{:?} jit=err:{}", v, truncate(&m, 120)))
        }
        (Interp::Err(_), JitOut::Err(_)) | (Interp::Err(_), JitOut::Gap(_)) => {
            (Verdict::BothFail, String::new())
        }
        (Interp::Err(m), JitOut::Ok(r)) => {
            (Verdict::Diverge, format!("one-sided: run=err:{} jit=ok:{}", truncate(&m, 120), show_ret(&r)))
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    let one_line: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    if one_line.len() <= n {
        one_line
    } else {
        one_line.chars().take(n).collect()
    }
}

/// Classify one program with a hang-safe bounded wait. Everything non-`Send`
/// stays inside the worker; only the `(Verdict, String)` result crosses back.
/// On timeout the worker is detached (it runs until process exit) and the
/// program is reported as a divergence.
fn classify(source: String, timeout: Duration) -> (Verdict, String) {
    let (tx, rx) = mpsc::channel();
    let worker = thread::Builder::new()
        .name("dmc-selftest-worker".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let out = eval_program(&source);
            let _ = tx.send(out);
        })
        .expect("failed to spawn selftest worker");

    match rx.recv_timeout(timeout) {
        Ok(v) => {
            let _ = worker.join();
            v
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            // Detach the (likely hung) worker; it will be reclaimed at process
            // exit. A timeout is itself the finding.
            (Verdict::Diverge, "timeout (possible interp/JIT hang)".into())
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            (Verdict::Diverge, "worker died before producing a verdict".into())
        }
    }
}

// ----------------------------------------------------------------------------
// Entry point.
// ----------------------------------------------------------------------------
pub fn run(cfg: &Config) -> i32 {
    // We capture Rust panics ourselves (per-backend) and fold them into the
    // verdict; silence the default hook so a caught panic doesn't spam a
    // backtrace for every buggy program.
    panic::set_hook(Box::new(|_| {}));

    if let Some(seed) = cfg.repro {
        let src = Gen::new(seed, cfg.floats).program();
        print!("{}", src);
        let (verdict, detail) = classify(src, cfg.timeout);
        println!("selftest: seed {}: {} {}", seed, verdict.tag(), detail);
        return if verdict == Verdict::Diverge { 1 } else { 0 };
    }

    if cfg.meta_test && !meta_test() {
        return 1;
    }

    let mut counts = [0usize; 4]; // ok, gap, both-fail, DIVERGE
    let mut diverged: Vec<(String, String, String)> = Vec::new();

    // Always run the pinned historical repros first.
    for (label, src) in REGRESSION_SEEDS {
        let (verdict, detail) = classify((*src).to_string(), cfg.timeout);
        if verdict == Verdict::Diverge {
            diverged.push((format!("regression:{}", label), (*src).to_string(), detail));
        } else if cfg.verbose {
            println!("selftest: regression {}: {} {}", label, verdict.tag(), detail);
        }
    }

    for idx in 0..cfg.iters {
        let seed = cfg.seed.wrapping_mul(1_000_003).wrapping_add(idx as u64);
        let src = Gen::new(seed, cfg.floats).program();
        let (verdict, detail) = classify(src.clone(), cfg.timeout);
        counts[verdict_index(verdict)] += 1;
        if verdict == Verdict::Diverge {
            diverged.push((format!("seed {}", seed), src, detail));
        } else if cfg.verbose {
            println!("selftest: seed {}: {} {}", seed, verdict.tag(), detail);
        }
    }

    println!(
        "selftest: {} ok, {} jit-gap, {} both-fail, {} divergence(s) over {} iters (seed base {})",
        counts[0], counts[1], counts[2], diverged.len(), cfg.iters, cfg.seed
    );
    for (tag, src, detail) in &diverged {
        eprintln!(
            "\nselftest: DIVERGENCE [{}]: {}\n--- repro (dmc selftest --repro <seed>) ---\n{}",
            tag, detail, src
        );
    }
    if diverged.is_empty() { 0 } else { 1 }
}

fn verdict_index(v: Verdict) -> usize {
    match v {
        Verdict::Ok => 0,
        Verdict::Gap => 1,
        Verdict::BothFail => 2,
        Verdict::Diverge => 3,
    }
}

/// Prove the differ has teeth: the comparison must flag values that differ and
/// accept values that match. A vacuously-green suite is the failure mode this
/// guards against.
fn meta_test() -> bool {
    let mut ok = true;
    if agree(&Value::Int(1), &ScalarRet::I64(2)) {
        eprintln!("selftest: meta-test: error: differ failed to flag i64 1 vs 2");
        ok = false;
    }
    if floats_agree(1.0, 2.0) {
        eprintln!("selftest: meta-test: error: differ failed to flag float 1.0 vs 2.0");
        ok = false;
    }
    // #473: the float compare is EXACT, so a one-ulp gap must be flagged. Under
    // the old tolerant compare this passed, which is why the divergence hid.
    if floats_agree(0.1_f32 as f64, 0.1_f64) {
        eprintln!("selftest: meta-test: error: differ failed to flag an f32/f64 ulp gap");
        ok = false;
    }
    if !floats_agree(f64::NAN, f64::NAN) {
        eprintln!("selftest: meta-test: error: differ failed to accept NaN vs NaN");
        ok = false;
    }
    if !agree(&Value::Int(5), &ScalarRet::I64(5)) {
        eprintln!("selftest: meta-test: error: differ failed to accept i64 5 vs 5");
        ok = false;
    }
    if !agree(&Value::Bool(true), &ScalarRet::Bool(true)) {
        eprintln!("selftest: meta-test: error: differ failed to accept bool true vs true");
        ok = false;
    }
    if ok {
        println!("selftest: meta-test: ok — divergent values flagged, matching values accepted");
    }
    ok
}

#[cfg(test)]
#[path = "selftest_tests.rs"]
mod selftest_tests;
