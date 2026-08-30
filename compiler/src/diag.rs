//! `--json` structured diagnostics (#485), schema 1.
//!
//! A second *encoding* of the diagnostics the human renderer already prints —
//! not a second diagnostic vocabulary. `SPEC.md §8` is unchanged, the set of
//! diagnostics is unchanged, and the exit code is unchanged; only the format
//! moves. Without `--json` nothing in this module runs.
//!
//! ## Wire format
//!
//! JSON Lines on **stderr**: one self-describing object per line, terminated by
//! exactly one `"kind": "summary"` object. Line-delimited rather than one
//! document because `dmc test` is streaming — a result per test as it runs —
//! and a consumer that reads a line at a time handles every command the same
//! way. `stdout` keeps whatever the *program* writes; the diagnostics channel
//! never shares it.
//!
//! Every object carries `"schema": 1` — a line torn out of a log is still
//! interpretable — and a `"kind"` discriminator. Field order is fixed, so a
//! test can assert a whole line.
//!
//! ## Kinds in schema 1
//!
//! | kind | what it is |
//! | --- | --- |
//! | `lex` | a lexer diagnostic |
//! | `parse` | a parser diagnostic |
//! | `check` | a type/shape checker diagnostic (`severity` error or warning) |
//! | `jit-ineligible` | the JIT declines to lower a construct; always has `code` |
//! | `jit-error` | the JIT failed on something it is supposed to handle (#480) |
//! | `cli` | a command-line usage error |
//! | `test` | one `dmc test` result; under `--jit` also the parity verdict |
//! | `unstructured` | a diagnostic whose category has no schema yet |
//! | `summary` | the terminal object; exactly one per run |
//!
//! `runtime` is **reserved** for a later slice. Until it exists, diagnostics
//! from that category are emitted as `unstructured` — the prose is preserved
//! verbatim, nothing is dropped, and the stream stays parseable. Adding a kind
//! is additive; consumers must ignore kinds they do not know.

use std::fmt::Write as _;

use crate::shape::SymDim;

/// The schema version, emitted on every object. Bumped only for a *breaking*
/// change: adding a kind, a code, or an optional field is not one.
pub const SCHEMA: u32 = 1;

/// A diagnostic's category — the `"kind"` discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Lex,
    Parse,
    Check,
    JitIneligible,
    JitError,
    Cli,
    /// A category with no schema yet (resolver prose, runtime errors, test
    /// output). Carries `message` only.
    Unstructured,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Lex => "lex",
            Kind::Parse => "parse",
            Kind::Check => "check",
            Kind::JitIneligible => "jit-ineligible",
            Kind::JitError => "jit-error",
            Kind::Cli => "cli",
            Kind::Unstructured => "unstructured",
        }
    }
}

/// `"severity"`. Only `warning` is non-fatal; `error` always contributes to a
/// nonzero exit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }
}

/// One diagnostic, ready to encode. Optional fields are *omitted* rather than
/// nulled: `code` appears only where the diagnostic has one, and the location
/// fields only where the diagnostic is located.
pub struct Diagnostic {
    pub kind: Kind,
    pub severity: Severity,
    pub code: Option<String>,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<usize>,
    pub col: Option<usize>,
    /// Byte offsets into the file, when the diagnostic carries a full span.
    pub start: Option<usize>,
    pub end: Option<usize>,
    pub hint: Option<String>,
    /// For a shape error: the two shapes, as pre-encoded JSON arrays
    /// (`shape_array`), so a consumer reads dims as data rather than re-parsing
    /// the rendered `Tensor[f32, [2, 3]]` out of `message`.
    pub expected: Option<String>,
    pub actual: Option<String>,
}

impl Diagnostic {
    pub fn new(kind: Kind, severity: Severity, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            kind,
            severity,
            code: None,
            message: message.into(),
            file: None,
            line: None,
            col: None,
            start: None,
            end: None,
            hint: None,
            expected: None,
            actual: None,
        }
    }

    /// A diagnostic in a category schema 1 does not model yet. Preserves the
    /// human text verbatim so `--json` never loses information.
    pub fn unstructured(message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(Kind::Unstructured, Severity::Error, message)
    }

    pub fn code(mut self, code: impl Into<String>) -> Diagnostic {
        self.code = Some(code.into());
        self
    }

    pub fn file(mut self, path: &std::path::Path) -> Diagnostic {
        self.file = Some(path.display().to_string());
        self
    }

    pub fn at(mut self, line: usize, col: usize) -> Diagnostic {
        self.line = Some(line);
        self.col = Some(col);
        self
    }

    pub fn bytes(mut self, start: usize, end: usize) -> Diagnostic {
        self.start = Some(start);
        self.end = Some(end);
        self
    }

    pub fn hint(mut self, hint: Option<&str>) -> Diagnostic {
        self.hint = hint.map(|h| h.to_string());
        self
    }

    /// Attach a shape error's two shapes as structured data.
    pub fn shapes(mut self, expected: &[SymDim], actual: &[SymDim]) -> Diagnostic {
        self.expected = Some(shape_array(expected));
        self.actual = Some(shape_array(actual));
        self
    }

    /// The JSON Lines encoding of this diagnostic — one line, no trailing
    /// newline. Key order is fixed and part of the format.
    pub fn to_json(&self) -> String {
        let mut o = Obj::new();
        o.num("schema", SCHEMA as u64);
        o.str("kind", self.kind.as_str());
        o.str("severity", self.severity.as_str());
        if let Some(c) = &self.code { o.str("code", c); }
        o.str("message", &self.message);
        if let Some(e) = &self.expected { o.raw("expected", e); }
        if let Some(a) = &self.actual { o.raw("actual", a); }
        if let Some(f) = &self.file { o.str("file", f); }
        if let Some(l) = self.line { o.num("line", l as u64); }
        if let Some(c) = self.col { o.num("col", c as u64); }
        if let Some(s) = self.start { o.num("start", s as u64); }
        if let Some(e) = self.end { o.num("end", e as u64); }
        if let Some(h) = &self.hint { o.str("hint", h); }
        o.finish()
    }
}

/// A shape as a JSON array: constant dims are numbers, symbolic dims (`n`,
/// `~`, `_`, arithmetic over parameters) are their `SPEC.md §8` rendering as
/// strings. `[2, "n"]` — data a consumer indexes, not prose it splits.
pub fn shape_array(dims: &[SymDim]) -> String {
    let mut out = String::from("[");
    for (i, d) in dims.iter().enumerate() {
        if i > 0 { out.push(','); }
        match d {
            SymDim::Const(n) => { write!(out, "{}", n).unwrap(); }
            other => out.push_str(&escape(&other.to_string())),
        }
    }
    out.push(']');
    out
}

/// The interp-vs-JIT verdict a `--jit` run attaches to each test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitVerdict {
    /// Compiled and agreed with the interpreter.
    Pass,
    /// Compiled but diverged (returned false, trapped, or blew `--forge`).
    Fail,
    /// The file is outside the JIT subset — not a failure.
    Skip,
}

impl JitVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            JitVerdict::Pass => "pass",
            JitVerdict::Fail => "fail",
            JitVerdict::Skip => "skip",
        }
    }
}

/// One `dmc test` result — the `test` kind. `status` is the interpreter's
/// verdict; under `--jit` the parity verdict rides along as `jit`, with its
/// own message when it fails, so one line per test carries both backends.
pub struct TestResult {
    pub name: String,
    pub file: String,
    pub pass: bool,
    /// Why the interpreter run failed. Omitted on a pass.
    pub message: Option<String>,
    /// `Some` exactly when the run is `--jit`.
    pub jit: Option<JitVerdict>,
    /// Why the JIT run failed. Omitted unless `jit` is `fail`.
    pub jit_message: Option<String>,
}

impl TestResult {
    pub fn to_json(&self) -> String {
        let mut o = Obj::new();
        o.num("schema", SCHEMA as u64);
        o.str("kind", "test");
        o.str("status", if self.pass { "pass" } else { "fail" });
        o.str("name", &self.name);
        o.str("file", &self.file);
        if let Some(m) = &self.message { o.str("message", m); }
        if let Some(j) = self.jit { o.str("jit", j.as_str()); }
        if let Some(m) = &self.jit_message { o.str("jit_message", m); }
        o.finish()
    }
}

/// The terminal object. Exactly one is emitted per `--json` run, after every
/// diagnostic, and it reports the process exit code the run is about to use —
/// so a consumer never has to infer the verdict from the diagnostics, and the
/// "exit codes are unchanged" promise is machine-checkable.
pub struct Summary {
    /// The subcommand as spelled on the command line (`--check`, `jit`, …).
    pub command: String,
    pub errors: usize,
    pub warnings: usize,
    /// `dmc test` only: the human summary line's tally, as data.
    pub passed: Option<usize>,
    pub failed: Option<usize>,
    /// `dmc test --jit` only: the parity counters.
    pub jit_ran: Option<usize>,
    pub jit_skipped: Option<usize>,
    /// Top-level items seen, for the commands that count them. `None` elsewhere.
    pub items: Option<usize>,
    pub exit: i32,
}

impl Summary {
    pub fn to_json(&self) -> String {
        let mut o = Obj::new();
        o.num("schema", SCHEMA as u64);
        o.str("kind", "summary");
        o.str("command", &self.command);
        o.str("status", if self.exit == 0 { "ok" } else { "failed" });
        o.num("errors", self.errors as u64);
        o.num("warnings", self.warnings as u64);
        if let Some(p) = self.passed { o.num("passed", p as u64); }
        if let Some(f) = self.failed { o.num("failed", f as u64); }
        if let Some(r) = self.jit_ran { o.num("jit_ran", r as u64); }
        if let Some(s) = self.jit_skipped { o.num("jit_skipped", s as u64); }
        if let Some(i) = self.items { o.num("items", i as u64); }
        // `dmc` exits 0 or 1; clamp anyway so a stray negative can never be
        // encoded as an enormous positive.
        o.num("exit", self.exit.max(0) as u64);
        o.finish()
    }
}

/// Collects diagnostics for one `--json` run and writes them to stderr.
///
/// Constructed only when `--json` is on. `emit` writes immediately, so a
/// consumer reading the pipe sees a long compile's diagnostics as they are
/// found; the counts it keeps along the way drive the summary.
pub struct Emitter {
    command: String,
    errors: usize,
    warnings: usize,
    passed: Option<usize>,
    failed: Option<usize>,
    jit_ran: Option<usize>,
    jit_skipped: Option<usize>,
    items: Option<usize>,
}

impl Emitter {
    pub fn new(command: impl Into<String>) -> Emitter {
        Emitter {
            command: command.into(),
            errors: 0,
            warnings: 0,
            passed: None,
            failed: None,
            jit_ran: None,
            jit_skipped: None,
            items: None,
        }
    }

    pub fn emit(&mut self, d: &Diagnostic) {
        match d.severity {
            Severity::Error => self.errors += 1,
            Severity::Warning => self.warnings += 1,
        }
        eprintln!("{}", d.to_json());
    }

    /// Write one `test` object. A failed test is not a *diagnostic* — it does
    /// not join the `errors` tally; the summary's `passed`/`failed` carry the
    /// verdict, exactly as the human `test result:` line does.
    pub fn emit_test(&mut self, t: &TestResult) {
        eprintln!("{}", t.to_json());
    }

    pub fn set_items(&mut self, items: usize) {
        self.items = Some(items);
    }

    /// `dmc test`: the same two numbers the human summary line prints.
    pub fn set_test_tally(&mut self, passed: usize, failed: usize) {
        self.passed = Some(passed);
        self.failed = Some(failed);
    }

    /// `dmc test --jit`: the same two numbers the human parity note prints.
    pub fn set_jit_parity(&mut self, ran: usize, skipped: usize) {
        self.jit_ran = Some(ran);
        self.jit_skipped = Some(skipped);
    }

    /// Write the summary and hand back the exit code, unchanged, for the caller
    /// to exit with.
    pub fn finish(self, exit: i32) -> i32 {
        let s = Summary {
            command: self.command,
            errors: self.errors,
            warnings: self.warnings,
            passed: self.passed,
            failed: self.failed,
            jit_ran: self.jit_ran,
            jit_skipped: self.jit_skipped,
            items: self.items,
            exit,
        };
        eprintln!("{}", s.to_json());
        exit
    }
}

// ── Out-of-band fatals ───────────────────────────────────────────────────────
//
// Not every diagnostic can reach the `Emitter`. The JIT's arena-exhaustion
// policy (`MEMORY.md §1.1`) ends the process from inside the allocation
// callback — that is, from inside compiled code's own call stack, with no way
// back to `real_main`. Left alone it would print a prose line into a stream a
// consumer is parsing as JSON, and then exit without an envelope: the two
// things a line-delimited format must never do.

static JSON_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static JSON_COMMAND: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Tell the out-of-band path that `--json` is on for `command`. Called once,
/// from argument parsing, before any compiled code can run.
pub fn arm_out_of_band(command: &str) {
    let _ = JSON_COMMAND.set(command.to_string());
    JSON_ARMED.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Report a fatal diagnostic raised somewhere the `Emitter` cannot be reached,
/// and exit 1 — exactly as the human path did.
///
/// The counts in the summary are the best this path can know: it has no access
/// to the emitter's tally, so it reports the one error it is holding. The exit
/// code, which is the field consumers act on, is exact.
pub fn out_of_band_fatal(msg: &str) -> ! {
    if JSON_ARMED.load(std::sync::atomic::Ordering::SeqCst) {
        eprintln!("{}", Diagnostic::unstructured(msg).to_json());
        let s = Summary {
            command: JSON_COMMAND.get().cloned().unwrap_or_default(),
            errors: 1,
            warnings: 0,
            passed: None,
            failed: None,
            jit_ran: None,
            jit_skipped: None,
            items: None,
            exit: 1,
        };
        eprintln!("{}", s.to_json());
    } else {
        eprintln!("{}", msg);
    }
    std::process::exit(1);
}

/// A diagnostic tag is the kebab-case word a message leads with, where the docs
/// give one — `port-forbidden` (`PORTS.md §5`), `decode-type` (`PORTS.md §6`),
/// `comptime-non-static` and `fuse-infeasible` (`SPEC.md §7.7`, `§7.8`). The
/// human renderer keeps it inline; `--json` lifts it into `code` so a consumer
/// does not have to split the prose.
///
/// Deliberately syntactic: any tag the docs add later is picked up without a
/// code change, and a message that merely *contains* a colon is not mistaken
/// for a tagged one — the prefix must be kebab-case, lowercase, and have at
/// least one hyphen.
pub fn tag_of(msg: &str) -> Option<&str> {
    let (head, _) = msg.split_once(": ")?;
    if !head.contains('-') {
        return None;
    }
    let ok = !head.is_empty()
        && head.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !head.starts_with('-')
        && !head.ends_with('-');
    if ok { Some(head) } else { None }
}

// ── Minimal JSON object writer ───────────────────────────────────────────────
//
// A hand-rolled writer rather than a dependency: the compiler has no serde, the
// shape here is one flat object of strings and non-negative integers, and the
// escaping rules that matter (quote, backslash, the C0 range) are the same ones
// `interp.rs` already implements for `json_encode`.

struct Obj {
    buf: String,
    first: bool,
}

impl Obj {
    fn new() -> Obj {
        Obj { buf: "{".to_string(), first: true }
    }

    fn sep(&mut self) {
        if self.first { self.first = false } else { self.buf.push(',') }
    }

    fn str(&mut self, key: &str, value: &str) {
        self.sep();
        write!(self.buf, "{}:{}", escape(key), escape(value)).unwrap();
    }

    fn num(&mut self, key: &str, value: u64) {
        self.sep();
        write!(self.buf, "{}:{}", escape(key), value).unwrap();
    }

    /// A value that is already valid JSON (a `shape_array`). The caller owns
    /// its well-formedness; nothing else may reach this.
    fn raw(&mut self, key: &str, json: &str) {
        self.sep();
        write!(self.buf, "{}:{}", escape(key), json).unwrap();
    }

    fn finish(mut self) -> String {
        self.buf.push('}');
        self.buf
    }
}

/// A JSON string literal, quotes included. Escapes the two mandatory
/// characters, the four short forms, and everything below U+0020; the
/// diagnostics carry `…`, `≥`, and box-drawing characters, which are emitted as
/// UTF-8 (JSON's default encoding) rather than `\u`-escaped.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => { write!(out, "\\u{:04x}", c as u32).unwrap(); }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
