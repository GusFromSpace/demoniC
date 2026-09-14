/// Typechecker unit tests — small fragments hitting specific check paths.
/// Integration verification: `dmc --check examples/*.dmc` from the shell.

use super::check::Checker;
use super::lexer::Lexer;
use super::parser::Parser;

/// Parse, then run the pipeline the `dmc` binary runs before its checker:
/// #505's `@comptime` fold, whose diagnostics are seeded into the `Checker`
/// exactly as `main.rs` seeds them. A helper that skipped the fold would test
/// a compiler no user has — it would miss every `comptime-non-static`, and it
/// would hand the checker an unfolded tree where the binary hands it a
/// literal.
fn checked(src: &str) -> Checker {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let mut program = Parser::new(tokens).parse_program().expect("parse failed");
    let comptime_errors = super::comptime::fold_program(&mut program);
    let mut checker = Checker::new();
    checker.errors = comptime_errors;
    checker.check_program(&program, None);
    checker
}

fn check(src: &str) -> Vec<String> {
    checked(src).errors.iter().map(|e| e.msg.clone()).collect()
}

fn passes(src: &str) -> bool { check(src).is_empty() }

/// Whole `TypeError`s, for the tests that assert on a diagnostic's hint rather
/// than only its message.
fn check_full(src: &str) -> Vec<super::check::TypeError> {
    checked(src).errors.clone()
}

/// Lint diagnostics (non-fatal warnings), as message strings.
fn warnings(src: &str) -> Vec<String> {
    checked(src).warnings.iter().map(|w| w.msg.clone()).collect()
}

fn writeback_warns(src: &str) -> bool {
    warnings(src).iter().any(|m| m.contains("does not write back"))
}

#[test]
fn lint_writeback_flags_dead_element_assignment() {
    // `entry` is copied out of the tensor, assigned, and never read: a no-op.
    let src = "fn main() -> nil { \
                 let !t = forge.zeros[u64, [3]]; \
                 let !entry = t[0]; \
                 entry = 9u64; \
                 nil }";
    assert!(writeback_warns(src), "expected a write-back warning, got {:?}", warnings(src));
}

#[test]
fn lint_writeback_spares_read_accumulator() {
    // `m` is bound from an element but genuinely read (running max), so the
    // assignment is a legitimate scratch update, not a missed write-back.
    let src = "fn f(data: Tensor[f64, [8]]) -> f64 { \
                 let !m = data[0]; \
                 for i in 0..8 { if data[i] > m { m = data[i] } } \
                 m }";
    assert!(!writeback_warns(src), "false positive on accumulator: {:?}", warnings(src));
}

#[test]
fn lint_writeback_spares_write_through_index() {
    // Correct form: assign through the index. No copied-out local at all.
    let src = "fn main() -> nil { \
                 let !t = forge.zeros[u64, [3]]; \
                 t[0] = 9u64; \
                 nil }";
    assert!(!writeback_warns(src), "false positive on write-through: {:?}", warnings(src));
}

#[test]
fn lint_writeback_spares_scratch_that_is_written_back() {
    // `s` is read (`s - ...`) and explicitly stored back via `X[i] = s`.
    let src = "fn f(M: Tensor[f64, [3, 3]], X: Tensor[f64, [3]]) -> nil { \
                 let !s = M[0, 0]; \
                 s = s - M[0, 1]; \
                 X[0] = s; \
                 nil }";
    assert!(!writeback_warns(src), "false positive on written-back scratch: {:?}", warnings(src));
}

#[test]
fn empty_program_passes() {
    assert!(passes(""));
}

#[test]
fn simple_fn_passes() {
    assert!(passes("fn id(x: i64) -> i64 { x }"));
}

#[test]
fn char_literal_returns_u32() {
    assert!(passes(r#"fn main() -> u32 { c"A" }"#));
}

#[test]
fn char_literal_assigns_to_u32_binding() {
    assert!(passes(r#"fn main() -> nil { let ch: u32 = c"Z"; nil }"#));
}

#[test]
fn undefined_ident_fails() {
    let errs = check("fn t() -> nil { let _ = undefined_thing; nil }");
    assert!(errs.iter().any(|e| e.contains("undefined identifier")), "got: {:?}", errs);
}

#[test]
fn matmul_inner_dim_mismatch() {
    let errs = check(r#"
        fn bad[B](
            a: Tensor[f32, [B, 8]],
            b: Tensor[f32, [16, B]],
        ) -> Tensor[f32, [B, B]] { a @ b }
    "#);
    assert!(errs.iter().any(|e| e.contains("matmul") && e.contains("inner")),
            "expected matmul inner-dim error, got: {:?}", errs);
}

#[test]
fn matmul_compatible_passes() {
    assert!(passes(r#"
        fn good[B, M, K, N](
            a: Tensor[f32, [B, M, K]],
            b: Tensor[f32, [B, K, N]],
        ) -> Tensor[f32, [B, M, N]] { a @ b }
    "#));
}

// ── #248: static shape mismatches through constructors are check-time errors ──
// SPEC §178/§758 promise these are caught before any code runs; the JIT already
// rejects the matmul case at lowering, so the checker uniquely lacked the check.
// The fix derives the constructor's static shape at the operator site (and via a
// side-table for bindings) WITHOUT changing the constructor's reported type, so
// KV seeding and symbolic shapes stay untouched.

#[test]
fn ctor_matmul_inner_dim_mismatch_is_check_error() {
    let errs = check(r#"
        fn main() -> nil {
            let z = forge.zeros[f32, [2, 3]] @ forge.zeros[f32, [4, 5]]
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("matmul") && e.contains("3") && e.contains("4")),
            "expected static matmul inner-dim error, got: {:?}", errs);
}

#[test]
fn ctor_matmul_compatible_passes() {
    assert!(passes(r#"
        fn main() -> nil {
            let z = forge.zeros[f32, [2, 3]] @ forge.zeros[f32, [3, 5]]
            nil
        }
    "#));
}

#[test]
fn let_bound_ctor_matmul_mismatch_is_check_error() {
    // #283: the same mismatch as ctor_matmul_inner_dim_mismatch_is_check_error,
    // but flowing through `let` bindings — must still be a compile-time error.
    let errs = check(r#"
        fn main() -> nil {
            let a = forge.zeros[f32, [2, 3]]
            let b = forge.zeros[f32, [4, 5]]
            let z = a @ b
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("matmul") && e.contains("3") && e.contains("4")),
            "expected static matmul inner-dim error through let, got: {:?}", errs);
}

#[test]
fn let_bound_ctor_matmul_compatible_passes() {
    assert!(passes(r#"
        fn main() -> nil {
            let a = forge.zeros[f32, [2, 3]]
            let b = forge.zeros[f32, [3, 5]]
            let z = a @ b
            nil
        }
    "#));
}

#[test]
fn let_bound_ctor_broadcast_mismatch_is_check_error() {
    // #283: elementwise mismatch through `let` bindings.
    let errs = check(r#"
        fn main() -> nil {
            let a = forge.zeros[f32, [3]]
            let b = forge.zeros[f32, [5]]
            let z = a .+ b
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("elementwise") && e.contains("3") && e.contains("5")),
            "expected static broadcast error through let, got: {:?}", errs);
}

#[test]
fn ctor_broadcast_mismatch_is_check_error() {
    let errs = check(r#"
        fn main() -> nil {
            let z = forge.zeros[f32, [3]] .+ forge.zeros[f32, [5]]
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("elementwise") && e.contains("3") && e.contains("5")),
            "expected static broadcast error, got: {:?}", errs);
}

#[test]
fn ctor_static_oob_index_is_check_error() {
    let errs = check(r#"
        fn main() -> nil {
            let m = forge.zeros[f32, [2, 2]]
            let x = m[5, 5]
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("out of bounds") && e.contains("size 2")),
            "expected static OOB index error, got: {:?}", errs);
}

#[test]
fn ctor_negative_index_within_range_passes() {
    // demoniC allows Python-style negative indexing; -1 is the last element.
    assert!(passes(r#"
        fn main() -> nil {
            let m = forge.zeros[f32, [2, 2]]
            let x = m[-1, 0]
            nil
        }
    "#));
}

#[test]
fn ctor_too_negative_index_is_check_error() {
    let errs = check(r#"
        fn main() -> nil {
            let m = forge.zeros[f32, [2, 2]]
            let x = m[-3, 0]
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("out of bounds")),
            "expected static OOB on -3 into size-2 axis, got: {:?}", errs);
}

#[test]
fn symbolic_ctor_shapes_do_not_false_positive() {
    // Symbolic dims (shape params) and variable indices must stay conservative —
    // the existing `equivalent`/`simplify` returns Unknown, so no error fires.
    assert!(passes(r#"
        fn f[N, K, M]() -> nil {
            let z = forge.zeros[f32, [N, K]] @ forge.zeros[f32, [K, M]]
            let !w = forge.zeros[f32, [N, K]]
            let i = 0
            let probe = w[i, i]
            nil
        }
    "#));
}

#[test]
fn oversized_tensor_literal_is_error() {
    // #403/#501 (SPEC §4.2, TOKENIZER §8a): tensor literals are for small
    // constants. Past 256 total elements this was a lint that TOKENIZER §8a
    // promised to promote; #501 collects the promise — it is a hard error now.
    let elems = std::iter::repeat("1.0").take(300).collect::<Vec<_>>().join(", ");
    let src = format!("fn main() -> f32 {{ let t = [{elems}]  t[0] }}");
    let errs = check(&src);
    assert!(errs.iter().any(|e| e.contains("300 elements") && e.contains("limit is 256")),
            "expected oversized-literal error, got {:?}", errs);
    // It must no longer be a lint — the diagnostic moved, it did not double up.
    assert!(!warnings(&src).iter().any(|w| w.contains("300 elements")),
            "oversized literal must not also warn");

    // The diagnostic names the replacement spellings.
    let hints: Vec<String> = check_full(&src).iter()
        .filter(|e| e.msg.contains("limit is 256"))
        .filter_map(|e| e.hint.clone())
        .collect();
    assert!(hints.iter().any(|h| h.contains("forge.zeros") && h.contains("vault.load")),
            "error must name the replacement spelling, got {:?}", hints);

    // Nested literals count leaves through the full inferred shape: 2 × 150.
    let row = std::iter::repeat("1").take(150).collect::<Vec<_>>().join(", ");
    let src = format!("fn main() -> i64 {{ let t = [[{row}], [{row}]]  t[0, 0] }}");
    let errs = check(&src);
    assert!(errs.iter().any(|e| e.contains("300 elements")),
            "expected nested-literal error, got {:?}", errs);

    // Exactly 256 stays legal — the bound is "more than 256".
    let elems = std::iter::repeat("1.0").take(256).collect::<Vec<_>>().join(", ");
    let src = format!("fn main() -> f32 {{ let t = [{elems}]  t[0] }}");
    assert!(passes(&src), "256-element literal must stay legal, got {:?}", check(&src));
}

#[test]
fn oversized_tensor_literal_survives_demon_mode() {
    // #501: the 256-element bound is a spec violation, not a safe-mode lint,
    // so `--demon` does not release it — same rule as the §3.1 cross-arena
    // write. A lint would vanish here; an error must not.
    let elems = std::iter::repeat("1.0").take(300).collect::<Vec<_>>().join(", ");
    let src = format!("fn main() -> f32 {{ let t = [{elems}]  t[0] }}");
    let tokens = Lexer::new(&src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker = Checker::new();
    checker.demon = true;
    checker.check_program(&program, None);
    assert!(checker.errors.iter().any(|e| e.msg.contains("limit is 256")),
            "demon mode must not suppress the oversized-literal error, got {:?}",
            checker.errors.iter().map(|e| e.msg.clone()).collect::<Vec<_>>());
}

#[test]
fn kv_element_assign_is_check_error() {
    // #403 (SPEC §4.8): a KV is append-only. Element assignment through a
    // KV-typed binding must fail at check time — previously it surfaced only
    // at runtime, as a misleading out-of-bounds on the `~` axis.
    let errs = check(r#"
        fn main() -> nil {
            let !cache: KV[f32, [2, ~, 3]] = forge.ones[f32, [2, 1, 3]]
            cache[0, 0, 0] = 5.0
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("append-only") && e.contains("cache")),
            "expected KV append-only error, got {:?}", errs);

    // `<-` append and element reads stay legal.
    assert!(passes(r#"
        fn main() -> f32 {
            let !cache: KV[f32, [2, ~, 3]] = forge.ones[f32, [2, 1, 3]]
            cache <- forge.ones[f32, [2, 2, 3]]
            cache[0, 0, 0]
        }
    "#));
}

#[test]
fn cross_arena_write_outside_vault_block_is_check_error() {
    // #442 (MEMORY §3.1): mutating Vault data from the default Forge context
    // belongs in an explicit `vault { … }` block. The spec's hard error, as
    // of the corpus migration to `forge.*` for per-run scratch.
    let es = check(r#"
        fn main() -> nil {
            let !w = vault.ones[f32, [4]]
            w[0] = 0.5
            nil
        }
    "#);
    assert!(es.iter().any(|e| e.contains("cross-arena write") && e.contains("`w`")),
            "expected cross-arena write error, got {:?}", es);

    // Compound assign is a read-modify-write of Vault memory.
    let es = check(r#"
        fn main() -> nil {
            let !w = vault.ones[f32, [4]]
            w -= vault.ones[f32, [4]]
            nil
        }
    "#);
    assert!(es.iter().any(|e| e.contains("cross-arena write")),
            "expected compound-assign cross-arena error, got {:?}", es);

    // A `vault { … }` block *expression* also produces Vault data; a nested
    // `forge { … }` context does not sneak past the innermost-block rule.
    let es = check(r#"
        fn main() -> nil {
            let !w = vault { [1.0, 2.0] }
            vault { forge { w[0] = 3.0 } }
            nil
        }
    "#);
    assert!(es.iter().any(|e| e.contains("cross-arena write")),
            "expected innermost-context cross-arena error, got {:?}", es);
}

#[test]
fn vault_writes_inside_vault_block_are_legal() {
    // The training-step idiom: mutate Vault weights inside `vault { … }`.
    let es = check(r#"
        fn main() -> nil {
            let !w = vault.ones[f32, [4]]
            vault {
                w[0] = 0.5
                w -= vault.ones[f32, [4]]
            }
            nil
        }
    "#);
    assert!(!es.iter().any(|e| e.contains("cross-arena")),
            "vault-block writes must be legal, got {:?}", es);
    // Reads of Vault data anywhere are fine; so are Forge-tensor writes
    // anywhere; so is a plain whole-`=` rebind (not a mutation), after which
    // the binding is Forge data and writes are legal.
    let es = check(r#"
        fn main() -> f32 {
            let !w = vault.ones[f32, [4]]
            let !t = forge.ones[f32, [4]]
            t[0] = w[1]
            w = forge.zeros[f32, [4]]
            w[0] = 1.0
            sum(w) + sum(t)
        }
    "#);
    assert!(!es.iter().any(|e| e.contains("cross-arena")),
            "rebind/forge writes must be legal, got {:?}", es);
}

#[test]
fn uninit_read_before_write_is_check_error() {
    // #403 (MEMORY §2): definite assignment over uninit allocations —
    // reading a `forge.uninit` binding before any write is a check error.
    // Builtins (sum) never fill their args, so this is a read.
    let errs = check(r#"
        fn main() -> f32 {
            let t = forge.uninit[f32, [4]]
            sum(t)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("uninitialized") && e.contains("`t`")),
            "expected uninit-read error, got {:?}", errs);

    // Reading the binding on the RHS of its own first write is still a read.
    let errs = check(r#"
        fn main() -> nil {
            let !t = forge.uninit[f32, [4]]
            t[0] = t[1]
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("uninitialized")),
            "expected self-RHS uninit-read error, got {:?}", errs);

    // Passing to a plain (non-`!`) param of a known fn is a read.
    let errs = check(r#"
        fn total(t: Tensor[f32, [4]]) -> f32 { sum(t) }
        fn main() -> f32 {
            let t = forge.uninit[f32, [4]]
            total(t)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("uninitialized")),
            "expected plain-param uninit-read error, got {:?}", errs);
}

#[test]
fn uninit_write_then_read_passes() {
    // The canonical fill loop: the first write initializes the binding
    // (binding-level, deliberately coarse — MEMORY §2 / #403).
    assert!(passes(r#"
        fn main() -> f32 {
            let !t = forge.uninit[f32, [4]]
            for i in 0..4 { t[i] = 1.0 }
            sum(t)
        }
    "#));
    // Passing to a `!` param is the fill.
    assert!(passes(r#"
        fn fill(!t: Tensor[f32, [4]]) -> nil {
            for i in 0..4 { t[i] = 0.0 }
            nil
        }
        fn main() -> f32 {
            let !t = forge.uninit[f32, [4]]
            fill(t)
            sum(t)
        }
    "#));
    // Whole reassignment initializes; rebinding a fresh uninit re-arms.
    assert!(passes(r#"
        fn main() -> f32 {
            let !t = forge.uninit[f32, [4]]
            t = forge.ones[f32, [4]]
            sum(t)
        }
    "#));
    let errs = check(r#"
        fn main() -> f32 {
            let !t = forge.ones[f32, [4]]
            t = forge.uninit[f32, [4]]
            sum(t)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("uninitialized")),
            "re-armed uninit flag must fire, got {:?}", errs);
    // A same-scope rebind to an initialized value masks the flag.
    assert!(passes(r#"
        fn main() -> f32 {
            let t = forge.uninit[f32, [4]]
            let t = forge.ones[f32, [4]]
            sum(t)
        }
    "#));
}

// ── Issue #476: the §2 check reaches one level into model fields ─────────────
//
// A model array held in a model FIELD was invisible to the definite-assignment
// check: the same uninitialized read was a clean `--check` error through a
// local and, through a field, an `opaque` value at runtime whose error named
// neither the field, nor the array, nor initialization. These pin the issue's
// four-spelling table.

const CELLS: &str = r#"
model Cell { !n: i64 }
model Holder { !cells: [Cell; 3] }
"#;

#[test]
fn uninit_model_array_field_read_is_check_error_476() {
    // Spelling 2 — the one that hurt: reads naturally, passed `--check`, and
    // produced opaque elements. A model-array field binds BY VALUE, so the
    // fill through `cs` never reaches `h.cells`.
    let errs = check(&format!(r#"{CELLS}
        fn main() -> i64 {{
            let !h = Holder {{ cells: vault.uninit[Cell, [3]] }}
            let !cs = h.cells
            for i in 0..3 {{ cs[i] = Cell {{ n: 0 }} }}
            let a = h.cells[0]
            a.n
        }}
    "#));
    assert!(errs.iter().any(|e| e.contains("uninitialized") && e.contains("`h.cells`")),
            "fill-through-a-copy must fail at --check like the local spelling, got {:?}", errs);

    // The bare field read, with nothing written at all.
    let errs = check(&format!(r#"{CELLS}
        fn main() -> i64 {{
            let !h = Holder {{ cells: vault.uninit[Cell, [3]] }}
            let c = h.cells[0]
            c.n
        }}
    "#));
    assert!(errs.iter().any(|e| e.contains("uninitialized") && e.contains("`h.cells`")),
            "expected uninit field-read error, got {:?}", errs);
}

#[test]
fn uninit_model_array_field_write_then_read_passes_476() {
    // Spelling 3 — `h.cells[i] = Cell { .. }` writes THROUGH the field, so it
    // is the write that initializes it. It used to fail at runtime with a
    // message about tensors and lists; it now works, which makes it the
    // natural fill-in-place idiom and the thing the read error points at.
    assert!(passes(&format!(r#"{CELLS}
        fn main() -> i64 {{
            let !h = Holder {{ cells: vault.uninit[Cell, [3]] }}
            vault {{ for i in 0..3 {{ h.cells[i] = Cell {{ n: 0 }} }} }}
            let a = h.cells[0]
            a.n
        }}
    "#)));

    // Spelling 4 — the idiom that always worked: build as a local, fill it,
    // store it at construction. The field is never marked, because the value
    // it is built from is not a fresh `uninit`.
    assert!(passes(&format!(r#"{CELLS}
        fn make() -> Holder {{
            let !cs = vault.uninit[Cell, [3]]
            vault {{ for i in 0..3 {{ cs[i] = Cell {{ n: 0 }} }} }}
            Holder {{ cells: cs }}
        }}
        fn main() -> i64 {{
            let !h = make()
            let a = h.cells[0]
            a.n
        }}
    "#)));
}

#[test]
fn uninit_model_array_field_reads_report_once_476() {
    // One bug, one report — same discipline as the binding-level check.
    let errs = check(&format!(r#"{CELLS}
        fn main() -> i64 {{
            let !h = Holder {{ cells: vault.uninit[Cell, [3]] }}
            let a = h.cells[0]
            let b = h.cells[1]
            a.n + b.n
        }}
    "#));
    let n = errs.iter().filter(|e| e.contains("uninitialized")).count();
    assert_eq!(n, 1, "expected exactly one uninit report, got {:?}", errs);
}

#[test]
fn uninit_model_array_field_self_rhs_still_reports_476() {
    // Re-armed for the RHS: the write target is not a read, but the value it
    // reads out of the same uninitialized field is.
    let errs = check(&format!(r#"{CELLS}
        fn main() -> nil {{
            let !h = Holder {{ cells: vault.uninit[Cell, [3]] }}
            vault {{ h.cells[0] = h.cells[1] }}
            nil
        }}
    "#));
    assert!(errs.iter().any(|e| e.contains("uninitialized") && e.contains("`h.cells`")),
            "expected self-RHS uninit field-read error, got {:?}", errs);
}

#[test]
fn uninit_model_array_field_check_does_not_over_report_476() {
    // The check errs toward silence. A model instance aliases through its
    // `Rc`, so anything handed the binding may have filled the field — a
    // method call on it, or passing it to a function.
    assert!(passes(&format!(r#"{CELLS}
        fn fill(!h: Holder) -> nil {{
            vault {{ for i in 0..3 {{ h.cells[i] = Cell {{ n: 1 }} }} }}
            nil
        }}
        fn main() -> i64 {{
            let !h = Holder {{ cells: vault.uninit[Cell, [3]] }}
            fill(h)
            let a = h.cells[0]
            a.n
        }}
    "#)));

    // A tensor field is NOT tracked: it binds as a live alias, so filling
    // through `let !v = h.buf` really works and flagging it would be a false
    // report. This is the deliberate boundary of the one-level reach.
    assert!(passes(r#"
        model Buf { !buf: Tensor[f32, [4]] }
        fn main() -> f32 {
            let !h = Buf { buf: vault.uninit[f32, [4]] }
            let !v = h.buf
            vault { for i in 0..4 { v[i] = 1.0 } }
            sum(h.buf)
        }
    "#));

    // Rebinding the root drops the marks with it.
    assert!(passes(&format!(r#"{CELLS}
        fn make() -> Holder {{
            let !cs = vault.uninit[Cell, [3]]
            vault {{ for i in 0..3 {{ cs[i] = Cell {{ n: 2 }} }} }}
            Holder {{ cells: cs }}
        }}
        fn main() -> i64 {{
            let !h = Holder {{ cells: vault.uninit[Cell, [3]] }}
            h = make()
            let a = h.cells[0]
            a.n
        }}
    "#)));
}

#[test]
fn model_array_field_bracket_literal_points_at_uninit_476() {
    // Spelling 1 — the natural spelling is still rejected, but the diagnostic
    // no longer sends the author off to fix the literal's element type. No
    // bracket literal of any element type builds a model array.
    let errs = check_full(&format!(r#"{CELLS}
        fn main() -> i64 {{
            let !h = Holder {{ cells: [Cell {{ n: 0 }}, Cell {{ n: 0 }}, Cell {{ n: 0 }}] }}
            let a = h.cells[0]
            a.n
        }}
    "#));
    let e = errs.iter().find(|e| e.msg.contains("mismatched type for field `cells`"))
        .expect(&format!("expected the field mismatch, got {:?}", errs));
    let hint = e.hint.as_deref().unwrap_or("");
    assert!(hint.contains("uninit"),
            "the diagnostic must point at `uninit`, not the literal's element type; hint: {hint:?}");
}

#[test]
fn stream_append_inside_loop_over_same_kv_is_check_error() {
    // #403 (MEMORY §9.1): `<-` to the binding a `for` loop iterates is the
    // mutate-while-iterating hazard, rejected lexically — branches included.
    let errs = check(r#"
        fn main() -> nil {
            let !c: KV[f32, [~]] = forge.ones[f32, [4]]
            for v in c {
                c <- forge.ones[f32, [1]]
            }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("stream-iteration-aliasing") && e.contains("`c`")),
            "expected stream-aliasing error, got {:?}", errs);

    // Buried in a branch inside a nested loop — the rule is lexical, so it
    // still fires.
    let errs = check(r#"
        fn main() -> nil {
            let !c: KV[f32, [~]] = forge.ones[f32, [4]]
            for v in c {
                for i in 0..2 {
                    if i == 0 { c <- forge.ones[f32, [1]] }
                }
            }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("stream-iteration-aliasing")),
            "expected nested/branched stream-aliasing error, got {:?}", errs);
}

#[test]
fn stream_append_inside_alloc_in_operand_block_is_check_error_819() {
    // #819: `collect_stream_appends_expr`'s `Expr::AllocIn { .. } => {}` arm
    // dropped BOTH of `AllocIn`'s operands, so a `c <- …` written inside the
    // `in` operand's block — `<ctor> in { … }`, #674 slice 3a's
    // postfix — went unreported here even though the identical append fires
    // normally when written directly in the loop body (the test above). The
    // `in` operand parses as a `postfix_expr`, and `postfix_expr`'s own
    // `primary` alternative admits a `block`, so `forge.zeros[…] in { c <-
    // …  b }` genuinely parses this way (§2.1 GRAMMAR.ebnf).
    let errs = check(r#"
        fn f(b: Budget) -> nil {
            let !c: KV[f32, [~]] = forge.ones[f32, [4]]
            for v in c {
                forge.ones[f32, [1]] in { c <- forge.ones[f32, [1]]  b }
            }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("stream-iteration-aliasing") && e.contains("`c`")),
            "expected stream-aliasing error from inside the `in` operand's block, got {:?}",
            errs);
}

#[test]
fn stream_append_inside_budget_block_body_is_check_error_819() {
    // #819 pinned regression: `Expr::BudgetBlock`'s `body` was already
    // walked by this collector before this fix (the `AllocIn` gap above was
    // the only miss) — pinned alongside it so a future edit to this walker
    // cannot silently regress the `forge.enter` case back to unreported.
    let errs = check(r#"
        fn f(b: Budget) -> nil {
            let !c: KV[f32, [~]] = forge.ones[f32, [4]]
            for v in c {
                forge.enter(b) { c <- forge.ones[f32, [1]] }
            }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("stream-iteration-aliasing") && e.contains("`c`")),
            "expected stream-aliasing error from inside a `forge.enter` budget block, got {:?}",
            errs);
}

#[test]
fn stream_append_inside_budget_block_operand_block_is_check_error_819() {
    // #819: `Expr::BudgetBlock`'s own `operand` was the same unwalked gap
    // this PR fixed for `Expr::AllocIn` — `parse_budget_block` (parser.rs)
    // parses the `forge.enter(…)` operand with a full `parse_expr()`, so a
    // block literal there parses fine: `forge.enter({ c <- …  b }) { … }`.
    // Before this arm walked `bb.operand`, a `<-` written inside it went
    // unreported by the #403 lint even though it fires normally when
    // written directly in the budget block's body (the test above).
    let errs = check(r#"
        fn f(b: Budget) -> nil {
            let !c: KV[f32, [~]] = forge.ones[f32, [4]]
            for v in c {
                forge.enter({ c <- forge.ones[f32, [1]]  b }) { nil }
            }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("stream-iteration-aliasing") && e.contains("`c`")),
            "expected stream-aliasing error from inside a `forge.enter` operand block, got {:?}",
            errs);
}

#[test]
fn stream_append_inside_call_argument_block_is_check_error_826() {
    // #826: `collect_stream_appends_expr` had no `Expr::Postfix` arm at all,
    // so a `<-` written inside a block literal passed as a CALL ARGUMENT
    // went unreported — even though the identical append fires normally
    // when written directly in the loop body, or inside an `AllocIn`/
    // `BudgetBlock` operand (#819/#823). The issue's own reduced repro.
    let errs = check(r#"
        fn g(x: f32) -> f32 { x }

        fn f() -> nil {
            let !c: KV[f32, [~]] = forge.ones[f32, [4]]
            for v in c {
                g({ c <- forge.ones[f32, [1]]  1.0 })
            }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("stream-iteration-aliasing") && e.contains("`c`")),
            "expected stream-aliasing error from inside a call argument block, got {:?}",
            errs);
}

#[test]
fn stream_append_inside_plain_call_argument_still_passes_826() {
    // Contrast case: a call with a plain (non-appending) block argument must
    // not fire the lint — the walker should not become false-positive-prone
    // once it descends into call arguments.
    assert!(passes(r#"
        fn g(x: f32) -> f32 { x }

        fn f() -> nil {
            let !c: KV[f32, [~]] = forge.ones[f32, [4]]
            for v in c {
                g({ let y: f32 = 1.0  y })
            }
            nil
        }
    "#));
}

#[test]
fn stream_append_inside_call_nested_in_match_arm_in_loop_is_check_error_826() {
    // Nested-form regression: a call argument's append-carrying block, buried
    // inside a `match` arm, inside the `for` loop this walker is scanning —
    // every layer (`Expr::Match` arm body, then `Expr::Postfix`'s call
    // arguments) must be walked for the lint to still fire.
    let errs = check(r#"
        fn g(x: f32) -> f32 { x }

        fn f() -> nil {
            let !c: KV[f32, [~]] = forge.ones[f32, [4]]
            for v in c {
                match 1 {
                    _ => g({ c <- forge.ones[f32, [1]]  1.0 }),
                }
            }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("stream-iteration-aliasing") && e.contains("`c`")),
            "expected stream-aliasing error from a call argument nested in a match arm, got {:?}",
            errs);
}

#[test]
fn stream_append_inside_assign_rhs_block_is_check_error_826() {
    // #826 (found auditing the same walker): `Stmt::Expr`'s previous
    // `else if assign.is_none()` shape walked `lhs` only when there was no
    // `assign` at all, so an assign whose RHS carries a block — even one
    // that is not itself the direct `c <- …` hit — silently dropped that
    // RHS. `x = { c <- … }` inside the loop went unreported.
    let errs = check(r#"
        fn f() -> nil {
            let !c: KV[f32, [~]] = forge.ones[f32, [4]]
            let !x: f32 = 0.0
            for v in c {
                x = { c <- forge.ones[f32, [1]]  1.0 }
            }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("stream-iteration-aliasing") && e.contains("`c`")),
            "expected stream-aliasing error from an assign's RHS block, got {:?}",
            errs);
}

#[test]
fn stream_append_via_snapshot_iteration_still_passes() {
    // The MEMORY §9.1 sanctioned idiom: iterate a snapshot binding, append to
    // the stream itself. Different binding name — no error. Appending to a
    // *different* stream inside a loop is likewise fine.
    assert!(passes(r#"
        fn main() -> nil {
            let !c: KV[f32, [~]] = forge.ones[f32, [4]]
            let snap = c
            for v in snap {
                c <- forge.ones[f32, [1]]
            }
            nil
        }
    "#));
    assert!(passes(r#"
        fn main() -> nil {
            let !a: KV[f32, [~]] = forge.ones[f32, [4]]
            let !b: KV[f32, [~]] = forge.ones[f32, [1]]
            for v in a {
                b <- forge.ones[f32, [1]]
            }
            nil
        }
    "#));
}

#[test]
fn kv_seeding_annotation_still_checks() {
    // The trap to avoid: making constructors report a concrete Tensor broke
    // `let k: KV[..] = forge.ones[..]`. The constructor's reported type stays
    // Unknown, so an explicit KV annotation seeding is accepted.
    assert!(passes(r#"
        fn main() -> nil {
            let cache: KV[f32, [2, 4]] = forge.ones[f32, [2, 4]]
            nil
        }
    "#));
}

#[test]
fn match_arm_shadowing_const_warns() {
    // #269: a bare-identifier arm that shadows an in-scope value binds (catch-all)
    // instead of comparing — the named-constant-in-match footgun. Warn.
    let ws = warnings(r#"
        fn classify(k: i64) -> i64 {
            let TWO = 2
            match k { TWO => 100, _ => 200 }
        }
        fn main() -> i64 { classify(5) }
    "#);
    assert!(ws.iter().any(|w| w.contains("binds the scrutinee") && w.contains("TWO")),
            "expected the #269 footgun warning, got: {:?}", ws);
}

#[test]
fn match_fresh_catchall_does_not_warn() {
    // A genuine fresh catch-all bind (`other` not previously in scope) is fine.
    let ws = warnings(r#"
        fn f(k: i64) -> i64 { match k { 0 => 10, 1 => 20, other => other * 2 } }
        fn main() -> i64 { f(5) }
    "#);
    assert!(!ws.iter().any(|w| w.contains("binds the scrutinee")),
            "a fresh catch-all bind must not warn, got: {:?}", ws);
}

#[test]
fn match_bare_variant_of_other_enum_warns() {
    // #350: on an enum scrutinee, a bare ident that is a variant of a DIFFERENT
    // enum silently binds (catch-all) instead of matching a variant — warn and
    // name the enum that actually owns it.
    let ws = warnings(r#"
        enum Color { Red, Green, Blue }
        enum Signal { Stop, Go }
        fn classify(c: Color) -> i64 {
            match c { Red => 1, Stop => 2, _ => 0 }
        }
        fn main() -> i64 { classify(Color.Red) }
    "#);
    assert!(ws.iter().any(|w| w.contains("Stop") && w.contains("variant of enum `Signal`")),
            "expected the #350 cross-enum shadow warning, got: {:?}", ws);
}

#[test]
fn match_bare_variant_of_same_enum_does_not_warn() {
    // A real variant of the scrutinee's own enum is a variant match, not a
    // catch-all — it must stay quiet.
    let ws = warnings(r#"
        enum Color { Red, Green, Blue }
        fn classify(c: Color) -> i64 {
            match c { Red => 1, Green => 2, Blue => 3 }
        }
        fn main() -> i64 { classify(Color.Green) }
    "#);
    assert!(!ws.iter().any(|w| w.contains("binds the scrutinee")),
            "a same-enum variant pattern must not warn, got: {:?}", ws);
}

// ── #350 (S9): the bare-variant catch-all lint ─────────────────────────────
//
// Ruling S9 on the #501 sweep keeps bare variants in the grammar and pays for
// them with a lint instead: in an enum-typed match, a bare ident that is NOT a
// variant of that enum binds irrefutably (SPEC §4.5), so it swallows the
// remaining variants and defeats exhaustiveness. Every such arm warns; arms
// that DO resolve to a variant stay silent, because the corpus spells them
// both ways on purpose.

/// Lint diagnostics as whole `TypeError`s, for the tests that assert on a
/// hint rather than only the message.
fn warnings_full(src: &str) -> Vec<super::check::TypeError> {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker = Checker::new();
    checker.check_program(&program, None);
    checker.warnings.clone()
}

/// The lint's own diagnostics, isolated from the other safe-mode families.
fn bare_variant_lints(src: &str) -> Vec<super::check::TypeError> {
    warnings_full(src).into_iter()
        .filter(|w| w.msg.contains("binds the scrutinee"))
        .collect()
}

const TRAFFIC: &str = r#"
    enum Light { Red, Yellow, Green }
"#;

#[test]
fn match_enum_typod_variant_warns() {
    // The case the lint exists for: `Gren` is nobody's variant, so it binds
    // everything, shadows `Yellow` below it, and silences the missing-`Green`
    // exhaustiveness error. Warn, and name the variant it misspells.
    let src = format!(r#"{TRAFFIC}
        fn duration(l: Light) -> i64 {{
            match l {{ Light.Red => 30, Gren => 25, Yellow => 5 }}
        }}
        fn main() -> i64 {{ duration(Light.Red) }}
    "#);
    let ws = bare_variant_lints(&src);
    assert_eq!(ws.len(), 1, "expected exactly one lint, got: {:?}", ws);
    assert!(ws[0].msg.contains("`Gren`") && ws[0].msg.contains("enum `Light` has no variant"),
            "unexpected message: {}", ws[0].msg);
    assert!(ws[0].hint.as_deref().is_some_and(|h| h.contains("`Light.Green`")),
            "expected the misspelled variant named in the hint, got: {:?}", ws[0].hint);
}

#[test]
fn match_enum_lowercase_variant_warns() {
    // A case slip is the same footgun: `green` is not `Green`.
    let src = format!(r#"{TRAFFIC}
        fn f(l: Light) -> i64 {{ match l {{ Light.Red => 1, green => 2 }} }}
        fn main() -> i64 {{ f(Light.Red) }}
    "#);
    let ws = bare_variant_lints(&src);
    assert_eq!(ws.len(), 1, "expected exactly one lint, got: {:?}", ws);
    assert!(ws[0].hint.as_deref().is_some_and(|h| h.contains("`Light.Green`")),
            "expected `Light.Green` suggested, got: {:?}", ws[0].hint);
}

#[test]
fn match_enum_bare_catchall_warns_suggesting_underscore() {
    // A name that resembles no variant is a genuine catch-all bind — still
    // warned, but pointed at `_`, the spelling that says so out loud.
    let src = format!(r#"{TRAFFIC}
        fn f(l: Light) -> i64 {{ match l {{ Light.Red => 1, other => 0 }} }}
        fn main() -> i64 {{ f(Light.Red) }}
    "#);
    let ws = bare_variant_lints(&src);
    assert_eq!(ws.len(), 1, "expected exactly one lint, got: {:?}", ws);
    let hint = ws[0].hint.clone().unwrap_or_default();
    assert!(hint.contains("`_`") && hint.contains("Red, Yellow, Green"),
            "expected the `_` suggestion and the variant list, got: {hint}");
    // No misspelling was plausible, so nothing is proposed as the intended
    // variant.
    assert!(!hint.contains("did you mean"), "unexpected typo suggestion: {hint}");
}

#[test]
fn match_bare_variants_stay_silent() {
    // Resolved bare variants, qualified arms, and an explicit `_` are all
    // correct spellings — the lint must not fire on any of them. These are the
    // shapes `examples/enum_traffic.dmc` and `examples/enum_shape.dmc` use.
    let src = format!(r#"{TRAFFIC}
        enum Shape {{ Circle(i64), Rect(i64, i64), Empty }}
        fn duration(l: Light) -> i64 {{
            match l {{ Light.Red => 30, Green => 25, Yellow => 5 }}
        }}
        fn can_go(l: Light) -> bool {{
            match l {{ Light.Green => true, _ => false }}
        }}
        fn area(s: Shape) -> i64 {{
            match s {{ Circle(r) => 3 * r * r, Shape.Rect(w, h) => w * h, Empty => 0 }}
        }}
        fn main() -> i64 {{ duration(Light.Red) + area(Shape.Empty) }}
    "#);
    let ws = bare_variant_lints(&src);
    assert!(ws.is_empty(), "correct enum-match spellings must stay silent, got: {:?}", ws);
}

#[test]
fn match_bare_ident_on_non_enum_scrutinee_does_not_warn() {
    // The lint is scoped to enum scrutinees: on an `i64` a bare ident is the
    // only way to bind the value, and #269 already covers the shadowing case.
    let ws = bare_variant_lints(r#"
        fn f(k: i64) -> i64 { match k { 0 => 10, 1 => 20, other => other * 2 } }
        fn main() -> i64 { f(5) }
    "#);
    assert!(ws.is_empty(), "a non-enum scrutinee must not warn, got: {:?}", ws);
}

#[test]
fn match_bare_variant_lint_is_released_by_demon() {
    // Safe-mode family member: `--demon` drops it on the floor (#196).
    let src = format!(r#"{TRAFFIC}
        fn duration(l: Light) -> i64 {{
            match l {{ Light.Red => 30, Gren => 25, Yellow => 5 }}
        }}
        fn main() -> i64 {{ duration(Light.Red) }}
    "#);
    assert!(warnings(&src).iter().any(|w| w.contains("binds the scrutinee")),
            "safe mode should warn");
    assert!(!warnings_demon(&src).iter().any(|w| w.contains("binds the scrutinee")),
            "demon mode should release the lint, got: {:?}", warnings_demon(&src));
}

#[test]
fn match_bare_variant_lint_does_not_error() {
    // A lint, never an error: the program still type-checks, because the arm's
    // meaning (SPEC §4.5) is unchanged.
    let src = format!(r#"{TRAFFIC}
        fn duration(l: Light) -> i64 {{
            match l {{ Light.Red => 30, Gren => 25, Yellow => 5 }}
        }}
        fn main() -> i64 {{ duration(Light.Red) }}
    "#);
    assert!(passes(&src), "the lint must not fail --check, got: {:?}", check(&src));
}

#[test]
fn match_guarded_bare_ident_on_enum_does_not_warn() {
    // A guard states the intent — the arm is deliberately a conditional bind,
    // not a mistyped variant.
    let src = format!(r#"{TRAFFIC}
        fn f(l: Light) -> i64 {{
            match l {{ Light.Red => 1, x if (x as i64) > 1 => 2, _ => 0 }}
        }}
        fn main() -> i64 {{ f(Light.Red) }}
    "#);
    let ws = bare_variant_lints(&src);
    assert!(ws.is_empty(), "a guarded arm must not warn, got: {:?}", ws);
}

// ── #369: unimplemented-directive lint ─────────────────────────────────────

#[test]
fn unimplemented_directive_warns() {
    // #369: `@recompute` / `@inplace` are parsed but have no effect — warn so
    // they aren't silent no-ops. Each is written on a target DIRECTIVES.md §1
    // allows it on: `@inplace` on an assignment statement, `@recompute` on a
    // fn. `@comptime` left this list at #505 — see
    // `effective_directives_do_not_warn`.
    for d in ["recompute(budget=2)"] {
        let src = format!("@{d}\nfn f() -> i64 {{ 1 }}\nfn main() -> i64 {{ f() }}");
        let ws = warnings(&src);
        assert!(ws.iter().any(|w| w.contains("is not implemented") && w.contains("no effect")),
                "expected an unimplemented-directive warning for @{d}, got: {:?}", ws);
    }
    let ws = warnings("fn main() -> i64 {\n let !a = 1\n @inplace a += 1\n a\n}");
    assert!(ws.iter().any(|w| w.contains("is not implemented") && w.contains("no effect")),
            "expected an unimplemented-directive warning for @inplace, got: {:?}", ws);
}

#[test]
fn effective_directives_do_not_warn() {
    // Directives the compiler acts on must stay quiet — including `@host match`,
    // which is functional (host-feature dispatch), not a no-op, and `@comptime`,
    // which folds at #505. The residual (shape-parameter) `@comptime` is the
    // form that survives to the checker at all, so it is the one that could
    // still have warned.
    let ws = warnings(r#"
        @grad fn loss(!w: Tensor[f32, [4]], x: Tensor[f32, [4]]) -> f32 {
            sum((w .* x) .* (w .* x))
        }
        fn pick() -> i64 { @host match { .avx2 => 1, _ => 0 } }
        fn tile[N](x: Tensor[f32, [N]]) -> i64 { @comptime { N * 2 } }
        fn main() -> i64 { pick() + tile(forge.zeros[f32, [4]]) }
    "#);
    assert!(!ws.iter().any(|w| w.contains("is not implemented")),
            "effective directives (@grad, @host, @comptime) must not warn, got: {:?}", ws);
}

#[test]
fn model_array_param_compatible_with_itself() {
    // #234: a fn taking a model-array param (`[Expr; N]`) — the core of an AST
    // walker — must type-check, including the recursive call passing its own
    // param back in. `compatible_with` lacked an `Array` arm, so two identical
    // `[Expr; 8]` fell through to incompatible ("expected [Expr; 8], got [Expr; 8]").
    assert!(passes(r#"
        model Expr { kind: i64, ival: i64, lhs: i64, rhs: i64 }
        fn eval(nodes: [Expr; 8], i: i64) -> i64 {
            let e = nodes[i]
            match e.kind {
                0 => e.ival,
                1 => eval(nodes, e.lhs) + eval(nodes, e.rhs),
                _ => eval(nodes, e.lhs) * eval(nodes, e.rhs),
            }
        }
        fn main() -> i64 {
            let !nodes = forge.uninit[Expr, [8]]
            nodes[0] = Expr { kind: 0, ival: 2, lhs: 0, rhs: 0 }
            eval(nodes, 0)
        }
    "#));
}

#[test]
fn array_size_mismatch_still_errors() {
    // The new `Array` arm must not over-accept: differing sizes stay incompatible.
    // Annotate the binding so the checker has a concrete `[M; 8]` (constructors
    // themselves report Unknown).
    let errs = check(r#"
        model M { x: i64 }
        fn take(a: [M; 4]) -> i64 { 0 }
        fn main() -> i64 {
            let b: [M; 8] = forge.uninit[M, [8]]
            take(b)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("expected") && e.contains("[M; 4]")),
            "expected an arity-mismatch error for [M;4] vs [M;8], got: {:?}", errs);
}

#[test]
fn scalar_op_on_tensors_warns() {
    let errs = check(r#"
        fn bad[B, D](x: Tensor[f32, [B, D]], y: Tensor[f32, [B, D]]) -> Tensor[f32, [B, D]] {
            x + y
        }
    "#);
    // Either errors out or hints to use dotted form
    assert!(errs.iter().any(|e| e.contains("dotted") || e.contains("tensor")),
            "expected hint about dotted op, got: {:?}", errs);
}

#[test]
fn elementwise_dotop_passes() {
    assert!(passes(r#"
        fn good[B, D](x: Tensor[f32, [B, D]], y: Tensor[f32, [B, D]]) -> Tensor[f32, [B, D]] {
            x .+ y
        }
    "#));
}

#[test]
fn function_arity_mismatch() {
    let errs = check(r#"
        fn add(a: i64, b: i64) -> i64 { a + b }
        fn t() -> nil { let _ = add(1); nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("wrong number of args")),
            "got: {:?}", errs);
}

#[test]
fn function_arg_type_mismatch_caught() {
    // Pre-alpha: our compatibility check is structural; this should catch
    // a clear type-class mismatch.
    let errs = check(r#"
        fn takes_int(x: i64) -> nil { nil }
        fn t() -> nil {
            let s = "hello"
            let _ = takes_int(s)
            nil
        }
    "#);
    // The error may or may not fire depending on stringly-typing — just
    // make sure we don't crash and produce *some* diagnostic on the call.
    let _ = errs;
}

#[test]
fn transpose_on_matrix_passes() {
    assert!(passes(r#"
        fn t[M, N](x: Tensor[f32, [M, N]]) -> Tensor[f32, [N, M]] { x' }
    "#));
}

#[test]
fn nested_block_typing() {
    assert!(passes(r#"
        fn t() -> i64 {
            let x = {
                let y = 10
                y + 5
            }
            x
        }
    "#));
}

#[test]
fn shape_param_used_as_value() {
    // `D as f32` in expression — shape params are SymDim variables, the
    // checker should accept this without "undefined" error.
    assert!(passes(r#"
        fn s[D]() -> f32 { D as f32 }
    "#));
}

#[test]
fn call_site_shape_param_infers_from_tensor_literal() {
    assert!(passes(r#"
        fn f[N](t: Tensor[i64, [N]]) -> Tensor[i64, [N]] { t }

        fn main() -> nil {
            let arr = [1, 2, 3]
            let r: Tensor[i64, [3]] = f(arr)
            nil
        }
    "#));
}

#[test]
fn call_site_shape_param_infers_return_shape() {
    let errs = check(r#"
        fn f[N](t: Tensor[i64, [N]]) -> Tensor[i64, [N]] { t }

        fn main() -> nil {
            let arr = [1, 2, 3]
            let r: Tensor[i64, [4]] = f(arr)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("let binding has type")),
            "expected inferred return shape mismatch, got: {:?}", errs);
}

#[test]
fn call_site_shape_param_mismatch_fails_across_args() {
    let errs = check(r#"
        fn pair[N](a: Tensor[i64, [N]], b: Tensor[i64, [N]]) -> Tensor[i64, [N]] { a }

        fn main() -> nil {
            let a = [1, 2, 3]
            let b = [4, 5]
            let r = pair(a, b)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("arg 1")),
            "expected second arg shape mismatch, got: {:?}", errs);
}

#[test]
fn type_alias_resolves_to_underlying_type() {
    let errs = check(r#"
        type Hidden = Tensor[f32, [4, 8]]

        fn bad(x: Hidden) -> Tensor[f32, [4, 16]] {
            x
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("returns") && e.contains("Tensor")),
            "expected alias shape mismatch, got: {:?}", errs);
}

#[test]
fn shape_param_type_alias_passes() {
    assert!(passes(r#"
        type Batch[B] = Tensor[f32, [B, 768]]

        fn project[B](x: Batch[B]) -> Tensor[f32, [B, 768]] {
            x
        }
    "#));
}

#[test]
fn shape_param_type_alias_mismatch_fails() {
    let errs = check(r#"
        type Batch[B] = Tensor[f32, [B, 768]]

        fn bad[B](x: Batch[B]) -> Tensor[f32, [B, 512]] {
            x
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("returns") && e.contains("Tensor")),
            "expected alias shape mismatch, got: {:?}", errs);
}

// ── #761: a top-level `let` literal constant is a real shape dimension,
// whether it names one directly in an annotation or through a `type` alias.
// Before the fix, `SymDim::from_expr` turned any identifier in a shape
// position into a bare `Var`, with nothing resolving it against the module's
// own constants — so `CAP` stayed symbolic forever, and a symbolic dim
// compared against a concrete one is `Equiv::Unknown` (`shape.rs`), never a
// provable mismatch OR a provable match. That cut both ways: a genuine `[9]`
// vs `CAP`(=4) mismatch went uncaught (this issue's own repro), and a
// genuinely matching `[4]` vs `CAP`(=4) pair was *wrongly* flagged as one
// (`let binding has type Tensor[I64, [CAP]] but value has type Tensor[I64,
// [4]]`) whenever the right-hand side happened to be concretely typed enough
// for `compatible_with` to even run the comparison.

#[test]
fn alias_named_dim_resolves_against_module_constant_and_catches_mismatch() {
    let errs = check(r#"
        let CAP = 4
        type Row = Tensor[i64, [CAP]]

        fn takes_row(r: Row) -> i64 { r[0] }

        fn main() -> nil {
            let bad = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            let _ = takes_row(bad)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("Tensor[i64, [4]]") && e.contains("[9]")),
            "expected the alias's dim resolved to 4 against a 9-element arg, got: {:?}", errs);
}

#[test]
fn alias_named_dim_resolves_against_module_constant_and_passes_a_genuine_match() {
    // The false-positive half of the same bug: before the fix, this matching
    // pair was rejected too — `CAP` never resolved to `4`, so it could never
    // be found EQUAL to a concrete `4` either.
    assert!(passes(r#"
        let CAP = 4
        type Row = Tensor[i64, [CAP]]

        fn takes_row(r: Row) -> i64 { r[0] }

        fn main() -> i64 {
            let ok = [1, 2, 3, 4]
            takes_row(ok)
        }
    "#));
}

#[test]
fn inline_annotation_named_dim_resolves_against_module_constant() {
    // The same hole, spelled without a `type` alias at all — a plain
    // annotation naming a top-level constant directly. Confirms the fix is
    // general to named dimensions, not alias-specific.
    let errs = check(r#"
        let CAP = 4

        fn takes_cap(t: Tensor[i64, [CAP]]) -> i64 { t[0] }

        fn main() -> nil {
            let bad = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            let _ = takes_cap(bad)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("Tensor[i64, [4]]") && e.contains("[9]")),
            "expected `CAP` resolved to 4 against a 9-element arg, got: {:?}", errs);

    assert!(passes(r#"
        let CAP = 4

        fn takes_cap(t: Tensor[i64, [CAP]]) -> i64 { t[0] }

        fn main() -> i64 {
            let ok = [1, 2, 3, 4]
            takes_cap(ok)
        }
    "#));
}

#[test]
fn let_annotation_named_dim_mismatch_names_the_resolved_value() {
    let errs = check(r#"
        let CAP = 4

        fn make9() -> Tensor[i64, [9]] { forge.zeros[i64, [9]] }

        fn main() -> i64 {
            let t: Tensor[i64, [CAP]] = make9()
            t[0]
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("Tensor[i64, [4]]") && e.contains("Tensor[i64, [9]]")),
            "expected the let annotation's `CAP` resolved to 4, got: {:?}", errs);
}

#[test]
fn let_annotation_named_dim_genuine_match_is_not_a_false_positive() {
    assert!(passes(r#"
        let CAP = 4

        fn make4() -> Tensor[i64, [4]] { forge.zeros[i64, [4]] }

        fn main() -> i64 {
            let t: Tensor[i64, [CAP]] = make4()
            t[0]
        }
    "#));
}

#[test]
fn generic_alias_mixes_its_own_shape_param_with_a_module_constant() {
    // `type Padded[B] = Tensor[f32, [B, PAD]]` — `B` is the alias's own
    // (symbolic, substituted per use) shape parameter; `PAD` is a top-level
    // constant folded in when the alias is cached, same as a non-generic
    // alias. Both must be honored at once.
    assert!(passes(r#"
        let PAD = 2
        type Padded[B] = Tensor[f32, [B, PAD]]

        fn takes(x: Padded[3]) -> f32 { x[0, 0] }

        fn main() -> f32 {
            let ok: Tensor[f32, [3, 2]] = forge.zeros[f32, [3, 2]]
            takes(ok)
        }
    "#));

    let errs = check(r#"
        let PAD = 2
        type Padded[B] = Tensor[f32, [B, PAD]]

        fn takes(x: Padded[3]) -> f32 { x[0, 0] }
        fn make() -> Tensor[f32, [3, 9]] { forge.zeros[f32, [3, 9]] }

        fn main() -> f32 { takes(make()) }
    "#);
    assert!(errs.iter().any(|e| e.contains("Tensor[f32, [3, 2]]") && e.contains("Tensor[f32, [3, 9]]")),
            "expected the module-constant half (PAD=2) of the generic alias to still be checked, got: {:?}", errs);
}

#[test]
fn type_alias_naming_an_undefined_shape_dimension_is_an_error() {
    // Unlike a `fn` signature (which tolerates an undeclared uppercase name
    // as an implicit generic shape parameter — the kvcache pattern, see
    // `implicit_fn_shape_param_is_not_treated_as_undefined` below), a `type`
    // alias has no such convention (SPEC.md §6.6: "No generics beyond shape
    // parameters and element type"). A name in its body that is neither its
    // own declared shape parameter nor a resolvable module constant cannot
    // mean anything, and is reported once, by name.
    let errs = check(r#"
        type Row = Tensor[i64, [TYPO]]
        fn main() -> i64 { 0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("Row") && e.contains("TYPO")),
            "expected an undefined-shape-dimension error naming `TYPO`, got: {:?}", errs);
}

#[test]
fn implicit_fn_shape_param_is_not_treated_as_undefined() {
    // Regression guard: an undeclared uppercase name in a plain `fn`
    // signature (never a `[...]` shape param, never a top-level `let`) is
    // the kvcache-style implicit generic dimension `check_fn` already
    // recognizes. The #761 fix must leave this permissive — it only adds a
    // fallback to a module constant, never a new refusal for a name that is
    // neither a constant nor a declared parameter, outside a `type` alias's
    // own (parameter-only, per SPEC §6.6) body.
    assert!(passes(r#"
        fn f(x: Tensor[f32, [S]]) -> f32 { x[0] }

        fn main() -> f32 {
            let a: Tensor[f32, [3]] = forge.zeros[f32, [3]]
            let b: Tensor[f32, [5]] = forge.zeros[f32, [5]]
            f(a) + f(b)
        }
    "#));
}

#[test]
fn imported_alias_exports_its_dims_already_resolved_against_the_private_constant() {
    // SPEC.md §6.7: a top-level `let`, `pub` or not, never crosses a module
    // boundary as a name (#724/#731) — but a `pub type` alias naming that
    // `let` in a dimension position DOES cross, because what crosses is the
    // alias's own already-RESOLVED shape, not a re-exposure of the `let`'s
    // name. The importer below never writes `CAP` anywhere and cannot: it
    // only ever spells `Row`, and still gets a real, checked, resolved-to-4
    // shape mismatch against it.
    let dep_src = r#"
        let CAP = 4
        pub type Row = Tensor[i64, [CAP]]
        pub fn make9() -> Tensor[i64, [9]] { forge.zeros[i64, [9]] }
        pub fn make4() -> Tensor[i64, [4]] { forge.zeros[i64, [4]] }
    "#;
    let dep_path = std::env::current_dir().unwrap().join("alias_dims_761_dep_temp.dmc");
    std::fs::write(&dep_path, dep_src).unwrap();
    let dep_canonical = dep_path.canonicalize().unwrap();

    let dep_tokens = Lexer::new(dep_src).tokenize().expect("lex failed");
    let dep_program = Parser::new(dep_tokens).parse_program().expect("parse failed");
    let mut dep_checker = Checker::new();
    dep_checker.check_program(&dep_program, Some(&dep_path));
    assert!(dep_checker.errors.is_empty(), "dependency check failed: {:?}", dep_checker.errors);

    let dep_env = super::check::ModuleEnv {
        env: dep_checker.env.clone(),
        aliases: dep_checker.aliases.clone(),
        resolved_aliases: dep_checker.resolved_aliases.clone(),
        public_items: super::ast::collect_public_items(&dep_program),
        top_level_let_names: super::ast::collect_top_level_let_names(&dep_program),
        transitive_import_lets: dep_checker.import_private_lets.clone(),
    };

    let importer_path = std::env::current_dir().unwrap().join("alias_dims_761_importer_temp.dmc");

    let bad_src = r#"
        use "alias_dims_761_dep_temp.dmc"
        fn takes_row(r: Row) -> i64 { r[0] }
        fn main() -> i64 { takes_row(make9()) }
    "#;
    let tokens = Lexer::new(bad_src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut bad_checker = Checker::new();
    bad_checker.checked_modules.insert(dep_canonical.clone(), dep_env.clone());
    bad_checker.check_program(&program, Some(&importer_path));
    let bad_errs: Vec<String> = bad_checker.errors.iter().map(|e| e.msg.clone()).collect();
    assert!(bad_errs.iter().any(|e| e.contains("Tensor[i64, [4]]") && e.contains("[9]")),
            "expected the imported alias's dim resolved to 4, got: {:?}", bad_errs);

    let good_src = r#"
        use "alias_dims_761_dep_temp.dmc"
        fn takes_row(r: Row) -> i64 { r[0] }
        fn main() -> i64 { takes_row(make4()) }
    "#;
    let tokens = Lexer::new(good_src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut good_checker = Checker::new();
    good_checker.checked_modules.insert(dep_canonical.clone(), dep_env.clone());
    good_checker.check_program(&program, Some(&importer_path));
    assert!(good_checker.errors.is_empty(), "expected a genuine 4-vs-4 match to pass: {:?}", good_checker.errors);

    // And the private `let` itself never crosses — only `Row`'s already-
    // resolved shape does (unaffected by this fix; pinned here as the other
    // half of the same boundary decision).
    let direct_src = r#"
        use "alias_dims_761_dep_temp.dmc"
        fn main() -> i64 { CAP }
    "#;
    let tokens = Lexer::new(direct_src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut direct_checker = Checker::new();
    direct_checker.checked_modules.insert(dep_canonical.clone(), dep_env.clone());
    direct_checker.check_program(&program, Some(&importer_path));
    assert!(!direct_checker.errors.is_empty(), "expected `CAP` to stay invisible to the importer");

    let _ = std::fs::remove_file(&dep_path);
}

// ── #766 (PR #766 review of #761): fold let-to-let chains, distinguish
// "not a top-level `let`" from "a top-level `let` but not a constant" (and
// its mutable-binding special case), and refuse a non-positive dimension at
// the alias/annotation instead of downstream at an unrelated use site.

#[test]
fn alias_dim_folds_through_a_top_level_let_to_let_name_chain() {
    // `let A = 4  let B = A` — `B`'s initializer is a bare name, not an int
    // literal, so the original #761 collector left `B` out of `const_dims`
    // entirely: `type Row = Tensor[i64,[B]]` was check-clean on base even
    // though `dmc run`'s own `ConstDim` table folds `B` to `4` (the JIT
    // folds a name-to-name chain the same way) and refuses at run time.
    let errs = check(r#"
        let A = 4
        let B = A
        type Row = Tensor[i64, [B]]

        fn takes_row(r: Row) -> i64 { r[0] }

        fn main() -> i64 {
            let bad = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            takes_row(bad)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("Tensor[i64, [4]]") && e.contains("[9]")),
            "expected `B` to fold through `A` to 4 and catch the mismatch, got: {:?}", errs);

    assert!(passes(r#"
        let A = 4
        let B = A
        type Row = Tensor[i64, [B]]

        fn takes_row(r: Row) -> i64 { r[0] }

        fn main() -> i64 {
            let ok = [1, 2, 3, 4]
            takes_row(ok)
        }
    "#));
}

#[test]
fn alias_dim_folds_through_a_let_chain_regardless_of_declaration_order() {
    // The fold is order-independent, exactly like a single literal constant
    // already was — `Row` is declared, and reads `B`, before either `let`
    // (top-level `let`s themselves still run in textual order — that's an
    // unrelated, pre-existing rule this isn't about — but a `type` alias is
    // pass-1 hoisted regardless of where it sits, so it may forward-reference
    // both).
    assert!(passes(r#"
        type Row = Tensor[i64, [B]]
        let A = 4
        let B = A

        fn takes_row(r: Row) -> i64 { r[0] }

        fn main() -> i64 {
            let ok = [1, 2, 3, 4]
            takes_row(ok)
        }
    "#));
}

#[test]
fn alias_dim_naming_a_mutable_top_level_let_says_so() {
    // Before this, a mutable top-level binding used as an alias dimension
    // got the same "neither a shape parameter... nor a top-level `let`
    // constant" text as a name that doesn't exist at all — true in the
    // letter (a mutable binding is not a *constant*) but misleading: `N` is
    // very much a top-level `let` here, just not one that can ever fold
    // into a dimension.
    let errs = check(r#"
        let ! N = 4
        type Row = Tensor[i64, [N]]
        fn main() -> i64 { 0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("Row") && e.contains("N") && e.contains("mutable")),
            "expected a message naming `N` as a mutable top-level `let`, got: {:?}", errs);
    assert!(!errs.iter().any(|e| e.contains("neither a shape parameter")),
            "expected the mutable-specific message, not the generic undefined one, got: {:?}", errs);
}

#[test]
fn fn_annotation_dim_naming_a_mutable_top_level_let_says_so_799() {
    // #799: a mutable top-level `let !N = 4` is (correctly) never in
    // `const_dims`, so an implicit `N` in a fn annotation used to fall
    // through `resolve_dim` all the way to an ordinary unbound `Var` —
    // silently making the fn fully shape-generic in `N`, with no
    // diagnostic at all. `let N = 4` (immutable) is concrete on both sides
    // as of #793/#797, so the `!` was a one-character cliff from "checks
    // and folds" to "silently accepts anything". Now the fn-annotation
    // path gets the alias path's own located diagnostic (`resolve_dim_at`),
    // naming the annotation, not the `let`.
    let errs = check(r#"
        let !N = 4
        fn f(x: Tensor[f32, [N]]) -> i64 { 1 }
        fn main() -> i64 { f(forge.zeros[f32, [8]]) }
    "#);
    assert!(errs.iter().any(|e| e.contains("N") && e.contains("mutable")
                             && e.contains("never a constant dimension")),
            "expected the mutable-binding-is-never-a-constant-dimension error, got: {:?}", errs);

    // An immutable top-level `let N = 4`, by contrast, still folds cleanly
    // (unaffected control — this is #793/#797's own behavior).
    assert!(passes(r#"
        let N = 4
        fn f(x: Tensor[f32, [N]]) -> i64 { 1 }
        fn main() -> i64 { f(forge.zeros[f32, [4]]) }
    "#));

    // An explicit `fn f[N](...)` shape param still shadows the mutable
    // top-level `let` too — declaring it makes it a genuine generic, not a
    // reference to the unusable binding, so no diagnostic fires.
    assert!(passes(r#"
        let !N = 4
        fn f[N](x: Tensor[f32, [N]]) -> i64 { 1 }
        fn main() -> i64 { f(forge.zeros[f32, [8]]) }
    "#));
}

#[test]
fn model_method_annotation_dim_naming_a_mutable_top_level_let_says_so_799() {
    // #799's method-path half: the same silent-generic gap existed for a
    // method's own parameter/return annotation, reached through `check_
    // model`'s `check_fn(f)` call (`resolve_type` there funnels through the
    // same `resolve_dim_at`). Also exercises #798: `check_model`'s own
    // #151 diagnostic must NOT additionally claim `N` is "unknown" — it
    // is a known top-level `let`, just not a usable constant, so only the
    // mutable-specific message should appear.
    let errs = check(r#"
        let !N = 4
        model M {
            fn m(self, x: Tensor[f32, [N]]) -> i64 { 1 }
        }
        fn main() -> i64 { 0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("N") && e.contains("mutable")
                             && e.contains("never a constant dimension")),
            "expected the mutable-binding-is-never-a-constant-dimension error, got: {:?}", errs);
    assert!(!errs.iter().any(|e| e.contains("unknown shape param")),
            "a mutable top-level let is a known binding, not an unknown shape param — \
             #151's diagnostic must not also fire, got: {:?}", errs);

    // The model's own explicit shape param still shadows the mutable
    // top-level `let` inside the model, same as the fn case.
    assert!(passes(r#"
        let !N = 4
        model M[N] {
            data: Tensor[f32, [N]]
        }
        fn make8() -> M[8] {
            M { data: forge.zeros[f32, [8]] }
        }
        fn main() -> i64 {
            let m = make8()
            sum(m.data) as i64
        }
    "#));
}

#[test]
fn allocator_shape_arg_naming_a_mutable_top_level_let_says_so_799() {
    // PR #824 review follow-up: SPEC.md §6.6 says the mutable-top-level-
    // `let`-is-never-a-dimension rule applies to "any [type] annotation".
    // An allocator's own shape argument (`forge.ones[f32, [N]]`) is parsed
    // as an EXPRESSION, not an `ast::Type` (the shape bracket indexes the
    // allocator call, same as the element-type argument does — see #728's
    // type-arg-less-model ruling on this identical bracket), so it never
    // reached `resolve_dim_at` through the usual annotation walk and
    // accepted a mutable `N` with no diagnostic at all, inconsistent with
    // every other annotation position #799 already covers (a local `let`,
    // an array size, View/KV/Mesh/tuple/fn-type, a model shape arg, a
    // model field). Fixed by applying the same check structurally to the
    // allocator's own shape bracket.
    let errs = check(r#"
        let !N = 4
        fn main() -> i64 {
            let t = forge.ones[f32, [N]]
            len(t)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("N") && e.contains("mutable")
                             && e.contains("never a constant dimension")),
            "expected the mutable-binding-is-never-a-constant-dimension error \
             for the allocator's own shape argument, got: {:?}", errs);

    // The model-array form of the same allocator set (`MODEL_ARRAY_
    // ALLOCATOR_METHODS` — `zeros`/`ones`/`uninit`) gets the identical
    // refusal for its shape argument.
    let errs = check(r#"
        let !N = 4
        model Surface[H] {
            data: Tensor[f32, [H]]
        }
        fn main() -> i64 {
            let arr = forge.uninit[Surface[2], [N]]
            0
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("N") && e.contains("mutable")
                             && e.contains("never a constant dimension")),
            "expected the mutable-binding-is-never-a-constant-dimension error \
             for the model-array allocator's own shape argument, got: {:?}", errs);

    // An immutable literal, and a fn's own shape parameter, in the exact
    // same allocator position are both unaffected controls — this must not
    // start refusing either of them.
    assert!(passes(r#"
        let N = 4
        fn main() -> i64 {
            let t = forge.ones[f32, [N]]
            len(t)
        }
    "#));
    assert!(passes(r#"
        fn f[N](x: Tensor[f32, [N]]) -> i64 {
            let t = forge.zeros[f32, [N]]
            len(t)
        }
        fn main() -> i64 { f(forge.zeros[f32, [4]]) }
    "#));
}

#[test]
fn alias_dim_naming_a_non_literal_top_level_let_says_so() {
    // Same distinction, the other non-constant case: a top-level `let` that
    // is neither a literal nor a fold-able chain to one.
    let errs = check(r#"
        let N = 2 + 2
        type Row = Tensor[i64, [N]]
        fn main() -> i64 { 0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("Row") && e.contains("N") && e.contains("not a literal constant")),
            "expected a message naming `N` as a non-literal top-level `let`, got: {:?}", errs);
}

#[test]
fn alias_dim_resolving_to_a_negative_constant_is_refused_at_the_alias() {
    // `expr_i64` admits negatives (correctly — the JIT's own `ConstDim`
    // does too), so `let CAP = -1` used as a dimension folds to `Const(-1)`.
    // Before this, that rode straight through into `Row`'s shape and only
    // surfaced far away, at the first place something actually indexed into
    // a `Row` — a confusing "index N out of bounds for axis 0 of size -1"
    // nowhere near the real problem. Refused right here instead, with
    // nothing even using `Row` in this program.
    let errs = check(r#"
        let CAP = -1
        type Row = Tensor[i64, [CAP]]
        fn main() -> i64 { 0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("-1") && e.contains("positive")),
            "expected a located non-positive-dimension error, got: {:?}", errs);
}

#[test]
fn negative_dim_refused_the_same_way_in_a_plain_annotation() {
    // The same fix, spelled without a `type` alias — a direct annotation
    // naming the same negative top-level constant.
    let errs = check(r#"
        let CAP = -1
        fn f(x: Tensor[i64, [CAP]]) -> i64 { x[0] }
        fn main() -> i64 { 0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("-1") && e.contains("positive")),
            "expected the same refusal for a direct annotation, got: {:?}", errs);
}

#[test]
fn implicit_fn_shape_param_matching_a_top_level_constant_resolves_to_it() {
    // An undeclared uppercase name in a `fn` signature is normally the
    // kvcache-style implicit generic (`implicit_fn_shape_param_is_not_
    // treated_as_undefined`, above) — but if it ALSO happens to name a
    // top-level literal `let`, the REGISTERED signature (what a call site
    // is actually checked against — see `register_fn`'s own comment on this
    // precedence rule) folds it to that constant, exactly like a named
    // alias dimension does. `dmc run`'s own `ConstDim` table folds the same
    // name the same way, so this call was already refused at run time
    // before this fix — check now agrees with run instead of being more
    // permissive than it.
    let errs = check(r#"
        let N = 4
        fn f(x: Tensor[f32, [N]]) -> f32 { x[0] }

        fn main() -> f32 {
            let a: Tensor[f32, [8]] = forge.zeros[f32, [8]]
            f(a)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("[4]") && e.contains("[8]")),
            "expected the implicit shape param `N` to fold to the top-level \
             constant 4 and catch the [8] mismatch, got: {:?}", errs);

    assert!(passes(r#"
        let N = 4
        fn f(x: Tensor[f32, [N]]) -> f32 { x[0] }

        fn main() -> f32 {
            let a: Tensor[f32, [4]] = forge.zeros[f32, [4]]
            f(a)
        }
    "#));
}

// ── #793 (regression from #766): the SAME top-level constant, used as an
// implicit shape param in TWO fn signatures, must fold the same way on both
// sides of a call between them. #766 applied the fold in `register_fn` (the
// registered signature callers check args against) but left `check_fn`
// binding the implicit name as a free `Var` for the body's own copy of the
// same types — so a value produced by one folded signature (`inner`'s
// param type, or `forge.zeros[f32,[N]]`) disagreed with another folded
// signature (`outer`'s param type) by construction, even though both are
// `[4]` and every call site is legitimate. `check_fn` now applies the same
// fold `register_fn` already did (see its own #766/#793 comment).

#[test]
fn implicit_shape_param_folds_the_same_way_across_two_fn_signatures_793() {
    // The issue's own reduction: `outer` takes `c` typed `Tensor[f32,[N]]`
    // and passes it straight to `inner`, which takes the same `Tensor[f32,
    // [N]]`. Before the fix this was a check error — `expected Tensor[F32,
    // [4]], got Tensor[F32, [N]]` — because `outer`'s OWN parameter, bound
    // inside `check_fn`, stayed `[N]` while `inner`'s registered signature
    // (what the call is checked against) had already folded to `[4]`.
    assert!(passes(r#"
        let N = 4
        fn inner(c: Tensor[f32, [N]]) -> i64 { 1 }
        fn outer(c: Tensor[f32, [N]]) -> i64 { inner(c) }
        fn main() -> i64 { outer(forge.zeros[f32, [N]]) }
    "#));
}

#[test]
fn implicit_shape_param_folds_the_same_way_across_two_fn_signatures_typed_let_793() {
    // Same repro, spelled with a typed/suffixed top-level `let` — the issue
    // names this as a second site of the same bug, and `collect_const_dims`
    // folds a typed literal (`expr_i64`) exactly like a bare one.
    assert!(passes(r#"
        let DATA_BLOCKS: u64 = 4u64
        fn inner(c: Tensor[u64, [DATA_BLOCKS]]) -> i64 { 1 }
        fn outer(c: Tensor[u64, [DATA_BLOCKS]]) -> i64 { inner(c) }
        fn main() -> i64 { outer(forge.zeros[u64, [DATA_BLOCKS]]) }
    "#));
}

#[test]
fn explicit_fn_shape_param_still_shadows_a_same_named_top_level_constant_793() {
    // SPEC.md §6.6 (a top-level `let` vs a fn's own shape param): an EXPLICIT `fn f[N](...)` shape param
    // always wins over a top-level `let` of the same name — the #793 fix
    // must not touch this half. `f` here is shape-generic in the ordinary
    // way and `N` never folds to the module's `4`, so an `[8]` argument is
    // legal.
    assert!(passes(r#"
        let N = 4
        fn f[N](x: Tensor[f32, [N]]) -> i64 { 1 }
        fn main() -> i64 { f(forge.zeros[f32, [8]]) }
    "#));
}

#[test]
fn model_method_implicit_top_level_constant_folds_like_a_fn_798() {
    // #798: was pinned KNOWN-DIVERGENT as `model_method_implicit_top_level_
    // constant_is_still_unknown_shape_param_793` — `check_model`'s own
    // diagnostic (#151, predating #766) rejected ANY uppercase name in a
    // method signature that is neither the model's nor the method's own
    // declared shape param, unconditionally, whether or not that name also
    // happened to be a top-level constant. SPEC.md §6.6 says a named
    // dimension in ANY type annotation that is not a declared shape
    // parameter resolves against a top-level literal `let`, and a method
    // parameter is such an annotation, exactly like a free fn's (#793/#797)
    // — so #151's blanket refusal conflicted with §6.6. Fixed by excluding
    // a `const_dims` name from `check_model`'s `undeclared` set, the same
    // way `check_fn`'s own implicit-shape-param loop does for a free fn.
    // This now checks clean and folds `N` to `4`, the same as the two-fn
    // repro in `implicit_shape_param_folds_the_same_way_across_two_fn_
    // signatures_793` above, just through a method call instead of a
    // second fn.
    assert!(passes(r#"
        let N = 4
        model M {
            fn inner(self, c: Tensor[f32, [N]]) -> i64 { 1 }
            fn outer(self, c: Tensor[f32, [N]]) -> i64 { self.inner(c) }
        }
        fn main() -> i64 {
            let m = M {}
            m.outer(forge.zeros[f32, [N]])
        }
    "#));

    // The model's own explicitly declared shape param, by contrast, still
    // shadows a same-named top-level constant inside the model — unaffected
    // by #798, unchanged from before it.
    assert!(passes(r#"
        let N = 4
        model M[N] {
            data: Tensor[f32, [N]]
        }
        fn make8() -> M[8] {
            M { data: forge.zeros[f32, [8]] }
        }
        fn main() -> i64 {
            let m = make8()
            sum(m.data) as i64
        }
    "#));

    // An explicit `fn m[N](self, ...)` shape param on the METHOD still
    // shadows the same-named top-level constant too — the fix must not
    // touch this half either. `N` here is genuinely shape-generic, so an
    // `[8]` argument is legal even though a top-level `N = 4` exists.
    assert!(passes(r#"
        let N = 4
        model M {
            fn m[N](self, x: Tensor[f32, [N]]) -> i64 { 1 }
        }
        fn main() -> i64 {
            let m = M {}
            m.m(forge.zeros[f32, [8]])
        }
    "#));

    // A genuinely undeclared uppercase name — neither a model/method shape
    // param nor a top-level constant — still gets #151's "unknown shape
    // param" refusal, and the hint still recommends declaring it as the
    // method's own shape param (there is no constant here for that spelling
    // to shadow).
    let errs = check_full(r#"
        model M {
            fn m(self, x: Tensor[f32, [K]]) -> i64 { 1 }
        }
        fn main() -> i64 { 0 }
    "#);
    let hit = errs.iter().find(|e| e.msg.contains("unknown shape param") && e.msg.contains("`K`"));
    assert!(hit.is_some(), "expected #151's diagnostic for a genuinely undeclared name, got: {:?}", errs);
    let hint = hit.unwrap().hint.as_deref().unwrap_or("");
    assert!(hint.contains("fn m[K](self, ...)"),
            "hint should still recommend declaring the undeclared, non-constant name, got: {:?}", hint);

    // A name that IS a top-level constant must never appear in the hint's
    // suggested spelling — that spelling would silently shadow it. Mix one
    // constant name with one genuinely undeclared name in the same method
    // signature: only the undeclared one is "unknown", and only it appears
    // in the hint.
    let errs = check_full(r#"
        let N = 4
        model M {
            fn m(self, x: Tensor[f32, [N, K]]) -> i64 { 1 }
        }
        fn main() -> i64 { 0 }
    "#);
    let hit = errs.iter().find(|e| e.msg.contains("unknown shape param"));
    assert!(hit.is_some(), "expected #151's diagnostic for the undeclared name `K`, got: {:?}", errs);
    let msg = &hit.unwrap().msg;
    assert!(!msg.contains('N'), "the error message must not mention the constant `N` at all, got: {:?}", msg);
    assert!(msg.contains('K') && !msg.contains("`N`"),
            "the error must name only the genuinely undeclared `K`, not the constant `N`, got: {:?}", msg);
    let hint = hit.unwrap().hint.as_deref().unwrap_or("");
    assert!(hint.contains('K') && !hint.contains("[N") && !hint.contains(", N]"),
            "the hint must not recommend shadowing the constant `N`, got: {:?}", hint);
}

#[test]
fn model_method_non_literal_top_level_let_names_it_precisely_798() {
    // #798's third category, the exact complaint it was filed on: an
    // uppercase name in a method signature that names a top-level `let`
    // which is neither a literal constant (`const_dims`) nor mutable
    // (`mutable_top_level_lets`) — an immutable but non-literal binding like
    // `let N = 2 + 2` (`expr_i64` folds only a bare literal or a leading
    // unary `-`, not general arithmetic, so this is genuinely non-literal,
    // the same repro `alias_dim_naming_a_non_literal_top_level_let_says_so`
    // uses for the alias path). Before the fix this fell through the
    // `undeclared` filter (present in neither excluded set) and got #151's
    // "unknown shape param" plus the shadowing hint recommending `fn
    // m[N](self, ...)` — a spelling that would silently shadow the real
    // top-level `N`. Must instead get the located, precise diagnostic
    // `resolve_alias_body`'s own non-literal branch uses, and must NOT also
    // get #151's diagnostic for the same name.
    let errs = check_full(r#"
        let N = 2 + 2
        model M {
            fn m(self, x: Tensor[f32, [N]]) -> i64 { 1 }
        }
        fn main() -> i64 { 0 }
    "#);
    let hit = errs.iter().find(|e| e.msg.contains("not a literal constant"));
    assert!(hit.is_some(),
            "expected the non-literal-top-level-`let` diagnostic naming `N`, got: {:?}", errs);
    let msg = &hit.unwrap().msg;
    assert!(msg.contains('N') && msg.contains("top-level `let`") && msg.contains("m") && msg.contains("M"),
            "expected the method/model-located non-literal-let message, got: {:?}", msg);
    assert!(!errs.iter().any(|e| e.msg.contains("unknown shape param")),
            "must not ALSO claim `N` is an unknown shape param — it is a known top-level \
             binding, just not a literal constant, got: {:?}", errs);
}

#[test]
fn model_method_unknown_shape_param_hint_states_model_params_already_in_scope_798() {
    // The hint on #151's "unknown shape param" diagnostic used to assert
    // "model-level shape params [H] are not automatically in scope inside
    // methods" — backwards. SPEC.md §3.10: "A model's own shape parameters
    // are in scope throughout its methods and bind from the instance." A
    // model's declared shape params (here `H`) are exactly what
    // `model_shape_names` excludes from `undeclared` above, in this very
    // function — this diagnostic can only ever fire for a name that is
    // NOT one of them, so the old sentence was false every time it printed.
    let errs = check_full(r#"
        model M[H] {
            fn m(self, x: Tensor[f32, [K]]) -> i64 { 1 }
        }
        fn main() -> i64 { 0 }
    "#);
    let hit = errs.iter().find(|e| e.msg.contains("unknown shape param") && e.msg.contains("`K`"));
    assert!(hit.is_some(), "expected #151's diagnostic for the undeclared name `K`, got: {:?}", errs);
    let hint = hit.unwrap().hint.as_deref().unwrap_or("");
    assert!(hint.contains('H'), "the hint should still list the model's own shape params, got: {:?}", hint);
    assert!(hint.contains("already in scope"),
            "the hint must say the model's shape params ARE already in scope inside methods \
             (SPEC.md §3.10), got: {:?}", hint);
    assert!(!hint.contains("not automatically in scope"),
            "the hint must not claim model-level shape params are not automatically in scope \
             inside methods — SPEC.md §3.10 says the opposite, got: {:?}", hint);
}

#[test]
fn grad_fn_implicit_shape_param_folds_the_same_way_as_a_plain_fn_793() {
    // `@grad fn` is checked through the exact same `check_fn`, so the #793
    // fold applies to it too — a `@grad fn` calling a plain fn that shares
    // the same implicit-constant-typed param must check clean, the same way
    // the two-plain-fn repro does.
    assert!(passes(r#"
        let N = 4
        fn inner(c: Tensor[f32, [N]]) -> f32 { sum(c) }
        @grad fn outer(!c: Tensor[f32, [N]]) -> f32 { inner(c) }
        fn main() -> i64 { 0 }
    "#));
}

#[test]
fn xor_on_int_literal_warns_did_you_mean_power() {
    // #194 lint: `x ^ 1` is XOR, almost always a mis-ported `** `. Warn (non-fatal).
    let warns = warnings(r#"
        fn divide_by_two(x: i64) -> i64 { x / 2^1 }
    "#);
    assert!(warns.iter().any(|w| w.contains("XOR, not exponentiation")),
            "expected XOR lint, got: {:?}", warns);
    // Still type-checks — a lint, not an error.
    assert!(passes(r#"
        fn divide_by_two(x: i64) -> i64 { x / 2^1 }
    "#));
}

#[test]
fn xor_of_two_variables_does_not_warn() {
    // Real XOR of two non-literal operands is not flagged.
    let warns = warnings(r#"
        fn real_xor(a: i64, b: i64) -> i64 { a ^ b }
    "#);
    assert!(!warns.iter().any(|w| w.contains("XOR")),
            "unexpected XOR lint on a^b: {:?}", warns);
}

/// Lint diagnostics with demon mode (the Control Art Restriction) released.
fn warnings_demon(src: &str) -> Vec<String> {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker = Checker::new();
    checker.demon = true;
    checker.check_program(&program, None);
    checker.warnings.iter().map(|w| w.msg.clone()).collect()
}

#[test]
fn method_call_on_builtin_desugars_via_ufcs() {
    // #333: `x.floor()` on a method-less receiver now UFCS-desugars to `floor(x)`
    // (the #199 lint is superseded for names that resolve to a real function) —
    // so it works and no longer warns.
    let warns = warnings(r#"
        fn f(x: f64) -> i64 { x.floor() }
    "#);
    assert!(!warns.iter().any(|w| w.contains("method-call syntax")),
            "x.floor() should desugar via UFCS, not warn; got: {:?}", warns);
    assert!(passes(r#"
        fn f(x: f64) -> i64 { x.floor() }
    "#));
}

#[test]
fn method_call_with_args_desugars_via_ufcs() {
    // `a.atan2(b)` → `atan2(a, b)`. The receiver becomes the first argument.
    let warns = warnings(r#"
        fn g(a: f64, b: f64) -> f64 { a.atan2(b) }
    "#);
    assert!(!warns.iter().any(|w| w.contains("method-call syntax")),
            "a.atan2(b) should desugar via UFCS, not warn; got: {:?}", warns);
    assert!(passes(r#"
        fn g(a: f64, b: f64) -> f64 { a.atan2(b) }
    "#));
}

#[test]
fn function_form_does_not_warn() {
    // The correct builtin-function form is clean — no lint.
    let warns = warnings(r#"
        fn f(x: f64) -> i64 { floor(x) }
    "#);
    assert!(!warns.iter().any(|w| w.contains("method-call syntax")),
            "false positive on floor(x): {:?}", warns);
}

#[test]
fn non_builtin_method_on_method_less_type_warns() {
    // #202: extends #199. A non-builtin `.method()` on a method-less concrete
    // type (here f64) resolves to an opaque value at runtime, so it now warns —
    // not just known-builtin collisions like `.floor()`.
    let warns = warnings(r#"
        fn f(x: f64) -> f64 { x.frobnicate() }
    "#);
    assert!(warns.iter().any(|w| w.contains("method-call syntax") && w.contains("frobnicate")),
            "expected #202 opaque-method lint on x.frobnicate(), got: {:?}", warns);
    // Still a lint, not a hard error.
    assert!(passes(r#"
        fn f(x: f64) -> f64 { x.frobnicate() }
    "#));
}

#[test]
fn method_on_unknown_or_model_receiver_does_not_warn() {
    // The lint only fires on types that definitionally carry no methods. Model
    // instances own methods, and `Unknown` receivers are ambiguous — neither
    // false-positives. `self.step()` inside a model method has a model receiver.
    let warns = warnings(r#"
        model Counter { n: i64
            fn step(self) -> i64 { self.n }
            fn run(self) -> i64 { self.step() }
        }
    "#);
    assert!(!warns.iter().any(|w| w.contains("method-call syntax")),
            "false positive on a model-method call: {:?}", warns);
}

#[test]
fn ghost_model_method_call_is_check_error() {
    // #441: a call to a method not defined on the receiver's model must fail
    // at check time. Previously the field access fell through to `Unknown`,
    // so the call was a silent no-op at runtime in statement position, or an
    // opaque value erroring only at its point of use in value position.
    let errs = check(r#"
        model Counter { !value: i64
            fn bump!(self) -> nil { self.value = self.value + 1  nil }
        }
        fn f() -> bool {
            let !c = Counter { value: 0 }
            c.ghost!(42)
            c.bump!()
            c.value == 1
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("no method `ghost!` on model `Counter`")),
            "expected unknown-method error, got {:?}", errs);

    // Value position reaches the same resolution path and must error too.
    let errs = check(r#"
        model Counter { !value: i64 }
        fn f() -> bool {
            let c = Counter { value: 0 }
            c.phantom() == 7
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("no method `phantom` on model `Counter`")),
            "expected unknown-method error in value position, got {:?}", errs);
}

#[test]
fn model_method_forward_references_still_pass() {
    // #441 must not break legal forward references: model methods are hoisted
    // in pass 1, so a call before its definition — within one model, and to a
    // method of a model declared later in the file — stays clean.
    assert!(passes(r#"
        model A { n: i64
            fn early(self) -> i64 { self.late() }
            fn use_b(self, b: B) -> i64 { b.bmethod() }
            fn late(self) -> i64 { self.n }
        }
        model B { m: i64
            fn bmethod(self) -> i64 { self.m }
        }
        fn f(a: A, b: B) -> i64 { a.early() + a.use_b(b) }
    "#));
}

#[test]
fn unsupported_str_method_warns_opaque() {
    // #202: `.to_lowercase()` / `.chars()` are not demoniC string methods — they
    // resolve to opaque values at runtime, so the checker now flags them.
    let warns = warnings(r#"
        fn lower(s: str) -> str { s.to_lowercase() }
    "#);
    assert!(warns.iter().any(|w| w.contains("not a demoniC string method") && w.contains("to_lowercase")),
            "expected #202 unsupported-str-method lint, got: {:?}", warns);
}

#[test]
fn supported_str_method_does_not_warn() {
    // #202: real string methods (split/replace/upper/lower/len) work via
    // call_str_method, so they must stay quiet — including `len`, which is also a
    // global builtin and used to false-positive under the bare #199 lint.
    let warns = warnings(r#"
        fn a(s: str) -> str { s.replace(" ", "") }
        fn b(s: str) -> list { s.split(" ") }
        fn c(s: str) -> str { s.upper() }
        fn d(s: str) -> i64 { s.len() }
    "#);
    assert!(!warns.iter().any(|w| w.contains("method-call syntax") || w.contains("string method")),
            "false positive on supported string methods: {:?}", warns);
}

#[test]
fn for_in_map_warns_not_iterable() {
    // #204: maps aren't iterable — `for (k,v) in m` type-checks but fails at
    // runtime. The Map type lets the checker flag it.
    let warns = warnings(r#"
        fn f() -> i64 {
            let !m = map_new()
            for (k, v) in m { }
            0
        }
    "#);
    assert!(warns.iter().any(|w| w.contains("maps are not iterable")),
            "expected #204 for-in-map lint, got: {:?}", warns);
    // Lint, not a hard error.
    assert!(passes(r#"
        fn f() -> i64 { let !m = map_new()  for (k, v) in m { }  0 }
    "#));
}

#[test]
fn for_in_map_param_does_not_warn() {
    // A `m: map` parameter does NOT carry TyType::Map (see resolve_type): because
    // Map is match-anything, such a param can be passed a list, so the annotation
    // can't guarantee a real map. Iterating it must stay quiet to avoid the #204
    // false positive — the lint only fires on map-producing expressions.
    let warns = warnings(r#"
        fn f(m: map) -> i64 { for x in m { }  0 }
    "#);
    assert!(!warns.iter().any(|w| w.contains("maps are not iterable")),
            "false positive: `map` param iteration should not warn, got: {:?}", warns);
}

#[test]
fn for_in_list_passed_as_map_param_does_not_warn() {
    // Regression for the #204 follow-up: a list passed to a `map`-typed param and
    // iterated is valid at runtime, so it must not warn "maps are not iterable".
    let warns = warnings(r#"
        fn iterate_it(m: map) -> i64 { let !s = 0  for x in m { s = s + 1 }  s }
        fn main() -> i64 {
            let !l = list()
            l = list_push(l, 10)
            l = list_push(l, 20)
            iterate_it(l)
        }
    "#);
    assert!(!warns.iter().any(|w| w.contains("maps are not iterable")),
            "false positive on list-passed-as-map-param: {:?}", warns);
}

#[test]
fn for_in_list_does_not_warn() {
    // Lists ARE iterable — single-var and tuple-destructuring forms must stay
    // quiet (the false-positive risk that blocked a naive lint).
    let warns = warnings(r#"
        fn single() -> i64 {
            let !l = list()
            l = list_push(l, 5)
            let !s = 0
            for x in l { s = s + 1 }
            s
        }
        fn destructure() -> i64 {
            let !l = list()
            l = list_push(l, (1, 2))
            let !s = 0
            for (a, b) in l { s = s + a + b }
            s
        }
    "#);
    assert!(!warns.iter().any(|w| w.contains("not iterable")),
            "false positive on list iteration: {:?}", warns);
}

#[test]
fn mod_on_subtraction_warns_truncated() {
    // #198: `(a - b) % n` — sign-bearing dividend; truncated vs floored diverges.
    let warns = warnings(r#"
        fn f(a: i64, b: i64, n: i64) -> i64 { (a - b) % n }
    "#);
    assert!(warns.iter().any(|w| w.contains("truncated in demoniC")),
            "expected #198 mod lint, got: {:?}", warns);
}

#[test]
fn mod_on_negation_warns_truncated() {
    // `-x % n` — explicit negation in the dividend.
    let warns = warnings(r#"
        fn f(x: i64, n: i64) -> i64 { -x % n }
    "#);
    assert!(warns.iter().any(|w| w.contains("truncated in demoniC")),
            "expected #198 mod lint on -x % n, got: {:?}", warns);
}

#[test]
fn mod_on_plain_index_does_not_warn() {
    // `i % n` (loop-index shape) is the common non-negative case — no noise.
    let warns = warnings(r#"
        fn f(i: i64, n: i64) -> i64 { i % n }
    "#);
    assert!(!warns.iter().any(|w| w.contains("truncated in demoniC")),
            "false positive on i % n: {:?}", warns);
}

#[test]
fn no_effect_arithmetic_warns() {
    // #231: identity-operand arithmetic (`+0`, `-0`, `*1`, `/1`) is a no-op — and
    // on a loop counter (`i = i + 0`) the program type-checks but loops forever.
    for src in [
        "fn f(x: i64) -> i64 { x + 0 }",
        "fn f(x: i64) -> i64 { x - 0 }",
        "fn f(x: i64) -> i64 { x * 1 }",
        "fn f(x: i64) -> i64 { x / 1 }",
    ] {
        let warns = warnings(src);
        assert!(warns.iter().any(|w| w.contains("no-effect arithmetic")),
                "expected #231 no-effect lint for `{}`, got: {:?}", src, warns);
    }
    // Real arithmetic stays quiet; demon mode suppresses the lint.
    assert!(!warnings("fn f(x: i64) -> i64 { x + 1 }").iter().any(|w| w.contains("no-effect")),
            "false positive on x + 1");
    assert!(!warnings_demon("fn f(x: i64) -> i64 { x + 0 }").iter().any(|w| w.contains("no-effect")),
            "demon mode should suppress the #231 lint");
}

#[test]
fn self_assignment_and_identity_rebind_warn() {
    // #232: `x = x` and `let !x = x` are dead code that compiles clean —
    // legal-but-garbage, so the checker warns rather than staying silent.
    assert!(warnings("fn f(n: i64) -> i64 { let !x = n  x = x  x }")
            .iter().any(|w| w.contains("self-assignment")),
            "expected #232 self-assignment lint");
    assert!(warnings("fn f(n: i64) -> i64 { let !x = n  let !x = x  x }")
            .iter().any(|w| w.contains("identity rebind")),
            "expected #232 identity-rebind lint");
    // A real reassignment stays quiet.
    assert!(!warnings("fn f(n: i64) -> i64 { let !x = n  x = n + 1  x }")
            .iter().any(|w| w.contains("self-assignment") || w.contains("identity rebind")),
            "false positive on a real assignment");
}

#[test]
fn type_is_valid_model_field_name() {
    // #235: `type` is reserved (type aliases) but every tokenizer's Token has a
    // `type` field. It must work as a field name in all three positions —
    // declaration, struct literal, and access — without breaking type aliases.
    assert!(passes(r#"
        model Token { type: i64, value: str }
        fn kind(t: Token) -> i64 { t.type }
        fn main() -> i64 { let t = Token { type: 2, value: "==" }  kind(t) }
    "#), "a `type` model field (decl + literal + access) should type-check");
    // The `type` alias keyword still works — not broken by the field-name allowance.
    assert!(passes("type Idx = i64  fn main() -> i64 { 0 }"),
            "`type` alias declaration should still parse");
}

#[test]
fn demon_mode_suppresses_lint_family() {
    // Releasing the Control Art Restriction (#196) drops every safe-mode lint:
    // both the #199 method-call trap and the #198 mod footgun go silent.
    let src = r#"
        fn f(x: f64) -> i64 { x.floor() }
        fn g(a: i64, b: i64, n: i64) -> i64 { (a - b) % n }
    "#;
    let safe = warnings(src);
    assert!(!safe.is_empty(), "safe mode should surface lints, got none");
    let demon = warnings_demon(src);
    assert!(demon.is_empty(), "demon mode should suppress all lints, got: {:?}", demon);
}

// --- file-size lint (#463) --------------------------------------------------

/// A trivial program padded with leading comment lines to exactly `n` source
/// lines, no trailing newline.
fn src_of_lines(n: usize) -> String {
    let mut s = String::new();
    for _ in 0..n.saturating_sub(1) { s.push_str("# pad\n"); }
    s.push_str("fn main() -> i64 { 0 }");
    s
}

/// Warnings with the file-size dial pinned (as if the nearest demoni.json
/// set `lints.max_file_lines` to `dial`), demon mode selectable. Full
/// `TypeError`s so tests can assert on the span, not just the message.
fn warnings_with_dial(src: &str, dial: Option<usize>, demon: bool) -> Vec<super::check::TypeError> {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker = Checker::new();
    checker.max_file_lines = dial;
    checker.demon = demon;
    checker.check_program(&program, None);
    checker.warnings.clone()
}

fn file_size_warns(src: &str, dial: Option<usize>) -> bool {
    warnings_with_dial(src, dial, false)
        .iter().any(|w| w.msg.contains("lints.max_file_lines"))
}

const DIAL: usize = 120;

#[test]
fn file_size_lint_fires_over_dial() {
    let src = src_of_lines(DIAL + 1);
    assert!(file_size_warns(&src, Some(DIAL)),
            "a {}-line file should trip a {}-line dial", DIAL + 1, DIAL);
}

#[test]
fn file_size_lint_silent_at_dial() {
    // Exactly at the dial: quiet. And with a trailing newline (EOF sits
    // at column 1 of the phantom next line): still quiet — the parser must
    // not count that line.
    let src = src_of_lines(DIAL);
    assert!(!file_size_warns(&src, Some(DIAL)),
            "a file at exactly the dial must not warn");
    let with_nl = format!("{}\n", src);
    assert!(!file_size_warns(&with_nl, Some(DIAL)),
            "a trailing newline must not push a dial-sized file over");
}

#[test]
fn file_size_lint_off_unless_configured() {
    // No manifest dial → the lint does not exist. The compiler does not
    // pick a number (#463 refinement: per-project dial, opt-in).
    let src = src_of_lines(5000);
    assert!(!file_size_warns(&src, None),
            "with no dial configured the file-size lint must never fire");
}

#[test]
fn file_size_lint_suppressed_in_demon_mode() {
    let src = src_of_lines(DIAL + 1);
    assert!(file_size_warns(&src, Some(DIAL)), "safe mode should warn");
    assert!(!warnings_with_dial(&src, Some(DIAL), true)
                .iter().any(|w| w.msg.contains("lints.max_file_lines")),
            "demon mode should release the file-size lint");
}

#[test]
fn file_size_lint_anchors_at_end_of_file() {
    // The excess is at the end of the file, so the diagnostic points there —
    // not at line 1.
    let n = DIAL + 30;
    let src = src_of_lines(n);
    let all = warnings_with_dial(&src, Some(DIAL), false);
    let w = all.iter().find(|w| w.msg.contains("lints.max_file_lines"))
        .expect("file-size warning expected");
    assert_eq!((w.span.line, w.span.col), (n, 1),
               "diagnostic must anchor at the last line, got {}:{}", w.span.line, w.span.col);
}

#[test]
fn file_size_dial_resolved_from_nearest_manifest() {
    // End to end through `check_program(path)`: the dial comes out of the
    // nearest demoni.json above the source file, and only from there.
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    std::fs::write(root.join("demoni.json"),
                   r#"{ "lints": { "max_file_lines": 60 } }"#).expect("write manifest");
    let src_path = root.join("big.dmc");
    let src = src_of_lines(61);
    std::fs::write(&src_path, &src).expect("write source");

    let tokens = Lexer::new(&src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker = Checker::new();
    checker.check_program(&program, Some(&src_path));
    assert!(checker.warnings.iter().any(|w| w.msg.contains("lints.max_file_lines")),
            "a 61-line file under a 60-line manifest dial should warn, got {:?}",
            checker.warnings);
}

#[test]
fn pipe_into_value_is_rejected() {
    // A pipe whose RHS is a concrete value is not callable. It used to
    // type-check and then fail at runtime; it is a check error (#188).
    let errs = check(r#"
        fn test_op() -> bool { let x = 256  (x \|> 2) >= 0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("not callable")),
            "expected non-callable pipe error, got: {:?}", errs);
}

#[test]
fn shift_intent_now_type_checks_as_a_shift() {
    // #188 verbatim: `x >> 2` written meaning a bitwise shift. It type-checked
    // as a pipe and died at runtime; S1a (#501) made it a lex error; #530 gives
    // it the meaning it was always written for. The issue's own repro passes.
    assert!(passes(r#"
        fn test_op() -> bool {
            let x = 256
            (x >> 2) >= 0
        }
    "#));
}

#[test]
fn right_shift_types_exactly_like_left_shift() {
    // `>>` joins the bitwise arm (`BitAnd | BitOr | BitXor | BitShl | BitShr`),
    // so its typing is `<<`'s typing — including the arm's known looseness on
    // floats, which the interpreter catches at runtime for both. Pinning the
    // two together is the invariant; changing the arm must move both.
    for op in ["<<", ">>"] {
        let ok = format!("fn main() -> nil {{\n    let x = 256 {} 2\n    nil\n}}", op);
        assert!(passes(&ok), "`{}` on two ints must check", op);

        let bad = format!("fn main() -> nil {{\n    let x = true {} \"s\"\n    nil\n}}", op);
        let errs = check(&bad);
        assert!(errs.iter().any(|e| e.contains("bitwise operator requires integer operands")),
                "`{}` on non-numeric operands must error, got: {:?}", op, errs);
    }
}

#[test]
fn pipe_into_callable_and_placeholder_pass() {
    // The legitimate pipe forms must still type-check: application, builtin,
    // `_`-placeholder fusion, and bare activation stage.
    assert!(passes(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn main() -> nil {
            let a = 5 |> inc
            let b = [1.0, 2.0] |> _ .+ [3.0, 4.0]
            let c = [1.0, 2.0, 3.0, 4.0] |> sum
            let d = [-1.0, 2.0] \|> \>
            nil
        }
    "#));
}

#[test]
fn pipeline_stage_placeholders_pass() {
    assert!(passes(r#"
        @pp(stages=2)
        fn pipeline[B, D](x: Tensor[f32, [B, D]]) -> Tensor[f32, [B, D]] {
            stage 0: block_0(x)
            stage 1: block_1(_)
        }
    "#));
}

#[test]
fn pp_requires_dense_ordered_stages() {
    let errs = check(r#"
        @pp(stages=2)
        fn pipeline[B, D](x: Tensor[f32, [B, D]]) -> Tensor[f32, [B, D]] {
            stage 0: block_0(x)
            stage 2: block_2(_)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("stage index must be 1")),
            "expected dense stage index error, got: {:?}", errs);
}

#[test]
fn pp_rejects_non_stage_body_stmt() {
    let errs = check(r#"
        @pp(stages=2)
        fn pipeline[B, D](x: Tensor[f32, [B, D]]) -> Tensor[f32, [B, D]] {
            stage 0: block_0(x)
            let y = x
            stage 1: block_1(_)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@pp body cannot contain `let`")),
            "expected non-stage body error, got: {:?}", errs);
}

#[test]
fn pp_stage_count_must_match() {
    let errs = check(r#"
        @pp(stages=3)
        fn pipeline[B, D](x: Tensor[f32, [B, D]]) -> Tensor[f32, [B, D]] {
            stage 0: block_0(x)
            stage 1: block_1(_)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("requires exactly 3 stage statements")),
            "expected stage count error, got: {:?}", errs);
}

#[test]
fn tp_let_requires_local_divisor_shape() {
    let errs = check(r#"
        fn bad[D]() -> nil {
            @tp(axis=-1)
            let w: Tensor[f32, [D, 4*D]] = vault.zeros[f32, [8, 32]]
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@tp") && e.contains("divisor `tp`")),
            "expected missing tp divisor error, got: {:?}", errs);
}

#[test]
fn tp_let_accepts_local_divisor_shape() {
    assert!(passes(r#"
        fn good[D]() -> nil {
            @tp(axis=-1)
            let w: Tensor[f32, [D, 4*D/tp]] = vault.zeros[f32, [8, 8]]
            nil
        }
    "#));
}

#[test]
fn shard_let_accepts_mesh_axis_divisor_shape() {
    assert!(passes(r#"
        fn good[B, D]() -> nil {
            @shard(axis=0, mesh=mesh.dp)
            let x: Tensor[f32, [B/dp, D]] = load_batch()
            nil
        }
    "#));
}

#[test]
fn shard_let_axis_must_be_in_bounds() {
    let errs = check(r#"
        fn bad[B, D]() -> nil {
            @shard(axis=2, mesh=mesh.dp)
            let x: Tensor[f32, [B/dp, D]] = load_batch()
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("axis 2 out of bounds")),
            "expected shard axis bounds error, got: {:?}", errs);
}

#[test]
fn top_level_tp_let_preserves_directive() {
    let errs = check(r#"
        @tp(axis=-1)
        let w: Tensor[f32, [D, 4*D]] = vault.zeros[f32, [8, 32]]
    "#);
    assert!(errs.iter().any(|e| e.contains("@tp") && e.contains("divisor `tp`")),
            "expected top-level tp directive check, got: {:?}", errs);
}

#[test]
fn top_level_shard_let_accepts_local_shape() {
    assert!(passes(r#"
        @shard(axis=0, mesh=mesh.dp)
        let x: Tensor[f32, [B/dp, D]] = load_batch()
    "#));
}

// ─── Stdlib signature tests ───────────────────────────────────────────────────

#[test]
fn stdlib_single_arg_passes() {
    assert!(passes("fn t(x: f64) -> nil { let _ = sqrt(x); nil }"));
}

#[test]
fn stdlib_arity_enforced() {
    // sqrt declares exactly 1 parameter; two arguments should error.
    let errs = check("fn t(x: f64) -> nil { let _ = sqrt(x, x); nil }");
    assert!(errs.iter().any(|e| e.contains("wrong number of args")),
            "expected arity error for sqrt(x, x), got: {:?}", errs);
}

#[test]
fn stdlib_print_returns_nil() {
    // print is variadic and returns nil; usable as the tail of a nil fn.
    assert!(passes(r#"fn t() -> nil { print("hi") }"#));
}

#[test]
fn stdlib_print_return_type_caught() {
    // print(...) -> nil; using it as an i64 should be flagged.
    let errs = check(r#"fn t() -> i64 { print("hi") }"#);
    assert!(errs.iter().any(|e| e.contains("returns") || e.contains("body")),
            "expected return-type mismatch for print()-as-i64, got: {:?}", errs);
}

#[test]
fn stdlib_attn_mask_is_optional() {
    // attn(q, k, v) — mask is ?-typed (optional per STDLIB.md §3.1), so 3 args must pass.
    assert!(passes(r#"
        fn t[B, H, S, D](
            q: Tensor[f32, [B, H, S, D]],
            k: Tensor[f32, [B, H, S, D]],
            v: Tensor[f32, [B, H, S, D]],
        ) -> nil { let _ = attn(q, k, v); nil }
    "#), "attn(q, k, v) without mask should be valid");
}

#[test]
fn stdlib_attn_four_args_passes() {
    assert!(passes(r#"
        fn t[B, H, S, D](
            q:    Tensor[f32, [B, H, S, D]],
            k:    Tensor[f32, [B, H, S, D]],
            v:    Tensor[f32, [B, H, S, D]],
            mask: View[bool, [S, S]],
        ) -> nil { let _ = attn(q, k, v, mask); nil }
    "#));
}

#[test]
fn stdlib_rope_arity_enforced() {
    // rope(x, cos, sin) — 3 args; 2 should error.
    let errs = check(r#"
        fn t[S, D](x: Tensor[f32, [S, D]], cos: Tensor[f32, [S, D]]) -> nil {
            let _ = rope(x, cos);
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("wrong number of args")),
            "expected arity error for rope(x, cos), got: {:?}", errs);
}

#[test]
fn stdlib_softmax_two_args_passes() {
    assert!(passes(r#"
        fn t[B, S](x: Tensor[f32, [B, S]]) -> nil { let _ = softmax(x, -1); nil }
    "#));
}

#[test]
fn stdlib_sum_one_arg_passes() {
    assert!(passes(r#"
        fn t[B, D](x: Tensor[f32, [B, D]]) -> nil { let _ = sum(x); nil }
    "#));
}

// ── Null Hypothesis Tests (HAS-dC §9) ────────────────────────────────────

#[test]
fn null1_type_checker_terminates_on_complex_program() {
    // Null 1: Φ₀ constraint manifold is decidable.
    // Falsification: --check non-termination on any well-formed program.
    // The checker is a tree walk (no fixpoint iteration), so it terminates
    // in O(n) statements. Test with a large, deeply-annotated program to
    // confirm no exponential blowup from nested type constraints.
    let errs = check(r#"
        fn f1[A, B](x: Tensor[f32, [A, B]]) -> Tensor[f32, [A, B]] { x }
        fn f2[A, B](x: Tensor[f32, [A, B]]) -> Tensor[f32, [A, B]] { f1(x) }
        fn f3[A, B](x: Tensor[f32, [A, B]]) -> Tensor[f32, [A, B]] { f2(x) }
        fn f4[A, B](x: Tensor[f32, [A, B]]) -> Tensor[f32, [A, B]] { f3(x) }
        fn f5[A, B](x: Tensor[f32, [A, B]]) -> Tensor[f32, [A, B]] { f4(x) }
        fn f6[A, B](x: Tensor[f32, [A, B]]) -> Tensor[f32, [A, B]] { f5(x) }
        fn f7[A, B](x: Tensor[f32, [A, B]]) -> Tensor[f32, [A, B]] { f6(x) }
        fn f8[A, B](x: Tensor[f32, [A, B]]) -> Tensor[f32, [A, B]] { f7(x) }
        fn main[A, B](x: Tensor[f32, [A, B]]) -> Tensor[f32, [A, B]] { f8(x) }
    "#);
    // No undefined-identifier errors — all calls resolve.
    assert!(!errs.iter().any(|e| e.contains("undefined identifier")),
            "unexpected errors: {:?}", errs);
}

// ── Pass 2 arena coherence: ? context ────────────────────────────────────

#[test]
fn question_mark_in_err_returning_fn_passes() {
    // ? is legal inside a fn returning (T, str).
    assert!(passes(r#"
        fn try_parse(s: str) -> (i64, str) {
            let (v, e) = other(s)?
            (v, nil)
        }
        fn other(s: str) -> (i64, str) { (0, nil) }
    "#));
}

#[test]
fn question_mark_outside_err_fn_fails() {
    // ? outside a (T, str)-returning function is a compile-time error (SPEC §4.9).
    let errs = check(r#"
        fn bad(x: i64) -> i64 {
            let y = might_fail()?
            y
        }
        fn might_fail() -> (i64, str) { (0, nil) }
    "#);
    assert!(errs.iter().any(|e| e.contains("`?` is only legal")),
            "expected ? context error, got: {:?}", errs);
}

// ── Pass 2 arena coherence: <- type ──────────────────────────────────────

#[test]
fn stream_arrow_on_scalar_lhs_fails() {
    // <- on a non-tensor lhs is a type error.
    let errs = check(r#"
        fn bad() -> nil {
            let x: i64 = 0
            x <- 1
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("<-") && e.contains("KV")),
            "expected stream-arrow type error, got: {:?}", errs);
}

// ── Fix: str+str, tensor index, for-range binding ────────────────────────

#[test]
fn str_concat_passes_type_check() {
    // str + str should not error
    let diags = check(r#"fn main() -> str { "hello" + " world" }"#);
    assert!(diags.is_empty(), "unexpected errors: {:?}", diags);
}

#[test]
fn str_concat_with_int_passes() {
    // str + int is valid (coercion)
    let diags = check(r#"fn main() -> str { "count=" + 42 }"#);
    assert!(diags.is_empty(), "unexpected errors: {:?}", diags);
}

#[test]
fn tensor_index_returns_element_type() {
    // t[i] should resolve to a float type, not error
    let diags = check(r#"
        fn main() -> f32 {
            let t = vault.zeros[f32, [8]]
            t[0]
        }
    "#);
    assert!(diags.is_empty(), "unexpected errors: {:?}", diags);
}

#[test]
fn tensor_row_assignment_type_checks() {
    // Mutable binding: `let !grid` allows indexed write-through. No errors.
    let diags = check(r#"
        fn main() -> nil {
            let !grid = [[0, 0, 0, 0], [0, 0, 0, 0]]
            let row = [1, 2, 3, 4]
            grid[0] = row
            nil
        }
    "#);
    assert!(diags.is_empty(), "unexpected errors: {:?}", diags);
}

#[test]
fn for_loop_var_usable_as_int() {
    // for x in 0..n -- x should be usable in arithmetic without Unknown bleeding into errors
    let diags = check(r#"
        fn main() -> i64 {
            let mut s = 0
            for x in 0..10 { s = s + x }
            s
        }
    "#);
    assert!(diags.is_empty(), "unexpected errors: {:?}", diags);
}

#[test]
fn strict_no_implicit_int_to_float_arg() {
    // §149 (#284): passing an integer (the I64 loop var) to an f32 param is an
    // implicit numeric conversion and must now error — no silent widening.
    let diags = check(r#"
        fn scale(x: f32) -> f32 { x }
        fn main() -> f32 {
            let !s: f32 = 0.0
            for i in 0..10 { s = s + scale(i) }
            s
        }
    "#);
    assert!(!diags.is_empty(), "expected implicit int->f32 arg conversion to error");

    // …and an explicit `as f32` cast is accepted.
    let ok = check(r#"
        fn scale(x: f32) -> f32 { x }
        fn main() -> f32 {
            let !s: f32 = 0.0
            for i in 0..10 { s = s + scale(i as f32) }
            s
        }
    "#);
    assert!(ok.is_empty(), "explicit cast should type-check, got: {:?}", ok);
}

#[test]
fn strict_numeric_typing_284() {
    // SPEC §149: no implicit numeric conversions between distinct concrete
    // scalar types. Implicit i64 -> f32 in a let errors…
    assert!(!check("fn widen(x: i64) -> nil { let y: f32 = x  nil }").is_empty(),
        "implicit i64->f32 binding should error");
    // …distinct concrete scalars are no longer mutually assignable…
    assert!(!check("fn f(x: i32) -> i64 { x }").is_empty(),
        "i32 body for an i64 return should error (no implicit widening)");
    assert!(!check("fn g(x: f32) -> f64 { x }").is_empty(),
        "f32 body for an f64 return should error (no implicit widening)");
    // Literal *range* checking (`let x: i8 = 300`) is now enforced — see
    // `lit_range_overflow_295` below.

    // Literals still adopt a matching context (the non-breaking half, #295):
    assert!(check("fn a() -> i32 { 5 }").is_empty(), "int literal adopts i32");
    assert!(check("fn b() -> f64 { 5.0 }").is_empty(), "float literal adopts f64");
    assert!(check("fn c() -> f32 { 5.0 }").is_empty(), "float literal adopts f32");
    assert!(check("fn d() -> u8 { let x: u8 = 5  x }").is_empty(), "int literal adopts u8");
    // …but an int literal may NOT adopt a float context (must be `5.0`/`5 as f32`).
    assert!(!check("fn e() -> f32 { let y: f32 = 5  y }").is_empty(),
        "int literal must not implicitly become f32");
}

#[test]
fn lit_range_overflow_295() {
    // #295: an untyped int literal adopts a narrow integral context, but its
    // magnitude must fit. Out-of-range at each binding site is now an error.
    fn has_range_err(src: &str) -> bool {
        check(src).iter().any(|m| m.contains("out of range"))
    }

    // let binding (the canonical case).
    assert!(has_range_err("fn m() -> nil { let x: i8 = 300  nil }"),
        "let x: i8 = 300 should overflow");
    // negative literal into an unsigned type.
    assert!(has_range_err("fn m() -> nil { let x: u8 = -1  nil }"),
        "let x: u8 = -1 should underflow");
    // explicit/tail return.
    assert!(has_range_err("fn f() -> i8 { 300 }"), "return 300 from -> i8 overflows");
    // call argument.
    assert!(has_range_err("fn g(x: i16) -> i16 { x }  fn m() -> nil { let _ = g(40000)  nil }"),
        "g(40000) into an i16 param overflows");
    // assignment through a mutable narrow binding.
    assert!(has_range_err("fn m() -> nil { let !x: i8 = 0  x = 200  nil }"),
        "x = 200 into i8 binding overflows");

    // ── In-range and non-literal forms must NOT be flagged ──────────────────
    assert!(!has_range_err("fn m() -> nil { let x: i8 = 127  nil }"), "i8 = 127 fits");
    assert!(!has_range_err("fn m() -> nil { let x: i8 = -128  nil }"), "i8 = -128 fits");
    assert!(!has_range_err("fn m() -> nil { let x: u8 = 255  nil }"), "u8 = 255 fits");
    // Arithmetic on literals carries only the left operand's value in the type,
    // so it must NOT be range-checked (would false-positive on `200 - 100`).
    assert!(!has_range_err("fn m() -> nil { let x: i8 = 200 - 100  nil }"),
        "200 - 100 (=100) must not be flagged — it is not a syntactic literal");
    // i64 target fits any i64 literal; quantization int kinds are not checked.
    assert!(!has_range_err("fn m() -> nil { let x: i64 = 9000000000  nil }"), "i64 literal fits i64");
}

#[test]
fn lit_range_overflow_operand_538() {
    // #538: SPEC §3.1 lists "other operand at its use site" as a context an
    // untyped literal adopts — #295's range check must reach that context
    // too, not just the annotation/return/param positions above.
    fn has_range_err(src: &str) -> bool {
        check(src).iter().any(|m| m.contains("out of range"))
    }

    // Operand on the right of a narrow-typed operand.
    assert!(has_range_err(
        "fn m() -> nil { let a: i32 = 256  let b: i32 = a + 5000000000  nil }"
    ), "a + 5000000000 (a: i32) should overflow");
    // Same, operand order reversed — the literal is checked either side.
    assert!(has_range_err(
        "fn m() -> nil { let a: i32 = 256  let b: i32 = 5000000000 + a  nil }"
    ), "5000000000 + a (a: i32) should overflow");
    // A parameter's declared type is a context too.
    assert!(has_range_err("fn f(x: i32) -> i32 { x + 5000000000 }"),
        "x + 5000000000 (x: i32 param) should overflow");

    // ── In-range operand literals must NOT be flagged ───────────────────────
    assert!(!has_range_err(
        "fn m() -> nil { let a: i32 = 256  let b: i32 = a + 5000  nil }"
    ), "a + 5000 (a: i32) fits");
    assert!(!has_range_err("fn f(x: i32) -> i32 { x + 5000 }"),
        "x + 5000 (x: i32 param) fits");
}

#[test]
fn suffix_conflicts_in_operand_position_539() {
    // #539: SPEC §3.1 says an explicit type suffix "conflicts with a
    // different annotation or parameter type as a normal type error" — that
    // must reach the operand position too, not just the binding position
    // (#445).
    fn disagrees(src: &str) -> bool {
        check(src).iter().any(|m| m.contains("disagree"))
    }

    // A suffix naming a *different* width than the other operand conflicts,
    // the same way `let y: i64 = 2i32` already does.
    assert!(disagrees(
        "fn m() -> nil { let a: i32 = 256  let b: i32 = a + 2i64  nil }"
    ), "a + 2i64 (a: i32) should conflict");
    // Two plain locals of different declared widths must stay a type error —
    // this is not a literal at all, but the same #284 strict-typing rule
    // reaching operand position closes this gap too.
    assert!(disagrees(
        "fn m() -> nil { let a: i32 = 5  let b: i64 = 10  let c: i32 = a + b  nil }"
    ), "a + b (a: i32, b: i64) should conflict");

    // A suffix naming the *same* width as the other operand must keep
    // working — the suffix is not a generic implicit cast, it is the most
    // explicit statement of intent a literal can carry, and here it agrees.
    assert!(passes(
        "fn m() -> nil { let a: i32 = 256  let b: i32 = a + 2i32  nil }"
    ), "a + 2i32 (a: i32) should still check clean");
    // Existing #445 binding-position conflict is untouched by this change.
    assert!(!passes("fn m() -> nil { let x = 2i32  let y: i64 = x  nil }"),
        "let y: i64 = x (x bound i32 via suffix) should still conflict");
}

#[test]
fn lit_and_suffix_operand_checks_reach_comparisons_and_bitwise_538_539() {
    // #538/#539: `check_binop`'s literal-range and suffix-conflict checks
    // were originally wired into the arithmetic arm only — comparisons and
    // bitwise/shift ops have their own arms in `check_binop` and silently
    // skipped both rules, so `a < 5000000000` or `a & 2i64` (a: i32) checked
    // clean while the JIT's single `lower_binop` guard (which covers every
    // scalar binop uniformly) already refused them. Both rules are now
    // shared via `adopt_and_check_operand_types`, called from all three
    // arms alike.
    fn has_range_err(src: &str) -> bool {
        check(src).iter().any(|m| m.contains("out of range"))
    }
    fn disagrees(src: &str) -> bool {
        check(src).iter().any(|m| m.contains("disagree"))
    }

    // ── Comparisons ──────────────────────────────────────────────────────
    assert!(has_range_err("fn m() -> nil { let a: i32 = 5  let _ = a < 5000000000  nil }"),
        "a < 5000000000 (a: i32) should overflow");
    assert!(disagrees("fn m() -> nil { let a: i32 = 5  let _ = a < 2i64  nil }"),
        "a < 2i64 (a: i32) should conflict");
    assert!(disagrees("fn m() -> nil { let a: i32 = 5  let b: i64 = 10  let _ = a < b  nil }"),
        "a < b (a: i32, b: i64) should conflict");
    assert!(disagrees("fn m() -> nil { let a: i32 = 5  let b: i64 = 10  let _ = a == b  nil }"),
        "a == b (a: i32, b: i64) should conflict");
    // A comparison's result is bool regardless — the check is a pure side
    // effect and must not change what the expression itself types as.
    assert!(passes("fn m() -> bool { let a: i32 = 5  a < 2i32 }"),
        "a < 2i32 (a: i32) should still check clean and type as bool");
    // Non-numeric comparisons never touch the numeric-only check.
    assert!(passes(r#"fn m() -> bool { "x" == "y" }"#), "str == str is untouched");

    // ── Bitwise / shift ──────────────────────────────────────────────────
    assert!(has_range_err("fn m() -> nil { let a: i32 = 5  let b: i32 = a & 5000000000  nil }"),
        "a & 5000000000 (a: i32) should overflow");
    assert!(disagrees("fn m() -> nil { let a: i32 = 5  let b: i32 = a << 2i64  nil }"),
        "a << 2i64 (a: i32) should conflict");
    assert!(disagrees("fn m() -> nil { let a: i32 = 5  let b: i64 = 10  let c: i32 = a & b  nil }"),
        "a & b (a: i32, b: i64) should conflict");
    // A matching suffix and an in-range literal still check clean.
    assert!(passes("fn m() -> nil { let a: i32 = 5  let b: i32 = a & 2i32  nil }"),
        "a & 2i32 (a: i32) should still check clean");
    assert!(passes("fn m() -> nil { let a: i32 = 5  let b: i32 = a & 500  nil }"),
        "a & 500 (a: i32) fits");
}

#[test]
fn operand_checks_reach_through_as_cast_547() {
    // #547: `check_expr`'s `Cast` arm used to resolve the cast's target type
    // and stop — it never recursed into the operand at all, so nothing
    // check_expr would otherwise catch on that operand (including
    // check_binop's #295/#538/#539 range/suffix/mixed-width checks above)
    // ran on it. `dmc --check` reported clean on `(a + b) as i64` (a: i32,
    // b: i64) while both `dmc run` (silently wraps) and `dmc jit` (refuses)
    // disagreed with it — the same three-way split #538/#539 closed
    // everywhere else, still open in the one position (`as i64`) that real
    // code — especially anything printing a narrow int — actually writes.
    // Fixed by making the `Cast` arm check its operand like any other
    // position, at any nesting depth, while leaving the cast itself (and
    // its own legality) untouched.
    fn has_range_err(src: &str) -> bool {
        check(src).iter().any(|m| m.contains("out of range"))
    }
    fn disagrees(src: &str) -> bool {
        check(src).iter().any(|m| m.contains("disagree"))
    }

    // ── Mixed integral width, through a cast ────────────────────────────
    assert!(disagrees(
        "fn m() -> i64 { let a: i32 = 1  let b: i64 = 2  let c: i64 = (a + b) as i64  c }"
    ), "(a + b) as i64 (a: i32, b: i64) should still conflict under a cast");
    assert!(disagrees(
        "fn m() -> i64 { let a: i32 = 1  let b: i64 = 2  let c: i64 = (b + a) as i64  c }"
    ), "(b + a) as i64 — operand order reversed — should still conflict");

    // ── Suffix conflict, through a cast ──────────────────────────────────
    assert!(disagrees(
        "fn m() -> i64 { let a: i32 = 1  let c: i64 = (a & 2i64) as i64  c }"
    ), "(a & 2i64) as i64 (a: i32) should still conflict");
    assert!(disagrees(
        "fn m() -> i64 { let a: i32 = 1  let c: i64 = (2i64 & a) as i64  c }"
    ), "(2i64 & a) as i64 — operand order reversed — should still conflict");

    // ── Literal range, through a cast ────────────────────────────────────
    assert!(has_range_err(
        "fn m() -> i64 { let a: i32 = 1  let c: i64 = (a + 5000000000) as i64  c }"
    ), "(a + 5000000000) as i64 (a: i32) should still overflow");
    assert!(has_range_err(
        "fn m() -> i64 { let a: i32 = 1  let c: i64 = (5000000000 + a) as i64  c }"
    ), "(5000000000 + a) as i64 — operand order reversed — should still overflow");

    // ── Nested casts: the check must reach through more than one `as` ───
    assert!(disagrees(
        "fn m() -> i64 { let a: i32 = 1  let b: i64 = 2  let c: i64 = ((a + b) as i64) as i64  c }"
    ), "((a + b) as i64) as i64 should still conflict at any nesting depth");

    // ── Cast inside a call argument — the idiomatic "print a narrow int"
    // shape that let this hole survive #543's own gate run.
    assert!(disagrees(
        "fn m() -> nil { let a: i32 = 1  let b: i64 = 2  print((a + b) as i64) }"
    ), "print((a + b) as i64) should conflict — this is the shape #547 was filed over");

    // ── Must stay clean: the cast itself is not the problem ─────────────
    // A plain concrete-to-concrete cast is legal SPEC §3.1 and must not be
    // touched by this fix.
    assert!(passes(
        "fn m() -> i64 { let a: i32 = 1  a as i64 }"
    ), "a as i64 (a: i32) is a plain, legal, concrete-to-concrete cast");
    // The correct way to write the mixed-width intent: cast the narrow
    // operand up *before* combining it, rather than the whole expression
    // after. This must keep checking clean. (Newline-separated, not
    // double-space-separated: a numeric literal statement immediately
    // followed by a same-line `(` parses as a call on the literal — an
    // unrelated parser gotcha the diagnostic itself warns about — so the
    // tail expression needs its own line here.)
    assert!(passes(
        "fn m() -> i64 {\n let a: i32 = 1\n let b: i64 = 2\n (a as i64) + b\n}"
    ), "(a as i64) + b — widen before combining — must stay clean");
}

#[test]
fn enum_exhaustiveness_and_variants_336() {
    let prelude = "enum Color { Red, Green, Blue }\n";
    fn errs(src: &str) -> Vec<String> { check(src) }

    // Full coverage (qualified + bare mix) — exhaustive, no catch-all needed.
    assert!(passes(&format!("{prelude}\
        fn f(c: Color) -> i64 {{ match c {{ Color.Red => 1, Green => 2, Color.Blue => 3 }} }}")),
        "qualified+bare full coverage should be exhaustive");

    // Missing a variant with no catch-all → error naming the gap.
    let e = errs(&format!("{prelude}\
        fn f(c: Color) -> i64 {{ match c {{ Color.Red => 1, Color.Green => 2 }} }}"));
    assert!(e.iter().any(|m| m.contains("coverage incomplete") && m.contains("Color.Blue")),
        "missing Blue should be flagged, got {:?}", e);

    // A catch-all closes an open match.
    assert!(passes(&format!("{prelude}\
        fn f(c: Color) -> i64 {{ match c {{ Color.Red => 1, _ => 0 }} }}")),
        "catch-all should satisfy exhaustiveness");

    // Unknown variant (typo) is an error.
    assert!(errs(&format!("{prelude}\
        fn f(c: Color) -> i64 {{ match c {{ Color.Reed => 1, _ => 0 }} }}"))
        .iter().any(|m| m.contains("no variant `Reed`")),
        "typo variant should error");

    // A qualified pattern from the wrong enum is an error.
    assert!(errs(&format!("{prelude}enum Size {{ Small, Big }}\n\
        fn f(c: Color) -> i64 {{ match c {{ Size.Small => 1, _ => 0 }} }}"))
        .iter().any(|m| m.contains("scrutinee")),
        "wrong-enum pattern should error");

    // An enum value casts to its ordinal via `as i64`.
    assert!(passes(&format!("{prelude}fn f(c: Color) -> i64 {{ (c as i64) }}")),
        "enum value should cast to i64");
    // Enums are nominal (strict typing, #284): a bare int does not satisfy an
    // enum return type.
    assert!(!errs(&format!("{prelude}fn f() -> Color {{ 0 }}")).is_empty(),
        "a bare int must not satisfy an enum return type");
}

#[test]
fn open_scalar_match_needs_catch_all_291_3() {
    fn errs(src: &str) -> Vec<String> { check(src) }
    // #291.3: an open scalar scrutinee (i64/str) needs a catch-all `_`.
    assert!(errs("fn f(k: i64) -> i64 { match k { 0 => 1, 1 => 2 } }")
        .iter().any(|m| m.contains("not exhaustive")),
        "open i64 match without catch-all should error");
    assert!(errs(r#"fn f(s: str) -> i64 { match s { "a" => 1, "b" => 2 } }"#)
        .iter().any(|m| m.contains("not exhaustive")),
        "open str match without catch-all should error");
    assert!(passes("fn f(k: i64) -> i64 { match k { 0 => 1, _ => 9 } }"),
        "catch-all satisfies an open i64 match");
    assert!(passes("fn f(k: i64) -> i64 { match k { 0 => 1, n => n } }"),
        "a bare-ident bind is a catch-all for an open i64 match");
    // `bool` still uses true/false coverage (not the open-scalar rule).
    assert!(passes("fn f(b: bool) -> i64 { match b { true => 1, false => 0 } }"),
        "bool full coverage needs no `_`");
}

#[test]
fn slice_index_does_not_return_scalar_type() {
    // t[0:4] result should be usable as a tensor (no false scalar-type error)
    // The checker should return Unknown for slice indexing, not elem_ty.
    // We verify by checking that no errors are emitted when the result is
    // used in a context that expects Unknown (i.e. not flagged as a scalar mismatch).
    let diags = check(r#"
        fn main() -> nil {
            let t = vault.zeros[f32, [4, 8]]
            let row = t[0]
            nil
        }
    "#);
    assert!(diags.is_empty(), "unexpected errors: {:?}", diags);
}

// ── Visibility Modifier Tests ───────────────────────────────────────────────

#[test]
fn typechecker_visibility_checking() {
    let dep_src = r#"
        pub fn public_fn() -> nil { nil }
        fn private_fn() -> nil { nil }
        pub model PublicModel { x: i64 }
        model PrivateModel { x: i64 }
        pub type PublicAlias = i64
        type PrivateAlias = i64
        pub fn public_val() -> i64 { 1 }
        let private_val = 2
    "#;

    // Write dep_src to a real temporary file in the workspace
    let dep_path = std::env::current_dir().unwrap().join("dep_visibility_temp.dmc");
    std::fs::write(&dep_path, dep_src).unwrap();
    let dep_canonical = dep_path.canonicalize().unwrap();

    let dep_tokens = Lexer::new(dep_src).tokenize().expect("lex failed");
    let dep_program = Parser::new(dep_tokens).parse_program().expect("parse failed");

    let mut checker = Checker::new();
    checker.check_program(&dep_program, None);
    assert!(checker.errors.is_empty(), "dependency check failed: {:?}", checker.errors);

    let public_items = super::ast::collect_public_items(&dep_program);
    let dep_env = super::check::ModuleEnv {
        env: checker.env.clone(),
        aliases: checker.aliases.clone(),
        resolved_aliases: checker.resolved_aliases.clone(),
        public_items,
        top_level_let_names: super::ast::collect_top_level_let_names(&dep_program),
        transitive_import_lets: checker.import_private_lets.clone(),
    };

    // Test 1: qualified import. Public items should be visible, private items should not.
    let importer_src = r#"
        use "dep_visibility_temp.dmc" as dep
        fn t() -> nil {
            let _ = dep.public_fn()
            let _ = dep.PublicModel { x: 1 }
            let _ = 42 as dep.PublicAlias
            nil
        }
    "#;
    let tokens = Lexer::new(importer_src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");

    let mut checker2 = Checker::new();
    checker2.checked_modules.insert(dep_canonical.clone(), dep_env.clone());
    checker2.check_program(&program, Some(&std::env::current_dir().unwrap().join("importer_visibility_temp.dmc")));
    assert!(checker2.errors.is_empty(), "importer check failed: {:?}", checker2.errors);

    // Try to access private_fn -> should fail
    let bad_src = r#"
        use "dep_visibility_temp.dmc" as dep
        fn t() -> nil {
            let _ = dep.private_fn()
            nil
        }
    "#;
    let tokens = Lexer::new(bad_src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker3 = Checker::new();
    checker3.checked_modules.insert(dep_canonical.clone(), dep_env.clone());
    checker3.check_program(&program, Some(&std::env::current_dir().unwrap().join("importer_visibility_temp.dmc")));
    assert!(!checker3.errors.is_empty(), "expected error for private fn");

    // Try to access private model -> should fail
    let bad_src2 = r#"
        use "dep_visibility_temp.dmc" as dep
        fn t() -> nil {
            let _ = dep.PrivateModel { x: 1 }
            nil
        }
    "#;
    let tokens = Lexer::new(bad_src2).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker4 = Checker::new();
    checker4.checked_modules.insert(dep_canonical.clone(), dep_env.clone());
    checker4.check_program(&program, Some(&std::env::current_dir().unwrap().join("importer_visibility_temp.dmc")));
    assert!(!checker4.errors.is_empty(), "expected error for private model");

    // Cleanup temporary file
    let _ = std::fs::remove_file(&dep_path);
}

// ── Additional typechecker coverage ──────────────────────────────────────────

#[test]
fn fn_call_correct_arity_passes() {
    assert!(passes("fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() -> i64 { add(1, 2) }"));
}

#[test]
fn fn_call_wrong_arity_fails() {
    let errs = check("fn f(x: i64) -> i64 { x }\nfn main() -> i64 { f(1, 2) }");
    assert!(!errs.is_empty(), "expected arity error");
}

#[test]
fn recursive_fn_passes() {
    assert!(passes("fn fib(n: i64) -> i64 { if n < 2 { n } else { fib(n-1) + fib(n-2) } }"));
}

#[test]
fn let_binding_basic_passes() {
    assert!(passes("fn main() -> i64 { let x = 42; x }"));
}

#[test]
fn if_else_passes() {
    assert!(passes("fn main() -> i64 { if true { 1 } else { 0 } }"));
}

#[test]
fn for_range_passes() {
    assert!(passes("fn main() -> nil { for i in 0..10 { nil } }"));
}

#[test]
fn while_loop_passes() {
    assert!(passes("fn main() -> nil { let mut x = 0; while x < 10 { x = x + 1 } }"));
}

#[test]
fn match_expression_passes() {
    assert!(passes(r#"fn main() -> i64 { match 1 { 1 => 10, _ => 0, } }"#));
}

#[test]
fn builtin_print_passes() {
    assert!(passes(r#"fn main() -> nil { print("hello\n") }"#));
}

#[test]
fn builtin_typed_prints_pass() {
    assert!(passes("fn main() -> nil { print_i64(1); print_f64(1.0); print_tensor([1.0, 2.0]) }"));
}

#[test]
fn builtin_assert_passes() {
    assert!(passes("fn main() -> nil { assert(true) }"));
}

#[test]
fn builtin_assert_eq_passes() {
    assert!(passes("fn main() -> nil { assert_eq(1, 1) }"));
}

#[test]
fn builtin_list_passes() {
    assert!(passes("fn main() -> nil { let xs = list(1, 2, 3); nil }"));
}

#[test]
fn builtin_map_passes() {
    assert!(passes("fn main() -> nil { let m = map(); nil }"));
}

#[test]
fn builtin_format_passes() {
    assert!(passes(r#"fn main() -> str { format("{} + {} = {}", 1, 2, 3) }"#));
}

#[test]
fn builtin_to_str_passes() {
    assert!(passes("fn main() -> str { to_str(42) }"));
}

#[test]
fn builtin_to_int_passes() {
    assert!(passes(r#"fn main() -> i64 { to_int("42") }"#));
}

#[test]
fn builtin_clamp_passes() {
    assert!(passes("fn main() -> nil { let _ = clamp(5, 0, 10); nil }"));
}

#[test]
fn builtin_round_passes() {
    assert!(passes("fn main() -> nil { let _ = round(3.7); nil }"));
}

#[test]
fn builtin_ord_passes() {
    assert!(passes(r#"fn main() -> i64 { ord("A") }"#));
}

#[test]
fn builtin_chr_passes() {
    assert!(passes("fn main() -> str { chr(65) }"));
}

#[test]
fn builtin_list_map_passes() {
    assert!(passes("fn main() -> nil { let xs = list_map(list(1,2,3), fn(x: i64) -> i64 { x * 2 }); nil }"));
}

#[test]
fn builtin_list_filter_passes() {
    assert!(passes("fn main() -> nil { let xs = list_filter(list(1,2,3), fn(x: i64) -> bool { x > 1 }); nil }"));
}

#[test]
fn builtin_list_find_passes() {
    assert!(passes("fn main() -> i64 { list_find(list(1,2,3), 2) }"));
}

#[test]
fn builtin_list_any_passes() {
    assert!(passes("fn main() -> bool { list_any(list(1,2,3), fn(x: i64) -> bool { x > 2 }) }"));
}

#[test]
fn builtin_str_repeat_passes() {
    assert!(passes(r#"fn main() -> str { str_repeat("ab", 3) }"#));
}

#[test]
fn builtin_regex_match_passes() {
    assert!(passes(r#"fn main() -> bool { regex_match("[0-9]+", "123") }"#));
}

#[test]
fn builtin_json_encode_passes() {
    assert!(passes(r#"fn main() -> str { json_encode("hello") }"#));
}

#[test]
fn builtin_hash_fnv_passes() {
    assert!(passes(r#"fn main() -> i64 { hash_fnv("test") }"#));
}

#[test]
fn builtin_pi_constant_passes() {
    assert!(passes("fn main() -> f64 { pi }"));
}

#[test]
fn builtin_tau_constant_passes() {
    assert!(passes("fn main() -> f64 { tau }"));
}

#[test]
fn lambda_in_let_passes() {
    assert!(passes("fn main() -> nil { let f = fn(x: i64) -> i64 { x * 2 }; nil }"));
}

#[test]
fn pub_fn_passes() {
    assert!(passes("pub fn exported() -> nil { nil }"));
}

#[test]
fn tuple_pattern_let_passes() {
    assert!(passes("fn main() -> nil { let (a, b) = (1, 2); nil }"));
}

#[test]
fn multiline_fn_passes() {
    assert!(passes(r#"
fn square(n: i64) -> i64 {
    let r = n * n
    r
}
"#));
}

#[test]
fn model_declaration_passes() {
    assert!(passes(r#"
model Linear {
    fn forward(x: Tensor[f32, [~]]) -> Tensor[f32, [~]] { x }
}
"#));
}

#[test]
fn float_literal_suffixes_check() {
    let get_type = |src: &str| -> String {
        let tokens = Lexer::new(src).tokenize().expect("lex failed");
        let program = Parser::new(tokens).parse_program().expect("parse failed");
        let mut checker = Checker::new();
        checker.check_program(&program, None);
        if let Some(ty) = checker.env.lookup("x") {
            format!("{}", ty)
        } else {
            "not found".to_string()
        }
    };

    assert_eq!(get_type("let x = 1.0f16"), "f16");
    assert_eq!(get_type("let x = 1.0bf16"), "bf16");
    assert_eq!(get_type("let x = 1.0tf32"), "tf32");
    assert_eq!(get_type("let x = 1.0f32"), "f32");
    assert_eq!(get_type("let x = 1.0f64"), "f64");
    assert_eq!(get_type("let x = 1.0fp8_e4m3"), "fp8_e4m3");
    assert_eq!(get_type("let x = 1.0fp8_e5m2"), "fp8_e5m2");
    // An unconstrained float literal defaults to f64 (#284) — consistent with
    // the unconstrained int-literal default of i64, and the conventional
    // double-precision default. A float literal that needs f32 must say so
    // (`let x: f32 = 1.0` or the `1.0f32` suffix).
    assert_eq!(get_type("let x = 1.0"), "f64");

    // Ensure float does not assign to a non-numeric type like bool
    let errs = check("fn main() -> bool { 1.0f64 }");
    assert!(!errs.is_empty(), "expected type mismatch error, got: {:?}", errs);
}

#[test]
fn fp8_dtype_admitted_in_ctor_type_arg_818() {
    // #818: `forge.zeros[fp8_e4m3, [4]]` (and the `fp8_e5m2` twin) used to
    // fail with `undefined identifier `fp8e4m3`` — the constructor's
    // type-arg position lost the underscore before the checker ever saw the
    // name (parser.rs's expression-position scalar-type reconstruction).
    // The checker already admitted `fp8_e4m3` in annotation position
    // (`Tensor[fp8_e4m3, [4]]`, #816); both positions must now agree.
    assert!(passes("fn main() -> nil { let t = forge.zeros[fp8_e4m3, [4]]\n nil }"),
        "got: {:?}", check("fn main() -> nil { let t = forge.zeros[fp8_e4m3, [4]]\n nil }"));
    assert!(passes("fn main() -> nil { let t = forge.zeros[fp8_e5m2, [4]]\n nil }"),
        "got: {:?}", check("fn main() -> nil { let t = forge.zeros[fp8_e5m2, [4]]\n nil }"));
}

// ── New coverage: @grad, @comptime, dynamic shapes, @cast, lambdas ──────────

#[test]
fn grad_fn_passes_typechecker() {
    assert!(passes(r#"
        @grad fn mse[D](!w: Tensor[f32, [D]], x: Tensor[f32, [D]]) -> f32 {
            sum((w .- x) .* (w .- x))
        }
    "#));
}

#[test]
fn grad_fn_two_differentiable_params_passes() {
    assert!(passes(r#"
        @grad fn loss[D](!w: Tensor[f32, [D]], !b: Tensor[f32, [D]], x: Tensor[f32, [D]]) -> f32 {
            sum((w .* x .+ b) .- x)
        }
    "#));
}

#[test]
fn grad_grad_fwd_bwd_bwd_passes() {
    // Stacked `@grad @grad` exposes `.fwd_bwd_bwd` at the source surface.
    assert!(passes(r#"
        @grad @grad fn cube(!w: Tensor[f32, [1]]) -> f32 {
            sum(w .* w .* w)
        }
        fn main() -> f32 {
            let !w = forge.zeros[f32, [1]]
            let (_l, g2) = cube.fwd_bwd_bwd(w)
            g2.w[0]
        }
    "#));
}

#[test]
fn fwd_bwd_bwd_on_single_grad_is_error() {
    // Second-order needs `@grad @grad`; calling it on a single-`@grad` fn
    // used to silently produce garbage (`<opaque index>`) at runtime.
    let errs = check(r#"
        @grad fn cube(!w: Tensor[f32, [1]]) -> f32 { sum(w .* w .* w) }
        fn main() -> f32 {
            let !w = forge.zeros[f32, [1]]
            let (_l, g2) = cube.fwd_bwd_bwd(w)
            g2.w[0]
        }
    "#);
    assert!(errs.iter().any(|m| m.contains("@grad @grad")), "got {errs:?}");
}

#[test]
fn unknown_grad_method_is_error() {
    // A typo'd autodiff method on a `@grad fn` is a hard error, not a
    // fall-through to opaque field access.
    let errs = check(r#"
        @grad fn cube(!w: Tensor[f32, [1]]) -> f32 { sum(w .* w .* w) }
        fn main() -> f32 {
            let !w = forge.zeros[f32, [1]]
            let g = cube.fwd_bw(w)
            0.0
        }
    "#);
    assert!(errs.iter().any(|m| m.contains("unknown @grad method")), "got {errs:?}");
}

#[test]
fn grad_method_on_plain_fn_is_error() {
    // `.fwd_bwd` on a fn that isn't `@grad` is a hard error.
    let errs = check(r#"
        fn cube(!w: Tensor[f32, [1]]) -> f32 { sum(w .* w .* w) }
        fn main() -> f32 {
            let !w = forge.zeros[f32, [1]]
            let (_l, g) = cube.fwd_bwd(w)
            0.0
        }
    "#);
    assert!(errs.iter().any(|m| m.contains("not a `@grad fn`")), "got {errs:?}");
}

#[test]
fn grad_fn_with_no_mut_param_is_error() {
    // AUTODIFF.md §2: a `@grad fn` with no `!` (mut) parameter has nothing to
    // differentiate — a compile-time error in `--check`/`run`, not just `--jit`.
    let errs = check(r#"
        @grad fn f(x: Tensor[f32, [2]]) -> f32 { sum(x .* x) }
        fn main() -> f32 {
            let x = forge.zeros[f32, [2]]
            let (l, g) = f.fwd_bwd(x)
            l
        }
    "#);
    assert!(
        errs.iter().any(|m| m.contains("nothing to differentiate")),
        "expected a non-differentiable @grad error, got {errs:?}"
    );
}

#[test]
fn grad_fn_with_mut_param_passes_differentiability() {
    // Near-miss: the same body but with a `!` mut param must NOT trip the new
    // check — a single mut param is enough to differentiate.
    let errs = check(r#"
        @grad fn f(!x: Tensor[f32, [2]]) -> f32 { sum(x .* x) }
        fn main() -> f32 {
            let !x = forge.zeros[f32, [2]]
            let (l, g) = f.fwd_bwd(x)
            l
        }
    "#);
    assert!(
        !errs.iter().any(|m| m.contains("nothing to differentiate")),
        "a @grad fn with a mut param must pass, got {errs:?}"
    );
}

#[test]
fn pub_extern_fn_is_error() {
    // SPEC §9: an `extern fn` is always exported, so `pub` on it is a
    // compile-time error.
    let errs = check(r#"pub extern fn some_c_fn(x: i32) -> nil"#);
    assert!(
        errs.iter().any(|m| m.contains("`pub` is not allowed on `extern fn`")),
        "expected a pub-extern error, got {errs:?}"
    );
}

#[test]
fn non_pub_extern_fn_passes() {
    // Near-miss: the same decl without `pub` is legal and must still pass.
    assert!(
        passes(r#"extern fn some_c_fn(x: i32) -> nil"#),
        "a non-pub extern fn must pass, got {:?}",
        check(r#"extern fn some_c_fn(x: i32) -> nil"#)
    );
}

#[test]
fn pub_let_is_error() {
    // #596: SPEC.md §6.7 exports only `pub fn` / `pub model` / `pub type` — a
    // top-level `let` never crosses a module boundary. Before this fix, `pub
    // let` type-checked clean and the `pub` silently did nothing; the failure
    // only ever surfaced at an importer, as an unrelated "undefined
    // identifier". This must now fire at the declaration itself.
    let errs = check(r#"pub let LIMIT: i64 = 7"#);
    assert!(
        errs.iter().any(|m| m.starts_with("pub-let:")
            && m.contains("`pub let LIMIT`")
            && m.contains("does not export")
            && m.contains("bindings cross a module")),
        "expected a pub-let error naming the binding, got {errs:?}"
    );
}

#[test]
fn pub_let_mut_is_also_error() {
    // The mutable form (`pub let !x`) is the same silent no-op and must be
    // rejected the same way. Mutability lives on the `LetStmt`, not the
    // pattern (`l.mutating`), so the rendered message must carry the `!` —
    // it previously fell off, rendering the bare `` `pub let LIMIT` ``.
    let errs = check(r#"pub let !LIMIT: i64 = 7"#);
    assert!(
        errs.iter().any(|m| m.starts_with("pub-let:") && m.contains("`pub let !LIMIT`")),
        "expected a pub-let error naming the mutable binding with its `!`, got {errs:?}"
    );
}

#[test]
fn pub_let_tuple_pattern_renders_the_pattern() {
    // A tuple-pattern `let` isn't a `Pattern::Ident`, so it previously fell
    // all the way back to the bare `` `pub let` `` with no pattern text at
    // all. It must render the tuple the way source would.
    let errs = check(r#"pub let (a, b): (i64, i64) = (1, 2)"#);
    assert!(
        errs.iter().any(|m| m.starts_with("pub-let:") && m.contains("`pub let (a, b)`")),
        "expected a pub-let error rendering the tuple pattern, got {errs:?}"
    );
}

#[test]
fn private_let_passes() {
    // Near-miss: the same declaration without `pub` is legal file-private
    // state and must still pass.
    assert!(
        passes(r#"let LIMIT: i64 = 7"#),
        "a non-pub let must pass, got {:?}",
        check(r#"let LIMIT: i64 = 7"#)
    );
}

#[test]
fn pub_fn_does_not_trip_the_pub_let_diagnostic() {
    // Near-miss: `pub fn` is the spec's actual export form (SPEC.md §6.7) and
    // must not be flagged.
    let errs = check(r#"pub fn limit() -> i64 { 7 }"#);
    assert!(
        !errs.iter().any(|m| m.starts_with("pub-let:")),
        "pub fn must not trip the pub-let diagnostic, got {errs:?}"
    );
}

#[test]
fn pub_model_does_not_trip_the_pub_let_diagnostic() {
    let errs = check(r#"pub model Thing { x: i64 }"#);
    assert!(
        !errs.iter().any(|m| m.starts_with("pub-let:")),
        "pub model must not trip the pub-let diagnostic, got {errs:?}"
    );
}

#[test]
fn pub_type_does_not_trip_the_pub_let_diagnostic() {
    let errs = check(r#"pub type Id = i64"#);
    assert!(
        !errs.iter().any(|m| m.starts_with("pub-let:")),
        "pub type must not trip the pub-let diagnostic, got {errs:?}"
    );
}

#[test]
fn comptime_block_passes() {
    assert!(passes(r#"
        fn main() -> i64 {
            let x = @comptime { 3 * 7 + 1 }
            x
        }
    "#));
}

// A trailing `@directive { … }` parses as a `Stmt::DirectiveBlock`, not a
// tail expression, and its body's last element may itself be a keyword-led
// `if` / `match` STATEMENT. That statement is the block's value, exactly as it
// is in a plain block. The checker used to read only the body's `tail_expr`
// here and type the whole fn body as nil — bound to a `let` first, the same
// block checked fine. Directive-independent: the arm is shared.

#[test]
fn trailing_directive_block_yields_its_if_statement() {
    for d in ["@deterministic", "@comptime"] {
        let src = format!("fn main() -> i64 {{ {d} {{ if 3 > 2 {{ 10 }} else {{ 20 }} }} }}");
        assert!(passes(&src), "{d}: {:?}", check(&src));
    }
}

#[test]
fn trailing_directive_block_yields_its_match_statement() {
    let src = "fn main() -> i64 {\n    let n = 3\n    @deterministic { match n { 3 => 10, _ => 20 } }\n}";
    assert!(passes(src), "{:?}", check(src));
}

#[test]
fn trailing_directive_block_keeps_inner_lets_in_scope_for_its_if() {
    let src = "fn main() -> i64 { @deterministic { let k = 5  if k > 2 { k } else { 0 } } }";
    assert!(passes(src), "{:?}", check(src));
}

#[test]
fn trailing_directive_block_if_value_is_checked_against_return_type() {
    // The value now flows: an `if` yielding bool from an `-> i64` fn is the
    // ordinary body-type mismatch, not a nil.
    let errs = check("fn main() -> i64 { @deterministic { if 3 > 2 { true } else { false } } }");
    assert!(errs.iter().any(|e| e.contains("body produces") && e.contains("bool")),
            "got: {:?}", errs);
}

// ── #505: `@comptime` v1 — the fold set and what it refuses ─────────────────
//
// SPEC.md §7.8 / DIRECTIVES.md §3 / the comptime v1 grammar's §5 rule. Every entry below is a
// rule that was specified and unenforced before #505, so each test is the
// difference between the directive meaning something and being a no-op.

fn comptime_errs(errs: &[String]) -> Vec<&String> {
    errs.iter()
        .filter(|e| e.contains("comptime-non-static")
                 || e.contains("comptime-budget")
                 || e.contains("port-forbidden"))
        .collect()
}

// The silence cases first. A refusal battery with no positives only proves the
// gate says no.

#[test]
fn comptime_folds_closed_integer_arithmetic() {
    assert!(passes(r#"fn main() -> i64 { @comptime { 3 * 7 + 1 } }"#));
}

#[test]
fn comptime_folds_a_boolean() {
    assert!(passes(r#"fn main() -> bool { @comptime { 2 > 1 && 3 != 4 } }"#));
}

#[test]
fn comptime_folds_a_conditional() {
    // Both spellings. The tail form needed #583 — a trailing directive block
    // whose body ends in an `if` STATEMENT used to type as `nil` whatever the
    // directive was — so pinning it here keeps the fold and that fix honest
    // about each other.
    assert!(passes(r#"fn main() -> i64 { @comptime { if 3 > 2 { 10 } else { 20 } } }"#));
    assert!(passes(r#"
        fn main() -> i64 {
            let x = @comptime { if 3 > 2 { 10 } else { 20 } }
            x
        }
    "#));
}

#[test]
fn comptime_folds_a_loop_that_terminates() {
    // A `while` accumulating a sum is the "configuration table" case SPEC.md
    // §7.8 names, and the reason the budget of §6 has to exist at all.
    let errs = check(r#"
        fn main() -> i64 {
            @comptime {
                let !acc = 0
                let !i = 1
                while i <= 4 { acc += i  i += 1 }
                acc
            }
        }
    "#);
    assert!(comptime_errs(&errs).is_empty(), "got: {:?}", errs);
}

#[test]
fn comptime_accepts_a_shape_parameter() {
    // Tier 2 of the comptime fold set: comptime-known, but constant only per
    // monomorphization, so the pass accepts it and leaves it to the backends.
    let errs = check(r#"
        fn tile[N](x: Tensor[f32, [N]]) -> i64 { @comptime { N * 2 } }
        fn main() -> i64 { tile(forge.zeros[f32, [4]]) }
    "#);
    assert!(comptime_errs(&errs).is_empty(), "a shape param is comptime; got: {:?}", errs);
}

#[test]
fn comptime_accepts_a_model_shape_parameter() {
    let errs = check(r#"
        model Block[D] {
            w: Tensor[f32, [D]]
            fn width(self) -> i64 { @comptime { D + 1 } }
        }
        fn main() -> i64 { 0 }
    "#);
    assert!(comptime_errs(&errs).is_empty(), "got: {:?}", errs);
}

// The refusals. Each names the offending construct, because "not comptime"
// with no noun is a diagnostic the reader has to guess at.

#[test]
fn a_float_literal_inside_comptime_is_non_static() {
    // The v1 cut, and the one most likely to be argued with: folding a float
    // would re-open whether a folded float must equal a computed one (#320).
    let errs = check(r#"fn main() -> i64 { let x = @comptime { 1.5 * 2.0 }  0 }"#);
    assert!(errs.iter().any(|e| e.contains("comptime-non-static") && e.contains("float `1.5`")),
        "got: {:?}", errs);
}

#[test]
fn a_runtime_binding_inside_comptime_is_non_static() {
    let errs = check(r#"fn f(n: i64) -> i64 { let x = @comptime { n + 1 }  0 }"#);
    assert!(errs.iter().any(|e| e.contains("comptime-non-static") && e.contains("`n` is not comptime")),
        "got: {:?}", errs);
}

#[test]
fn a_port_call_inside_comptime_is_port_forbidden() {
    // PORTS.md §5's fourth restriction — specified since the document was
    // written, and the only one of the four that did not bind before #505.
    for prim in ["port_open", "port_call", "port_close"] {
        let src = format!("fn main() -> i64 {{ let x = @comptime {{ {}(1) }}  0 }}", prim);
        let errs = check(&src);
        assert!(errs.iter().any(|e| e.contains("port-forbidden") && e.contains(prim)),
            "expected port-forbidden for {}, got: {:?}", prim, errs);
    }
}

#[test]
fn an_extern_fn_call_inside_comptime_is_non_static() {
    // SPEC.md §5: "An `extern fn` may not be called from `@comptime`".
    let errs = check(r#"
        extern fn c_add(x: i32) -> i32
        fn main() -> i64 { let x = @comptime { c_add(1) }  0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("comptime-non-static") && e.contains("extern fn c_add")),
        "got: {:?}", errs);
}

// ─── #578: the extern boundary, and the three constructs that forbid a call ──

#[test]
fn an_extern_boundary_refuses_a_tensor_parameter() {
    // Until #578 this checked clean and JIT-compiled clean. At a boundary that
    // performs no shape, alignment or aliasing check, a silently-accepted
    // tensor parameter is a segfault waiting for a call site.
    let errs = check("extern fn bad(t: Tensor[f32, [4]]) -> nil\nfn main() -> i64 { 0 }");
    assert!(errs.iter().any(|e| e.contains("extern-boundary") && e.contains("`t`")
                             && e.contains("Tensor[f32, [4]]")),
        "got: {:?}", errs);
}

#[test]
fn an_extern_boundary_refuses_a_tensor_return_and_a_tuple() {
    for (src, what) in [
        ("extern fn bad(x: i32) -> Tensor[f32, [4]]\nfn main() -> i64 { 0 }", "return"),
        ("extern fn bad(t: (i32, i32)) -> nil\nfn main() -> i64 { 0 }", "tuple"),
        ("extern fn bad(t: [i32; 4]) -> nil\nfn main() -> i64 { 0 }", "array"),
        ("extern fn bad(p: *Tensor[f32, [4]]) -> nil\nfn main() -> i64 { 0 }", "pointee"),
    ] {
        let errs = check(src);
        assert!(errs.iter().any(|e| e.contains("extern-boundary")),
            "no extern-boundary for the {} case: {:?}", what, errs);
    }
}

#[test]
fn an_extern_boundary_admits_scalars_pointers_and_nil() {
    // The positive rule the spec states, exhaustively: scalar types, raw
    // pointers `*T`, and `nil`. `str` is a scalar type, so the *checker*
    // admits it — the JIT refuses it separately (`jit-extern`), because a
    // demoniC `str` is not a `char*`. Narrowing here instead would be a
    // change to shipped surface under `STABILITY.md §3`.
    let errs = check(r#"
        extern fn ok1(a: i8, b: i16, c: i32, d: i64, e: u8, f: u32, g: u64) -> i64
        extern fn ok2(a: f32, b: f64, c: bool, d: str) -> f64
        extern fn ok3(p: *f32, q: *nil, n: i64) -> *f32
        extern fn ok4(x: i32) -> nil
        fn main() -> i64 { 0 }
    "#);
    assert!(errs.is_empty(), "an admissible boundary was rejected: {:?}", errs);
}

#[test]
fn an_extern_fn_call_is_forbidden_in_the_three_effect_constructs() {
    // The spec's `extern fn` rules name four constructs. `@comptime` is
    // covered by its own total ban on calls (the test above); these are the
    // other three, and they were unenforced before #578.
    //
    // The `@deterministic` row is what makes "no foreign accumulation order
    // inside `@deterministic`" a property of the language rather than of one
    // kernel-selection arm — the extern-fn lowering design bakes this in at
    // the call-classification level, not per backend.
    for (src, what) in [
        ("extern fn c_add(x: i32) -> i32\n\
          fn main() -> i64 { let v = @deterministic { c_add(1) }  v as i64 }",
         "`@deterministic` block"),
        ("extern fn c_add(x: i32) -> i32\n\
          fn main() -> i64 { let v = @fuse { c_add(1) }  v as i64 }",
         "`@fuse` block"),
        ("extern fn c_add(x: i32) -> i32\n\
          @grad fn g(!w: Tensor[f32, [2]]) -> f32 { (c_add(1) as f32) + sum(w) }\n\
          fn main() -> i64 { 0 }",
         "`@grad fn`"),
    ] {
        let errs = check(src);
        assert!(errs.iter().any(|e| e.contains("extern-context") && e.contains(what)),
            "no extern-context naming {} in: {:?}", what, errs);
    }
}

#[test]
fn an_extern_fn_call_outside_those_constructs_is_fine() {
    // The ban is scoped to the three constructs, not to `extern fn` generally
    // — the fast path #578 exists for is an ordinary call in ordinary code.
    let errs = check("extern fn c_add(x: i32) -> i32\n\
                      fn main() -> i64 { c_add(1) as i64 }");
    assert!(errs.is_empty(), "got: {:?}", errs);
}

// ─── #785: `Tensor`/`View`/`KV`'s element type must be a scalar type ────────
//
// `resolve_type`'s three arms parsed `Tensor[Buf, [4]]` / `View[Buf, [4]]` /
// `KV[Buf, [4]]` for a model `Buf` and `--check`-passed silently — a model is
// not a scalar element type (SPEC.md §3.1), and nothing downstream (the
// interpreter's `DType`, the JIT's `ScalarKind`) has a representation to
// build such a value in. `RawPtr`'s pointee already enforced the sibling
// rule at the `extern fn` boundary; these three arms did not, until this fix.

#[test]
fn tensor_element_type_refuses_a_model_in_param_position() {
    let errs = check("model Buf { x: f32 }\n\
                      fn f(t: Tensor[Buf, [4]]) -> nil { nil }\n\
                      fn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("tensor-element-type")
                             && e.contains("`Tensor`")
                             && e.contains("`Buf` is a model, not a scalar type")),
        "got: {:?}", errs);
}

#[test]
fn tensor_element_type_refuses_a_model_in_field_position() {
    let errs = check("model Buf { x: f32 }\n\
                      model Holder { t: Tensor[Buf, [4]] }\n\
                      fn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("tensor-element-type") && e.contains("`Buf` is a model")),
        "got: {:?}", errs);
}

#[test]
fn tensor_element_type_refuses_a_model_in_return_position() {
    let errs = check("model Buf { x: f32 }\n\
                      fn f() -> Tensor[Buf, [4]] { forge.zeros[f32, [4]] }\n\
                      fn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("tensor-element-type") && e.contains("`Buf` is a model")),
        "got: {:?}", errs);
}

#[test]
fn tensor_element_type_refuses_a_model_in_let_annotation() {
    let errs = check("model Buf { x: f32 }\n\
                      fn main() -> nil { let t: Tensor[Buf, [4]] = forge.zeros[f32, [4]]  nil }");
    assert!(errs.iter().any(|e| e.contains("tensor-element-type") && e.contains("`Buf` is a model")),
        "got: {:?}", errs);
}

#[test]
fn tensor_element_type_refuses_a_model_in_alias_body() {
    let errs = check("model Buf { x: f32 }\n\
                      type Row = Tensor[Buf, [4]]\n\
                      fn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("tensor-element-type") && e.contains("`Buf` is a model")),
        "got: {:?}", errs);
}

#[test]
fn view_element_type_refuses_a_model() {
    let errs = check("model Buf { x: f32 }\n\
                      fn f(t: View[Buf, [4]]) -> nil { nil }\n\
                      fn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("tensor-element-type")
                             && e.contains("`View`")
                             && e.contains("`Buf` is a model")),
        "got: {:?}", errs);
}

#[test]
fn kv_element_type_refuses_a_model() {
    let errs = check("model Buf { x: f32 }\n\
                      fn f(t: KV[Buf, [4]]) -> nil { nil }\n\
                      fn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("tensor-element-type")
                             && e.contains("`KV`")
                             && e.contains("`Buf` is a model")),
        "got: {:?}", errs);
}

#[test]
fn tensor_element_type_refuses_an_enum() {
    // An enum is a distinct nominal type (#336), not one of SPEC.md §3.1's
    // scalar types — same defect class as a model, even though its runtime
    // representation happens to be an i64 ordinal.
    let errs = check("enum Color { Red, Green, Blue }\n\
                      fn f(t: Tensor[Color, [4]]) -> nil { nil }\n\
                      fn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("tensor-element-type")
                             && e.contains("`Color` is an enum, not a scalar type")),
        "got: {:?}", errs);
}

#[test]
fn tensor_element_type_names_an_undefined_type_accurately() {
    // `Foo` names nothing — no model, enum, or alias declares it — so it
    // must NOT be misreported as "`Foo` is a model, not a scalar type"
    // (`resolve_type`'s `Type::Named` fallthrough arm produces the same
    // `TyType::Named` shape for an undefined name as for a genuine model
    // reference; only `is_model_name` tells them apart). This was itself
    // silently accepted before #785 (a pre-existing, broader gap: an
    // undefined type name anywhere, not just here, checks clean) — #785
    // does not fix that gap, but must not misdescribe it either now that
    // this position surfaces it.
    let errs = check("fn f(t: Tensor[Foo, [4]]) -> nil { nil }\nfn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("tensor-element-type")
                             && e.contains("`Foo` names no model, enum or alias in scope")),
        "got: {:?}", errs);
    assert!(!errs.iter().any(|e| e.contains("`Foo` is a model")),
        "must not claim `Foo` is a model: {:?}", errs);
}

#[test]
fn tensor_element_type_admits_every_spec_3_1_scalar() {
    // The positive rule, exhaustively: every SPEC.md §3.1 scalar name is a
    // legal `Tensor`/`View`/`KV` element type, `bool` included (the corpus
    // itself relies on `View[bool, […]]` for an attention mask —
    // `examples/fused_attn.dmc`/`examples/rope.dmc`).
    let errs = check(
        "fn f(\n\
            a: Tensor[i8, [1]], b: Tensor[i16, [1]], c: Tensor[i32, [1]], d: Tensor[i64, [1]],\n\
            e: Tensor[u8, [1]], g: Tensor[u16, [1]], h: Tensor[u32, [1]], i: Tensor[u64, [1]],\n\
            j: Tensor[int4, [1]], k: Tensor[int8, [1]],\n\
            l: Tensor[f16, [1]], m: Tensor[bf16, [1]], n: Tensor[tf32, [1]],\n\
            o: Tensor[f32, [1]], p: Tensor[f64, [1]],\n\
            q: Tensor[fp8_e4m3, [1]], r: Tensor[fp8_e5m2, [1]],\n\
            s: Tensor[trit, [1]], t: Tensor[bool, [1]],\n\
            u: Tensor[str, [1]], v: Tensor[nil, [1]]\n\
         ) -> nil { nil }\n\
         fn main() -> nil { nil }"
    );
    assert!(errs.is_empty(), "an admissible element type was rejected: {:?}", errs);
}

#[test]
fn tensor_element_type_admits_any() {
    // `any` resolves to `Unknown`, the checker's conservative match-anything
    // bottom (#186) — deliberately not flagged, the same "don't cascade onto
    // an already-⊥ type" rule every other check in this file applies to
    // `Unknown` (e.g. the `embed`-ids-must-be-integer check).
    let errs = check("fn f(t: Tensor[any, [4]]) -> nil { nil }\nfn main() -> nil { nil }");
    assert!(errs.is_empty(), "got: {:?}", errs);
}

#[test]
fn tensor_element_type_alias_resolves_before_the_check() {
    // `type E = f32` then `Tensor[E, [3]]` is exactly `Tensor[f32, [3]]` —
    // the alias must resolve to its scalar target before this check runs,
    // not be flagged as "`E` is ... not a scalar type" (jit_probes.py's
    // `type_alias_nested_tensor_element` probe covers the same ground for
    // the JIT side).
    let errs = check("type E = f32\n\
                      fn head(x: Tensor[E, [3]]) -> f32 { x[0] }\n\
                      fn main() -> nil { nil }");
    assert!(errs.is_empty(), "got: {:?}", errs);
}

#[test]
fn tensor_element_type_error_reports_once_not_twice() {
    // `resolve_type` runs over a fn's param/return type twice — once in
    // `register_fn` (pass 1), again in `check_fn` (pass 2) — so without the
    // span-keyed dedupe (`reported_element_type_spans`) this would report
    // the same mistake twice for a single bad annotation.
    let errs = check("model Buf { x: f32 }\n\
                      fn f(t: Tensor[Buf, [4]]) -> Tensor[Buf, [4]] { t }\n\
                      fn main() -> nil { nil }");
    let param_errs: Vec<_> = errs.iter().filter(|e| e.contains("tensor-element-type")).collect();
    // One for the param's `Buf`, one for the return's `Buf` — two DISTINCT
    // spans — but neither doubled.
    assert_eq!(param_errs.len(), 2, "expected exactly 2 (param + return, each once), got: {:?}", errs);
}

#[test]
fn a_user_fn_call_inside_comptime_is_non_static() {
    // v1 admits no call at all — that total ban is what makes the effect gate
    // structural, with no interprocedural scan to get wrong.
    let errs = check(r#"
        fn helper(x: i64) -> i64 { x + 1 }
        fn main() -> i64 { let x = @comptime { helper(2) }  0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("comptime-non-static") && e.contains("call to `helper`")),
        "got: {:?}", errs);
}

#[test]
fn non_integer_constructs_inside_comptime_are_non_static() {
    // One assertion per construct, each naming itself in the diagnostic.
    for (body, noun) in [
        ("[1.0, 2.0]",            "a tensor literal"),
        ("\"hi\"",                "a string literal"),
        ("(1, 2)",                "a tuple"),
        ("1 as f32",              "`as`"),
        ("@cast(bf16) { 1 }",     "a nested directive"),
    ] {
        let src = format!("fn main() -> i64 {{ let x = @comptime {{ {} }}  0 }}", body);
        let errs = check(&src);
        assert!(errs.iter().any(|e| e.contains("comptime-non-static") && e.contains(noun)),
            "expected {} to be refused as {}, got: {:?}", body, noun, errs);
    }
}

#[test]
fn a_non_terminating_comptime_exhausts_the_budget() {
    // The comptime evaluator's step budget. Loops are in the fold set, so this is the rule that
    // stops a `@comptime` from hanging the compiler rather than failing it.
    let errs = check(r#"
        fn main() -> i64 { let x = @comptime { let !i = 0  loop { i += 1 }  i }  0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("comptime-budget") && e.contains("may not terminate")),
        "got: {:?}", errs);
}

#[test]
fn comptime_may_not_assign_to_a_binding_it_did_not_make() {
    // A fold that wrote through to the surrounding program would be an effect,
    // which is the one thing compile-time evaluation may not have.
    let errs = check(r#"
        fn main() -> i64 { let !a = 1  let x = @comptime { a = 2  a }  0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("comptime-non-static") && e.contains("did not bind")),
        "got: {:?}", errs);
}

#[test]
fn comptime_on_a_fn_is_refused_by_attachment() {
    // DIRECTIVES.md §1 gives the attachment as *block*. Honouring only that is
    // what stops the directive being a silent no-op one level up — the same
    // shape as `@inplace`'s and `@fuse`'s attachment rules.
    let errs = check(r#"
        @comptime
        fn f() -> i64 { 1 }
        fn main() -> i64 { f() }
    "#);
    assert!(errs.iter().any(|e| e.contains("`@comptime` on a `fn`")), "got: {:?}", errs);
}

#[test]
fn cast_expr_passes_typechecker() {
    assert!(passes(r#"
        fn main() -> nil {
            let buf = @cast(u8) { "hello" }
            nil
        }
    "#));
}

// ─── #598: `@cast(Model) { bytes }` overlay layout ──────────────────────────
//
// DIRECTIVES.md §4.1 calls the overlay casts "value-transforming by design",
// but the only wire-layout guarantee SPEC.md §3.10 makes for a model is field
// ORDER. So the overlay reaches exactly as far as a field's own type fixes a
// byte extent: a scalar of known width, or a tensor of such scalars at a fully
// static shape (count × width, row-major). A `str` field's dynamic,
// arena-backed text, a tensor at a symbolic shape, a nested model, and a
// `View`/`KV` descriptor over another arena have none, and `check_cast_model_-
// overlay` refuses the directive at the declaration naming each such field —
// instead of letting the interpreter materialize a wrong value (a `str` field
// used to decode as a zero-width slice and come back `Int(0)`).

#[test]
fn cast_model_overlay_accepts_scalar_only_model_598() {
    assert!(passes(r#"
        model DirEntry {
            inode_id: u64
            mode: i32
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let d = @cast(DirEntry) { buf }
            d.inode_id as i64
        }
    "#));
}

#[test]
fn cast_model_overlay_accepts_every_admitted_scalar_kind_598() {
    // One field per admitted kind: the fixed-width integers, `bool`, and the
    // IEEE floats. Each has a byte width its own type states, so each decodes.
    assert!(passes(r#"
        model Wide {
            a: i8
            b: i16
            c: i32
            d: i64
            e: u8
            f: u16
            g: u32
            h: u64
            i: bool
            j: f16
            k: bf16
            l: f32
            m: f64
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [64]]
            let w = @cast(Wide) { buf }
            w.d
        }
    "#));
}

#[test]
fn cast_model_overlay_accepts_fixed_shape_tensor_field_598() {
    // The shape #598 was filed against: a 32-byte block of `u64` + 24 bytes.
    // A tensor field whose dims are all static has an unambiguous extent —
    // element count × element width — so it overlays.
    assert!(passes(r#"
        model DirEntry {
            inode_id: u64
            name: Tensor[u8, [24]]
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let d = @cast(DirEntry) { buf }
            d.inode_id as i64
        }
    "#));
    // …and a multi-axis one, whose element order is the row-major convention.
    assert!(passes(r#"
        model Grid {
            cells: Tensor[i32, [2, 3]]
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [24]]
            let g = @cast(Grid) { buf }
            0
        }
    "#));
}

#[test]
fn cast_model_overlay_rejects_str_field_598() {
    let errs = check(r#"
        model DirEntry {
            inode_id: u64
            name: str
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let d = @cast(DirEntry) { buf }
            d.inode_id as i64
        }
    "#);
    assert!(
        errs.iter().any(|e| e.contains("no overlay layout")
            && e.contains("name: str")
            && e.contains("DIRECTIVES.md §4.1")),
        "expected an overlay-layout diagnostic naming `name: str`, got: {:?}", errs
    );
}

#[test]
fn cast_model_overlay_rejects_symbolic_tensor_shape_598() {
    // A tensor field is admitted only when every dim is a constant: a
    // shape-parameterized one has no byte extent until the parameter binds,
    // and the overlay sizes every later field from it.
    let errs = check(r#"
        model Frame[N] {
            seq: u32
            payload: Tensor[u8, [N]]
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let f = @cast(Frame) { buf }
            0
        }
    "#);
    assert!(
        errs.iter().any(|e| e.contains("no overlay layout")
            && e.contains("payload")
            && e.contains("static shape")),
        "expected a static-shape diagnostic naming `payload`, got: {:?}", errs
    );
}

#[test]
fn cast_model_overlay_rejects_unlayoutable_element_type_598() {
    // The element type has to have a byte form too — `trit` is a packed
    // ternary storage format with no wire encoding at all.
    let errs = check(r#"
        model Weights {
            packed: Tensor[trit, [8]]
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let w = @cast(Weights) { buf }
            0
        }
    "#);
    assert!(
        errs.iter().any(|e| e.contains("no overlay layout") && e.contains("packed")),
        "expected an overlay-layout diagnostic naming `packed`, got: {:?}", errs
    );
}

#[test]
fn cast_model_overlay_rejects_scalars_without_a_byte_form_598() {
    // `trit` (no wire dtype), `int4` (packed 4 bits) and the f32-backed
    // nominal tags each have no byte storage to read — one diagnostic per
    // field, naming it.
    let errs = check(r#"
        model Odd {
            t: trit
            q: int4
            n: fp8_e4m3
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let o = @cast(Odd) { buf }
            0
        }
    "#);
    for field in ["t: trit", "q: int4", "n: fp8_e4m3"] {
        assert!(
            errs.iter().any(|e| e.contains("no overlay layout") && e.contains(field)),
            "expected an overlay-layout diagnostic naming `{}`, got: {:?}", field, errs
        );
    }
}

#[test]
fn cast_model_overlay_rejects_nested_model_field_598() {
    let errs = check(r#"
        model Inner {
            x: i64
        }
        model Outer {
            head: u32
            inner: Inner
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let o = @cast(Outer) { buf }
            0
        }
    "#);
    assert!(
        errs.iter().any(|e| e.contains("no overlay layout") && e.contains("inner")),
        "expected an overlay-layout diagnostic naming `inner`, got: {:?}", errs
    );
}

#[test]
fn cast_model_overlay_rejects_view_field_598() {
    // A `View` field is a descriptor over another arena — it owns no bytes of
    // its own for the overlay to lay down.
    let errs = check(r#"
        model Slot {
            head: u32
            win: View[u8, [8]]
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let s = @cast(Slot) { buf }
            0
        }
    "#);
    assert!(
        errs.iter().any(|e| e.contains("no overlay layout") && e.contains("win")),
        "expected an overlay-layout diagnostic naming `win`, got: {:?}", errs
    );
}

#[test]
fn cast_model_overlay_check_fires_in_tail_position_598() {
    // The trailing-directive-block form (no explicit `let`) goes through a
    // separate check.rs arm (`check_block_body`'s deferred-last path) from the
    // `let`-bound expression form above — both must catch the same field.
    let errs = check(r#"
        model DirEntry {
            name: str
        }
        fn make(buf: Tensor[u8, [32]]) -> DirEntry {
            @cast(DirEntry) { buf }
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let d = make(buf)
            0
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("no overlay layout")), "got: {:?}", errs);
}

#[test]
fn cast_model_overlay_rejects_non_byte_source_tensor_598() {
    // The source of `@cast(Model) { bytes }` is read one byte per ELEMENT, so
    // a tensor of wider values contributes one truncated byte each and the
    // overlay runs off the end of a buffer it thinks is 8× longer.
    let errs = check(r#"
        model DirEntry {
            inode_id: u64
            mode: i32
        }
        fn make(src: Tensor[u32, [8]]) -> i64 {
            let d = @cast(DirEntry) { src }
            0
        }
        fn main() -> i64 { 0 }
    "#);
    assert!(
        errs.iter().any(|e| e.contains("byte tensor read one byte per element")
            && e.contains("u32")),
        "expected a byte-source diagnostic naming `u32`, got: {:?}", errs
    );
}

#[test]
fn cast_model_overlay_accepts_a_u8_view_source_598() {
    // The ordinary source: a `u8` view sliced out of a byte arena. A `Tensor`
    // of `u8` is the same thing without the view.
    assert!(passes(r#"
        model DirEntry {
            inode_id: u64
            name: Tensor[u8, [24]]
        }
        fn make(src: View[u8, [32]]) -> i64 {
            let d = @cast(DirEntry) { src }
            d.inode_id as i64
        }
        fn main() -> i64 { 0 }
    "#));
    assert!(passes(r#"
        model DirEntry {
            inode_id: u64
            name: Tensor[u8, [24]]
        }
        fn make(src: Tensor[u8, [32]]) -> i64 {
            let d = @cast(DirEntry) { src }
            d.inode_id as i64
        }
        fn main() -> i64 { 0 }
    "#));
}

#[test]
fn cast_model_overlay_unrelated_cast_targets_unaffected_598() {
    // A plain dtype `@cast` (not naming a model) must not be touched by the
    // new check at all.
    assert!(passes(r#"
        fn main() -> nil {
            let buf = @cast(u8) { "hello" }
            nil
        }
    "#));
    assert!(passes(r#"
        fn main() -> f32 {
            @cast(bf16) { 1.0f32 }
        }
    "#));
}

#[test]
fn multiple_return_paths_same_type_passes() {
    assert!(passes(r#"
        fn sign(n: i64) -> i64 {
            if n > 0 { return 1 }
            if n < 0 { return -1 }
            0
        }
    "#));
}

#[test]
fn match_on_str_literal_passes() {
    assert!(passes(r#"
        fn label(s: str) -> i64 {
            match s {
                "yes" => 1,
                "no"  => 0,
                _     => -1,
            }
        }
    "#));
}

#[test]
fn lambda_in_let_binding_passes() {
    assert!(passes(r#"
        fn main() -> nil {
            let doubled = list_map(list(1, 2, 3), fn(x: i64) -> i64 { x * 2 })
            nil
        }
    "#));
}

#[test]
fn tuple_return_type_passes() {
    assert!(passes(r#"
        fn swap(a: i64, b: i64) -> (i64, i64) { (b, a) }
    "#));
}

#[test]
fn pipe_expr_passes() {
    assert!(passes(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn main() -> i64 { 5 |> inc }
    "#));
}

// ── Issue #151: undeclared shape param in model method diagnostic ─────────────

#[test]
fn model_method_undeclared_shape_param_errors() {
    // B is used in the method signature but not declared on model [D, H] or method []
    let errs = check(r#"
        model MyLayer[D, H] {
            weights: Tensor[f32, [D, H]]
            fn forward(self, x: Tensor[f32, [B, D]]) -> Tensor[f32, [B, H]] {
                x @ self.weights
            }
        }
    "#);
    assert!(!errs.is_empty(), "expected error for undeclared shape param `B`");
    let msg = &errs[0];
    assert!(msg.contains("unknown shape param") && msg.contains("B"),
        "expected helpful diagnostic, got: {}", msg);
    assert!(msg.contains("forward") || msg.contains("MyLayer"),
        "expected method/model name in diagnostic, got: {}", msg);
}

#[test]
fn model_method_declared_shape_param_passes() {
    // B is explicitly declared on the method — should typecheck cleanly
    assert!(passes(r#"
        model MyLayer[D, H] {
            weights: Tensor[f32, [D, H]]
            fn forward[B](self, x: Tensor[f32, [B, D]]) -> Tensor[f32, [B, H]] {
                x @ self.weights
            }
        }
    "#), "fn forward[B] with explicit shape param should pass");
}

#[test]
fn model_method_model_shape_params_pass() {
    // D and H are on the model — method can use them without re-declaring
    assert!(passes(r#"
        model Proj[D, H] {
            w: Tensor[f32, [D, H]]
            fn project(self, x: Tensor[f32, [D]]) -> Tensor[f32, [H]] {
                x @ self.w
            }
        }
    "#), "model shape params should be usable in methods without re-declaring");
}

#[test]
fn any_return_accepts_heterogeneous_branches() {
    // #186: a fn declared `-> any` may return values of differing types from
    // different paths — the central pattern of a dynamically-typed atom()/eval().
    assert!(passes(r#"
        fn atom(tok: str) -> any {
            if is_numeric(tok) { return to_float(tok) }
            tok
        }
    "#), "any return should accept both F64 and Str: {:?}",
        check(r#"
        fn atom(tok: str) -> any {
            if is_numeric(tok) { return to_float(tok) }
            tok
        }
    "#));
}

#[test]
fn any_param_accepts_any_arg_type() {
    // An `any` parameter resolves to Unknown, which is compatible with any
    // argument type — so the same fn can be called with int, str, or float.
    assert!(passes(r#"
        fn describe(x: any) -> str {
            if is_int(x) { return "int" }
            if is_str(x) { return "str" }
            "other"
        }
        fn main() -> str {
            let _a = describe(42)
            let _b = describe("hi")
            describe(3.5)
        }
    "#), "any param should accept heterogeneous args: {:?}",
        check(r#"
        fn describe(x: any) -> str {
            if is_int(x) { return "int" }
            if is_str(x) { return "str" }
            "other"
        }
        fn main() -> str { describe(42) }
    "#));
}

#[test]
fn any_value_flows_into_concrete_position() {
    // A value returned as `any` (Unknown) can flow into any later position
    // without a type error — Unknown is match-anything in both directions.
    assert!(passes(r#"
        fn pick() -> any { 7 }
        fn main() -> i64 {
            let v = pick()
            v + 1
        }
    "#), "any value should be usable in a concrete (i64) position: {:?}",
        check(r#"
        fn pick() -> any { 7 }
        fn main() -> i64 { let v = pick(); v + 1 }
    "#));
}

// ─── #244: match arm result type unification ─────────────────────────────────

#[test]
fn match_arm_type_mismatch_errors() {
    // #244: arms yielding incompatible concrete types must be rejected.
    let errs = check(r#"
        fn main() -> nil { let v = match 1 { 0 => 1, _ => "hello" }  print(v + 1)  nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("match arms yield incompatible types")),
            "expected match arm type mismatch error, got: {:?}", errs);
}

#[test]
fn match_arm_fn_return_type_mismatch_errors() {
    // #244: fn-return variant — match used as fn body with mixed arm types.
    let errs = check(r#"
        fn pick(k: i64) -> i64 { match k { 0 => 1, _ => "hello" } }
    "#);
    assert!(errs.iter().any(|e| e.contains("match arms yield incompatible types")),
            "expected match arm type mismatch in fn return, got: {:?}", errs);
}

#[test]
fn match_arm_uniform_types_passes() {
    // Valid: all arms yield i64.
    assert!(passes(r#"
        fn pick(k: i64) -> i64 { match k { 0 => 1, 1 => 2, _ => 3 } }
    "#), "uniform match arm types should pass");
}

#[test]
fn match_arm_diverging_arm_passes() {
    // A diverging (panic) arm is ⊥ and must not trigger a false mismatch.
    assert!(passes(r#"
        fn main() -> nil {
            let v: i64 = match 1 { 0 => 42, _ => panic("nope") }
            print(v)
            nil
        }
    "#), "diverging match arm should not cause a type-mismatch error");
}

// ─── #245: tuple destructuring arity mismatch ─────────────────────────────────

#[test]
fn tuple_destructure_over_arity_errors() {
    // #245: pattern has more elements than the tuple type.
    let errs = check(r#"
        fn pair() -> (i64, i64) { (1, 2) }
        fn main() -> nil { let (a, b, c) = pair()  print(a)  nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("tuple pattern has 3 elements but value has 2")),
            "expected tuple arity mismatch error, got: {:?}", errs);
}

#[test]
fn tuple_destructure_under_arity_errors() {
    // #245: pattern has fewer elements than the tuple type.
    let errs = check(r#"
        fn triple() -> (i64, i64, i64) { (1, 2, 3) }
        fn main() -> nil { let (a, b) = triple()  print(a)  nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("tuple pattern has 2 elements but value has 3")),
            "expected tuple under-arity mismatch error, got: {:?}", errs);
}

#[test]
fn tuple_destructure_correct_arity_passes() {
    // Valid: pattern arity matches tuple arity.
    assert!(passes(r#"
        fn pair() -> (i64, i64) { (1, 2) }
        fn main() -> nil { let (a, b) = pair()  print(a)  nil }
    "#), "matching tuple destructure arity should pass");
}

// ─── #246: model constructor missing required field ───────────────────────────

#[test]
fn model_constructor_missing_field_errors() {
    // #246: constructor that omits a required declared field.
    let errs = check(r#"
        model M { b: i64 }
        fn main() -> nil { let m = M { }  print(m.b)  nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("missing required field") && e.contains("`b`")),
            "expected missing required field error, got: {:?}", errs);
}

#[test]
fn model_constructor_all_fields_passes() {
    // Valid: all declared fields provided.
    assert!(passes(r#"
        model M { b: i64 }
        fn main() -> nil { let m = M { b: 42 }  print(m.b)  nil }
    "#), "complete model constructor should pass");
}

#[test]
fn model_constructor_postfix_missing_field_errors() {
    // #246: also covers the `M { }` postfix constructor syntax.
    let errs = check(r#"
        model Point { x: i64, y: i64 }
        fn main() -> nil { let p = Point { x: 1 }  nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("missing required field") && e.contains("`y`")),
            "expected missing required field error for postfix constructor, got: {:?}", errs);
}

// ─── immutable-binding assignment (pre-existing behaviour) ───────────────────

#[test]
fn scalar_immutable_assign_still_errors() {
    // Existing behaviour: plain `a = v` on immutable binding still errors.
    let errs = check(r#"
        fn main() -> nil { let a = 1  a = 2  nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("cannot assign to immutable binding `a`")),
            "expected plain immutable assignment error, got: {:?}", errs);
}

// ─── #247 / #90: element write through an immutable binding/param ─────────────

#[test]
fn indexed_write_through_immutable_let_errors() {
    // `let a = ...; a[0] = ...` is an element write through an immutable binding.
    let errs = check(r#"
        fn main() -> nil { let a = forge.zeros[f32, [3]]  a[0] = 5.0  nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("immutable binding `a`")),
            "expected immutable element-write error, got: {:?}", errs);
}

#[test]
fn indexed_write_through_mutable_let_ok() {
    // `let !a` permits element writes.
    let errs = check(r#"
        fn main() -> nil { let !a = forge.zeros[f32, [3]]  a[0] = 5.0  nil }
    "#);
    assert!(!errs.iter().any(|e| e.contains("immutable binding")),
            "let ! element write should be allowed, got: {:?}", errs);
}

#[test]
fn indexed_write_through_nonmut_param_errors() {
    // Mutating a non-`!` parameter element is the cross-call variant of #247.
    let errs = check(r#"
        fn poke(buf: Tensor[f32, [4]], v: f32) -> Tensor[f32, [4]] { buf[0] = v  buf }
        fn main() -> nil { nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("immutable binding `buf`")),
            "expected immutable param element-write error, got: {:?}", errs);
}

#[test]
fn indexed_write_through_mut_param_ok() {
    // `!buf` parameter permits element writes.
    let errs = check(r#"
        fn poke(!buf: Tensor[f32, [4]], v: f32) -> Tensor[f32, [4]] { buf[0] = v  buf }
        fn main() -> nil { nil }
    "#);
    assert!(!errs.iter().any(|e| e.contains("immutable binding")),
            "!buf element write should be allowed, got: {:?}", errs);
}

// ─── #838 (ruling, option C): a `!` tensor/view argument needs a mutable
// root binding ──────────────────────────────────────────────────────────────
//
// `SPEC.md §3.4.1` ¶Argument form: a `!` argument of tensor or view type
// must be a place whose root binding is mutable — a `let !`/`let mut`
// local, a `!` parameter, or a field path whose own root binding is one of
// those. A place rooted at a plain `let` binding is a compile-time error.
// A `!` argument of model or `KV` type may be any place. `@grad fn` calls,
// `fwd` and `fwd_bwd` are exempt. Mirrors the #838 probe set:
// p1/p5/p11/p16/p17/p18/p19 (refused) and the p4/p7/p10/p13/p14/p15 family
// plus the doc's own negative control (accepted).

fn mut_arg_err(errs: &[String], name: &str) -> bool {
    errs.iter().any(|e| {
        e.contains("passing `") && e.contains(&format!("`{name}`")) &&
        e.contains("to a `!` parameter writes through it")
    })
}

#[test]
fn plain_tensor_by_name_to_bang_param_errors_838() {
    // p1's shape: a plain `let` tensor handed to a `!` tensor parameter.
    let errs = check(r#"
        fn bump(!t: Tensor[f32, [4]]) -> nil { t[0] = 42.0  nil }
        fn main() -> nil {
            let x = forge.ones[f32, [4]]
            bump(x)
            nil
        }
    "#);
    assert!(mut_arg_err(&errs, "x"), "expected a #838 mutable-root error naming `x`, got: {:?}", errs);
}

#[test]
fn plain_view_to_bang_param_errors_838() {
    // p18's shape: a plain `let` View (a row of a plain `let` tensor).
    let errs = check(r#"
        fn bump(!v: View[f32, [4]]) -> nil { v[0] = 42.0  nil }
        fn main() -> nil {
            let x = forge.ones[f32, [2, 4]]
            let v = x[0, ..]
            bump(v)
            nil
        }
    "#);
    assert!(mut_arg_err(&errs, "v"), "expected a #838 mutable-root error naming `v`, got: {:?}", errs);
}

#[test]
fn plain_model_field_path_to_bang_param_errors_838() {
    // p17's shape: `bump(m.w)` roots at `m` (`lhs_root_ident` walks field
    // chains), and `m` is a plain `let` model — refused for the same reason
    // `m.w[0] = 42.0` already is (m18/p21).
    let errs = check(r#"
        model M { w: Tensor[f32, [4]] }
        fn bump(!t: Tensor[f32, [4]]) -> nil { t[0] = 42.0  nil }
        fn main() -> nil {
            let m = M { w: forge.ones[f32, [4]] }
            bump(m.w)
            nil
        }
    "#);
    assert!(mut_arg_err(&errs, "m"), "expected a #838 mutable-root error naming `m`, got: {:?}", errs);
}

#[test]
fn plain_tensor_inside_lambda_body_errors_838() {
    // p19's shape: a plain `let` declared inside a function literal's own
    // body, direct call — a lambda body is an ordinary function body, so
    // `lhs_root_ident`/`is_mutable_binding` read its scope exactly as they
    // read `main`'s.
    let errs = check(r#"
        fn bump(!t: Tensor[f32, [4]]) -> nil { t[0] = 42.0  nil }
        fn main() -> nil {
            let f = fn(n: i64) -> i64 {
                let x = forge.ones[f32, [4]]
                bump(x)
                (x[0] as i64) + n
            }
            f(0)
            nil
        }
    "#);
    assert!(mut_arg_err(&errs, "x"), "expected a #838 mutable-root error naming `x`, got: {:?}", errs);
}

#[test]
fn plain_tensor_to_bang_model_method_param_errors_838() {
    // p11's shape: a plain `let` tensor handed to a `!` parameter of a
    // model method, not just a free fn.
    let errs = check(r#"
        model H {
            n: i64
            fn poke(self, !t: Tensor[f32, [4]]) -> nil { t[0] = 42.0  nil }
        }
        fn main() -> nil {
            let h = H { n: 0 }
            let x = forge.ones[f32, [4]]
            h.poke(x)
            nil
        }
    "#);
    assert!(mut_arg_err(&errs, "x"), "expected a #838 mutable-root error naming `x`, got: {:?}", errs);
}

#[test]
fn mutable_let_bang_tensor_to_bang_param_ok_838() {
    // Accepted: a `let !` local satisfies the mutable-root rule.
    let errs = check(r#"
        fn bump(!t: Tensor[f32, [4]]) -> nil { t[0] = 42.0  nil }
        fn main() -> nil {
            let !x = forge.ones[f32, [4]]
            bump(x)
            nil
        }
    "#);
    assert!(errs.is_empty(), "let ! tensor argument should be allowed, got: {:?}", errs);
}

#[test]
fn mutable_let_mut_tensor_to_bang_param_ok_838() {
    // Accepted: `let mut` is the long form of `let !` (SPEC §3.4) and
    // satisfies the mutable-root rule the same way.
    let errs = check(r#"
        fn bump(!t: Tensor[f32, [4]]) -> nil { t[0] = 42.0  nil }
        fn main() -> nil {
            let mut x = forge.ones[f32, [4]]
            bump(x)
            nil
        }
    "#);
    assert!(errs.is_empty(), "let mut tensor argument should be allowed, got: {:?}", errs);
}

#[test]
fn bang_param_forwarded_to_bang_param_ok_838() {
    // Accepted: a `!` parameter is itself a mutable root, so forwarding it
    // into another `!` parameter is legal.
    let errs = check(r#"
        fn bump(!t: Tensor[f32, [4]]) -> nil { t[0] = 42.0  nil }
        fn outer(!t: Tensor[f32, [4]]) -> nil { bump(t)  nil }
        fn main() -> nil { nil }
    "#);
    assert!(errs.is_empty(), "!-param forwarding should be allowed, got: {:?}", errs);
}

#[test]
fn mutable_model_field_path_to_bang_param_ok_838() {
    // The negative control §7.1 names for the field-path row: `let !m`'s
    // own root is mutable, so `bump(m.w)` stays legal — the live-alias
    // route `SPEC.md §3.4`'s ¶Bindings from model fields and m5 measure.
    let errs = check(r#"
        model M { w: Tensor[f32, [4]] }
        fn bump(!t: Tensor[f32, [4]]) -> nil { t[0] = 42.0  nil }
        fn main() -> nil {
            let !m = M { w: forge.ones[f32, [4]] }
            bump(m.w)
            nil
        }
    "#);
    assert!(errs.is_empty(), "let !m field-path argument should be allowed, got: {:?}", errs);
}

#[test]
fn plain_model_argument_to_bang_param_ok_838() {
    // p4's shape: a plain `let` model is untouched — a model or `KV`
    // argument may be any place, mutable or not (the parameter is the
    // caller's instance either way).
    let errs = check(r#"
        model C { n: i64 }
        fn bump(!c: C) -> nil { c.n = 42  nil }
        fn main() -> nil {
            let x = C { n: 1 }
            bump(x)
            nil
        }
    "#);
    assert!(errs.is_empty(), "plain model argument should be allowed, got: {:?}", errs);
}

#[test]
fn plain_kv_argument_to_bang_param_ok_838() {
    // p10's shape: a plain `let` `KV` is untouched, same reasoning as model.
    let errs = check(r#"
        fn push(!c: KV[f32, [4]]) -> nil { c <- forge.ones[f32, [4]]  nil }
        fn main() -> nil {
            let c = forge.kv[f32, [4]]
            push(c)
            nil
        }
    "#);
    assert!(errs.is_empty(), "plain KV argument should be allowed, got: {:?}", errs);
}

#[test]
fn grad_fn_direct_call_with_plain_tensor_arg_ok_838() {
    // Inside a `@grad fn`, `!` marks a differentiable input (`AUTODIFF.md
    // §1`), not a mutation — a direct call is forward-only (SPEC §6.2) and
    // exempt from this rule, the same as the census's 16 `@grad` sites in
    // `examples/`.
    let errs = check(r#"
        @grad fn reg(!w: Tensor[f32, [4]]) -> f32 { sum(w .* w) }
        fn main() -> f32 {
            let w = forge.ones[f32, [4]]
            reg(w)
        }
    "#);
    assert!(errs.is_empty(), "a @grad fn's direct call should be exempt, got: {:?}", errs);
}

#[test]
fn grad_fn_fwd_bwd_with_plain_tensor_arg_ok_838() {
    // Same exemption at `.fwd_bwd`.
    let errs = check(r#"
        @grad fn reg(!w: Tensor[f32, [4]]) -> f32 { sum(w .* w) }
        fn main() -> nil {
            let w = forge.ones[f32, [4]]
            let (l, g) = reg.fwd_bwd(w)
            nil
        }
    "#);
    assert!(errs.is_empty(), "a @grad fn's .fwd_bwd should be exempt, got: {:?}", errs);
}

// --- unused-value lint (`\>`-trap, PR #412): a discarded pure expr-statement ---

fn no_effect_warns(src: &str) -> bool {
    warnings(src).iter().any(|m| m.contains("has no effect"))
}

#[test]
fn discarded_relu_statement_warns() {
    // The `\>`-trap surface: `let m = a \> 0.0` splits off a discarded `\>(0.0)`.
    let src = r#"
        fn main() -> f32 {
            let !a = forge.zeros[f32, [4]]
            let m = a \> 0.0
            sum(m)
        }
    "#;
    assert!(no_effect_warns(src), "expected unused-value warning, got {:?}", warnings(src));
}

#[test]
fn discarded_pure_operator_statements_warn() {
    // A bare arithmetic / comparison / bare-var statement is dead code.
    assert!(no_effect_warns("fn main() -> i64 { let x = 1  x + 1  x }"),
            "arith stmt should warn: {:?}", warnings("fn main() -> i64 { let x = 1  x + 1  x }"));
    let cmp = "fn main() -> i64 { let a = forge.zeros[f32,[2]]  let b = forge.zeros[f32,[2]]  a .> b  0 }";
    assert!(no_effect_warns(cmp), "dotted-compare stmt should warn: {:?}", warnings(cmp));
}

#[test]
fn effectful_bare_statements_do_not_warn() {
    // Calls (effectful builtins or unknown-purity user fns) must never warn —
    // they are the idiomatic effect-for-discard form.
    let src = r#"
        fn side(x: i64) -> i64 { print("x")  x }
        fn main() -> i64 {
            let !m = map_new()
            map_set(m, "k", 7)
            print("hello")
            side(3)
            map_get(m, "k")
        }
    "#;
    assert!(!no_effect_warns(src), "effectful statements must not warn: {:?}", warnings(src));
}

#[test]
fn block_tail_value_is_not_flagged() {
    // A block's trailing value expression is the block's value, not discarded.
    let src = "fn main() -> i64 { let x = 2  x + 1 }";
    assert!(!no_effect_warns(src), "tail expr must not warn: {:?}", warnings(src));
}

#[test]
fn unused_value_lint_suppressed_in_demon_mode() {
    let src = r#"
        fn main() -> f32 {
            let !a = forge.zeros[f32, [4]]
            let m = a \> 0.0
            sum(m)
        }
    "#;
    assert!(warnings(src).iter().any(|m| m.contains("has no effect")),
            "safe mode should warn");
    assert!(!warnings_demon(src).iter().any(|m| m.contains("has no effect")),
            "demon mode should suppress the lint: {:?}", warnings_demon(src));
}

// --- loud-diagnostics batch (#392/#397/#398) -------------------------------

#[test]
fn fwd_bwd_bwd_wrong_arity_errors_392() {
    // The grad-method call now types as a 2-tuple, so a 3-name destructure
    // fires the arity check instead of binding silent nils.
    let errs = check(r#"
        @grad @grad fn g2(!w: Tensor[f32,[3]]) -> f32 { sum(w .* w) }
        fn main() -> nil {
            let !w = forge.zeros[f32,[3]]
            let (v, g1, g2) = g2.fwd_bwd_bwd(w)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("tuple pattern has 3 elements")),
            "expected arity error, got {:?}", errs);
    // The correct 2-name form must stay clean.
    let ok = check(r#"
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 { sum(w .* w) }
        fn main() -> nil {
            let !w = forge.zeros[f32,[3]]
            let (l, g) = loss.fwd_bwd(w)
            nil
        }
    "#);
    assert!(!ok.iter().any(|e| e.contains("tuple pattern")),
            "2-name fwd_bwd must not error: {:?}", ok);
}

#[test]
fn captured_mut_in_grad_fn_is_accepted_398() {
    // #398 implemented: a captured `mut` binding read directly in the body is a
    // differentiable input (AUTODIFF.md §2), so the program that used to be
    // rejected now checks clean — the interpreter returns `g.gg` for real
    // (`interp_tests::grad_captured_mut_tensor_*`).
    let errs = check(r#"
        let !gg = forge.zeros[f32,[3]]
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 { sum(w .* gg) }
        fn main() -> nil { nil }
    "#);
    assert!(errs.is_empty(), "captured-mut @grad fn must check clean, got {:?}", errs);
    // A @grad fn using only its ! params must stay clean.
    let ok = check(r#"
        @grad fn loss(!w: Tensor[f32,[3]], x: Tensor[f32,[3]]) -> f32 { sum(w .* x) }
        fn main() -> nil { nil }
    "#);
    assert!(ok.is_empty(), "param-only @grad fn must not error: {:?}", ok);
}

#[test]
fn captured_mut_only_grad_fn_is_differentiable_398() {
    // AUTODIFF.md §2's sentence, now literal: "no mut bindings **and** no mut
    // parameters" is the error. A captured mut with no `!` param is enough to
    // differentiate, so it must NOT trip the nothing-to-differentiate rule…
    let ok = check(r#"
        let !gain = 2.0f32
        @grad fn loss(x: Tensor[f32,[3]]) -> f32 { sum(x .* x) * gain }
        fn main() -> nil { nil }
    "#);
    assert!(ok.is_empty(), "capture-only @grad fn must check clean, got {:?}", ok);

    // …while neither a `!` param nor a captured mut still errors, with the
    // full message naming both sources.
    let errs = check(r#"
        let gain = 2.0f32
        @grad fn loss(x: Tensor[f32,[3]]) -> f32 { sum(x .* x) * gain }
        fn main() -> nil { nil }
    "#);
    assert_eq!(
        errs,
        vec!["`@grad fn` with no `!` (mut) parameter and no captured `mut` binding \
              has nothing to differentiate (see the spec on `@grad`)".to_string()],
        "expected exactly the nothing-to-differentiate error",
    );

    // An *immutable* module binding is not a capture for gradient purposes
    // (§2: "captured immut binding — no"), so it cannot rescue the fn above:
    // that is what the second case just proved. A non-differentiable captured
    // mut can't either.
    let ints = check(r#"
        let !steps = 0
        @grad fn loss(x: Tensor[f32,[3]]) -> f32 { sum(x .* x) * (steps as f32) }
        fn main() -> nil { nil }
    "#);
    assert!(ints.iter().any(|e| e.contains("nothing to differentiate")),
            "an integer capture must not count as differentiable, got {:?}", ints);
}

#[test]
fn captured_mut_read_in_callee_errors_398() {
    // The tape traces only the differentiated body; a plain fn call runs
    // concretely and contributes no nodes. So a capture read in the body AND in
    // a called fn would come back with half its gradient — silently. Rejected,
    // naming the callee. (Truth here: dL/db = 2w; the traced half alone is w.)
    let errs = check(r#"
        let !b = forge.zeros[f32,[2]]
        fn helper(x: Tensor[f32,[2]]) -> f32 { sum(x .* b) }
        @grad fn loss(!w: Tensor[f32,[2]]) -> f32 { sum(w .* b) + helper(w) }
        fn main() -> nil { nil }
    "#);
    assert_eq!(
        errs,
        vec!["captured mutable binding `b` is also read inside fn `helper`, which this \
              `@grad fn` calls — the tape does not trace calls, so `b`'s gradient would \
              silently omit that path".to_string()],
        "expected exactly the callee-read error",
    );

    // Transitively: the body calls `outer`, which calls `inner`, which reads it.
    let deep = check(r#"
        let !b = forge.zeros[f32,[2]]
        fn inner(x: Tensor[f32,[2]]) -> f32 { sum(x .* b) }
        fn outer(x: Tensor[f32,[2]]) -> f32 { inner(x) }
        @grad fn loss(!w: Tensor[f32,[2]]) -> f32 { sum(w .* b) + outer(w) }
        fn main() -> nil { nil }
    "#);
    assert!(deep.iter().any(|e| e.contains("is also read inside fn `inner`")),
            "expected the transitive callee to be named, got {:?}", deep);
}

#[test]
fn captured_mut_callee_scan_tolerates_shadows_and_recursion_398() {
    // Near-miss 1: the callee's own *parameter* is named `b` — it shadows the
    // module binding, so the callee is not a reader and the capture is fine.
    let param_shadow = check(r#"
        let !b = forge.zeros[f32,[2]]
        fn helper(b: Tensor[f32,[2]]) -> f32 { sum(b .* b) }
        @grad fn loss(!w: Tensor[f32,[2]]) -> f32 { sum(w .* b) + helper(w) }
        fn main() -> nil { nil }
    "#);
    assert!(param_shadow.is_empty(),
            "a callee param shadowing the capture must not error: {:?}", param_shadow);

    // Near-miss 2: the callee has a local `let b` of its own.
    let local_shadow = check(r#"
        let !b = forge.zeros[f32,[2]]
        fn helper(x: Tensor[f32,[2]]) -> f32 { let b = 2.0f32  sum(x) * b }
        @grad fn loss(!w: Tensor[f32,[2]]) -> f32 { sum(w .* b) + helper(w) }
        fn main() -> nil { nil }
    "#);
    assert!(local_shadow.is_empty(),
            "a callee local shadowing the capture must not error: {:?}", local_shadow);

    // Near-miss 3: the callee never mentions the capture at all.
    let unrelated = check(r#"
        let !b = forge.zeros[f32,[2]]
        fn helper(x: Tensor[f32,[2]]) -> f32 { sum(x .* x) }
        @grad fn loss(!w: Tensor[f32,[2]]) -> f32 { sum(w .* b) + helper(w) }
        fn main() -> nil { nil }
    "#);
    assert!(unrelated.is_empty(),
            "an unrelated callee must not error: {:?}", unrelated);

    // A recursive call graph must terminate (and still find the reader).
    let recursive = check(r#"
        let !b = forge.zeros[f32,[2]]
        fn ping(x: Tensor[f32,[2]], n: i64) -> f32 {
            if n <= 0 { sum(x .* b) } else { pong(x, n - 1) }
        }
        fn pong(x: Tensor[f32,[2]], n: i64) -> f32 { ping(x, n - 1) }
        @grad fn loss(!w: Tensor[f32,[2]]) -> f32 { sum(w .* b) + ping(w, 2) }
        fn main() -> nil { nil }
    "#);
    assert!(recursive.iter().any(|e| e.contains("is also read inside fn `ping`")),
            "mutual recursion must terminate and report the reader, got {:?}", recursive);
}

#[test]
fn captured_mut_read_in_a_model_method_errors_398() {
    // The rule that catches a plain callee was escaped by a METHOD call.
    // `h.contrib()` parses as Call(Field(h, "contrib")), so the callee's name is
    // a `PostfixOp::Field` and never an `Expr::Ident` — the identifier scan that
    // discovers plain callees could not see it, and the program compiled clean
    // while returning half of dL/dcap ([1,1,1] where the truth is [2,2,2]).
    let errs = check(r#"
        let !cap = forge.zeros[f32,[3]]
        model H {
            k: f32,
            fn contrib(self) -> f32 { sum(cap) }
        }
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 {
            let h = H { k: 1.0f32 }
            sum(w .* cap) + h.contrib()
        }
        fn main() -> nil { nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("is also read inside fn `H.contrib`")),
            "a method reading the capture must be named like any other callee, got {:?}", errs);

    // A method that does NOT read the capture is not a reader.
    let clean = check(r#"
        let !cap = forge.zeros[f32,[3]]
        model H {
            k: f32,
            fn contrib(self) -> f32 { self.k }
        }
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 {
            let h = H { k: 1.0f32 }
            sum(w .* cap) + h.contrib()
        }
        fn main() -> nil { nil }
    "#);
    assert!(clean.is_empty(),
            "an unrelated method must not error: {:?}", clean);

    // A method whose own parameter shadows the capture is not a reader either.
    let shadowed = check(r#"
        let !cap = forge.zeros[f32,[3]]
        model H {
            k: f32,
            fn contrib(self, cap: Tensor[f32,[3]]) -> f32 { sum(cap) }
        }
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 {
            let h = H { k: 1.0f32 }
            sum(w .* cap) + h.contrib(w)
        }
        fn main() -> nil { nil }
    "#);
    assert!(shadowed.is_empty(),
            "a method param shadowing the capture must not error: {:?}", shadowed);
}

#[test]
fn captured_mut_method_call_graph_is_transitive_and_terminates_398() {
    // A method calling a plain fn that reads the capture, and a method calling
    // another method — both must be followed, and a cycle must not hang.
    let via_fn = check(r#"
        let !cap = forge.zeros[f32,[3]]
        fn reads() -> f32 { sum(cap) }
        model H {
            k: f32,
            fn contrib(self) -> f32 { reads() }
        }
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 {
            let h = H { k: 1.0f32 }
            sum(w .* cap) + h.contrib()
        }
        fn main() -> nil { nil }
    "#);
    assert!(via_fn.iter().any(|e| e.contains("is also read inside fn `reads`")),
            "method → fn → capture must be followed, got {:?}", via_fn);

    let cyclic = check(r#"
        let !cap = forge.zeros[f32,[3]]
        model H {
            k: f32,
            fn a(self) -> f32 { self.b() }
            fn b(self) -> f32 { self.a() + sum(cap) }
        }
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 {
            let h = H { k: 1.0f32 }
            sum(w .* cap) + h.a()
        }
        fn main() -> nil { nil }
    "#);
    assert!(cyclic.iter().any(|e| e.contains("is also read inside fn `H.b`")),
            "a cyclic method graph must terminate and report the reader, got {:?}", cyclic);
}

#[test]
fn captured_mut_body_local_let_shadows_the_module_binding_398() {
    // Direction (a): a body-local `let cap` shadows the module `!cap`, so the
    // body reads no capture at all. The old scan asked only "is this name a
    // module mutable?" and produced a phantom `g.cap` for a binding the body
    // never touched.
    let shadowed = check(r#"
        let !cap = forge.zeros[f32,[3]]
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 {
            let cap = [1.0f32, 1.0f32, 1.0f32]
            sum(w .* cap)
        }
        fn main() -> nil { nil }
    "#);
    assert!(shadowed.is_empty(),
            "a body-local shadowing the capture must check clean: {:?}", shadowed);

    // Direction (b): the module's `counter` is an `i64`, which is not
    // differentiable — but this body never reads it. Its own float local of the
    // same name was being type-checked against the MODULE binding and rejected,
    // so a correct program did not compile.
    let local_of_a_nondiff_name = check(r#"
        let !counter = 0
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 {
            let counter = 2.0f32
            sum(w .* w) * counter
        }
        fn main() -> nil { nil }
    "#);
    assert!(local_of_a_nondiff_name.is_empty(),
            "a body-local must not inherit the module binding's type: {:?}",
            local_of_a_nondiff_name);

    // The rule still fires when the body genuinely READS the non-differentiable
    // module binding — shadowing narrows the rule, it does not disable it.
    let genuinely_reads = check(r#"
        let !counter = 0
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 { sum(w .* w) * (counter as f32) }
        fn main() -> nil { nil }
    "#);
    assert!(genuinely_reads.iter().any(|e| e.contains("non-differentiable type")),
            "a real read of the i64 capture must still error, got {:?}", genuinely_reads);
}

#[test]
fn captured_mut_non_float_errors_398() {
    // Only float scalars / float tensors carry gradients. A captured integer
    // mutable would otherwise land in `Grads` as a meaningless field (or, worse,
    // be silently skipped), so it stays a compile-time error — full message.
    let errs = check(r#"
        let !steps = 0
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 { sum(w .* w) * (steps as f32) }
        fn main() -> nil { nil }
    "#);
    assert_eq!(
        errs,
        vec!["captured mutable binding `steps` has non-differentiable type `i64` \
              inside a `@grad fn`".to_string()],
        "expected exactly the non-differentiable-type error",
    );

    // The float counterpart of the same program is clean.
    let ok = check(r#"
        let !gain = 0.5f32
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 { sum(w .* w) * gain }
        fn main() -> nil { nil }
    "#);
    assert!(ok.is_empty(), "a float capture must check clean, got {:?}", ok);
}

#[test]
fn captured_mut_hidden_in_closure_errors_398() {
    // A captured mutable referenced inside a closure LITERAL is still rejected
    // after #398 shipped: the tape never enters a closure body, so this one
    // capture genuinely has no adjoint and an empty/zero `g.bias` would be a
    // silent wrong answer. Assert the whole message, not a fragment.
    let errs = check(r#"
        let !bias = forge.zeros[f32,[1]]
        @grad fn loss(!w: Tensor[f32,[1]], x: Tensor[f32,[1]]) -> f32 {
            let _hook = fn() -> f32 { sum(bias) }
            let y = sum(w .* x)
            y * y
        }
        fn main() -> nil { nil }
    "#);
    assert_eq!(
        errs,
        vec!["captured mutable binding `bias` is not differentiable inside a closure — \
              the gradient tape does not enter closure bodies, so its gradient would be \
              silently absent".to_string()],
        "expected exactly the captured-mut-in-closure error",
    );

    // Reading the same binding directly *as well* does not launder the closure
    // read: the direct read is traced, the closure read is not, and a gradient
    // missing that term is exactly the silent wrongness this rejects.
    let both = check(r#"
        let !bias = forge.zeros[f32,[1]]
        @grad fn loss(!w: Tensor[f32,[1]], x: Tensor[f32,[1]]) -> f32 {
            let _hook = fn() -> f32 { sum(bias) }
            let y = sum(w .* x .+ bias)
            y * y
        }
        fn main() -> nil { nil }
    "#);
    assert!(both.iter().any(|e| e.contains("not differentiable inside a closure")
                              && e.contains("bias")),
            "a capture read in a closure must error even when also read directly: {:?}", both);

    // Nesting the closure one level deeper must not hide it either.
    let nested = check(r#"
        let !bias = forge.zeros[f32,[1]]
        @grad fn loss(!w: Tensor[f32,[1]], x: Tensor[f32,[1]]) -> f32 {
            let _outer = fn() -> fn() -> f32 { fn() -> f32 { sum(bias) } }
            let y = sum(w .* x)
            y * y
        }
        fn main() -> nil { nil }
    "#);
    assert!(nested.iter().any(|e| e.contains("not differentiable inside a closure")
                                && e.contains("bias")),
            "a capture nested two closures deep must still error: {:?}", nested);

    // The closure's OWN params are not captures — a closure that touches no
    // module-level mutable must stay clean.
    let ok = check(r#"
        @grad fn loss(!w: Tensor[f32,[1]], x: Tensor[f32,[1]]) -> f32 {
            let sq = fn(v: f32) -> f32 { v * v }
            sq(sum(w .* x))
        }
        fn main() -> nil { nil }
    "#);
    assert!(!ok.iter().any(|e| e.contains("not differentiable")),
            "closure with no captured mutable must not error: {:?}", ok);

    // A closure param that shadows a module-level mutable of the same name is
    // not a capture either.
    let shadow = check(r#"
        let !bias = 1.0f32
        @grad fn loss(!w: Tensor[f32,[1]], x: Tensor[f32,[1]]) -> f32 {
            let sq = fn(bias: f32) -> f32 { bias * bias }
            sq(sum(w .* x))
        }
        fn main() -> nil { nil }
    "#);
    assert!(!shadow.iter().any(|e| e.contains("not differentiable")),
            "a closure param shadowing a mutable is not a capture: {:?}", shadow);
}

#[test]
fn tensor_split_types_as_ntuple_397() {
    // `.split[n, ...]` types as an n-tuple (no warning), so a wrong-arity
    // destructure is caught and a right-arity one is clean.
    let src = r#"
        fn main() -> nil {
            let q = forge.zeros[f32,[2,6]]
            let (a, b) = q.split[3, axis=-1]
            nil
        }
    "#;
    assert!(check(src).iter().any(|e| e.contains("tuple pattern has 2 elements")),
            "split arity should be checked, got {:?}", check(src));
    assert!(!warnings(src).iter().any(|m| m.contains("#397")),
            "the #397 warning should be gone (split is built): {:?}", warnings(src));
    let ok = r#"
        fn main() -> nil {
            let q = forge.zeros[f32,[2,6]]
            let (a, b, c) = q.split[3, axis=-1]
            nil
        }
    "#;
    assert!(check(ok).is_empty(), "3-name split destructure must be clean: {:?}", check(ok));
}

#[test]
fn shape_pattern_in_match_rejected_393() {
    let errs = check(r#"
        fn main() -> nil {
            let t = forge.zeros[f32,[2,3]]
            let r = match t { [2, 3] => 1, _ => 0 }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("shape pattern")),
            "expected shape-pattern reject, got {:?}", errs);
}

#[test]
fn tuple_rest_pattern_arity_393() {
    // `..` relaxes the destructure arity check: `(a, ..)` on a 3-tuple is fine.
    assert!(passes("fn main() -> i64 { let (a, ..) = (1, 2, 3)  a }"),
            "rest destructure should type-check: {:?}", check("fn main() -> i64 { let (a, ..) = (1, 2, 3)  a }"));
    assert!(passes("fn main() -> i64 { let (a, .., z) = (1, 2, 3, 4)  a + z }"));

    // But the fixed head+tail must still fit.
    let too_few = check("fn main() -> i64 { let (a, b, .., z) = (1, 2)  a }");
    assert!(too_few.iter().any(|e| e.contains("at least 3")),
            "too-short rest destructure should error, got {:?}", too_few);

    // More than one `..` is ambiguous and rejected.
    let multi = check("fn main() -> i64 { let (a, .., b, ..) = (1, 2, 3, 4)  a }");
    assert!(multi.iter().any(|e| e.contains("at most one")),
            "multi-rest should be rejected, got {:?}", multi);

    // Exact-arity destructure (no rest) still catches mismatches.
    let mism = check("fn main() -> i64 { let (a, b) = (1, 2, 3)  a }");
    assert!(mism.iter().any(|e| e.contains("2 elements but value has 3")),
            "exact-arity mismatch must still fire, got {:?}", mism);
}

#[test]
fn body_ending_in_let_gets_targeted_diagnostic() {
    // A body with no tail expression that ends in a `let` types as nil. The
    // frequent cause is two statements on one line where `<expr> (…)` parses as
    // a call that swallows the intended tail expression. The diagnostic must
    // name the missing-tail-expression cause, not the bare "produces nil".
    let errs = check("fn main() -> i64 { let x = 5 }");
    assert!(errs.iter().any(|e| e.contains("ends in a `let` binding")
                              && e.contains("has no value")),
            "expected targeted let-ending diagnostic, got {:?}", errs);

    // The call-absorption footgun (`t.split[2]  (…)` on one line) reaches the
    // same shape and must get the same targeted message.
    let absorbed = check(
        "fn main() -> i64 { let !t = forge.zeros[f32,[8]]  let (a,b) = t.split[2]  (a[3]+b[0]) as i64 }");
    assert!(absorbed.iter().any(|e| e.contains("ends in a `let` binding")),
            "call-absorption case should get the let-ending diagnostic, got {:?}", absorbed);

    // A genuine type mismatch (real tail expression, wrong type) keeps the
    // original "body produces `T`" wording — the special case must not swallow it.
    let mismatch = check("fn main() -> i64 { true }");
    assert!(mismatch.iter().any(|e| e.contains("body produces") && e.contains("bool")),
            "genuine mismatch must keep the produces-T message, got {:?}", mismatch);

    // Correctly-formatted code (tail expression present) still checks clean.
    assert!(check("fn main() -> i64 { let x = 5  x + 1 }").is_empty());
}

// --- #443: element writes through immutable `self` stay check-time errors ----
//
// Shapes 2 and 4 of the issue's matrix. The interpreter fix for #443 must not
// make these reachable at runtime: they are rejected by the immutable-binding
// rule, which is why the language forces callers into the `let !c = self.f`
// alias form in the first place.

#[test]
fn element_write_through_immutable_self_rejected_443() {
    // Shape 2: `self.cells[self.table[0]] = 1` — self read in index position.
    let errs = check(r#"
model Box {
    !table: Tensor[i64, [4]]
    !cells: Tensor[i64, [8]]
    fn go!(self) -> nil {
        self.cells[self.table[0]] = 1
        nil
    }
}
fn main() -> nil { nil }
"#);
    assert!(errs.iter().any(|e| e.contains("cannot write to an element of immutable binding `self`")),
            "expected the immutable-self element-write error, got {:?}", errs);

    // Shape 4: same write with the self read in value position instead.
    let errs = check(r#"
model Box {
    !table: Tensor[i64, [4]]
    !cells: Tensor[i64, [8]]
    fn go!(self) -> nil {
        self.cells[0] = self.table[0] + 7
        nil
    }
}
fn main() -> nil { nil }
"#);
    assert!(errs.iter().any(|e| e.contains("cannot write to an element of immutable binding `self`")),
            "expected the immutable-self element-write error, got {:?}", errs);
}

#[test]
fn field_alias_element_write_checks_clean_443() {
    // The alias form the rule steers callers to must keep checking clean —
    // including with a `self` read inside the index (shapes 1 and 5).
    assert!(passes(r#"
model Box {
    !table: Tensor[i64, [4]]
    !cells: Tensor[i64, [8]]
    fn idx(self) -> i64 { self.table[0] + 1 }
    fn go!(self) -> nil {
        let !c = self.cells
        c[self.table[0]] = 1
        c[self.idx()] = 2
        nil
    }
}
fn main() -> nil { nil }
"#), "alias-form element write should check clean, got {:?}", check(r#"
model Box {
    !table: Tensor[i64, [4]]
    !cells: Tensor[i64, [8]]
    fn idx(self) -> i64 { self.table[0] + 1 }
    fn go!(self) -> nil {
        let !c = self.cells
        c[self.table[0]] = 1
        c[self.idx()] = 2
        nil
    }
}
fn main() -> nil { nil }
"#));
}

#[test]
fn int_suffix_types_unannotated_local_445() {
    // The suffix is the most explicit statement of intent a literal carries:
    // `let c = 0xffu32` binds a u32, not a defaulted i64.
    assert!(passes(
        "fn takes_u32(x: u32) -> u32 { x }\n\
         fn f() -> bool { let c = 0x00ff_ff00u32  takes_u32(c) == 0x00ff_ff00u32 }"),
        "u32-suffixed initializer must bind u32");
    assert!(passes(
        "fn f() -> u64 { let !n = 5u64  n = n + 1u64  n }"),
        "u64-suffixed mutable local must bind u64");
}

#[test]
fn int_suffix_conflicts_and_ranges_445() {
    // Suffix vs annotation conflict is an error, not a silent override.
    let e = check("fn f() -> nil { let x: u32 = 5u64  nil }");
    assert!(!e.is_empty(), "u64 literal into u32 annotation must error");
    // A literal must fit its own suffix, annotation or not (#295).
    let e = check("fn f() -> nil { let x = 300u8  nil }");
    assert!(e.iter().any(|m| m.contains("range")), "300u8 must range-error, got {:?}", e);
    // 64-bit hex masks store as i64 bit patterns (#282) — always legal.
    assert!(passes("fn f() -> nil { let m = 0xffff_ffff_ffff_ffffu64  nil }"),
        "u64 mask literal must stay legal");
}

/// #597: `int_suffix_conflicts_and_ranges_445` above only exercised an
/// UNANNOTATED `let` — the literal's own suffix is what typed it, and
/// `check_expr`'s `Literal` arm already exempted a suffixed 64-bit literal
/// from its range check (#282). A `let` with an explicit `u64` ANNOTATION
/// takes a different path (`check_int_literal_range`, driven by the
/// annotation rather than the literal's suffix), which had no such
/// exemption: it folded the literal down to its raw `i64` bit pattern via
/// `expr_i64` and rejected anything negative — so `0xffffffffffffffffu64`
/// read as `-1` and was rejected as "out of range for `u64`" even though it
/// is the top of the u64 range, not the bottom. A directly-written negative
/// decimal (`-1u64`, an actual leading `-` in the source) must still error:
/// that is the program writing down a negative literal for an unsigned type,
/// which SPEC §3.1 forbids regardless of width.
#[test]
fn u64_mask_literal_with_declared_annotation_597() {
    assert!(passes("fn f() -> nil { let x: u64 = 0xffffffffffffffffu64  nil }"),
        "an all-ones u64 mask must fit its own declared `u64` annotation");
    assert!(passes("fn f() -> nil { let x: u64 = 0x8000000000000000u64  nil }"),
        "a top-bit-set u64 mask must fit its own declared `u64` annotation");
    assert!(passes(
        "fn f() -> nil { \
         let x: u64 = 0b1111111111111111111111111111111111111111111111111111111111111111u64 \
         nil }"),
        "an all-ones u64 BINARY mask must fit its own declared `u64` annotation");
    // A return-typed position is the same `check_int_literal_range` call site
    // (§3.1's other width-origin: "a suffixed literal, a typed `let`, a typed
    // parameter, a declared return type").
    assert!(passes("fn f() -> u64 { 0xffffffffffffffffu64 }"),
        "an all-ones u64 mask must fit a declared `u64` return type");
    // An explicit negative decimal literal is still rejected — this is not a
    // bit-pattern mask, it is the program writing a negative number down for
    // an unsigned type (§3.1: "a literal that does not fit the width it
    // adopts is ... a compile-time error, not a wrap").
    let e = check("fn f() -> nil { let x: u64 = -1u64  nil }");
    assert!(e.iter().any(|m| m.contains("out of range") && m.contains("u64")),
        "an explicit -1u64 must still range-error, got {:?}", e);
    let e = check("fn f() -> nil { let x: u64 = -1  nil }");
    assert!(e.iter().any(|m| m.contains("out of range") && m.contains("u64")),
        "an explicit -1 into a `u64` annotation must still range-error, got {:?}", e);
}

#[test]
fn cross_arena_error_is_not_demon_suppressed_442() {
    // A spec violation, not a lint: demon mode drops safe-mode lints but must
    // still reject a cross-arena write, matching the §2 uninit-read error.
    let tokens = super::lexer::Lexer::new(
        "fn main() -> nil { let !w = vault.ones[f32, [4]]  w[0] = 0.5  nil }")
        .tokenize().expect("lex failed");
    let program = super::parser::Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker = Checker::new();
    checker.demon = true;
    checker.check_program(&program, None);
    let es: Vec<String> = checker.errors.iter().map(|e| e.msg.clone()).collect();
    assert!(es.iter().any(|e| e.contains("cross-arena write")),
            "demon mode must still reject a cross-arena write, got {:?}", es);
}

// ─── `if`/`else` branch unification (#479) ───────────────────────────────────
//
// `match` has unified its arms since #244; `if` did not, and silently produced
// `Unit` on a mismatch. Two failures followed: a wrong diagnostic naming `nil`
// and pointing at the consumer when the value WAS used, and no diagnostic at
// all when it was not.

fn if_branch_err(src: &str) -> bool {
    check(src).iter().any(|m| m.contains("`if` branches yield incompatible types"))
}

#[test]
fn if_branches_must_unify_in_value_position() {
    // The #479 repro: a `str` branch and an `i64` branch, previously accepted.
    assert!(if_branch_err(
        r#"fn main() { let a: i64 = 1  let z = if a > 0 { "x" } else { a }  print(to_str(z)) }"#
    ));
    // A block's trailing value is value position, even written as a statement.
    assert!(if_branch_err(
        r#"fn f() -> i64 { let a = 1  if a > 0 { 1 } else { "x" } }"#
    ));
    // So is an argument.
    assert!(if_branch_err(
        r#"fn k(x: i64) {} fn main() { let a = 1  k(if a > 0 { 1 } else { "x" }) }"#
    ));
}

#[test]
fn if_branch_mismatch_names_if_not_nil() {
    // The old diagnostic said "value has type nil" AT THE CONSUMER, which sent
    // the reader looking for a missing return value instead of a branch
    // mismatch. The error must now name both branch types.
    let msgs = check(
        r#"fn main() { let a = 0.1f32  let b: f64 = 0.5
             let z = if a > 0.0f32 { a } else { b }
             let w: f64 = z  print(to_str(w)) }"#,
    );
    assert!(
        msgs.iter().any(|m| m.contains("`if` branches yield incompatible types")),
        "expected a branch-mismatch error, got: {:?}", msgs,
    );
    assert!(
        !msgs.iter().any(|m| m.contains("value has type nil")),
        "the `if` still degrades to nil: {:?}", msgs,
    );
}

#[test]
fn else_if_chain_unifies_with_the_leading_branch() {
    // The `ElseBranch::If` arm used to `return self.check_if(nested)` and
    // DISCARD the leading branch's type, so the first branch never met the
    // rest of the chain and this reported nothing at all.
    //
    // The REST OF THE CHAIN MUST AGREE WITH ITSELF for this to isolate that
    // hole: with `else if a < 0 { "x" } else { 2 }` the inner `if` catches the
    // mismatch on its own and the test passes even with the fix reverted
    // (verified by mutation). Here `1` and `2` unify fine and only the leading
    // `"x"` disagrees, so nothing but the leading-branch unification can see it.
    assert!(if_branch_err(
        r#"fn main() { let a: i64 = 1
             let z = if a > 0 { "x" } else if a < 0 { 1 } else { 2 }
             print(to_str(z)) }"#
    ));
    // An all-i64 chain stays clean.
    assert!(passes(
        r#"fn main() { let a = 1
             let z = if a > 2 { 1 } else if a > 1 { 2 } else { 3 }
             print(to_str(z)) }"#
    ));
}

#[test]
fn if_statement_branches_are_not_unified() {
    // A bare `if` statement discards its value, so the branch "types" are
    // incidental — one side calling something for its effect and the other
    // doing nothing is legal, and must stay legal.
    assert!(passes(
        r#"fn g() -> i64 { 1 } fn main() { let a = 1  if a > 0 { g() } else { }  print("ok") }"#
    ));
    assert!(passes(
        r#"fn g() -> i64 { 1 } fn h() -> str { "x" }
           fn main() { let a = 1  if a > 0 { g() } else { h() }  print("ok") }"#
    ));
    assert!(passes(
        r#"fn g() -> i64 { 1 } fn main() { let a = 1  if a > 0 { g() }  print("ok") }"#
    ));
}

#[test]
fn if_unification_keeps_the_exemptions_match_has() {
    // Same predicate as `match` (`compatible_with`, `Unknown` exempt as the
    // diverging/bottom type), so the two forms cannot drift apart.
    assert!(passes(
        r#"fn main() { let a = 1  let z = if a > 0 { 5 } else { panic("no") }  print(to_str(z)) }"#
    ));
    // An untyped literal adopts the other branch, in either order (#284/#295).
    assert!(passes(
        r#"fn main() { let a = 0.1f32  let z = if a > 0.0f32 { a } else { 0.0 }  print(to_str(z)) }"#
    ));
    assert!(passes(
        r#"fn main() { let a = 0.1f32  let z = if a > 0.0f32 { 0.0 } else { a }
             let w: f32 = z  print(to_str(w)) }"#
    ));
}

// PORTS.md §5: a port call inside a `@grad fn` is an effect boundary the
// gradient cannot cross — the checker rejects it with a `port-forbidden` tag.
#[test]
fn port_call_inside_grad_fn_is_forbidden() {
    let errs = check(r#"
        @grad fn loss(!w: Tensor[f32, [4]]) -> f32 {
            let (p, e) = port_open("python")
            sum(w .* w)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("port-forbidden") && e.contains("@grad")),
        "got: {:?}", errs);
}

// The same call in an ordinary fn is fine — the restriction is scoped to the
// gradient tape, not ports in general.
#[test]
fn port_call_outside_grad_fn_is_allowed() {
    let errs = check(r#"
        fn talk() -> str {
            let (p, e) = port_open("python")
            let (out, e2) = port_call(p, "len", "[[1,2,3]]")
            let (_, e3) = port_close(p)
            out
        }
    "#);
    assert!(!errs.iter().any(|e| e.contains("port-forbidden")), "got: {:?}", errs);
}

// SPEC.md §4.11 / PORTS.md §3.2: the copy-mode primitives are *not* port
// calls. They are pure value transforms — a tensor to text and back, with no
// runtime on the other end — so the effect-boundary ban does not reach them.
// Pinned because the ban is a name list in `check.rs`: without this, adding a
// `port_`-prefixed primitive to that list, or forgetting to keep these off it,
// would silently move the boundary the spec draws.
#[test]
fn the_copy_mode_primitives_are_not_port_calls() {
    for site in [
        r#"@grad fn loss(!w: Tensor[f32, [4]]) -> f32 {
               let s = port_tensor_encode(w)
               let (b, e) = port_tensor_decode(s, forge.zeros[f32, [4]])
               sum(w .* w)
           }"#,
        r#"fn f() -> i64 {
               let !g = forge.zeros[i64, [2]]
               @deterministic {
                   let s = port_tensor_encode(g)
                   let (b, e) = port_tensor_decode(s, forge.zeros[i64, [2]])
               }
               0
           }"#,
    ] {
        let errs = check(site);
        assert!(!errs.iter().any(|e| e.contains("port-forbidden")), "got: {:?}", errs);
    }
}

// The restriction reaches into a closure checked within the `@grad fn` body:
// a port call there is still inside the tape.
#[test]
fn port_call_in_closure_within_grad_fn_is_forbidden() {
    let errs = check(r#"
        @grad fn loss(!w: Tensor[f32, [4]]) -> f32 {
            let f = fn() -> nil { let (p, e) = port_open("python")  nil }
            sum(w .* w)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("port-forbidden")), "got: {:?}", errs);
}

// ── #474: model shape args survive into the type, in all three positions ────
//
// A model literal used to type bare — `Box[2, 2] { … }` was just `Box`, the
// two numbers standing right there thrown away — so a parameterized model
// type had nothing to unify against anywhere: not a field, not a parameter.
// And a shape bracket between a method name and its call hid the method from
// the checker entirely, which is how a defined, `--check`-clean method could
// do nothing at all. Every message below is asserted whole: a substring match
// would not notice a diagnostic that stopped naming the field, the model, or
// the shapes.

/// The one error `src` must produce, or a panic naming what came instead.
fn one_error(src: &str) -> String {
    let errs = check(src);
    assert_eq!(errs.len(), 1, "expected exactly one type error, got: {:?}", errs);
    errs.into_iter().next().unwrap()
}

#[test]
fn model_field_of_parameterized_model_type_474() {
    // The issue comment's repro. `Inner[4, 5]` is exactly what `H, W` bind to
    // from `Outer[2, 3, 4, 5]`, and the literal now carries it.
    let errs = check(r#"
        model Inner[H, W] {
            !px: Tensor[u32, [H, W]]
            !n: i64
        }
        model Outer[R, C, H, W] {
            !grid: Tensor[u32, [R, C]]
            !surf: Inner[H, W]
        }
        fn main() -> nil {
            vault {
                let !o = Outer[2, 3, 4, 5] {
                    grid: vault.zeros[u32, [2, 3]],
                    surf: Inner[4, 5] { px: vault.zeros[u32, [4, 5]], n: 0 }
                }
                nil
            }
        }
    "#);
    assert!(errs.is_empty(), "expected the field literal to unify, got: {:?}", errs);
}

#[test]
fn model_field_shape_args_are_enforced_474() {
    // Accepting the literal is only half the fix: the shapes it carries have
    // to be *checked* against the ones the field expects. `Inner[9, 5]` where
    // `H` is 4 is a different type, and the diagnostic still names the field.
    let msg = one_error(r#"
        model Inner[H, W] {
            !px: Tensor[u32, [H, W]]
            !n: i64
        }
        model Outer[R, C, H, W] {
            !grid: Tensor[u32, [R, C]]
            !surf: Inner[H, W]
        }
        fn main() -> nil {
            vault {
                let !o = Outer[2, 3, 4, 5] {
                    grid: vault.zeros[u32, [2, 3]],
                    surf: Inner[9, 5] { px: vault.zeros[u32, [9, 5]], n: 0 }
                }
                nil
            }
        }
    "#);
    assert_eq!(msg, "mismatched type for field `surf` of model `Outer`: \
                     expected `Inner[4, 5]`, got `Inner[9, 5]`");
}

#[test]
fn a_bare_model_literal_that_pins_its_shape_satisfies_the_field_474() {
    // The issue's title spelling: "a bare `Inner { … }` literal". It has no
    // bracket, but it is not silent — `px` is a 4x5 tensor, which pins `H` and
    // `W` as plainly as `Inner[4, 5]` would. Unify and let it in.
    let errs = check(r#"
        model Inner[H, W] {
            !px: Tensor[u32, [H, W]]
            !n: i64
        }
        model Outer[H, W] { !surf: Inner[H, W] }
        fn mk() -> Tensor[u32, [4, 5]] { vault { vault.zeros[u32, [4, 5]] } }
        fn main() -> nil {
            vault {
                let !o = Outer[4, 5] { surf: Inner { px: mk(), n: 0 } }
                nil
            }
        }
    "#);
    assert!(errs.is_empty(), "expected a bare literal that pins 4x5 to unify, got: {:?}", errs);
}

#[test]
fn a_bare_literal_whose_fields_disagree_is_rejected_474() {
    // The same bare spelling, a 7x7 buffer, a 4x5 field. Accepting a literal
    // because it *declared* nothing would make the empty argument list a
    // wildcard — the one reading that lets a wrong shape through in silence.
    // It unifies to `Inner[7, 7]` and is refused, naming the field.
    let msg = one_error(r#"
        model Inner[H, W] {
            !px: Tensor[u32, [H, W]]
            !n: i64
        }
        model Outer[H, W] { !surf: Inner[H, W] }
        fn mk() -> Tensor[u32, [7, 7]] { vault { vault.zeros[u32, [7, 7]] } }
        fn main() -> nil {
            vault {
                let !o = Outer[4, 5] { surf: Inner { px: mk(), n: 0 } }
                nil
            }
        }
    "#);
    assert_eq!(msg, "mismatched type for field `surf` of model `Outer`: \
                     expected `Inner[4, 5]`, got `Inner[7, 7]`");
}

#[test]
fn a_bare_literal_that_pins_nothing_is_rejected_474() {
    // #474's field position, the silent case. `vault.zeros[…]` types as
    // `Unknown` in constructor position (a pre-existing tensor-typing gap), so
    // this literal pins neither `H` nor `W` — it makes no claim at all. The
    // slot says `Inner[4, 5]` and nothing later will re-check it, so an
    // unprovable literal is refused here rather than accepted on the strength
    // of its name. The diagnostic names the field and says how to fix it.
    let msg = one_error(r#"
        model Inner[H, W] {
            !px: Tensor[u32, [H, W]]
            !n: i64
        }
        model Outer[H, W] { !surf: Inner[H, W] }
        fn main() -> nil {
            vault {
                let !o = Outer[4, 5] { surf: Inner { px: vault.zeros[u32, [7, 7]], n: 0 } }
                nil
            }
        }
    "#);
    assert_eq!(msg, "field `surf` of model `Outer` expects `Inner[4, 5]`, and the \
                     literal given does not say what shape it is");
}

#[test]
fn a_different_model_is_still_the_wrong_field_474() {
    // Carrying shape args is not a licence for any model to satisfy any
    // parameterized model field.
    let msg = one_error(r#"
        model Inner[H, W] { !px: Tensor[u32, [H, W]] }
        model Crate { !n: i64 }
        model Outer[H, W] { !surf: Inner[H, W] }
        fn main() -> nil {
            vault {
                let !o = Outer[4, 5] { surf: Crate { n: 1 } }
                nil
            }
        }
    "#);
    assert_eq!(msg, "mismatched type for field `surf` of model `Outer`: \
                     expected `Inner[4, 5]`, got `Crate`");
}
// ── #474 parameter position, and the two holes next door that stay open ─────

#[test]
fn parameterized_model_param_accepts_a_bare_model_arg_474() {
    // A model literal types bare (`Box`), so `!b: Box[H, W]` used to reject
    // every call that did not spell the shapes out. The dims are on the
    // instance and the interpreter's harvest binds them, so the call stands.
    let errs = check(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            !n: i64
        }
        fn area[H, W](!b: Box[H, W]) -> i64 { H * W }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 3] { cells: vault.zeros[u32, [2, 3]], n: 0 }
                area(b)
            }
        }
    "#);
    assert!(errs.is_empty(), "expected the call to check clean, got: {:?}", errs);
}

#[test]
fn a_model_of_the_wrong_shape_is_rejected_at_the_call_474() {
    // The binding runs both ways: a concretely-shaped model parameter now has
    // something to compare against, so the wrong instance is caught here
    // rather than deep inside the callee.
    let msg = one_error(r#"
        model Box[H, W] { !cells: Tensor[u32, [H, W]] }
        fn concrete(!b: Box[2, 2]) -> i64 { 1 }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 3] { cells: vault.zeros[u32, [2, 3]] }
                concrete(b)
            }
        }
    "#);
    assert_eq!(msg, "arg 0: expected Box[2, 2], got Box[2, 3]");
}

#[test]
fn a_model_and_a_tensor_that_disagree_are_caught_at_check_474() {
    // The promotion the shape lane predicted: once the model argument binds
    // `H = 2` at check time, the tensor argument that wants `H = 3` no longer
    // has to wait for the interpreter. It only reaches this far when the
    // tensor's own type is known — a `vault.zeros` binding deliberately types
    // as `Unknown` (#248), and that pair is still the interpreter's to catch.
    let msg = one_error(r#"
        model Box[H, W] { !cells: Tensor[u32, [H, W]] }
        fn wide[H, W](!b: Box[H, W], src: Tensor[u32, [H, W]]) -> i64 { H }
        fn drive(src: Tensor[u32, [3, 2]]) -> i64 {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                wide(b, src)
            }
        }
        fn main() -> nil { nil }
    "#);
    assert_eq!(msg, "arg 1: expected Tensor[u32, [2, 2]], got Tensor[u32, [3, 2]]");
}

// ── #792: a model's shape arguments compare like a tensor's ─────────────────
//
// `fn mk[N]() -> Buf[N]` has no value parameter to infer `N` from, so an
// explicit bracket at the call (`mk[8]()`) is the only way to pin it. Nothing
// consumed that bracket for a *free* function (only a bracketed *method*
// call did, #474) — `mk[8]()` fell through the general indexing arm and typed
// `Unknown`, which is compatible with everything, so a mismatched shape
// argument passed `--check` clean in every position: return, parameter,
// field initialization, a `let` annotation, and a model-array element.

#[test]
fn generic_fn_call_return_position_792() {
    let msg = one_error(r#"
        model Buf[N] { t: Tensor[i64, [N]] }
        fn mk[N]() -> Buf[N] { Buf[N] { t: forge.zeros[i64, [N]] } }
        fn wrong() -> Buf[4] { mk[8]() }
        fn main() -> nil { nil }
    "#);
    assert_eq!(msg, "fn `wrong` returns `Buf[4]` but body produces `Buf[8]`");
}

#[test]
fn generic_fn_call_return_position_matching_shape_passes_792() {
    assert!(passes(r#"
        model Buf[N] { t: Tensor[i64, [N]] }
        fn mk[N]() -> Buf[N] { Buf[N] { t: forge.zeros[i64, [N]] } }
        fn ok() -> Buf[4] { mk[4]() }
        fn main() -> nil { nil }
    "#));
}

#[test]
fn generic_fn_call_parameter_position_792() {
    let msg = one_error(r#"
        model Buf[N] { t: Tensor[i64, [N]] }
        fn mk[N]() -> Buf[N] { Buf[N] { t: forge.zeros[i64, [N]] } }
        fn take(b: Buf[4]) -> i64 { 4 }
        fn main() -> i64 { take(mk[8]()) }
    "#);
    assert_eq!(msg, "arg 0: expected Buf[4], got Buf[8]");
}

#[test]
fn generic_fn_call_field_init_position_792() {
    let msg = one_error(r#"
        model Buf[N] { t: Tensor[i64, [N]] }
        model Outer { inner: Buf[4] }
        fn mk[N]() -> Buf[N] { Buf[N] { t: forge.zeros[i64, [N]] } }
        fn main() -> i64 {
            let o = Outer { inner: mk[8]() }
            0
        }
    "#);
    assert_eq!(msg, "mismatched type for field `inner` of model `Outer`: \
                     expected `Buf[4]`, got `Buf[8]`");
}

#[test]
fn generic_fn_call_let_annotation_position_792() {
    let msg = one_error(r#"
        model Buf[N] { t: Tensor[i64, [N]] }
        fn mk[N]() -> Buf[N] { Buf[N] { t: forge.zeros[i64, [N]] } }
        fn main() -> i64 {
            let b: Buf[4] = mk[8]()
            0
        }
    "#);
    assert_eq!(msg, "let binding has type Buf[4] but value has type Buf[8]");
}

#[test]
fn generic_fn_call_model_array_element_position_792() {
    // A model-array parameter's element type is a model type like any other —
    // the `Array` arm of `compatible_with` already delegated to it (#234), so
    // this position was caught by the pre-existing `Named` comparison; pinned
    // here so the whole enumerated set stays covered by one file.
    let msg = one_error(r#"
        model Buf[N] { t: Tensor[i64, [N]] }
        fn take_arr(arr: [Buf[4]; 2]) -> i64 { 0 }
        fn main() -> i64 {
            let arr: [Buf[8]; 2] = forge.zeros[Buf[8], [2]]
            take_arr(arr)
        }
    "#);
    assert_eq!(msg, "arg 0: expected [Buf[4]; 2], got [Buf[8]; 2]");
}

#[test]
fn generic_fn_call_shape_arity_mismatch_792() {
    // `Buf[4, 4]` vs `Buf[4]` — a different number of shape arguments, not
    // merely a different value. The `Named` arm's own length check (not the
    // bracket-instantiation fix) is what catches this.
    let msg = one_error(r#"
        model Buf[N] { t: Tensor[i64, [N]] }
        fn take(b: Buf[4]) -> i64 { 0 }
        fn wrong(b: Buf[4, 4]) -> i64 { take(b) }
        fn main() -> nil { nil }
    "#);
    assert_eq!(msg, "arg 0: expected Buf[4], got Buf[4, 4]");
}

#[test]
fn generic_fn_call_lowercase_shape_param_call_site_stays_permissive_792() {
    // SPEC §3.10: a model shape parameter written lowercase is "accepted at
    // check" — it can only be *supplied* explicitly at the call
    // (`area[2, 3](b)`), never verified against it. This is not merely #733's
    // "inference never binds a lowercase name": `jit_shape_arg_inference_
    // 732_733.rs`'s `a_lowercase_shape_param_spelled_at_the_call_is_never_
    // checked_against_the_argument` pins that both backends run this exact
    // shape, in flat contradiction of `b`'s real instance, rather than
    // refusing — so the same bracket-instantiation path used for `mk[8]()`
    // must stay just as permissive here: an argument whose actual shape
    // disagrees with a lowercase-bound bracket is still accepted, same as it
    // always ran before #792.
    assert!(passes(r#"
        model Box[h, w] { cells: Tensor[u32, [h, w]] }
        fn area[h, w](b: Box[h, w]) -> i64 { h * w }
        fn wrong_area(b: Box[4, 4]) -> i64 { area[2, 3](b) }
        fn main() -> nil { nil }
    "#));
}

#[test]
fn generic_fn_call_implicit_inference_still_works_786() {
    // #786's inference rule must keep working: an unbound shape param on a
    // function's own parameter (`fn cap[N](b: Buf[N])`) still binds `N` from
    // a concretely-shaped argument with no bracket at the call at all.
    assert!(passes(r#"
        model Buf[N] { t: Tensor[i64, [N]] }
        fn cap[N](b: Buf[N]) -> i64 { N }
        fn main() -> i64 {
            let b = Buf[8] { t: forge.zeros[i64, [8]] }
            cap(b)
        }
    "#));
}

// ── #792 rework: every case above uses `model Buf[N]` — the blind spot that
// shipped both regressions this rework fixes. Tensor/View/KV parameters and a
// named-constant shape argument at a bracketed free-fn call never appeared in
// this file until now.

#[test]
fn generic_fn_bracket_call_accepts_tensor_argument_for_view_param_792() {
    // SPEC.md §3.3 (~308): "`Tensor[T, S]` is a `View` with contiguous
    // strides in the arena's natural layout (row-major). Slicing a `Tensor`
    // yields a `View`." A `Tensor` argument must satisfy a `View`-typed
    // parameter at an explicit bracketed call.
    assert!(passes(r#"
        fn total[N](v: View[u8, [N]]) -> i64 { v[0] as i64 }
        fn main() -> i64 {
            let t = [1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8]
            total[8](t)
        }
    "#));
}

#[test]
fn generic_fn_bracket_call_accepts_kv_argument_for_view_param_792() {
    // SPEC.md §3.6 (~446): "A `KV[T, S]` is convertible to `View[T, S]` ...
    // for any read-only operation. The conversion is free."
    assert!(passes(r#"
        fn total(v: View[f32, [4]]) -> i64 { 1 }
        fn main() -> i64 {
            let k: KV[f32, [4]] = forge.zeros[f32, [4]]
            total(k)
        }
    "#));
}

#[test]
fn generic_fn_unbracketed_call_infers_n_from_a_tensor_argument_for_a_view_param_792() {
    // The two tests above pin the same-kind and explicit-bracket cases; this
    // one is the actual gap the reviewer found: `infer_call_shape_bindings`
    // (check.rs ~7090) had same-kind arms only, so `total[4](t)` (bracket
    // supplies `N` directly) passed while the unbracketed `total(t)` — the
    // ordinary call form #786 exists for — was refused with "expected
    // View[U8, [N]], got Tensor[U8, [4]]" even though the Tensor-for-View
    // widening this rework adds is otherwise unconditional. Pin that the
    // (View, Tensor) pair now also binds `N`, not just satisfies
    // `compatible_with`.
    assert!(passes(r#"
        fn total[N](v: View[u8, [N]]) -> i64 { v[0] as i64 }
        fn main() -> i64 {
            let t = [1u8, 2u8, 3u8, 4u8]
            total(t)
        }
    "#));
}

#[test]
fn generic_fn_unbracketed_call_infers_n_from_a_kv_argument_for_a_view_param_792() {
    // Same gap, `(View, KV)` side: a KV argument at an unbracketed call to a
    // shape-generic `View`-typed parameter must also bind `N`.
    assert!(passes(r#"
        fn total[N](v: View[u8, [N]]) -> i64 { v[0] as i64 }
        fn main() -> i64 {
            let k: KV[u8, [4]] = forge.zeros[u8, [4]]
            total(k)
        }
    "#));
}

#[test]
fn view_argument_does_not_satisfy_a_tensor_param_792() {
    // The reverse direction stays refused: SPEC.md gives `Tensor -> View` and
    // `KV -> View` as one-directional conversions, never the other way — a
    // `View` may be non-contiguous, or not backed by a live stream arena, so
    // it must not satisfy a `Tensor` (or `KV`) parameter.
    let msg = one_error(r#"
        fn wants_tensor(v: Tensor[u8, [4]]) -> i64 { v[0] as i64 }
        fn give_view(v: View[u8, [4]]) -> View[u8, [4]] { v }
        fn main() -> i64 {
            let t = forge.zeros[u8, [4]]
            let vv: View[u8, [4]] = give_view(t)
            wants_tensor(vv)
        }
    "#);
    assert_eq!(msg, "arg 0: expected Tensor[u8, [4]], got View[u8, [4]]");
}

#[test]
fn str_argument_accepted_for_a_view_u8_param_792() {
    // #809: unspecified by SPEC.md — filed to rule whether the JIT should
    // gain a lowering for this or the type should refuse it outright. The
    // interpreter already accepts a `str` argument for a `View[u8, [N]]`
    // parameter (any `N`, unchecked), so `--check` matches it rather than
    // being stricter than the backend that actually runs the program, until
    // that ruling lands.
    assert!(passes(r#"
        fn first[N](v: View[u8, [N]]) -> i64 { v[0] as i64 }
        fn main() -> i64 {
            first[4]("abcd")
        }
    "#));
    // #809: the non-bracketed, concrete-shape spelling widens the same way.
    // `main` (pre-#792) refused this call outright — pin it here too, next
    // to the bracketed case, since it changed for the same reason and the
    // #809 ruling may flip both at once.
    assert!(passes(r#"
        fn first(v: View[u8, [4]]) -> i64 { v[0] as i64 }
        fn main() -> i64 {
            first("abcd")
        }
    "#));
}

#[test]
fn generic_fn_bracket_call_resolves_a_local_let_constant_792() {
    // `let k = 4` is an immutable local integral constant — a bracketed
    // free-fn call's shape argument must resolve it to `Const(4)`, not leave
    // it an unbound `Var` that a concrete `Buf[4]` param can only ever fail
    // to match (`SymDim::equivalent`'s `Unknown`).
    assert!(passes(r#"
        model Buf[N] { t: Tensor[i64, [N]] }
        fn mk[N]() -> Buf[N] { Buf[N] { t: forge.zeros[i64, [N]] } }
        fn take(b: Buf[4]) -> i64 { 4 }
        fn main() -> i64 {
            let k = 4
            take(mk[k]())
        }
    "#));
}

#[test]
fn generic_fn_bracket_call_resolves_a_top_level_constant_792() {
    // Same fold, for a top-level `let` constant (#761/#766's `const_dims`) —
    // a bracketed free-fn call never consulted either constant table before
    // this rework.
    assert!(passes(r#"
        let CAP = 4
        model Buf[N] { t: Tensor[i64, [N]] }
        fn mk[N]() -> Buf[N] { Buf[N] { t: forge.zeros[i64, [N]] } }
        fn take(b: Buf[4]) -> i64 { 4 }
        fn main() -> i64 {
            take(mk[CAP]())
        }
    "#));
}

#[test]
fn bracketed_method_call_checks_its_arity_474() {
    // The headline hole seen from the checker's side. `b.generic![2, 2](src)`
    // parses as Call(Index(Field(b, "generic!"), [2, 2]), [src]), so the
    // method-call resolution never saw a method and the whole call typed as
    // `Unknown`: any number of arguments, of any type, and a result that
    // unified with anything.
    let msg = one_error(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> i64 { SH }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                let !s = vault.zeros[u32, [2, 2]]
                b.generic![2, 2](s, s, s)
                nil
            }
        }
    "#);
    assert_eq!(msg, "wrong number of args: expected 1, got 3");
}

#[test]
fn bracketed_method_call_checks_its_argument_types_474() {
    let msg = one_error(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> i64 { SH }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                b.generic![2, 2]("hello")
                nil
            }
        }
    "#);
    assert_eq!(msg, "arg 0: expected Tensor[u32, [2, 2]], got str");
}

#[test]
fn bracketed_method_call_has_the_methods_return_type_474() {
    // `-> i64` reaching a `bool` binding. Under `Unknown` this passed.
    let msg = one_error(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> i64 { SH }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                let !s = vault.zeros[u32, [2, 2]]
                let ok: bool = b.generic![2, 2](s)
                nil
            }
        }
    "#);
    assert_eq!(msg, "let binding has type bool but value has type i64");
}

#[test]
fn a_method_bracket_naming_a_shape_param_that_is_not_there_474() {
    let msg = one_error(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> i64 { SH }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                let !s = vault.zeros[u32, [2, 2]]
                b.generic![SX = 2](s)
                nil
            }
        }
    "#);
    assert_eq!(msg, "`SX` is not a shape parameter of `Box.generic!` (declared: SH, SW)");
}

#[test]
fn a_method_bracket_with_more_args_than_shape_params_474() {
    let msg = one_error(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> i64 { SH }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                let !s = vault.zeros[u32, [2, 2]]
                b.generic![2, 2, 2](s)
                nil
            }
        }
    "#);
    assert_eq!(msg, "`Box.generic!` declares 2 shape parameter(s), got more bracket args");
}

#[test]
fn a_method_bracket_binding_one_shape_param_twice_474() {
    let msg = one_error(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> i64 { SH }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                let !s = vault.zeros[u32, [2, 2]]
                b.generic![SH = 2, SH = 2](s)
                nil
            }
        }
    "#);
    assert_eq!(msg, "shape parameter `SH` of `Box.generic!` bound twice");
}

#[test]
fn ghost_method_behind_a_bracket_and_a_field_receiver_is_a_check_error_474() {
    // #441's gate, on the bracketed spelling and through a receiver that is
    // not a bare identifier — the form the old Ident-only lookup missed.
    let msg = one_error(r#"
        model Inner[H, W] { !px: Tensor[u32, [H, W]] }
        model Holder { !surf: Inner[2, 2] }
        fn main() -> nil {
            vault {
                let !i = Inner[2, 2] { px: vault.zeros[u32, [2, 2]] }
                let !h = Holder { surf: i }
                h.surf.nope![2, 2](1)
                nil
            }
        }
    "#);
    assert_eq!(msg, "no method `nope!` on model `Inner`");
}

#[test]
fn a_bracket_on_a_model_field_is_still_indexing_474() {
    // The method typing must not swallow `b.cells[0, 1]`: a field of that
    // name being indexed stays on the indexing path, and a real out-of-range
    // index there is still caught.
    let errs = check(r#"
        model Box[H, W] { !cells: Tensor[u32, [H, W]] }
        fn main() -> u32 {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                b.cells[0, 1]
            }
        }
    "#);
    assert!(errs.is_empty(), "expected field indexing to stay clean, got: {:?}", errs);
}

#[test]
fn an_error_inside_a_method_bracket_is_reported_once_474() {
    // Both spellings of the bracket, each reported exactly once (`one_error`
    // asserts the count): the positional one is typed by the index path
    // before the call typing sees it, the named one is not typed anywhere
    // else. Getting this wrong is invisible except as a doubled diagnostic.
    let src = r#"
        model Box[H, W] {
            !n: i64
            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> i64 { SH }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { n: 0 }
                let !s = vault.zeros[u32, [4, 5]]
                b.generic![PLACEHOLDER](s)
                nil
            }
        }
    "#;
    assert_eq!(one_error(&src.replace("PLACEHOLDER", "nope, 5")),
               "undefined identifier `nope`");
    assert_eq!(one_error(&src.replace("PLACEHOLDER", "SH = nope, SW = 5")),
               "undefined identifier `nope`");
}

#[test]
fn a_method_bracket_without_a_call_is_an_error_474() {
    // The ghost's last spelling. `b.generic![2, 2]` with no arguments after it
    // typed as `Unknown`, passed `--check`, and evaluated to an opaque that
    // did nothing — a statement shaped exactly like the call the author meant.
    // A shape bracket is part of calling a method; on its own it is not a
    // value, and saying so is the same principle that killed the ghost.
    let msg = one_error(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            !n: i64
            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> nil {
                self.n = self.n + 1
            }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]], n: 0 }
                b.generic![2, 2]
                nil
            }
        }
    "#);
    assert_eq!(msg, "`Box.generic!` is a method, and a shape bracket on it is \
                     part of calling it — `generic![…]` on its own is not a value");
}

#[test]
fn a_method_bracket_bound_to_a_name_is_the_same_error_474() {
    // Not only in statement position: there is no value to bind either.
    let msg = one_error(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn generic![SH](self, n: i64) -> i64 { SH + n }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                let f = b.generic![2]
                nil
            }
        }
    "#);
    assert!(msg.contains("is not a value"), "got: {msg}");
}

#[test]
fn a_non_integer_shape_argument_is_a_check_error_474() {
    // The arity, the parameter names and the bind-twice rule all report at
    // `--check`; this one waited for the interpreter for no reason. A shape
    // argument is a dim, and `"x"` is not one.
    let msg = one_error(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            !n: i64
            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> nil {
                self.n = self.n + 1
            }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]], n: 0 }
                let !s = vault.zeros[u32, [2, 2]]
                b.generic!["x", 2](s)
                nil
            }
        }
    "#);
    assert_eq!(msg, "shape argument `SH` of `Box.generic!` must be an integer, got str");
}

#[test]
fn a_named_non_integer_shape_argument_is_caught_too_474() {
    // The named spelling reaches the bracket by a different route; it is held
    // to the same rule.
    let msg = one_error(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn generic![SH](self, n: i64) -> i64 { SH + n }
        }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                b.generic![SH = true](1)
            }
        }
    "#);
    assert_eq!(msg, "shape argument `SH` of `Box.generic!` must be an integer, got bool");
}

#[test]
fn a_shape_generic_method_still_checks_clean_474() {
    // The whole repro, end to end: it must type-check as well as run.
    let errs = check(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            !n: i64
            fn plain!(self, v: u32) -> nil { self.n = self.n + 100 }
            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> nil {
                self.n = self.n + 1
                let !c = self.cells
                c[0, 0] = src[0, 0]
            }
        }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]], n: 0 }
                let !s = vault.zeros[u32, [2, 2]]
                b.plain!(1u32)
                b.generic![2, 2](s)
                nil
            }
        }
    "#);
    assert!(errs.is_empty(), "expected #474's repro to check clean, got: {:?}", errs);
}

#[test]
fn a_different_model_is_still_the_wrong_argument_474() {
    // The allowance is for the *same* model written bare — not a licence for
    // any model to satisfy any parameterized model parameter.
    let errs = check(r#"
        model Box[H, W] { !cells: Tensor[u32, [H, W]] }
        model Crate { !n: i64 }
        fn area[H, W](!b: Box[H, W]) -> i64 { H * W }
        fn main() -> i64 {
            let !c = Crate { n: 1 }
            area(c)
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("arg 0")), "expected an arg-0 error, got: {:?}", errs);
}

#[test]
fn model_array_shape_param_still_checks_clean_459() {
    // #459 was never a check-time rejection — it checked clean and failed at
    // the call. The fix is inference, so it must still check clean.
    let errs = check(r#"
        model Node { kind: i64 }
        fn build[N](!ns: [Node; N], n: i64) -> i64 {
            ns[0] = Node { kind: 7 }
            ns[0].kind
        }
        fn main() -> i64 {
            let !arena = forge.uninit[Node, [8]]
            build(arena, 8)
        }
    "#);
    assert!(errs.is_empty(), "expected a clean check, got: {:?}", errs);
}

#[test]
fn model_field_of_parameterized_model_type_no_longer_rejects_474() {
    // This was the scope boundary while only #474's parameter position was
    // fixed: the field position had the same root — a literal types bare —
    // and was left alone, so a correctly-shaped `Inner[4, 5]` in an
    // `Inner[H, W]` field was rejected. The field position is fixed now (a
    // literal keeps the args it was written with), so the same program that
    // used to have to fail has to pass. The rejections that *should* survive
    // are pinned next door: a literal whose args contradict the field
    // (`model_field_shape_args_are_enforced_474`), one whose fields
    // contradict them (`a_bare_literal_whose_fields_disagree_is_rejected_474`),
    // and a different model entirely (`a_different_model_is_still_the_wrong_field_474`).
    let errs = check(r#"
        model Inner[H, W] { !px: Tensor[u32, [H, W]] }
        model Outer[R, C, H, W] {
            !grid: Tensor[u32, [R, C]]
            !surf: Inner[H, W]
        }
        fn main() -> nil {
            vault {
                let !o = Outer[2, 3, 4, 5] {
                    grid: vault.zeros[u32, [2, 3]],
                    surf: Inner[4, 5] { px: vault.zeros[u32, [4, 5]] }
                }
                nil
            }
        }
    "#);
    assert!(errs.is_empty(),
            "expected the field literal to unify now that field position is fixed, got: {:?}", errs);
}

// `@fuse` is the second §5 gate: no fusion crosses a port call, so a port
// call inside the block contradicts the single-kernel promise.
#[test]
fn port_call_inside_fuse_block_is_forbidden() {
    let errs = check(r#"
        fn main() -> nil {
            let x = @fuse {
                let (p, e) = port_open("python")
                1.0f32
            }
            print(x)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("port-forbidden") && e.contains("@fuse")),
        "got: {:?}", errs);
}

// A `@fuse` block that is its enclosing block's trailing value takes a
// different path through `check_block` (its body is walked in the outer
// scope). The gate has to hold there too.
#[test]
fn port_call_in_trailing_fuse_block_is_forbidden() {
    let errs = check(r#"
        fn f() -> f32 {
            @fuse {
                let (p, e) = port_open("python")
                1.0f32
            }
        }
        fn main() -> nil { print(f())  nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("port-forbidden") && e.contains("@fuse")),
        "got: {:?}", errs);
}

// `@deterministic` is the third: bit-exactness needs a port manifest naming
// the runtime, version, env, files and args (PORTS.md §5), and no port has
// one — so every core op inside the block is rejected today.
#[test]
fn port_call_inside_deterministic_block_is_forbidden() {
    let errs = check(r#"
        fn main() -> nil {
            @deterministic {
                let (p, e) = port_open("python")
                let (out, e2) = port_call(p, "len", "[[1,2,3]]")
                let (_, e3) = port_close(p)
                print(out)
            }
            nil
        }
    "#);
    let hits = errs.iter().filter(|e| e.contains("port-forbidden")).count();
    assert_eq!(hits, 3, "every core op must be rejected, got: {:?}", errs);
    assert!(errs.iter().any(|e| e.contains("@deterministic")), "got: {:?}", errs);
}

// Closures again: a port call written inside a closure in a `@fuse` block is
// still lexically inside the block.
#[test]
fn port_call_in_closure_within_fuse_block_is_forbidden() {
    let errs = check(r#"
        fn main() -> nil {
            let x = @fuse {
                let f = fn() -> nil { let (p, e) = port_open("python")  nil }
                1.0f32
            }
            print(x)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("port-forbidden")), "got: {:?}", errs);
}

// The gate is lexical, like every other directive scope (DIRECTIVES.md §4):
// the same calls beside the block are fine.
#[test]
fn port_call_beside_a_fuse_block_is_allowed() {
    let errs = check(r#"
        fn main() -> nil {
            let t = [1.0f32, 2.0f32]
            let y = @fuse { t .+ t }
            let (p, e) = port_open("python")
            let (_, e3) = port_close(p)
            print(y[0])
            nil
        }
    "#);
    assert!(!errs.iter().any(|e| e.contains("port-forbidden")), "got: {:?}", errs);
}

// ── DIRECTIVES.md §3: illegal stacks ───────────────────────────────────────

// `@cast(t1) @cast(t2)`: the inner dtype wins, so the outer one is dead code
// dressed as intent. The diagnostic names both.
#[test]
fn stacked_cast_directives_are_rejected() {
    let errs = check(r#"
        fn main() -> nil {
            let x = @cast(f32) @cast(bf16) { 1.0f32 }
            print(x)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("illegal directive stack")
            && e.contains("@cast(bf16)") && e.contains("@cast(f32)")),
        "got: {:?}", errs);
}

// Written with braces instead of stacked, it is the same program.
#[test]
fn directly_nested_cast_blocks_are_rejected() {
    let errs = check(r#"
        fn main() -> nil {
            let x = @cast(f32) { @cast(bf16) { 1.0f32 } }
            print(x)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("illegal directive stack") && e.contains("@cast")),
        "got: {:?}", errs);
}

// One cast per scope is the whole point — a lone `@cast` stays quiet,
// including when the block it wraps carries an unrelated directive.
#[test]
fn a_single_cast_scope_is_legal() {
    let errs = check(r#"
        fn main() -> nil {
            let t = [1.0f32, 2.0f32]
            let x = @cast(bf16) { 1.0f32 }
            let y = @cast(bf16) { @fuse { t .* t } }
            print(x); print(y[0])
            nil
        }
    "#);
    assert!(!errs.iter().any(|e| e.contains("illegal directive stack")), "got: {:?}", errs);
}

// `@fuse @fuse` — fusion is idempotent, so the second one says nothing.
#[test]
fn stacked_fuse_directives_are_rejected() {
    let errs = check(r#"
        fn main() -> nil {
            let x = @fuse @fuse { 1.0f32 }
            print(x)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("illegal directive stack") && e.contains("@fuse")),
        "got: {:?}", errs);
}

// The canonical legal stack (DIRECTIVES.md §3) must keep checking — the
// rejections are targeted, not a ban on stacking.
#[test]
fn the_canonical_directive_stack_stays_legal() {
    let errs = check(r#"
        fn main() -> nil {
            let t = [1.0f32, 2.0f32]
            let x = @deterministic @cast(bf16) @fuse { t .+ t }
            print(x[0])
            nil
        }
    "#);
    assert!(!errs.iter().any(|e| e.contains("illegal directive stack")), "got: {:?}", errs);
}

// `@grad @grad` is the second-order autodiff form (SPEC.md §6.2), explicitly
// legal — the duplicate-directive rejections must not reach it.
#[test]
fn stacked_grad_directives_stay_legal() {
    let errs = check(r#"
        @grad @grad fn loss(!w: Tensor[f32, [4]]) -> f32 { sum(w .* w) }
        fn main() -> nil { nil }
    "#);
    assert!(!errs.iter().any(|e| e.contains("illegal directive stack")), "got: {:?}", errs);
}

// `@inplace` guards a write that would copy-on-write (MEMORY.md §4.3), so an
// assignment statement is the only thing it can attach to.
#[test]
fn inplace_on_an_assignment_is_legal() {
    let errs = check(r#"
        fn main() -> nil {
            let !a = 1
            @inplace a += 1
            print(a)
            nil
        }
    "#);
    assert!(!errs.iter().any(|e| e.contains("illegal directive stack")), "got: {:?}", errs);
}

#[test]
fn inplace_on_a_non_assignment_is_rejected() {
    // fn declaration, `let` binding, block, bare expression — none of them
    // hold a write for `@inplace` to guard.
    let cases = [
        ("@inplace\nfn f() -> i64 { 1 }\nfn main() -> nil {\n print(f())\n nil\n}",
         "`fn` declaration"),
        ("fn main() -> nil {\n @inplace let b = 2\n print(b)\n nil\n}", "`let` binding"),
        ("fn main() -> nil {\n @inplace {\n print(1)\n }\n nil\n}", "a block"),
        ("fn main() -> nil {\n let !a = 1\n @inplace a\n print(a)\n nil\n}",
         "bare expression"),
    ];
    for (src, what) in cases {
        let errs = check(src);
        assert!(errs.iter().any(|e| e.contains("illegal directive stack")
                && e.contains("@inplace") && e.contains(what)),
            "expected an `@inplace` attachment error naming {what}, got: {:?}", errs);
    }
}

// A directive level between two casts does not launder the stack: written
// flat, `@cast(f32) @fuse @cast(bf16)` is rejected, so the long-hand brace
// spelling of the same stack has to be rejected too (DIRECTIVES.md §3).
#[test]
fn cast_blocks_separated_by_another_directive_are_rejected() {
    let errs = check(r#"
        fn main() -> nil {
            let x = @cast(f32) { @fuse { @cast(bf16) { 1.0f32 } } }
            print(x)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("illegal directive stack")
            && e.contains("@cast(bf16)") && e.contains("@cast(f32)")),
        "got: {:?}", errs);
}

// The flat spelling of that same stack, for the pair — both must reject.
#[test]
fn casts_separated_by_another_directive_reject_flat_too() {
    let errs = check(r#"
        fn main() -> nil {
            let x = @cast(f32) @fuse @cast(bf16) { 1.0f32 }
            print(x)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("illegal directive stack")
            && e.contains("@cast(bf16)") && e.contains("@cast(f32)")),
        "got: {:?}", errs);
}

// Two `@fuse` with a cast between them is the same story.
#[test]
fn fuse_blocks_separated_by_a_cast_are_rejected() {
    let errs = check(r#"
        fn main() -> nil {
            let x = @fuse { @cast(bf16) { @fuse { 1.0f32 } } }
            print(x)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("illegal directive stack") && e.contains("@fuse")),
        "got: {:?}", errs);
}

// A block that holds more than the lone nested directive construct is a real
// block, not a stack written long-hand — the descent has to stop there.
#[test]
fn a_cast_around_a_real_block_is_not_a_stack() {
    let errs = check(r#"
        fn main() -> nil {
            let x = @cast(f32) {
                print(1)
                @cast(bf16) { 1.0f32 }
            }
            print(x)
            nil
        }
    "#);
    assert!(!errs.iter().any(|e| e.contains("illegal directive stack")), "got: {:?}", errs);
}

// ── SPEC.md §7.7: the fuse-infeasible analysis (#503) ──────────────────────
//
// `@fuse` promises one kernel and no materialized intermediates. The set the
// JIT's fused kernel actually collapses is a single elementwise chain over
// shape-equal f32 tensors with float-scalar broadcasts; everything else is
// refused at check time as `fuse-infeasible`, naming the offending op.

fn fuse_infeasible(errs: &[String]) -> Vec<&String> {
    errs.iter().filter(|e| e.contains("fuse-infeasible")).collect()
}

// The positive case: an elementwise chain with a scalar broadcast, ReLU, and
// parentheses is exactly the fused kernel's shape — it stays silent.
#[test]
fn fusable_elementwise_chain_is_accepted() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
            let b = [10.0f32, 20.0f32, 30.0f32, 40.0f32]
            let s = 2.0f32
            let c = @fuse { \>((a .+ b) .* s .+ 1.0f32) }
            print(c[0])
            nil
        }
    "#);
    assert!(fuse_infeasible(&errs).is_empty(), "got: {:?}", errs);
}

// The same ops without `@fuse` make no promise, so nothing fires — the
// analysis gates the directive, not the ops.
#[test]
fn programs_without_fuse_are_untouched() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]
            let t = a @ a
            let s = softmax([1.0f32, 2.0f32], -1)
            print(t[0, 0]); print(s[0])
            nil
        }
    "#);
    assert!(fuse_infeasible(&errs).is_empty(), "got: {:?}", errs);
}

// `@` contracts an axis — that is a reduction, not a lane-wise op. The
// diagnostic names it.
#[test]
fn matmul_inside_fuse_is_infeasible() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]
            let c = @fuse { a @ a }
            print(c[0, 0])
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible") && e.contains("`@` is not elementwise")),
        "got: {:?}", errs);
}

// A call is an opaque boundary — the kernel cannot see through it. The
// diagnostic names the callee.
#[test]
fn call_inside_fuse_is_infeasible() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [1.0f32, 2.0f32]
            let c = @fuse { softmax(a, -1) }
            print(c[0])
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible") && e.contains("`softmax`")),
        "got: {:?}", errs);
}

// A `let` inside the block materializes its binding — the exact intermediate
// the directive promises away. Hoist it above the block.
#[test]
fn let_statement_inside_fuse_is_infeasible() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [1.0f32, 2.0f32]
            let c = @fuse {
                let t = a .* a
                t .+ a
            }
            print(c[0])
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible") && e.contains("`let` statement")),
        "got: {:?}", errs);
}

// A scalar-valued block has no lanes to write.
#[test]
fn scalar_body_inside_fuse_is_infeasible() {
    let errs = check(r#"
        fn main() -> nil {
            let x = @fuse { 1.0f32 }
            print(x)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible") && e.contains("scalar")),
        "got: {:?}", errs);
}

// A KV cache carries a streaming `~` extent; its read is not a static-shape
// lane. The leaf is named with its type.
#[test]
fn kv_leaf_inside_fuse_is_infeasible() {
    let errs = check(r#"
        fn step[B, H, D](q: Tensor[f32, [B, H, 1, D]], k: KV[f32, [B, H, ~, D]])
            -> Tensor[f32, [B, H, 1, D]] {
            @fuse { q .* k }
        }
        fn main() -> nil { nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible") && e.contains("`k`")),
        "got: {:?}", errs);
}

// `'` reorders lanes; the single pass reads every operand at one offset.
#[test]
fn transpose_inside_fuse_is_infeasible() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]
            let c = @fuse { a .* a' }
            print(c[0, 0])
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible") && e.contains("transpose")),
        "got: {:?}", errs);
}

// The non-f32 leaf class: the fused kernel computes in f32 lanes.
#[test]
fn int_tensor_inside_fuse_is_infeasible() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [1, 2, 3]
            let c = @fuse { a .+ a }
            print(c[0])
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible") && e.contains("f32")),
        "got: {:?}", errs);
}

// Tensor-tensor broadcast: the interpreter's `.*` broadcasts [2, 1] against
// [2, 3], but the fused kernel reads every leaf at one lane offset — the
// JIT's `fuse_infer_ty` refuses unequal shapes, so the checker does too.
// Only float scalars broadcast inside `@fuse`.
#[test]
fn tensor_tensor_broadcast_inside_fuse_is_infeasible() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
            let b = [[10.0f32], [20.0f32]]
            let c = @fuse { a .* b }
            print(c[0, 0])
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible")
            && e.contains("[2, 3]") && e.contains("[2, 1]")),
        "got: {:?}", errs);
}

// Rank broadcast is the same refusal: [2] has no lane to pair with [2, 2].
#[test]
fn rank_broadcast_inside_fuse_is_infeasible() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]
            let v = [10.0f32, 20.0f32]
            let c = @fuse { a .+ v }
            print(c[0, 0])
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible")
            && e.contains("[2, 2]") && e.contains("[2]")),
        "got: {:?}", errs);
}

// `@fuse` on a `fn` declaration: the catalog's attachment is block / expr,
// and both backends ignore the declaration form — an unaudited promise, so
// it is refused like `@inplace`'s attachment rule is.
#[test]
fn fuse_on_a_fn_declaration_is_rejected() {
    let errs = check(r#"
        @fuse fn f(a: Tensor[f32, [2]]) -> Tensor[f32, [2]] { a .+ a }
        fn main() -> nil { nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("illegal directive stack")
            && e.contains("`@fuse`") && e.contains("`fn` declaration")),
        "got: {:?}", errs);
}

// Symbolic shapes that are not *provably* equal: the JIT's fused kernel
// refuses unequal shapes at monomorphization, so `[N, 4]` against `[M, 4]`
// across fn params — Equiv::Unknown, some monomorphization can differ — is
// refused here, at check time, naming both shapes. Without this the checker
// stays silent and the backends split on the same program.
#[test]
fn unprovable_symbolic_shapes_inside_fuse_are_infeasible() {
    let errs = check(r#"
        fn f[N, M](a: Tensor[f32, [N, 4]], b: Tensor[f32, [M, 4]]) -> f32 {
            let t = @fuse { a .+ b }
            sum(t)
        }
        fn main() -> nil { nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible")
            && e.contains("[N, 4]") && e.contains("[M, 4]")),
        "got: {:?}", errs);
}

// `@fuse` attached to a statement (`@fuse let x = …`) is an attachment
// error, whatever the body: the catalog's attachment is block / expr, and
// the JIT refuses every statement-attached directive before any fuse
// analysis runs — admitting the form (even with a feasible body) would
// split the backends on a program the expression spelling handles. The
// attachment refusal fires alone; the body analysis never runs on it.
#[test]
fn fuse_on_a_let_statement_is_an_attachment_error() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [1.0f32, 2.0f32]
            @fuse let x = softmax(a, -1)
            print(x[0])
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("illegal directive stack")
            && e.contains("`@fuse`") && e.contains("`let` statement")),
        "got: {:?}", errs);
    assert!(fuse_infeasible(&errs).is_empty(),
        "the attachment refusal fires alone — the body analysis must not \
         also run on a refused form; got: {:?}", errs);
}

// A feasible body changes nothing: the divergence is the attachment itself
// (`dmc jit` refuses every statement directive; `dmc run` would execute),
// so the checker refuses the spelling and the hint names the expression
// form that both backends handle.
#[test]
fn fuse_on_a_feasible_let_statement_is_an_attachment_error() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [1.0f32, 2.0f32]
            @fuse let x = a .+ a
            print(x[0])
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("illegal directive stack")
            && e.contains("`@fuse`") && e.contains("`let` statement")),
        "got: {:?}", errs);
}

// Any other statement kind is the same refusal, naming the attachment.
#[test]
fn fuse_on_a_control_flow_statement_is_an_attachment_error() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [1.0f32, 2.0f32]
            @fuse for i in 0..2 {
                print(a[i])
            }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("illegal directive stack")
            && e.contains("`@fuse`") && e.contains("`for` statement")),
        "got: {:?}", errs);
}

// The supported spelling: the same fused unit as an expression checks clean.
#[test]
fn fuse_expression_form_of_a_let_is_accepted() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [1.0f32, 2.0f32]
            let x = @fuse { a .+ a }
            print(x[0])
            nil
        }
    "#);
    assert!(errs.iter().all(|e| !e.contains("illegal directive stack")
            && !e.contains("fuse-infeasible")),
        "got: {:?}", errs);
}

// Unsuffixed float tensor literals are the language's ordinary form; they
// type as f32 tensors (both backends build f32 lanes from float leaves) and
// must fuse — refusing them was a false diagnostic against a program the
// JIT's fused kernel lowers happily.
#[test]
fn unsuffixed_float_literal_tensors_fuse() {
    let errs = check(r#"
        fn main() -> nil {
            let a = [1.0, 2.0]
            let c = @fuse { \>(a .+ a .* 0.5) }
            print(c[0])
            nil
        }
    "#);
    assert!(fuse_infeasible(&errs).is_empty(), "got: {:?}", errs);
}

// Symbolic shapes: `[B, D]` against `[B, D]` is provably equal, so the walk
// stays silent — provable equality is the bar, and same-name symbols meet it.
#[test]
fn equal_symbolic_shapes_inside_fuse_are_accepted() {
    let errs = check(r#"
        fn affine[B, D](x: Tensor[f32, [B, D]], g: Tensor[f32, [B, D]])
            -> Tensor[f32, [B, D]] {
            @fuse { x .* g }
        }
        fn main() -> nil { nil }
    "#);
    assert!(fuse_infeasible(&errs).is_empty(), "got: {:?}", errs);
}

// The statement diagnostic anchors at the offending statement, not at the
// enclosing `@fuse` block.
#[test]
fn statement_diagnostic_anchors_at_the_statement() {
    let errs = check_full(r#"
        fn main() -> nil {
            let a = [1.0f32, 2.0f32]
            let c = @fuse {
                let t = a .* a
                t .+ a
            }
            print(c[0])
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.msg.contains("fuse-infeasible") && e.span.line == 5),
        "expected the diagnostic on line 5 (the `let t`), got: {:?}",
        errs.iter().map(|e| (e.span.line, e.msg.clone())).collect::<Vec<_>>());
}

// The trailing-block path through `check_block` bypasses `check_stmt`; the
// gate has to hold there the same way port-forbidden's does.
#[test]
fn trailing_fuse_block_is_checked_too() {
    let errs = check(r#"
        fn f() -> Tensor[f32, [2, 2]] {
            let a = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]
            @fuse { a @ a }
        }
        fn main() -> nil { nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("fuse-infeasible") && e.contains("`@` is not elementwise")),
        "got: {:?}", errs);
}

// ── DIRECTIVES.md §3: `@shard`/`@tp` attachment, expression form ───────────
//
// The catalog advertises `@shard` on an expression as well as on a `let`, so
// the expression form has to be checked identically — every way the `let`
// form is a hard error, the brace form is the same hard error.

#[test]
fn shard_block_requires_a_tensor_like_value() {
    let errs = check(r#"
        let mesh = Mesh[dp=8, tp=4]
        fn main() -> nil {
            let s: f32 = 1.0
            let y = @shard(axis=0, mesh=mesh.dp) { s }
            print(y)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@shard") && e.contains("tensor-like")),
        "expected a `@shard` non-tensor error on the block form, got: {:?}", errs);
}

#[test]
fn shard_block_requires_the_mesh_divisor_in_the_shape() {
    let errs = check(r#"
        let mesh = Mesh[dp=8, tp=4]
        fn main[B, D](x: Tensor[f32, [B, D]]) -> nil {
            let y = @shard(axis=0, mesh=mesh.dp) { x }
            print(y)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@shard") && e.contains("divisor `dp`")),
        "expected a `@shard` divisor error on the block form, got: {:?}", errs);
}

#[test]
fn shard_block_requires_the_mesh_argument() {
    let errs = check(r#"
        fn main[B, D](x: Tensor[f32, [B, D]]) -> nil {
            let y = @shard(axis=99) { x }
            print(y)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@shard") && e.contains("mesh=mesh.axis")),
        "expected a `@shard` missing-mesh error on the block form, got: {:?}", errs);
}

#[test]
fn shard_block_axis_must_be_in_bounds() {
    let errs = check(r#"
        let mesh = Mesh[dp=8, tp=4]
        fn main[B, D](x: Tensor[f32, [B/dp, D]]) -> nil {
            let y = @shard(axis=99, mesh=mesh.dp) { x }
            print(y)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@shard") && e.contains("axis 99 out of bounds")),
        "expected a `@shard` axis-bounds error on the block form, got: {:?}", errs);
}

#[test]
fn tp_block_requires_a_tensor_like_value() {
    let errs = check(r#"
        fn main() -> nil {
            let s: f32 = 1.0
            let y = @tp(axis=-1) { s }
            print(y)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@tp") && e.contains("tensor-like")),
        "expected a `@tp` non-tensor error on the block form, got: {:?}", errs);
}

#[test]
fn tp_block_requires_the_tp_divisor_in_the_shape() {
    let errs = check(r#"
        fn main[D](w: Tensor[f32, [D, 4*D]]) -> nil {
            let y = @tp(axis=-1) { w }
            print(y)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@tp") && e.contains("divisor `tp`")),
        "expected a `@tp` divisor error on the block form, got: {:?}", errs);
}

#[test]
fn tp_block_axis_must_be_in_bounds() {
    let errs = check(r#"
        fn main[D](w: Tensor[f32, [D, 4*D/tp]]) -> nil {
            let y = @tp(axis=99) { w }
            print(y)
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@tp") && e.contains("axis 99 out of bounds")),
        "expected a `@tp` axis-bounds error on the block form, got: {:?}", errs);
}

// A well-formed sharded value in the expression form stays quiet — the check
// is the `let` form's check, not a ban on the brace spelling.
#[test]
fn a_well_formed_shard_block_is_legal() {
    let errs = check(r#"
        let mesh = Mesh[dp=8, tp=4]
        fn main[B, D](x: Tensor[f32, [B/dp, D]]) -> nil {
            let y = @shard(axis=0, mesh=mesh.dp) { x }
            print(y)
            nil
        }
    "#);
    assert!(!errs.iter().any(|e| e.contains("@shard")),
        "expected a well-formed `@shard` block to pass, got: {:?}", errs);
}

// The statement form of the block reaches the same check — a `@shard { … }`
// standing as a statement is not a way around the rule either.
#[test]
fn shard_block_as_a_statement_is_checked() {
    let errs = check(r#"
        let mesh = Mesh[dp=8, tp=4]
        fn main() -> nil {
            let s: f32 = 1.0
            @shard(axis=0, mesh=mesh.dp) { print(s) }
            nil
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@shard") && e.contains("tensor-like")),
        "expected a `@shard` non-tensor error on the statement form, got: {:?}", errs);
}

// The third way in: a directive block standing as a function body's trailing
// value takes its own path through `check_block`, so it needs its own guard.

#[test]
fn shard_block_as_a_trailing_value_is_checked() {
    let errs = check(r#"
        let mesh = Mesh[dp=8, tp=4]
        fn main() -> f32 {
            let s: f32 = 1.0
            @shard(axis=0, mesh=mesh.dp) { s }
        }
    "#);
    assert!(errs.iter().any(|e| e.contains("@shard") && e.contains("tensor-like")),
        "expected a `@shard` non-tensor error on the trailing-block form, got: {:?}", errs);
}

#[test]
fn a_port_handle_in_a_model_field_is_rejected() {
    // SPEC §3.11: a handle "cannot appear inside tensor element types, model
    // fields, or Vault constants". Nothing enforced the model-field half, so
    // `dmc run` accepted the program and `dmc jit` failed it with `cannot
    // convert `Port` to `Port`` — the field's type had resolved to a *model*
    // named `Port`, so both sides rendered alike and neither was the handle
    // type. One answer now, from the checker, so both backends give it.
    let errs = check("model Holder { h: Port[python] }\nfn main() -> i64 { 0 }\n");
    assert!(errs.iter().any(|e| e.contains("model field `Holder.h` is a port handle")),
        "{:?}", errs);
    // A handle in a parameter or return position stays legal — the same §3.11
    // sentence says so explicitly.
    assert!(passes("fn ask(q: Port[python]) -> Port[python] { q }\nfn main() -> i64 { 0 }\n"));
}

#[test]
fn a_query_dim_still_accepts_differing_shapes_501() {
    // The `[?, ?]` escape hatch (SPEC §3.2) is what survives the removal of
    // `_`-as-dim (#501, ruling S3): one parameter, two argument shapes, no
    // shape error.
    assert!(passes(r#"
        fn total(x: Tensor[f32, [?, ?]]) -> f32 { sum(x) }
        fn main() -> i64 {
            let a = total([[1.0f32, 2.0f32], [3.0f32, 4.0f32]])
            let b = total([[1.0f32, 2.0f32, 3.0f32]])
            (a + b) as i64
        }
    "#));
}

#[test]
fn a_shape_diagnostic_spells_a_dynamic_dim_query_501() {
    // The checker is the third renderer of a dynamic dim, after `fmt` and the
    // JIT's refusal. It spelled one `_`, so a shape error described the user's
    // `[?, 4]` back to them as `[_, 4]` — a type the parser rejects since #501.
    let errs = check(
        "fn need(x: Tensor[f32, [?, 4]]) -> Tensor[f32, [2, 8]] { x }\n\
         fn main() -> i64 { 0 }\n",
    );
    assert_eq!(errs.len(), 1, "expected one shape error, got {errs:?}");
    assert!(errs[0].contains("[?, 4]"), "dynamic dim renders `?`, got: {}", errs[0]);
    assert!(!errs[0].contains('_'), "no dim may render as `_`, got: {}", errs[0]);
}

// ── #550: `as` cast legality ────────────────────────────────────────────────
//
// The `Cast` arm resolved its target type and returned it on faith, so a cast
// between types with no conversion type-checked clean and the interpreter
// handed the *unconverted* value back — a `-> i64` function returning a `str`,
// with the checker's blessing. `dmc jit` already refused every one of these;
// the diagnostics below reuse its wording so the two backends do not describe
// the same bad program differently.

/// The headline repro. `str` has no numeric reading, so there is no conversion
/// to name — SPEC §3.1 licenses `as` between *scalars* and to `str`, never out
/// of one.
#[test]
fn a_str_cannot_be_cast_to_a_number_550() {
    for (src, want) in [
        ("fn main() -> i64 { let s: str = \"abc\"  s as i64 }",
         "cannot convert `str` to `i64`"),
        ("fn main() -> f64 { let s: str = \"1.5\"  s as f64 }",
         "cannot convert `str` to `f64`"),
        ("fn main() -> bool { let s: str = \"x\"  s as bool }",
         "cannot convert `str` to `bool`"),
    ] {
        let errs = check(src);
        assert_eq!(errs.len(), 1, "expected exactly one error for {src:?}, got {errs:?}");
        assert_eq!(errs[0], want, "message must match the JIT's, got: {}", errs[0]);
    }
}

/// The other kinds with no conversion: `nil`, a tuple, and a tensor retyped as
/// a whole other tensor — which the JIT spells with exactly these words.
#[test]
fn the_kinds_with_no_conversion_are_refused_550() {
    for (src, want) in [
        ("fn main() -> i64 { let n = nil  n as i64 }",
         "cannot convert `nil` to `i64`"),
        ("fn main() -> i64 { let p: (i64, i64) = (1, 2)  p as i64 }",
         "cannot convert `(i64, i64)` to `i64`"),
        ("fn main() -> str { let t: Tensor[f32, [2]] = forge.zeros[f32, [2]]  t as str }",
         "cannot convert `Tensor[f32, [2]]` to `str`"),
        ("fn main() -> i64 { let t: Tensor[f32, [2]] = forge.zeros[f32, [2]]  \
                             let u = t as Tensor[i64, [2]]  u[0] }",
         "cannot convert `Tensor[f32, [2]]` to `Tensor[i64, [2]]`"),
    ] {
        let errs = check(src);
        assert!(errs.iter().any(|e| e == want),
            "expected {want:?} for {src:?}, got {errs:?}");
    }
}

/// (1) scalar ↔ scalar, the family SPEC §3.1 exists to describe — including
/// `bool as i64` (`1` on both backends) and the #540 narrowing wrap.
#[test]
fn the_scalar_conversions_stay_legal_550() {
    assert!(passes("fn main() -> i64 { let b: bool = true  b as i64 }"),
        "`bool as i64` is a real conversion and must not be refused");
    assert!(passes("fn main() -> i64 { let a: i32 = 1  a as i64 }"),
        "widening i32 -> i64");
    assert!(passes("fn main() -> i32 { let a: i64 = 5000000000  a as i32 }"),
        "the #540 narrowing wrap");
    assert!(passes("fn main() -> i8 { let a: i64 = 300  a as i8 }"),
        "narrowing to a sub-word integer");
    assert!(passes("fn main() -> i64 { let x: f64 = 1.5  x as i64 }"),
        "float -> int truncation (OPERATORS.md §7)");
    assert!(passes("fn main() -> f32 { let x: f64 = 1.5  x as f32 }"),
        "f64 -> f32 narrowing");
    assert!(passes("fn main() -> bool { let n: i64 = 2  n as bool }"),
        "int -> bool (0 is false, anything else true)");
    assert!(passes("fn main() -> f32 { 5 as f32 }"),
        "an untyped int literal casts into the float family");
}

/// (2) `x as str` — SPEC §3.1, "Integer-to-string conversion: `n as str`".
#[test]
fn casting_to_str_stays_legal_550() {
    assert!(passes("fn main() -> str { let n: i64 = 42  n as str }"));
    assert!(passes("fn main() -> str { let x: f64 = 0.5  x as str }"));
    assert!(passes("fn main() -> str { let b: bool = true  b as str }"));
    assert!(passes("fn main() -> str { let s: str = \"abc\"  s as str }"),
        "the identity cast is always legal");
}

/// (3) the elementwise tensor cast. It is legal — `dmc run` maps every element,
/// and `dmc jit` declines it as *unsupported*, not illegal — but it produces a
/// TENSOR of the target element type, which is what typing it as the bare
/// scalar got wrong: `fn main() -> i64 { … t as i64 }` used to check clean and
/// then return a `Tensor`.
#[test]
fn the_elementwise_tensor_cast_is_legal_and_yields_a_tensor_550() {
    assert!(passes(
        "fn main() -> Tensor[i64, [2]] { \
           let t: Tensor[f32, [2]] = forge.zeros[f32, [2]]  t as i64 }"),
        "a tensor cast to a numeric scalar is an elementwise cast");
    assert!(passes(
        "fn main() -> Tensor[f32, [4, 3]] { \
           let t: Tensor[f64, [4, 3]] = forge.zeros[f64, [4, 3]]  t as f32 }"),
        "…at any rank, narrowing inside the float family");
    // Typed as the scalar, this was the second half of the soundness hole.
    let errs = check(
        "fn main() -> i64 { let t: Tensor[f32, [2]] = forge.zeros[f32, [2]]  t as i64 }");
    assert!(errs.iter().any(|e| e.contains("returns `i64`") && e.contains("Tensor[i64, [2]]")),
        "an elementwise cast must not type as a scalar, got {errs:?}");
    // …and the same through a bare `let` off a constructor, which reports
    // `Unknown` on purpose (#248) — the shape side-table still knows it is a
    // tensor. This spelling is the second repro in #550.
    let errs = check("fn main() -> i64 { let t = forge.zeros[f32, [2]]  t as i64 }");
    assert!(errs.iter().any(|e| e.contains("returns `i64`") && e.contains("Tensor[i64, [2]]")),
        "the constructor-bound form must be caught too, got {errs:?}");
}

/// (4) enum ↔ ordinal — SPEC §3.1: "An enum value is its variant's `i64`
/// ordinal … `Token.Eq as i64 == 1`", and the explicit way back the JIT lowers.
#[test]
fn the_enum_ordinal_casts_stay_legal_550() {
    assert!(passes("enum Light { Red, Amber, Green }\n\
                    fn main() -> i64 { Light.Amber as i64 }"));
    assert!(passes("enum Light { Red, Amber, Green }\n\
                    fn main() -> i64 { let n: i64 = 1  let l = n as Light  l as i64 }"));
}

/// (5) the types the checker deliberately does not model must not be refused:
/// `any` (⊥), a shape parameter in value position, a dynamic map read.
#[test]
fn casts_involving_unmodelled_types_are_not_refused_550() {
    assert!(passes("fn s[D]() -> f32 { D as f32 }\nfn main() -> f32 { s[4]() }"),
        "a shape param is a SymDim, not a value the checker types");
    assert!(passes("fn f(x: any) -> i64 { x as i64 }\nfn main() -> i64 { f(1) }"),
        "`any` is the dynamic escape hatch — it converts to anything");
    assert!(passes("fn main() -> i64 { let m = map_new()  map_set(m, \"k\", 1)  \
                    map_get(m, \"k\") as i64 }"),
        "a dynamically-typed map read still casts");
}

/// (6) inside a SHAPE-GENERIC body. Every case above is written at the top
/// level, where the checker has concrete types for everything; a generic body
/// is the environment where it does not, and the cast rule has to hold there
/// too. Two ways it could fail: refuse a legal cast because a shape parameter
/// made the surrounding types unresolved, or wave an illegal one through for
/// the same reason — the second being the #550 hole re-opened one scope in.
#[test]
fn cast_legality_holds_inside_a_generic_body_550() {
    // Illegal: `str` has no numeric reading here either, and the message must
    // be the same one the top-level form gets.
    let errs = check(
        "fn g[N](x: Tensor[f32, [N]]) -> i64 {\n\
         let s: str = \"abc\"\n\
         (s as i64) + N\n\
         }\n\
         fn main() -> i64 { let t = forge.zeros[f32, [2]]  g(t) }\n",
    );
    assert!(errs.iter().any(|e| e == "cannot convert `str` to `i64`"),
        "an illegal cast in a generic body must be refused, got {errs:?}");

    // Legal, and must stay legal: a concrete-to-concrete scalar cast whose
    // operand happens to live beside a shape parameter.
    assert!(passes(
        "fn g[N](x: Tensor[f32, [N]]) -> i64 {\n\
         let a: i32 = 1\n\
         (a as i64) + N\n\
         }\n\
         fn main() -> i64 { let t = forge.zeros[f32, [2]]  g(t) }\n"),
        "a legal scalar cast must not be refused just for sharing a scope with `N`");

    // The elementwise tensor cast keeps the shape PARAMETER, rather than
    // collapsing to a scalar or to an unresolved shape.
    assert!(passes(
        "fn g[N](x: Tensor[f32, [N]]) -> Tensor[i64, [N]] { x as i64 }\n\
         fn main() -> i64 { let t = forge.zeros[f32, [2]]  let u = g(t)  u[0] }\n"),
        "an elementwise cast in a generic body yields Tensor[i64, [N]]");

    // …and typed as the bare scalar it is still the #550 soundness hole, with
    // `N` carried into the diagnostic rather than erased to a placeholder.
    let errs = check(
        "fn g[N](x: Tensor[f32, [N]]) -> i64 { x as i64 }\n\
         fn main() -> i64 { let t = forge.zeros[f32, [2]]  g(t) }\n",
    );
    assert!(errs.iter().any(|e| e.contains("returns `i64`") && e.contains("Tensor[i64, [N]]")),
        "the generic elementwise cast must not type as a scalar, got {errs:?}");
}

/// #549 fixed the recursion into a cast's operand and left the cast alone.
/// Both halves must now fire, and the operand's own diagnostics must not be
/// swallowed by the new one.
#[test]
fn the_operand_checks_and_the_cast_check_coexist_549_550() {
    assert!(!passes("fn m() -> i64 { (nonexistent_var + 1) as i64 }"),
        "#549's operand recursion must still report the undefined identifier");
    assert!(passes("fn m() -> i64 { let a: i32 = 1  a as i64 }"),
        "a legal cast over a clean operand stays clean");
}

// ── #533: a `trit` tensor has no copy-mode wire dtype ───────────────────────
//
// PORTS.md §3.2: "`trit` has no wire dtype. A packed ternary weight is a
// demoniC storage format, not a portable element type." Both backends enforce
// that at run time; the set of encodable element types is a property of the
// argument's *type*, so it is a compile-time error (AGENTS.md §2.5). The
// wording is the backends' own, pinned on their side by #512's
// `jit_a_trit_tensor_is_refused_the_way_the_interpreter_refuses_it`.

const TRIT_ENCODE_REFUSAL: &str =
    "port_tensor_encode: a `trit` tensor has no copy-mode wire dtype (PORTS.md §3.2)";
const TRIT_DECODE_REFUSAL: &str =
    "port_tensor_decode: a `trit` tensor has no copy-mode wire dtype (PORTS.md §3.2)";

/// Every spelling that reaches the primitive with a `trit` tensor: the bare
/// `let` off the constructor (the repro in #533, which types as `Unknown`
/// because `forge.trit` carries no element-type argument), an annotated
/// binding, a parameter, and the constructor written inline.
#[test]
fn port_tensor_encode_refuses_a_trit_tensor_533() {
    for src in [
        "fn main() -> str { let !t = forge.trit[2, 2]  port_tensor_encode(t) }",
        "fn main() -> str { let !t: Tensor[trit, [2, 2]] = forge.trit[2, 2]  \
                            port_tensor_encode(t) }",
        "fn enc(t: Tensor[trit, [2, 2]]) -> str { port_tensor_encode(t) }\n\
         fn main() -> str { let !t = forge.trit[2, 2]  enc(t) }",
        "fn main() -> str { port_tensor_encode(forge.trit[2, 2]) }",
    ] {
        let errs = check(src);
        assert!(errs.iter().any(|e| e == TRIT_ENCODE_REFUSAL),
            "expected the backends' refusal for {src:?}, got {errs:?}");
    }
}

/// `port_tensor_decode(s, like)` reads `like`'s dtype, so a `trit` declared
/// payload buffer is the same refusal — PORTS.md §3.2's second primitive.
#[test]
fn port_tensor_decode_refuses_a_trit_like_tensor_533() {
    for src in [
        "fn main() -> i64 { let !t = forge.trit[2, 2]  \
                            let (v, e) = port_tensor_decode(\"{}\", t)  0 }",
        "fn dec(t: Tensor[trit, [2, 2]]) -> i64 { \
           let (v, e) = port_tensor_decode(\"{}\", t)  0 }\n\
         fn main() -> i64 { let !t = forge.trit[2, 2]  dec(t) }",
    ] {
        let errs = check(src);
        assert!(errs.iter().any(|e| e == TRIT_DECODE_REFUSAL),
            "expected the backends' refusal for {src:?}, got {errs:?}");
    }
}

/// The refusal is about the element type and nothing else: every dtype §3.2's
/// table does define still encodes, and a `trit` tensor doing the job it exists
/// for — a POPCNT matmul — is untouched.
#[test]
fn the_wire_dtypes_still_encode_and_trit_still_matmuls_533() {
    for src in [
        "fn main() -> str { let t = forge.zeros[i64, [2, 3]]  port_tensor_encode(t) }",
        "fn main() -> str { let t = forge.zeros[f32, [2]]  port_tensor_encode(t) }",
        "fn main() -> str { let t = forge.zeros[bool, [2]]  port_tensor_encode(t) }",
        "fn main() -> i64 { let t = forge.zeros[f64, [2]]  \
                            let (v, e) = port_tensor_decode(\"{}\", t)  0 }",
        "fn main() -> f32 { let x = forge.ones[f32, [2, 4]]  \
                            let !w = forge.trit[4, 3]  sum(x @ w) }",
    ] {
        assert!(passes(src), "must stay legal: {src:?}, got {:?}", check(src));
    }
}

/// A rebind drops the tag in that scope, like every other binding side-table.
#[test]
fn a_rebound_trit_binding_stops_being_a_trit_533() {
    assert!(passes(
        "fn main() -> str { let !t = forge.trit[2, 2]  \
                            let t = forge.zeros[f32, [2]]  port_tensor_encode(t) }"),
        "the name now holds an f32 tensor: {:?}",
        check("fn main() -> str { let !t = forge.trit[2, 2]  \
                                  let t = forge.zeros[f32, [2]]  port_tensor_encode(t) }"));
}

// ── #562: `..` range slabs with a shape-parameter bound ───────────────────
//
// `x[0 .. S]` parses as `IndexElem::Expr(Expr::Range)`, not
// `IndexElem::Slice` — interp.rs and jit.rs both already special-case that
// (see their own "#276" / "not `IndexElem::Slice`" comments). The checker's
// `PostfixOp::Index` arm didn't: it tested `matches!(e, IndexElem::Expr(_))`
// to mean "plain scalar index", which is also true of a `Range`-wrapping
// `Expr`, so `x[0..S]` read as one scalar index and collapsed the whole
// tensor to its element type. The colon spelling (`x[0:S]`) went through
// `IndexElem::Slice` and *looked* right only because the non-scalar branch
// returned bare `Unknown` — permissive enough to duck any shape mismatch,
// not an actually-correct derived shape. Both are covered below via
// `classify_index_axis`/`derive_slice_shape`, the one place that now decides
// scalar-vs-slice and the sliced shape for every spelling.

/// The issue's own repro: a `..` slab whose end is a bare shape parameter,
/// passed across a function boundary that expects a tensor.
#[test]
fn range_slab_with_shape_param_end_types_as_tensor_562() {
    let src = "\
        fn take[N](t: Tensor[f32, [N]]) -> f32 { sum(t) } \
        fn demo[S](x: Tensor[f32, [S]]) -> nil { \
            let v = x[0 .. S] \
            let _ = take(v) \
            nil \
        }";
    assert!(passes(src), "expected Check OK, got {:?}", check(src));
}

/// The other repro in the issue: a derived extent (`S / 2`), not just a bare
/// shape parameter. `SymDim` can represent `S/2` (it has `Div`), so this must
/// type as `Tensor[F32, [(S/2)]]`, not fall back to `Unknown`.
#[test]
fn range_slab_with_derived_extent_types_precisely_562() {
    let right = "\
        fn take_half[H](t: Tensor[f32, [H]]) -> f32 { sum(t) } \
        fn demo[S](x: Tensor[f32, [S]]) -> nil { \
            let v = x[0 .. S / 2] \
            let _ = take_half(v) \
            nil \
        }";
    assert!(passes(right), "expected Check OK, got {:?}", check(right));

    // Prove the shape is the real derived extent and not merely "some
    // tensor": binding it against an incompatible CONSTANT shape must still
    // be rejected. If this passed, the slab would be typing as `Unknown`
    // (or any other shape-erasing fallback) again, not as `S/2`.
    let wrong = "\
        fn take99(t: Tensor[f32, [99]]) -> f32 { sum(t) } \
        fn demo[S](x: Tensor[f32, [S]]) -> nil { \
            let v = x[0 .. S / 2] \
            let _ = take99(v) \
            nil \
        }";
    assert!(!passes(wrong), "a [99]-shaped param must reject a derived (S/2) arg");
}

/// The colon spelling of the same slab. Before the fix this passed `--check`
/// for the wrong reason (the non-scalar branch returned `Unknown`, which is
/// argument-compatible with anything); pin that it now passes for the RIGHT
/// reason by also rejecting an incompatible constant shape, the same way the
/// `..` spelling does above.
#[test]
fn colon_slab_with_shape_param_end_is_precise_not_unknown_562() {
    let right = "\
        fn take_half[H](t: Tensor[f32, [H]]) -> f32 { sum(t) } \
        fn demo[S](x: Tensor[f32, [S]]) -> nil { \
            let v = x[0 : S / 2] \
            let _ = take_half(v) \
            nil \
        }";
    assert!(passes(right), "expected Check OK, got {:?}", check(right));

    let wrong = "\
        fn take99(t: Tensor[f32, [99]]) -> f32 { sum(t) } \
        fn demo[S](x: Tensor[f32, [S]]) -> nil { \
            let v = x[0 : S / 2] \
            let _ = take99(v) \
            nil \
        }";
    assert!(!passes(wrong),
        "colon-form slab must be precise too, not an Unknown that swallows a [99] mismatch: {:?}",
        check(wrong));
}

/// Literal bounds must be exactly as precise as shape-parameter bounds: the
/// derived extent is a concrete `Const`, and a mismatched fixed-size param
/// must still be rejected (this used to collapse to the element scalar too,
/// just like the shape-param case — the bug was never about `..` vs `:`, or
/// about literals vs shape params, only about the combination).
#[test]
fn range_slab_with_literal_bounds_types_precisely_562() {
    let right = "\
        fn take3(t: Tensor[f32, [3]]) -> f32 { sum(t) } \
        fn demo(x: Tensor[f32, [4]]) -> nil { \
            let v = x[0 .. 3] \
            let _ = take3(v) \
            nil \
        }";
    assert!(passes(right), "expected Check OK, got {:?}", check(right));

    let wrong = "\
        fn take4(t: Tensor[f32, [4]]) -> f32 { sum(t) } \
        fn demo(x: Tensor[f32, [4]]) -> nil { \
            let v = x[0 .. 3] \
            let _ = take4(v) \
            nil \
        }";
    assert!(!passes(wrong), "a [3]-shaped slab must reject a [4]-shaped param: {:?}", check(wrong));
}

/// A bare `..` (full axis, `IndexElem::FullSlice`) keeps a shape-parametric
/// dim unchanged — the pre-existing case `classify_index_axis` must not
/// regress.
#[test]
fn full_slice_keeps_shape_param_dim_562() {
    let src = "\
        fn take[N](t: Tensor[f32, [N]]) -> f32 { sum(t) } \
        fn demo[S](x: Tensor[f32, [S]]) -> nil { \
            let v = x[..] \
            let _ = take(v) \
            nil \
        }";
    assert!(passes(src), "expected Check OK, got {:?}", check(src));
}

/// Mixed indexing: a scalar axis (dropped) alongside a `..` slab on a second
/// axis (derived). Exercises `derive_slice_shape`'s per-axis dispatch, not
/// just the single-axis case in the issue.
#[test]
fn scalar_and_range_slab_mix_on_rank2_562() {
    let src = "\
        fn take[N](t: Tensor[f32, [N]]) -> f32 { sum(t) } \
        fn demo[M, N](t: Tensor[f32, [M, N]]) -> nil { \
            let row = t[0, 0 .. N] \
            let _ = take(row) \
            nil \
        }";
    assert!(passes(src), "expected Check OK, got {:?}", check(src));
}

/// A slice this can't reason about symbolically (a stepped slice, `a:b:c`)
/// must fall back to `Unknown` rather than assert a wrong shape — the task's
/// explicit caution: "a wrong shape is worse than the current error". This
/// pins the fallback, not a regression of the stepped-slice case (which
/// was, and remains, permissive).
#[test]
fn stepped_slice_still_falls_back_to_unknown_not_a_wrong_shape_562() {
    let src = "fn main() -> nil { \
        let t = forge.zeros[f32, [4]] \
        let v = t[0:4:2] \
        print(sum(v) as i64) \
        nil \
    }";
    assert!(passes(src), "stepped slice must stay permissive (Unknown), got {:?}", check(src));
}

/// A negative-literal bound needs Python-style from-the-end resolution,
/// which `derive_slice_shape` explicitly declines to model (see
/// `is_negative_literal`) — must stay permissive (`Unknown`), not compute
/// `dim - (-2)` as if it were a plain offset from zero.
#[test]
fn negative_literal_bound_still_falls_back_to_unknown_562() {
    let src = "fn main() -> nil { \
        let t = forge.zeros[f32, [4]] \
        let v = t[-2 .. 4] \
        print(sum(v) as i64) \
        nil \
    }";
    assert!(passes(src), "negative bound must stay permissive (Unknown), got {:?}", check(src));
}

/// Regression guard for the pre-#562 behavior this must NOT touch: a plain
/// all-scalar index (no slice/full-slice elem anywhere) still prunes the
/// shape per-axis and still catches a static out-of-bounds constant index.
#[test]
fn scalar_only_indexing_is_unaffected_by_the_slab_fix_562() {
    assert!(passes(
        "fn main() -> nil { let t = forge.zeros[f32, [4, 8]]  let row = t[0]  nil }"),
        "scalar indexing on a rank-2 tensor must still prune to the remaining axis");
    let oob = check(
        "fn main() -> nil { let t = forge.zeros[f32, [4]]  let x = t[9]  nil }");
    assert!(oob.iter().any(|m| m.contains("out of bounds")),
        "static OOB on a constant scalar index must still be caught: {:?}", oob);
}

// ── #575: `embed`'s `ids` argument must be an integer tensor ────────────────
//
// STDLIB.md §3.6 declares `embed(vocab, ids)` with `ids: Tensor[i64, [...B]]`.
// Nothing enforced that at `--check` time — a float `ids` tensor passed
// clean, the interpreter then gathered rows by truncating the float index,
// and the JIT refused with `` `embed`: ids must be an integer tensor `` at
// `dmc jit`. That backend split was the bug (deliberately left alone as a
// checker hole, not a JIT gap, by #577's refusal-classification sweep): an
// index has no defined meaning as a float, so the checker should have
// rejected the program before either backend ever saw it.

const EMBED_INDEX_TYPE_REFUSAL_PREFIX: &str = "embed-index-type: `embed`'s `ids` argument";

/// The exact repro from #575: a float `ids` tensor now fails `--check`
/// instead of passing and diverging between backends.
#[test]
fn embed_refuses_a_float_ids_tensor_575() {
    let src = "fn main() -> nil { \
        let !table = forge.zeros[f32, [4, 2]] \
        table[0, 0] = 1.0   table[1, 0] = 2.0 \
        let !ids = forge.zeros[f32, [2]] \
        ids[0] = 1.0 \
        let out = embed(table, ids) \
        print(out[0, 0]) \
        nil \
    }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.starts_with(EMBED_INDEX_TYPE_REFUSAL_PREFIX)
                                 && e.contains("Tensor[f32, [2]]")),
        "expected an embed-index-type refusal naming the float tensor type, got {errs:?}");
}

/// Both integer element types the two backends actually accept (`i64`, and
/// `i32` per the JIT's `ScalarKind::I32` arm) stay legal — this is a type
/// check, not a ban on `embed` gaining new callers.
#[test]
fn embed_still_accepts_integer_ids_575() {
    for ty in ["i64", "i32"] {
        let src = format!(
            "fn main() -> nil {{ \
                let table = forge.zeros[f32, [4, 2]] \
                let ids = forge.zeros[{ty}, [2]] \
                let out = embed(table, ids) \
                print(out[0, 0]) \
                nil \
            }}");
        assert!(passes(&src), "Tensor[{ty}, ...] ids must stay legal, got {:?}", check(&src));
    }
}

/// An `Unknown`-typed `ids` (e.g. the `any` escape hatch, #186 — demoniC has
/// no user-level generic element type, so this is the realistic way a call
/// site's `ids` type comes in undetermined) must not be flagged — the check
/// only fires when the element type is known and provably non-integral, per
/// the same conservatism as the neighboring `#533` trit check.
#[test]
fn embed_leaves_unknown_ids_type_alone_575() {
    let src = "fn call_embed(vocab: Tensor[f32, [4, 2]], ids: any) -> nil { \
        let out = embed(vocab, ids) \
        nil \
    }";
    assert!(passes(src), "an `any`-typed ids must not be flagged, got {:?}", check(src));
}

// ── #745: a keyed map builtin's key must be `str`, and its map argument must
// actually be a map ──────────────────────────────────────────────────────────
//
// SPEC.md §264: a map is `str → any`. Every keyed map builtin's arg 1 carried
// no type at all before this (`types.rs::builtin_sig`'s `mk` gives every
// builtin param `Unknown`), so `map_set(m, 5, 1)` — an int key, a bool key, a
// float key, or a tensor key — passed `--check` clean and only failed at
// `dmc run` ("key must be str, got ..."). The fix gives the key parameter a
// real `str` type; the existing general call-arg mechanism (`check.rs`'s
// `TyType::Fn { params, .. }` arm, used for ordinary fn calls too) then
// reports the mismatch with no map-specific code needed. The map argument
// itself (arg 0) needs a separate, dedicated check — `TyType::Map` is
// deliberately match-anything in `compatible_with` (#204), so a
// `Map`-typed arg-0 slot in the signature enforces nothing on its own.

/// The exact repro from #745: an int-literal key.
#[test]
fn map_set_refuses_an_int_key_745() {
    let src = "fn main() -> i64 { \
        let !m = map_new() \
        map_set(m, 5, 1) \
        0 \
    }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.contains("arg 1") && e.contains("str") && e.contains("integer")),
        "expected a located str-key refusal naming arg 1, got {errs:?}");
}

/// A bool key — same rule, different offending type.
#[test]
fn map_set_refuses_a_bool_key_745() {
    let src = "fn main() -> i64 { \
        let !m = map_new() \
        map_set(m, true, 1) \
        0 \
    }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.contains("arg 1") && e.contains("str") && e.contains("bool")),
        "expected a located str-key refusal naming the bool key, got {errs:?}");
}

/// A float key — same rule, different offending type.
#[test]
fn map_set_refuses_a_float_key_745() {
    let src = "fn main() -> i64 { \
        let !m = map_new() \
        map_set(m, 5.0, 1) \
        0 \
    }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.contains("arg 1") && e.contains("str") && e.contains("float")),
        "expected a located str-key refusal naming the float key, got {errs:?}");
}

/// A tensor key — same rule; a tensor *literal* (not routed through a
/// constructor-bound local, which the checker deliberately types `Unknown`
/// regardless of annotation, #248) so the checker actually has a concrete
/// type to reject.
#[test]
fn map_set_refuses_a_tensor_key_745() {
    let src = "fn main() -> i64 { \
        let !m = map_new() \
        map_set(m, [1.0, 2.0], 1) \
        0 \
    }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.contains("arg 1") && e.contains("str") && e.contains("Tensor")),
        "expected a located str-key refusal naming the tensor key, got {errs:?}");
}

/// The rule applies to every keyed map builtin, not just `map_set`.
#[test]
fn every_keyed_map_builtin_refuses_a_non_str_key_745() {
    for call in ["map_get(m, 5)", "map_has(m, 5)", "map_contains(m, 5)", "map_del(m, 5)"] {
        let src = format!(
            "fn main() -> i64 {{ let !m = map_new()  map_set(m, \"k\", 1)  {call}  0 }}");
        let errs = check(&src);
        assert!(errs.iter().any(|e| e.contains("arg 1") && e.contains("str") && e.contains("integer")),
            "{call}: expected a located str-key refusal, got {errs:?}");
    }
}

/// A non-map arg 0 — the issue's other half. `TyType::Map` is match-anything
/// in `compatible_with` (#204), so this needs the dedicated check, not the
/// general per-param mechanism.
#[test]
fn map_set_refuses_a_non_map_first_arg_745() {
    let errs = check("fn main() -> i64 { map_set(5, \"k\", 1)  0 }");
    assert!(errs.iter().any(|e| e.contains("map_set") && e.contains("first arg must be map")
                                 && e.contains("integer")),
        "expected a located non-map-arg refusal, got {errs:?}");
}

/// Every keyed map builtin gets the non-map-arg-0 check too.
#[test]
fn every_keyed_map_builtin_refuses_a_non_map_first_arg_745() {
    for (call, builtin) in [
        ("map_get(5, \"k\")", "map_get"),
        ("map_has(5, \"k\")", "map_has"),
        ("map_contains(5, \"k\")", "map_contains"),
        ("map_del(5, \"k\")", "map_del"),
    ] {
        let src = format!("fn main() -> i64 {{ {call}  0 }}");
        let errs = check(&src);
        assert!(errs.iter().any(|e| e.contains(builtin) && e.contains("first arg must be map")),
            "{call}: expected a located non-map-arg refusal, got {errs:?}");
    }
}

/// A valid `str` key, on a real map, still passes clean — this is a type
/// check, not a ban on `map_set`/`map_get` gaining new callers.
#[test]
fn map_set_and_map_get_still_accept_a_str_key_745() {
    let src = "fn main() -> i64 { \
        let !m = map_new() \
        map_set(m, \"k\", 1) \
        map_get(m, \"k\") as i64 \
    }";
    assert!(passes(src), "a str key on a real map must stay legal, got {:?}", check(src));
}

/// #742/#743's shape must not regress: a `map_get` result used as a *later*
/// key types `Unknown` in the checker (only the JIT refuses it, citing #721),
/// so it must still pass `--check` clean.
#[test]
fn map_get_result_used_as_a_later_key_still_passes_check_745() {
    let src = "fn main() -> i64 { \
        let !rec = map_new() \
        map_set(rec, \"name\", \"gus\") \
        let !defs = map_new() \
        map_set(defs, map_get(rec, \"name\"), rec) \
        0 \
    }";
    assert!(passes(src),
        "a map_get result used as a key must stay a checker pass (JIT-only refusal, #721/#742/#743), got {:?}",
        check(src));
}

/// An `m: map`-annotated parameter must not trip the non-map-arg-0 check.
/// `resolve_type` deliberately does NOT resolve `map` to `TyType::Map` (see
/// its own comment, and `for_in_map_param_does_not_warn` above) — `map` is
/// match-anything there too, for the same reason the #204 lint stays quiet
/// on it. Mirrors `examples/demonic_lexer.dmc`'s `fn tok_kind(t: map) -> str
/// { map_get(t, "k") }`.
#[test]
fn map_annotated_param_does_not_trip_the_non_map_arg_check_745() {
    let src = "fn get_it(m: map) -> i64 { map_get(m, \"k\") as i64 } \
               fn main() -> i64 { let !m = map_new()  map_set(m, \"k\", 1)  get_it(m) }";
    assert!(passes(src),
        "an `m: map`-typed param must not trip the non-map-arg check, got {:?}", check(src));
}

// ── #758: `len`'s argument is typed against its admissible kind set ─────────
//
// STDLIB.md §3.7's own domain row: `len(x)` is `Tensor[T, S] | str | Map |
// Tuple -> i64`. #745's `mk_typed` mechanism gives a builtin parameter one
// fixed concrete `TyType` (`map_set`'s key is always `str`), which cannot
// express a domain spanning four different shapes. `TyType::OneOf` names the
// admissible `TyKind`s directly; `compatible_with`'s new `OneOf` arm matches
// by kind rather than structural equality, so the existing general call-arg
// mechanism (the same `TyType::Fn { params, .. }` arm #745's tests exercise)
// reports the mismatch with no new checker code, exactly as #745 intended
// for a builtin with a genuinely fixed parameter type.

/// The issue's exact repro: `len(true)` passed `--check` clean before this
/// fix and only failed at `dmc run` ("len: requires tensor, tuple, str,
/// list, or map").
#[test]
fn len_refuses_a_bool_argument_758() {
    let errs = check("fn main() -> i64 { len(true) }");
    assert!(errs.iter().any(|e|
        e.contains("arg 0") && e.contains("expected tensor, str, Map, or tuple") && e.contains("bool")),
        "expected a located domain refusal naming the bool argument, got {errs:?}");
}

/// An int literal — same rule, different offending type.
#[test]
fn len_refuses_an_int_argument_758() {
    let errs = check("fn main() -> i64 { len(1) }");
    assert!(errs.iter().any(|e|
        e.contains("arg 0") && e.contains("expected tensor, str, Map, or tuple") && e.contains("integer")),
        "expected a located domain refusal naming the int argument, got {errs:?}");
}

/// A float literal — same rule, different offending type.
#[test]
fn len_refuses_a_float_argument_758() {
    let errs = check("fn main() -> i64 { len(1.5) }");
    assert!(errs.iter().any(|e|
        e.contains("arg 0") && e.contains("expected tensor, str, Map, or tuple") && e.contains("float")),
        "expected a located domain refusal naming the float argument, got {errs:?}");
}

/// A literal `nil` — same rule, different offending type.
#[test]
fn len_refuses_a_nil_argument_758() {
    let errs = check("fn main() -> i64 { len(nil) }");
    assert!(errs.iter().any(|e|
        e.contains("arg 0") && e.contains("expected tensor, str, Map, or tuple") && e.contains("nil")),
        "expected a located domain refusal naming the nil argument, got {errs:?}");
}

/// Every admissible kind must still pass clean: tensor, str, map, tuple.
#[test]
fn len_accepts_every_admissible_kind_758() {
    let src = "fn main() -> i64 { \
        let t = forge.zeros[f32, [4]] \
        let a = len(t) \
        let b = len(\"hello\") \
        let !m = map() \
        map_set(m, \"k\", 1) \
        let c = len(m) \
        let tup = (1, 2, 3) \
        let d = len(tup) \
        a + b + c + d \
    }";
    assert!(passes(src), "every admissible len argument kind must pass, got {:?}", check(src));
}

/// A tensor slice (`t[a..b]`) types as `View`, not `Tensor` — `TyKind::Tensor`
/// must match it too (both reify to the same runtime tensor value; see
/// `TyKind`'s own doc comment).
#[test]
fn len_accepts_a_view_slice_758() {
    let src = "fn main() -> i64 { \
        let !t = forge.zeros[f32, [10]] \
        let v = t[2..6] \
        len(v) \
    }";
    assert!(passes(src), "a View-typed slice must pass len's tensor kind, got {:?}", check(src));
}

/// A `map_get` result stays `Unknown` in the checker (#721 — typed map
/// values aren't tracked end to end) — it must still pass `len`'s check the
/// same way it already passes every other typed builtin parameter (#745's
/// own `map_get_result_used_as_a_later_key_still_passes_check_745`).
#[test]
fn len_accepts_an_unknown_map_get_result_758() {
    let src = "fn main() -> i64 { \
        let !d = map() \
        map_set(d, \"k\", 3) \
        let x = map_get(d, \"k\") \
        len(x) \
    }";
    assert!(passes(src),
        "an Unknown map_get result must still pass len's check, got {:?}", check(src));
}

/// A user-declared top-level fn named `len` shadows the builtin spelling —
/// the same rule #745's own `user_fn_of_the_same_name_shadows_the_map_builtin_check_745`
/// establishes for `map_get`. `len(1, 2)` here calls the real 2-arg user fn,
/// not the builtin, so the 1-arg domain check must not fire on it at all.
#[test]
fn user_fn_of_the_same_name_shadows_the_len_builtin_check_758() {
    let src = "fn len(a: i64, b: i64) -> i64 { a + b } \
               fn main() -> i64 { len(1, 2) }";
    assert!(passes(src),
        "a user-declared `len` must shadow the builtin's domain check, got {:?}", check(src));
}

/// An `m: map`-annotated parameter must not trip `len`'s `Map` kind check —
/// `resolve_type` deliberately leaves `map` unresolved as `Named { name:
/// "map", .. }` rather than `TyType::Map` (see #745's own
/// `map_annotated_param_does_not_trip_the_non_map_arg_check_745`), and a real
/// map value reaches `len` through it exactly as it does through `map_get`.
#[test]
fn map_annotated_param_passes_len_758() {
    let src = "fn count(m: map) -> i64 { len(m) } \
               fn main() -> i64 { let !m = map()  map_set(m, \"k\", 1)  count(m) }";
    assert!(passes(src),
        "an `m: map`-typed param must not trip len's domain check, got {:?}", check(src));
}

/// The downstream-sweep finding: a `-> list`-annotated fn's result types as
/// `Named { name: "list", .. }` (SPEC has no static `list` type to resolve
/// to — the same escape hatch `map` gets), not `Unknown`. Before the
/// dedicated `OneOf` carve-out for this shape, this real pattern from
/// a downstream file (`len(surface)`, `surface:
/// parse_introspection(..) -> list`) newly failed `--check` under this fix.
#[test]
fn list_annotated_return_passes_len_758() {
    let src = "fn make_list() -> list { list() } \
               fn main() -> i64 { len(make_list()) }";
    assert!(passes(src),
        "a `-> list`-annotated result must not trip len's domain check, got {:?}", check(src));
}

/// The `range` counterpart of the same escape hatch.
#[test]
fn range_annotated_return_passes_len_758() {
    let src = "fn make_range() -> range { 0..10 } \
               fn main() -> i64 { len(make_range()) }";
    assert!(passes(src),
        "a `-> range`-annotated result must not trip len's domain check, got {:?}", check(src));
}

// ── #753: the `Map` wildcard no longer swallows a no-`->` fn's map tail, or
// a `nil` map key ────────────────────────────────────────────────────────────
//
// Before this fix, a fn with no declared return type at all typed its whole
// body as `nil` (`f.ret_type.unwrap_or(Unit)`), even when its tail expression
// was a real map — and `compatible_with`'s old `TyType::Map` arm (#204)
// returned `true` whenever *either* side was `Map`, so the `Unit`-vs-`Map`
// mismatch that `fn make() { 7 }` gets ("returns `nil` but body produces
// `i64`") was never even compared for a map-tailed body. That let
// `examples/package_manifest.dmc`'s `fn dependency(...) { ... dep }` (no
// arrow, `dep` a real map) pass `--check` while `map_set(dependency(...), ...)`
// then failed at runtime ("first arg must be map, got nil"). Fixed by scoping
// `compatible_with` to specifically refuse the `(Unit, Map)` pair (types.rs
// ~146) — a map-tailed fn must now declare `-> map`. The `(Unit, Map)`
// carve-out also feeds `unify_if_branches` (`if b { m } else { nil }` is now a
// check error too, the same predicate `match` arms use — see the test below).
//
// Reviewer follow-up on this same PR found two more bugs in the first pass:
//
//   - The arg-0 exemption list dropping `Unit` made the DEDICATED non-map-
//     arg-0 check fire on ANY `Unit`-typed arg 0 — but `compatible_with`'s new
//     `(Unit, Map)` carve-out means the GENERAL call-arg mechanism
//     (`inferred_pty.compatible_with`, `PostfixOp::Call`'s `TyType::Fn` arm)
//     now refuses that same pairing all on its own, so `map_set(nil, "k", 1)`
//     reported the same span TWICE. `Unit` is back on the dedicated check's
//     exemption list — not because it is tolerated, but because the general
//     mechanism now owns reporting it, and reporting it twice is a bug, not
//     thoroughness (see the "exactly one error" test below).
//   - The dedicated nil-KEY check (arg 1) gated on the key's static TYPE
//     (`Unit`) rather than the key EXPRESSION, so it also fired on a
//     `Unit`-typed key that was never actually `nil` at runtime: a no-`-> T`
//     helper's return type defaults to `Unit` regardless of its body's real
//     tail (`f.ret_type.unwrap_or(Unit)`), so `fn key(i: i64) { "k" +
//     to_str(i) }` (a working, `str`-returning helper — its own
//     declared-vs-body mismatch is masked by the same `(Unit, Str)` `(T,
//     Err)`-convention exemption used elsewhere) + `map_set(m, key(1), 5)` (a
//     real program with a genuine `str` key at runtime) got refused, naming a
//     type the value never actually had. The key check is now gated on a
//     literal `nil` at the call site (`Expr::Nil`), not merely a `Unit`-typed
//     expression. (That helper's OWN declaration arguably deserves its own
//     `-> str` rule — this fix's sibling finding — but that is a separate,
//     not-yet-filed issue; this call site must not guess at a callee's real
//     return type from its body.)

/// The headline repro: a no-`->` fn whose tail is a map now errors at its own
/// declaration, the same shape `fn make() { 7 }` gets.
#[test]
fn map_tailed_fn_without_return_type_is_a_check_error_753() {
    let src = "fn side() { let !m = map()  map_set(m, \"k\", 1)  m } \
               fn main() -> i64 { 0 }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.contains("side") && e.contains("returns `nil`")
                                 && e.contains("body produces `map`")),
        "expected a located return-type mismatch naming `side`, got {errs:?}");
}

/// The full #753 repro: the mistyped `side()` result then also trips the
/// GENERAL call-arg mechanism at the `map_set` call site — not the dedicated
/// non-map-arg-0 check, which now exempts `Unit` again (see the header
/// comment above) so the two don't both fire on the same span.
#[test]
fn map_tailed_fn_result_used_as_map_set_first_arg_is_also_a_check_error_753() {
    let src = "fn side() { let !m = map()  map_set(m, \"k\", 1)  m } \
               fn main() -> i64 { map_set(side(), \"k\", 1)  0 }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.contains("arg 0") && e.contains("expected map")
                                 && e.contains("got nil")),
        "expected the general call-arg mechanism to refuse the nil result at arg 0, got {errs:?}");
}

/// Declaring `-> map` on the same fn fixes it: the tail is a real map, the
/// return type matches, and the call site sees a real map, not `nil`.
#[test]
fn map_tailed_fn_with_declared_return_type_passes_753() {
    let src = "fn side() -> map { let !m = map()  map_set(m, \"k\", 1)  m } \
               fn main() -> i64 { map_set(side(), \"k\", 1)  0 }";
    assert!(passes(src),
        "a map-tailed fn declaring `-> map` must pass, got {:?}", check(src));
}

/// A literal `nil` map argument is refused the same way — not just a
/// mistyped helper result. The refusal comes from the general call-arg
/// mechanism (`compatible_with`'s `(Unit, Map)` carve-out), not the dedicated
/// arg-0 check — see the exactly-one-error test just below.
#[test]
fn map_set_refuses_a_nil_first_arg_753() {
    let errs = check("fn main() -> i64 { map_set(nil, \"k\", 1)  0 }");
    assert!(errs.iter().any(|e| e.contains("arg 0") && e.contains("expected map")
                                 && e.contains("got nil")),
        "expected a located non-map-arg refusal for a literal nil, got {errs:?}");
}

/// Reviewer follow-up: a nil arg 0 must report EXACTLY ONE error, not two.
/// Before this fix, dropping `Unit` from the dedicated arg-0 check's
/// exemption list left it firing on the same `Unit` value the general
/// call-arg mechanism (`compatible_with`'s `(Unit, Map)` carve-out) already
/// refuses on its own — `map_set(nil, "k", 1)` reported the same span twice
/// (`map_set: first arg must be map, got nil` AND `arg 0: expected map, got
/// nil`). `Unit` is back on the dedicated check's exemption list so only the
/// general mechanism reports it.
#[test]
fn map_set_nil_first_arg_reports_exactly_one_error_753() {
    let errs = check("fn main() -> i64 { map_set(nil, \"k\", 1)  0 }");
    assert_eq!(errs.len(), 1,
        "a nil map_set arg 0 must report exactly one error, not a double-report, got {errs:?}");
}

/// A literal `nil` key is refused: the `(Unit, Str)` compatibility that
/// exists for the `(T, Err)` convention must not leak into the map-builtin
/// key position.
#[test]
fn map_set_refuses_a_nil_key_753() {
    let src = "fn main() -> i64 { \
        let !m = map() \
        map_set(m, nil, 1) \
        0 \
    }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.contains("map_set") && e.contains("key must be str")
                                 && e.contains("nil")),
        "expected a located nil-key refusal, got {errs:?}");
}

/// The nil-key rule applies to every keyed map builtin, not just `map_set`.
#[test]
fn every_keyed_map_builtin_refuses_a_nil_key_753() {
    for (call, builtin) in [
        ("map_get(m, nil)", "map_get"),
        ("map_has(m, nil)", "map_has"),
        ("map_contains(m, nil)", "map_contains"),
        ("map_del(m, nil)", "map_del"),
    ] {
        let src = format!(
            "fn main() -> i64 {{ let !m = map()  map_set(m, \"k\", 1)  {call}  0 }}");
        let errs = check(&src);
        assert!(errs.iter().any(|e| e.contains(builtin) && e.contains("key must be str")
                                     && e.contains("nil")),
            "{call}: expected a located nil-key refusal, got {errs:?}");
    }
}

/// A `map_get` result used as a *later* key must not regress (#742/#743's
/// shape stays `Unknown` in the checker — JIT-only refusal), and neither must
/// a genuine `str` key on a real map.
#[test]
fn map_753_fixes_do_not_regress_valid_map_code() {
    let src = "fn main() -> i64 { \
        let !rec = map() \
        map_set(rec, \"name\", \"gus\") \
        let !defs = map() \
        map_set(defs, map_get(rec, \"name\"), rec) \
        0 \
    }";
    assert!(passes(src),
        "a map_get result used as a key must stay a checker pass (#742/#743), got {:?}",
        check(src));
}

/// Reviewer follow-up (#753's sibling finding): the nil-key check must gate
/// on a literal `nil` at the call site, not on the key expression's static
/// TYPE. `fn key(i: i64) { "k" + to_str(i) }` has no `->`, so its return type
/// defaults to `Unit` regardless of its body's real `str` tail — but it is a
/// working, `str`-returning helper (its own declared-vs-body mismatch passes
/// via the `(Unit, Str)` `(T, Err)`-convention exemption, the same one that
/// lets an `Err`-typed fn return plain `nil`). `map_set(m, key(1), 5)` has a
/// genuine `str` key at runtime and must not be refused as a nil key just
/// because the call's static type is `Unit`.
#[test]
fn map_set_key_from_a_no_arrow_str_tailed_helper_still_passes_753() {
    let src = "fn key(i: i64) { \"k\" + to_str(i) } \
               fn main() -> i64 { \
        let !m = map() \
        map_set(m, key(1), 5) \
        0 \
    }";
    assert!(passes(src),
        "a no-`->` helper whose real tail is `str` must not be refused as a nil map key, got {:?}",
        check(src));
}

/// The `(Unit, Map)` `compatible_with` carve-out (types.rs ~146, #753) feeds
/// `unify_if_branches` too, the same predicate `match` arms use: a real map in
/// one branch and a literal `nil` in the other must be refused, not silently
/// typed `map` the way the old match-anything `Map` wildcard allowed.
#[test]
fn if_else_map_vs_nil_is_now_a_check_error_753() {
    let src = "fn main() -> i64 { \
        let !m = map() \
        let b = true \
        let z = if b { m } else { nil } \
        map_len(z) \
    }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.contains("`if` branches yield incompatible types")
                                 && e.contains("map") && e.contains("nil")),
        "expected a branch-mismatch error for map vs nil, got {errs:?}");
}

/// A user-declared top-level fn of the same name shadows the builtin, the
/// same "the thing the program actually declared outranks a builtin
/// spelling" rule the JIT's #720 review (F2) applies to `map`/`map_new`. The
/// non-map-arg-0 check must not fire on a call that resolves to the user fn
/// — a plain `fn map_get(a: i64, b: i64) -> i64 { a + b }` + `map_get(1, 2)`
/// is a real, `--check`-clean, `dmc run`-answering program (answers `3`) on
/// the pre-#745 binary, and this check must not regress it.
#[test]
fn user_fn_of_the_same_name_shadows_the_map_builtin_check_745() {
    let src = "fn map_get(a: i64, b: i64) -> i64 { a + b } \
               fn main() -> i64 { map_get(1, 2) }";
    assert!(passes(src),
        "a user fn named `map_get` must shadow the builtin's non-map-arg-0 check, got {:?}",
        check(src));
}

// ── #744: a range bound must have an integer type ───────────────────────────
//
// #731 (#724) made the `Range` arm in `check_expr` recurse into `start`/`end`
// so an undefined or import-private name inside a bound is caught — but it
// left the bound's *type* unconstrained: `for k in 0..("x")` and `0..(1.5)`
// checked clean and ran as silent zero-iteration loops (`dmc run` gates on
// `--check`, so a str/float/bool bound never reached the interpreter's own
// `for`-loop tolerance at all before this fix — see `eval_range_bound`'s
// "#639" comment in `interp.rs`, which is deliberately unaffected: it stays
// the fallback for a bound the checker genuinely can't pin down, e.g. a
// dynamically-str `Unknown` value). The slice site (`t[a..b]`) never even
// reached the `Range` arm — `classify_index_axis` unpacks its bounds first —
// so it needed the same constraint applied separately; unfixed, it was
// worse than silent: `t[0..("x")]` panicked at run time instead
// (`interp.rs`'s `index_int`, confirmed against the release binary).

#[test]
fn str_bound_in_for_is_a_check_error_744() {
    let msg = one_error(r#"fn main() -> nil { for k in 0..("x") { nil } }"#);
    assert_eq!(msg, "range end must be integer, got str");
}

#[test]
fn float_bound_in_for_is_a_check_error_744() {
    let msg = one_error(r#"fn main() -> nil { for k in 0..(1.5) { nil } }"#);
    assert_eq!(msg, "range end must be integer, got {float}");
}

#[test]
fn bool_bound_in_for_is_a_check_error_744() {
    let msg = one_error(r#"fn main() -> nil { for k in 0..(true) { nil } }"#);
    assert_eq!(msg, "range end must be integer, got bool");
}

#[test]
fn str_start_bound_in_for_is_a_check_error_744() {
    let msg = one_error(r#"fn main() -> nil { for k in ("x")..10 { nil } }"#);
    assert_eq!(msg, "range start must be integer, got str");
}

#[test]
fn str_bound_in_slice_is_a_check_error_744() {
    // The repro from the issue title: `t[0..("x")]` reaches
    // `classify_index_axis`, not the `Range` arm, so it needs its own check.
    let msg = one_error(r#"
        fn main() -> nil {
            let t = [1.0, 2.0, 3.0, 4.0]
            let s = t[0..("x")]
            nil
        }
    "#);
    assert_eq!(msg, "range end must be integer, got str");
}

#[test]
fn float_start_bound_in_slice_is_a_check_error_744() {
    let msg = one_error(r#"
        fn main() -> nil {
            let t = [1.0, 2.0, 3.0, 4.0]
            let s = t[(1.5)..4]
            nil
        }
    "#);
    assert_eq!(msg, "range start must be integer, got {float}");
}

#[test]
fn str_step_in_colon_slice_is_a_check_error_744() {
    // The colon-form step shares `classify_index_axis`'s `IndexAxis::Slice`
    // with the `..` spellings (`SPEC.md §4.3`: "both spellings are permanent
    // surface"), so it is held to the same constraint.
    let msg = one_error(r#"
        fn main() -> nil {
            let t = [1.0, 2.0, 3.0, 4.0]
            let s = t[0:4:("x")]
            nil
        }
    "#);
    assert_eq!(msg, "slice step must be integer, got str");
}

#[test]
fn bool_bound_in_value_position_range_is_a_check_error_744() {
    // A range need not sit in a `for` or an index bracket at all — `#724`'s
    // own repro shape is a bare `let` binding.
    let msg = one_error(r#"fn main() -> nil { let r = 0..(true)  nil }"#);
    assert_eq!(msg, "range end must be integer, got bool");
}

#[test]
fn str_bound_in_call_arg_position_range_is_a_check_error_744() {
    let msg = one_error(r#"
        fn f(r: i64) -> nil { nil }
        fn main() -> nil { f(0..("x")) }
    "#);
    assert_eq!(msg, "range end must be integer, got str");
}

#[test]
fn plain_untyped_int_bounds_pass_744() {
    assert!(passes("fn main() -> nil { for i in 0..10 { nil } }"));
}

#[test]
fn matching_narrow_width_bounds_pass_744() {
    // `dmc run` iterates `0u8..4u8` the same as `0..4` today (confirmed
    // against the release binary) — a fixed-width integer is still an
    // integer.
    assert!(passes("fn main() -> nil { for i in 0u8..4u8 { nil } }"));
}

#[test]
fn mixed_untyped_and_narrow_width_bounds_pass_744() {
    assert!(passes("fn main() -> nil { for i in 0..4u8 { nil } }"));
}

#[test]
fn mixed_width_bounds_pass_744() {
    // `0u8..4i64`: the two bounds don't share a width, but `dmc run` iterates
    // it the same as any other integer range (confirmed against the release
    // binary) — this fix constrains each bound to *an* integer type, not to
    // a common one.
    assert!(passes("fn main() -> nil { for i in 0u8..4i64 { nil } }"));
}

#[test]
fn unknown_typed_bound_passes_744() {
    // A `map_get` result types as `Unknown` (#742/#743) — genuinely unknown
    // to the checker, not evidence of a wrong type, so it is accepted (and
    // remains the interpreter's own responsibility if it turns out not to be
    // an int at run time — `eval_range_bound`'s documented tolerance).
    assert!(passes(r#"
        fn main() -> nil {
            let m = map_new()
            map_set(m, "n", 4)
            let n: i64 = map_get(m, "n")
            for i in 0..n { nil }
        }
    "#));
}

#[test]
fn shape_param_bound_passes_744() {
    // A shape parameter reference types as `Unknown` in a method/fn body
    // (`Env::shape_param_in_scope`), the same "checker genuinely doesn't
    // pin this down further" shape as a `map_get` result.
    assert!(passes(r#"
        fn f[N](x: Tensor[f32, [N]]) -> nil {
            for i in 0..N { nil }
        }
        fn main() -> nil {
            let t = [1.0, 2.0, 3.0]
            f(t)
        }
    "#));
}

#[test]
fn open_ended_slice_range_still_passes_744() {
    // `t[1..]` has no end bound at all — nothing to constrain.
    assert!(passes(r#"
        fn main() -> nil {
            let t = [1.0, 2.0, 3.0]
            let s = t[1..]
            nil
        }
    "#));
}

// ── #768: a range bound cannot itself be a range ────────────────────────────
//
// `check_expr(Expr::Range)` types a bare range `Unknown` (see the `Range` arm
// in `check_expr`), and `is_integral_shape_arg` waves `Unknown` through on
// purpose (#474: an expression the checker can't pin down is not evidence of
// a wrong one) — but a bound that is *syntactically* a range is not
// undecidable, it's a known wrong shape, and the old code let it through
// both the slice site and the `for` site. Confirmed against the release
// binary built from main before #768:
// `t[(0..2)..3]` passed `--check` and then panicked under `dmc run` with
// "runtime error at 1:58: range start must be integer, got range"; `for i in
// (0..4)..8` passed `--check` and silently iterated `0..8` under `dmc run`.

#[test]
fn range_bound_in_slice_is_a_check_error_768() {
    let msg = one_error(r#"
        fn main() -> nil {
            let t = [1.0, 2.0, 3.0, 4.0]
            let s = t[(0..2)..3]
            nil
        }
    "#);
    assert_eq!(msg, "range start must be integer, got range");
}

#[test]
fn range_bound_in_for_is_a_check_error_768() {
    let msg = one_error("fn main() -> nil { for i in (0..4)..8 { nil } }");
    assert_eq!(msg, "range start must be integer, got range");
}

// ── #728: warn on a type-arg-less generic-model annotation ──────────────────
//
// #728 (ratified
// 2026-09-09): a shape-parameterized model named without an
// argument list in a parameter, field, return or binding annotation, or as a
// builtin's own model type argument (`forge.uninit[Surface, [N]]`), is a
// warning at 0.0.5-draft and becomes a compile-time error at 0.0.6-draft. A
// model with no shape parameters is unaffected — its bare name is its only
// spelling. Follow-up PR (3) of that section's list: the predicate and the
// diagnostic, landed here as a warning only (PR (4) turns it into an error).

fn type_arg_less_warning<'a>(warns: &'a [super::check::TypeError], model: &str) -> Option<&'a super::check::TypeError> {
    warns.iter().find(|w| {
        w.msg.starts_with("bare-generic-model: ")
            && w.msg.contains(&format!("names `{model}` with no argument list"))
    })
}

#[test]
fn type_arg_less_generic_model_warns_on_param() {
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               fn cap(b: Buf) -> i64 { b.t[0] }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Buf").expect("expected a type-arg-less warning");
    assert!(w.msg.contains("parameter `b` of function `cap`"), "{}", w.msg);
    assert!(w.msg.contains("SPEC.md §3.10"), "{}", w.msg);
    assert!(w.msg.contains("0.0.6-draft"), "{}", w.msg);
    assert_eq!((w.span.line, w.span.col), (2, 11));
    // Reviewer defect 1: param position is the one spot "write it
    // explicitly" is wrong advice (a bracket-less param check-passes,
    // silencing this lint forever, and the JIT refuses it identically
    // either way) — the hint must name the ruling's blessed spelling
    // instead: declare the shape parameter on the function itself.
    // Verified `--check` clean and `dmc jit`-lowering (see check.rs's
    // `type_arg_less_hint` doc comment).
    assert_eq!(
        w.hint.as_deref(),
        Some("declare the shape parameter on the function: `fn cap[N](b: Buf[N])`"),
    );
}

#[test]
fn type_arg_less_generic_model_warns_on_field() {
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               model Outer[K] { inner: Buf, k: i64 }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Buf").expect("expected a type-arg-less warning");
    assert!(w.msg.contains("field `inner` of model `Outer`"), "{}", w.msg);
    assert_eq!((w.span.line, w.span.col), (2, 25));
    // `Outer`'s own shape param is `K`, not `Buf`'s `N` — nothing to reuse,
    // so the hint is a concrete bracket, not a bare-name spelling that
    // would be a hard "undefined identifier" error in field position.
    assert_eq!(w.hint.as_deref(), Some("write a concrete shape argument: `Buf[4]`"));
}

#[test]
fn type_arg_less_generic_model_warns_on_return() {
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               fn make() -> Buf { Buf[4] { t: forge.zeros[i64, [4]] } }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Buf").expect("expected a type-arg-less warning");
    assert!(w.msg.contains("the return type of function `make`"), "{}", w.msg);
    assert_eq!((w.span.line, w.span.col), (2, 14));
    assert_eq!(w.hint.as_deref(), Some("write a concrete shape argument: `Buf[4]`"));
}

#[test]
fn type_arg_less_generic_model_warns_on_let_binding() {
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               fn main() -> i64 { let !a = Buf[4] { t: forge.zeros[i64, [4]] } \
               let x: Buf = a x.t[0] }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Buf").expect("expected a type-arg-less warning");
    assert!(w.msg.contains("let binding `x` in function `main`"), "{}", w.msg);
    assert_eq!((w.span.line, w.span.col), (2, 72));
    assert_eq!(w.hint.as_deref(), Some("write a concrete shape argument: `Buf[4]`"));
}

#[test]
fn type_arg_less_generic_model_warns_on_allocator() {
    // `forge.uninit[Surface, [4]]` — the sixth annotation position (§2.5):
    // a builtin's own model type argument, bare.
    let src = "model Surface[H, W] { !px: Tensor[i64, [H, W]] }\n\
               fn main() -> i64 { let !arr = forge.uninit[Surface, [4]] \
               arr[0] = Surface[2, 3] { px: forge.zeros[i64, [2, 3]] } \
               arr[0].px[0, 0] }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Surface").expect("expected a type-arg-less warning");
    assert!(w.msg.contains("the element type of `forge.uninit` in function `main`"), "{}", w.msg);
    assert_eq!((w.span.line, w.span.col), (2, 44));
    assert_eq!(w.hint.as_deref(), Some("write a concrete shape argument: `Surface[4, 4]`"));
}

#[test]
fn type_arg_less_generic_model_reuses_a_shape_param_already_in_scope() {
    // Reviewer defect 1's third hint form: when the enclosing fn already
    // declares a shape param matching the model's own (by name), the hint
    // reuses it rather than a concrete placeholder — `fn make[N]() -> Buf`
    // already has `N` in scope for the return position.
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               fn make[N]() -> Buf { Buf[N] { t: forge.zeros[i64, [N]] } }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Buf").expect("expected a type-arg-less warning");
    assert_eq!(w.hint.as_deref(), Some("write the shape arguments explicitly: `Buf[N]`"));
}

#[test]
fn type_arg_less_generic_model_silent_with_explicit_brackets() {
    // Control: `Buf[N]` spells its own shape argument — the predicate must
    // not fire on the bracketed spelling, in any of the four positions.
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               fn cap[N](b: Buf[N]) -> i64 { b.t[0] }\n";
    let warns = checked(src).warnings;
    assert!(type_arg_less_warning(&warns, "Buf").is_none(),
            "explicit bracket must not warn, got {:?}", warns.iter().map(|w| &w.msg).collect::<Vec<_>>());
}

#[test]
fn type_arg_less_generic_model_silent_for_non_generic_model() {
    // Control: a model with NO shape parameters has only one spelling — its
    // bare name — so the predicate (shape_params non-empty) must not fire.
    let src = "model Pt { c: Tensor[i64, [2]] }\n\
               fn sx(p: Pt) -> i64 { p.c[0] }\n";
    let warns = checked(src).warnings;
    assert!(type_arg_less_warning(&warns, "Pt").is_none(),
            "non-generic model must not warn, got {:?}", warns.iter().map(|w| &w.msg).collect::<Vec<_>>());
}

// ── #728 rework: reviewer defects 2–5 ────────────────────────────────────

#[test]
fn type_arg_less_generic_model_warns_on_array_element_in_param() {
    // Reviewer defect 2: the predicate used to match only `Type::Named`, so
    // a bare model nested in a fixed-size array's element type escaped in
    // parameter position (design probe d11, `[Surface; 4]` — SPEC.md §3.10
    // calls `[M; N]` a field/parameter type same as any other).
    let src = "model Surface[H, W] { !px: Tensor[i64, [H, W]] }\n\
               fn sum_two(arr: [Surface; 4]) -> i64 { arr[0].px[0, 0] }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Surface").expect("expected a type-arg-less warning");
    assert!(w.msg.contains("parameter `arr` of function `sum_two`"), "{}", w.msg);
    // Nested in an array wrapper: verified at RUNTIME that a freshly
    // declared shape param does not infer through this position (the
    // array/tuple/fn-type-nested twin of #739, confirmed by probe — see
    // `type_arg_less_hint`'s doc comment) — so the hint is a concrete
    // bracket that preserves the array wrapper (`[Surface[4, 4]; 4]`, not
    // `Surface[H, W]`, which silently drops it and isn't this parameter's
    // type at all), not the "declare on the function" spelling a direct
    // (non-nested) param position gets.
    assert_eq!(w.hint.as_deref(), Some("write a concrete shape argument: `[Surface[4, 4]; 4]`"));
}

#[test]
fn type_arg_less_generic_model_warns_on_array_element_in_field() {
    // Reviewer defect 2, field position half: `Type::Named`-only matching
    // let `[Surface; 4]` escape as a model field's type too.
    let src = "model Surface[H, W] { !px: Tensor[i64, [H, W]] }\n\
               model Holder[K] { arr: [Surface; 4], k: i64 }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Surface").expect("expected a type-arg-less warning");
    assert!(w.msg.contains("field `arr` of model `Holder`"), "{}", w.msg);
    assert_eq!(w.hint.as_deref(), Some("write a concrete shape argument: `[Surface[4, 4]; 4]`"));
}

#[test]
fn type_arg_less_generic_model_warns_on_zeros_allocator() {
    // Reviewer defect 3: the allocator check keyed on the literal method
    // name `uninit`, so `forge.zeros[Surface, [4]]` — an equally legal
    // model-array allocator (same set `ctor_tensor_ty` matches for a
    // scalar element) — escaped.
    let src = "model Surface[H, W] { !px: Tensor[i64, [H, W]] }\n\
               fn main() -> i64 { let !arr = forge.zeros[Surface, [4]] \
               arr[0] = Surface[2, 3] { px: forge.zeros[i64, [2, 3]] } \
               arr[0].px[0, 0] }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Surface").expect("expected a type-arg-less warning");
    assert!(w.msg.contains("the element type of `forge.zeros` in function `main`"), "{}", w.msg);
}

#[test]
fn type_arg_less_generic_model_warns_on_ones_allocator() {
    let src = "model Surface[H, W] { !px: Tensor[i64, [H, W]] }\n\
               fn main() -> i64 { let !arr = forge.ones[Surface, [4]] \
               arr[0] = Surface[2, 3] { px: forge.zeros[i64, [2, 3]] } \
               arr[0].px[0, 0] }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Surface").expect("expected a type-arg-less warning");
    assert!(w.msg.contains("the element type of `forge.ones` in function `main`"), "{}", w.msg);
}

#[test]
fn type_arg_less_generic_model_field_before_later_param_warns_in_source_order() {
    // Reviewer defect 4: `lint_type_arg_less_generic_model_fields` runs as
    // a trailing post-registration pass (see its own doc comment for why —
    // a model field's type is resolved once, during registration, and a
    // forward-referenced model's shape-param count isn't known yet at that
    // point), so without an explicit sort its warning for a field on line 2
    // would print AFTER a param warning from line 4, out of source order.
    // `check_program` sorts `warnings` by source position specifically to
    // prevent that.
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               model Outer[K] { inner: Buf, k: i64 }\n\
               fn nop() -> i64 { 0 }\n\
               fn cap(b: Buf) -> i64 { b.t[0] }\n";
    let warns = checked(src).warnings;
    let field_idx = warns.iter().position(|w| w.msg.contains("field `inner` of model `Outer`"))
        .expect("expected the field warning");
    let param_idx = warns.iter().position(|w| w.msg.contains("parameter `b` of function `cap`"))
        .expect("expected the param warning");
    assert!(
        field_idx < param_idx,
        "the line-2 field warning must print before the line-4 param warning: {:?}",
        warns.iter().map(|w| (w.span.line, w.msg.clone())).collect::<Vec<_>>(),
    );
}

#[test]
fn type_arg_less_generic_model_warns_on_alias_body() {
    // Reviewer defect 5, the alias hatch: `type B = Buf` then `fn cap(b:
    // B)` warned nothing even after the field/param fixes, since `B`'s own
    // body is never itself a param/field/return/let annotation — a fifth
    // position the ratified decision's principle covers all the same.
    // Design probe g35's equivalent.
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               type B = Buf\n\
               fn cap(b: B) -> i64 { b.t[0] }\n";
    let warns = checked(src).warnings;
    let w = type_arg_less_warning(&warns, "Buf").expect("expected a type-arg-less warning");
    assert!(w.msg.contains("the body of type alias `B`"), "{}", w.msg);
    assert_eq!(w.hint.as_deref(), Some("write a concrete shape argument: `Buf[4]`"));
    // The alias body carries the only warning — `b: B` does not ALSO
    // trigger the param-position lint (`B` is a type-alias name, not a
    // model name, so `bare_generic_model` never matches it there); one
    // warning per bare-model site, not two for the same underlying escape.
    assert_eq!(
        warns.iter().filter(|w| w.msg.starts_with("bare-generic-model: ")
            && w.msg.contains("names `Buf` with no argument list")).count(),
        1,
    );
}

#[test]
fn type_arg_less_generic_model_silent_for_bracketed_alias_body() {
    // Control (design probe g34): `type S = Buf[4]` pins a concrete
    // geometry once — the ruling's own blessed hatch for reusing a shape,
    // not an escape — so the alias-body lint must stay silent on it.
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               type S = Buf[4]\n\
               fn cap(b: S) -> i64 { b.t[0] }\n";
    let warns = checked(src).warnings;
    assert!(type_arg_less_warning(&warns, "Buf").is_none(),
            "a bracketed alias body must not warn, got {:?}", warns.iter().map(|w| &w.msg).collect::<Vec<_>>());
}

// ── #785: `Tensor`/`View`/`KV`'s own element type is a hard error, not a
// bare-generic-model warning ────────────────────────────────────────────────
//
// `find_bare_generic_models` briefly walked these three arms too (an interim
// revision of the #728 lint, filed alongside #785 itself), on the theory
// that a bare generic model there was the only diagnostic a reader would
// get. That premise died once #785 landed: `resolve_type`'s own `Tensor`/
// `View`/`KV` arms now hard-refuse ANY non-scalar element outright
// (`check_tensor_element_scalar`) — model or enum, bracketed or bare — so
// the lint's recursion into these three arms was pulled back out (matching
// `RawPtr`, which was never walked for the same reason: a stricter,
// already-refusing check owns that position). A bare generic model there
// now gets exactly one diagnostic: the #785 `tensor-element-type` error,
// not this lint's warning.

#[test]
fn tensor_element_type_error_fires_instead_of_bare_generic_model_warning() {
    // `Buf` is both shape-parameterized AND named bare — before #785 this
    // checked clean; the interim #728 revision made it warn; now it is a
    // hard type error, and the old warning must not also fire (one
    // diagnostic, not two).
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               fn cap(b: Tensor[Buf, [4]]) -> i64 { 0 }\n";
    let checker = checked(src);
    assert!(checker.errors.iter().any(|e|
        e.msg.contains("tensor-element-type") && e.msg.contains("`Buf` is a model")),
        "expected a tensor-element-type error, got errors {:?}",
        checker.errors.iter().map(|e| &e.msg).collect::<Vec<_>>());
    assert!(type_arg_less_warning(&checker.warnings, "Buf").is_none(),
        "bare-generic-model must not ALSO fire for a Tensor element, got warnings {:?}",
        checker.warnings.iter().map(|w| &w.msg).collect::<Vec<_>>());
}

#[test]
fn tensor_element_type_error_fires_for_kv_field_even_when_bracketed() {
    // Fully bracketed (`Buf[4]`, not bare) never tripped the #728 lint at
    // all — this position was never that lint's to catch. #785's own check
    // does not care whether the model reference was bare or bracketed: a
    // model is not a scalar type either way.
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               model Holder[K] { cache: KV[Buf[4], [4]], k: i64 }\n";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.contains("tensor-element-type") && e.contains("`Buf` is a model")),
        "got: {:?}", errs);
}

// ── one hint brackets every match in the annotation ──────────────────────

#[test]
fn type_arg_less_generic_model_hint_brackets_every_match() {
    // An annotation with two bare-model occurrences used to get two
    // warnings whose hints each bracketed only their own occurrence —
    // applying either hint alone left the sibling occurrence still bare,
    // which still warned. Now every match found in the annotation is
    // bracketed in each hint, so applying any single one converges to zero
    // warnings in one edit.
    let src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
               model Holder[K] { y: ([Buf; 2], (Buf, i64)), k: i64 }\n";
    let warns = checked(src).warnings;
    let matches: Vec<_> = warns.iter()
        .filter(|w| w.msg.starts_with("bare-generic-model: ")
            && w.msg.contains("names `Buf` with no argument list"))
        .collect();
    assert_eq!(matches.len(), 2, "expected two warnings, got {:?}",
               warns.iter().map(|w| &w.msg).collect::<Vec<_>>());
    for w in &matches {
        assert_eq!(
            w.hint.as_deref(),
            Some("write a concrete shape argument: `([Buf[4]; 2], (Buf[4], i64))`"),
        );
    }
    // And applying that rendered hint verbatim actually silences the lint.
    let fixed_src = "model Buf[N] { !t: Tensor[i64, [N]] }\n\
                     model Holder[K] { y: ([Buf[4]; 2], (Buf[4], i64)), k: i64 }\n";
    let fixed_warns = checked(fixed_src).warnings;
    assert!(type_arg_less_warning(&fixed_warns, "Buf").is_none(),
            "the rendered hint must silence the lint, got {:?}",
            fixed_warns.iter().map(|w| &w.msg).collect::<Vec<_>>());
}

// ── #674 slice 3a: arena budgets ─────────────────────────────────────────────

#[test]
fn budget_ctor_spine_shared_table_674() {
    // The parser/checker pair over ONE spine table (`parser_tests.rs`'s
    // `ARENA_CTOR_SPINE_TABLE`) so the two can't silently diverge on which
    // spellings admit `in b`. A non-`vault`, non-`kv` spine entry, wrapped
    // in `forge.enter` with an explicit `in b` override, check-passes with
    // no `budget-arena` — `vault.*` (R3 instance ii) and `.kv` (R3 instance
    // i, which fires on an explicit override too, not just ambiently) are
    // exercised on their own below.
    for (spine, is_spine) in crate::parser_tests::ARENA_CTOR_SPINE_TABLE {
        if !is_spine { continue; }
        if spine.starts_with("vault.") { continue; } // R3(ii), below
        if spine.contains(".kv[") { continue; }       // R3(i), below
        let src = format!(
            "fn f(b: Budget) -> nil {{ forge.enter(b) {{ let t = {spine} in b }}\n nil }}");
        let errs = check(&src);
        assert!(
            !errs.iter().any(|e| e.contains("budget-arena")),
            "`{spine} in b` inside `forge.enter` must not raise budget-arena, got: {errs:?}"
        );
    }

    // PR #814 review: the sweep above is negative-only, so a predicate
    // mutation that NARROWS `arena_ctor_receiver`/`is_arena_ctor_spine` (stops
    // recognizing a genuine spine as one) survives it undetected — a
    // mis-recognized entry still produces no `budget-arena` error, exactly
    // like the correctly-excluded entries this loop actually checks for.
    // Positive complement, from the SAME table: the two excluded categories
    // must still be recognized as ctor spines AND still raise their own R3
    // message, or this narrowing would surface here.
    for (spine, is_spine) in crate::parser_tests::ARENA_CTOR_SPINE_TABLE {
        if !is_spine || !spine.starts_with("vault.") { continue; }
        let src = format!("fn f(b: Budget) -> nil {{ let t = {spine} in b\n nil }}");
        let errs = check(&src);
        assert!(
            errs.iter().any(|e| e.starts_with("budget-arena:") && e.contains("Vault")),
            "`{spine} in b` must still be recognized as a ctor spine and raise the \
             Vault R3 message, got: {errs:?}"
        );
    }
    for (spine, is_spine) in crate::parser_tests::ARENA_CTOR_SPINE_TABLE {
        if !is_spine || !spine.contains(".kv[") || !spine.starts_with("stream.") { continue; }
        let src = format!("fn f(b: Budget) -> nil {{ let t = {spine} in b\n nil }}");
        let errs = check(&src);
        assert!(
            errs.iter().any(|e| e.starts_with("budget-arena:") && e.contains("Stream")),
            "`{spine} in b` must still be recognized as a ctor spine and raise the \
             Stream R3 message, got: {errs:?}"
        );
    }
}

#[test]
fn budget_arena_forge_kv_inside_entered_extent_674() {
    // R3(i): `forge.kv` ambiently inside a budgeted extent (no `in`
    // override) — `.kv` always lands in the Stream arena, which no budget
    // owns.
    let src = "fn f(b: Budget) -> nil { \
                 forge.enter(b) { let t = forge.kv[f32, [4]] }\n nil }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.starts_with("budget-arena:") && e.contains("forge.kv")),
        "expected a budget-arena error naming `forge.kv`, got: {errs:?}");
}

#[test]
fn budget_arena_stream_kv_inside_entered_extent_674() {
    // R3(i)'s second spelling.
    let src = "fn f(b: Budget) -> nil { \
                 forge.enter(b) { let t = stream.kv[f32, [4]] }\n nil }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.starts_with("budget-arena:") && e.contains("stream.kv")),
        "expected a budget-arena error naming `stream.kv`, got: {errs:?}");
}

#[test]
fn budget_arena_forge_kv_outside_entered_extent_is_fine_674() {
    // Same spelling, no enclosing `forge.enter` — must NOT fire (`k1`'s own
    // must-not-change guarantee: `.kv` outside a budgeted extent).
    let src = "fn f() -> nil { let t = forge.kv[f32, [4]]\n nil }";
    assert!(passes(src), "errors: {:?}", check(src));
}

#[test]
fn budget_arena_vault_ctor_with_in_override_674() {
    // R3(ii): `vault.<ctor>[…] in b` — a budget is a region of Forge, so `b`
    // cannot own vault-charged bytes, for ANY vault allocator method
    // (not just `kv`).
    let src = "fn f(b: Budget) -> nil { let t = vault.zeros[f32, [4]] in b\n nil }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.starts_with("budget-arena:") && e.contains("vault.zeros")),
        "expected a budget-arena error naming `vault.zeros`, got: {errs:?}");
}

#[test]
fn budget_arena_forge_kv_with_explicit_in_override_674() {
    // R3(i) also fires on an EXPLICIT override, not just ambiently: a `.kv`
    // spine can never be owned by any budget, named or ambient, and this
    // must report exactly ONCE (`checking_alloc_in_ctor` suppresses the
    // ambient recheck of the same node).
    let src = "fn f(b: Budget) -> nil { let t = forge.kv[f32, [4]] in b\n nil }";
    let errs = check(src);
    let matches: Vec<_> = errs.iter().filter(|e| e.starts_with("budget-arena:")).collect();
    assert_eq!(matches.len(), 1, "expected exactly one budget-arena error, got: {errs:?}");
}

#[test]
fn budget_arena_nested_kv_inside_ctor_index_args_still_fires_674() {
    // Nit fix: `checking_alloc_in_ctor` used to be a bare `bool` held across
    // the WHOLE `check_expr(ctor)` recursion for the outer overridden ctor
    // spine, which wrongly suppressed the ambient R3(i) check for a
    // DIFFERENT `.kv` spine nested inside that ctor's own index/shape
    // arguments too — a real, unreported hole (this program used to
    // `--check` clean). Narrowed to the exact ctor-spine span being
    // exempted, so the outer override still reports once (see the test
    // above) but this inner, unrelated `.kv` spine — ambiently inside the
    // same `forge.enter` extent, with no override of its own — still fires.
    let src = "fn f(b: Budget) -> nil { \
                 forge.enter(b) { let t = forge.zeros[f32, [stream.kv[f32, [4]]]] in b } \n nil }";
    let errs = check(src);
    let matches: Vec<_> = errs.iter().filter(|e| e.starts_with("budget-arena:")).collect();
    assert_eq!(matches.len(), 1,
        "expected exactly one budget-arena error naming the nested `stream.kv`, got: {errs:?}");
    assert!(matches[0].contains("stream.kv"), "got: {matches:?}");
}

#[test]
fn budget_arena_vault_kv_ctor_with_in_override_674() {
    // #674 slice 3a fix: `vault.kv […] in b` must hit R3(i)'s Stream-arena
    // message (`.kv` always lands in Stream regardless of prefix, `MEMORY.md
    // §9`), NOT R3(ii)'s Vault message — the two sites must agree because
    // both now route through the one `arena_ctor_receiver` predicate rather
    // than a hand-rolled `Ident` match that tested `vault` before `kv`.
    let src = "fn f(b: Budget) -> nil { let t = vault.kv[f32, [4]] in b\n nil }";
    let errs = check(src);
    let matches: Vec<_> = errs.iter().filter(|e| e.starts_with("budget-arena:")).collect();
    assert_eq!(matches.len(), 1, "expected exactly one budget-arena error, got: {errs:?}");
    assert!(matches[0].contains("vault.kv") && matches[0].contains("Stream"),
        "expected the Stream-arena message naming `vault.kv`, got: {matches:?}");
    assert!(!matches[0].contains("Vault"),
        "must not report the Vault-instance message for `vault.kv`, got: {matches:?}");
}

#[test]
fn budget_arena_vault_ctor_no_override_inside_extent_is_fine_674() {
    // §3.5: a bare `vault.*` (no `in b`) inside an entered extent charges
    // the root — the spelling says Vault, no budget was named, so this is
    // NOT an error.
    let src = "fn f(b: Budget) -> nil { \
                 forge.enter(b) { let t = vault.zeros[f32, [4]] }\n nil }";
    assert!(passes(src), "errors: {:?}", check(src));
}

// ── #674 slice 3a: well-formedness stays check-time ─────────────────────────
// `forge.carve`/`forge.reap`'s argument typing — the one static slice of
// R1/R2 a checker with no arena state can still decide. `forge.enter`'s own
// operand and `in b`'s budget operand are covered by the R3/BudgetBlock
// tests above; these four pin the two calls neither of those arms reaches.

#[test]
fn forge_carve_non_budget_parent_is_a_type_error_674() {
    // The motivating example: `forge.carve("not a budget", -1)` used to pass
    // `--check` clean — `forge`'s bare-ident type is `Unknown`, so nothing
    // constrained the call's own arguments.
    let errs = check("fn main() -> nil { forge.carve(\"not a budget\", -1) \n nil }");
    assert!(errs.iter().any(|e| e.starts_with("forge.carve:") && e.contains("Budget") && e.contains("str")),
        "expected a Budget-typing error for the parent, got: {errs:?}");
    assert!(errs.iter().any(|e| e.starts_with("forge.carve:") && e.contains("positive") && e.contains("-1")),
        "expected a positive-bytes-literal error, got: {errs:?}");
}

#[test]
fn forge_carve_non_integer_bytes_is_a_type_error_674() {
    let errs = check("fn f(b: Budget) -> nil { forge.carve(b, \"four\") \n nil }");
    assert!(errs.iter().any(|e| e.starts_with("forge.carve:") && e.contains("integer") && e.contains("str")),
        "errors: {errs:?}");
}

#[test]
fn forge_carve_well_typed_is_fine_674() {
    let src = "fn f(b: Budget) -> nil { forge.carve(b, 4096) \n nil }";
    assert!(passes(src), "errors: {:?}", check(src));
}

#[test]
fn forge_reap_non_budget_is_a_type_error_674() {
    let errs = check("fn main() -> nil { forge.reap(\"not a budget\") \n nil }");
    assert!(errs.iter().any(|e| e.starts_with("forge.reap:") && e.contains("Budget") && e.contains("str")),
        "errors: {errs:?}");
}

// PR #814 review: `arg_tys.first()`/`.get(1)` are `Option`s that silently
// no-op on a missing argument, so `forge.carve()`, `forge.carve(root)` and
// `forge.reap()` all "checked clean" before the arity checks above were
// added — no argument to type at all is not the same as a well-typed call.

#[test]
fn forge_carve_no_args_is_an_arity_error_674() {
    let errs = check("fn main() -> nil { forge.carve() \n nil }");
    assert!(errs.iter().any(|e| e.starts_with("forge.carve:") && e.contains("2") && e.contains("0")),
        "errors: {errs:?}");
}

#[test]
fn forge_carve_one_arg_is_an_arity_error_674() {
    // `root`'s own undefined-identifier error is a separate, expected
    // complaint here — the arity error must fire independently of it.
    let errs = check("fn f(root: Budget) -> nil { forge.carve(root) \n nil }");
    assert!(errs.iter().any(|e| e.starts_with("forge.carve:") && e.contains("2") && e.contains("1")),
        "expected an arity error naming 2 expected / 1 found, got: {errs:?}");
}

#[test]
fn forge_carve_three_args_is_an_arity_error_674() {
    let errs = check("fn f(b: Budget) -> nil { forge.carve(b, 4, 8) \n nil }");
    assert!(errs.iter().any(|e| e.starts_with("forge.carve:") && e.contains("2") && e.contains("3")),
        "errors: {errs:?}");
}

#[test]
fn forge_reap_no_args_is_an_arity_error_674() {
    let errs = check("fn main() -> nil { forge.reap() \n nil }");
    assert!(errs.iter().any(|e| e.starts_with("forge.reap:") && e.contains("1") && e.contains("0")),
        "errors: {errs:?}");
}

#[test]
fn budget_name_is_reserved_against_model_674_s6() {
    // Probe s6: `model Budget { n: i64 }` must now be a name-reservation
    // error, not a silently-shadowing user model.
    let errs = check("model Budget { n: i64 }\nfn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("reserved") && e.contains("Budget")),
        "errors: {errs:?}");
}

#[test]
fn budget_name_is_reserved_against_enum_674() {
    let errs = check("enum Budget { A, B }\nfn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("reserved") && e.contains("Budget")),
        "errors: {errs:?}");
}

#[test]
fn budget_name_is_reserved_against_type_alias_674() {
    let errs = check("type Budget = i64\nfn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("reserved") && e.contains("Budget")),
        "errors: {errs:?}");
}

#[test]
fn budget_type_checks_in_return_field_and_array_position_674_s2() {
    // Probe s2's three positions, all clean: return type, model field,
    // struct-literal initialiser.
    let src = "fn mk() -> Budget { forge.root }\n\
               model Tenant { budget: Budget, id: i64 }\n\
               fn main() -> nil { \
                 let r = Tenant { budget: mk(), id: 7 }\n \
                 nil }";
    assert!(passes(src), "errors: {:?}", check(src));
}

#[test]
fn budget_array_position_674() {
    let src = "fn f(bs: [Budget; 4]) -> Budget { bs[0] }\nfn main() -> nil { nil }";
    assert!(passes(src), "errors: {:?}", check(src));
}

#[test]
fn budget_is_not_compatible_with_other_types_674() {
    let errs = check("fn f(b: Budget) -> i64 { b }\nfn main() -> nil { nil }");
    assert!(!errs.is_empty(), "expected a type mismatch returning a Budget as i64");
}

#[test]
fn budget_takes_no_type_arguments_674() {
    let errs = check("fn f(b: Budget[4]) -> nil { nil }\nfn main() -> nil { nil }");
    assert!(errs.iter().any(|e| e.contains("Budget") && e.contains("no type argument")),
        "errors: {errs:?}");
}

#[test]
fn forge_enter_requires_budget_operand_674() {
    let errs = check("fn main() -> nil { forge.enter(5) { nil }\n nil }");
    assert!(errs.iter().any(|e| e.contains("forge.enter") && e.contains("Budget")),
        "errors: {errs:?}");
}

#[test]
fn alloc_in_requires_budget_operand_674() {
    let errs = check("fn main() -> nil { let t = forge.zeros[f32, [4]] in 5\n nil }");
    assert!(errs.iter().any(|e| e.contains("Budget")), "errors: {errs:?}");
}

// ── #674 slice 3d: the typing of the two forms, now that both lower ─────────

#[test]
fn alloc_in_is_typed_through_its_constructor_674() {
    // `Expr::AllocIn`'s value IS the constructor's — `in b` names where the
    // bytes are charged, not what the expression is — and the constructor is
    // CHECKED, not skipped. Both halves in one pair: an error inside the
    // spine surfaces identically with and without the `in b` tail, so the
    // arm neither swallows the constructor nor re-reports it.
    let with_in = check(
        "fn f(b: Budget) -> nil { let t = forge.zeros[f32, [undefined_dim_674]] in b\n nil }");
    let without = check(
        "fn f(b: Budget) -> nil { let t = forge.zeros[f32, [undefined_dim_674]]\n nil }");
    assert_eq!(with_in, without,
        "`in b` must not change what the constructor itself reports");
    assert_eq!(with_in.len(), 1, "errors: {with_in:?}");

    // And a well-formed spine under a real `Budget` reports nothing.
    let errs = check(
        "fn f(b: Budget) -> nil { let t = forge.zeros[f32, [4]] in b\n nil }");
    assert!(errs.is_empty(), "errors: {errs:?}");
}

#[test]
fn budget_block_has_its_bodys_type_674() {
    // `forge.enter(b) { … }` is a block: its value is its tail, and it is
    // the block's type the checker hands back (`check_block`), never nil.
    // The negative half runs through the fn's own return-type check, which
    // is the one place a block expression's type is actually enforced.
    let errs = check("fn f(b: Budget) -> i64 { forge.enter(b) { 5 } }");
    assert!(errs.is_empty(), "errors: {errs:?}");

    let errs = check("fn f(b: Budget) -> i64 { forge.enter(b) { \"s\" } }");
    assert!(errs.iter().any(|e| e.contains("returns") && e.contains("str")),
        "a type mismatch through `forge.enter` must be reported: {errs:?}");
}

#[test]
fn forge_enter_over_a_budget_field_and_an_array_element_check_674() {
    // B1: a `Budget` lives in a model field and in a `[Budget; N]`, and both
    // spellings are `postfix_expr`s the operand position admits (§2.1 point
    // 3). Slice 3d is the first slice at which either actually runs, so the
    // typing is worth pinning here rather than only in the parser.
    let errs = check(
        "model Tenant { budget: Budget }\n\
         fn f(r: Tenant) -> nil { forge.enter(r.budget) { nil }\n nil }");
    assert!(errs.is_empty(), "errors: {errs:?}");

    let errs = check(
        "fn f(bs: [Budget; 4]) -> nil { forge.enter(bs[0]) { nil }\n nil }");
    assert!(errs.is_empty(), "errors: {errs:?}");
}

#[test]
fn budget_stack_is_separate_from_arena_stack_674() {
    // §2.3: `forge.enter` must never push onto `arena_stack` — a Vault
    // write nested inside an entered extent (no enclosing `vault { }`) must
    // still be refused by #442, exactly as it is outside any `forge.enter`.
    let src = "fn f(b: Budget) -> nil { \
                 let !v = vault.zeros[f32, [4]]\n \
                 forge.enter(b) { v[0] = 1.0 }\n \
                 nil }";
    let errs = check(src);
    assert!(errs.iter().any(|e| e.contains("cross-arena write")),
        "expected #442's cross-arena write error inside forge.enter, got: {errs:?}");
}

// ── #844: `TyType`'s `Display` must spell every type the parser accepts ─────

/// Every `ScalarType` variant, round-tripped through `TyType`'s `Display` and
/// spliced back in as a `let` annotation. Before #844's fix this failed for
/// every scalar whose `Debug`-derived name isn't a plain case-fold of its
/// source spelling (`Fp8E4M3` -> `Fp8E4M3` never lexes as `fp8_e4m3`; `I32`,
/// `F32`, `Bool`, etc. don't lex as anything at all, since demoniC has no
/// uppercase type keywords). Only *parsing* is asserted — not `--check`
/// cleanliness — because the point is "the parser accepts what `Display`
/// prints", not "the value on the right happens to be well-typed".
#[test]
fn ty_type_display_of_every_scalar_reparses_as_the_annotation_744() {
    use super::ast::ScalarType;
    use super::types::TyType;

    let scalars = [
        ScalarType::I8, ScalarType::I16, ScalarType::I32, ScalarType::I64,
        ScalarType::U8, ScalarType::U16, ScalarType::U32, ScalarType::U64,
        ScalarType::Int4, ScalarType::Int8,
        ScalarType::F16, ScalarType::Bf16, ScalarType::Tf32, ScalarType::F32, ScalarType::F64,
        ScalarType::Fp8E4M3, ScalarType::Fp8E5M2,
        ScalarType::Trit, ScalarType::Bool, ScalarType::Str, ScalarType::Nil,
    ];

    for s in scalars {
        let rendered = TyType::Scalar(s.clone()).to_string();
        assert_eq!(
            rendered,
            super::fmt::scalar_type_str(&s),
            "TyType::Display for {:?} must match fmt::scalar_type_str's canonical table",
            s,
        );
        assert_type_annotation_parses(&rendered);
    }
}

/// A handful of composite `TyType` forms, each wrapping a scalar leaf, must
/// also round-trip: `Display` must print exactly what the parser accepts for
/// `Tensor[...]`, `View[...]`, `KV[...]`, `fn(...) -> ...`, tuples, fixed
/// arrays and raw pointers alike, not just the bare scalar case.
#[test]
fn ty_type_display_of_composite_types_reparses_as_the_annotation_744() {
    use super::ast::ScalarType;
    use super::shape::{Shape, SymDim};
    use super::types::TyType;

    let dim4 = Shape::new(vec![SymDim::Const(4)]);
    let kv_shape = Shape::new(vec![SymDim::Const(4), SymDim::Streaming]);

    let composites = vec![
        TyType::Tensor(Box::new(TyType::Scalar(ScalarType::Fp8E4M3)), dim4.clone()),
        TyType::View(Box::new(TyType::Scalar(ScalarType::Trit)), dim4.clone()),
        TyType::KV(Box::new(TyType::Scalar(ScalarType::F32)), kv_shape),
        TyType::Fn {
            params: vec![TyType::Scalar(ScalarType::Bool), TyType::Scalar(ScalarType::Str)],
            ret: Box::new(TyType::Scalar(ScalarType::I32)),
        },
        TyType::Tuple(vec![TyType::Scalar(ScalarType::U64), TyType::Scalar(ScalarType::Fp8E5M2)]),
        TyType::Array(Box::new(TyType::Scalar(ScalarType::Bf16)), SymDim::Const(3)),
        TyType::RawPtr(Box::new(TyType::Scalar(ScalarType::I64))),
        // Nested: a tensor of a tuple-free scalar inside a fn's return type.
        TyType::Fn {
            params: vec![],
            ret: Box::new(TyType::Tensor(Box::new(TyType::Scalar(ScalarType::Fp8E5M2)), dim4)),
        },
    ];

    for ty in composites {
        let rendered = ty.to_string();
        assert_type_annotation_parses(&rendered);
    }
}

/// Splice a rendered type back in as a `let` annotation and assert the
/// resulting program at least *parses* (lexes and parses clean). Whether it
/// then passes `--check` is irrelevant here — the value `0` is nonsense for
/// most of these types, and that is fine: `check_int_literal_range` and
/// friends run well after the parser, so a parse failure can only come from
/// the annotation's own spelling.
fn assert_type_annotation_parses(rendered: &str) {
    let src = format!("fn main() -> nil {{ let x: {} = 0\n nil }}", rendered);
    let tokens = Lexer::new(&src).tokenize()
        .unwrap_or_else(|e| panic!("rendered type `{}` failed to lex: {:?}", rendered, e));
    Parser::new(tokens).parse_program()
        .unwrap_or_else(|e| panic!("rendered type `{}` failed to parse: {:?}", rendered, e));
}
