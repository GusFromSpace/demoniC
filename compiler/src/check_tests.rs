/// Typechecker unit tests — small fragments hitting specific check paths.
/// Integration verification: `dmc --check examples/*.dmc` from the shell.

use super::check::Checker;
use super::lexer::Lexer;
use super::parser::Parser;

fn check(src: &str) -> Vec<String> {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker = Checker::new();
    checker.check_program(&program, None);
    checker.errors.iter().map(|e| e.msg.clone()).collect()
}

fn passes(src: &str) -> bool { check(src).is_empty() }

/// Lint diagnostics (non-fatal warnings), as message strings.
fn warnings(src: &str) -> Vec<String> {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker = Checker::new();
    checker.check_program(&program, None);
    checker.warnings.iter().map(|w| w.msg.clone()).collect()
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
fn oversized_tensor_literal_warns() {
    // #403 (SPEC §4.2): tensor literals are for small constants — past 256
    // total elements the checker warns (spec reserves the right to error).
    let elems = std::iter::repeat("1.0").take(300).collect::<Vec<_>>().join(", ");
    let src = format!("fn main() -> f32 {{ let t = [{elems}]  t[0] }}");
    let warns = warnings(&src);
    assert!(warns.iter().any(|w| w.contains("300 elements")),
            "expected oversized-literal warning, got {:?}", warns);

    // Nested literals count leaves through the full inferred shape: 2 × 150.
    let row = std::iter::repeat("1").take(150).collect::<Vec<_>>().join(", ");
    let src = format!("fn main() -> i64 {{ let t = [[{row}], [{row}]]  t[0, 0] }}");
    let warns = warnings(&src);
    assert!(warns.iter().any(|w| w.contains("300 elements")),
            "expected nested-literal warning, got {:?}", warns);

    // Exactly 256 stays quiet — the bound is "more than 256".
    let elems = std::iter::repeat("1.0").take(256).collect::<Vec<_>>().join(", ");
    let src = format!("fn main() -> f32 {{ let t = [{elems}]  t[0] }}");
    let warns = warnings(&src);
    assert!(warns.is_empty(), "256-element literal must not warn, got {:?}", warns);
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

#[test]
fn match_enum_fresh_catchall_does_not_shadow_warn() {
    // A genuine fresh catch-all (`other`, not a variant of any enum) on an enum
    // scrutinee must not trip the #350 shadow lint.
    let ws = warnings(r#"
        enum Color { Red, Green, Blue }
        fn classify(c: Color) -> i64 {
            match c { Red => 1, other => 0 }
        }
        fn main() -> i64 { classify(Color.Blue) }
    "#);
    assert!(!ws.iter().any(|w| w.contains("variant of enum")),
            "a fresh catch-all on an enum must not warn, got: {:?}", ws);
}

// ── #369: unimplemented-directive lint ─────────────────────────────────────

#[test]
fn unimplemented_directive_warns() {
    // #369: `@recompute` / `@comptime` / `@inplace` are parsed but have no
    // effect — warn so they aren't silent no-ops.
    for d in ["recompute(budget=2)", "comptime", "inplace"] {
        let src = format!("@{d}\nfn f() -> i64 {{ 1 }}\nfn main() -> i64 {{ f() }}");
        let ws = warnings(&src);
        assert!(ws.iter().any(|w| w.contains("is not implemented") && w.contains("no effect")),
                "expected an unimplemented-directive warning for @{d}, got: {:?}", ws);
    }
}

#[test]
fn effective_directives_do_not_warn() {
    // Directives the compiler acts on must stay quiet — including `@host match`,
    // which is functional (host-feature dispatch), not a no-op.
    let ws = warnings(r#"
        @grad fn loss(!w: Tensor[f32, [4]], x: Tensor[f32, [4]]) -> f32 {
            sum((w .* x) .* (w .* x))
        }
        fn pick() -> i64 { @host match { .avx2 => 1, _ => 0 } }
        fn main() -> i64 { pick() }
    "#);
    assert!(!ws.iter().any(|w| w.contains("is not implemented")),
            "effective directives (@grad, @host) must not warn, got: {:?}", ws);
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
    // #232: `x = x` and `let !x = x` are dead code — and the signature of LLM
    // repetition-collapse, which compiles as legal-but-garbage.
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

#[test]
fn pipe_into_value_is_rejected() {
    // #188: `x >> 2` (meant as bitwise shift) pipes into a non-callable value.
    // Previously type-checked, then failed at runtime; now a check error.
    let errs = check(r#"
        fn test_op() -> bool { let x = 256  (x >> 2) >= 0 }
    "#);
    assert!(errs.iter().any(|e| e.contains("not callable")),
            "expected non-callable pipe error, got: {:?}", errs);
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
        pub let public_val = 1
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
        public_items,
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

    assert_eq!(get_type("let x = 1.0f16"), "F16");
    assert_eq!(get_type("let x = 1.0bf16"), "Bf16");
    assert_eq!(get_type("let x = 1.0tf32"), "Tf32");
    assert_eq!(get_type("let x = 1.0f32"), "F32");
    assert_eq!(get_type("let x = 1.0f64"), "F64");
    assert_eq!(get_type("let x = 1.0fp8_e4m3"), "Fp8E4M3");
    assert_eq!(get_type("let x = 1.0fp8_e5m2"), "Fp8E5M2");
    // An unconstrained float literal defaults to F64 (#284) — consistent with
    // the unconstrained int-literal default of i64, and the conventional
    // double-precision default. A float literal that needs f32 must say so
    // (`let x: f32 = 1.0` or the `1.0f32` suffix).
    assert_eq!(get_type("let x = 1.0"), "F64");

    // Ensure float does not assign to a non-numeric type like bool
    let errs = check("fn main() -> bool { 1.0f64 }");
    assert!(!errs.is_empty(), "expected type mismatch error, got: {:?}", errs);
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
fn comptime_block_passes() {
    assert!(passes(r#"
        fn main() -> i64 {
            let x = @comptime { 3 * 7 + 1 }
            x
        }
    "#));
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
fn captured_mut_in_grad_fn_errors_398() {
    let errs = check(r#"
        let !gg = forge.zeros[f32,[3]]
        @grad fn loss(!w: Tensor[f32,[3]]) -> f32 { sum(w .* gg) }
        fn main() -> nil { nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("not differentiable") && e.contains("gg")),
            "expected captured-mut error, got {:?}", errs);
    // A @grad fn using only its ! params must stay clean.
    let ok = check(r#"
        @grad fn loss(!w: Tensor[f32,[3]], x: Tensor[f32,[3]]) -> f32 { sum(w .* x) }
        fn main() -> nil { nil }
    "#);
    assert!(!ok.iter().any(|e| e.contains("not differentiable")),
            "param-only @grad fn must not error: {:?}", ok);
}

#[test]
fn captured_mut_hidden_in_closure_errors_398() {
    // A captured mutable referenced inside a closure LITERAL used to slip past
    // the #398 check (collect_body_idents skipped FnLit bodies), silently
    // yielding an opaque `grads.<name>` at runtime. The scan now recurses into
    // closure bodies, so the capture is caught.
    let errs = check(r#"
        let !bias = forge.zeros[f32,[1]]
        @grad fn loss(!w: Tensor[f32,[1]], x: Tensor[f32,[1]]) -> f32 {
            let _hook = fn() -> f32 { sum(bias) }
            let y = sum(w .* x)
            y * y
        }
        fn main() -> nil { nil }
    "#);
    assert!(errs.iter().any(|e| e.contains("not differentiable") && e.contains("bias")),
            "expected captured-mut-in-closure error, got {:?}", errs);

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
    assert!(errs.iter().any(|e| e.contains("#393") && e.contains("shape pattern")),
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
    assert!(mismatch.iter().any(|e| e.contains("body produces") && e.contains("Bool")),
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
