/// Parser unit tests — small fragments that hit specific grammar productions.
/// Pre-alpha; not exhaustive. Integration verification is done by parsing all
/// 12 files in /examples in CI.

use super::ast::*;
use super::lexer::Lexer;
use super::parser::Parser;

fn parse(src: &str) -> Program {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    Parser::new(tokens).parse_program().expect("parse failed")
}

fn parse_err(src: &str) -> String {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    Parser::new(tokens).parse_program().err().expect("expected parse failure").msg
}

#[test]
fn empty_program() {
    let p = parse("");
    assert!(p.items.is_empty());
}

#[test]
fn simple_fn() {
    let p = parse("fn id(x: i64) -> i64 { x }");
    assert_eq!(p.items.len(), 1);
    match &p.items[0] {
        Item::Fn(f) => {
            assert_eq!(f.name, "id");
            assert_eq!(f.params.len(), 1);
            assert_eq!(f.params[0].name, "x");
            assert!(f.body.tail_expr.is_some());
        }
        _ => panic!("expected fn"),
    }
}

#[test]
fn fn_with_shape_params() {
    let p = parse("fn f[B, D](x: Tensor[f32, [B, D]]) -> nil { nil }");
    if let Item::Fn(f) = &p.items[0] {
        assert_eq!(f.shape_params.len(), 2);
        assert_eq!(f.shape_params[0].name, "B");
    } else { panic!() }
}

#[test]
fn fn_with_directive() {
    let p = parse("@grad fn forward(x: f32) -> f32 { x }");
    if let Item::Fn(f) = &p.items[0] {
        assert_eq!(f.directives.len(), 1);
        assert_eq!(f.directives[0].name, "grad");
    } else { panic!() }
}

#[test]
fn precedence_matmul_then_dotmul() {
    // `q @ k .* s` should be `(q @ k) .* s` because matmul binds tighter than product? No —
    // per grammar, product (which includes .*) is below matmul, so it's `(q @ k) .* s`.
    let p = parse("fn t() -> nil { let _ = q @ k .* s; nil }");
    if let Item::Fn(f) = &p.items[0] {
        if let Stmt::Let(l) = &f.body.stmts[0] {
            // top-level operator should be DotMul
            match &l.value {
                Expr::BinOp { op: BinOp::DotMul, .. } => {}
                other => panic!("expected DotMul at top, got {:?}", other),
            }
        } else { panic!() }
    } else { panic!() }
}

/// #530: `>>` parses as `BinOp::BitShr` — the mirror of `<<`/`BitShl`, and
/// NOT the pipe-era `BinOp::RShift`, which ruling S16 keeps unconstructible.
#[test]
fn right_shift_parses_as_bitshr() {
    let p = parse("fn t() -> nil { let _ = a >> b; nil }");
    if let Item::Fn(f) = &p.items[0] {
        if let Stmt::Let(l) = &f.body.stmts[0] {
            match &l.value {
                Expr::BinOp { op: BinOp::BitShr, .. } => {}
                other => panic!("expected BitShr at top, got {:?}", other),
            }
        } else { panic!() }
    } else { panic!() }
}

/// `>>` sits at the bitshift level with `<<` (OPERATORS §1, row 12): looser
/// than `+`, tighter than `&`. Both neighbours are asserted, so a `>>` arm
/// added at the wrong rung — the pipe level it used to occupy, say — fails.
#[test]
fn right_shift_binds_below_sum_and_above_bitand() {
    // `a + b >> c` is `(a + b) >> c`.
    let p = parse("fn t() -> nil { let _ = a + b >> c; nil }");
    if let Item::Fn(f) = &p.items[0] {
        if let Stmt::Let(l) = &f.body.stmts[0] {
            match &l.value {
                Expr::BinOp { op: BinOp::BitShr, lhs, .. } =>
                    assert!(matches!(**lhs, Expr::BinOp { op: BinOp::Add, .. }),
                            "expected `+` under the shift, got {:?}", lhs),
                other => panic!("expected BitShr at top, got {:?}", other),
            }
        } else { panic!() }
    } else { panic!() }
    // `a & b >> c` is `a & (b >> c)`.
    let p = parse("fn t() -> nil { let _ = a & b >> c; nil }");
    if let Item::Fn(f) = &p.items[0] {
        if let Stmt::Let(l) = &f.body.stmts[0] {
            match &l.value {
                Expr::BinOp { op: BinOp::BitAnd, rhs, .. } =>
                    assert!(matches!(**rhs, Expr::BinOp { op: BinOp::BitShr, .. }),
                            "expected the shift under `&`, got {:?}", rhs),
                other => panic!("expected BitAnd at top, got {:?}", other),
            }
        } else { panic!() }
    } else { panic!() }
}

#[test]
fn transpose_postfix() {
    let p = parse("fn t() -> nil { let _ = k'; nil }");
    if let Item::Fn(f) = &p.items[0] {
        if let Stmt::Let(l) = &f.body.stmts[0] {
            assert!(matches!(l.value, Expr::Postfix { op: PostfixOp::Transpose, .. }));
        } else { panic!() }
    } else { panic!() }
}

#[test]
fn model_decl() {
    let p = parse(r#"
        model Block[D] {
            ln: Tensor[f32, [D]]
            fn forward(self, x: Tensor[f32, [D]]) -> Tensor[f32, [D]] { x }
        }
    "#);
    if let Item::Model(m) = &p.items[0] {
        assert_eq!(m.name, "Block");
        assert_eq!(m.members.len(), 2);
    } else { panic!() }
}

#[test]
fn cast_expr() {
    let p = parse("fn t() -> nil { let _ = B as f32; nil }");
    if let Item::Fn(f) = &p.items[0] {
        if let Stmt::Let(l) = &f.body.stmts[0] {
            assert!(matches!(l.value, Expr::Cast { .. }));
        } else { panic!() }
    } else { panic!() }
}

#[test]
fn mesh_type() {
    let p = parse("fn t() -> Mesh[dp=8, tp=4] { nil }");
    if let Item::Fn(f) = &p.items[0] {
        assert!(matches!(&f.ret_type, Some(Type::Mesh(_, _))));
    } else { panic!() }
}

#[test]
fn arena_block_stmt() {
    let p = parse("fn t() -> nil { vault { let _ = 1 } nil }");
    if let Item::Fn(f) = &p.items[0] {
        assert!(f.body.stmts.len() >= 1);
    } else { panic!() }
}

#[test]
fn arena_block_expr_in_let_binding() {
    let p = parse(r#"
        model Item { !x: i64 }
        fn t() -> nil {
            let item = vault {
                Item { x: 11 }
            }
            nil
        }
    "#);
    if let Item::Fn(f) = &p.items[1] {
        if let Stmt::Let(l) = &f.body.stmts[0] {
            assert!(matches!(l.value, Expr::ArenaBlock(_)));
        } else { panic!("expected let") }
    } else { panic!("expected fn") }
}

#[test]
fn nested_arena_block_expr_in_let_binding() {
    parse(r#"
        model Item { !x: i64 }
        fn t() -> nil {
            let !item = vault {
                let i = forge { Item { x: 11 } }
                i
            }
            nil
        }
    "#);
}

#[test]
fn match_with_shape_pat() {
    let p = parse(r#"
        fn t() -> nil {
            match dims {
                [B, S] => 1,
                _ => 0,
            }
            nil
        }
    "#);
    let _ = p;
}

#[test]
fn pipe_with_underscore() {
    let p = parse("fn t() -> nil { let _ = x |> _ .+ b; nil }");
    let _ = p;
}

#[test]
fn err_unterminated_paren() {
    let e = parse_err("fn t() -> nil { let _ = (1 + 2 ; nil }");
    assert!(e.contains("expected") || e.contains("found"));
}

#[test]
fn else_on_new_line_parses() {
    // `else` on its own line (after a closing brace) must parse correctly.
    parse("fn t() -> i64 { if x > 0 {\n 1\n}\nelse {\n 2\n} }");
}

#[test]
fn else_if_on_new_line_parses() {
    parse("fn t() -> i64 { if x > 2 {\n 3\n}\nelse if x > 1 {\n 2\n}\nelse {\n 1\n} }");
}

// ── Bug regression tests (issue #42) ─────────────────────────────────────────

/// B3: `?` wildcard inside shape literals (Spec §3.2)
#[test]
fn b3_dynamic_shape_wildcard() {
    let p = parse("fn shape_of(x: Tensor[f32, [?, ?]]) -> i64 { 99 }");
    if let Item::Fn(f) = &p.items[0] {
        assert_eq!(f.name, "shape_of");
        // Verify the shape spec was parsed (type annotation exists on param)
        assert!(f.params[0].ty.is_some());
    } else {
        panic!("expected fn");
    }
}

// ── S3: `_` is not a dimension (#501) ────────────────────────────────────────

/// `_` in a *type*'s shape was a second spelling of `?`. It is gone; the
/// diagnostic names the survivor.
#[test]
fn underscore_dim_in_type_is_rejected_501() {
    let e = parse_err("fn rows(x: Tensor[f32, [_, _]]) -> i64 { 1 }");
    assert_eq!(e, "`_` is not a dimension; a dynamic dim is `?`");
}

/// The rejection is about shape position, not about `Tensor`: `View` and `KV`
/// share the production and reject it identically.
#[test]
fn underscore_dim_rejected_in_view_and_kv_501() {
    for src in [
        "fn v(x: View[f32, [_, 4]]) -> i64 { 1 }",
        "fn k(c: KV[f32, [_, ~, 4]]) -> i64 { 1 }",
    ] {
        assert_eq!(parse_err(src), "`_` is not a dimension; a dynamic dim is `?`", "for {src}");
    }
}

/// The break is about the spelling, not about first position: `_` is rejected
/// wherever it hides inside a type's shape element, not only as its lead token.
#[test]
fn underscore_dim_rejected_in_non_leading_position_501() {
    for src in [
        "fn f(x: Tensor[f32, [(_), 2]]) -> i64 { 1 }",
        "fn f(x: Tensor[f32, [1 + _, 2]]) -> i64 { 1 }",
        "fn f(x: Tensor[f32, [2, -_]]) -> i64 { 1 }",
        "fn f(x: Tensor[f32, [2 * (3 + _)]]) -> i64 { 1 }",
    ] {
        assert_eq!(parse_err(src), "`_` is not a dimension; a dynamic dim is `?`", "for {src}");
    }
}

/// …and the guard is scoped to types: the same shapes in *pattern* position are
/// a different production and keep parsing.
#[test]
fn underscore_inside_shape_pattern_expr_still_parses_501() {
    parse(r#"
        fn t[S](x: Tensor[f32, [2, S, 768]]) -> nil {
            match x.shape {
                [1 + _, S, 768] => print("ok"),
                _               => panic("drift"),
            }
            nil
        }
    "#);
}

/// The survivor still parses to the same node it always did.
#[test]
fn query_dim_still_parses_501() {
    let p = parse("fn rows(x: Tensor[f32, [?, 4]]) -> i64 { 1 }");
    if let Item::Fn(f) = &p.items[0] {
        match f.params[0].ty.as_ref().expect("param is annotated") {
            Type::Tensor(_, shape, _) => {
                assert!(matches!(shape.elems[0], ShapeElem::Wildcard(_)));
                assert!(matches!(shape.elems[1], ShapeElem::Expr(_)));
            }
            other => panic!("expected a tensor type, got {other:?}"),
        }
    } else { panic!("expected fn") }
}

/// Regression: `_` in *shape-pattern* position is a different production and is
/// untouched — `examples/slice.dmc:20` matches on `x.shape` this way.
#[test]
fn underscore_in_shape_pattern_still_parses_501() {
    let p = parse(r#"
        fn t[S](x: Tensor[f32, [2, S, 768]]) -> nil {
            match x.shape {
                [_, S, 768] => print("ok"),
                _           => panic("drift"),
            }
            nil
        }
    "#);
    if let Item::Fn(f) = &p.items[0] {
        if let Some(Stmt::Match(me)) = f.body.stmts.first() {
            match &me.arms[0].pattern {
                Pattern::Shape(elems, _) => {
                    assert!(matches!(elems[0], ShapeElem::Wildcard(_)), "leading `_` stays a wildcard");
                    assert_eq!(elems.len(), 3);
                }
                other => panic!("expected a shape pattern, got {other:?}"),
            }
        } else { panic!("expected match stmt") }
    } else { panic!("expected fn") }
}

/// Regression: the bare `_` catch-all pattern is untouched.
#[test]
fn underscore_catchall_pattern_still_parses_501() {
    let p = parse("fn t(n: i64) -> i64 { match n { 0 => 1, _ => 2 } }");
    if let Item::Fn(f) = &p.items[0] {
        if let Some(Stmt::Match(me)) = f.body.stmts.first() {
            assert!(matches!(me.arms[1].pattern, Pattern::Wildcard(_)));
        } else { panic!("expected match stmt") }
    } else { panic!("expected fn") }
}

/// Regression: `_` as an *expression* — the pipe-stage placeholder (SPEC §7.6)
/// — is untouched.
#[test]
fn underscore_expr_still_parses_501() {
    let p = parse("fn t(x: i64) -> i64 { x |> add(_, 1) }");
    let printed = crate::fmt::pretty_print_program(&p);
    assert!(printed.contains("add(_, 1)"),
            "the stage placeholder must survive parse and print, got: {printed}");
    parse(&printed);
}

/// B4: `..` rest pattern in match arms (Spec §4.5)
#[test]
fn b4_dotdot_rest_pattern_in_match() {
    // `..` parses as a distinct `Rest` pattern (not `Wildcard`), so `(a, ..)`
    // tuples are not confused with fixed-arity `(a, _)`. Standalone it still acts
    // as a catch-all — see interp/check tests — but the AST node is `Rest`.
    let p = parse(r#"
        fn t(n: i64) -> i64 {
            match n {
                0  => 100,
                .. => 999,
            }
        }
    "#);
    if let Item::Fn(f) = &p.items[0] {
        if let Some(Stmt::Match(me)) = f.body.stmts.first() {
            assert_eq!(me.arms.len(), 2);
            assert!(matches!(me.arms[1].pattern, Pattern::Rest(_)));
        } else {
            panic!("expected match stmt");
        }
    } else {
        panic!("expected fn");
    }
}

/// B2: `@host match { ... }` directive form (Spec §7.3)
#[test]
fn b2_directive_match_form() {
    let p = parse(r#"
        fn t() -> i64 {
            let chosen = @host match {
                .avx2 => 1,
                .neon => 2,
                _     => 0,
            }
            chosen
        }
    "#);
    if let Item::Fn(f) = &p.items[0] {
        // Should have a let stmt binding a DirectiveBlock containing a match
        assert!(!f.body.stmts.is_empty());
    } else {
        panic!("expected fn");
    }
}

/// B7: Unicode XID identifiers (Spec §2.3)
#[test]
fn b7_unicode_xid_identifiers() {
    // Greek letters are valid XID_Start/XID_Continue characters
    let p = parse(r#"
        fn t() -> i64 {
            let π = 3
            let θ = 4
            let δ = π + θ
            δ
        }
    "#);
    if let Item::Fn(f) = &p.items[0] {
        // 3 let stmts + tail expr
        assert!(f.body.stmts.len() >= 3);
    } else {
        panic!("expected fn");
    }
}

// ── Visibility Modifier Tests ───────────────────────────────────────────────

#[test]
fn parser_pub_modifiers() {
    let p = parse(r#"
        pub fn test_fn() -> nil { nil }
        pub model Point { x: i64 }
        pub type Custom = i64
        pub let global_val = 100
        @pp pub fn pp_fn() -> nil { nil }
        pub @pp fn pp_fn_2() -> nil { nil }
    "#);

    assert_eq!(p.items.len(), 6);
    assert!(matches!(p.items[0], Item::Pub(..)));
    assert!(matches!(p.items[1], Item::Pub(..)));
    assert!(matches!(p.items[2], Item::Pub(..)));
    assert!(matches!(p.items[3], Item::Pub(..)));

    // Item 4: @pp pub fn -> Pub wrapping Fn which contains the directive
    if let Item::Pub(inner) = &p.items[4] {
        if let Item::Fn(f) = inner.as_ref() {
            assert_eq!(f.directives.len(), 1);
            assert_eq!(f.directives[0].name, "pp");
        } else {
            panic!("expected fn");
        }
    } else {
        panic!("expected pub wrapping fn");
    }

    // Item 5: pub @pp fn -> Pub wrapping Fn which contains the directive
    if let Item::Pub(inner) = &p.items[5] {
        if let Item::Fn(f) = inner.as_ref() {
            assert_eq!(f.directives.len(), 1);
            assert_eq!(f.directives[0].name, "pp");
        } else {
            panic!("expected fn");
        }
    } else {
        panic!("expected pub wrapping fn");
    }
}

#[test]
fn parser_pub_arena_errors() {
    let e = parse_err("pub vault { let x = 1 }");
    assert!(e.contains("visibility modifier not allowed on arena blocks"));

    let e = parse_err("pub forge { let x = 1 }");
    assert!(e.contains("visibility modifier not allowed on arena blocks"));

    let e = parse_err("pub stream { let x = 1 }");
    assert!(e.contains("visibility modifier not allowed on arena blocks"));
}

#[test]
fn parser_pub_use_errors() {
    let e = parse_err("pub use \"other.dmc\"");
    assert!(e.contains("visibility modifier not allowed on use statements"));
}

// -- Char literal parser tests

#[test]
fn parser_char_lit_ascii() {
    let p = parse(r#"fn main() -> u32 { c"A" }"#);
    if let Item::Fn(f) = &p.items[0] {
        if let Some(Expr::Literal(Literal::Char('A'), _)) = &f.body.tail_expr.as_deref() {
            return;
        }
    }
    panic!("expected Literal::Char('A')");
}

#[test]
fn parser_char_lit_escape() {
    let p = parse(r#"fn main() -> u32 { c"\n" }"#);
    if let Item::Fn(f) = &p.items[0] {
        if let Some(Expr::Literal(Literal::Char('\n'), _)) = &f.body.tail_expr.as_deref() {
            return;
        }
    }
    panic!("expected Literal::Char(newline)");
}

#[test]
fn parser_char_lit_in_binding() {
    let p = parse(r#"fn main() -> nil { let ch = c"Z"; nil }"#);
    assert_eq!(p.items.len(), 1);
}

// ── Additional parser tests ──────────────────────────────────────────────────

#[test]
fn parse_let_with_type_annotation() {
    let p = parse("fn f() -> nil { let x: i64 = 42; nil }");
    if let Item::Fn(f) = &p.items[0] {
        let stmt = &f.body.stmts[0];
        if let Stmt::Let(l) = stmt {
            assert!(l.ty.is_some());
        } else { panic!("expected let stmt"); }
    }
}

#[test]
fn parse_tuple_literal() {
    let p = parse("fn f() -> nil { let t = (1, 2, 3); nil }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_list_literal() {
    let p = parse("fn f() -> nil { let xs = list(1, 2, 3); nil }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_if_else_chain() {
    let p = parse(r#"
fn classify(n: i64) -> str {
    if n < 0 { "neg" }
    else if n == 0 { "zero" }
    else { "pos" }
}
"#);
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_for_range_loop() {
    let p = parse("fn f() -> nil { for i in 0..10 { nil } }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_while_loop() {
    let p = parse("fn f() -> nil { let x = 0; while x < 10 { x = x + 1 } }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_match_expr() {
    let p = parse(r#"
fn f(x: i64) -> str {
    match x {
        0 => "zero",
        1 => "one",
        _ => "other",
    }
}
"#);
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_lambda_expr() {
    let p = parse("fn f() -> nil { let add = fn(x: i64, y: i64) -> i64 { x + y }; nil }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_method_call_chain() {
    let p = parse(r#"fn f(s: str) -> str { s.upper().trim() }"#);
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_pipe_operator() {
    let p = parse("fn f() -> nil { let x = 5 |> to_str; nil }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_two_functions() {
    let p = parse("fn a() -> nil { nil }\nfn b() -> nil { nil }");
    assert_eq!(p.items.len(), 2);
}

#[test]
fn parse_block_as_expression() {
    let p = parse("fn f() -> i64 { let x = { 42 }; x }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_tuple_destructure() {
    let p = parse("fn f() -> nil { let (a, b) = (1, 2); nil }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_return_statement() {
    let p = parse("fn f() -> i64 { return 42 }");
    if let Item::Fn(f) = &p.items[0] {
        assert!(f.body.stmts.iter().any(|s| matches!(s, Stmt::Return { .. })));
    }
}

#[test]
fn parse_neg_int_literal() {
    let p = parse("fn f() -> i64 { -42 }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_string_literal() {
    let p = parse(r#"fn f() -> str { "hello world" }"#);
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_bool_literal() {
    let p = parse("fn f() -> bool { true }");
    if let Item::Fn(f) = &p.items[0] {
        if let Some(Expr::Literal(Literal::Bool(true), _)) = f.body.tail_expr.as_deref() {}
        else { panic!("expected bool literal"); }
    }
}

#[test]
fn parse_nil_literal() {
    let p = parse("fn f() -> nil { nil }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_binary_arithmetic() {
    let p = parse("fn f() -> i64 { 1 + 2 * 3 - 4 / 2 }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_index_expression() {
    let p = parse("fn f(t: Tensor[f32, [3, 4]]) -> nil { let _ = t[0]; nil }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn parse_use_declaration() {
    let p = parse(r#"use "other.dmc""#);
    assert!(p.items.iter().any(|item| matches!(item, Item::Use(_))));
}

#[test]
fn parse_pub_fn() {
    let p = parse("pub fn f() -> nil { nil }");
    assert!(p.items.iter().any(|item| matches!(item, Item::Pub(_))));
}

#[test]
fn parse_break_continue() {
    let p = parse("fn f() -> nil { for i in 0..10 { if i == 5 { break } else { continue } } }");
    assert_eq!(p.items.len(), 1);
}

// ── Issue #131: binary operator as continuation-line leader ─────────────────

#[test]
fn binop_continuation_line() {
    // Dot-ops and arithmetic ops can lead continuation lines (closes #131).
    // The operator must still be at the END of the previous line for `@` (matmul),
    // since `@` is also a directive prefix and cannot safely span newlines.
    let p = parse(r#"
        fn f(a: i64, b: i64, c: i64) -> i64 {
            a
            + b
            + c
        }
    "#);
    assert_eq!(p.items.len(), 1);
}

#[test]
fn dotop_continuation_line() {
    let p = parse(r#"
        fn g(a: f32, b: f32, c: f32) -> f32 {
            a
            .+ b
            .- c
        }
    "#);
    assert_eq!(p.items.len(), 1);
}

// ─── Precedence & associativity regressions (#278, #279) ─────────────────────

/// Extract the tail expression of the first fn item.
fn tail_of(p: &Program) -> &Expr {
    match &p.items[0] {
        Item::Fn(f) => f.body.tail_expr.as_ref().expect("expected tail expr"),
        _ => panic!("expected fn item"),
    }
}

#[test]
fn power_is_right_associative() {
    // #278 / OPERATORS.md §1 row 16: `a ** b ** c` is `a ** (b ** c)`.
    let p = parse("fn f(a: i64, b: i64, c: i64) -> i64 { a ** b ** c }");
    match tail_of(&p) {
        Expr::BinOp { op: BinOp::StarStar, lhs, rhs, .. } => {
            assert!(matches!(**lhs, Expr::Ident(ref n, _) if n == "a"),
                    "lhs must be the bare `a`, got: {:?}", lhs);
            assert!(matches!(**rhs, Expr::BinOp { op: BinOp::StarStar, .. }),
                    "rhs must be the nested `b ** c`, got: {:?}", rhs);
        }
        other => panic!("expected top-level `**`, got: {:?}", other),
    }
}

#[test]
fn dot_pow_is_right_associative() {
    // #278: `.^` shares row 16 with `**`.
    let p = parse("fn f(a: f32, b: f32, c: f32) -> f32 { a .^ b .^ c }");
    match tail_of(&p) {
        Expr::BinOp { op: BinOp::DotPow, rhs, .. } => {
            assert!(matches!(**rhs, Expr::BinOp { op: BinOp::DotPow, .. }),
                    "rhs must be the nested `b .^ c`, got: {:?}", rhs);
        }
        other => panic!("expected top-level `.^`, got: {:?}", other),
    }
}

#[test]
fn range_binds_looser_than_bitor() {
    // #279 / OPERATORS.md §1, GRAMMAR.ebnf: `..` (8) is looser than `|` (9),
    // so `a | b .. c | d` is `(a | b) .. (c | d)`.
    let p = parse("fn f(a: i64, b: i64, c: i64, d: i64) -> i64 { a | b .. c | d }");
    match tail_of(&p) {
        Expr::Range { start, end, inclusive: false, .. } => {
            assert!(matches!(start.as_deref(), Some(Expr::BinOp { op: BinOp::BitOr, .. })),
                    "start must be `a | b`, got: {:?}", start);
            assert!(matches!(end.as_deref(), Some(Expr::BinOp { op: BinOp::BitOr, .. })),
                    "end must be `c | d`, got: {:?}", end);
        }
        other => panic!("expected top-level range, got: {:?}", other),
    }
}

#[test]
fn range_still_binds_tighter_than_equality() {
    // #279: `==` (6) is looser than `..` (8): `a .. b == c .. d` keeps `==` on top.
    let p = parse("fn f(a: i64, b: i64, c: i64, d: i64) -> bool { (a .. b) == (c .. d) }");
    match tail_of(&p) {
        Expr::BinOp { op: BinOp::Eq, .. } => {}
        other => panic!("expected top-level `==`, got: {:?}", other),
    }
}

#[test]
fn range_over_sums_unchanged() {
    // Re-wiring the ladder must not disturb the common `lo + 1 .. hi - 1` form.
    let p = parse("fn f(lo: i64, hi: i64) -> nil { for i in lo + 1 .. hi - 1 { } nil }");
    assert_eq!(p.items.len(), 1);
}

// ── #446: wrapped signatures — `->` may begin its own line ──────────────────

#[test]
fn arrow_on_own_line_after_wrapped_params_446() {
    // Newlines are insignificant inside `( )`; the one after `)` used to end
    // the signature early, so the body's `{` read as missing.
    let p = parse("fn wide(a: i64, b: i64,\n        c: i64, d: i64)\n        -> i64 {\n    a + b + c + d\n}");
    assert_eq!(p.items.len(), 1);
    match &p.items[0] {
        Item::Fn(f) => {
            assert_eq!(f.params.len(), 4, "all four params must survive");
            assert!(f.ret_type.is_some(), "return type must be attached");
        }
        other => panic!("expected a fn item, got {:?}", other),
    }
}

#[test]
fn arrow_on_own_line_in_fn_literal_446() {
    let p = parse("fn main() -> nil { let f = fn(a: i64,\n b: i64)\n -> i64 { a * b }  nil }");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn arrow_on_own_line_in_extern_decl_446() {
    let p = parse("extern fn ext(a: i32,\n              b: i32)\n              -> nil");
    assert_eq!(p.items.len(), 1);
}

#[test]
fn no_return_type_fn_unaffected_446() {
    // The newline-tolerant eat must restore position when no `->` follows,
    // so a body-only signature still parses and the NEXT item is intact.
    let p = parse("fn noret(a: i64) { print(a) }\nfn other() -> i64 { 1 }");
    assert_eq!(p.items.len(), 2, "both items must parse");
    match &p.items[0] {
        Item::Fn(f) => assert!(f.ret_type.is_none(), "first fn has no return type"),
        other => panic!("expected a fn item, got {:?}", other),
    }
}

#[test]
fn missing_body_brace_names_the_signature_446() {
    // The old message ("expected LBrace before block") pointed at the body;
    // the real problem is that the signature ended early.
    let e = parse_err("fn f(a: i64) -> i64\n{\n    a\n}");
    assert!(e.contains("`f`"), "diagnostic must name the fn, got: {e}");
    assert!(e.contains("signature"), "diagnostic must point at the signature, got: {e}");
}

// ── slicing seam (#501 S6/S7) ────────────────────────────────────────────────
// Both slicing families are permanent surface (SPEC §4.3, OPERATORS §9), but
// they do not mix: `..`/`..=` is the unstepped range form, `:` is the
// stepped/full form. A `..` range in a `:` slice bound used to parse into
// `IndexElem::Slice { start: Expr::Range, .. }` — check-clean, and fatal only
// at run time ("slice start must be integer, got range") or in the JIT.

#[test]
fn range_as_slice_start_is_a_parse_error_501() {
    let e = parse_err("fn f() -> nil { let b = a[0..100:2]\n nil }");
    assert!(e.contains("`a[0:100:2]`"), "diagnostic must name the colon form, got: {e}");
    assert!(e.contains("`a[0..100:2]`"), "diagnostic must name the rejected form, got: {e}");
}

#[test]
fn range_as_colon_colon_slice_start_is_a_parse_error_501() {
    // `x[-1..0::2]` takes the `start::step` shorthand branch.
    let e = parse_err("fn f() -> nil { let b = a[-1..0::2]\n nil }");
    assert!(e.contains("`a[0:100:2]`"), "diagnostic must name the colon form, got: {e}");
}

#[test]
fn range_as_slice_end_or_step_is_a_parse_error_501() {
    let end = parse_err("fn f() -> nil { let b = a[0:1..2]\n nil }");
    assert!(end.contains("`a[0:100:2]`"), "range in end bound must be rejected, got: {end}");
    let step = parse_err("fn f() -> nil { let b = a[0:4:1..2]\n nil }");
    assert!(step.contains("`a[0:100:2]`"), "range in step must be rejected, got: {step}");
    let no_start = parse_err("fn f() -> nil { let b = a[:1..2]\n nil }");
    assert!(no_start.contains("`a[0:100:2]`"), "start-less form must be rejected, got: {no_start}");
}

#[test]
fn both_slicing_families_still_parse_501() {
    // The ruling keeps both. Neither the range form nor the colon form moves.
    for src in [
        "fn f() -> nil { let b = a[0..100]\n nil }",
        "fn f() -> nil { let b = a[0..=99]\n nil }",
        "fn f() -> nil { let b = a[0..]\n nil }",
        "fn f() -> nil { let b = a[.., 3]\n nil }",
        "fn f() -> nil { let b = a[0:100]\n nil }",
        "fn f() -> nil { let b = a[0:100:2]\n nil }",
        "fn f() -> nil { let b = a[:]\n nil }",
        "fn f() -> nil { let b = a[:50]\n nil }",
        "fn f() -> nil { let b = a[10:]\n nil }",
        "fn f() -> nil { let b = a[0::2]\n nil }",
        "fn f() -> nil { let b = a[.., n - 1::-1, ..]\n nil }",
    ] {
        let p = parse(src);
        assert_eq!(p.items.len(), 1, "must parse: {src}");
    }
}

// ── #674 slice 3a: arena budgets — grammar ───────────────────────────────────

/// The shared spine table `check_tests.rs`'s twin test walks too — `parser_
/// tests.rs`/`check_tests.rs` pair over one table so the parser's `in` arm
/// and the checker's `is_arena_ctor_spine`-gated sites cannot silently
/// diverge on which spellings are on the spine.
pub(crate) const ARENA_CTOR_SPINE_TABLE: &[(&str, bool)] = &[
    ("forge.zeros[f32, [4]]", true),
    ("forge.ones[f32, [4]]", true),
    ("forge.uninit[f32, [4]]", true),
    ("forge.identity[f32, 4]", true),
    ("forge.kv[f32, [4]]", true),
    ("forge.trit[4, 4]", true),
    ("vault.zeros[f32, [4]]", true),
    ("stream.kv[f32, [4]]", true),
    // Not on the spine: wrong base, wrong method, or no bracket at all.
    ("r.normal[f32, [4]]", false),
    ("forge.reset()", false),
    ("forge.root", false),
    ("forge.carve(forge.root, 4M)", false),
];

#[test]
fn budget_alloc_in_shared_spine_table_674() {
    // Every spine entry produces `Expr::AllocIn` for a trailing `in b`.
    // #674 slice 3a fix: every non-spine entry is now a LOCATED PARSE ERROR
    // naming the ctor-spine shape — before this fix `in` there silently
    // ended the expression one token early and reparsed as its own
    // (undefined) identifier statement (probe `s3`'s pre-3a shape,
    // `forge.reshape[…] in b` yielding `undefined identifier `in`` plus a
    // discarded-value warning); that silent reading is gone.
    for (spine, is_spine) in ARENA_CTOR_SPINE_TABLE {
        let src = format!("fn f(b: Budget) -> nil {{ let t = {spine} in b\n nil }}");
        if *is_spine {
            let p = parse(&src);
            let Item::Fn(f) = &p.items[0] else { panic!("expected fn") };
            let Stmt::Let(l) = &f.body.stmts[0] else { panic!("expected let") };
            assert!(matches!(l.value, Expr::AllocIn { .. }),
                "`{spine} in b`: expected AllocIn, got {:?}", l.value);
        } else {
            let msg = parse_err(&src);
            assert!(
                msg.contains("applies only to an arena constructor"),
                "`{spine} in b`: expected the ctor-spine parse error, got: {msg}"
            );
        }
    }
}

#[test]
fn non_spine_in_tail_is_a_located_parse_error_674() {
    // The motivating example for the grammar's side condition on `in`:
    // `forge.reshape` is not on the ctor spine (not in
    // `ARENA_CTOR_ALLOC_METHODS`), so `in b` after it must be refused at
    // parse time rather than silently read as a second, discarded-value
    // statement.
    let msg = parse_err("fn f(b: Budget) -> nil { let t = forge.reshape[f32, [4]] in b\n nil }");
    assert!(
        msg.contains("`in` applies only to an arena constructor") && msg.contains("forge.zeros"),
        "got: {msg}"
    );
}

#[test]
fn ctor_with_trailing_call_before_in_gets_its_own_message_674() {
    // PR #814 review: `forge.kv[…](capacity=N) in b` — a genuine ctor
    // spine, but the trailing call takes `expr` off the spine by the time
    // this loop reaches `in` (`is_arena_ctor_spine(&expr)` sees the CALL as
    // the outer node, not the `[…]`). Unlike `forge.reshape[…] in b` above
    // (never a spine at all), the user here DID write an arena constructor —
    // the generic "applies only to an arena constructor" message would be
    // actively misleading, so this shape gets its own, naming the actual
    // mistake (the call sits between `[…]` and `in`).
    let msg = parse_err(
        "fn f(b: Budget) -> nil { let t = forge.kv[f32, [4]](capacity=8) in b\n nil }");
    assert!(
        msg.contains("the `in` tail must follow the `[…]` directly"),
        "got: {msg}"
    );
    assert!(
        !msg.contains("applies only to an arena constructor"),
        "must not fall back to the generic non-spine message, got: {msg}"
    );
}

#[test]
fn budget_block_parses_as_statement_674() {
    let p = parse("fn f(b: Budget) -> nil { forge.enter(b) { let t = forge.zeros[f32, [4]] }\n nil }");
    let Item::Fn(f) = &p.items[0] else { panic!("expected fn") };
    match &f.body.stmts[0] {
        Stmt::Expr { lhs: Expr::BudgetBlock(bb), .. } => {
            assert!(matches!(bb.operand.as_ref(), Expr::Ident(n, _) if n == "b"));
            assert_eq!(bb.body.stmts.len(), 1);
        }
        other => panic!("expected a BudgetBlock statement, got {:?}", other),
    }
}

#[test]
fn budget_block_parses_as_expression_674() {
    // Genuine expression position: a `let` RHS is parsed through
    // `parse_expr` → … → `parse_primary` directly, never through
    // `try_parse_keyword_stmt` (whose own guard always wins at STATEMENT
    // position — the same reason a bare tail-position `forge { … }` is
    // ALSO always a `Stmt::Expr`, never a block's `tail_expr`, today).
    let p = parse("fn f(b: Budget) -> nil { let x = forge.enter(b) { 5 }\n nil }");
    let Item::Fn(f) = &p.items[0] else { panic!("expected fn") };
    let Stmt::Let(l) = &f.body.stmts[0] else { panic!("expected let") };
    assert!(matches!(l.value, Expr::BudgetBlock(_)), "got {:?}", l.value);
}

// PR #814 review: `forge.enter`'s arity of exactly 1 (the `Budget` operand)
// is not a `check.rs` rule at all, unlike `forge.carve`/`forge.reap` —
// `parse_budget_block` calls `parse_expr` once for the operand and then
// requires `)`, so 0 or 2+ operands are structurally impossible to parse,
// not merely a later diagnostic. Pinned here rather than in `check_tests.rs`
// because that harness's `check()`/`checked()` both `.expect("parse
// failed")` — a parse failure there is a test-harness panic, not a value to
// assert on.

#[test]
fn budget_block_no_operand_is_a_parse_error_674() {
    let msg = parse_err("fn f() -> nil { forge.enter() { 5 }\n nil }");
    assert!(msg.contains("expected expression"), "got: {msg}");
}

#[test]
fn budget_block_two_operands_is_a_parse_error_674() {
    // `parse_budget_block` parses exactly one `parse_expr()` between the
    // parens, then expects `)` outright — a second operand leaves the `,`
    // itself as the unexpected token.
    let msg = parse_err("fn f(a: Budget, b: Budget) -> nil { forge.enter(a, b) { 5 }\n nil }");
    assert!(msg.contains("after budget block operand"), "got: {msg}");
}

#[test]
fn plain_forge_block_still_parses_674() {
    // The pre-existing `forge { … }` arena block must still parse — the new
    // `Dot`+`Ident("enter")` guard must not swallow it.
    let p = parse("fn f() -> nil { forge { let t = forge.zeros[f32, [4]] }\n nil }");
    let Item::Fn(f) = &p.items[0] else { panic!("expected fn") };
    assert!(matches!(&f.body.stmts[0], Stmt::Expr { lhs: Expr::ArenaBlock(_), .. }));
}

#[test]
fn for_ctor_in_budget_is_one_reading_674() {
    // §2.1 point 2 / §6 Q2: `for x in <ctor> in b { }` has exactly one
    // reading — the loop's own `in` is consumed before the iterable, so the
    // second `in` is unambiguously the postfix override.
    let p = parse("fn f(b: Budget) -> nil { for x in forge.zeros[f32, [4]] in b { }\n nil }");
    let Item::Fn(f) = &p.items[0] else { panic!("expected fn") };
    match &f.body.stmts[0] {
        Stmt::For { iter, .. } => assert!(matches!(iter, Expr::AllocIn { .. })),
        other => panic!("expected a for stmt, got {:?}", other),
    }
}

#[test]
fn in_arm_does_not_skip_newlines_674_z2() {
    // Probe z2's shape: a statement HEADED by `in`, right after a
    // constructor statement with no `in b` on it, is legal today and must
    // stay that way — the `in` arm must not `skip_newlines()`, or this
    // would silently reparse as `forge.zeros[...] in in`.
    let p = parse("fn f() -> nil {\n    let in = 7\n    let t = forge.zeros[f32, [4]]\n    in\n    nil\n}");
    let Item::Fn(f) = &p.items[0] else { panic!("expected fn") };
    // `let t = forge.zeros[...]` must NOT have absorbed the next line's `in`.
    let Stmt::Let(t_let) = &f.body.stmts[1] else { panic!("expected the second let") };
    assert!(matches!(t_let.value, Expr::Postfix { .. }), "must not be AllocIn: {:?}", t_let.value);
    // The bare `in` on its own line is its own (discarded-value) statement.
    match &f.body.stmts[2] {
        Stmt::Expr { lhs: Expr::Ident(n, _), .. } => assert_eq!(n, "in"),
        other => panic!("expected a bare `in` ident statement, got {:?}", other),
    }
}

#[test]
fn in_is_still_a_usable_identifier_674_z1() {
    // Probe z1: `in` is not a reserved word.
    let p = parse("fn f() -> nil { let in = 5\n print(\"z1 \" + to_str(in))\n nil }");
    let Item::Fn(f) = &p.items[0] else { panic!("expected fn") };
    let Stmt::Let(l) = &f.body.stmts[0] else { panic!("expected let") };
    assert!(matches!(&l.pattern, Pattern::Ident(n, _) if n == "in"));
}

#[test]
fn alloc_in_operand_is_postfix_not_full_expr_674() {
    // §2.1 point 3: `b + 1` does not parse as an AllocIn operand — there is
    // no `Budget` arithmetic. `forge.zeros[f32,[4]] in b + 1` must parse as
    // `(forge.zeros[f32,[4]] in b) + 1` (an ill-typed but SYNTACTICALLY
    // valid expression), not as `... in (b + 1)`.
    let p = parse("fn f(b: Budget) -> nil { let t = forge.zeros[f32, [4]] in b + 1\n nil }");
    let Item::Fn(f) = &p.items[0] else { panic!("expected fn") };
    let Stmt::Let(l) = &f.body.stmts[0] else { panic!("expected let") };
    match &l.value {
        Expr::BinOp { op: BinOp::Add, lhs, .. } => {
            assert!(matches!(lhs.as_ref(), Expr::AllocIn { .. }));
        }
        other => panic!("expected `(... in b) + 1`, got {:?}", other),
    }
}

#[test]
fn fp8_scalar_type_keeps_underscore_in_ctor_type_arg_818() {
    // #818: a scalar-type token in expression position (a constructor's
    // element-type argument, e.g. `forge.zeros[fp8_e4m3, [4]]`) used to be
    // reconstructed by Debug-formatting the token kind and lowercasing it
    // (`Fp8E4M3` -> `fp8e4m3`), silently dropping the underscore — the
    // checker then saw an `Expr::Ident("fp8e4m3", ..)` that named no scalar
    // type. Every other scalar-type token happened to survive that
    // round-trip (`F16` -> `f16`, `Bf16` -> `bf16`, `Trit` -> `trit`, ...);
    // only the two fp8 kinds have a token-kind spelling that isn't just a
    // case-fold of the source spelling. Pin both.
    for (src, want) in [
        ("fn f() -> nil { let t = forge.zeros[fp8_e4m3, [4]]\n nil }", "fp8_e4m3"),
        ("fn f() -> nil { let t = forge.zeros[fp8_e5m2, [4]]\n nil }", "fp8_e5m2"),
    ] {
        let p = parse(src);
        let Item::Fn(f) = &p.items[0] else { panic!("expected fn") };
        let Stmt::Let(l) = &f.body.stmts[0] else { panic!("expected let") };
        let Expr::Postfix { op: PostfixOp::Index(idxs), .. } = &l.value else {
            panic!("expected a postfix index ctor, got {:?}", l.value)
        };
        let IndexElem::Expr(elem) = &idxs[0] else { panic!("expected an expr index elem") };
        match elem {
            Expr::Ident(name, _) => assert_eq!(name, want, "src: {src}"),
            other => panic!("expected Expr::Ident({want:?}), got {:?}", other),
        }
    }
}
