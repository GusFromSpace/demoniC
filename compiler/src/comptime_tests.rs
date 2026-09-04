//! #505: the `@comptime` fold itself — that the AST node is *replaced*.
//!
//! `check_tests` pins what the gate refuses and accepts; this file pins the
//! substitution, which is the part neither backend can observe once it has
//! happened. A fold that silently stopped replacing nodes would leave every
//! refusal test green and quietly restore the parity hole #505 closed, because
//! `dmc jit` would be back to seeing a directive block.

use super::ast::*;
use super::comptime::fold_program;
use super::lexer::Lexer;
use super::parser::Parser;

fn folded(src: &str) -> Program {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let mut program = Parser::new(tokens).parse_program().expect("parse failed");
    let errs = fold_program(&mut program);
    assert!(errs.is_empty(), "expected a clean fold, got: {:?}",
            errs.iter().map(|e| e.msg.clone()).collect::<Vec<_>>());
    program
}

/// The value of the first `let` in the first `fn` of a program.
fn first_let_value(p: &Program) -> &Expr {
    for item in &p.items {
        if let Item::Fn(f) = item {
            for s in &f.body.stmts {
                if let Stmt::Let(l) = s {
                    return &l.value;
                }
            }
        }
    }
    panic!("no `let` found");
}

#[test]
fn a_closed_integer_block_becomes_an_int_literal() {
    let p = folded(r#"fn main() -> i64 { let x = @comptime { 3 * 7 + 1 }  x }"#);
    match first_let_value(&p) {
        Expr::Literal(Literal::Int(22, None), _) => {}
        other => panic!("expected the folded literal 22, got {:?}", other),
    }
}

#[test]
fn a_closed_boolean_block_becomes_a_bool_literal() {
    let p = folded(r#"fn main() -> bool { let x = @comptime { 2 > 1 && 3 != 4 }  x }"#);
    match first_let_value(&p) {
        Expr::Literal(Literal::Bool(true), _) => {}
        other => panic!("expected the folded literal true, got {:?}", other),
    }
}

#[test]
fn the_folded_literal_keeps_the_blocks_span() {
    // Diagnostics raised on the folded value must still point at the
    // `@comptime` the reader wrote, not at line 0 or at some inner operand.
    let src = "fn main() -> i64 {\n    let x = @comptime { 1 + 1 }\n    x\n}";
    let p = folded(src);
    match first_let_value(&p) {
        Expr::Literal(Literal::Int(2, None), sp) => {
            assert_eq!(sp.line, 2, "the folded literal must carry the block's line");
        }
        other => panic!("expected a folded literal, got {:?}", other),
    }
}

#[test]
fn the_int_literal_is_unsuffixed_so_it_still_defaults_to_i64() {
    // #284/#316: an unconstrained integer literal defaults to i64, which is
    // what `let x = @comptime { … }` inferred before #505. Splicing a
    // *suffixed* literal would silently change the type of every folded
    // binding.
    let p = folded(r#"fn main() -> i64 { let x = @comptime { 40 + 2 }  x }"#);
    match first_let_value(&p) {
        Expr::Literal(Literal::Int(42, suffix), _) =>
            assert!(suffix.is_none(), "the folded literal must be unsuffixed, got {:?}", suffix),
        other => panic!("expected a folded literal, got {:?}", other),
    }
}

#[test]
fn a_loop_folds_to_its_accumulated_value() {
    // The "configuration table" case SPEC.md §7.8 names: real evaluation, not
    // constant-propagation over a literal expression.
    let p = folded(r#"
        fn main() -> i64 {
            let x = @comptime {
                let !acc = 0
                let !i = 1
                while i <= 4 { acc += i  i += 1 }
                acc
            }
            x
        }
    "#);
    match first_let_value(&p) {
        Expr::Literal(Literal::Int(10, None), _) => {}
        other => panic!("expected the folded sum 10, got {:?}", other),
    }
}

#[test]
fn integer_semantics_are_the_interpreters_own() {
    // The point of evaluating with the reference interpreter rather than a
    // second const-folder: division truncates toward zero and `%` takes the
    // dividend's sign, exactly as `dmc run` does. `int_div_floor_neg` is
    // already a jit_probe, so a fold that disagreed here would put the two
    // backends back into divergence through the front door.
    let p = folded(r#"fn main() -> i64 { let x = @comptime { (0 - 7) / 2 }  x }"#);
    match first_let_value(&p) {
        Expr::Literal(Literal::Int(-3, None), _) => {}
        other => panic!("expected -3 (truncating division), got {:?}", other),
    }
    let p = folded(r#"fn main() -> i64 { let x = @comptime { (0 - 7) % 2 }  x }"#);
    match first_let_value(&p) {
        Expr::Literal(Literal::Int(-1, None), _) => {}
        other => panic!("expected -1 (dividend's sign), got {:?}", other),
    }
}

#[test]
fn a_residual_shape_param_block_is_not_spliced() {
    // Tier 2 (COMPTIME_V1.md §4). The pass cannot pick a value for `N` — that
    // is fixed per monomorphization — so the node must survive for the
    // backends to lower. Splicing something here would be picking a shape.
    let p = folded(r#"
        fn tile[N](x: Tensor[f32, [N]]) -> i64 { let k = @comptime { N * 2 }  k }
        fn main() -> i64 { tile(forge.zeros[f32, [4]]) }
    "#);
    match first_let_value(&p) {
        Expr::DirectiveBlock { directives, .. } =>
            assert!(directives.iter().any(|d| d.name == "comptime"),
                    "the residual node must keep its directive"),
        other => panic!("a shape-parameter block must not be folded, got {:?}", other),
    }
}

#[test]
fn folding_reaches_a_nested_block() {
    // The walker must descend into every construct, not only a fn's top level
    // — a block it fails to reach is a block `dmc jit` still refuses.
    let p = folded(r#"
        fn main() -> i64 {
            let !acc = 0
            for i in 0..2 {
                acc += @comptime { 5 * 5 }
            }
            acc
        }
    "#);
    let mut found = false;
    if let Some(Item::Fn(f)) = p.items.first() {
        for s in &f.body.stmts {
            if let Stmt::For { body, .. } = s {
                for inner in &body.stmts {
                    if let Stmt::Expr { assign: Some((_, rhs)), .. } = inner {
                        assert!(matches!(rhs, Expr::Literal(Literal::Int(25, None), _)),
                                "expected the folded literal 25, got {:?}", rhs);
                        found = true;
                    }
                }
            }
        }
    }
    assert!(found, "the fold did not reach inside the `for` body");
}
