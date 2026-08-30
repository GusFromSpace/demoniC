/// Integration tests for the `dmc fmt` pretty-printer.
///
/// Each test parses a source string, formats it, then re-parses the formatted
/// output to verify round-trip parse stability.

#[cfg(test)]
mod fmt_tests {
    use crate::fmt::pretty_print_program;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse_program(src: &str) -> Result<crate::ast::Program, crate::parser::ParseError> {
        let tokens = Lexer::new(src).tokenize().expect("lex failed");
        Parser::new(tokens).parse_program()
    }

    #[test]
    fn fmt_roundtrip_simple_fn() {
        let src = r#"fn add(a: i64, b: i64) -> i64 { a + b }"#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_model() {
        let src = r#"
model Counter {
    n: i64
    fn inc(self) -> i64 { self.n + 1 }
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_let_and_return() {
        let src = r#"
fn compute(x: i64) -> i64 {
    let y: i64 = x + 1
    return y
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_char_literal() {
        let src = r#"fn newline() -> u32 { c"\n" }"#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        assert!(formatted.contains(r#"c"\n""#), "got: {}", formatted);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_if_else() {
        let src = r#"
fn abs(x: i64) -> i64 {
    if x < 0 { -x } else { x }
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_for_loop() {
        let src = r#"
fn sum(n: i64) -> i64 {
    let mut acc: i64 = 0
    for i in 0..n {
        acc += i
    }
    acc
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_type_alias() {
        let src = r#"type MyFloat = f32"#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_multiple_items() {
        let src = r#"
fn foo(x: i64) -> i64 { x }
fn bar(y: i64) -> i64 { y + 1 }
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_float_suffixes() {
        let src = r#"
fn test_floats() {
    let a: f16 = 1.0f16
    let b: bf16 = 2.5bf16
    let c: tf32 = 3.14tf32
    let d: f32 = 4.0f32
    let e: f64 = 5.0f64
    let f: fp8_e4m3 = 0.5fp8_e4m3
    let g: fp8_e5m2 = 0.25fp8_e5m2
}
"#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        assert!(formatted.contains("1.0f16"), "got: {}", formatted);
        assert!(formatted.contains("2.5bf16"), "got: {}", formatted);
        assert!(formatted.contains("3.14tf32"), "got: {}", formatted);
        assert!(formatted.contains("4.0f32"), "got: {}", formatted);
        assert!(formatted.contains("5.0f64"), "got: {}", formatted);
        assert!(formatted.contains("0.5fp8_e4m3"), "got: {}", formatted);
        assert!(formatted.contains("0.25fp8_e5m2"), "got: {}", formatted);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_match_expr() {
        let src = r#"
fn classify(n: i64) -> i64 {
    match n {
        0 => 0,
        1 => 1,
        _ => -1,
    }
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_while_break_continue() {
        let src = r#"
fn first_positive(n: i64) -> i64 {
    let mut result = -1
    let mut i = 0
    while i < n {
        if i == 0 { i += 1; continue }
        result = i
        break
    }
    result
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_pipe_operator() {
        let src = r#"
fn inc(x: i64) -> i64 { x + 1 }
fn double(x: i64) -> i64 { x * 2 }
fn main() -> i64 { 5 |> inc |> double }
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    /// #501. The formatter's job on the pipe is to collapse the surviving
    /// spellings (`\|>` and the bare `|>`) onto the canonical one — `\|>`, per
    /// TOKENIZER §2–§3. It was emitting the bare form, i.e. normalizing away
    /// from canon. The third spelling, `>>`, is gone entirely (ruling S1a).
    #[test]
    fn fmt_normalizes_every_pipe_spelling_to_canonical() {
        for src in [
            r#"fn inc(x: i64) -> i64 { x + 1 }
fn main() -> i64 { 5 \|> inc }"#,
            r#"fn inc(x: i64) -> i64 { x + 1 }
fn main() -> i64 { 5 |> inc }"#,
        ] {
            let prog = parse_program(src).expect("parse failed");
            let formatted = pretty_print_program(&prog);
            assert!(formatted.contains(r"\|>"),
                    "expected canonical `\\|>` in output, got: {}", formatted);
            parse_program(&formatted).expect("re-parse of formatted output failed");
        }
    }

    /// #501. `\>` / `\<` are canonical (TOKENIZER §3 lists `relu(x)` as the
    /// *alternate*); the formatter was rewriting the operator into the call,
    /// inverting the rule against all 30 canonical uses in the corpus.
    #[test]
    fn fmt_keeps_activation_operators_canonical() {
        let src = r#"
fn main() -> f64 {
    let a = [1.0, -2.0]
    let b = \> a
    let c = \< a
    b[0] + c[0]
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        assert!(formatted.contains(r"\>"), "expected `\\>` preserved, got: {}", formatted);
        assert!(formatted.contains(r"\<"), "expected `\\<` preserved, got: {}", formatted);
        assert!(!formatted.contains("relu("), "formatter must not emit the alternate spelling: {}", formatted);
        assert!(!formatted.contains("gelu("), "formatter must not emit the alternate spelling: {}", formatted);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    /// #501. XOR is `^` (OPERATORS §8a). The formatter emitted `^^`, which is
    /// not a demoniC operator — `dmc fmt` output did not lex. No round-trip
    /// test covered `^`, which is how it survived.
    #[test]
    fn fmt_roundtrip_bitwise_operators() {
        let src = r#"
fn main() -> i64 {
    let a = 6
    let b = 3
    let x = a ^ b
    let y = a & b
    let z = a | b
    let w = a << b
    x + y + z + w
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        assert!(!formatted.contains("^^"), "`^^` is not a demoniC operator: {}", formatted);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_lambda() {
        let src = r#"
fn apply(x: i64) -> i64 {
    let f = fn(n: i64) -> i64 { n * n }
    f(x)
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_pub_fn() {
        let src = r#"pub fn add(a: i64, b: i64) -> i64 { a + b }"#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        assert!(formatted.contains("pub"), "expected `pub` in output, got: {}", formatted);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_use_decl() {
        let src = r#"use "other.dmc""#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_grad_fn() {
        let src = r#"
@grad fn mse(!w: Tensor[f32, [4]], x: Tensor[f32, [4]]) -> f32 {
    sum((w .- x) .* (w .- x))
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        assert!(formatted.contains("@grad"), "expected `@grad` in output, got: {}", formatted);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_inclusive_range() {
        let src = r#"
fn sum_to(n: i64) -> i64 {
    let mut acc = 0
    for i in 0..=n { acc += i }
    acc
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        assert!(formatted.contains("..="), "expected inclusive range in output, got: {}", formatted);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_tuple_return() {
        let src = r#"fn swap(a: i64, b: i64) -> (i64, i64) { (b, a) }"#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_else_if_chain() {
        let src = r#"
fn sign(n: i64) -> i64 {
    if n > 0 { 1 }
    else if n < 0 { -1 }
    else { 0 }
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_tensor_type_in_signature() {
        let src = r#"
fn dot[N](a: Tensor[f32, [N]], b: Tensor[f32, [N]]) -> f32 {
    sum(a .* b)
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_string_escapes() {
        // #290: a string literal containing decoded control chars must
        // re-emit them as escape sequences — the lexer rejects a raw
        // newline inside a string literal.
        let src = r#"fn main() -> nil { let s = "a\nb\tc\r\0\\\""  nil }"#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        let reparsed = parse_program(&formatted)
            .expect("re-parse of formatted string-literal output failed");
        // And the round-trip must be lossless: fmt(parse(fmt)) == fmt.
        assert_eq!(pretty_print_program(&reparsed), formatted);
    }

    // ── #501 S16: the reserved `BinOp` variants ──────────────────────────
    //
    // No parser path constructs `Pow` or `RShift`, so these tests build the
    // AST nodes directly. Before the fix, `fmt` spelled `Pow` as `^` (which
    // parses as `BitXor`) and `RShift` as `>>` (which parsed as `BinOp::Pipe`
    // until S1a freed the token): re-parseable, and a different operator.
    //
    // `BitShr` left this set in #530: `>>` is the right shift now, the parser
    // constructs `BitShr` from it, and `>>` is its true spelling — covered by
    // `fmt_rshift_round_trips` below rather than by a marker.

    use crate::ast::BinOp;
    use crate::fmt::{binop_str, RESERVED_POW, RESERVED_RSHIFT};

    const RESERVED: [(BinOp, &str); 2] = [
        (BinOp::Pow, RESERVED_POW),
        (BinOp::RShift, RESERVED_RSHIFT),
    ];

    /// The markers must not lex, let alone re-parse as another operator.
    /// This is the release-build half of the guarantee, and it holds in
    /// either build.
    #[test]
    fn fmt_reserved_binop_markers_do_not_relex() {
        for (op, marker) in RESERVED {
            let src = format!("fn main() -> i64 {{ 1 {} 2 }}", marker);
            assert!(Lexer::new(&src).tokenize().is_err(),
                    "marker for BinOp::{:?} lexed: {}", op, marker);
        }
        // The old spellings were not inert: `^` still parses, as `BitXor`.
        assert!(matches!(try_parse_binop("1 ^ 2"), Some(BinOp::BitXor)));
        // `>>` is the other half, and #530 settled which node it denotes:
        // `BitShr`, never the pipe-era `RShift`. That is exactly why `RShift`
        // may not be printed as `>>` — it would round-trip into a shift.
        assert!(matches!(try_parse_binop("1 >> 2"), Some(BinOp::BitShr)));
    }

    /// #530: `>>` is the right shift's real spelling, so it must survive a
    /// format/re-parse round trip as the same node — the property the reserved
    /// markers exist to deny `RShift`.
    #[test]
    fn fmt_rshift_round_trips() {
        assert_eq!(binop_str(&BinOp::BitShr), ">>");
        let src = "fn main() -> i64 { 256 >> 2 }";
        let formatted = pretty_print_program(&parse_program(src).expect("parse failed"));
        assert!(formatted.contains("256 >> 2"), "formatted: {}", formatted);
        assert!(matches!(try_parse_binop("1 >> 2"), Some(BinOp::BitShr)));
    }

    /// `None` when `expr` does not lex or parse as a binop — a reserved
    /// spelling is a lex error, not a parse of some other operator.
    fn try_parse_binop(expr: &str) -> Option<BinOp> {
        let src = format!("fn main() -> i64 {{ {} }}", expr);
        let tokens = Lexer::new(&src).tokenize().ok()?;
        let prog = Parser::new(tokens).parse_program().ok()?;
        match &prog.items[0] {
            crate::ast::Item::Fn(f) => match &**f.body.tail_expr.as_ref()? {
                crate::ast::Expr::BinOp { op, .. } => Some(op.clone()),
                _ => None,
            },
            other => panic!("expected fn item, got: {:?}", other),
        }
    }

    /// The debug-build half: surfacing a reserved variant trips the assert
    /// in `fmt::reserved_binop` instead of quietly emitting wrong code.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "reserved BinOp::Pow reached the formatter")]
    fn fmt_reserved_pow_trips_debug_assert() { let _ = binop_str(&BinOp::Pow); }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "reserved BinOp::RShift reached the formatter")]
    fn fmt_reserved_rshift_trips_debug_assert() { let _ = binop_str(&BinOp::RShift); }

    /// With the assert compiled out, `fmt` still renders the marker rather
    /// than a re-parseable operator.
    #[cfg(not(debug_assertions))]
    #[test]
    fn fmt_reserved_binops_render_as_markers() {
        for (op, marker) in RESERVED {
            assert_eq!(binop_str(&op), marker, "BinOp::{:?}", op);
        }
    }

    #[test]
    fn fmt_prints_dynamic_dim_as_query_501() {
        // A dynamic dim in a *type* prints `?`. It used to print `_`, which the
        // parser now rejects — the printer would have emitted unparseable code.
        let src = "fn rows(x: Tensor[f32, [?, ?]]) -> i64 { 1 }";
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        assert!(formatted.contains("[?, ?]"), "dynamic dim prints `?`, got: {formatted}");
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_prints_shape_pattern_wildcard_as_underscore_501() {
        // In *pattern* position the same node is the wildcard pattern and keeps
        // its `_` spelling — the shape pattern at `examples/slice.dmc:20` comes
        // back out of `dmc fmt` unchanged. (That file is not round-trip clean
        // overall: it is one of 15 in tree that `fmt` mangles at `..` slicing,
        // a pre-existing bug unrelated to shapes.)
        let src = r#"
fn t[S](x: Tensor[f32, [2, S, 768]]) -> nil {
    match x.shape {
        [_, S, 768] => print("ok"),
        _           => panic("drift"),
    }
    nil
}
        "#;
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        assert!(formatted.contains("[_, S, 768]"), "pattern wildcard prints `_`, got: {formatted}");
        parse_program(&formatted).expect("re-parse of formatted output failed");
    }

    #[test]
    fn fmt_roundtrip_colon_slices_501() {
        // GRAMMAR_SWEEP N1: the printer emitted `..` for `IndexElem::Slice`,
        // so `a[0:100:2]` came back out as `a[0..100::2]` — a range in the
        // start bound, i.e. the invalid mixed form the parser now rejects.
        // Both slicing families are permanent surface (OPERATORS §9), so the
        // formatter must preserve which one the author wrote.
        let src = "fn main() -> nil {\n\
                   let s1 = a[0:100:2]\n\
                   let s2 = a[:]\n\
                   let s3 = a[:50]\n\
                   let s4 = a[10:]\n\
                   let s5 = a[0::2]\n\
                   let r1 = a[0..100]\n\
                   let r2 = a[0..=99]\n\
                   let r3 = a[.., 3]\n\
                   nil\n}";
        let prog = parse_program(src).expect("parse failed");
        let formatted = pretty_print_program(&prog);
        for spelling in ["a[0:100:2]", "a[:]", "a[:50]", "a[10:]", "a[0::2]",
                         "a[0..100]", "a[0..=99]", "a[.., 3]"] {
            assert!(formatted.contains(spelling),
                "fmt must preserve `{spelling}`, got:\n{formatted}");
        }
        let reparsed = parse_program(&formatted)
            .expect("re-parse of formatted slice output failed");
        assert_eq!(pretty_print_program(&reparsed), formatted);
    }
}
