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
}
