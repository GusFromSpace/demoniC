/// demoniC lexer tests — driven by examples/hello.dmc and spec fixtures
///
/// Run: cargo test

#[cfg(test)]
mod tests {
    use super::super::lexer::{Lexer, TokenKind};

    fn lex(src: &str) -> Vec<TokenKind> {
        Lexer::new(src)
            .tokenize()
            .expect("lex failed")
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    fn lex_kinds_noeol(src: &str) -> Vec<TokenKind> {
        lex(src).into_iter().filter(|k| *k != TokenKind::Newline && *k != TokenKind::Eof).collect()
    }

    // ── Literals ──────────────────────────────────────────────────────────

    #[test]
    fn int_decimal() {
        assert_eq!(lex_kinds_noeol("42"), vec![TokenKind::IntLit(42)]);
    }

    #[test]
    fn int_with_underscores() {
        assert_eq!(lex_kinds_noeol("1_000_000"), vec![TokenKind::IntLit(1_000_000)]);
    }

    #[test]
    fn int_hex() {
        assert_eq!(lex_kinds_noeol("0xdead_beef"), vec![TokenKind::IntLit(0xdead_beef)]);
    }

    #[test]
    fn int_binary() {
        assert_eq!(lex_kinds_noeol("0b1010_1100"), vec![TokenKind::IntLit(0b1010_1100)]);
    }

    #[test]
    fn float_decimal() {
        let toks = lex_kinds_noeol("3.14159");
        assert!(matches!(toks[0], TokenKind::FloatLit(f, _) if (f - 3.14159).abs() < 1e-5));
    }

    #[test]
    fn float_leading_dot() {
        // `.5f64` — leading dot, explicit dtype
        let toks = lex_kinds_noeol(".5f64");
        assert!(matches!(toks[0], TokenKind::FloatLit(f, _) if (f - 0.5).abs() < 1e-10));
    }

    #[test]
    fn float_scientific() {
        let toks = lex_kinds_noeol("1e-9");
        assert!(matches!(toks[0], TokenKind::FloatLit(f, _) if (f - 1e-9).abs() < 1e-20));
    }

    #[test]
    fn str_literal() {
        let toks = lex_kinds_noeol(r#""demoniC\n""#);
        assert_eq!(toks, vec![TokenKind::StrLit("demoniC\n".into())]);
    }

    #[test]
    fn bool_nil() {
        let toks = lex_kinds_noeol("true false nil");
        assert_eq!(toks, vec![TokenKind::True, TokenKind::False, TokenKind::Nil]);
    }

    // ── Keywords ──────────────────────────────────────────────────────────

    #[test]
    fn keywords_roundtrip() {
        let pairs = [
            ("fn",       TokenKind::Fn),
            ("let",      TokenKind::Let),
            ("mut",      TokenKind::Mut),
            ("if",       TokenKind::If),
            ("else",     TokenKind::Else),
            ("return",   TokenKind::Return),
            ("vault",    TokenKind::Vault),
            ("forge",    TokenKind::Forge),
            ("stream",   TokenKind::Stream),
            ("model",    TokenKind::Model),
        ];
        for (src, expected) in pairs {
            let toks = lex_kinds_noeol(src);
            assert_eq!(toks, vec![expected], "failed on kw `{src}`");
        }
    }

    // ── Identifiers ───────────────────────────────────────────────────────

    #[test]
    fn plain_ident() {
        assert_eq!(
            lex_kinds_noeol("my_var"),
            vec![TokenKind::Ident("my_var".into())]
        );
    }

    #[test]
    fn mutating_fn_ident() {
        // `push!` is a valid identifier with trailing `!`
        assert_eq!(
            lex_kinds_noeol("push!"),
            vec![TokenKind::Ident("push!".into())]
        );
    }

    // ── Operators from TOKENIZER.md §2 ────────────────────────────────────

    #[test]
    fn relu_operator() {
        // `\>` lexes as a single ReLU token
        assert_eq!(lex_kinds_noeol(r"\>"), vec![TokenKind::ReLU]);
    }

    #[test]
    fn gelu_operator() {
        assert_eq!(lex_kinds_noeol(r"\<"), vec![TokenKind::GeLU]);
    }

    #[test]
    fn pipe_operator() {
        assert_eq!(lex_kinds_noeol("|>"), vec![TokenKind::Pipe]);
    }

    #[test]
    fn dot_elementwise() {
        let toks = lex_kinds_noeol(".+ .- .* ./");
        assert_eq!(toks, vec![
            TokenKind::DotAdd,
            TokenKind::DotSub,
            TokenKind::DotMul,
            TokenKind::DotDiv,
        ]);
    }

    #[test]
    fn dot_pow_variants() {
        let toks = lex_kinds_noeol(".^ .**");
        assert_eq!(toks, vec![TokenKind::DotPow, TokenKind::DotPow2]);
    }

    #[test]
    fn dot_comparison_operators() {
        let toks = lex_kinds_noeol(".> .< .>= .<=");
        assert_eq!(toks, vec![
            TokenKind::DotGt,
            TokenKind::DotLt,
            TokenKind::DotGe,
            TokenKind::DotLe,
        ]);
    }

    #[test]
    fn transpose_operator() {
        // `A'` — the `'` is a postfix transpose, not a char literal
        let toks = lex_kinds_noeol("A'");
        assert_eq!(toks, vec![TokenKind::Ident("A".into()), TokenKind::Transpose]);
    }

    #[test]
    fn range_operators() {
        let toks = lex_kinds_noeol(".. ..=");
        assert_eq!(toks, vec![TokenKind::DotDot, TokenKind::DotDotEq]);
    }

    #[test]
    fn arrow_operators() {
        let toks = lex_kinds_noeol("-> => <-");
        assert_eq!(toks, vec![TokenKind::Arrow, TokenKind::FatArrow, TokenKind::StreamArrow]);
    }

    // ── Comments ──────────────────────────────────────────────────────────

    #[test]
    fn line_comment_skipped() {
        let toks = lex_kinds_noeol("42 # this is a comment");
        assert_eq!(toks, vec![TokenKind::IntLit(42)]);
    }

    #[test]
    fn block_comment_skipped() {
        let toks = lex_kinds_noeol("1 #{ skip me }# 2");
        assert_eq!(toks, vec![TokenKind::IntLit(1), TokenKind::IntLit(2)]);
    }

    #[test]
    fn nested_block_comment() {
        // From hello.dmc fixture: nested block comments must work
        let src = "#{ outer #{ inner }# still outer }# 99";
        assert_eq!(lex_kinds_noeol(src), vec![TokenKind::IntLit(99)]);
    }

    // ── Newline significance ───────────────────────────────────────────────

    #[test]
    fn newline_inside_parens_insignificant() {
        // Inside `( )` newlines are ignored (SPEC §0)
        let src = "(\n42\n)";
        let toks = lex(src)
            .into_iter()
            .filter(|k| *k != TokenKind::Eof)
            .collect::<Vec<_>>();
        assert!(!toks.contains(&TokenKind::Newline));
    }

    #[test]
    fn newline_outside_parens_significant() {
        let src = "42\n43";
        let toks = lex(src)
            .into_iter()
            .filter(|k| *k != TokenKind::Eof)
            .collect::<Vec<_>>();
        assert!(toks.contains(&TokenKind::Newline));
    }

    // ── hello.dmc full fixture ────────────────────────────────────────────

    // Fixtures are in-crate snapshots of examples/{hello,matmul}.dmc so the
    // compiler/ crate builds and tests standalone with no reach into the repo
    // examples/ tree (#408). They only need to be representative full programs
    // for the lexer to chew on, not byte-current with examples/.
    #[test]
    fn hello_dmc_lexes_without_error() {
        let src = include_str!("../tests/fixtures/hello.dmc");
        let result = Lexer::new(src).tokenize();
        assert!(result.is_ok(), "hello.dmc lex error: {:?}", result.err());
    }

    #[test]
    fn matmul_dmc_lexes_without_error() {
        let src = include_str!("../tests/fixtures/matmul.dmc");
        let result = Lexer::new(src).tokenize();
        assert!(result.is_ok(), "matmul.dmc lex error: {:?}", result.err());
    }

    // ── Scalar type tokens ────────────────────────────────────────────────

    #[test]
    fn scalar_types_lex() {
        let types = ["i8","i16","i32","i64","u8","u16","u32","u64",
                        "f16","bf16","tf32","f32","f64","bool","str"];
        for t in &types {
            let toks = lex_kinds_noeol(t);
            assert_eq!(toks.len(), 1, "scalar type `{t}` should be single token");
        }
    }

    // ── Null Hypothesis 9: Tokenizer stability ────────────────────────────
    // Null 9: Tokenizer stability under BPE — operator splits post-canonicalization.
    // Falsification: any multi-char operator that splits into shorter tokens.

    #[test]
    fn null9_dotge_is_single_token() {
        // `.>=` must lex as DotGe, not DotGt + Assign or Dot + Ge.
        assert_eq!(lex_kinds_noeol(".>="), vec![TokenKind::DotGe]);
    }

    #[test]
    fn null9_dotle_is_single_token() {
        assert_eq!(lex_kinds_noeol(".<="), vec![TokenKind::DotLe]);
    }

    #[test]
    fn null9_dotgt_does_not_absorb_eq() {
        // `.> =` (with space) must be DotGt + Eq, not DotGe.
        let toks = lex_kinds_noeol(".> =");
        assert_eq!(toks, vec![TokenKind::DotGt, TokenKind::Eq]);
    }

    #[test]
    fn null9_pipe_is_single_token() {
        // `|>` must be Pipe, not BitOr + Gt.
        assert_eq!(lex_kinds_noeol("|>"), vec![TokenKind::Pipe]);
    }

    #[test]
    fn null9_relu_is_single_token() {
        // `\>` must be ReLU, not Backslash + Gt.
        assert_eq!(lex_kinds_noeol(r"\>"), vec![TokenKind::ReLU]);
    }

    #[test]
    fn null9_fat_arrow_is_single_token() {
        // `=>` must be FatArrow, not Assign + Gt.
        assert_eq!(lex_kinds_noeol("=>"), vec![TokenKind::FatArrow]);
    }

    #[test]
    fn null9_dotdoteq_is_single_token() {
        // `..=` must be DotDotEq, not DotDot + Assign.
        assert_eq!(lex_kinds_noeol("..="), vec![TokenKind::DotDotEq]);
    }

    #[test]
    fn null9_stream_arrow_is_single_token() {
        // `<-` must be StreamArrow, not Lt + Minus.
        assert_eq!(lex_kinds_noeol("<-"), vec![TokenKind::StreamArrow]);
    }

    // -- Char literals

    #[test]
    fn char_lit_ascii() {
        assert_eq!(lex_kinds_noeol(r#"c"A""#), vec![TokenKind::CharLit('A')]);
    }

    #[test]
    fn char_lit_digit() {
        assert_eq!(lex_kinds_noeol(r#"c"0""#), vec![TokenKind::CharLit('0')]);
    }

    #[test]
    fn char_lit_escape_newline() {
        assert_eq!(lex_kinds_noeol(r#"c"\n""#), vec![TokenKind::CharLit('\n')]);
    }

    #[test]
    fn char_lit_escape_tab() {
        assert_eq!(lex_kinds_noeol(r#"c"\t""#), vec![TokenKind::CharLit('\t')]);
    }

    #[test]
    fn char_lit_space() {
        assert_eq!(lex_kinds_noeol(r#"c" ""#), vec![TokenKind::CharLit(' ')]);
    }

    #[test]
    fn char_lit_empty_is_error() {
        let result = Lexer::new(r#"c"""#).tokenize();
        assert!(result.is_err(), "empty char literal should be a lex error");
    }

    #[test]
    fn char_lit_multi_char_is_error() {
        let result = Lexer::new(r#"c"AB""#).tokenize();
        assert!(result.is_err(), "multi-char literal should be a lex error");
    }

    // ── Byte literals `b'x'` (#334) ───────────────────────────────────────

    #[test]
    fn byte_lit_basic() {
        assert_eq!(lex_kinds_noeol("b'0'"), vec![TokenKind::IntLit(48)]);
        assert_eq!(lex_kinds_noeol("b'A'"), vec![TokenKind::IntLit(65)]);
        assert_eq!(lex_kinds_noeol("b' '"), vec![TokenKind::IntLit(32)]);
    }

    #[test]
    fn byte_lit_escapes() {
        assert_eq!(lex_kinds_noeol(r"b'\n'"),  vec![TokenKind::IntLit(10)]);
        assert_eq!(lex_kinds_noeol(r"b'\t'"),  vec![TokenKind::IntLit(9)]);
        assert_eq!(lex_kinds_noeol(r"b'\\'"),  vec![TokenKind::IntLit(92)]);
        assert_eq!(lex_kinds_noeol(r"b'\''"),  vec![TokenKind::IntLit(39)]);
        assert_eq!(lex_kinds_noeol(r"b'\0'"),  vec![TokenKind::IntLit(0)]);
    }

    #[test]
    fn byte_lit_slots_in_as_int() {
        assert_eq!(
            lex_kinds_noeol("x == b'0'"),
            vec![TokenKind::Ident("x".into()), TokenKind::EqEq, TokenKind::IntLit(48)]
        );
    }

    #[test]
    fn byte_lit_does_not_steal_transpose() {
        // `b'` = transpose of a tensor named `b` (common in ML) must survive.
        assert_eq!(
            lex_kinds_noeol("a @ b'"),
            vec![TokenKind::Ident("a".into()), TokenKind::At,
                 TokenKind::Ident("b".into()), TokenKind::Transpose]
        );
        // `b'` followed by an operator is transpose, not a byte char.
        assert_eq!(
            lex_kinds_noeol("b' .+ c"),
            vec![TokenKind::Ident("b".into()), TokenKind::Transpose,
                 TokenKind::DotAdd, TokenKind::Ident("c".into())]
        );
    }

    #[test]
    fn byte_lit_empty_and_unterminated_are_errors() {
        assert!(Lexer::new("b''").tokenize().is_ok(),  "b'' is b-transpose-transpose, not a byte lit");
        assert!(Lexer::new(r"b'\q'").tokenize().is_err(), "invalid escape must error");
    }

    #[test]
    fn char_lit_c_ident_without_quote_is_ident() {
        let kinds = lex_kinds_noeol("c");
        assert_eq!(kinds, vec![TokenKind::Ident("c".to_string())]);
    }

    #[test]
    fn char_lit_cool_ident_is_ident() {
        let kinds = lex_kinds_noeol("cool");
        assert_eq!(kinds, vec![TokenKind::Ident("cool".to_string())]);
    }

    // ── Additional literal coverage ───────────────────────────────────────

    #[test]
    fn int_zero() {
        assert_eq!(lex_kinds_noeol("0"), vec![TokenKind::IntLit(0)]);
    }

    #[test]
    fn float_scientific_notation() {
        let toks = lex_kinds_noeol("1.5e3");
        assert!(matches!(toks[0], TokenKind::FloatLit(f, _) if (f - 1500.0).abs() < 1.0));
    }

    #[test]
    fn float_neg_exponent() {
        let toks = lex_kinds_noeol("1e-3");
        assert!(matches!(toks[0], TokenKind::FloatLit(f, _) if (f - 0.001).abs() < 1e-6));
    }

    #[test]
    fn string_with_escape_newline() {
        let toks = lex_kinds_noeol(r#""\n""#);
        assert_eq!(toks, vec![TokenKind::StrLit("\n".to_string())]);
    }

    #[test]
    fn string_with_escape_tab() {
        let toks = lex_kinds_noeol(r#""\t""#);
        assert_eq!(toks, vec![TokenKind::StrLit("\t".to_string())]);
    }

    #[test]
    fn string_with_escaped_quote() {
        let toks = lex_kinds_noeol(r#""he said \"hi\"""#);
        assert_eq!(toks, vec![TokenKind::StrLit("he said \"hi\"".to_string())]);
    }

    #[test]
    fn string_empty() {
        assert_eq!(lex_kinds_noeol(r#""""#), vec![TokenKind::StrLit("".to_string())]);
    }

    #[test]
    fn bool_true_token() {
        assert_eq!(lex_kinds_noeol("true"), vec![TokenKind::True]);
    }

    #[test]
    fn bool_false_token() {
        assert_eq!(lex_kinds_noeol("false"), vec![TokenKind::False]);
    }

    #[test]
    fn nil_token() {
        assert_eq!(lex_kinds_noeol("nil"), vec![TokenKind::Nil]);
    }

    #[test]
    fn char_lit_digit_value() {
        assert_eq!(lex_kinds_noeol(r#"c"5""#), vec![TokenKind::CharLit('5')]);
    }

    #[test]
    fn char_lit_lowercase_letter() {
        assert_eq!(lex_kinds_noeol(r#"c"z""#), vec![TokenKind::CharLit('z')]);
    }

    #[test]
    fn char_lit_escape_tab_char() {
        assert_eq!(lex_kinds_noeol(r#"c"\t""#), vec![TokenKind::CharLit('\t')]);
    }

    #[test]
    fn char_lit_escape_backslash_char() {
        assert_eq!(lex_kinds_noeol(r#"c"\\""#), vec![TokenKind::CharLit('\\')]);
    }

    // ── Keyword coverage ──────────────────────────────────────────────────

    #[test]
    fn keyword_fn() {
        assert_eq!(lex_kinds_noeol("fn"), vec![TokenKind::Fn]);
    }

    #[test]
    fn keyword_let() {
        assert_eq!(lex_kinds_noeol("let"), vec![TokenKind::Let]);
    }

    #[test]
    fn keyword_if_else() {
        let toks = lex_kinds_noeol("if x else y");
        assert!(toks.contains(&TokenKind::If));
        assert!(toks.contains(&TokenKind::Else));
    }

    #[test]
    fn keyword_for_in() {
        let toks = lex_kinds_noeol("for x in xs");
        assert!(toks.contains(&TokenKind::For));
        // 'in' lexes as Ident("in")
        assert!(toks.contains(&TokenKind::Ident("in".to_string())));
    }

    #[test]
    fn keyword_while() {
        assert_eq!(lex_kinds_noeol("while"), vec![TokenKind::While]);
    }

    #[test]
    fn keyword_return() {
        assert_eq!(lex_kinds_noeol("return"), vec![TokenKind::Return]);
    }

    #[test]
    fn keyword_match() {
        assert_eq!(lex_kinds_noeol("match"), vec![TokenKind::Match]);
    }

    #[test]
    fn keyword_pub() {
        assert_eq!(lex_kinds_noeol("pub"), vec![TokenKind::Pub]);
    }

    #[test]
    fn keyword_use() {
        assert_eq!(lex_kinds_noeol("use"), vec![TokenKind::Use]);
    }

    // ── Operator coverage ─────────────────────────────────────────────────

    #[test]
    fn comparison_operators() {
        let toks = lex_kinds_noeol("< > <= >= == !=");
        assert_eq!(toks, vec![
            TokenKind::Lt, TokenKind::Gt,
            TokenKind::LtEq, TokenKind::GtEq,
            TokenKind::EqEq, TokenKind::BangEq,
        ]);
    }

    #[test]
    fn logical_operators() {
        let toks = lex_kinds_noeol("&& || !");
        assert_eq!(toks, vec![TokenKind::AndAnd, TokenKind::OrOr, TokenKind::Bang]);
    }

    #[test]
    fn bitwise_shift_operators() {
        let toks = lex_kinds_noeol("<< >>");
        assert_eq!(toks, vec![TokenKind::LtLt, TokenKind::RShift]);
    }

    #[test]
    fn arrow_and_fat_arrow() {
        let toks = lex_kinds_noeol("-> =>");
        assert_eq!(toks, vec![TokenKind::Arrow, TokenKind::FatArrow]);
    }

    #[test]
    fn inclusive_range_operator() {
        let toks = lex_kinds_noeol("0..=10");
        assert!(toks.contains(&TokenKind::DotDotEq));
    }

    #[test]
    fn pipe_right_operator() {
        let toks = lex_kinds_noeol("|>");
        assert_eq!(toks, vec![TokenKind::Pipe]);
    }

    #[test]
    fn assignment_operators() {
        let toks = lex_kinds_noeol("+= -= *= /=");
        assert_eq!(toks, vec![
            TokenKind::PlusEq, TokenKind::MinusEq,
            TokenKind::StarEq, TokenKind::SlashEq,
        ]);
    }

    #[test]
    fn stream_arrow_token() {
        let toks = lex_kinds_noeol("<-");
        assert_eq!(toks, vec![TokenKind::StreamArrow]);
    }

    #[test]
    fn string_with_newline_escape_sequence() {
        // Escape sequences in strings are resolved at lex time
        let toks = lex_kinds_noeol(r#""\nhello""#);
        assert_eq!(toks, vec![TokenKind::StrLit("\nhello".to_string())]);
    }

    #[test]
    fn ident_with_underscore() {
        let toks = lex_kinds_noeol("my_var");
        assert_eq!(toks, vec![TokenKind::Ident("my_var".to_string())]);
    }

    #[test]
    fn ident_with_digit_suffix() {
        let toks = lex_kinds_noeol("x2");
        assert_eq!(toks, vec![TokenKind::Ident("x2".to_string())]);
    }

    #[test]
    fn comment_is_skipped() {
        let toks = lex_kinds_noeol("42 # this is a comment");
        assert_eq!(toks, vec![TokenKind::IntLit(42)]);
    }

    #[test]
    fn model_keyword() {
        assert_eq!(lex_kinds_noeol("model"), vec![TokenKind::Model]);
    }

    #[test]
    fn break_continue_tokens() {
        let toks = lex_kinds_noeol("break continue");
        assert_eq!(toks, vec![TokenKind::Break, TokenKind::Continue]);
    }

    #[test]
    fn colon_and_double_colon() {
        let toks = lex_kinds_noeol(": ::");
        assert_eq!(toks, vec![TokenKind::Colon, TokenKind::ColonColon]);
    }

    #[test]
    fn at_sign_token() {
        let toks = lex_kinds_noeol("@grad");
        // @ is parsed as directive prefix
        assert!(!toks.is_empty());
    }

    // ── Integer suffix boundaries (#280) ──────────────────────────────────

    #[test]
    fn int_suffix_does_not_eat_adjacent_keyword() {
        // #280: `1if` is IntLit(1) + `if`, not IntLit(1) + Ident("f").
        let toks = lex_kinds_noeol("1if true { 2 } else { 3 }");
        assert_eq!(toks[0], TokenKind::IntLit(1));
        assert_eq!(toks[1], TokenKind::If, "the `i` of `if` was eaten as a suffix");
    }

    #[test]
    fn int_suffix_does_not_eat_adjacent_ident() {
        // #280: `1in` must keep `in` intact (`in` lexes as a plain ident).
        let toks = lex_kinds_noeol("1in");
        assert_eq!(toks, vec![TokenKind::IntLit(1), TokenKind::Ident("in".to_string())]);
    }

    #[test]
    fn int_suffix_still_consumed() {
        // Real suffixes still attach to the literal.
        for src in ["1i8", "1i16", "1i32", "1i64", "1u8", "1u16", "1u32", "1u64"] {
            let toks = lex_kinds_noeol(src);
            assert_eq!(toks, vec![TokenKind::IntLit(1)], "suffix not consumed in `{src}`");
        }
    }

    #[test]
    fn int_suffix_requires_token_boundary() {
        // #280: `1i32abc` is IntLit(1) + Ident("i32abc"), not a split suffix.
        let toks = lex_kinds_noeol("1i32abc");
        assert_eq!(toks, vec![TokenKind::IntLit(1), TokenKind::Ident("i32abc".to_string())]);
    }

    // ── 64-bit bit-pattern literals (#282) ────────────────────────────────

    #[test]
    fn hex_literal_high_bit() {
        // #282: all-ones and sign-bit hex patterns are valid i64 bit patterns.
        assert_eq!(lex_kinds_noeol("0xffffffffffffffff"), vec![TokenKind::IntLit(-1)]);
        assert_eq!(lex_kinds_noeol("0x8000000000000000"), vec![TokenKind::IntLit(i64::MIN)]);
        assert_eq!(lex_kinds_noeol("0xFFFF_FFFF_FFFF_FFFF"), vec![TokenKind::IntLit(-1)]);
    }

    #[test]
    fn binary_literal_high_bit() {
        let ones = "0b".to_string() + &"1".repeat(64);
        assert_eq!(lex_kinds_noeol(&ones), vec![TokenKind::IntLit(-1)]);
    }

    #[test]
    fn hex_literal_too_wide_still_errors() {
        // 65 bits must still be rejected.
        let r = Lexer::new("0x1ffffffffffffffff").tokenize();
        assert!(r.is_err(), "65-bit hex literal must be out of range");
    }

    #[test]
    fn hex_leading_underscore_rejected() {
        // #291.6: grammar is `"0x" hex_digit { hex_digit | "_" }` — an
        // underscore may only sit between digits, so `0x_ff` is invalid.
        assert!(Lexer::new("0x_ff").tokenize().is_err(), "`0x_ff` must be rejected");
        // grammar-legal underscores (between/trailing) still lex fine.
        assert_eq!(lex_kinds_noeol("0xf_f"), vec![TokenKind::IntLit(0xff)]);
        assert_eq!(lex_kinds_noeol("0xff_"), vec![TokenKind::IntLit(0xff)]);
    }

    #[test]
    fn binary_leading_underscore_rejected() {
        // #291.6: same rule for `"0b" bin_digit { bin_digit | "_" }`.
        assert!(Lexer::new("0b_11").tokenize().is_err(), "`0b_11` must be rejected");
        assert_eq!(lex_kinds_noeol("0b1_1"), vec![TokenKind::IntLit(0b11)]);
    }

    // ── Column counting in characters (#281) ──────────────────────────────

    #[test]
    fn col_counts_chars_not_bytes() {
        // #281: `é` is 2 bytes but 1 column; `+` is at col 2, `1` at col 3.
        let toks = Lexer::new("é+1").tokenize().unwrap();
        let plus = toks.iter().find(|t| t.kind == TokenKind::Plus).unwrap();
        assert_eq!(plus.span.col, 2, "`+` must be column 2 after a 2-byte char");
        let one = toks.iter().find(|t| t.kind == TokenKind::IntLit(1)).unwrap();
        assert_eq!(one.span.col, 3);
    }

    // ── #401: binary size suffixes (K/M/G) for byte-count literals ─────────

    #[test]
    fn int_size_suffix_g_m_k() {
        assert_eq!(lex_kinds_noeol("4G"), vec![TokenKind::IntLit(4 * 1024 * 1024 * 1024)]);
        assert_eq!(lex_kinds_noeol("2M"), vec![TokenKind::IntLit(2 * 1024 * 1024)]);
        assert_eq!(lex_kinds_noeol("3K"), vec![TokenKind::IntLit(3 * 1024)]);
        assert_eq!(lex_kinds_noeol("1Gi"), vec![TokenKind::IntLit(1024 * 1024 * 1024)]);
    }

    #[test]
    fn int_size_suffix_needs_token_boundary() {
        // `4Gx` must NOT eat `G` as a suffix — it stays IntLit(4) + Ident("Gx").
        assert_eq!(
            lex_kinds_noeol("4Gx"),
            vec![TokenKind::IntLit(4), TokenKind::Ident("Gx".into())]
        );
    }
}
