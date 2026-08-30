/// demoniC parser — spec 0.0.4-draft
///
/// Companion to: GRAMMAR.ebnf, SPEC.md
///
/// Hand-written recursive descent. One function per non-terminal where
/// useful; precedence ladder is one function per level (parse_logic_or,
/// parse_logic_and, …) for readability. Mirrors the structure of lexer.rs.
///
/// Pragmatic extensions to bare EBNF (documented inline at each use):
///   - shape elements may contain arithmetic exprs (`B/dp`, `H*D`)
///   - call and bracket arg lists allow named args (`dp=8`, `axis=-1`)
///   - `[T; N]` array type (Rust-style fixed-size array)
///   - `...` spread in call arg lists (lex: DotDot then Dot)
///   - `start::step` and `::step` indexing shorthand (lex: ColonColon; #529
///     added the start-omitted form so `a[::-1]` parses)
///   - `expr as type` postfix cast (used heavily in examples)
///
/// Pre-alpha goal: parse all 12 files in `examples/*.dmc` cleanly.

use std::fmt;

use crate::ast::*;
use crate::lexer::{Span, Token, TokenKind};

// ─── Error ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ParseError {
    pub msg: String,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at {}:{}: {}", self.line, self.col, self.msg)
    }
}

pub type ParseResult<T> = Result<T, ParseError>;

// ─── Parser ──────────────────────────────────────────────────────────────────

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    struct_literal_allowed: bool,
    /// Current expression-nesting depth, bounded by `MAX_EXPR_DEPTH` so pathological
    /// input can't overflow the native stack during recursive-descent parsing (#214).
    depth: usize,
}

/// Maximum expression nesting depth. Far above anything real code reaches, far
/// below the native-stack overflow threshold — so deeply nested input gets a
/// clean parse error instead of a SIGABRT.
const MAX_EXPR_DEPTH: usize = 1024;

/// Which construct a shape-element list belongs to. The two share every element
/// spelling but `_`: in a shape *pattern* it is the wildcard pattern, in a
/// tensor *type* it is not a dimension at all — the dynamic dim is `?` (#501).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShapeCtx {
    /// `Tensor[f32, [ … ]]` and friends — the shape of a type.
    Type,
    /// `[ … ]` in pattern position, e.g. a `match` arm over `x.shape`.
    Pattern,
}

/// One wording for the S3 break, wherever `_` surfaces in a type's shape.
const UNDERSCORE_NOT_A_DIM: &str = "`_` is not a dimension; a dynamic dim is `?`";

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            struct_literal_allowed: true,
            depth: 0,
        }
    }

    // ── Primitives ────────────────────────────────────────────────────────

    fn peek(&self) -> &TokenKind {
        &self.tokens.get(self.pos).map(|t| &t.kind).unwrap_or(&TokenKind::Eof)
    }

    fn peek_at(&self, offset: usize) -> &TokenKind {
        &self.tokens.get(self.pos + offset).map(|t| &t.kind).unwrap_or(&TokenKind::Eof)
    }

    fn peek_span(&self) -> Span {
        self.tokens.get(self.pos)
            .map(|t| t.span.clone())
            .unwrap_or_else(|| self.tokens.last().map(|t| t.span.clone())
                .unwrap_or(Span { start: 0, end: 0, line: 1, col: 1 }))
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        if self.pos < self.tokens.len() - 1 { self.pos += 1; }
        tok
    }

    fn eat(&mut self, kind: &TokenKind) -> bool {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(kind) {
            self.advance();
            true
        } else { false }
    }

    fn expect(&mut self, kind: &TokenKind, ctx: &str) -> ParseResult<Token> {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(kind) {
            Ok(self.advance())
        } else {
            Err(self.err(format!("expected {:?} {}, found {:?}", kind, ctx, self.peek())))
        }
    }

    fn err(&self, msg: impl Into<String>) -> ParseError {
        let sp = self.peek_span();
        ParseError { msg: msg.into(), line: sp.line, col: sp.col }
    }

    /// `err`, but pointing at a span already consumed — for a construct that is
    /// only known to be wrong after it has been parsed.
    fn err_at(&self, sp: &Span, msg: impl Into<String>) -> ParseError {
        ParseError { msg: msg.into(), line: sp.line, col: sp.col }
    }

    fn span_from(&self, start: &Span) -> Span {
        let end = self.tokens.get(self.pos.saturating_sub(1))
            .map(|t| t.span.end).unwrap_or(start.end);
        Span { start: start.start, end, line: start.line, col: start.col }
    }

    /// Skip any consecutive Newline tokens. Newlines are statement-terminators
    /// in block context but should be ignored everywhere a stmt is *not*
    /// expected next (e.g. between fn decls, before a `}`, between match arms).
    fn skip_newlines(&mut self) {
        while matches!(self.peek(), TokenKind::Newline) { self.advance(); }
    }

    /// Like `eat`, but skips over any intervening newlines first.
    /// If the next non-newline token matches `kind`, consume the newlines and
    /// the token and return true. Otherwise leave position unchanged.
    fn eat_over_newlines(&mut self, kind: &TokenKind) -> bool {
        let saved = self.pos;
        while matches!(self.peek(), TokenKind::Newline) { self.advance(); }
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(kind) {
            self.advance();
            true
        } else {
            self.pos = saved;
            false
        }
    }

    fn at_eof(&self) -> bool { matches!(self.peek(), TokenKind::Eof) }

    // ─── Entry ────────────────────────────────────────────────────────────

    pub fn parse_program(&mut self) -> ParseResult<Program> {
        let start = self.peek_span();
        let mut items = Vec::new();
        self.skip_newlines();
        while !self.at_eof() {
            items.push(self.parse_item()?);
            self.skip_newlines();
        }
        // #463: the EOF token records where the source ends. A trailing
        // newline parks EOF at column 1 of the line past the last real one —
        // don't count that phantom line.
        let eof = self.peek_span();
        let source_lines =
            if eof.col == 1 && eof.line > 1 { eof.line - 1 } else { eof.line };
        let mut program = Program { items, span: self.span_from(&start), source_lines };
        // #333: UFCS — rewrite `x.f(args)` → `f(x, args)` once, post-parse, so
        // check/interp/jit all inherit the desugared AST.
        crate::desugar::desugar_ufcs(&mut program);
        Ok(program)
    }

    // ─── Items ────────────────────────────────────────────────────────────

    fn parse_item(&mut self) -> ParseResult<Item> {
        let is_pub = self.eat(&TokenKind::Pub);
        self.skip_newlines();

        // collect leading directives (zero or more)
        let mut directives = Vec::new();
        while matches!(self.peek(), TokenKind::At) {
            // is this `@ident ...` (directive) or just `@`? At item level it's always directive.
            directives.push(self.parse_directive()?);
            self.skip_newlines();
        }

        // What if pub was specified after directives? E.g., `@pp pub fn foo()`
        let is_pub = is_pub || {
            let p = self.eat(&TokenKind::Pub);
            if p {
                self.skip_newlines();
            }
            p
        };

        let inner = match self.peek() {
            TokenKind::Fn      => Ok(Item::Fn(self.parse_fn_decl(directives)?)),
            TokenKind::Extern  => {
                if !directives.is_empty() {
                    return Err(self.err("directives on extern fn not supported"));
                }
                Ok(Item::ExternFn(self.parse_extern_fn_decl()?))
            }
            TokenKind::Model   => Ok(Item::Model(self.parse_model_decl(directives)?)),
            TokenKind::Type    => {
                if !directives.is_empty() {
                    return Err(self.err("directives on type alias not supported"));
                }
                Ok(Item::TypeAlias(self.parse_type_alias()?))
            }
            TokenKind::Enum    => {
                if !directives.is_empty() {
                    return Err(self.err("directives on enum not supported"));
                }
                Ok(Item::Enum(self.parse_enum_decl()?))
            }
            TokenKind::Vault | TokenKind::Forge | TokenKind::Stream => {
                if is_pub {
                    return Err(self.err("visibility modifier not allowed on arena blocks"));
                }
                if !directives.is_empty() {
                    return Err(self.err("directives on arena block not supported"));
                }
                Ok(Item::Arena(self.parse_arena_block()?))
            }
            TokenKind::Let     => {
                let let_stmt = self.parse_let_stmt()?;
                let span = let_stmt.span.clone();
                let item = Item::Let(let_stmt);
                if directives.is_empty() {
                    Ok(item)
                } else {
                    Ok(Item::Directive {
                        directives,
                        inner: Box::new(item),
                        span,
                    })
                }
            }
            TokenKind::Use     => {
                if is_pub {
                    return Err(self.err("visibility modifier not allowed on use statements"));
                }
                if !directives.is_empty() {
                    return Err(self.err("directives on use statement not supported"));
                }
                Ok(Item::Use(self.parse_use_stmt()?))
            }
            other => Err(self.err(format!("expected item (fn/extern/model/type/let/use/vault/forge/stream), found {:?}", other))),
        }?;

        if is_pub {
            Ok(Item::Pub(Box::new(inner)))
        } else {
            Ok(inner)
        }
    }

    fn parse_use_stmt(&mut self) -> ParseResult<UseStmt> {
        let start = self.peek_span();
        self.expect(&TokenKind::Use, "in use stmt")?;
        let path = match self.peek().clone() {
            TokenKind::StrLit(s) => {
                self.advance();
                s
            }
            other => return Err(self.err(format!("expected string literal for import path, found {:?}", other))),
        };
        let mut alias = None;
        if self.eat(&TokenKind::As) {
            alias = Some(self.expect_ident("alias name after `as`")?);
        }
        Ok(UseStmt {
            path,
            alias,
            span: self.span_from(&start),
        })
    }

    fn parse_fn_decl(&mut self, directives: Vec<Directive>) -> ParseResult<FnDecl> {
        let start = self.peek_span();
        self.expect(&TokenKind::Fn, "in fn decl")?;
        let name = self.expect_ident("fn name")?;
        let mutates_self = name.ends_with('!') || self.eat(&TokenKind::Bang);
        let shape_params = if matches!(self.peek(), TokenKind::LBracket) {
            self.parse_shape_params()?
        } else { Vec::new() };
        self.expect(&TokenKind::LParen, "before fn params")?;
        let params = if matches!(self.peek(), TokenKind::RParen) {
            Vec::new()
        } else {
            self.parse_params()?
        };
        self.expect(&TokenKind::RParen, "after fn params")?;
        // #446: a multi-line parameter list often wants the return arrow on
        // its own line. Newlines are insignificant inside `( )`, so the one
        // *after* `)` used to end the signature early and the body's `{` then
        // read as missing. `eat_over_newlines` restores position when the
        // next non-newline token is not `->`, so a genuinely absent return
        // type is unaffected.
        let ret_type = if self.eat_over_newlines(&TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else { None };
        self.expect_fn_body_brace(&name)?;
        let body = self.parse_block()?;
        Ok(FnDecl {
            directives, name, mutates_self, shape_params, params, ret_type, body,
            span: self.span_from(&start),
        })
    }

    fn parse_extern_fn_decl(&mut self) -> ParseResult<ExternFnDecl> {
        let start = self.peek_span();
        self.expect(&TokenKind::Extern, "in extern fn decl")?;
        // Optional ABI string: extern "cuda" fn ...
        let abi = if let TokenKind::StrLit(s) = self.peek().clone() {
            self.advance();
            Some(s)
        } else {
            None
        };
        self.expect(&TokenKind::Fn, "in extern fn decl")?;
        let name = self.expect_ident("extern fn name")?;
        let shape_params = if matches!(self.peek(), TokenKind::LBracket) {
            self.parse_shape_params()?
        } else { Vec::new() };
        self.expect(&TokenKind::LParen, "before extern fn params")?;
        let params = if matches!(self.peek(), TokenKind::RParen) {
            Vec::new()
        } else {
            self.parse_params()?
        };
        self.expect(&TokenKind::RParen, "after extern fn params")?;
        // #446: same wrapped-signature allowance as `parse_fn_decl`.
        let ret_type = if self.eat_over_newlines(&TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else { None };
        Ok(ExternFnDecl { abi, name, shape_params, params, ret_type, span: self.span_from(&start) })
    }

    fn parse_shape_params(&mut self) -> ParseResult<Vec<ShapeParam>> {
        self.expect(&TokenKind::LBracket, "in shape params")?;
        let mut out = Vec::new();
        if !matches!(self.peek(), TokenKind::RBracket) {
            loop {
                let start = self.peek_span();
                let name = self.expect_ident("shape param name")?;
                let default = if self.eat(&TokenKind::Eq) {
                    self.skip_newlines();
                    Some(self.parse_expr()?)
                } else { None };
                out.push(ShapeParam { name, default, span: self.span_from(&start) });
                if !self.eat(&TokenKind::Comma) { break; }
            }
        }
        self.expect(&TokenKind::RBracket, "after shape params")?;
        Ok(out)
    }

    fn parse_params(&mut self) -> ParseResult<Vec<Param>> {
        let mut out = Vec::new();
        loop {
            let start = self.peek_span();
            let mutating = self.eat(&TokenKind::Bang);
            let (is_self, name) = if self.eat(&TokenKind::SelfKw) {
                (true, "self".to_string())
            } else {
                (false, self.expect_ident("param name")?)
            };
            let ty = if self.eat(&TokenKind::Colon) {
                Some(self.parse_type()?)
            } else { None };
            out.push(Param { mutating, is_self, name, ty, span: self.span_from(&start) });
            // allow trailing newlines/comma; stop at )
            self.skip_newlines();
            if !self.eat(&TokenKind::Comma) { break; }
            self.skip_newlines();
            if matches!(self.peek(), TokenKind::RParen) { break; }
        }
        Ok(out)
    }

    fn parse_model_decl(&mut self, directives: Vec<Directive>) -> ParseResult<ModelDecl> {
        let start = self.peek_span();
        self.expect(&TokenKind::Model, "in model decl")?;
        let name = self.expect_ident("model name")?;
        let shape_params = if matches!(self.peek(), TokenKind::LBracket) {
            self.parse_shape_params()?
        } else { Vec::new() };
        self.expect(&TokenKind::LBrace, "before model body")?;
        let mut members = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
            members.push(self.parse_model_member()?);
            self.skip_newlines();
            // optional separating comma between fields
            if self.eat(&TokenKind::Comma) { self.skip_newlines(); }
        }
        self.expect(&TokenKind::RBrace, "after model body")?;
        Ok(ModelDecl { directives, name, shape_params, members, span: self.span_from(&start) })
    }

    fn parse_model_member(&mut self) -> ParseResult<ModelMember> {
        if matches!(self.peek(), TokenKind::Fn) {
            Ok(ModelMember::Method(self.parse_fn_decl(Vec::new())?))
        } else {
            let start = self.peek_span();
            let mutating = self.eat(&TokenKind::Bang);
            // #235: accept keyword-as-field-name (`type`, `shape`, `dtype`, …) — the
            // same set field *access* already allows. Every tokenizer's Token has a
            // `type` field; `type` is reserved (type aliases) but is fine as a member.
            let name = self.parse_field_name()?;
            self.expect(&TokenKind::Colon, "after model field name")?;
            let ty = self.parse_type()?;
            Ok(ModelMember::Field { mutating, name, ty, span: self.span_from(&start) })
        }
    }

    /// `enum Color { Red, Green, Blue }` (#336). Variants are bare identifiers,
    /// newline- or comma-separated, with an optional trailing separator.
    fn parse_enum_decl(&mut self) -> ParseResult<EnumDecl> {
        let start = self.peek_span();
        self.expect(&TokenKind::Enum, "in enum declaration")?;
        let name = self.expect_ident("enum name")?;
        self.expect(&TokenKind::LBrace, "before enum variants")?;
        self.skip_newlines();
        let mut variants: Vec<EnumVariant> = Vec::new();
        while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
            let vstart = self.peek_span();
            let vname = self.expect_ident("enum variant name")?;
            if variants.iter().any(|v| v.name == vname) {
                return Err(self.err(&format!("duplicate enum variant `{}`", vname)));
            }
            // #350 Part 2: optional positional payload `Circle(f32)` /
            // `Rect(f32, f32)`. No parens = a tag-only C-like variant (#336).
            let mut fields = Vec::new();
            if self.eat(&TokenKind::LParen) {
                self.skip_newlines();
                if matches!(self.peek(), TokenKind::RParen) {
                    return Err(self.err("payload-carrying enum variant needs at least one field type (or drop the parens for a tag-only variant)"));
                }
                loop {
                    fields.push(self.parse_type()?);
                    self.skip_newlines();
                    if !self.eat(&TokenKind::Comma) { break; }
                    self.skip_newlines();
                    if matches!(self.peek(), TokenKind::RParen) { break; }
                }
                self.expect(&TokenKind::RParen, "after enum variant payload types")?;
            }
            variants.push(EnumVariant { name: vname, fields, span: self.span_from(&vstart) });
            self.skip_newlines();
            if !self.eat(&TokenKind::Comma) { break; }
            self.skip_newlines();
        }
        self.expect(&TokenKind::RBrace, "after enum variants")?;
        if variants.is_empty() {
            return Err(self.err("enum must have at least one variant"));
        }
        Ok(EnumDecl { name, variants, span: self.span_from(&start) })
    }

    fn parse_type_alias(&mut self) -> ParseResult<TypeAlias> {
        let start = self.peek_span();
        self.expect(&TokenKind::Type, "in type alias")?;
        let name = self.expect_ident("type name")?;
        let shape_params = if matches!(self.peek(), TokenKind::LBracket) {
            self.parse_shape_params()?
        } else { Vec::new() };
        self.expect(&TokenKind::Eq, "in type alias")?;
        self.skip_newlines();
        let ty = self.parse_type()?;
        Ok(TypeAlias { name, shape_params, ty, span: self.span_from(&start) })
    }

    fn parse_arena_block(&mut self) -> ParseResult<ArenaBlock> {
        let start = self.peek_span();
        let kind = match self.advance().kind {
            TokenKind::Vault  => ArenaKind::Vault,
            TokenKind::Forge  => ArenaKind::Forge,
            TokenKind::Stream => ArenaKind::Stream,
            _ => unreachable!(),
        };
        let body = self.parse_block()?;
        Ok(ArenaBlock { kind, body, span: self.span_from(&start) })
    }

    // ─── Directives ───────────────────────────────────────────────────────

    fn parse_directive(&mut self) -> ParseResult<Directive> {
        let start = self.peek_span();
        self.expect(&TokenKind::At, "in directive")?;
        let name = self.expect_ident("directive name")?;
        let mut args = Vec::new();
        if self.eat(&TokenKind::LParen) {
            if !matches!(self.peek(), TokenKind::RParen) {
                loop {
                    args.push(self.parse_darg()?);
                    if !self.eat(&TokenKind::Comma) { break; }
                }
            }
            self.expect(&TokenKind::RParen, "after directive args")?;
        }
        Ok(Directive { name, args, span: self.span_from(&start) })
    }

    fn parse_darg(&mut self) -> ParseResult<DArg> {
        let start = self.peek_span();
        // try `ident "=" expr` first
        if matches!(self.peek(), TokenKind::Ident(_))
            && matches!(self.peek_at(1), TokenKind::Eq)
        {
            let name = self.expect_ident("darg name")?;
            self.advance(); // eat =
            self.skip_newlines();
            let value = self.parse_expr()?;
            return Ok(DArg::Named { name, value, span: self.span_from(&start) });
        }
        Ok(DArg::Positional(self.parse_expr()?))
    }

    // ─── Statements / Block ───────────────────────────────────────────────

    /// #446: report a missing function-body brace against the SIGNATURE.
    /// `parse_block`'s generic "expected LBrace" sent readers looking at the
    /// body when the real problem was that the signature ended early.
    fn expect_fn_body_brace(&mut self, what: &str) -> ParseResult<()> {
        if matches!(self.peek(), TokenKind::Newline) {
            return Err(self.err(format!(
                "`{}`: expected `{{` to open the function body, found end of line \
                 — the opening brace must be on the same line as the end of the \
                 signature (a wrapped parameter list may put `->` on its own line)",
                what)));
        }
        Ok(())
    }

    fn parse_block(&mut self) -> ParseResult<Block> {
        let start = self.peek_span();
        self.expect(&TokenKind::LBrace, "before block")?;
        let mut stmts = Vec::new();
        let mut tail_expr: Option<Box<Expr>> = None;
        self.skip_newlines();
        while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
            // Try to parse a stmt. Stmts that start with a keyword are
            // unambiguous. Otherwise we parse an expr; if followed by an
            // assign-op or `<-`, it's an Expr stmt; else it's the trailing
            // expression of the block.
            let stmt_start = self.peek_span();
            if let Some(s) = self.try_parse_keyword_stmt()? {
                stmts.push(s);
            } else {
                // expression — may be tail or expr-stmt
                let e = self.parse_expr()?;
                let assign = self.try_parse_assign_tail()?;
                // Detect tail: if no assign and we're at } directly (after newlines), it's tail
                self.skip_newlines();
                if assign.is_none() && matches!(self.peek(), TokenKind::RBrace) {
                    tail_expr = Some(Box::new(e));
                    break;
                }
                stmts.push(Stmt::Expr {
                    lhs: e, assign,
                    span: self.span_from(&stmt_start),
                });
            }
            // consume one terminator (semicolon or newline) then skip more
            if self.eat(&TokenKind::Semicolon) {
                self.skip_newlines();
            } else if matches!(self.peek(), TokenKind::Newline) {
                self.skip_newlines();
            } else if !matches!(self.peek(), TokenKind::RBrace) {
                // be lenient: allow no terminator if next is `}` already
                self.skip_newlines();
            }
        }
        self.expect(&TokenKind::RBrace, "after block")?;
        Ok(Block { stmts, tail_expr, span: self.span_from(&start) })
    }

    /// Returns Some(stmt) if the current token starts an unambiguous keyword-led stmt.
    fn try_parse_keyword_stmt(&mut self) -> ParseResult<Option<Stmt>> {
        let start = self.peek_span();
        let s = match self.peek() {
            TokenKind::Let      => Some(Stmt::Let(self.parse_let_stmt()?)),
            TokenKind::If       => {
                let if_e = self.parse_if_expr()?;
                Some(Stmt::If(if_e))
            }
            TokenKind::Match    => Some(Stmt::Match(self.parse_match_expr()?)),
            TokenKind::For      => Some(self.parse_for_stmt()?),
            TokenKind::While    => Some(self.parse_while_stmt()?),
            TokenKind::Loop     => Some(self.parse_loop_stmt()?),
            TokenKind::Stage    => Some(self.parse_stage_stmt()?),
            TokenKind::Break    => { self.advance(); Some(Stmt::Break(self.span_from(&start))) }
            TokenKind::Continue => { self.advance(); Some(Stmt::Continue(self.span_from(&start))) }
            TokenKind::Return   => {
                self.advance();
                let value = if matches!(self.peek(), TokenKind::Semicolon | TokenKind::Newline | TokenKind::RBrace) {
                    None
                } else { Some(self.parse_expr()?) };
                Some(Stmt::Return { value, span: self.span_from(&start) })
            }
            TokenKind::Vault | TokenKind::Forge | TokenKind::Stream
                if matches!(self.peek_at(1), TokenKind::LBrace) =>
            {
                // arena block as a statement — only when followed by `{`.
                // Otherwise `vault.zeros[...]` / `forge.reset()` etc. are
                // expressions starting with the arena keyword as a value.
                let arena = self.parse_arena_block()?;
                Some(Stmt::Expr {
                    lhs: Expr::ArenaBlock(arena),
                    assign: None,
                    span: self.span_from(&start),
                })
            }
            TokenKind::At       => {
                // directive-prefixed stmt: directives then stmt or block
                let mut directives = Vec::new();
                while matches!(self.peek(), TokenKind::At) {
                    directives.push(self.parse_directive()?);
                    self.skip_newlines();
                }
                if matches!(self.peek(), TokenKind::LBrace) {
                    let body = self.parse_block()?;
                    Some(Stmt::DirectiveBlock { directives, body, span: self.span_from(&start) })
                } else if let Some(inner) = self.try_parse_keyword_stmt()? {
                    Some(Stmt::Directive { directives, inner: Box::new(inner), span: self.span_from(&start) })
                } else {
                    // fall back to expr-stmt
                    let e = self.parse_expr()?;
                    let assign = self.try_parse_assign_tail()?;
                    Some(Stmt::Directive {
                        directives,
                        inner: Box::new(Stmt::Expr { lhs: e, assign, span: self.span_from(&start) }),
                        span: self.span_from(&start),
                    })
                }
            }
            _ => None,
        };
        Ok(s)
    }

    fn parse_let_stmt(&mut self) -> ParseResult<LetStmt> {
        let start = self.peek_span();
        self.expect(&TokenKind::Let, "in let")?;
        let mutating = self.eat(&TokenKind::Bang);
        let is_mut = !mutating && self.eat(&TokenKind::Mut);
        let pattern = self.parse_pattern()?;
        let ty = if self.eat(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else { None };
        self.expect(&TokenKind::Eq, "after let pattern/type")?;
        self.skip_newlines();
        let value = self.parse_expr()?;
        Ok(LetStmt { mutating, is_mut, pattern, ty, value, span: self.span_from(&start) })
    }

    fn try_parse_assign_tail(&mut self) -> ParseResult<Option<(AssignOp, Expr)>> {
        let op = match self.peek() {
            TokenKind::Eq          => AssignOp::Eq,
            TokenKind::ColonEq     => AssignOp::ColonEq,
            TokenKind::PlusEq      => AssignOp::PlusEq,
            TokenKind::MinusEq     => AssignOp::MinusEq,
            TokenKind::StarEq      => AssignOp::StarEq,
            TokenKind::SlashEq     => AssignOp::SlashEq,
            TokenKind::StreamArrow => AssignOp::StreamArrow,
            TokenKind::AmpEq       => AssignOp::AmpEq,
            TokenKind::BarEq       => AssignOp::BarEq,
            TokenKind::CaretEq     => AssignOp::CaretEq,
            _ => return Ok(None),
        };
        self.advance();
        self.skip_newlines();
        let rhs = self.parse_expr()?;
        Ok(Some((op, rhs)))
    }

    fn parse_for_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.peek_span();
        self.expect(&TokenKind::For, "in for")?;
        let pattern = self.parse_pattern()?;
        // Allow `in` as a bare ident; the EBNF uses `in` as a keyword but the
        // lexer doesn't reserve it. We accept Ident("in").
        match self.peek() {
            TokenKind::Ident(s) if s == "in" => { self.advance(); }
            _ => return Err(self.err("expected `in` in for stmt")),
        }
        let iter = self.parse_expr_no_struct()?;
        let body = self.parse_block()?;
        Ok(Stmt::For { pattern, iter, body, span: self.span_from(&start) })
    }

    fn parse_while_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.peek_span();
        self.expect(&TokenKind::While, "in while")?;
        let cond = self.parse_expr_no_struct()?;
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body, span: self.span_from(&start) })
    }

    fn parse_loop_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.peek_span();
        self.expect(&TokenKind::Loop, "in loop")?;
        let body = self.parse_block()?;
        Ok(Stmt::Loop { body, span: self.span_from(&start) })
    }

    fn parse_stage_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.peek_span();
        self.expect(&TokenKind::Stage, "in stage")?;
        let stage = match self.advance().kind {
            TokenKind::IntLit(n, _) => n,
            other => return Err(self.err(format!("expected int after `stage`, found {:?}", other))),
        };
        self.expect(&TokenKind::Colon, "after stage number")?;
        let body = self.parse_expr()?;
        Ok(Stmt::Stage { stage, body, span: self.span_from(&start) })
    }

    // ─── If / Match ──────────────────────────────────────────────────────

    fn parse_if_expr(&mut self) -> ParseResult<IfExpr> {
        let start = self.peek_span();
        self.expect(&TokenKind::If, "in if")?;
        let cond = self.parse_expr_no_struct()?;
        let then_branch = self.parse_block()?;
        let else_branch = if self.eat_over_newlines(&TokenKind::Else) {
            if matches!(self.peek(), TokenKind::If) {
                Some(ElseBranch::If(Box::new(self.parse_if_expr()?)))
            } else {
                Some(ElseBranch::Block(self.parse_block()?))
            }
        } else { None };
        Ok(IfExpr { cond, then_branch, else_branch, span: self.span_from(&start) })
    }

    fn parse_match_expr(&mut self) -> ParseResult<MatchExpr> {
        let start = self.peek_span();
        self.expect(&TokenKind::Match, "in match")?;
        // Scrutinee is optional when `{` follows immediately — e.g. `@host match { ... }`
        // uses an implicit host-feature dispatch target (Spec §7.3).
        let scrutinee = if matches!(self.peek(), TokenKind::LBrace) {
            Expr::Nil(self.peek_span())
        } else {
            self.parse_expr_no_struct()?
        };
        self.expect(&TokenKind::LBrace, "before match arms")?;
        let mut arms = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek(), TokenKind::RBrace | TokenKind::Eof) {
            let arm_start = self.peek_span();
            let pattern = self.parse_pattern()?;
            let guard = if self.eat(&TokenKind::If) {
                Some(self.parse_expr()?)
            } else { None };
            self.expect(&TokenKind::FatArrow, "in match arm")?;
            let body = self.parse_expr()?;
            arms.push(MatchArm { pattern, guard, body, span: self.span_from(&arm_start) });
            self.skip_newlines();
            if !self.eat(&TokenKind::Comma) { break; }
            self.skip_newlines();
        }
        self.expect(&TokenKind::RBrace, "after match arms")?;
        Ok(MatchExpr { scrutinee, arms, span: self.span_from(&start) })
    }

    // ─── Patterns ────────────────────────────────────────────────────────

    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        let start = self.peek_span();
        let base = self.parse_pattern_atom()?;
        // `pat @ pat` bind form
        if matches!(self.peek(), TokenKind::At) {
            // only treat as bind if next token can start a pattern (not @ident which is directive — but @ inside pattern always means bind)
            self.advance();
            let rhs = self.parse_pattern()?;
            return Ok(Pattern::Bind(Box::new(base), Box::new(rhs), self.span_from(&start)));
        }
        Ok(base)
    }

    /// Parse the optional `( pat, pat, ... )` payload-binding list of an
    /// enum-variant pattern (#350 Part 2). Returns an empty vec when there is
    /// no `(` — a tag-only variant pattern. An empty `()` is rejected: a payload
    /// variant always has at least one field.
    fn parse_variant_pattern_bindings(&mut self) -> ParseResult<Vec<Pattern>> {
        if !self.eat(&TokenKind::LParen) {
            return Ok(Vec::new());
        }
        if matches!(self.peek(), TokenKind::RParen) {
            return Err(self.err("payload-variant pattern needs at least one binding (or drop the parens for a tag-only variant)"));
        }
        let mut bindings = Vec::new();
        loop {
            bindings.push(self.parse_pattern()?);
            if !self.eat(&TokenKind::Comma) { break; }
            if matches!(self.peek(), TokenKind::RParen) { break; }
        }
        self.expect(&TokenKind::RParen, "after payload-variant pattern bindings")?;
        Ok(bindings)
    }

    fn parse_pattern_atom(&mut self) -> ParseResult<Pattern> {
        let start = self.peek_span();
        match self.peek().clone() {
            TokenKind::Ident(s) if s == "_" => {
                self.advance();
                Ok(Pattern::Wildcard(self.span_from(&start)))
            }
            TokenKind::Ident(s) => {
                self.advance();
                // `Color.Red` — qualified enum-variant pattern (#336). The bare
                // tag-only form `Red` stays an `Ident` and is resolved by the
                // checker.
                if matches!(self.peek(), TokenKind::Dot) {
                    self.advance();
                    let variant = self.expect_ident("variant name after `.` in enum pattern")?;
                    // #350 Part 2: optional payload bindings `Shape.Circle(r)`.
                    let bindings = self.parse_variant_pattern_bindings()?;
                    return Ok(Pattern::EnumVariant { enum_name: s, variant, bindings, span: self.span_from(&start) });
                }
                // #350 Part 2: bare payload form `Circle(r)` — an ident directly
                // followed by `(`. The enum is resolved by the checker against
                // the scrutinee (empty `enum_name`). A bare ident *without* parens
                // stays an `Ident` (tag-only variant or catch-all bind, as #336).
                if matches!(self.peek(), TokenKind::LParen) {
                    let bindings = self.parse_variant_pattern_bindings()?;
                    return Ok(Pattern::EnumVariant { enum_name: String::new(), variant: s, bindings, span: self.span_from(&start) });
                }
                Ok(Pattern::Ident(s, self.span_from(&start)))
            }
            TokenKind::IntLit(n, ref suffix) => {
                let ty = parse_int_suffix(suffix);
                self.advance();
                Ok(Pattern::Literal(Literal::Int(n, ty), self.span_from(&start)))
            }
            TokenKind::FloatLit(f, ref suffix) => {
                self.advance();
                let ty = parse_float_suffix(suffix);
                Ok(Pattern::Literal(Literal::Float(f, ty), self.span_from(&start)))
            }
            TokenKind::StrLit(s) => {
                self.advance();
                Ok(Pattern::Literal(Literal::Str(s), self.span_from(&start)))
            }
            TokenKind::CharLit(c) => {
                self.advance();
                Ok(Pattern::Literal(Literal::Char(c), self.span_from(&start)))
            }
            TokenKind::True  => { self.advance(); Ok(Pattern::Literal(Literal::Bool(true), self.span_from(&start))) }
            TokenKind::False => { self.advance(); Ok(Pattern::Literal(Literal::Bool(false), self.span_from(&start))) }
            TokenKind::Nil   => { self.advance(); Ok(Pattern::Literal(Literal::Nil, self.span_from(&start))) }
            TokenKind::LParen => {
                self.advance();
                let mut elems = Vec::new();
                if !matches!(self.peek(), TokenKind::RParen) {
                    loop {
                        elems.push(self.parse_pattern()?);
                        if !self.eat(&TokenKind::Comma) { break; }
                        if matches!(self.peek(), TokenKind::RParen) { break; }
                    }
                }
                self.expect(&TokenKind::RParen, "after tuple pattern")?;
                Ok(Pattern::Tuple(elems, self.span_from(&start)))
            }
            TokenKind::LBracket => {
                self.advance();
                let elems = self.parse_shape_elems(TokenKind::RBracket, ShapeCtx::Pattern)?;
                self.expect(&TokenKind::RBracket, "after shape pattern")?;
                Ok(Pattern::Shape(elems, self.span_from(&start)))
            }
            // `..` rest / catch-all pattern (Spec §4.5). Standalone it is a
            // catch-all; inside a tuple it absorbs the unmatched middle. Kept as
            // a distinct `Rest` (not `Wildcard`) so `(a, ..)` is not confused with
            // the fixed-arity `(a, _)` — the two mean different things (#393).
            TokenKind::DotDot => {
                self.advance();
                Ok(Pattern::Rest(self.span_from(&start)))
            }
            // `.variant` — enum/feature-flag patterns (e.g. `@host match { .avx2 => ... }`)
            TokenKind::Dot => {
                self.advance();
                let name = self.expect_ident("variant name after `.` in pattern")?;
                Ok(Pattern::Ident(format!(".{}", name), self.span_from(&start)))
            }
            // Negative integer literal patterns: `-N`
            TokenKind::Minus => {
                self.advance();
                match self.advance().kind {
                    TokenKind::IntLit(n, suffix) => Ok(Pattern::Literal(Literal::Int(-n, parse_int_suffix(&suffix)), self.span_from(&start))),
                    TokenKind::FloatLit(f, suffix) => {
                        let ty = parse_float_suffix(&suffix);
                        Ok(Pattern::Literal(Literal::Float(-f, ty), self.span_from(&start)))
                    }
                    other => Err(self.err(format!("expected number after `-` in pattern, found {:?}", other))),
                }
            }
            other => Err(self.err(format!("expected pattern, found {:?}", other))),
        }
    }

    fn parse_shape_elems(&mut self, terminator: TokenKind, ctx: ShapeCtx) -> ParseResult<Vec<ShapeElem>> {
        let mut out = Vec::new();
        while std::mem::discriminant(self.peek()) != std::mem::discriminant(&terminator) {
            let start = self.peek_span();
            let elem = match self.peek() {
                // `_` is the wildcard *pattern*, legal only in a shape pattern. As a
                // dimension inside a type it was a second spelling of `?`; removed in
                // the pre-0.1.0 redundancy sweep (#501, ruling S3).
                TokenKind::Ident(s) if s == "_" => match ctx {
                    ShapeCtx::Pattern => { self.advance(); ShapeElem::Wildcard(self.span_from(&start)) }
                    ShapeCtx::Type    => return Err(self.err(UNDERSCORE_NOT_A_DIM)),
                },
                TokenKind::DotDot               => { self.advance(); ShapeElem::Spread(self.span_from(&start)) }
                TokenKind::Tilde                => { self.advance(); ShapeElem::Streaming(self.span_from(&start)) }
                // `?` inside a shape literal is the dynamic-dimension escape hatch (Spec §3.2)
                TokenKind::Query                => { self.advance(); ShapeElem::Wildcard(self.span_from(&start)) }
                _ => {
                    let e = self.parse_expr()?;
                    // A leading `_` is caught above; this catches every other
                    // position it can hide in — `(_)`, `1 + _`, `-_`, `_ as i64`
                    // — so the break is total rather than a first-token check.
                    // Such an element used to parse and then reject every
                    // argument shape as `SymDim::Unknown`; the refusal belongs
                    // at the spelling, not three phases later.
                    if ctx == ShapeCtx::Type && crate::ast::expr_contains_underscore(&e) {
                        return Err(self.err_at(&start, UNDERSCORE_NOT_A_DIM));
                    }
                    ShapeElem::Expr(Box::new(e))
                }
            };
            out.push(elem);
            if !self.eat(&TokenKind::Comma) { break; }
        }
        Ok(out)
    }

    // ─── Types ───────────────────────────────────────────────────────────

    fn parse_type(&mut self) -> ParseResult<Type> {
        let start = self.peek_span();
        match self.peek().clone() {
            TokenKind::Fn => {
                self.advance();
                self.expect(&TokenKind::LParen, "after fn in fn type")?;
                let mut args = Vec::new();
                if !matches!(self.peek(), TokenKind::RParen) {
                    loop {
                        args.push(self.parse_type()?);
                        if !self.eat(&TokenKind::Comma) { break; }
                    }
                }
                self.expect(&TokenKind::RParen, "after fn type args")?;
                self.skip_newlines();   // #446
                self.expect(&TokenKind::Arrow, "in fn type")?;
                let ret = self.parse_type()?;
                Ok(Type::Fn(args, Box::new(ret), self.span_from(&start)))
            }
            TokenKind::LParen => {
                // tuple type
                self.advance();
                let mut elems = Vec::new();
                if !matches!(self.peek(), TokenKind::RParen) {
                    loop {
                        elems.push(self.parse_type()?);
                        if !self.eat(&TokenKind::Comma) { break; }
                        if matches!(self.peek(), TokenKind::RParen) { break; }
                    }
                }
                self.expect(&TokenKind::RParen, "after tuple type")?;
                Ok(Type::Tuple(elems, self.span_from(&start)))
            }
            TokenKind::LBracket => {
                // `[T; N]` array type — pragmatic extension
                self.advance();
                let elem_ty = self.parse_type()?;
                self.expect(&TokenKind::Semicolon, "in array type [T; N]")?;
                let size_expr = self.parse_expr()?;
                self.expect(&TokenKind::RBracket, "after array type")?;
                Ok(Type::Array(Box::new(elem_ty), Box::new(size_expr), self.span_from(&start)))
            }
            TokenKind::Star => {
                // `*T` raw pointer — extern fn boundary only (§3.12)
                self.advance();
                let inner = self.parse_type()?;
                Ok(Type::RawPtr(Box::new(inner), self.span_from(&start)))
            }
            k if Self::is_scalar_type_kind(&k) => {
                self.advance();
                Ok(Type::Scalar(Self::scalar_from_kind(&k), self.span_from(&start)))
            }
            TokenKind::Ident(name) => {
                self.advance();
                match name.as_str() {
                    "Tensor" | "View" | "KV" => {
                        self.expect(&TokenKind::LBracket, "after tensor-like type")?;
                        let inner = self.parse_type()?;
                        self.expect(&TokenKind::Comma, "in tensor-like type args")?;
                        self.expect(&TokenKind::LBracket, "before shape spec")?;
                        let elems_start = self.peek_span();
                        let elems = self.parse_shape_elems(TokenKind::RBracket, ShapeCtx::Type)?;
                        self.expect(&TokenKind::RBracket, "after shape spec")?;
                        self.expect(&TokenKind::RBracket, "after tensor-like type")?;
                        let shape = ShapeSpec { elems, span: self.span_from(&elems_start) };
                        let span = self.span_from(&start);
                        match name.as_str() {
                            "Tensor" => Ok(Type::Tensor(Box::new(inner), shape, span)),
                            "View"   => Ok(Type::View  (Box::new(inner), shape, span)),
                            "KV"     => Ok(Type::KV    (Box::new(inner), shape, span)),
                            _ => unreachable!(),
                        }
                    }
                    "Mesh" => {
                        self.expect(&TokenKind::LBracket, "after Mesh")?;
                        let mut axes = Vec::new();
                        if !matches!(self.peek(), TokenKind::RBracket) {
                            loop {
                                let ax_start = self.peek_span();
                                let ax_name = self.expect_ident("mesh axis name")?;
                                self.expect(&TokenKind::Eq, "after mesh axis name")?;
                                self.skip_newlines();
                                let size = self.parse_expr()?;
                                axes.push(MeshAxis { name: ax_name, size: Box::new(size), span: self.span_from(&ax_start) });
                                if !self.eat(&TokenKind::Comma) { break; }
                            }
                        }
                        self.expect(&TokenKind::RBracket, "after Mesh axes")?;
                        Ok(Type::Mesh(axes, self.span_from(&start)))
                    }
                    _ => {
                        // generic named type
                        let mut args = Vec::new();
                        if self.eat(&TokenKind::LBracket) {
                            if !matches!(self.peek(), TokenKind::RBracket) {
                                loop {
                                    args.push(self.parse_type_arg()?);
                                    if !self.eat(&TokenKind::Comma) { break; }
                                }
                            }
                            self.expect(&TokenKind::RBracket, "after type args")?;
                        }
                        Ok(Type::Named { name, args, span: self.span_from(&start) })
                    }
                }
            }
            other => Err(self.err(format!("expected type, found {:?}", other))),
        }
    }

    fn parse_type_arg(&mut self) -> ParseResult<TypeArg> {
        let start = self.peek_span();
        // `ident "=" expr` — named
        if matches!(self.peek(), TokenKind::Ident(_))
            && matches!(self.peek_at(1), TokenKind::Eq)
        {
            let name = self.expect_ident("type arg name")?;
            self.advance(); // =
            self.skip_newlines();
            let value = self.parse_expr()?;
            return Ok(TypeArg::Named { name, value: Box::new(value), span: self.span_from(&start) });
        }
        // try type first; if that fails fall back to expr.
        // Cheap heuristic: if current token starts a type, parse type; else expr.
        if self.token_can_start_type() {
            Ok(TypeArg::Type(self.parse_type()?))
        } else {
            Ok(TypeArg::Expr(Box::new(self.parse_expr()?)))
        }
    }

    fn token_can_start_type(&self) -> bool {
        matches!(self.peek(),
            TokenKind::Fn | TokenKind::LParen | TokenKind::LBracket
        ) || Self::is_scalar_type_kind(self.peek())
            || matches!(self.peek(), TokenKind::Ident(_))
    }

    fn is_scalar_type_kind(k: &TokenKind) -> bool {
        matches!(k,
            TokenKind::I8 | TokenKind::I16 | TokenKind::I32 | TokenKind::I64 |
            TokenKind::U8 | TokenKind::U16 | TokenKind::U32 | TokenKind::U64 |
            TokenKind::Int4 | TokenKind::Int8 |
            TokenKind::F16 | TokenKind::Bf16 | TokenKind::Tf32 | TokenKind::F32 | TokenKind::F64 |
            TokenKind::Fp8E4M3 | TokenKind::Fp8E5M2 | TokenKind::Trit |
            TokenKind::Bool | TokenKind::Str | TokenKind::Nil
        )
    }

    fn scalar_from_kind(k: &TokenKind) -> ScalarType {
        match k {
            TokenKind::I8 => ScalarType::I8, TokenKind::I16 => ScalarType::I16,
            TokenKind::I32 => ScalarType::I32, TokenKind::I64 => ScalarType::I64,
            TokenKind::U8 => ScalarType::U8, TokenKind::U16 => ScalarType::U16,
            TokenKind::U32 => ScalarType::U32, TokenKind::U64 => ScalarType::U64,
            TokenKind::Int4 => ScalarType::Int4,
            TokenKind::Int8 => ScalarType::Int8,
            TokenKind::F16 => ScalarType::F16, TokenKind::Bf16 => ScalarType::Bf16,
            TokenKind::Tf32 => ScalarType::Tf32, TokenKind::F32 => ScalarType::F32,
            TokenKind::F64 => ScalarType::F64,
            TokenKind::Fp8E4M3 => ScalarType::Fp8E4M3, TokenKind::Fp8E5M2 => ScalarType::Fp8E5M2,
            TokenKind::Trit => ScalarType::Trit,
            TokenKind::Bool => ScalarType::Bool, TokenKind::Str => ScalarType::Str,
            TokenKind::Nil => ScalarType::Nil,
            _ => unreachable!(),
        }
    }

    // ─── Expressions ─────────────────────────────────────────────────────
    //
    // Precedence ladder (low → high), per GRAMMAR.ebnf lines 97–110:
    //   pipe → logic_or → logic_and → compare → equality → range
    //        → sum → product → matmul → power → unary → postfix → primary
    //
    // `cast` (`expr as type`) is parsed as a postfix-level operator since
    // examples use it inside arithmetic (`(B as f32)`), which suggests it
    // binds tighter than arithmetic.

    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        // Depth guard (#214): every nesting level flows through here (the grouped
        // `(`/`[`/`{` forms in parse_primary recurse back into parse_expr), so a
        // single counter here bounds total recursion and prevents a stack-overflow
        // SIGABRT on pathologically nested input.
        self.depth += 1;
        if self.depth > MAX_EXPR_DEPTH {
            self.depth -= 1;
            return Err(self.err("expression nesting too deep"));
        }
        let result = self.parse_pipe_expr();
        self.depth -= 1;
        result
    }

    fn parse_expr_no_struct(&mut self) -> ParseResult<Expr> {
        let prev = self.struct_literal_allowed;
        self.struct_literal_allowed = false;
        let res = self.parse_expr();
        self.struct_literal_allowed = prev;
        res
    }

    fn parse_pipe_expr(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut lhs = self.parse_logic_or()?;
        loop {
            // Pipe operators commonly continue across newlines:
            //   x @ w1
            //       \|> _ .+ b1
            // Save position, skip newlines, check for op; if not found, restore.
            let saved = self.pos;
            self.skip_newlines();
            let op = match self.peek() {
                // `\|>` and the bare `|>` share TokenKind::Pipe (TOKENIZER §2–§3).
                // `>>` was a third spelling until #501 ruling S1a; it is the
                // right-shift operator since #530 and binds at the bitshift
                // level, so TokenKind::RShift never reaches the pipe level.
                TokenKind::Pipe    => BinOp::Pipe,
                _ => { self.pos = saved; break; }
            };
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_logic_or()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        }
        Ok(lhs)
    }

    fn parse_logic_or(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut lhs = self.parse_logic_and()?;
        loop {
            let saved = self.pos;
            self.skip_newlines();
            if !matches!(self.peek(), TokenKind::OrOr) { self.pos = saved; break; }
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_logic_and()?;
            lhs = Expr::BinOp { op: BinOp::Or, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        }
        Ok(lhs)
    }

    fn parse_logic_and(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut lhs = self.parse_compare()?;
        loop {
            let saved = self.pos;
            self.skip_newlines();
            if !matches!(self.peek(), TokenKind::AndAnd) { self.pos = saved; break; }
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_compare()?;
            lhs = Expr::BinOp { op: BinOp::And, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        }
        Ok(lhs)
    }

    fn parse_compare(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let lhs = self.parse_equality()?;
        let saved = self.pos;
        self.skip_newlines();
        let op = match self.peek() {
            TokenKind::Lt    => BinOp::Lt,
            TokenKind::Gt    => BinOp::Gt,
            TokenKind::LtEq  => BinOp::LtEq,
            TokenKind::GtEq  => BinOp::GtEq,
            TokenKind::DotLt => BinOp::DotLt,
            TokenKind::DotGt => BinOp::DotGt,
            TokenKind::DotLe => BinOp::DotLe,
            TokenKind::DotGe => BinOp::DotGe,
            _ => { self.pos = saved; return Ok(lhs); }
        };
        self.advance();
        self.skip_newlines();
        let rhs = self.parse_equality()?;
        let result = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        // Spec OPERATORS.md §1: `<`, `<=`, `>`, `>=` are none-associative — chains are parse errors.
        {
            let saved2 = self.pos;
            self.skip_newlines();
            if matches!(self.peek(),
                TokenKind::Lt | TokenKind::Gt | TokenKind::LtEq | TokenKind::GtEq |
                TokenKind::DotLt | TokenKind::DotGt | TokenKind::DotLe | TokenKind::DotGe
            ) {
                return Err(self.err(
                    "chained comparison is not allowed; `<`, `<=`, `>`, `>=` are none-associative\n  hint: use `&&`, e.g. `a < b && b < c`"
                ));
            }
            self.pos = saved2;
        }
        Ok(result)
    }

    fn parse_equality(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        // #279: range sits between equality and bitwise_or (GRAMMAR.ebnf:
        // `equality = range_e {...}`, `range_e = bitwise_or [...]`).
        let lhs = self.parse_range()?;
        let saved = self.pos;
        self.skip_newlines();
        let op = match self.peek() {
            TokenKind::EqEq   => BinOp::Eq,
            TokenKind::BangEq => BinOp::NotEq,
            _ => { self.pos = saved; return Ok(lhs); }
        };
        self.advance();
        self.skip_newlines();
        let rhs = self.parse_range()?;
        let result = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        // Spec OPERATORS.md §1: `==` and `!=` are none-associative.
        {
            let saved2 = self.pos;
            self.skip_newlines();
            if matches!(self.peek(), TokenKind::EqEq | TokenKind::BangEq) {
                return Err(self.err(
                    "chained equality is not allowed; `==` and `!=` are none-associative\n  hint: use `&&`, e.g. `a == b && b == c`"
                ));
            }
            self.pos = saved2;
        }
        Ok(result)
    }

    fn parse_bitwise_or(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut lhs = self.parse_bitwise_xor()?;
        loop {
            let saved = self.pos;
            self.skip_newlines();
            if !matches!(self.peek(), TokenKind::Bar) { self.pos = saved; break; }
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_bitwise_xor()?;
            lhs = Expr::BinOp { op: BinOp::BitOr, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        }
        Ok(lhs)
    }

    fn parse_bitwise_xor(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut lhs = self.parse_bitwise_and()?;
        loop {
            let saved = self.pos;
            self.skip_newlines();
            if !matches!(self.peek(), TokenKind::Caret) { self.pos = saved; break; }
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_bitwise_and()?;
            lhs = Expr::BinOp { op: BinOp::BitXor, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        }
        Ok(lhs)
    }

    fn parse_bitwise_and(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut lhs = self.parse_bitshift()?;
        loop {
            let saved = self.pos;
            self.skip_newlines();
            if !matches!(self.peek(), TokenKind::Amp) { self.pos = saved; break; }
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_bitshift()?;
            lhs = Expr::BinOp { op: BinOp::BitAnd, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        }
        Ok(lhs)
    }

    fn parse_bitshift(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut lhs = self.parse_sum()?;
        loop {
            let saved = self.pos;
            self.skip_newlines();
            let op = match self.peek() {
                TokenKind::LtLt => BinOp::BitShl,
                // #530: `>>` is the arithmetic right shift, the mirror of `<<`
                // and at the same precedence level (OPERATORS §1, row 12).
                // `BinOp::BitShr` — not the pipe-era `BinOp::RShift`, which
                // ruling S16 keeps reserved and which nothing constructs.
                TokenKind::RShift => BinOp::BitShr,
                _ => { self.pos = saved; break; }
            };
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_sum()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        }
        Ok(lhs)
    }

    fn parse_range(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let lhs = self.parse_bitwise_or()?;
        match self.peek() {
            TokenKind::DotDot => {
                self.advance();
                self.skip_newlines();
                // optional end expr — could be missing in `start..`
                if self.peek_can_start_expr() {
                    let end = self.parse_bitwise_or()?;
                    Ok(Expr::Range {
                        start: Some(Box::new(lhs)),
                        end: Some(Box::new(end)),
                        inclusive: false,
                        span: self.span_from(&start),
                    })
                } else {
                    Ok(Expr::Range { start: Some(Box::new(lhs)), end: None, inclusive: false, span: self.span_from(&start) })
                }
            }
            TokenKind::DotDotEq => {
                self.advance();
                self.skip_newlines();
                let end = self.parse_bitwise_or()?;
                Ok(Expr::Range {
                    start: Some(Box::new(lhs)),
                    end: Some(Box::new(end)),
                    inclusive: true,
                    span: self.span_from(&start),
                })
            }
            _ => Ok(lhs),
        }
    }

    fn parse_sum(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut lhs = self.parse_product()?;
        loop {
            let saved = self.pos;
            self.skip_newlines();
            let op = match self.peek() {
                TokenKind::Plus    => BinOp::Add,
                TokenKind::Minus   => BinOp::Sub,
                TokenKind::DotAdd  => BinOp::DotAdd,
                TokenKind::DotSub  => BinOp::DotSub,
                _ => { self.pos = saved; break; }
            };
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_product()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        }
        Ok(lhs)
    }

    fn parse_product(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut lhs = self.parse_matmul()?;
        loop {
            let saved = self.pos;
            self.skip_newlines();
            let op = match self.peek() {
                TokenKind::Star    => BinOp::Mul,
                TokenKind::Slash   => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                TokenKind::DotMul  => BinOp::DotMul,
                TokenKind::DotDiv  => BinOp::DotDiv,
                _ => { self.pos = saved; break; }
            };
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_matmul()?;
            lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        }
        Ok(lhs)
    }

    fn parse_matmul(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut lhs = self.parse_power()?;
        // `@` is dual-use: matmul in expressions, directive prefix at statement/item level.
        // Do NOT skip newlines before `@` — a leading `@` on the next line is a directive,
        // not a matmul continuation. The operator must appear on the same line as the LHS.
        while matches!(self.peek(), TokenKind::At) {
            self.advance();
            self.skip_newlines();
            let rhs = self.parse_power()?;
            lhs = Expr::BinOp { op: BinOp::Matmul, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) };
        }
        Ok(lhs)
    }

    fn parse_power(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let lhs = self.parse_unary()?;
        // #278: `**` and `.^` are right-associative (OPERATORS.md §1
        // row 16, §2.4): `a ** b ** c` is `a ** (b ** c)`. Recurse into
        // parse_power for the rhs instead of left-folding over parse_unary.
        let saved = self.pos;
        self.skip_newlines();
        let op = match self.peek() {
            TokenKind::StarStar => BinOp::StarStar,
            TokenKind::DotPow   => BinOp::DotPow,
            _ => { self.pos = saved; return Ok(lhs); }
        };
        self.advance();
        self.skip_newlines();
        let rhs = self.parse_power()?;
        Ok(Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span: self.span_from(&start) })
    }

    fn parse_unary(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        // `~` as bitwise NOT — but only when next token can start an expression.
        // In shape literal positions, `~` is streaming axis; parse_primary handles
        // it as Expr::Ident("~") which is used in shape contexts.
        if matches!(self.peek(), TokenKind::Tilde) {
            // Peek ahead: if what follows can start an expression, treat as BitNot.
            // But we need to be careful — in primary position `~` is a bare ident.
            // We intercept here in unary position to catch `~a`, `~(expr)`, etc.
            let saved = self.pos;
            self.advance(); // eat `~`
            if self.peek_can_start_expr() {
                let operand = self.parse_unary()?;
                return Ok(Expr::UnOp { op: UnOp::BitNot, operand: Box::new(operand), span: self.span_from(&start) });
            }
            // Not followed by expression — restore and fall through to primary
            self.pos = saved;
        }
        let op = match self.peek() {
            TokenKind::Minus => Some(UnOp::Neg),
            TokenKind::Bang  => Some(UnOp::Not),
            TokenKind::Star  => Some(UnOp::Deref),
            TokenKind::ReLU  => Some(UnOp::ReLU),
            TokenKind::GeLU  => Some(UnOp::GeLU),
            _ => None,
        };
        if let Some(op) = op {
            // ReLU/GeLU can also appear as standalone values inside a pipe
            // stage (`\|> \>`). If the next token can't start an expression,
            // treat as a bare identifier ("\\>" or "\\<") instead.
            let is_activation = matches!(op, UnOp::ReLU | UnOp::GeLU);
            self.advance();
            if is_activation && !self.peek_can_start_expr() {
                let name = if matches!(op, UnOp::ReLU) { "\\>" } else { "\\<" };
                return Ok(Expr::Ident(name.to_string(), self.span_from(&start)));
            }
            let operand = self.parse_unary()?;
            // Rewrite `-(expr as T)` → `(-expr) as T` so that narrowing casts like
            // `-5 as u8` produce the expected two's-complement wrapping (251), not
            // the identity result from negating an already-cast positive value.
            if matches!(op, UnOp::Neg) {
                if let Expr::Cast { expr: inner, ty, span: cast_span } = operand {
                    let neg_inner = Expr::UnOp {
                        op: UnOp::Neg,
                        operand: inner,
                        span: self.span_from(&start),
                    };
                    return Ok(Expr::Cast {
                        expr: Box::new(neg_inner),
                        ty,
                        span: cast_span,
                    });
                }
            }
            return Ok(Expr::UnOp { op, operand: Box::new(operand), span: self.span_from(&start) });
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek() {
                TokenKind::Transpose => {
                    self.advance();
                    expr = Expr::Postfix {
                        expr: Box::new(expr), op: PostfixOp::Transpose,
                        span: self.span_from(&start),
                    };
                }
                TokenKind::Query => {
                    self.advance();
                    expr = Expr::Postfix {
                        expr: Box::new(expr), op: PostfixOp::Query,
                        span: self.span_from(&start),
                    };
                }
                TokenKind::LBracket => {
                    self.advance();
                    let elems = self.parse_index_or_bracket_args()?;
                    self.expect(&TokenKind::RBracket, "after index/bracket args")?;
                    // Decide flavor: if any elem is a named arg, treat as BracketArgs;
                    // else as Index. Pre-alpha: both fields possible.
                    let any_named = elems.iter().any(|e| matches!(e, IndexOrArg::Named { .. }));
                    if any_named {
                        let args = elems.into_iter().map(IndexOrArg::into_call_arg).collect();
                        expr = Expr::Postfix {
                            expr: Box::new(expr), op: PostfixOp::BracketArgs(args),
                            span: self.span_from(&start),
                        };
                    } else {
                        let idx_elems = elems.into_iter().map(IndexOrArg::into_index_elem).collect();
                        expr = Expr::Postfix {
                            expr: Box::new(expr), op: PostfixOp::Index(idx_elems),
                            span: self.span_from(&start),
                        };
                    }
                }
                TokenKind::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    self.skip_newlines();
                    if !matches!(self.peek(), TokenKind::RParen) {
                        loop {
                            args.push(self.parse_call_arg()?);
                            self.skip_newlines();
                            if !self.eat(&TokenKind::Comma) { break; }
                            self.skip_newlines();
                            if matches!(self.peek(), TokenKind::RParen) { break; }
                        }
                    }
                    self.expect(&TokenKind::RParen, "after call args")?;
                    expr = Expr::Postfix {
                        expr: Box::new(expr), op: PostfixOp::Call(args),
                        span: self.span_from(&start),
                    };
                }
                TokenKind::Dot => {
                    self.advance();
                    // Accept idents OR keywords-as-field-names: many demoniC
                    // keywords (shape, view, dtype, etc.) are valid member names.
                    let name = self.parse_field_name()?;
                    expr = Expr::Postfix {
                        expr: Box::new(expr), op: PostfixOp::Field(name),
                        span: self.span_from(&start),
                    };
                }
                TokenKind::As => {
                    self.advance();
                    let ty = self.parse_type()?;
                    expr = Expr::Cast { expr: Box::new(expr), ty, span: self.span_from(&start) };
                }
                // Model constructor `Name { field: val, ... }` or `Name[T] { ... }`.
                // `struct_literal_allowed` prevents ambiguity with block expressions
                // after `for`/`while`/`match` scrutinee (set via parse_expr_no_struct).
                TokenKind::LBrace
                    if self.struct_literal_allowed && extract_ctor_name(&expr).is_some()
                => {
                    self.advance(); // eat `{`
                    let mut fields = Vec::new();
                    self.skip_newlines();
                    if !matches!(self.peek(), TokenKind::RBrace) {
                        loop {
                            // #235: keyword-as-field-name (`type`, …) in `T { type: v }`,
                            // matching the field declaration and access positions.
                            let field_name = self.parse_field_name()?;
                            self.expect(&TokenKind::Colon, "after constructor field name")?;
                            self.skip_newlines();
                            let value = self.parse_expr()?;
                            fields.push((field_name, value));
                            self.skip_newlines();
                            if !self.eat(&TokenKind::Comma) { break; }
                            self.skip_newlines();
                            if matches!(self.peek(), TokenKind::RBrace) { break; }
                        }
                    }
                    self.expect(&TokenKind::RBrace, "after constructor body")?;
                    let name = extract_ctor_name(&expr).unwrap();
                    let type_args = extract_ctor_type_args(&expr);
                    expr = Expr::StructLit { name, type_args, fields, span: self.span_from(&start) };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_call_arg(&mut self) -> ParseResult<CallArg> {
        let start = self.peek_span();
        // `...` spread (lexes as DotDot then Dot)
        if matches!(self.peek(), TokenKind::DotDot) && matches!(self.peek_at(1), TokenKind::Dot) {
            self.advance(); self.advance();
            return Ok(CallArg::Spread(self.span_from(&start)));
        }
        // named `ident = expr`
        if matches!(self.peek(), TokenKind::Ident(_))
            && matches!(self.peek_at(1), TokenKind::Eq)
        {
            let name = self.expect_ident("call arg name")?;
            self.advance(); // =
            self.skip_newlines();
            let value = self.parse_expr()?;
            return Ok(CallArg::Named { name, value, span: self.span_from(&start) });
        }
        Ok(CallArg::Positional(self.parse_expr()?))
    }

    fn parse_index_or_bracket_args(&mut self) -> ParseResult<Vec<IndexOrArg>> {
        let mut out = Vec::new();
        self.skip_newlines();
        if matches!(self.peek(), TokenKind::RBracket) { return Ok(out); }
        loop {
            out.push(self.parse_index_elem_or_arg()?);
            self.skip_newlines();
            if !self.eat(&TokenKind::Comma) { break; }
            self.skip_newlines();
            if matches!(self.peek(), TokenKind::RBracket) { break; }
        }
        Ok(out)
    }

    fn parse_index_elem_or_arg(&mut self) -> ParseResult<IndexOrArg> {
        let start = self.peek_span();
        // `..` = full axis
        if matches!(self.peek(), TokenKind::DotDot) {
            self.advance();
            return Ok(IndexOrArg::Idx(IndexElem::FullSlice(self.span_from(&start))));
        }
        // bare `:` or `:end` or `:end:step` slice forms where start is omitted
        if matches!(self.peek(), TokenKind::Colon) {
            self.advance();
            let end = if !matches!(self.peek(), TokenKind::Colon | TokenKind::Comma | TokenKind::RBracket) {
                Some(Box::new(self.parse_expr()?))
            } else { None };
            let step = if self.eat(&TokenKind::Colon) {
                if !matches!(self.peek(), TokenKind::Comma | TokenKind::RBracket) {
                    Some(Box::new(self.parse_expr()?))
                } else { None }
            } else { None };
            reject_range_bound(&start, [end.as_deref(), step.as_deref()])?;
            return Ok(IndexOrArg::Idx(IndexElem::Slice {
                start: None,
                end,
                step,
                span: self.span_from(&start),
            }));
        }
        // `::step` — start AND stop omitted, step present (lex: ColonColon).
        // `::` is one token, so `a[::-1]` never reaches the `Colon` branch
        // above; without this arm it falls through to `parse_expr` and dies
        // on "expected expression, found ColonColon" (#529). Mirrors the
        // `expr "::" expr` arm below: the step is mandatory here too, so
        // `a[::]` (no step at all) is still a parse error, same as `a[0::]`.
        if matches!(self.peek(), TokenKind::ColonColon) {
            self.advance();
            let step = self.parse_expr()?;
            reject_range_bound(&start, [Some(&step)])?;
            return Ok(IndexOrArg::Idx(IndexElem::Slice {
                start: None,
                end: None,
                step: Some(Box::new(step)),
                span: self.span_from(&start),
            }));
        }
        // named `ident = expr`
        if matches!(self.peek(), TokenKind::Ident(_))
            && matches!(self.peek_at(1), TokenKind::Eq)
        {
            let name = self.expect_ident("named bracket arg")?;
            self.advance(); // =
            self.skip_newlines();
            let value = self.parse_expr()?;
            return Ok(IndexOrArg::Named { name, value, span: self.span_from(&start) });
        }
        // `expr [':' expr ...]` or `expr '::' expr` slice forms
        let first = self.parse_expr()?;
        // `start::step` shorthand (lex: ColonColon)
        if matches!(self.peek(), TokenKind::ColonColon) {
            self.advance();
            let step = self.parse_expr()?;
            reject_range_bound(&start, [Some(&first), Some(&step)])?;
            return Ok(IndexOrArg::Idx(IndexElem::Slice {
                start: Some(Box::new(first)),
                end: None,
                step: Some(Box::new(step)),
                span: self.span_from(&start),
            }));
        }
        // canonical `start:end[:step]`
        if matches!(self.peek(), TokenKind::Colon) {
            self.advance();
            let end = if !matches!(self.peek(), TokenKind::Colon | TokenKind::Comma | TokenKind::RBracket) {
                Some(Box::new(self.parse_expr()?))
            } else { None };
            let step = if self.eat(&TokenKind::Colon) {
                if !matches!(self.peek(), TokenKind::Comma | TokenKind::RBracket) {
                    Some(Box::new(self.parse_expr()?))
                } else { None }
            } else { None };
            reject_range_bound(&start, [Some(&first), end.as_deref(), step.as_deref()])?;
            return Ok(IndexOrArg::Idx(IndexElem::Slice {
                start: Some(Box::new(first)),
                end, step,
                span: self.span_from(&start),
            }));
        }
        Ok(IndexOrArg::Idx(IndexElem::Expr(first)))
    }

    fn parse_primary(&mut self) -> ParseResult<Expr> {
        let start = self.peek_span();
        match self.peek().clone() {
            TokenKind::IntLit(n, ref suffix) => { let ty = parse_int_suffix(suffix); self.advance(); Ok(Expr::Literal(Literal::Int(n, ty), self.span_from(&start))) }
            TokenKind::FloatLit(f, suffix) => { self.advance(); let ty = parse_float_suffix(&suffix); Ok(Expr::Literal(Literal::Float(f, ty), self.span_from(&start))) }
            TokenKind::StrLit(s)   => { self.advance(); Ok(Expr::Literal(Literal::Str(s),   self.span_from(&start))) }
            TokenKind::CharLit(c)  => { self.advance(); Ok(Expr::Literal(Literal::Char(c),  self.span_from(&start))) }
            TokenKind::True        => { self.advance(); Ok(Expr::Literal(Literal::Bool(true),  self.span_from(&start))) }
            TokenKind::False       => { self.advance(); Ok(Expr::Literal(Literal::Bool(false), self.span_from(&start))) }
            TokenKind::Nil         => { self.advance(); Ok(Expr::Nil(self.span_from(&start))) }
            TokenKind::Ident(s) if s == "_" => { self.advance(); Ok(Expr::Underscore(self.span_from(&start))) }
            TokenKind::Ident(s) => { self.advance(); Ok(Expr::Ident(s, self.span_from(&start))) }
            // Scalar type names can appear as expressions in cast/type-arg contexts
            k if Self::is_scalar_type_kind(&k) => {
                self.advance();
                let name = format!("{:?}", k).to_lowercase()
                    .replace("kind::", "")
                    .replace("tokenkind::", "");
                Ok(Expr::Ident(name, self.span_from(&start)))
            }
            TokenKind::SelfKw => { self.advance(); Ok(Expr::Ident("self".to_string(), self.span_from(&start))) }
            TokenKind::LParen => {
                self.advance();
                self.skip_newlines();
                let mut elems = Vec::new();
                if !matches!(self.peek(), TokenKind::RParen) {
                    loop {
                        elems.push(self.parse_expr()?);
                        self.skip_newlines();
                        if !self.eat(&TokenKind::Comma) { break; }
                        self.skip_newlines();
                        if matches!(self.peek(), TokenKind::RParen) { break; }
                    }
                }
                self.expect(&TokenKind::RParen, "after parenthesized expr")?;
                Ok(Expr::Tuple(elems, self.span_from(&start)))
            }
            TokenKind::LBracket => {
                // tensor literal `[e1, e2, ...]`
                self.advance();
                self.skip_newlines();
                let mut elems = Vec::new();
                if !matches!(self.peek(), TokenKind::RBracket) {
                    loop {
                        elems.push(self.parse_expr()?);
                        self.skip_newlines();
                        if !self.eat(&TokenKind::Comma) { break; }
                        self.skip_newlines();
                        if matches!(self.peek(), TokenKind::RBracket) { break; }
                    }
                }
                self.expect(&TokenKind::RBracket, "after tensor literal")?;
                Ok(Expr::TensorLit(elems, self.span_from(&start)))
            }
            TokenKind::LBrace => Ok(Expr::Block(Box::new(self.parse_block()?))),
            TokenKind::If     => Ok(Expr::If(Box::new(self.parse_if_expr()?))),
            TokenKind::Match  => Ok(Expr::Match(Box::new(self.parse_match_expr()?))),
            TokenKind::Fn     => Ok(Expr::FnLit(Box::new(self.parse_fn_lit()?))),
            TokenKind::At     => {
                // `@cast(bf16) { ... }` as expression — directive(s) followed by block or match/if.
                // Spec §7.3: `@host match { .avx2 => 1, ... }` is also legal in expression position.
                let mut directives = Vec::new();
                while matches!(self.peek(), TokenKind::At) {
                    directives.push(self.parse_directive()?);
                }
                if matches!(self.peek(), TokenKind::LBrace) {
                    let body = self.parse_block()?;
                    Ok(Expr::DirectiveBlock { directives, body, span: self.span_from(&start) })
                } else if matches!(self.peek(), TokenKind::Match | TokenKind::If) {
                    // @directive match/if — parse the control-flow expression, wrap in a
                    // DirectiveBlock so callers can still see the directive annotation.
                    let inner_expr = if matches!(self.peek(), TokenKind::Match) {
                        Expr::Match(Box::new(self.parse_match_expr()?))
                    } else {
                        Expr::If(Box::new(self.parse_if_expr()?))
                    };
                    let span = self.span_from(&start);
                    Ok(Expr::DirectiveBlock {
                        directives,
                        body: Block {
                            stmts: Vec::new(),
                            tail_expr: Some(Box::new(inner_expr)),
                            span: span.clone(),
                        },
                        span,
                    })
                } else {
                    Err(self.err("expected `{`, `match`, or `if` after directive(s) in expression position"))
                }
            }
            TokenKind::Vault | TokenKind::Forge | TokenKind::Stream
                if matches!(self.peek_at(1), TokenKind::LBrace) =>
            {
                Ok(Expr::ArenaBlock(self.parse_arena_block()?))
            }
            // Arena keywords used as values in expressions (e.g. `vault.zeros[...]`)
            TokenKind::Vault  => {
                self.advance();
                Ok(Expr::Ident("vault".to_string(), self.span_from(&start)))
            }
            TokenKind::Forge  => {
                self.advance();
                Ok(Expr::Ident("forge".to_string(), self.span_from(&start)))
            }
            TokenKind::Stream => {
                self.advance();
                Ok(Expr::Ident("stream".to_string(), self.span_from(&start)))
            }
            // `~` in expression position is the streaming-axis marker (used
            // inside generic instantiations like `forge.kv[i32, [B, ~]]`).
            TokenKind::Tilde  => { self.advance(); Ok(Expr::Ident("~".to_string(), self.span_from(&start))) }
            other => Err(self.err(format!("expected expression, found {:?}", other))),
        }
    }

    fn parse_fn_lit(&mut self) -> ParseResult<FnLit> {
        let start = self.peek_span();
        self.expect(&TokenKind::Fn, "in fn literal")?;
        let shape_params = if matches!(self.peek(), TokenKind::LBracket) {
            self.parse_shape_params()?
        } else { Vec::new() };
        self.expect(&TokenKind::LParen, "in fn literal")?;
        let params = if matches!(self.peek(), TokenKind::RParen) {
            Vec::new()
        } else { self.parse_params()? };
        self.expect(&TokenKind::RParen, "in fn literal")?;
        // #446: same wrapped-signature allowance as `parse_fn_decl`.
        let ret_type = if self.eat_over_newlines(&TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else { None };
        self.expect_fn_body_brace("fn literal")?;
        let body = self.parse_block()?;
        Ok(FnLit { shape_params, params, ret_type, body, span: self.span_from(&start) })
    }

    fn peek_can_start_expr(&self) -> bool {
        match self.peek() {
            TokenKind::IntLit(..) | TokenKind::FloatLit(..) | TokenKind::StrLit(_) |
            TokenKind::CharLit(_) |
            TokenKind::True | TokenKind::False | TokenKind::Nil |
            TokenKind::Ident(_) | TokenKind::SelfKw |
            TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace |
            TokenKind::If | TokenKind::Match | TokenKind::Fn |
            TokenKind::At |
            TokenKind::Vault | TokenKind::Forge | TokenKind::Stream |
            TokenKind::Minus | TokenKind::Bang | TokenKind::Star |
            TokenKind::ReLU | TokenKind::GeLU | TokenKind::Tilde => true,
            k if Self::is_scalar_type_kind(k) => true,
            _ => false,
        }
    }

    fn expect_ident(&mut self, ctx: &str) -> ParseResult<String> {
        match self.peek().clone() {
            TokenKind::Ident(s) => { self.advance(); Ok(s) }
            other => Err(self.err(format!("expected identifier {}, found {:?}", ctx, other))),
        }
    }

    /// Field-name parsing: accept any ident OR a small set of keywords that
    /// are legitimately used as member/method names (`x.shape`, `x.dtype`, etc.).
    fn parse_field_name(&mut self) -> ParseResult<String> {
        let s = match self.peek().clone() {
            TokenKind::Ident(s) => s,
            TokenKind::Shape  => "shape".to_string(),
            TokenKind::Dtype  => "dtype".to_string(),
            TokenKind::View   => "view".to_string(),
            TokenKind::Stream => "stream".to_string(),
            TokenKind::Vault  => "vault".to_string(),
            TokenKind::Forge  => "forge".to_string(),
            TokenKind::Model  => "model".to_string(),
            TokenKind::Stage  => "stage".to_string(),
            TokenKind::Type   => "type".to_string(),
            TokenKind::SelfKw => "self".to_string(),
            TokenKind::As     => "as".to_string(),
            TokenKind::Trit   => "trit".to_string(),
            other => return Err(self.err(format!("expected field name, found {:?}", other))),
        };
        self.advance();
        Ok(s)
    }
}

/// The two slicing surfaces never mix (OPERATORS §9, SPEC §4.3): `..`/`..=`
/// is the unstepped range form, `:` is the stepped/full form. `a[0..100:2]`
/// otherwise parses as a *range expression sitting in the start bound* of a
/// `:` slice — type-clean at check time, and fatal only much later ("slice
/// start must be integer, got range" under `dmc run`, "slice start must be a
/// literal int in slice 4" under `dmc jit`). Reject it at the seam, where the
/// confusion is, and name the spelling the author meant.
fn reject_range_bound<const N: usize>(at: &Span, bounds: [Option<&Expr>; N]) -> ParseResult<()> {
    for b in bounds.into_iter().flatten() {
        if matches!(b, Expr::Range { .. }) {
            return Err(ParseError {
                msg: "`..` range used as a `:` slice bound; write the whole slice \
                      with colons (`a[0:100:2]`, not `a[0..100:2]`)".into(),
                line: at.line,
                col: at.col,
            });
        }
    }
    Ok(())
}

/// #445: map an integer literal's explicit type suffix. Mirrors
/// `parse_float_suffix`; the lexer only emits these eight strings.
fn parse_int_suffix(suffix: &Option<String>) -> Option<ScalarType> {
    suffix.as_ref().map(|s| match s.as_str() {
        "i8" => ScalarType::I8,
        "i16" => ScalarType::I16,
        "i32" => ScalarType::I32,
        "i64" => ScalarType::I64,
        "u8" => ScalarType::U8,
        "u16" => ScalarType::U16,
        "u32" => ScalarType::U32,
        "u64" => ScalarType::U64,
        _ => ScalarType::I64,
    })
}

fn parse_float_suffix(suffix: &Option<String>) -> Option<ScalarType> {
    suffix.as_ref().map(|s| match s.as_str() {
        "f16" => ScalarType::F16,
        "bf16" => ScalarType::Bf16,
        "tf32" => ScalarType::Tf32,
        "f32" => ScalarType::F32,
        "f64" => ScalarType::F64,
        "fp8_e4m3" => ScalarType::Fp8E4M3,
        "fp8_e5m2" => ScalarType::Fp8E5M2,
        _ => ScalarType::F32,
    })
}

// Helper enum: bracket contents are ambiguously index or arg list
enum IndexOrArg {
    Idx(IndexElem),
    Named { name: String, value: Expr, span: Span },
}

/// Extract the base identifier from an expression that may have been postfixed
/// with generic bracket args (e.g. `Foo[16, 32]` → `"Foo"`). Returns `None`
/// if the expression doesn't bottom out in a plain identifier.
fn extract_ctor_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident(name, _) => Some(name.clone()),
        Expr::Postfix { expr, op: PostfixOp::BracketArgs(_), .. }
        | Expr::Postfix { expr, op: PostfixOp::Index(_), .. } => extract_ctor_name(expr),
        Expr::Postfix { expr: inner, op: PostfixOp::Field(name), .. } => {
            if let Some(inner_name) = extract_ctor_name(inner) {
                Some(format!("{}.{}", inner_name, name))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract shape type args from `M[3]` or `M[N, K]` when used as a constructor prefix.
/// Returns the positional index expressions; named args and BracketArgs are ignored.
fn extract_ctor_type_args(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::Postfix { expr: inner, op: PostfixOp::Index(elems), .. } => {
            // First check if the inner already has type args (e.g., nested generics — rare).
            let inner_args = extract_ctor_type_args(inner);
            if !inner_args.is_empty() {
                return inner_args;
            }
            elems.iter().filter_map(|e| {
                if let IndexElem::Expr(expr) = e { Some(expr.clone()) } else { None }
            }).collect()
        }
        Expr::Postfix { expr: inner, op: PostfixOp::BracketArgs(_), .. } => {
            extract_ctor_type_args(inner)
        }
        _ => Vec::new(),
    }
}

impl IndexOrArg {
    fn into_index_elem(self) -> IndexElem {
        match self {
            IndexOrArg::Idx(e) => e,
            IndexOrArg::Named { name, value, span } => {
                // shouldn't happen after the any_named check; preserve as Expr fallback
                IndexElem::Expr(Expr::BinOp {
                    op: BinOp::Eq,
                    lhs: Box::new(Expr::Ident(name, span.clone())),
                    rhs: Box::new(value),
                    span,
                })
            }
        }
    }
    fn into_call_arg(self) -> CallArg {
        match self {
            IndexOrArg::Named { name, value, span } => CallArg::Named { name, value, span },
            IndexOrArg::Idx(IndexElem::Expr(e)) => CallArg::Positional(e),
            IndexOrArg::Idx(IndexElem::FullSlice(sp)) => {
                CallArg::Positional(Expr::Range { start: None, end: None, inclusive: false, span: sp })
            }
            IndexOrArg::Idx(IndexElem::Slice { start, end, step, span }) => {
                // collapse to Range expr; pre-alpha approximation
                let _ = (start.clone(), end.clone(), step.clone());
                CallArg::Positional(Expr::Range {
                    start, end, inclusive: false, span,
                })
            }
        }
    }
}
