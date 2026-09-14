// Module-wide dead-code allowance — deliberate, and scoped to this data model
// only. The AST is a *complete, consistent* representation: every node carries a
// `span` for diagnostics even though not every downstream pass reads it, and a
// few variants are reserved or superseded (e.g. `BinOp::RShift`, the pipe-era
// node for `>>`; #501 ruling S1a freed that spelling and #530 gave it to
// `BinOp::BitShr`, so `RShift` itself is unconstructible and ruling S16 keeps
// it reserved) yet kept wired into the backends' exhaustive matches. These are
// not stale helpers — deleting them would make the model inconsistent. Real
// dead code (unused helpers/methods) is NOT suppressed anywhere else in the
// crate.
#![allow(dead_code)]
/// demoniC AST — spec 0.0.5-draft
///
/// Companion to: GRAMMAR.ebnf, SPEC.md
///
/// One enum per grammar non-terminal where useful. Every node carries a
/// Span so diagnostics can point back at source. Pretty-printable via
/// `#[derive(Debug)]` — call sites use `{:#?}` for tree dumps.

use crate::lexer::Span;
use std::collections::HashSet;

// ─── Program / Items ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
    pub span: Span,
    /// #463: line count of the source this program was parsed from, measured
    /// at the EOF token. Feeds the safe-mode file-size lint in `check.rs`;
    /// carried on the AST so the checker needs no path or re-read of source.
    pub source_lines: usize,
}

#[derive(Debug, Clone)]
pub enum Item {
    Fn(FnDecl),
    ExternFn(ExternFnDecl),
    Model(ModelDecl),
    TypeAlias(TypeAlias),
    Enum(EnumDecl),
    Arena(ArenaBlock),
    Let(LetStmt),
    Use(UseStmt),
    Directive { directives: Vec<Directive>, inner: Box<Item>, span: Span },
    Pub(Box<Item>),
}

#[derive(Debug, Clone)]
pub struct ExternFnDecl {
    pub abi: Option<String>,
    pub name: String,
    pub shape_params: Vec<ShapeParam>,
    pub params: Vec<Param>,
    pub ret_type: Option<Type>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct UseStmt {
    pub path: String,
    pub alias: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub directives: Vec<Directive>,
    pub name: String,
    pub mutates_self: bool,     // trailing `!`
    pub shape_params: Vec<ShapeParam>,
    pub params: Vec<Param>,
    pub ret_type: Option<Type>,
    pub body: Block,
    pub span: Span,
    /// The file this fn was declared in, as the interpreter numbers files
    /// while loading them. A body resolves names against its own file
    /// (SPEC §6.7), so the callee must know where it was written. `0` in a
    /// freshly parsed AST; the loader stamps it.
    pub module: usize,
}

#[derive(Debug, Clone)]
pub struct ModelDecl {
    pub directives: Vec<Directive>,
    pub name: String,
    pub shape_params: Vec<ShapeParam>,
    pub members: Vec<ModelMember>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ModelMember {
    Field { mutating: bool, name: String, ty: Type, span: Span },
    Method(FnDecl),
}

#[derive(Debug, Clone)]
pub struct TypeAlias {
    pub name: String,
    pub shape_params: Vec<ShapeParam>,
    pub ty: Type,
    pub span: Span,
}

/// `enum Color { Red, Green, Blue }` — a C-like enum (#336). Variants are
/// ordered named constants; an enum value is its variant's i64 ordinal (index
/// in `variants`), so the interpreter and JIT reuse all integer machinery.
/// Payload-carrying variants (tagged unions) are a tracked follow-up.
#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: String,
    pub variants: Vec<EnumVariant>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: String,
    /// Positional payload types (#350 Part 2). Empty = a C-like tag-only
    /// variant (#336); non-empty = a tuple-style payload, e.g. `Circle(f32)`.
    pub fields: Vec<Type>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ArenaBlock {
    pub kind: ArenaKind,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArenaKind { Vault, Forge, Stream }

/// `forge.enter(b) { … }` (#674, the arena-budget grammar's `budget_block`
/// rule). Pushes `b` as the
/// entered budget for `body` and pops it on every exit edge (normal, `return`,
/// `break`, `continue`, `?`); nothing is reaped and `b` outlives the block.
/// Deliberately its own AST node rather than a fourth `ArenaKind`: `enter`
/// selects a *budget* (a host-lifetime handle, `TyType::Budget`), not one of
/// the three arenas `ArenaBlock`/`ArenaKind` name — conflating the two would
/// make the checker's `budget_stack` (a SEPARATE lexical stack from
/// `arena_stack`, `check.rs`) push an `ArenaKind` it must never push.
#[derive(Debug, Clone)]
pub struct BudgetBlock {
    /// The `Budget`-typed expression named in `forge.enter(…)`.
    pub operand: Box<Expr>,
    pub body: Block,
    pub span: Span,
}

/// #674 slice 3a: the arena-constructor spine `(forge|vault|stream).<alloc_
/// method>[…]` — the ONE shape both the `in b` budget-override postfix
/// (`Expr::AllocIn`, §2.1 GRAMMAR.ebnf's side condition) and the checker's
/// `budget-arena` (R3) diagnostic recognize. `parser.rs:parse_postfix`'s `in`
/// arm and `check.rs`'s R3 sites both call THIS function rather than
/// restating the pattern, per the design's own rule: "One definition of the
/// spine, exported, not restated."
///
/// `MODEL_ARRAY_ALLOCATOR_METHODS` (below) is NOT this set: it is the
/// narrower three-method set a *model-array* allocator uses
/// (`check.rs:ctor_tensor_ty`/`model_array_alloc_arena_of`, which also admit
/// only `forge`/`vault`, never `stream`). `identity` and `trit` are charged
/// constructors on the spine (`interp.rs:try_arena_constructor`,
/// `charge_ctor`) that `MODEL_ARRAY_ALLOCATOR_METHODS` has never covered.
pub const ARENA_CTOR_ALLOC_METHODS: [&str; 6] =
    ["zeros", "ones", "uninit", "identity", "kv", "trit"];

/// The `(arena, method)` a postfix receiver names, if it is the FIELD half
/// of an arena-constructor spine — `(forge|vault|stream).<alloc_method>`,
/// with no `Index` wrapper yet applied. Shared by `is_arena_ctor_spine`
/// (which adds the `Index` check) and by `check.rs`'s narrower model-array
/// recognizers, so every caller answers from the same one definition of
/// "which arena, which method" rather than a hand-rolled `Ident` match that
/// could drift from it.
pub fn arena_ctor_receiver(expr: &Expr) -> Option<(&str, &str)> {
    let Expr::Postfix { expr: inner, op: PostfixOp::Field(method), .. } = expr else { return None };
    if !ARENA_CTOR_ALLOC_METHODS.contains(&method.as_str()) { return None; }
    let Expr::Ident(arena, _) = inner.as_ref() else { return None };
    if !matches!(arena.as_str(), "forge" | "vault" | "stream") { return None; }
    Some((arena.as_str(), method.as_str()))
}

/// #674 (the arena-budget grammar's constructor-spine rule): is `expr` an arena
/// constructor spine — `(forge|vault|stream).<alloc_method>[…]`? The ONE
/// predicate the `in` postfix (grammar side condition) and the checker's R3
/// (`budget-arena`) both gate on, so a later generalization of `in` is a
/// single-site code change rather than a silent reinterpretation (§7 R3).
pub fn is_arena_ctor_spine(expr: &Expr) -> bool {
    matches!(expr, Expr::Postfix { expr: base, op: PostfixOp::Index(_), .. }
        if arena_ctor_receiver(base).is_some())
}

/// #728 rework (reviewer defect 3): the builtin allocator method names that
/// can build a *model* array (`arena.<method>[Model, [dims]]`) — the same
/// set `check.rs:ctor_tensor_ty` matches for a *scalar*-element allocator.
/// Moved here (#674 slice 3a) beside `is_arena_ctor_spine`, which it is
/// NOT the method set of (see that fn's doc comment) — kept beside it only
/// so the two related-but-different allocator-method sets stay visibly
/// paired rather than drifting apart in two separate files.
pub const MODEL_ARRAY_ALLOCATOR_METHODS: [&str; 3] = ["zeros", "ones", "uninit"];

#[derive(Debug, Clone)]
pub struct ShapeParam {
    pub name: String,
    pub default: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub mutating: bool,         // leading `!`
    pub is_self: bool,          // the `self` keyword
    pub name: String,
    pub ty: Option<Type>,
    pub span: Span,
}

// ─── Directives ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Directive {
    pub name: String,
    pub args: Vec<DArg>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum DArg {
    Positional(Expr),
    Named { name: String, value: Expr, span: Span },
}

// ─── Statements / Blocks ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail_expr: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(LetStmt),
    /// `expr` or `expr ASSIGN expr` or `expr <- expr`
    Expr {
        lhs: Expr,
        assign: Option<(AssignOp, Expr)>,
        span: Span,
    },
    If(IfExpr),
    Match(MatchExpr),
    For {
        pattern: Pattern,
        iter: Expr,
        body: Block,
        span: Span,
    },
    While { cond: Expr, body: Block, span: Span },
    Loop  { body: Block, span: Span },
    Stage { stage: i64, body: Expr, span: Span },     // inside @pp fn
    Directive { directives: Vec<Directive>, inner: Box<Stmt>, span: Span },
    DirectiveBlock { directives: Vec<Directive>, body: Block, span: Span },
    Break(Span),
    Continue(Span),
    Return { value: Option<Expr>, span: Span },
}

#[derive(Debug, Clone)]
pub struct LetStmt {
    pub mutating: bool,         // `!`
    pub is_mut: bool,           // `mut`
    pub pattern: Pattern,
    pub ty: Option<Type>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignOp { Eq, ColonEq, PlusEq, MinusEq, StarEq, SlashEq, StreamArrow, AmpEq, BarEq, CaretEq }

// ─── Patterns ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard(Span),
    Ident(String, Span),
    Literal(Literal, Span),
    Tuple(Vec<Pattern>, Span),
    Shape(Vec<ShapeElem>, Span),
    /// `pat @ pat` — bind a name to a sub-pattern
    Bind(Box<Pattern>, Box<Pattern>, Span),
    /// `..` — a rest pattern. Standalone it is a catch-all (matches anything,
    /// binds nothing, like `_`). Inside a tuple pattern it absorbs zero or more
    /// consecutive elements: `(a, ..)` matches any tuple of length ≥ 1, `(a, .., z)`
    /// any of length ≥ 2. At most one `..` per tuple (enforced at check time).
    Rest(Span),
    /// `Color.Red` — a qualified enum-variant pattern (#336). The bare tag-only
    /// form `Red` parses as `Ident` and is resolved to a variant by the checker
    /// when the scrutinee is an enum.
    ///
    /// `bindings` (#350 Part 2) holds the sub-patterns for a payload variant:
    /// `Circle(r)` / `Shape.Circle(r)` → `bindings: [Ident(r)]`. Empty = a
    /// tag-only pattern. `enum_name` is empty for the bare payload form
    /// (`Circle(r)`), which the checker resolves against the scrutinee's enum.
    EnumVariant { enum_name: String, variant: String, bindings: Vec<Pattern>, span: Span },
}

// ─── Expressions ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal, Span),
    Ident(String, Span),
    Underscore(Span),           // `_` in pipe stages
    Spread(Span),                // `...` in call arg lists
    Nil(Span),

    /// `(e)` or `(e1, e2, ...)` — parser builds Tuple of 1 for grouping;
    /// later passes can collapse.
    Tuple(Vec<Expr>, Span),
    TensorLit(Vec<Expr>, Span),
    Block(Box<Block>),

    If(Box<IfExpr>),
    Match(Box<MatchExpr>),
    FnLit(Box<FnLit>),
    ArenaBlock(ArenaBlock),
    /// `forge.enter(b) { … }` (#674 slice 3a). See `BudgetBlock`'s own doc.
    BudgetBlock(BudgetBlock),
    /// `<ctor_spine> in <budget>` (#674 slice 3a, §2.1 GRAMMAR.ebnf) — the
    /// per-constructor budget override. `expr` is always an arena-constructor
    /// spine (`is_arena_ctor_spine(&expr)` held at parse time); `budget` is a
    /// `postfix_expr`, not a general `expr` (`b`, `r.budget`, `bs[i]`,
    /// `forge.root`, `mk()` all parse; `b + 1` does not — there is no
    /// `Budget` arithmetic to want, §2.1).
    AllocIn { expr: Box<Expr>, budget: Box<Expr>, span: Span },
    /// `@cast(bf16) { ... }` and similar — directive applied to block,
    /// used as an expression.
    DirectiveBlock { directives: Vec<Directive>, body: Block, span: Span },

    BinOp { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr>, span: Span },
    UnOp  { op: UnOp,  operand: Box<Expr>, span: Span },

    Postfix { expr: Box<Expr>, op: PostfixOp, span: Span },

    /// `x as f32`
    Cast { expr: Box<Expr>, ty: Type, span: Span },

    /// `ModelName { field: value, ... }` — model constructor (Spec §6.4)
    /// `type_args` carries the `[N, ...]` shape params from `M[3] { ... }`.
    StructLit { name: String, type_args: Vec<Expr>, fields: Vec<(String, Expr)>, span: Span },

    /// `start..end` or `start..=end`
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub struct IfExpr {
    pub cond: Expr,
    pub then_branch: Block,
    pub else_branch: Option<ElseBranch>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ElseBranch {
    Block(Block),
    If(Box<IfExpr>),
}

#[derive(Debug, Clone)]
pub struct MatchExpr {
    pub scrutinee: Expr,
    pub arms: Vec<MatchArm>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FnLit {
    pub shape_params: Vec<ShapeParam>,
    pub params: Vec<Param>,
    pub ret_type: Option<Type>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum PostfixOp {
    Transpose,
    Query,
    Index(Vec<IndexElem>),
    Call(Vec<CallArg>),
    Field(String),
    /// `expr[args...]` — generic instantiation or shape literal in call position
    /// We disambiguate from indexing by tracking whether all elements are
    /// type-args or named (e.g. `Transformer[L=24, D=2048]`). Parser uses
    /// Index() for the literal form and a Type-arg path emerges in typechecker.
    /// Pre-alpha: same node, semantics decided later.
    BracketArgs(Vec<CallArg>),
    Constructor(Vec<(String, Expr)>),
}

#[derive(Debug, Clone)]
pub enum IndexElem {
    /// `..` — full axis
    FullSlice(Span),
    /// `start..end` style (also handles `start::step` via missing end)
    Slice {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        step: Option<Box<Expr>>,
        span: Span,
    },
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub enum CallArg {
    Positional(Expr),
    Named { name: String, value: Expr, span: Span },
    Spread(Span),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Pow, StarStar,
    DotAdd, DotSub, DotMul, DotDiv, DotPow,
    DotGt, DotLt, DotGe, DotLe,
    Matmul,
    And, Or,
    Eq, NotEq, Lt, Gt, LtEq, GtEq,
    Pipe, RShift,
    // Bitwise operators
    BitAnd, BitOr, BitXor, BitShl, BitShr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnOp { Neg, Not, Deref, ReLU, GeLU, BitNot }

// ─── Types ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Type {
    Scalar(ScalarType, Span),
    Tensor(Box<Type>, ShapeSpec, Span),
    View(Box<Type>, ShapeSpec, Span),
    KV(Box<Type>, ShapeSpec, Span),
    Mesh(Vec<MeshAxis>, Span),
    Fn(Vec<Type>, Box<Type>, Span),
    Tuple(Vec<Type>, Span),
    /// `[T; N]` — fixed-size array of T (Rust-style; pragmatic extension to grammar)
    Array(Box<Type>, Box<Expr>, Span),
    /// `*T` — raw pointer, extern fn boundary only (§3.12)
    RawPtr(Box<Type>, Span),
    /// `Ident` or `Ident[type_args]` — named type with optional generic args
    Named { name: String, args: Vec<TypeArg>, span: Span },
}

#[derive(Debug, Clone)]
pub enum TypeArg {
    Type(Type),
    Expr(Box<Expr>),
    Named { name: String, value: Box<Expr>, span: Span },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Int4, Int8,
    F16, Bf16, Tf32, F32, F64,
    Fp8E4M3, Fp8E5M2,
    Trit,
    Bool, Str, Nil,
}

impl ScalarType {
    /// The byte width this scalar occupies inside a `@cast(Model) { bytes }`
    /// overlay, or `None` when it has no byte form at all (#598).
    ///
    /// One table, consulted by both the checker (which refuses a model field
    /// it answers `None` for, naming the field) and the interpreter (which
    /// decodes with it), so the two can never disagree about which fields the
    /// overlay admits or how wide they are.
    ///
    /// Admitted: the fixed-width integers, `bool`, and the IEEE floats —
    /// every kind `docs/PORTS.md`'s element table gives a width for, plus the
    /// narrow integer kinds it does not carry on the wire but whose declared
    /// width is unambiguous all the same.
    ///
    /// Refused, and why each has no width to state:
    ///   - `str`  — dynamic, arena-backed text; nothing fixes its extent.
    ///   - `nil`  — the unit type occupies nothing.
    ///   - `trit` — packed ternary is a demoniC storage format with no byte
    ///     encoding (`PORTS.md`: "`trit` has no wire dtype").
    ///   - `int4` — a packed 4-bit type; how two of them share a byte is not
    ///     stated anywhere.
    ///   - `tf32`, `fp8_e4m3`, `fp8_e5m2` — f32-backed *nominal tags*
    ///     (`DIRECTIVES.md §4.1`, the #179 convention). They have no distinct
    ///     storage form to read bytes out of.
    pub fn overlay_byte_width(&self) -> Option<usize> {
        use ScalarType::*;
        Some(match self {
            I8 | U8 | Int8 | Bool => 1,
            I16 | U16 | F16 | Bf16 => 2,
            I32 | U32 | F32 => 4,
            I64 | U64 | F64 => 8,
            Str | Nil | Trit | Int4 | Tf32 | Fp8E4M3 | Fp8E5M2 => return None,
        })
    }

    /// Why the overlay has no layout for this scalar — the tail of the
    /// `--check` diagnostic and of the interpreter's backstop error. `None`
    /// when the kind IS admitted (`overlay_byte_width` answers `Some`).
    pub fn overlay_refusal_reason(&self) -> Option<&'static str> {
        use ScalarType::*;
        Some(match self {
            Str => "a `str` field's text is dynamic and arena-backed, so it has no fixed width",
            Nil => "`nil` occupies no bytes",
            Trit => "packed ternary has no byte encoding (PORTS.md: `trit` has no wire dtype)",
            Int4 => "`int4` is packed 4 bits wide, and how two share a byte is unstated",
            Tf32 | Fp8E4M3 | Fp8E5M2 =>
                "an f32-backed nominal tag (DIRECTIVES.md §4.1) has no distinct byte storage",
            _ => return None,
        })
    }

    /// Map a constructor's value-position element-type NAME (the `f32` in
    /// `forge.zeros[f32, […]]`) to its `ScalarType`. Returns `None` for
    /// anything that is not a known constructor scalar-type name.
    ///
    /// This is the ONE table both `check.rs`'s `scalar_type_from_ident`
    /// (which types the constructor at `--check`) and `interp.rs`'s
    /// `arena_ctor_dtype` (which tags the constructed tensor's `DType`, #834)
    /// resolve a bracket name through — so a name the checker admits here and
    /// a name the interpreter accepts a bracket argument for can never
    /// diverge into two different tables again (`arena_ctor_dtype` used to
    /// keep its own ad hoc lexical copy of this list, missing `fp8_e4m3`/
    /// `fp8_e5m2` and silently defaulting anything else to `DType::F32`).
    ///
    /// `str`/`nil` are deliberately absent: #817 is an open ruling on
    /// whether either has a real tensor-element representation at all, and
    /// this table does not decide that question either way — the checker
    /// still admits both names in this (constructor value) position, and
    /// the interpreter falls back to `DType::F32` for them as it did before
    /// #834. Annotation position
    /// (`Tensor[str, …]`) is a different table (`resolve_type`) and does
    /// admit them (SPEC.md §3.2).
    pub fn from_ctor_name(name: &str) -> Option<ScalarType> {
        use ScalarType::*;
        Some(match name {
            "i8" => I8, "i16" => I16, "i32" => I32, "i64" => I64,
            "u8" => U8, "u16" => U16, "u32" => U32, "u64" => U64,
            "int4" => Int4, "int8" => Int8,
            "f16" => F16, "bf16" => Bf16, "tf32" => Tf32, "f32" => F32, "f64" => F64,
            "fp8_e4m3" => Fp8E4M3, "fp8_e5m2" => Fp8E5M2, "trit" => Trit,
            "bool" => Bool,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ShapeSpec {
    pub elems: Vec<ShapeElem>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ShapeElem {
    Wildcard(Span),
    Spread(Span),            // `..`
    Streaming(Span),         // `~`
    /// Pragmatic extension: allow arithmetic (`B/dp`, `H*D`, `4*D/tp`) — full expressions.
    /// Spec EBNF restricts to ident|int_lit; examples clearly need more.
    /// Boxed because ShapeElem is reachable from Type, which is reachable
    /// from Expr, so we need a heap indirection here to break the cycle.
    Expr(Box<Expr>),
}

#[derive(Debug, Clone)]
pub struct MeshAxis {
    pub name: String,
    pub size: Box<Expr>,           // boxed; reachable from Type
    pub span: Span,
}

// ─── Literals ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// The optional ScalarType is an explicit type suffix (`42u32`) — the
    /// most explicit statement of intent a literal can carry (#445).
    Int(i64, Option<ScalarType>),
    Float(f64, Option<ScalarType>),
    Str(String),
    /// Char literal `c"x"` — Unicode scalar value, type `u32`.
    Char(char),
    Bool(bool),
    Nil,
}

// ─── Public Item Collection Helpers ──────────────────────────────────────────

/// Every name a top-level `let` binds in this file, `pub` or not. `pub let`
/// is itself a compile-time error (#616: `SPEC.md §6.7` exports only `pub
/// fn`/`pub model`/`pub type`), so in practice this is always the *private*
/// set — collected independent of visibility anyway so the invariant this
/// leans on stays true even if that rule ever narrows. `check.rs`'s import
/// machinery (#724) uses this to name a top-level `let` that leaked across a
/// module boundary with its own diagnostic rather than a bare `undefined
/// identifier`.
pub fn collect_top_level_let_names(program: &Program) -> HashSet<String> {
    let mut names = HashSet::new();
    for item in &program.items {
        collect_top_level_let_names_in_item(item, &mut names);
    }
    names
}

fn collect_top_level_let_names_in_item(item: &Item, names: &mut HashSet<String>) {
    match item {
        Item::Let(l) => collect_pattern_names(&l.pattern, names),
        Item::Pub(inner) | Item::Directive { inner, .. } => {
            collect_top_level_let_names_in_item(inner, names);
        }
        _ => {}
    }
}

pub fn collect_public_items(program: &Program) -> HashSet<String> {
    let mut public_names = HashSet::new();
    for item in &program.items {
        collect_public_items_in_item(item, &mut public_names, false);
    }
    public_names
}

fn collect_public_items_in_item(item: &Item, names: &mut HashSet<String>, is_pub: bool) {
    match item {
        Item::Pub(inner) => {
            collect_public_items_in_item(inner, names, true);
        }
        Item::Directive { inner, .. } => {
            collect_public_items_in_item(inner, names, is_pub);
        }
        other => {
            if is_pub {
                collect_names_in_item(other, names);
            }
        }
    }
}

fn collect_names_in_item(item: &Item, names: &mut HashSet<String>) {
    match item {
        Item::Fn(f) => {
            names.insert(f.name.clone());
        }
        Item::ExternFn(e) => {
            names.insert(e.name.clone());
        }
        Item::Model(m) => {
            names.insert(m.name.clone());
        }
        Item::TypeAlias(t) => {
            names.insert(t.name.clone());
        }
        Item::Enum(e) => {
            names.insert(e.name.clone());
        }
        Item::Let(l) => {
            collect_pattern_names(&l.pattern, names);
        }
        Item::Pub(inner) => {
            collect_names_in_item(inner, names);
        }
        Item::Directive { inner, .. } => {
            collect_names_in_item(inner, names);
        }
        Item::Arena(_) | Item::Use(_) => {}
    }
}

/// Split a tuple pattern's elements around a single `..` rest, if present.
/// Returns `(before, after, has_rest)`: with a rest the tuple matches any value
/// of length ≥ `before.len() + after.len()` (the rest absorbs the middle);
/// without, it is exact-arity. If more than one `..` appears, the first governs
/// (the checker rejects the multi-rest case, so valid code never relies on it).
pub fn tuple_rest_split(pats: &[Pattern]) -> (&[Pattern], &[Pattern], bool) {
    if let Some(i) = pats.iter().position(|p| matches!(p, Pattern::Rest(_))) {
        (&pats[..i], &pats[i + 1..], true)
    } else {
        (pats, &[], false)
    }
}

fn collect_pattern_names(pat: &Pattern, names: &mut HashSet<String>) {
    match pat {
        Pattern::Wildcard(_) => {}
        Pattern::Ident(name, _) => {
            names.insert(name.clone());
        }
        Pattern::Literal(_, _) => {}
        Pattern::Tuple(pats, _) => {
            for p in pats {
                collect_pattern_names(p, names);
            }
        }
        Pattern::Shape(_, _) => {}
        Pattern::Bind(p1, p2, _) => {
            collect_pattern_names(p1, names);
            collect_pattern_names(p2, names);
        }
        Pattern::EnumVariant { .. } => {}
        Pattern::Rest(_) => {}  // binds no name
    }
}

/// True if `e` syntactically uses a `_` pipe placeholder at this stage level.
///
/// Distinguishes the placeholder-fusion pipe form (`x |> _ .+ b`, where the RHS
/// is a stage expression) from the application form (`x |> f`, where the RHS is
/// a callable). Does not descend into closures (`FnLit`) — they carry their own
/// pipe scope. Block-like stages (block/if/match) are not treated as fusion
/// stages and report `false`.
///
/// The parser reuses it for a second question with the same shape: whether a
/// shape element in a *type* hides a `_`, which is not a dimension (#501, S3).
pub(crate) fn expr_contains_underscore(e: &Expr) -> bool {
    match e {
        Expr::Underscore(_) => true,
        Expr::Tuple(es, _) | Expr::TensorLit(es, _) => es.iter().any(expr_contains_underscore),
        Expr::BinOp { lhs, rhs, .. } => {
            expr_contains_underscore(lhs) || expr_contains_underscore(rhs)
        }
        Expr::UnOp { operand, .. } => expr_contains_underscore(operand),
        Expr::Cast { expr, .. } => expr_contains_underscore(expr),
        Expr::Postfix { expr, op, .. } => {
            expr_contains_underscore(expr) || postfix_contains_underscore(op)
        }
        Expr::Range { start, end, .. } => {
            start.as_deref().is_some_and(expr_contains_underscore)
                || end.as_deref().is_some_and(expr_contains_underscore)
        }
        _ => false,
    }
}

fn postfix_contains_underscore(op: &PostfixOp) -> bool {
    match op {
        PostfixOp::Index(elems) => elems.iter().any(|el| match el {
            IndexElem::Expr(e) => expr_contains_underscore(e),
            IndexElem::Slice { start, end, step, .. } => [start, end, step]
                .iter()
                .any(|o| o.as_deref().is_some_and(expr_contains_underscore)),
            IndexElem::FullSlice(_) => false,
        }),
        PostfixOp::Call(args) | PostfixOp::BracketArgs(args) => args.iter().any(|a| match a {
            CallArg::Positional(e) => expr_contains_underscore(e),
            CallArg::Named { value, .. } => expr_contains_underscore(value),
            CallArg::Spread(_) => false,
        }),
        PostfixOp::Constructor(fields) => {
            fields.iter().any(|(_, v)| expr_contains_underscore(v))
        }
        PostfixOp::Transpose | PostfixOp::Query | PostfixOp::Field(_) => false,
    }
}
