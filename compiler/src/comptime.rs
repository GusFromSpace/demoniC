//! #505: `@comptime` folding — the reference interpreter at compile time.
//!
//! `SPEC.md §7.8` has specified this directive since the first draft;
//! `DIRECTIVES.md §3` names its rejection, `PORTS.md §5` forbids port calls
//! inside it, and `diag.rs` already lifts the tag into `--json`'s `code`.
//! None of it was enforced: `@comptime` parsed, warned, and did nothing — and
//! the two backends disagreed about what nothing meant. `dmc run` evaluated
//! the block; `dmc jit` refused it as an unsupported directive block. A
//! program that ran was a program that would not compile.
//!
//! This pass closes that. It runs after parse (and after the UFCS desugar
//! `parse_program` applies) and before `Checker::check_program`, over
//! `&mut Program`, so both backends lower the same folded tree. Its errors are
//! `check::TypeError`, seeded into the checker before it runs, so the human
//! renderer, the exit code and the `--json` stream are unchanged code paths.
//!
//! Two tiers (`COMPTIME_V1.md §4`):
//!
//! - **closed** — the body names no free identifier, so it has one value now.
//!   The whole `DirectiveBlock` node is replaced by an integer or boolean
//!   literal carrying the block's original span. No backend sees a directive.
//! - **residual** — the body reads a shape parameter of the enclosing `fn` or
//!   `model`. That is comptime-*known* but not comptime-*constant* until
//!   monomorphization (`SPEC.md §7.8`: "a compile-time constant in the
//!   surrounding monomorphization"), so this pass cannot pick a value. It
//!   gates the body and leaves the node for the backends, which after #505
//!   both lower it — the interpreter binds the shape parameter in scope, and
//!   inside the JIT a shape parameter *is* a constant.
//!
//! The v1 fold set is integers and booleans (`COMPTIME_V1.md §2`). The float
//! cut is the design, not an omission: it makes the #320-class question —
//! must a folded float equal a computed one? — unaskable rather than
//! answered. Widening it is a spec amendment, not a patch.
//!
//! The gate is **structural**. No call of any kind may appear in a `@comptime`
//! body, so no port, no `extern fn`, no `rng.*`, no file read and no
//! allocation can appear *even transitively* — the effect gate `PORTS.md §5`
//! asks for is total without any interprocedural analysis to be wrong about.

use std::collections::HashSet;

use crate::ast::*;
use crate::check::TypeError;
use crate::lexer::Span;
use crate::interp::{ComptimeStop, Interpreter, Value};

/// `COMPTIME_V1.md §6`: evaluator steps one `@comptime` block may take before
/// the compile fails as `comptime-budget`. A body may loop, so the compiler
/// must not hang on one. Per block, not per program — a program is not
/// penalized for folding many things. Fixed, not a dial: a fold that needs a
/// million steps of integer arithmetic is a design error, not a tuning
/// problem.
pub const STEP_BUDGET: u64 = 1_000_000;

/// Fold every `@comptime` block in `prog`, in place. Returns the diagnostics
/// the fold raised; an empty vec means every block either folded or was
/// accepted as residual.
pub fn fold_program(prog: &mut Program) -> Vec<TypeError> {
    let mut externs = HashSet::new();
    for item in &prog.items {
        collect_externs(item, &mut externs);
    }
    let mut cx = Folder { externs, errors: Vec::new(), shape_params: Vec::new() };
    for item in &mut prog.items {
        cx.item(item);
    }
    cx.errors
}

fn collect_externs(item: &Item, out: &mut HashSet<String>) {
    match item {
        Item::ExternFn(e) => { out.insert(e.name.clone()); }
        Item::Pub(inner) | Item::Directive { inner, .. } => collect_externs(inner, out),
        _ => {}
    }
}

struct Folder {
    /// Declared `extern fn` names — the one refusal that needs a fact from
    /// outside the block, so it can be reported as the foreign call it is
    /// (`SPEC.md §5`) rather than as an anonymous call.
    externs: HashSet<String>,
    errors: Vec<TypeError>,
    /// Shape parameters of the enclosing `fn` / `model`, innermost last. These
    /// are the only free identifiers a `@comptime` body may read.
    shape_params: Vec<HashSet<String>>,
}

/// What the static gate concluded about one `@comptime` body.
enum Gate {
    /// Every operand is a literal or block-bound — evaluate it now.
    Closed,
    /// Legal, but reads a shape parameter, so its value is fixed only per
    /// monomorphization. Leave it for the backends.
    Residual,
    /// Refused; diagnostics already recorded.
    Rejected,
}

impl Folder {
    fn error(&mut self, msg: String, span: &Span) {
        self.errors.push(TypeError { msg, span: span.clone(), hint: None, shapes: None });
    }

    fn error_hint(&mut self, msg: String, span: &Span, hint: &str) {
        self.errors.push(TypeError {
            msg, span: span.clone(), hint: Some(hint.to_string()), shapes: None,
        });
    }

    fn is_shape_param(&self, name: &str) -> bool {
        self.shape_params.iter().any(|s| s.contains(name))
    }

    // ── Walking ─────────────────────────────────────────────────────────────

    fn item(&mut self, item: &mut Item) {
        match item {
            Item::Fn(f) => self.fn_decl(f),
            Item::Model(m) => {
                let own: HashSet<String> =
                    m.shape_params.iter().map(|p| p.name.clone()).collect();
                self.shape_params.push(own);
                for mem in &mut m.members {
                    if let ModelMember::Method(f) = mem { self.fn_decl(f) }
                }
                self.shape_params.pop();
            }
            Item::Arena(a) => self.block(&mut a.body),
            Item::Let(l) => self.expr(&mut l.value),
            Item::Directive { directives, inner, span } => {
                // `DIRECTIVES.md §1` gives `@comptime`'s attachment as *block*.
                // An item is not one, and the directive would silently do
                // nothing there — the same defect, one level up. Refused the
                // way `@inplace`'s and `@fuse`'s attachment rules are.
                let _ = span;
                self.check_attachment(directives, "an item");
                self.item(inner);
            }
            Item::Pub(inner) => self.item(inner),
            Item::ExternFn(_) | Item::TypeAlias(_) | Item::Enum(_) | Item::Use(_) => {}
        }
    }

    fn fn_decl(&mut self, f: &mut FnDecl) {
        self.check_attachment(&f.directives, "a `fn`");
        let own: HashSet<String> = f.shape_params.iter().map(|p| p.name.clone()).collect();
        self.shape_params.push(own);
        self.block(&mut f.body);
        self.shape_params.pop();
    }

    /// `@comptime` on anything but a block. The catalog lists one attachment;
    /// honoring only that is what stops the directive from being a no-op in a
    /// second place.
    fn check_attachment(&mut self, directives: &[Directive], what: &str) {
        let spans: Vec<Span> = directives.iter()
            .filter(|d| d.name == "comptime").map(|d| d.span.clone()).collect();
        for sp in spans {
            self.error_hint(
                format!("`@comptime` on {} — the directive attaches to a block, and {} \
                         holds no expression to fold (DIRECTIVES.md §1)", what, what),
                &sp,
                "write `@comptime { … }` around the expression instead",
            );
        }
    }

    fn block(&mut self, b: &mut Block) {
        for s in &mut b.stmts { self.stmt(s) }
        if let Some(t) = &mut b.tail_expr { self.expr(t) }
    }

    fn stmt(&mut self, s: &mut Stmt) {
        match s {
            Stmt::Let(l) => self.expr(&mut l.value),
            Stmt::Expr { lhs, assign, .. } => {
                self.expr(lhs);
                if let Some((_, rhs)) = assign { self.expr(rhs) }
            }
            Stmt::If(i) => self.if_expr(i),
            Stmt::Match(m) => self.match_expr(m),
            Stmt::For { iter, body, .. } => { self.expr(iter); self.block(body) }
            Stmt::While { cond, body, .. } => { self.expr(cond); self.block(body) }
            Stmt::Loop { body, .. } => self.block(body),
            Stmt::Stage { body, .. } => self.expr(body),
            Stmt::Directive { directives, inner, span } => {
                let _ = span;
                self.check_attachment(directives, "a statement");
                self.stmt(inner);
            }
            Stmt::DirectiveBlock { directives, body, span } => {
                if has_comptime(directives) {
                    // A statement-position `@comptime` folds to a value nothing
                    // reads. Gate it (so its refusals still fire) and leave it;
                    // both backends lower a directive block as a statement.
                    if let Gate::Rejected = self.gate_block(body) {}
                    return;
                }
                let _ = span;
                self.block(body);
            }
            Stmt::Return { value, .. } => { if let Some(v) = value { self.expr(v) } }
            Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }

    fn if_expr(&mut self, i: &mut IfExpr) {
        self.expr(&mut i.cond);
        self.block(&mut i.then_branch);
        match &mut i.else_branch {
            Some(ElseBranch::Block(b)) => self.block(b),
            Some(ElseBranch::If(inner)) => self.if_expr(inner),
            None => {}
        }
    }

    fn match_expr(&mut self, m: &mut MatchExpr) {
        self.expr(&mut m.scrutinee);
        for arm in &mut m.arms {
            if let Some(g) = &mut arm.guard { self.expr(g) }
            self.expr(&mut arm.body);
        }
    }

    fn expr(&mut self, e: &mut Expr) {
        // The fold itself: a `@comptime` block is the one node this pass may
        // replace, so it is handled before the structural recursion.
        if let Expr::DirectiveBlock { directives, body, span } = e {
            if has_comptime(directives) {
                match self.gate_block(body) {
                    Gate::Closed => {
                        if let Some(folded) = self.evaluate(body, span) {
                            *e = folded;
                        }
                    }
                    Gate::Residual | Gate::Rejected => {}
                }
                return;
            }
        }
        match e {
            Expr::Tuple(xs, _) | Expr::TensorLit(xs, _) => {
                for x in xs { self.expr(x) }
            }
            Expr::Block(b) => self.block(b),
            Expr::If(i) => self.if_expr(i),
            Expr::Match(m) => self.match_expr(m),
            Expr::FnLit(f) => self.block(&mut f.body),
            Expr::ArenaBlock(a) => self.block(&mut a.body),
            Expr::DirectiveBlock { body, .. } => self.block(body),
            Expr::BinOp { lhs, rhs, .. } => { self.expr(lhs); self.expr(rhs) }
            Expr::UnOp { operand, .. } => self.expr(operand),
            Expr::Cast { expr, .. } => self.expr(expr),
            Expr::Postfix { expr, op, .. } => {
                self.expr(expr);
                match op {
                    PostfixOp::Call(args) | PostfixOp::BracketArgs(args) => {
                        for a in args {
                            match a {
                                CallArg::Positional(x) => self.expr(x),
                                CallArg::Named { value, .. } => self.expr(value),
                                CallArg::Spread(_) => {}
                            }
                        }
                    }
                    PostfixOp::Index(elems) => {
                        for el in elems { self.index_elem(el) }
                    }
                    PostfixOp::Constructor(fields) => {
                        for (_, v) in fields { self.expr(v) }
                    }
                    PostfixOp::Transpose | PostfixOp::Query | PostfixOp::Field(_) => {}
                }
            }
            Expr::StructLit { type_args, fields, .. } => {
                for t in type_args { self.expr(t) }
                for (_, v) in fields { self.expr(v) }
            }
            Expr::Range { start, end, .. } => {
                if let Some(s) = start { self.expr(s) }
                if let Some(x) = end { self.expr(x) }
            }
            Expr::Literal(..) | Expr::Ident(..) | Expr::Underscore(_)
            | Expr::Spread(_) | Expr::Nil(_) => {}
        }
    }

    fn index_elem(&mut self, el: &mut IndexElem) {
        match el {
            IndexElem::Expr(e) => self.expr(e),
            IndexElem::Slice { start, end, step, .. } => {
                if let Some(x) = start { self.expr(x) }
                if let Some(x) = end { self.expr(x) }
                if let Some(x) = step { self.expr(x) }
            }
            IndexElem::FullSlice(_) => {}
        }
    }

    // ── The static gate ─────────────────────────────────────────────────────

    /// Walk a `@comptime` body against the v1 grammar (`COMPTIME_V1.md §5`),
    /// recording a diagnostic for every construct outside it. Nothing is
    /// evaluated here: the gate is what guarantees a fold cannot reach an
    /// effect, so it must conclude before the interpreter is handed anything.
    fn gate_block(&mut self, body: &Block) -> Gate {
        let mut g = GateCx {
            bound: vec![HashSet::new()],
            reads_shape_param: false,
            rejected: false,
            errors: Vec::new(),
        };
        g.block(body, self);
        let rejected = g.rejected;
        let residual = g.reads_shape_param;
        self.errors.append(&mut g.errors);
        if rejected { Gate::Rejected } else if residual { Gate::Residual } else { Gate::Closed }
    }

    /// Run a closed body and turn its value into a literal. `None` if it did
    /// not produce one — the diagnostic is recorded here.
    fn evaluate(&mut self, body: &Block, span: &Span) -> Option<Expr> {
        match Interpreter::new().eval_comptime(body, STEP_BUDGET) {
            Ok(Value::Int(n, _)) => Some(Expr::Literal(Literal::Int(n, None), span.clone())),
            Ok(Value::Bool(b)) => Some(Expr::Literal(Literal::Bool(b), span.clone())),
            Ok(other) => {
                // The backstop of `COMPTIME_V1.md §5` R8. With the gate in
                // force this should be unreachable; it is what keeps the fold
                // set honest if the body grammar is ever widened without
                // revisiting §2.
                self.error(
                    format!("comptime-non-static: `@comptime` produced {}, which v1 does \
                             not fold — integers and booleans only (DIRECTIVES.md §3)",
                            value_kind(&other)),
                    span,
                );
                None
            }
            Err(ComptimeStop::Budget) => {
                self.error_hint(
                    format!("comptime-budget: `@comptime` evaluation exceeded {} steps — \
                             the block may not terminate (DIRECTIVES.md §3)", STEP_BUDGET),
                    span,
                    "compile-time evaluation is bounded; bound the loop or move the work \
                     to run time",
                );
                None
            }
            Err(ComptimeStop::Failed(e)) => {
                self.error(
                    format!("comptime-non-static: `@comptime` evaluation failed — {} \
                             (DIRECTIVES.md §3)", e.msg),
                    span,
                );
                None
            }
        }
    }
}

/// Scope state for one `@comptime` body's gate walk.
struct GateCx {
    /// Names the block itself introduced, innermost last. The only *other*
    /// legal free name is a shape parameter; everything else is a runtime
    /// binding and refuses the fold.
    bound: Vec<HashSet<String>>,
    reads_shape_param: bool,
    rejected: bool,
    errors: Vec<TypeError>,
}

impl GateCx {
    fn reject(&mut self, msg: String, span: &Span, hint: Option<&str>) {
        self.rejected = true;
        self.errors.push(TypeError {
            msg, span: span.clone(), hint: hint.map(str::to_string), shapes: None,
        });
    }

    /// The general refusal. Every construct outside the v1 grammar lands here,
    /// named, so the diagnostic says what was written rather than that
    /// something was.
    fn reject_construct(&mut self, what: &str, span: &Span) {
        self.reject(
            format!("comptime-non-static: {} is not comptime — v1 folds integer and \
                     boolean expressions only (DIRECTIVES.md §3)", what),
            span,
            Some("move it outside the `@comptime` block; v1 folds ints, bools and \
                  shape arithmetic"),
        );
    }

    fn bind(&mut self, p: &Pattern) {
        match p {
            Pattern::Ident(n, _) => { self.bound.last_mut().unwrap().insert(n.clone()); }
            Pattern::Tuple(ps, _) => for q in ps { self.bind(q) },
            Pattern::Bind(a, b, _) => { self.bind(a); self.bind(b) }
            _ => {}
        }
    }

    fn is_bound(&self, name: &str) -> bool {
        self.bound.iter().any(|s| s.contains(name))
    }

    fn block(&mut self, b: &Block, f: &Folder) {
        self.bound.push(HashSet::new());
        for s in &b.stmts { self.stmt(s, f) }
        if let Some(t) = &b.tail_expr { self.expr(t, f) }
        self.bound.pop();
    }

    fn stmt(&mut self, s: &Stmt, f: &Folder) {
        match s {
            Stmt::Let(l) => {
                self.expr(&l.value, f);
                self.bind(&l.pattern);
            }
            Stmt::Expr { lhs, assign, span } => {
                // An assignment is legal only onto a name the block itself
                // introduced — writing through to an outer binding would make
                // the fold an effect on the surrounding program.
                if let Some((_, rhs)) = assign {
                    match lhs {
                        Expr::Ident(n, sp) if self.is_bound(n) => { let _ = sp; }
                        Expr::Ident(n, sp) => self.reject(
                            format!("comptime-non-static: `@comptime` assigns to `{}`, which \
                                     it did not bind — a fold may not write to the program \
                                     around it (DIRECTIVES.md §3)", n),
                            sp, None),
                        _ => self.reject_construct("this assignment target", span),
                    }
                    self.expr(rhs, f);
                } else {
                    self.expr(lhs, f);
                }
            }
            Stmt::If(i) => self.if_expr(i, f),
            Stmt::While { cond, body, .. } => { self.expr(cond, f); self.block(body, f) }
            Stmt::Loop { body, .. } => self.block(body, f),
            Stmt::For { pattern, iter, body, .. } => {
                self.expr(iter, f);
                self.bound.push(HashSet::new());
                self.bind(pattern);
                self.block(body, f);
                self.bound.pop();
            }
            Stmt::Break(_) | Stmt::Continue(_) => {}
            Stmt::Return { span, .. } => self.reject_construct("`return`", span),
            Stmt::Match(m) => self.reject_construct("`match`", &m.span),
            Stmt::Stage { span, .. } => self.reject_construct("`stage`", span),
            Stmt::Directive { span, .. } | Stmt::DirectiveBlock { span, .. } =>
                self.reject_construct("a nested directive", span),
        }
    }

    fn if_expr(&mut self, i: &IfExpr, f: &Folder) {
        self.expr(&i.cond, f);
        self.block(&i.then_branch, f);
        match &i.else_branch {
            Some(ElseBranch::Block(b)) => self.block(b, f),
            Some(ElseBranch::If(inner)) => self.if_expr(inner, f),
            None => {}
        }
    }

    fn expr(&mut self, e: &Expr, f: &Folder) {
        match e {
            Expr::Literal(Literal::Int(..), _) | Expr::Literal(Literal::Bool(_), _) => {}
            // `{:?}` not `{}`: `2.0f64` renders as `2` under `Display`, and a
            // diagnostic that calls `2.0` a float while printing `2` reads as
            // a compiler bug rather than as the rule it is.
            Expr::Literal(Literal::Float(v, _), sp) => self.reject(
                format!("comptime-non-static: float `{:?}` is not comptime — v1 folds \
                         integers and booleans only (DIRECTIVES.md §3)", v),
                sp,
                Some("the float cut is deliberate: compile-time and run-time float \
                      evaluation are not the same context"),
            ),
            Expr::Literal(Literal::Str(_), sp) => self.reject_construct("a string literal", sp),
            Expr::Literal(Literal::Char(_), sp) => self.reject_construct("a char literal", sp),
            Expr::Literal(Literal::Nil, sp) | Expr::Nil(sp) =>
                self.reject_construct("`nil`", sp),

            Expr::Ident(name, sp) => {
                if self.is_bound(name) {
                    // introduced by this block — fine
                } else if f.is_shape_param(name) {
                    self.reads_shape_param = true;
                } else {
                    self.reject(
                        format!("comptime-non-static: `{}` is not comptime — it is a runtime \
                                 binding (DIRECTIVES.md §3)", name),
                        sp,
                        Some("only literals, names the block binds, and shape parameters \
                              of the enclosing `fn` or `model` are comptime"),
                    );
                }
            }

            Expr::BinOp { op, lhs, rhs, span } => {
                if !is_comptime_op(op) {
                    self.reject_construct(&format!("`{}`", op_name(op)), span);
                }
                self.expr(lhs, f);
                self.expr(rhs, f);
            }
            Expr::UnOp { op, operand, span } => {
                if !matches!(op, UnOp::Neg | UnOp::Not) {
                    self.reject_construct(&format!("`{}`", unop_name(op)), span);
                }
                self.expr(operand, f);
            }

            Expr::If(i) => self.if_expr(i, f),
            Expr::Block(b) => self.block(b, f),
            Expr::Tuple(xs, _) if xs.len() == 1 => self.expr(&xs[0], f),

            // Everything below is outside the v1 grammar. Each is named rather
            // than lumped, because "not comptime" without a noun is a
            // diagnostic the reader has to guess at.
            Expr::Tuple(_, sp) => self.reject_construct("a tuple", sp),
            Expr::TensorLit(_, sp) => self.reject_construct("a tensor literal", sp),
            Expr::Match(m) => self.reject_construct("`match`", &m.span),
            Expr::FnLit(l) => self.reject_construct("a lambda", &l.span),
            Expr::ArenaBlock(a) => self.reject_construct("an arena block", &a.span),
            Expr::DirectiveBlock { span, .. } =>
                self.reject_construct("a nested directive", span),
            Expr::Cast { span, .. } => self.reject_construct("`as`", span),
            Expr::StructLit { span, .. } => self.reject_construct("a model literal", span),
            Expr::Range { span, .. } => self.reject_construct("a range", span),
            Expr::Underscore(sp) => self.reject_construct("`_`", sp),
            Expr::Spread(sp) => self.reject_construct("`...`", sp),

            Expr::Postfix { expr, op, span } => self.postfix(expr, op, span, f),
        }
    }

    /// Calls are the whole effect gate. v1 admits none — that is what makes
    /// `PORTS.md §5`'s "no `@comptime` evaluation may call a port" total
    /// without an interprocedural scan, and it is why nothing in a legal body
    /// can reach the port registry, the RNG, the clock, an arena, or a file.
    /// The three named cases exist so the diagnostic says which rule the
    /// program hit, not merely that it hit one.
    fn postfix(&mut self, recv: &Expr, op: &PostfixOp, span: &Span, f: &Folder) {
        match op {
            PostfixOp::Call(_) | PostfixOp::BracketArgs(_) => {
                match recv {
                    Expr::Ident(n, sp)
                        if matches!(n.as_str(), "port_open" | "port_call" | "port_close") =>
                    {
                        self.reject(
                            format!("port-forbidden: `{}` is illegal inside a `@comptime` \
                                     block — a port call is an effect boundary compile-time \
                                     evaluation cannot cross (PORTS.md §5)", n),
                            sp, None,
                        );
                    }
                    Expr::Ident(n, sp) if f.externs.contains(n) => {
                        self.reject(
                            format!("comptime-non-static: `extern fn {}` is not comptime — a \
                                     foreign call cannot run at compile time (SPEC.md §5)", n),
                            sp, None,
                        );
                    }
                    Expr::Ident(n, sp) => self.reject(
                        format!("comptime-non-static: call to `{}` is not comptime — v1 folds \
                                 literal integer and boolean expressions, not calls \
                                 (DIRECTIVES.md §3)", n),
                        sp,
                        Some("hoist the call above the `@comptime` block"),
                    ),
                    _ => self.reject_construct("a call", span),
                }
            }
            PostfixOp::Index(_) => self.reject_construct("an index", span),
            PostfixOp::Field(name) =>
                self.reject_construct(&format!("field access `.{}`", name), span),
            PostfixOp::Constructor(_) => self.reject_construct("a constructor", span),
            PostfixOp::Transpose => self.reject_construct("`'`", span),
            PostfixOp::Query => self.reject_construct("`?`", span),
        }
    }
}

fn has_comptime(directives: &[Directive]) -> bool {
    directives.iter().any(|d| d.name == "comptime")
}

/// The v1 operator set: integer arithmetic, comparison, and boolean logic.
/// The `.`-prefixed elementwise family and `@` are tensor operators, so they
/// are outside the fold set by construction rather than by omission.
fn is_comptime_op(op: &BinOp) -> bool {
    matches!(op,
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod
        | BinOp::Pow | BinOp::StarStar
        | BinOp::And | BinOp::Or
        | BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq
        | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::BitShl | BinOp::BitShr)
}

fn op_name(op: &BinOp) -> &'static str {
    match op {
        BinOp::DotAdd => ".+", BinOp::DotSub => ".-", BinOp::DotMul => ".*",
        BinOp::DotDiv => "./", BinOp::DotPow => ".^",
        BinOp::DotGt => ".>", BinOp::DotLt => ".<",
        BinOp::DotGe => ".>=", BinOp::DotLe => ".<=",
        BinOp::Matmul => "@", BinOp::Pipe => "|>", BinOp::RShift => ">>",
        _ => "this operator",
    }
}

fn unop_name(op: &UnOp) -> &'static str {
    match op {
        UnOp::Deref => "*", UnOp::ReLU => "\\>", UnOp::GeLU => "\\~",
        UnOp::BitNot => "~", UnOp::Neg => "-", UnOp::Not => "!",
    }
}

/// Name a value's kind for the R8 backstop diagnostic.
fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::Float(..) => "a float",
        Value::Str(_) => "a string",
        Value::Tensor(_) => "a tensor",
        Value::Tuple(_) => "a tuple",
        Value::Struct(_) => "a model value",
        Value::Nil => "`nil`",
        _ => "a value outside the fold set",
    }
}

