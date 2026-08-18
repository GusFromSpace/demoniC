//! #333: UFCS desugaring — rewrite `recv.f(args)` → `f(recv, args)`.
//!
//! demoniC has no method-call syntax, but anyone arriving from Rust or Python
//! reaches for `x.sort()`, `x.to_string()`, `x.floor()` constantly — the most
//! common way otherwise-correct code fails to compile. This pass — run once after
//! parse, so every backend (check/interp/jit) inherits it — turns a method call
//! on a non-`model` receiver into the free-function form the language already
//! supports, when the method name resolves to a real function.
//!
//! It fires only when the name is unambiguous: a known function (top-level fn or
//! builtin) that is NOT a model member (genuine method/field stays a method) and
//! NOT a supported string method (`s.split()` etc. are real and must survive).
//! The only residual imprecision is a function-valued model *field* whose name
//! collides with a global function — contrived, and excluded by the member set.

use std::collections::HashSet;
use crate::ast::*;

pub fn desugar_ufcs(prog: &mut Program) {
    let mut funcs = HashSet::new();
    let mut members = HashSet::new();
    for item in &prog.items {
        collect_item(item, &mut funcs, &mut members);
    }
    let ctx = Ctx { funcs, members };
    for item in &mut prog.items {
        ctx.item(item);
    }
}

/// Gather top-level function names and model member (method + field) names.
fn collect_item(item: &Item, funcs: &mut HashSet<String>, members: &mut HashSet<String>) {
    match item {
        Item::Fn(f) => { funcs.insert(f.name.clone()); }
        Item::ExternFn(f) => { funcs.insert(f.name.clone()); }
        Item::Model(m) => {
            for mem in &m.members {
                match mem {
                    ModelMember::Field { name, .. } => { members.insert(name.clone()); }
                    ModelMember::Method(f) => { members.insert(f.name.clone()); }
                }
            }
        }
        Item::Pub(inner) => collect_item(inner, funcs, members),
        Item::Directive { inner, .. } => collect_item(inner, funcs, members),
        Item::TypeAlias(_) | Item::Enum(_) | Item::Arena(_) | Item::Let(_) | Item::Use(_) => {}
    }
}

/// `allreduce`/`allgather`/… are distributed collectives, not UFCS receivers.
/// A method-form call on one (`allreduce.sum(x)`) must not be rewritten to
/// `sum(allreduce, x)` (which silently yields 0.0 — #396); it stays a method
/// call so the interpreter's collective guard reports a real error.
fn is_collective_ident(e: &Expr) -> bool {
    matches!(e, Expr::Ident(n, _)
        if matches!(n.as_str(), "allreduce" | "allgather" | "reducescatter" | "broadcast"))
}

struct Ctx {
    funcs: HashSet<String>,
    members: HashSet<String>,
}

impl Ctx {
    /// Should `recv.name(args)` desugar to `name(recv, args)`?
    fn should(&self, name: &str) -> bool {
        !self.members.contains(name)                       // genuine model method/field
            && !crate::check::is_supported_str_method(name) // real string method (`s.split()`)
            && (self.funcs.contains(name) || crate::interp::is_builtin(name))
    }

    fn item(&self, item: &mut Item) {
        match item {
            Item::Fn(f) => self.block(&mut f.body),
            Item::Model(m) => {
                for mem in &mut m.members {
                    if let ModelMember::Method(f) = mem {
                        self.block(&mut f.body);
                    }
                }
            }
            Item::Arena(a) => self.block(&mut a.body),
            Item::Let(l) => self.expr(&mut l.value),
            Item::Pub(inner) => self.item(inner),
            Item::Directive { inner, .. } => self.item(inner),
            Item::ExternFn(_) | Item::TypeAlias(_) | Item::Enum(_) | Item::Use(_) => {}
        }
    }

    fn block(&self, b: &mut Block) {
        for s in &mut b.stmts {
            self.stmt(s);
        }
        if let Some(e) = &mut b.tail_expr {
            self.expr(e);
        }
    }

    fn stmt(&self, s: &mut Stmt) {
        match s {
            Stmt::Let(l) => self.expr(&mut l.value),
            Stmt::Expr { lhs, assign, .. } => {
                self.expr(lhs);
                if let Some((_, rhs)) = assign {
                    self.expr(rhs);
                }
            }
            Stmt::If(i) => self.ifexpr(i),
            Stmt::Match(m) => self.matchexpr(m),
            Stmt::For { iter, body, .. } => { self.expr(iter); self.block(body); }
            Stmt::While { cond, body, .. } => { self.expr(cond); self.block(body); }
            Stmt::Loop { body, .. } => self.block(body),
            Stmt::Stage { body, .. } => self.expr(body),
            Stmt::Directive { inner, .. } => self.stmt(inner),
            Stmt::DirectiveBlock { body, .. } => self.block(body),
            Stmt::Return { value, .. } => { if let Some(e) = value { self.expr(e); } }
            Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }

    fn ifexpr(&self, i: &mut IfExpr) {
        self.expr(&mut i.cond);
        self.block(&mut i.then_branch);
        match &mut i.else_branch {
            Some(ElseBranch::Block(b)) => self.block(b),
            Some(ElseBranch::If(e)) => self.ifexpr(e),
            None => {}
        }
    }

    fn matchexpr(&self, m: &mut MatchExpr) {
        self.expr(&mut m.scrutinee);
        for arm in &mut m.arms {
            if let Some(g) = &mut arm.guard {
                self.expr(g);
            }
            self.expr(&mut arm.body);
        }
    }

    fn callargs(&self, args: &mut [CallArg]) {
        for a in args {
            match a {
                CallArg::Positional(e) => self.expr(e),
                CallArg::Named { value, .. } => self.expr(value),
                CallArg::Spread(_) => {}
            }
        }
    }

    fn expr(&self, e: &mut Expr) {
        // 1. Recurse into children first (so nested/receiver method calls desugar
        //    before this node restructures them).
        match e {
            Expr::Literal(..) | Expr::Ident(..) | Expr::Underscore(..)
            | Expr::Spread(..) | Expr::Nil(..) => {}
            Expr::Tuple(es, _) | Expr::TensorLit(es, _) => {
                for x in es { self.expr(x); }
            }
            Expr::Block(b) => self.block(b),
            Expr::If(i) => self.ifexpr(i),
            Expr::Match(m) => self.matchexpr(m),
            Expr::FnLit(f) => self.block(&mut f.body),
            Expr::ArenaBlock(a) => self.block(&mut a.body),
            Expr::DirectiveBlock { body, .. } => self.block(body),
            Expr::BinOp { lhs, rhs, .. } => { self.expr(lhs); self.expr(rhs); }
            Expr::UnOp { operand, .. } => self.expr(operand),
            Expr::Postfix { expr, op, .. } => {
                self.expr(expr);
                match op {
                    PostfixOp::Call(args) | PostfixOp::BracketArgs(args) => self.callargs(args),
                    _ => {}
                }
            }
            Expr::Cast { expr, .. } => self.expr(expr),
            Expr::StructLit { type_args, fields, .. } => {
                for t in type_args { self.expr(t); }
                for (_, v) in fields { self.expr(v); }
            }
            Expr::Range { start, end, .. } => {
                if let Some(s) = start { self.expr(s); }
                if let Some(en) = end { self.expr(en); }
            }
        }

        // 2. Rewrite `recv.f(args)` → `f(recv, args)` at this node, if eligible.
        let do_rewrite = if let Expr::Postfix { expr: callee, op: PostfixOp::Call(_), .. } = &*e {
            if let Expr::Postfix { expr: recv, op: PostfixOp::Field(n), .. } = callee.as_ref() {
                // #396: `allreduce.sum(x)` must NOT desugar to `sum(allreduce, x)`
                // — `sum` of a non-tensor receiver silently returns 0.0. Leave
                // distributed collectives in method form so the interpreter's
                // collective guard fires with a real error.
                self.should(n) && !is_collective_ident(recv)
            } else {
                false
            }
        } else {
            false
        };
        if !do_rewrite {
            return;
        }
        let outer_span = match &*e {
            Expr::Postfix { span, .. } => span.clone(),
            _ => unreachable!(),
        };
        if let Expr::Postfix { expr: callee, op: PostfixOp::Call(args), span } =
            std::mem::replace(e, Expr::Nil(outer_span))
        {
            if let Expr::Postfix { expr: recv, op: PostfixOp::Field(name), span: fspan } = *callee {
                let mut new_args = Vec::with_capacity(args.len() + 1);
                new_args.push(CallArg::Positional(*recv));
                new_args.extend(args);
                *e = Expr::Postfix {
                    expr: Box::new(Expr::Ident(name, fspan)),
                    op: PostfixOp::Call(new_args),
                    span,
                };
            }
        }
    }
}
