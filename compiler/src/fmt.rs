/// demoniC AST pretty-printer — `dmc fmt`
///
/// Produces canonical source text from a parsed AST.  The output is
/// designed to be stable under round-trips: `parse → fmt → parse` should
/// always succeed and produce an identical formatted string.

use crate::ast::*;

// ─── Public API ──────────────────────────────────────────────────────────────

pub fn pretty_print_program(program: &Program) -> String {
    let mut p = Printer::new();
    p.print_program(program);
    p.finish()
}

// ─── Printer ─────────────────────────────────────────────────────────────────

struct Printer {
    buf: String,
    indent: usize,
}

impl Printer {
    fn new() -> Self {
        Self { buf: String::new(), indent: 0 }
    }

    fn finish(self) -> String {
        self.buf
    }

    // ── Indentation helpers ───────────────────────────────────────────────

    fn indent_str(&self) -> String {
        "    ".repeat(self.indent)
    }

    fn push_indent(&mut self) { self.indent += 1; }
    fn pop_indent(&mut self)  { if self.indent > 0 { self.indent -= 1; } }

    fn write(&mut self, s: &str) {
        self.buf.push_str(s);
    }

    fn writeln_indented(&mut self, s: &str) {
        let ind = self.indent_str();
        self.buf.push_str(&ind);
        self.buf.push_str(s);
        self.buf.push('\n');
    }

    // ── Program ───────────────────────────────────────────────────────────

    fn print_program(&mut self, prog: &Program) {
        for (i, item) in prog.items.iter().enumerate() {
            if i > 0 { self.write("\n"); }
            self.print_item(item);
        }
    }

    // ── Items ─────────────────────────────────────────────────────────────

    fn print_item(&mut self, item: &Item) {
        self.print_item_with_vis(item, false);
    }

    fn print_item_with_vis(&mut self, item: &Item, is_public: bool) {
        match item {
            Item::Fn(fd)       => self.print_fn_decl(fd, false, is_public),
            Item::ExternFn(e)  => self.print_extern_fn_decl(e, is_public),
            Item::Model(md)    => self.print_model_decl(md, is_public),
            Item::TypeAlias(ta) => self.print_type_alias(ta, is_public),
            Item::Enum(ed) => self.print_enum_decl(ed, is_public),
            Item::Arena(ab) => self.print_arena_block(ab),
            Item::Let(ls) => self.print_let_stmt(ls, is_public),
            Item::Use(us) => {
                let ind = self.indent_str();
                self.buf.push_str(&ind);
                self.buf.push_str("use ");
                self.buf.push_str(&format!("\"{}\"", us.path));
                if let Some(alias) = &us.alias {
                    self.buf.push_str(" as ");
                    self.buf.push_str(alias);
                }
                self.buf.push('\n');
            }
            Item::Directive { directives, inner, .. } => {
                for d in directives { self.print_directive_line(d); }
                self.print_item_with_vis(inner, is_public);
            }
            Item::Pub(inner) => {
                self.print_item_with_vis(inner, true);
            }
        }
    }

    fn print_fn_decl(&mut self, fd: &FnDecl, is_method: bool, is_public: bool) {
        let ind = self.indent_str();
        // directives on their own lines
        for d in &fd.directives {
            self.buf.push_str(&ind);
            self.print_directive_inline(d);
            self.buf.push('\n');
        }

        self.buf.push_str(&ind);
        if is_public { self.buf.push_str("pub "); }
        self.buf.push_str("fn ");
        self.buf.push_str(&fd.name);
        if fd.mutates_self { self.buf.push('!'); }

        // shape params
        if !fd.shape_params.is_empty() {
            self.buf.push('[');
            for (i, sp) in fd.shape_params.iter().enumerate() {
                if i > 0 { self.buf.push_str(", "); }
                self.buf.push_str(&sp.name);
                if let Some(def) = &sp.default {
                    self.buf.push_str(" = ");
                    let s = self.expr_to_string(def);
                    self.buf.push_str(&s);
                }
            }
            self.buf.push(']');
        }

        // params
        self.buf.push('(');
        for (i, p) in fd.params.iter().enumerate() {
            if i > 0 { self.buf.push_str(", "); }
            let s = self.param_to_string(p, is_method);
            self.buf.push_str(&s);
        }
        self.buf.push(')');

        // return type
        if let Some(ret) = &fd.ret_type {
            self.buf.push_str(" -> ");
            let s = self.type_to_string(ret);
            self.buf.push_str(&s);
        }

        self.buf.push(' ');
        self.print_block(&fd.body);
        self.buf.push('\n');
    }

    fn print_extern_fn_decl(&mut self, e: &ExternFnDecl, is_public: bool) {
        let ind = self.indent_str();
        self.buf.push_str(&ind);
        if is_public { self.buf.push_str("pub "); }
        self.buf.push_str("extern ");
        if let Some(abi) = &e.abi {
            self.buf.push('"');
            self.buf.push_str(abi);
            self.buf.push_str("\" ");
        }
        self.buf.push_str("fn ");
        self.buf.push_str(&e.name);
        if !e.shape_params.is_empty() {
            self.buf.push('[');
            for (i, sp) in e.shape_params.iter().enumerate() {
                if i > 0 { self.buf.push_str(", "); }
                self.buf.push_str(&sp.name);
            }
            self.buf.push(']');
        }
        self.buf.push('(');
        for (i, p) in e.params.iter().enumerate() {
            if i > 0 { self.buf.push_str(", "); }
            let s = self.param_to_string(p, false);
            self.buf.push_str(&s);
        }
        self.buf.push(')');
        if let Some(ret) = &e.ret_type {
            self.buf.push_str(" -> ");
            let s = self.type_to_string(ret);
            self.buf.push_str(&s);
        }
        self.buf.push('\n');
    }

    fn print_model_decl(&mut self, md: &ModelDecl, is_public: bool) {
        let ind = self.indent_str();
        for d in &md.directives {
            self.buf.push_str(&ind);
            self.print_directive_inline(d);
            self.buf.push('\n');
        }

        self.buf.push_str(&ind);
        if is_public { self.buf.push_str("pub "); }
        self.buf.push_str("model ");
        self.buf.push_str(&md.name);

        if !md.shape_params.is_empty() {
            self.buf.push('[');
            for (i, sp) in md.shape_params.iter().enumerate() {
                if i > 0 { self.buf.push_str(", "); }
                self.buf.push_str(&sp.name);
                if let Some(def) = &sp.default {
                    self.buf.push_str(" = ");
                    let s = self.expr_to_string(def);
                    self.buf.push_str(&s);
                }
            }
            self.buf.push(']');
        }

        self.buf.push_str(" {\n");
        self.push_indent();
        for member in &md.members {
            match member {
                ModelMember::Field { mutating, name, ty, .. } => {
                    let ind2 = self.indent_str();
                    self.buf.push_str(&ind2);
                    if *mutating { self.buf.push('!'); }
                    self.buf.push_str(name);
                    self.buf.push_str(": ");
                    let s = self.type_to_string(ty);
                    self.buf.push_str(&s);
                    self.buf.push('\n');
                }
                ModelMember::Method(fd) => {
                    self.print_fn_decl(fd, true, false);
                }
            }
        }
        self.pop_indent();
        let ind = self.indent_str();
        self.buf.push_str(&ind);
        self.buf.push_str("}\n");
    }

    fn print_enum_decl(&mut self, ed: &EnumDecl, is_public: bool) {
        let ind = self.indent_str();
        self.buf.push_str(&ind);
        if is_public { self.buf.push_str("pub "); }
        self.buf.push_str("enum ");
        self.buf.push_str(&ed.name);
        self.buf.push_str(" {\n");
        for v in &ed.variants {
            self.buf.push_str(&ind);
            self.buf.push_str("    ");
            self.buf.push_str(&v.name);
            // #350 Part 2: positional payload types `Circle(f32)` / `Rect(f32, f32)`.
            if !v.fields.is_empty() {
                let tys: Vec<String> = v.fields.iter().map(|t| self.type_to_string(t)).collect();
                self.buf.push('(');
                self.buf.push_str(&tys.join(", "));
                self.buf.push(')');
            }
            self.buf.push_str(",\n");
        }
        self.buf.push_str(&ind);
        self.buf.push_str("}\n");
    }

    fn print_type_alias(&mut self, ta: &TypeAlias, is_public: bool) {
        let ind = self.indent_str();
        self.buf.push_str(&ind);
        if is_public { self.buf.push_str("pub "); }
        self.buf.push_str("type ");
        self.buf.push_str(&ta.name);

        if !ta.shape_params.is_empty() {
            self.buf.push('[');
            for (i, sp) in ta.shape_params.iter().enumerate() {
                if i > 0 { self.buf.push_str(", "); }
                self.buf.push_str(&sp.name);
            }
            self.buf.push(']');
        }

        self.buf.push_str(" = ");
        let s = self.type_to_string(&ta.ty);
        self.buf.push_str(&s);
        self.buf.push('\n');
    }

    fn print_arena_block(&mut self, ab: &ArenaBlock) {
        let ind = self.indent_str();
        self.buf.push_str(&ind);
        let kw = match ab.kind {
            ArenaKind::Vault  => "vault",
            ArenaKind::Forge  => "forge",
            ArenaKind::Stream => "stream",
        };
        self.buf.push_str(kw);
        self.buf.push(' ');
        self.print_block(&ab.body);
        self.buf.push('\n');
    }

    fn print_let_stmt(&mut self, ls: &LetStmt, is_public: bool) {
        let ind = self.indent_str();
        self.buf.push_str(&ind);
        if is_public { self.buf.push_str("pub "); }
        self.buf.push_str("let ");
        if ls.mutating { self.buf.push('!'); }
        if ls.is_mut { self.buf.push_str("mut "); }
        let pat = self.pattern_to_string(&ls.pattern);
        self.buf.push_str(&pat);
        if let Some(ty) = &ls.ty {
            self.buf.push_str(": ");
            let s = self.type_to_string(ty);
            self.buf.push_str(&s);
        }
        self.buf.push_str(" = ");
        let val = self.expr_to_string(&ls.value);
        self.buf.push_str(&val);
        self.buf.push('\n');
    }

    fn print_directive_line(&mut self, d: &Directive) {
        let ind = self.indent_str();
        self.buf.push_str(&ind);
        self.print_directive_inline(d);
        self.buf.push('\n');
    }

    fn print_directive_inline(&mut self, d: &Directive) {
        self.buf.push('@');
        self.buf.push_str(&d.name);
        if !d.args.is_empty() {
            self.buf.push('(');
            for (i, arg) in d.args.iter().enumerate() {
                if i > 0 { self.buf.push_str(", "); }
                match arg {
                    DArg::Positional(e) => {
                        let s = self.expr_to_string(e);
                        self.buf.push_str(&s);
                    }
                    DArg::Named { name, value, .. } => {
                        self.buf.push_str(name);
                        self.buf.push('=');
                        let s = self.expr_to_string(value);
                        self.buf.push_str(&s);
                    }
                }
            }
            self.buf.push(')');
        }
    }

    // ── Block ─────────────────────────────────────────────────────────────

    fn print_block(&mut self, block: &Block) {
        self.buf.push_str("{\n");
        self.push_indent();
        for stmt in &block.stmts {
            self.print_stmt(stmt);
        }
        if let Some(tail) = &block.tail_expr {
            let ind = self.indent_str();
            self.buf.push_str(&ind);
            let s = self.expr_to_string(tail);
            self.buf.push_str(&s);
            self.buf.push('\n');
        }
        self.pop_indent();
        let ind = self.indent_str();
        self.buf.push_str(&ind);
        self.buf.push('}');
    }

    // ── Statements ────────────────────────────────────────────────────────

    fn print_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(ls) => self.print_let_stmt(ls, false),

            Stmt::Expr { lhs, assign, .. } => {
                let ind = self.indent_str();
                self.buf.push_str(&ind);
                let s = self.expr_to_string(lhs);
                self.buf.push_str(&s);
                if let Some((op, rhs)) = assign {
                    self.buf.push(' ');
                    self.buf.push_str(assignop_str(op));
                    self.buf.push(' ');
                    let r = self.expr_to_string(rhs);
                    self.buf.push_str(&r);
                }
                self.buf.push('\n');
            }

            Stmt::If(ie) => {
                let ind = self.indent_str();
                self.buf.push_str(&ind);
                let s = self.if_expr_to_string(ie);
                self.buf.push_str(&s);
                self.buf.push('\n');
            }

            Stmt::Match(me) => {
                let ind = self.indent_str();
                self.buf.push_str(&ind);
                let s = self.match_expr_to_string(me);
                self.buf.push_str(&s);
                self.buf.push('\n');
            }

            Stmt::For { pattern, iter, body, .. } => {
                let ind = self.indent_str();
                self.buf.push_str(&ind);
                self.buf.push_str("for ");
                let pat = self.pattern_to_string(pattern);
                self.buf.push_str(&pat);
                self.buf.push_str(" in ");
                let it = self.expr_to_string(iter);
                self.buf.push_str(&it);
                self.buf.push(' ');
                self.print_block(body);
                self.buf.push('\n');
            }

            Stmt::While { cond, body, .. } => {
                let ind = self.indent_str();
                self.buf.push_str(&ind);
                self.buf.push_str("while ");
                let c = self.expr_to_string(cond);
                self.buf.push_str(&c);
                self.buf.push(' ');
                self.print_block(body);
                self.buf.push('\n');
            }

            Stmt::Loop { body, .. } => {
                let ind = self.indent_str();
                self.buf.push_str(&ind);
                self.buf.push_str("loop ");
                self.print_block(body);
                self.buf.push('\n');
            }

            Stmt::Break(_) => {
                self.writeln_indented("break");
            }

            Stmt::Continue(_) => {
                self.writeln_indented("continue");
            }

            Stmt::Return { value, .. } => {
                let ind = self.indent_str();
                self.buf.push_str(&ind);
                self.buf.push_str("return");
                if let Some(v) = value {
                    self.buf.push(' ');
                    let s = self.expr_to_string(v);
                    self.buf.push_str(&s);
                }
                self.buf.push('\n');
            }

            Stmt::Stage { stage, body, .. } => {
                let ind = self.indent_str();
                self.buf.push_str(&ind);
                self.buf.push_str(&format!("@stage({}) ", stage));
                let s = self.expr_to_string(body);
                self.buf.push_str(&s);
                self.buf.push('\n');
            }

            Stmt::Directive { directives, inner, .. } => {
                for d in directives { self.print_directive_line(d); }
                self.print_stmt(inner);
            }

            Stmt::DirectiveBlock { directives, body, .. } => {
                let ind = self.indent_str();
                self.buf.push_str(&ind);
                for (i, d) in directives.iter().enumerate() {
                    if i > 0 { self.buf.push(' '); }
                    self.print_directive_inline(d);
                }
                self.buf.push(' ');
                self.print_block(body);
                self.buf.push('\n');
            }
        }
    }

    // ── Expressions (return String) ───────────────────────────────────────

    fn expr_to_string(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Literal(lit, _) => lit_to_string(lit),

            Expr::Ident(name, _) => name.clone(),

            Expr::Underscore(_) => "_".to_string(),

            Expr::Spread(_) => "...".to_string(),

            Expr::Nil(_) => "nil".to_string(),

            Expr::Tuple(elems, _) => {
                if elems.len() == 1 {
                    // Parenthesised expression, not a tuple
                    format!("({})", self.expr_to_string(&elems[0]))
                } else {
                    let parts: Vec<String> = elems.iter()
                        .map(|e| self.expr_to_string(e))
                        .collect();
                    format!("({})", parts.join(", "))
                }
            }

            Expr::TensorLit(elems, _) => {
                let parts: Vec<String> = elems.iter()
                    .map(|e| self.expr_to_string(e))
                    .collect();
                format!("[{}]", parts.join(", "))
            }

            Expr::Block(block) => {
                // Inline block — we need to print the block inline but we
                // can't because print_block uses push/pop indent.  Emit a
                // block on the current line (common in expr position).
                let saved_indent = self.indent;
                let saved_buf = std::mem::take(&mut self.buf);
                self.print_block(block);
                let block_str = std::mem::replace(&mut self.buf, saved_buf);
                self.indent = saved_indent;
                block_str
            }

            Expr::If(ie) => self.if_expr_to_string(ie),

            Expr::Match(me) => self.match_expr_to_string(me),

            Expr::FnLit(fl) => {
                let mut s = String::from("fn");
                if !fl.shape_params.is_empty() {
                    s.push('[');
                    for (i, sp) in fl.shape_params.iter().enumerate() {
                        if i > 0 { s.push_str(", "); }
                        s.push_str(&sp.name);
                    }
                    s.push(']');
                }
                s.push('(');
                for (i, p) in fl.params.iter().enumerate() {
                    if i > 0 { s.push_str(", "); }
                    s.push_str(&self.param_to_string(p, false));
                }
                s.push(')');
                if let Some(ret) = &fl.ret_type {
                    s.push_str(" -> ");
                    s.push_str(&self.type_to_string(ret));
                }
                s.push(' ');
                let saved_indent = self.indent;
                let saved_buf = std::mem::take(&mut self.buf);
                self.print_block(&fl.body);
                let block_str = std::mem::replace(&mut self.buf, saved_buf);
                self.indent = saved_indent;
                s.push_str(&block_str);
                s
            }

            Expr::ArenaBlock(ab) => {
                let kind = match ab.kind {
                    ArenaKind::Vault  => "vault",
                    ArenaKind::Forge  => "forge",
                    ArenaKind::Stream => "stream",
                };
                let saved_indent = self.indent;
                let saved_buf = std::mem::take(&mut self.buf);
                self.print_block(&ab.body);
                let block_str = std::mem::replace(&mut self.buf, saved_buf);
                self.indent = saved_indent;
                format!("{} {}", kind, block_str)
            }

            Expr::DirectiveBlock { directives, body, .. } => {
                let mut s = String::new();
                for (i, d) in directives.iter().enumerate() {
                    if i > 0 { s.push(' '); }
                    let saved_buf = std::mem::take(&mut self.buf);
                    self.print_directive_inline(d);
                    let ds = std::mem::replace(&mut self.buf, saved_buf);
                    s.push_str(&ds);
                }
                s.push(' ');
                let saved_indent = self.indent;
                let saved_buf = std::mem::take(&mut self.buf);
                self.print_block(body);
                let block_str = std::mem::replace(&mut self.buf, saved_buf);
                self.indent = saved_indent;
                s.push_str(&block_str);
                s
            }

            Expr::BinOp { op, lhs, rhs, .. } => {
                let l = self.expr_to_string_parens(lhs, op);
                let r = self.expr_to_string_parens_right(rhs, op);
                format!("{} {} {}", l, binop_str(op), r)
            }

            Expr::UnOp { op, operand, .. } => {
                let inner = self.expr_to_string(operand);
                match op {
                    UnOp::Neg => format!("-{}", inner),
                    UnOp::Not => format!("!{}", inner),
                    UnOp::Deref => format!("*{}", inner),
                    UnOp::ReLU => format!("relu({})", inner),
                    UnOp::GeLU => format!("gelu({})", inner),
                    UnOp::BitNot => format!("~{}", inner),
                }
            }

            Expr::Postfix { expr, op, .. } => {
                let base = self.expr_to_string(expr);
                self.postfix_to_string(&base, op)
            }

            Expr::Cast { expr, ty, .. } => {
                let e = self.expr_to_string(expr);
                let t = self.type_to_string(ty);
                format!("{} as {}", e, t)
            }

            Expr::StructLit { name, type_args, fields, .. } => {
                let mut s = name.clone();
                if !type_args.is_empty() {
                    s.push('[');
                    for (i, a) in type_args.iter().enumerate() {
                        if i > 0 { s.push_str(", "); }
                        s.push_str(&self.expr_to_string(a));
                    }
                    s.push(']');
                }
                s.push_str(" {");
                for (i, (fname, fval)) in fields.iter().enumerate() {
                    if i > 0 { s.push_str(", "); }
                    s.push(' ');
                    s.push_str(fname);
                    s.push_str(": ");
                    s.push_str(&self.expr_to_string(fval));
                }
                if !fields.is_empty() { s.push(' '); }
                s.push('}');
                s
            }

            Expr::Range { start, end, inclusive, .. } => {
                let mut s = String::new();
                if let Some(st) = start {
                    s.push_str(&self.expr_to_string(st));
                }
                if *inclusive {
                    s.push_str("..=");
                } else {
                    s.push_str("..");
                }
                if let Some(en) = end {
                    s.push_str(&self.expr_to_string(en));
                }
                s
            }
        }
    }

    /// Like expr_to_string but adds parens when the child operator has lower
    /// precedence than the parent (simple heuristic — add parens for binary
    /// ops inside binary ops when they'd otherwise be ambiguous).
    fn expr_to_string_parens(&mut self, expr: &Expr, parent_op: &BinOp) -> String {
        if let Expr::BinOp { op, .. } = expr {
            if binop_precedence(op) < binop_precedence(parent_op) {
                return format!("({})", self.expr_to_string(expr));
            }
        }
        self.expr_to_string(expr)
    }

    fn expr_to_string_parens_right(&mut self, expr: &Expr, parent_op: &BinOp) -> String {
        if let Expr::BinOp { op, .. } = expr {
            if binop_precedence(op) <= binop_precedence(parent_op)
                && !binop_is_assoc(parent_op)
            {
                return format!("({})", self.expr_to_string(expr));
            }
        }
        self.expr_to_string(expr)
    }

    fn if_expr_to_string(&mut self, ie: &IfExpr) -> String {
        let saved_indent = self.indent;
        let saved_buf = std::mem::take(&mut self.buf);

        self.buf.push_str("if ");
        let cond = self.expr_to_string(&ie.cond);
        self.buf.push_str(&cond);
        self.buf.push(' ');
        self.print_block(&ie.then_branch);

        if let Some(else_br) = &ie.else_branch {
            self.buf.push_str(" else ");
            match else_br {
                ElseBranch::Block(b) => self.print_block(b),
                ElseBranch::If(inner_ie) => {
                    let s = self.if_expr_to_string(inner_ie);
                    self.buf.push_str(&s);
                }
            }
        }

        let result = std::mem::replace(&mut self.buf, saved_buf);
        self.indent = saved_indent;
        result
    }

    fn match_expr_to_string(&mut self, me: &MatchExpr) -> String {
        let saved_indent = self.indent;
        let saved_buf = std::mem::take(&mut self.buf);

        let scr = self.expr_to_string(&me.scrutinee);
        self.buf.push_str("match ");
        self.buf.push_str(&scr);
        self.buf.push_str(" {\n");
        self.push_indent();

        for arm in &me.arms {
            let ind = self.indent_str();
            self.buf.push_str(&ind);
            let pat = self.pattern_to_string(&arm.pattern);
            self.buf.push_str(&pat);
            if let Some(guard) = &arm.guard {
                self.buf.push_str(" if ");
                let g = self.expr_to_string(guard);
                self.buf.push_str(&g);
            }
            self.buf.push_str(" => ");
            let body = self.expr_to_string(&arm.body);
            self.buf.push_str(&body);
            self.buf.push_str(",\n");
        }

        self.pop_indent();
        let ind = self.indent_str();
        self.buf.push_str(&ind);
        self.buf.push('}');

        let result = std::mem::replace(&mut self.buf, saved_buf);
        self.indent = saved_indent;
        result
    }

    fn postfix_to_string(&mut self, base: &str, op: &PostfixOp) -> String {
        match op {
            PostfixOp::Transpose => format!("{}.T", base),
            PostfixOp::Query => format!("{}?", base),
            PostfixOp::Field(name) => format!("{}.{}", base, name),

            PostfixOp::Call(args) => {
                let mut s = format!("{}(", base);
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { s.push_str(", "); }
                    match arg {
                        CallArg::Positional(e) => s.push_str(&self.expr_to_string(e)),
                        CallArg::Named { name, value, .. } => {
                            s.push_str(name);
                            s.push('=');
                            s.push_str(&self.expr_to_string(value));
                        }
                        CallArg::Spread(_) => s.push_str("..."),
                    }
                }
                s.push(')');
                s
            }

            PostfixOp::Index(elems) => {
                let mut s = format!("{}[", base);
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 { s.push_str(", "); }
                    match elem {
                        IndexElem::FullSlice(_) => s.push_str(".."),
                        IndexElem::Expr(e) => s.push_str(&self.expr_to_string(e)),
                        IndexElem::Slice { start, end, step, .. } => {
                            if let Some(st) = start {
                                s.push_str(&self.expr_to_string(st));
                            }
                            s.push_str("..");
                            if let Some(en) = end {
                                s.push_str(&self.expr_to_string(en));
                            }
                            if let Some(st) = step {
                                s.push_str("::");
                                s.push_str(&self.expr_to_string(st));
                            }
                        }
                    }
                }
                s.push(']');
                s
            }

            PostfixOp::BracketArgs(args) => {
                let mut s = format!("{}[", base);
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { s.push_str(", "); }
                    match arg {
                        CallArg::Positional(e) => s.push_str(&self.expr_to_string(e)),
                        CallArg::Named { name, value, .. } => {
                            s.push_str(name);
                            s.push('=');
                            s.push_str(&self.expr_to_string(value));
                        }
                        CallArg::Spread(_) => s.push_str("..."),
                    }
                }
                s.push(']');
                s
            }

            PostfixOp::Constructor(fields) => {
                let mut s = format!("{} {{", base);
                for (i, (fname, fval)) in fields.iter().enumerate() {
                    if i > 0 { s.push_str(", "); }
                    s.push(' ');
                    s.push_str(fname);
                    s.push_str(": ");
                    s.push_str(&self.expr_to_string(fval));
                }
                if !fields.is_empty() { s.push(' '); }
                s.push('}');
                s
            }
        }
    }

    // ── Types ─────────────────────────────────────────────────────────────

    fn type_to_string(&mut self, ty: &Type) -> String {
        match ty {
            Type::Scalar(s, _) => scalar_type_str(s).to_string(),

            Type::Tensor(elem, shape, _) => {
                let e = self.type_to_string(elem);
                let sh = self.shape_spec_to_string(shape);
                format!("Tensor[{}, {}]", e, sh)
            }

            Type::View(elem, shape, _) => {
                let e = self.type_to_string(elem);
                let sh = self.shape_spec_to_string(shape);
                format!("View[{}, {}]", e, sh)
            }

            Type::KV(elem, shape, _) => {
                let e = self.type_to_string(elem);
                let sh = self.shape_spec_to_string(shape);
                format!("KV[{}, {}]", e, sh)
            }

            Type::Mesh(axes, _) => {
                let parts: Vec<String> = axes.iter()
                    .map(|a| {
                        let sz = self.expr_to_string(&a.size);
                        format!("{}: {}", a.name, sz)
                    })
                    .collect();
                format!("Mesh[{}]", parts.join(", "))
            }

            Type::Fn(params, ret, _) => {
                let ps: Vec<String> = params.iter()
                    .map(|t| self.type_to_string(t))
                    .collect();
                let r = self.type_to_string(ret);
                format!("fn({}) -> {}", ps.join(", "), r)
            }

            Type::Tuple(types, _) => {
                let parts: Vec<String> = types.iter()
                    .map(|t| self.type_to_string(t))
                    .collect();
                format!("({})", parts.join(", "))
            }

            Type::Array(elem, size, _) => {
                let e = self.type_to_string(elem);
                let s = self.expr_to_string(size);
                format!("[{}; {}]", e, s)
            }

            Type::RawPtr(inner, _) => {
                format!("*{}", self.type_to_string(inner))
            }

            Type::Named { name, args, .. } => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let parts: Vec<String> = args.iter()
                        .map(|a| self.type_arg_to_string(a))
                        .collect();
                    format!("{}[{}]", name, parts.join(", "))
                }
            }
        }
    }

    fn type_arg_to_string(&mut self, arg: &TypeArg) -> String {
        match arg {
            TypeArg::Type(t) => self.type_to_string(t),
            TypeArg::Expr(e) => self.expr_to_string(e),
            TypeArg::Named { name, value, .. } => {
                let v = self.expr_to_string(value);
                format!("{}={}", name, v)
            }
        }
    }

    fn shape_spec_to_string(&mut self, ss: &ShapeSpec) -> String {
        let parts: Vec<String> = ss.elems.iter()
            .map(|e| self.shape_elem_to_string(e))
            .collect();
        format!("[{}]", parts.join(", "))
    }

    fn shape_elem_to_string(&mut self, elem: &ShapeElem) -> String {
        match elem {
            ShapeElem::Wildcard(_) => "_".to_string(),
            ShapeElem::Spread(_) => "..".to_string(),
            ShapeElem::Streaming(_) => "~".to_string(),
            ShapeElem::Expr(e) => self.expr_to_string(e),
        }
    }

    // ── Patterns ──────────────────────────────────────────────────────────

    fn pattern_to_string(&mut self, pat: &Pattern) -> String {
        match pat {
            Pattern::Wildcard(_) => "_".to_string(),
            Pattern::Rest(_) => "..".to_string(),
            Pattern::Ident(name, _) => name.clone(),
            Pattern::Literal(lit, _) => lit_to_string(lit),
            Pattern::Tuple(pats, _) => {
                let parts: Vec<String> = pats.iter()
                    .map(|p| self.pattern_to_string(p))
                    .collect();
                format!("({})", parts.join(", "))
            }
            Pattern::Shape(elems, _) => {
                let parts: Vec<String> = elems.iter()
                    .map(|e| self.shape_elem_to_string(e))
                    .collect();
                format!("[{}]", parts.join(", "))
            }
            Pattern::Bind(outer, inner, _) => {
                let o = self.pattern_to_string(outer);
                let i = self.pattern_to_string(inner);
                format!("{} @ {}", o, i)
            }
            Pattern::EnumVariant { enum_name, variant, bindings, .. } => {
                // `enum_name` is empty for the bare payload form `Circle(r)`.
                let head = if enum_name.is_empty() {
                    variant.clone()
                } else {
                    format!("{}.{}", enum_name, variant)
                };
                if bindings.is_empty() {
                    head
                } else {
                    // #350 Part 2: payload bindings `Circle(r)` / `Rect(w, h)`.
                    let bs: Vec<String> = bindings.iter().map(|b| self.pattern_to_string(b)).collect();
                    format!("{}({})", head, bs.join(", "))
                }
            }
        }
    }

    // ── Helper: param ─────────────────────────────────────────────────────

    fn param_to_string(&mut self, p: &Param, _is_method: bool) -> String {
        if p.is_self {
            if p.mutating {
                return "!self".to_string();
            }
            return "self".to_string();
        }
        let mut s = String::new();
        if p.mutating { s.push('!'); }
        s.push_str(&p.name);
        if let Some(ty) = &p.ty {
            s.push_str(": ");
            s.push_str(&self.type_to_string(ty));
        }
        s
    }
}

// ─── Pure helpers (no &mut self needed) ──────────────────────────────────────

fn lit_to_string(lit: &Literal) -> String {
    match lit {
        Literal::Int(n)   => n.to_string(),
        Literal::Float(f, suffix) => {
            // Preserve at least one decimal place so the lexer sees it as float
            let mut s = format!("{}", f);
            if !s.contains('.') && !s.contains('e') {
                s.push_str(".0");
            }
            if let Some(t) = suffix {
                s.push_str(scalar_type_str(t));
            }
            s
        }
        // #290: control characters must round-trip as escape sequences — the
        // lexer rejects raw newlines inside string literals, so emitting the
        // decoded value verbatim breaks the parse -> fmt -> parse guarantee.
        Literal::Str(s)   => {
            let escaped: String = s.chars().map(|c| match c {
                '\\' => "\\\\".to_string(),
                '"'  => "\\\"".to_string(),
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                '\0' => "\\0".to_string(),
                ch   => ch.to_string(),
            }).collect();
            format!("\"{}\"", escaped)
        }
        Literal::Char(c)  => {
            let escaped = match c {
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                '\\' => "\\\\".to_string(),
                '"'  => "\\\"".to_string(),
                '\0' => "\\0".to_string(),
                ch   => ch.to_string(),
            };
            format!("c\"{}\"", escaped)
        }
        Literal::Bool(b)  => if *b { "true".to_string() } else { "false".to_string() },
        Literal::Nil      => "nil".to_string(),
    }
}

fn scalar_type_str(s: &ScalarType) -> &'static str {
    match s {
        ScalarType::I8    => "i8",
        ScalarType::I16   => "i16",
        ScalarType::I32   => "i32",
        ScalarType::I64   => "i64",
        ScalarType::U8    => "u8",
        ScalarType::U16   => "u16",
        ScalarType::U32   => "u32",
        ScalarType::U64   => "u64",
        ScalarType::Int4  => "int4",
        ScalarType::Int8  => "int8",
        ScalarType::F16   => "f16",
        ScalarType::Bf16  => "bf16",
        ScalarType::Tf32  => "tf32",
        ScalarType::F32   => "f32",
        ScalarType::F64   => "f64",
        ScalarType::Fp8E4M3 => "fp8_e4m3",
        ScalarType::Fp8E5M2 => "fp8_e5m2",
        ScalarType::Trit  => "trit",
        ScalarType::Bool  => "bool",
        ScalarType::Str   => "str",
        ScalarType::Nil   => "nil",
    }
}

fn binop_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add     => "+",
        BinOp::Sub     => "-",
        BinOp::Mul     => "*",
        BinOp::Div     => "/",
        BinOp::Mod     => "%",
        BinOp::Pow     => "^",
        BinOp::StarStar => "**",
        BinOp::DotAdd  => ".+",
        BinOp::DotSub  => ".-",
        BinOp::DotMul  => ".*",
        BinOp::DotDiv  => "./",
        BinOp::DotPow  => ".^",
        BinOp::DotPow2 => ".**",
        BinOp::DotGt   => ".>",
        BinOp::DotLt   => ".<",
        BinOp::DotGe   => ".>=",
        BinOp::DotLe   => ".<=",
        BinOp::Matmul  => "@",
        BinOp::And     => "&&",
        BinOp::Or      => "||",
        BinOp::Eq      => "==",
        BinOp::NotEq   => "!=",
        BinOp::Lt      => "<",
        BinOp::Gt      => ">",
        BinOp::LtEq    => "<=",
        BinOp::GtEq    => ">=",
        BinOp::Pipe    => "|>",
        BinOp::RShift  => ">>",
        BinOp::BitAnd  => "&",
        BinOp::BitOr   => "|",
        BinOp::BitXor  => "^^",
        BinOp::BitShl  => "<<",
        BinOp::BitShr  => ">>",
    }
}

fn binop_precedence(op: &BinOp) -> u8 {
    match op {
        BinOp::Or     => 1,
        BinOp::And    => 2,
        BinOp::BitOr  => 3,
        BinOp::BitXor => 4,
        BinOp::BitAnd => 5,
        BinOp::Eq | BinOp::NotEq => 6,
        BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq
        | BinOp::DotGt | BinOp::DotLt | BinOp::DotGe | BinOp::DotLe => 7,
        BinOp::BitShl | BinOp::BitShr | BinOp::RShift => 8,
        BinOp::Add | BinOp::Sub | BinOp::DotAdd | BinOp::DotSub => 9,
        BinOp::Mul | BinOp::Div | BinOp::Mod
        | BinOp::DotMul | BinOp::DotDiv => 10,
        BinOp::Pow | BinOp::StarStar | BinOp::DotPow | BinOp::DotPow2 => 11,
        BinOp::Matmul => 12,
        BinOp::Pipe => 0,
    }
}

fn binop_is_assoc(op: &BinOp) -> bool {
    matches!(op,
        BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or
        | BinOp::DotAdd | BinOp::DotMul | BinOp::BitAnd
        | BinOp::BitOr | BinOp::BitXor
    )
}

fn assignop_str(op: &AssignOp) -> &'static str {
    match op {
        AssignOp::Eq          => "=",
        AssignOp::ColonEq     => ":=",
        AssignOp::PlusEq      => "+=",
        AssignOp::MinusEq     => "-=",
        AssignOp::StarEq      => "*=",
        AssignOp::SlashEq     => "/=",
        AssignOp::StreamArrow => "<-",
        AssignOp::AmpEq       => "&=",
        AssignOp::BarEq       => "|=",
        AssignOp::CaretEq     => "^=",
    }
}
