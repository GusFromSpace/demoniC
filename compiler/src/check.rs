/// demoniC typechecker — Phase 3 (pre-alpha).
///
/// Three-pass design per HAS-dC §10:
///   Pass 1 — collect_signatures: hoist all fn/model decls so forward references resolve
///   Pass 2 — check_types: walk every item checking shapes, arity, ? context, <- type
///   Pass 3 — arena_coherence: verify arena-specific rules (cross-arena writes deferred;
///             ? / <- pass-level checks already complete after pass 2)
///
/// Doesn't yet handle:
///   - @grad adjoint type derivation (Phase 3.x)
///   - @shard / @tp / @pp mesh feasibility (Phase 3.x)
///   - Full Presburger / SMT shape proofs (currently structural + algebraic)
///   - Stdlib return-type generics (Unknown until type vars land; arity now enforced)
///   - Cross-arena write detection (requires arena-tag propagation, Phase 3.x)

use std::{collections::{HashMap, HashSet}, fmt, path::{Path, PathBuf}};

use crate::ast::*;
use crate::lexer::Span;
use crate::shape::{Shape, ShapeError, SymDim};
use crate::types::{Env, FnSig, ModelInfo, TyType};

// ─── Diagnostic ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TypeError {
    pub msg: String,
    pub span: Span,
    pub hint: Option<String>,
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type error at {}:{}: {}", self.span.line, self.span.col, self.msg)?;
        if let Some(h) = &self.hint { write!(f, "\n  hint: {}", h)?; }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ModuleEnv {
    pub env: Env,
    pub aliases: HashMap<String, TypeAlias>,
    pub public_items: HashSet<String>,
}

// ─── Checker ─────────────────────────────────────────────────────────────────

pub struct Checker {
    pub env: Env,
    pub aliases: HashMap<String, TypeAlias>,
    resolving_aliases: Vec<String>,
    pub errors: Vec<TypeError>,
    /// Non-fatal lint diagnostics. Surfaced to the user but never block
    /// execution (`dmc run` proceeds even when warnings are present).
    pub warnings: Vec<TypeError>,
    /// Return type of the innermost function being checked; used for `?` context validation.
    current_fn_ret: Option<TyType>,
    /// Top-level user fns → number of stacked `@grad` directives (0 = plain
    /// fn). Drives check-time validation of the autodiff calling conventions
    /// (`f.fwd_bwd(...)` etc.): an unknown method name on a `@grad fn` used
    /// to fall through to generic field access and silently produce garbage
    /// at runtime.
    grad_fn_counts: HashMap<String, usize>,
    /// #403 (MEMORY §2): per-fn param mutability (`!` markers), by position.
    /// Lets call sites treat an uninit binding passed to a `!` param as the
    /// fill that initializes it, and one passed to a plain param as a read.
    fn_mut_params: HashMap<String, Vec<bool>>,
    /// #442 (MEMORY §3.1): the lexically enclosing arena blocks, innermost
    /// last. Empty = the default Forge context (MEMORY §3 rule 3).
    arena_stack: Vec<ArenaKind>,
    /// PORTS.md §5: set while checking a `@grad fn` body (closures included).
    /// A port call inside the tape is an effect the gradient cannot cross, so
    /// `port_open`/`port_call`/`port_close` are rejected here (`port-forbidden`).
    in_grad_fn: bool,
    pub checked_modules: HashMap<PathBuf, ModuleEnv>,
    /// Demon mode: the Control Art Restriction released. When set, the
    /// safe-mode lint family (`warn()`) is suppressed entirely — raw, full
    /// speed, no guardrails (issue #196). Hard type errors still fire.
    pub demon: bool,
}

impl Checker {
    pub fn new() -> Self {
        Self {
            env: Env::new(),
            aliases: HashMap::new(),
            resolving_aliases: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            current_fn_ret: None,
            grad_fn_counts: HashMap::new(),
            fn_mut_params: HashMap::new(),
            arena_stack: Vec::new(),
            in_grad_fn: false,
            checked_modules: HashMap::new(),
            demon: false,
        }
    }

    pub fn check_program(&mut self, program: &Program, path: Option<&Path>) {
        self.process_imports(program, path);
        self.pass1_collect_signatures(program);
        self.pass2_check_types(program);
        self.pass3_arena_coherence(program);
        self.lint_writeback(program);
    }

    /// Lint pass: flag the "dead write-back" pattern — a mutable binding
    /// copied out of a tensor element (`let !x = t[i]`) that is then assigned
    /// (`x = v`) but never read. Scalar indexing copies the element (it does
    /// not alias the tensor; see `MEMORY.md §4`), so such an assignment can
    /// never reach the tensor. It is a silent no-op, almost always a missed
    /// `t[i] = ...` write-through. Gating on "never read" spares legitimate
    /// scratch accumulators (`let !m = x[0]; ... if x[i] > m { m = x[i] }`),
    /// which are read.
    fn lint_writeback(&mut self, program: &Program) {
        for item in &program.items {
            lint_item_writeback(item, &mut self.warnings);
        }
    }

    fn process_imports(&mut self, program: &Program, path: Option<&Path>) {
        for item in &program.items {
            if let Item::Use(us) = item {
                if let Some(current_path) = path {
                    let parent_dir = current_path.parent().unwrap_or_else(|| Path::new(""));
                    let import_path = parent_dir.join(&us.path);
                    let canonical_path = match import_path.canonicalize() {
                        Ok(p) => p,
                        Err(e) => {
                            self.error(
                                format!("error resolving import {:?} in file {:?}: {}", us.path, current_path, e),
                                us.span.clone(),
                            );
                            continue;
                        }
                    };
                    if let Some(imported_env) = self.checked_modules.get(&canonical_path).cloned() {
                        if let Some(alias) = &us.alias {
                            self.env.bind(alias.clone(), TyType::Module { alias: alias.clone(), path: canonical_path.clone() });

                            let imported_model_names: std::collections::HashSet<String> = imported_env.env.models.keys().cloned().collect();

                            // Prefix functions
                            for (name, sig) in &imported_env.env.functions {
                                if !imported_env.public_items.contains(name) {
                                    continue;
                                }
                                let prefixed_params = sig.params.iter().map(|(pname, pty)| {
                                    (pname.clone(), prefix_types_in_ty(pty.clone(), alias, &imported_model_names))
                                }).collect();
                                let prefixed_ret = prefix_types_in_ty(sig.ret.clone(), alias, &imported_model_names);
                                self.env.functions.insert(
                                    format!("{}.{}", alias, name),
                                    FnSig {
                                        shape_params: sig.shape_params.clone(),
                                        params: prefixed_params,
                                        ret: prefixed_ret,
                                    }
                                );
                            }

                            // Prefix models
                            for (name, info) in &imported_env.env.models {
                                if !imported_env.public_items.contains(name) {
                                    continue;
                                }
                                let mut prefixed_fields = HashMap::new();
                                for (fname, fty) in &info.fields {
                                    prefixed_fields.insert(fname.clone(), prefix_types_in_ty(fty.clone(), alias, &imported_model_names));
                                }
                                let mut prefixed_methods = HashMap::new();
                                for (mname, msig) in &info.methods {
                                    let prefixed_mparams = msig.params.iter().map(|(pname, pty)| {
                                        (pname.clone(), prefix_types_in_ty(pty.clone(), alias, &imported_model_names))
                                    }).collect();
                                    let prefixed_mret = prefix_types_in_ty(msig.ret.clone(), alias, &imported_model_names);
                                    prefixed_methods.insert(
                                        mname.clone(),
                                        FnSig {
                                            shape_params: msig.shape_params.clone(),
                                            params: prefixed_mparams,
                                            ret: prefixed_mret,
                                        }
                                    );
                                }
                                self.env.models.insert(
                                    format!("{}.{}", alias, name),
                                    ModelInfo {
                                        shape_params: info.shape_params.clone(),
                                        fields: prefixed_fields,
                                        methods: prefixed_methods,
                                    }
                                );
                            }

                            for (name, type_alias) in &imported_env.aliases {
                                if !imported_env.public_items.contains(name) {
                                    continue;
                                }
                                self.aliases.insert(format!("{}.{}", alias, name), type_alias.clone());
                            }
                        } else {
                            // Unqualified imports: merge everything directly.
                            // Error if a name is already exported by a different module —
                            // use `use "..." as alias` to disambiguate.
                            for (name, sig) in imported_env.env.functions {
                                if !imported_env.public_items.contains(&name) {
                                    continue;
                                }
                                if self.env.functions.contains_key(&name) {
                                    self.error(
                                        format!("ambiguous import: `{}` is exported by multiple modules; \
                                            use aliased imports (`use \"...\" as alias`) to disambiguate", name),
                                        us.span.clone(),
                                    );
                                } else {
                                    self.env.functions.insert(name, sig);
                                }
                            }
                            for (name, info) in imported_env.env.models {
                                if !imported_env.public_items.contains(&name) {
                                    continue;
                                }
                                if self.env.models.contains_key(&name) {
                                    self.error(
                                        format!("ambiguous import: `{}` is exported by multiple modules; \
                                            use aliased imports (`use \"...\" as alias`) to disambiguate", name),
                                        us.span.clone(),
                                    );
                                } else {
                                    self.env.models.insert(name, info);
                                }
                            }
                            for (name, type_alias) in imported_env.aliases {
                                if !imported_env.public_items.contains(&name) {
                                    continue;
                                }
                                if self.aliases.contains_key(&name) {
                                    self.error(
                                        format!("ambiguous import: `{}` is exported by multiple modules; \
                                            use aliased imports (`use \"...\" as alias`) to disambiguate", name),
                                        us.span.clone(),
                                    );
                                } else {
                                    self.aliases.insert(name, type_alias);
                                }
                            }
                        }
                    } else {
                        self.error(
                            format!("imported module not found / checked yet: {:?}", canonical_path),
                            us.span.clone(),
                        );
                    }
                } else {
                    self.error(
                        "cannot resolve imports without a file path context",
                        us.span.clone(),
                    );
                }
            }
        }
    }

    // ── Pass 1: hoist all declarations so forward references resolve ──────

    fn pass1_collect_signatures(&mut self, program: &Program) {
        for item in &program.items {
            self.collect_alias_item(item);
        }
        for item in &program.items {
            self.register_item_signatures(item);
        }
    }

    fn collect_alias_item(&mut self, item: &Item) {
        match item {
            Item::TypeAlias(alias) => {
                self.aliases.insert(alias.name.clone(), alias.clone());
            }
            // #336: register enums in this first pass1 loop so a later
            // model field or fn signature can resolve an enum-typed annotation.
            Item::Enum(e) => self.register_enum(e),
            Item::Directive { inner, .. } => self.collect_alias_item(inner),
            Item::Pub(inner) => self.collect_alias_item(inner),
            _ => {}
        }
    }

    fn register_item_signatures(&mut self, item: &Item) {
        match item {
            Item::Fn(f)        => self.register_fn(f),
            Item::ExternFn(e)  => self.register_extern_fn(e),
            Item::Model(m)     => self.register_model(m),
            Item::Directive { inner, .. } => self.register_item_signatures(inner),
            Item::Pub(inner) => self.register_item_signatures(inner),
            // Enums are registered earlier (collect_alias_item).
            Item::Enum(_) | Item::TypeAlias(_) | Item::Arena(_) | Item::Let(_) | Item::Use(_) => {}
        }
    }

    // ── Pass 2: type-check all fn bodies, lets, arenas ───────────────────

    fn pass2_check_types(&mut self, program: &Program) {
        for item in &program.items {
            self.check_item(item);
        }
    }

    // ── Pass 3: arena coherence ───────────────────────────────────────────
    // `?` and `<-` violations are reported during pass 2 via current_fn_ret.
    // #442: the binding-level cross-arena write check also runs in pass 2
    // (Vault-origin tags + the lexical `arena_stack`). What remains here is
    // arena-tag *propagation* — through aliases, `!` params, and model
    // fields — which needs arena-qualified types (#442 follow-ups).

    fn pass3_arena_coherence(&mut self, _program: &Program) {}

    fn register_fn(&mut self, f: &FnDecl) {
        let shape_params: Vec<String> = f.shape_params.iter().map(|sp| sp.name.clone()).collect();
        // To resolve param/ret types, push the shape params into scope.
        self.env.push_shape_scope();
        for sp in &shape_params {
            self.env.bind_shape_param(sp, None);
        }
        let params: Vec<(String, TyType)> = f.params.iter().map(|p| {
            let ty = p.ty.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unknown);
            (p.name.clone(), ty)
        }).collect();
        let ret = f.ret_type.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unit);
        self.env.pop_shape_scope();
        self.env.functions.insert(f.name.clone(), FnSig { shape_params, params, ret });
        self.grad_fn_counts.insert(
            f.name.clone(),
            f.directives.iter().filter(|d| d.name == "grad").count(),
        );
        self.fn_mut_params.insert(
            f.name.clone(),
            f.params.iter().map(|p| p.mutating).collect(),
        );
    }

    fn register_extern_fn(&mut self, e: &ExternFnDecl) {
        let shape_params: Vec<String> = e.shape_params.iter().map(|sp| sp.name.clone()).collect();
        self.env.push_shape_scope();
        for sp in &shape_params {
            self.env.bind_shape_param(sp, None);
        }
        let params: Vec<(String, TyType)> = e.params.iter().map(|p| {
            let ty = p.ty.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unknown);
            (p.name.clone(), ty)
        }).collect();
        let ret = e.ret_type.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unit);
        self.env.pop_shape_scope();
        self.env.functions.insert(e.name.clone(), FnSig { shape_params, params, ret });
        self.fn_mut_params.insert(
            e.name.clone(),
            e.params.iter().map(|p| p.mutating).collect(),
        );
    }

    fn register_model(&mut self, m: &ModelDecl) {
        let shape_params: Vec<String> = m.shape_params.iter().map(|sp| sp.name.clone()).collect();
        self.env.push_shape_scope();
        for sp in &shape_params {
            self.env.bind_shape_param(sp, None);
        }
        let mut fields = std::collections::HashMap::new();
        let mut methods = std::collections::HashMap::new();
        for member in &m.members {
            match member {
                ModelMember::Field { name, ty, .. } => {
                    fields.insert(name.clone(), self.resolve_type(ty));
                }
                ModelMember::Method(f) => {
                    let m_shape: Vec<String> = f.shape_params.iter().map(|sp| sp.name.clone()).collect();
                    self.env.push_shape_scope();
                    for sp in &m_shape {
                        self.env.bind_shape_param(sp, None);
                    }
                    let params: Vec<(String, TyType)> = f.params.iter().map(|p| {
                        let ty = p.ty.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unknown);
                        (p.name.clone(), ty)
                    }).collect();
                    let ret = f.ret_type.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unit);
                    self.env.pop_shape_scope();
                    methods.insert(f.name.clone(), FnSig { shape_params: m_shape, params, ret });
                }
            }
        }
        self.env.pop_shape_scope();
        self.env.models.insert(m.name.clone(), ModelInfo { shape_params, fields, methods });
    }

    /// #336: register a C-like enum — name → ordered variant names. The
    /// parser already rejects duplicate variants and empty enums; here we only
    /// flag a name that collides with an existing enum or model.
    fn register_enum(&mut self, e: &EnumDecl) {
        if self.env.enums.contains_key(&e.name) || self.env.models.contains_key(&e.name) {
            self.error(format!("type `{}` is already declared", e.name), e.span.clone());
            return;
        }
        let variants: Vec<String> = e.variants.iter().map(|v| v.name.clone()).collect();
        self.env.enums.insert(e.name.clone(), variants);
        // #350 Part 2: record positional payload types per variant (raw AST,
        // resolved lazily at use sites). Only variants that carry data appear.
        let payloads: HashMap<String, Vec<Type>> = e.variants.iter()
            .filter(|v| !v.fields.is_empty())
            .map(|v| (v.name.clone(), v.fields.clone()))
            .collect();
        if !payloads.is_empty() {
            self.env.enum_payloads.insert(e.name.clone(), payloads);
        }
    }

    fn check_item(&mut self, item: &Item) {
        self.check_item_vis(item, false);
    }

    /// Like `check_item`, but threads whether the item is under a `pub`. Only
    /// `extern fn` cares: SPEC §9 makes `pub extern fn` a compile-time error.
    fn check_item_vis(&mut self, item: &Item, is_public: bool) {
        match item {
            Item::Fn(f)       => self.check_fn(f),
            Item::ExternFn(e) => {
                // SPEC.md §6.8 / §9: an `extern fn` is always exported in the
                // linkage sense (foreign symbols are a process-wide resource),
                // so the `pub` keyword on it is meaningless — and a compile-time
                // error. Nothing else to check: an extern decl has no body.
                if is_public {
                    self.error(
                        format!(
                            "`pub` is not allowed on `extern fn` `{}` — an `extern fn` is \
                             always exported; the `pub` keyword on it is a compile-time error \
                             (see the spec's `extern fn` rules)",
                            e.name),
                        e.span.clone(),
                    );
                }
            }
            Item::Model(m)    => self.check_model(m),
            Item::TypeAlias(_) => {}
            Item::Enum(_) => {} // registered in pass 1; nothing to check in a body
            Item::Arena(a) => {
                self.arena_stack.push(a.kind.clone());
                let _ = self.check_block(&a.body);
                self.arena_stack.pop();
            }
            Item::Let(l)   => { self.check_let(l); }
            Item::Use(_)   => {}
            Item::Directive { directives, inner, .. } => {
                self.check_directive_item(directives, inner, is_public);
            }
            Item::Pub(inner) => self.check_item_vis(inner, true),
        }
    }

    /// #369: directives the compiler actually acts on. Any `@ident` parses, so
    /// a documented-but-unimplemented directive (`@comptime`, `@inplace`,
    /// `@recompute`) or a typo is otherwise a silent no-op — warn instead of
    /// quietly doing nothing. (`@host` stays effective via `@host match`.)
    fn lint_unimplemented_directives(&mut self, directives: &[Directive]) {
        const EFFECTIVE: &[&str] =
            &["grad", "cast", "shard", "tp", "pp", "fuse", "deterministic", "host"];
        for d in directives {
            if !EFFECTIVE.contains(&d.name.as_str()) {
                self.warn(
                    format!("directive `@{}` is not implemented — it is parsed but has no effect", d.name),
                    d.span.clone(),
                    Some("remove it; it has no effect in this version".to_string()),
                );
            }
        }
    }

    fn check_directive_item(&mut self, directives: &[Directive], inner: &Item, is_public: bool) {
        self.lint_unimplemented_directives(directives);
        match inner {
            Item::Let(l) => {
                let ty = self.check_let(l);
                self.check_sharding_directives(directives, &ty, l.span.clone());
            }
            Item::Pub(inner_inner) => self.check_directive_item(directives, inner_inner, true),
            _ => self.check_item_vis(inner, is_public),
        }
    }

    fn check_fn(&mut self, f: &FnDecl) {
        self.lint_unimplemented_directives(&f.directives);
        self.check_grad_fn_differentiable(f);
        self.check_pp_directive(f);
        self.env.push_scope();
        self.env.push_shape_scope();
        for sp in &f.shape_params {
            self.env.bind_shape_param(&sp.name, None);
        }
        // Implicit shape params: collect any uppercase single-word identifiers
        // referenced in param types and bind them as shape params if not
        // already declared. Matches the pattern in kvcache.dmc where `S`
        // appears in `Tensor[i32, [B, S]]` without being in `[B, H, D]`.
        let mut implicit = std::collections::HashSet::new();
        for p in &f.params {
            if let Some(t) = &p.ty {
                collect_shape_vars(t, &mut implicit);
            }
        }
        if let Some(rt) = &f.ret_type {
            collect_shape_vars(rt, &mut implicit);
        }
        for name in &implicit {
            if !self.env.shape_param_in_scope(name) {
                self.env.bind_shape_param(name, None);
            }
        }
        for p in &f.params {
            let ty = p.ty.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unknown);
            if p.mutating {
                self.env.bind_mutable(p.name.clone(), ty);
            } else {
                self.env.bind(&p.name, ty);
            }
        }
        // Track the current fn's return type for ? context validation (pass 2 / arena coherence).
        let declared_ret = f.ret_type.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unit);
        let outer_fn_ret = self.current_fn_ret.replace(declared_ret.clone());
        // PORTS.md §5: port calls are an effect boundary a gradient cannot
        // cross. Track the `@grad fn` body (and any closures it checks) so the
        // call site can reject them.
        let outer_in_grad = self.in_grad_fn;
        self.in_grad_fn = f.directives.iter().any(|d| d.name == "grad");
        let body_ty = self.check_block(&f.body);
        self.in_grad_fn = outer_in_grad;
        self.current_fn_ret = outer_fn_ret;
        // #398: a captured mutable binding (a module-level `let !`) referenced in
        // a `@grad fn` is NOT differentiable — only the fn's `!` params enter the
        // gradient tape, so its gradient silently comes back absent (a downstream
        // `grads.<name>` reads as an opaque nil). Flag it at compile time. This
        // runs after `check_block`, so body-local bindings are already popped from
        // the env: a mutable binding here that is neither a param nor a local can
        // only be a captured module-level `let !`.
        if self.grad_fn_counts.get(&f.name).copied().unwrap_or(0) >= 1 {
            let params: std::collections::HashSet<&str> =
                f.params.iter().map(|p| p.name.as_str()).collect();
            let mut refs = std::collections::HashSet::new();
            collect_body_idents(&f.body, &mut refs);
            let mut captured: Vec<&String> = refs.iter()
                .filter(|n| !params.contains(n.as_str()) && self.env.is_mutable_binding(n))
                .collect();
            captured.sort();
            for name in captured {
                self.error_with_hint(
                    format!("captured mutable binding `{}` is not differentiable inside a \
                             `@grad fn` — its gradient is silently absent", name),
                    f.body.span.clone(),
                    Some(format!("pass `{}` as a `!` parameter so it enters the gradient tape", name)),
                );
            }
        }
        // If the body ends with `return X` and has no tail expression, the
        // function always exits through that return statement. The return
        // statement's own check (above in check_stmt) already verified the
        // type; suppress the false body-type mismatch here.
        let effective_body_ty = if f.body.tail_expr.is_none() {
            match f.body.stmts.last() {
                Some(Stmt::Return { value: Some(_), .. }) => TyType::Unknown,
                _ => body_ty,
            }
        } else { body_ty };
        // Declared return type must be compatible with the body's produced type.
        // Unknown body type (e.g. block with only statements) is always accepted.
        if !declared_ret.compatible_with(&effective_body_ty) && !matches!(effective_body_ty, TyType::Unknown) {
            // A body that ends in a `let` binding has no value, so it types as
            // `nil` and trips this mismatch. The bare "body produces nil" is
            // misleading — it points at the return type when the real problem is
            // a missing tail expression. A frequent cause: two statements share a
            // line, and `<expr> (…)` / `<expr> [.…]` is parsed as a call/index
            // that swallows what was meant to be the final expression (a newline
            // between them disambiguates). Give a diagnostic that names this.
            let ends_in_let = f.body.tail_expr.is_none()
                && matches!(f.body.stmts.last(), Some(Stmt::Let(_)));
            if ends_in_let && matches!(effective_body_ty, TyType::Unit) {
                self.error_with_hint(
                    format!("fn `{}` returns `{}` but its body ends in a `let` binding, which has no value",
                            f.name, declared_ret),
                    f.body.span.clone(),
                    Some("a function body must end in an expression. If you meant to write two \
                          statements, put them on separate lines — a value followed by `(…)` or \
                          `[…]` on the same line parses as a call/index and can swallow the next line."
                          .to_string()),
                );
            } else {
                self.error_with_hint(
                    format!("fn `{}` returns `{}` but body produces `{}`", f.name, declared_ret, effective_body_ty),
                    f.body.span.clone(),
                    Some(format!("declared return type at line {}", f.span.line)),
                );
            }
        }
        // #295: `fn f() -> i8 { 300 }` — a bare-literal tail expression that
        // overflows the declared return type (the implicit-return analog of the
        // `Stmt::Return` check).
        if let Some(tail) = &f.body.tail_expr {
            self.check_int_literal_range(&declared_ret, tail);
        }
        self.env.pop_shape_scope();
        self.env.pop_scope();
    }

    /// AUTODIFF.md §2: a `@grad fn` that captures **no** mut bindings and has
    /// **no** mut (`!`) parameters is a compile-time error — there is nothing
    /// for the backward to produce. The JIT already enforces this in
    /// `declare_grad_fn` (jit.rs); mirror it here so `--check` and the
    /// tree-walking `run` reject it too, not just `--jit`. Captured mut bindings
    /// are specified-but-not-yet-implemented (#398) and top-level/method fns
    /// have no captures, so the mut-parameter set is the only source of
    /// differentiable inputs today — matching the JIT's own condition.
    fn check_grad_fn_differentiable(&mut self, f: &FnDecl) {
        let is_grad = f.directives.iter().any(|d| d.name == "grad");
        if !is_grad {
            return;
        }
        let has_mut_param = f.params.iter().any(|p| p.mutating);
        if !has_mut_param {
            self.error(
                "`@grad fn` with no `!` (mut) parameter has nothing to differentiate \
                 (see the spec on `@grad`)",
                f.span.clone(),
            );
        }
    }

    fn check_pp_directive(&mut self, f: &FnDecl) {
        let Some(pp) = f.directives.iter().find(|d| d.name == "pp") else {
            return;
        };
        let Some(stages) = directive_i64_arg(pp, "stages") else {
            self.error("@pp requires integer `stages` argument", pp.span.clone());
            return;
        };
        if stages <= 0 {
            self.error("@pp `stages` must be positive", pp.span.clone());
            return;
        }
        if f.body.tail_expr.is_some() {
            self.error("@pp body must contain only `stage K:` statements", f.body.span.clone());
        }
        if f.body.stmts.len() != stages as usize {
            self.error(
                format!("@pp(stages={}) requires exactly {} stage statements", stages, stages),
                f.body.span.clone(),
            );
        }
        for (idx, stmt) in f.body.stmts.iter().enumerate() {
            match stmt {
                Stmt::Stage { stage, span, .. } => {
                    if *stage != idx as i64 {
                        self.error(
                            format!("@pp stage index must be {}; got {}", idx, stage),
                            span.clone(),
                        );
                    }
                }
                other => {
                    self.error(
                        format!("@pp body cannot contain `{}`", stmt_kind(other)),
                        f.body.span.clone(),
                    );
                }
            }
        }
    }

    fn check_model(&mut self, m: &ModelDecl) {
        self.lint_unimplemented_directives(&m.directives);
        self.env.push_shape_scope();
        for sp in &m.shape_params {
            self.env.bind_shape_param(&sp.name, None);
        }
        let model_shape_names: std::collections::HashSet<String> =
            m.shape_params.iter().map(|sp| sp.name.clone()).collect();
        for member in &m.members {
            if let ModelMember::Method(f) = member {
                // Detect shape vars used in method signature that are not declared on the
                // model or the method itself. Emit a targeted diagnostic before body checking
                // so users aren't left wondering where an unfamiliar symbol came from.
                let method_shape_names: std::collections::HashSet<String> =
                    f.shape_params.iter().map(|sp| sp.name.clone()).collect();
                let mut all_used = std::collections::HashSet::new();
                for p in &f.params {
                    if let Some(t) = &p.ty { collect_shape_vars(t, &mut all_used); }
                }
                if let Some(rt) = &f.ret_type { collect_shape_vars(rt, &mut all_used); }
                let mut undeclared: Vec<String> = all_used
                    .into_iter()
                    .filter(|v| !method_shape_names.contains(v) && !model_shape_names.contains(v))
                    .collect();
                if !undeclared.is_empty() {
                    undeclared.sort();
                    let model_params: Vec<&str> = m.shape_params.iter()
                        .map(|sp| sp.name.as_str()).collect();
                    self.error_with_hint(
                        format!(
                            "unknown shape param{} `{}` in method `{}` of model `{}`",
                            if undeclared.len() == 1 { "" } else { "s" },
                            undeclared.join("`, `"),
                            f.name,
                            m.name,
                        ),
                        f.span.clone(),
                        Some(format!(
                            "did you mean `fn {}[{}](self, ...)`? \
                             model-level shape params [{}] are not automatically in scope inside methods",
                            f.name,
                            undeclared.join(", "),
                            model_params.join(", "),
                        )),
                    );
                }
                // Bind self → the model type with this model's shape params as args
                self.env.push_scope();
                let self_args: Vec<TyType> = m.shape_params.iter()
                    .map(|_sp| TyType::Unknown)  // generic-pass-through; real binding in instantiation
                    .collect();
                let _ = self_args;
                self.env.bind("self", TyType::Named {
                    name: m.name.clone(),
                    args: Vec::new(),
                });
                self.check_fn(f);
                self.env.pop_scope();
            }
        }
        self.env.pop_shape_scope();
    }

    /// #295: a directly-written integer literal (`300`, `-200`) must fit the
    /// concrete narrow integral type it is bound to — `let x: i8 = 300` is a
    /// compile-time error. Only fires for *syntactic* literals (`expr_i64`, which
    /// folds an int literal and a leading unary `-`), so arithmetic like
    /// `200 - 100` — whose `IntLit` type carries only the left operand's value —
    /// is never falsely flagged. No-op for non-integral or wide (`i64`-fitting)
    /// targets, and for the quantization int kinds `int_scalar_range` skips.
    fn check_int_literal_range(&mut self, target: &TyType, value_expr: &Expr) {
        let TyType::Scalar(s) = target else { return };
        let Some((lo, hi)) = int_scalar_range(s.clone()) else { return };
        let Some(n) = expr_i64(value_expr) else { return };
        let v = n as i128;
        if v < lo || v > hi {
            self.error(
                format!(
                    "integer literal {} out of range for `{}` (valid range {}..={})",
                    n, target, lo, hi,
                ),
                value_expr.span_of(),
            );
        }
    }

    fn check_let(&mut self, l: &LetStmt) -> TyType {
        let value_ty = self.check_expr(&l.value);
        // Footgun lint (#232): `let !x = x` rebinds a name to a copy of itself —
        // redundant dead code that type-checks clean, so nothing else catches
        // it. Only fire when the RHS name actually resolves, so
        // a genuine `let x = x` on an undefined `x` stays a plain undefined error.
        if let (Pattern::Ident(pname, _), Expr::Ident(vname, _)) = (&l.pattern, &l.value) {
            if pname == vname && self.env.lookup(vname).is_some() {
                self.warn(
                    format!("identity rebind `let {} = {}` is a redundant self-copy", pname, vname),
                    l.span.clone(),
                    Some("drop the rebind — dead code (also the signature of degenerate codegen)".to_string()),
                );
            }
        }
        let declared = l.ty.as_ref().map(|t| self.resolve_type(t));
        let final_ty = if let Some(d) = declared {
            if !d.compatible_with(&value_ty) {
                self.error(
                    format!("let binding has type {} but value has type {}", d, value_ty),
                    l.span.clone(),
                );
            }
            // #295: catch `let x: i8 = 300` — a literal that doesn't fit its
            // annotated narrow type (compatible_with only checks integrality).
            self.check_int_literal_range(&d, &l.value);
            d
        } else {
            // No annotation: an untyped int literal defaults to i64 (#295).
            concretize(value_ty)
        };
        // #245: tuple destructuring arity check.
        // When both the pattern arity and the bound type's arity are statically
        // known (concrete tuple on each side), mismatches are a compile-time error.
        if let (Pattern::Tuple(pats, _), TyType::Tuple(tys)) = (&l.pattern, &final_ty) {
            self.check_tuple_pattern_arity(pats, tys.len(), &final_ty, &l.span);
        }
        // #248: record (or clear) the static constructor shape for a simple
        // binding, so static OOB indexing through it is catchable. Kept separate
        // from `final_ty` — see `Env::set_ctor_shape`. Only when there's no
        // annotation: an explicit `let k: KV[..] = forge.ones[..]` must not be
        // treated as a plain tensor (and would round-trip the wrong type anyway).
        if let Pattern::Ident(name, _) = &l.pattern {
            let ctor_shape = if l.ty.is_none() {
                ctor_tensor_ty(&l.value).and_then(|t| match t {
                    TyType::Tensor(_, s) => Some(s),
                    _ => None,
                })
            } else {
                None
            };
            self.env.set_ctor_shape(name.clone(), ctor_shape);
        }
        // #403 (MEMORY §2): a fresh `forge.uninit`/`vault.uninit` allocation is
        // undefined until written — mark simple-ident bindings for the
        // definite-assignment check (reads before the first write error). A
        // same-scope rebind to anything else masks the flag in this scope;
        // inner-scope shadows are masked by the scope walk itself.
        if let Pattern::Ident(name, _) = &l.pattern {
            if is_uninit_ctor(&l.value) {
                self.env.mark_uninit(name.clone());
            } else {
                self.env.unmark_uninit_here(name);
            }
            // #442 (MEMORY §3.1): tag bindings whose value lives in the Vault,
            // so mutating them outside a `vault { … }` block can error.
            if is_vault_ctor(&l.value) {
                self.env.mark_vault_origin(name.clone());
            } else {
                self.env.unmark_vault_origin_here(name);
            }
        }
        // Bind pattern names; `let !` and `let mut` both produce mutable bindings.
        if l.mutating || l.is_mut {
            if let Pattern::Ident(name, _) = &l.pattern {
                self.env.bind_mutable(name.clone(), final_ty.clone());
                return final_ty;
            }
        }
        self.bind_pattern(&l.pattern, &final_ty);
        final_ty
    }

    fn check_directive_stmt(&mut self, directives: &[Directive], inner: &Stmt) {
        self.lint_unimplemented_directives(directives);
        if let Stmt::Let(l) = inner {
            let ty = self.check_let(l);
            self.check_sharding_directives(directives, &ty, l.span.clone());
        } else {
            self.check_stmt(inner);
        }
    }

    fn check_sharding_directives(&mut self, directives: &[Directive], ty: &TyType, span: Span) {
        for directive in directives {
            match directive.name.as_str() {
                "shard" => {
                    let Some(axis) = directive_i64_arg(directive, "axis") else {
                        self.error("@shard requires integer `axis` argument", directive.span.clone());
                        continue;
                    };
                    let Some(divisor) = directive_mesh_axis_arg(directive) else {
                        self.error("@shard requires `mesh=mesh.axis` argument", directive.span.clone());
                        continue;
                    };
                    self.check_sharded_axis("@shard", ty, axis, &divisor, span.clone());
                }
                "tp" => {
                    let Some(axis) = directive_i64_arg(directive, "axis") else {
                        self.error("@tp requires integer `axis` argument", directive.span.clone());
                        continue;
                    };
                    self.check_sharded_axis("@tp", ty, axis, "tp", span.clone());
                }
                _ => {}
            }
        }
    }

    fn check_sharded_axis(
        &mut self,
        directive: &str,
        ty: &TyType,
        raw_axis: i64,
        divisor: &str,
        span: Span,
    ) {
        let Some((_elem, shape)) = ty.as_tensor_like() else {
            self.error(format!("{} requires tensor-like let binding; got `{}`", directive, ty), span);
            return;
        };
        let Some(axis) = normalize_axis(raw_axis, shape.rank()) else {
            self.error(
                format!("{} axis {} out of bounds for rank {}", directive, raw_axis, shape.rank()),
                span,
            );
            return;
        };
        if !symdim_divided_by(&shape.dims[axis], divisor) {
            self.error(
                format!(
                    "{} axis {} shape `{}` must include divisor `{}`",
                    directive, raw_axis, shape.dims[axis], divisor,
                ),
                span,
            );
        }
    }

    fn bind_pattern(&mut self, pat: &Pattern, ty: &TyType) {
        match pat {
            // #336: a bare ident that names a variant of the scrutinee's enum is
            // a *variant* match, not a fresh binding — don't shadow it.
            Pattern::Ident(name, _)
                if matches!(ty, TyType::Enum(en) if self.env.enums.get(en).is_some_and(|vs| vs.contains(name))) => {}
            Pattern::Ident(name, _) if name != "_" => {
                self.env.bind(name, ty.clone());
            }
            Pattern::EnumVariant { .. } => {}
            Pattern::Wildcard(_) => {}
            Pattern::Tuple(pats, _) => {
                if let TyType::Tuple(tys) = ty {
                    let (before, after, has_rest) = crate::ast::tuple_rest_split(pats);
                    let ok = if has_rest { tys.len() >= before.len() + after.len() }
                             else { pats.len() == tys.len() };
                    if ok {
                        if has_rest {
                            for (p, t) in before.iter().zip(tys) { self.bind_pattern(p, t); }
                            let tail = tys.len() - after.len();
                            for (p, t) in after.iter().zip(&tys[tail..]) { self.bind_pattern(p, t); }
                        } else {
                            for (p, t) in pats.iter().zip(tys) { self.bind_pattern(p, t); }
                        }
                        return;
                    }
                }
                // Unknown tuple shape — bind each named element as Unknown.
                for p in pats {
                    self.bind_pattern(p, &TyType::Unknown);
                }
            }
            Pattern::Shape(_, _) => {
                // Shape patterns destructure shape elements; we don't bind names.
            }
            Pattern::Literal(_, _) => {}
            Pattern::Ident(_, _) => {}
            Pattern::Bind(inner, _, _) => self.bind_pattern(inner, ty),
            Pattern::Rest(_) => {}  // `..` binds no name
        }
    }

    /// Arity-check a tuple destructuring pattern against a known tuple width.
    /// A `..` rest relaxes the check to "at least the fixed head+tail count";
    /// more than one `..` in a tuple is rejected (its meaning is ambiguous).
    fn check_tuple_pattern_arity(&mut self, pats: &[Pattern], width: usize, ty: &TyType, span: &Span) {
        let rests = pats.iter().filter(|p| matches!(p, Pattern::Rest(_))).count();
        if rests > 1 {
            self.error(
                format!("tuple pattern has {} `..` rest elements; at most one is allowed", rests),
                span.clone(),
            );
            return;
        }
        let (before, after, has_rest) = crate::ast::tuple_rest_split(pats);
        if has_rest {
            let fixed = before.len() + after.len();
            if width < fixed {
                self.error(
                    format!("tuple pattern needs at least {} element(s) but value has {} (type `{}`)",
                            fixed, width, ty),
                    span.clone(),
                );
            }
        } else if pats.len() != width {
            self.error(
                format!("tuple pattern has {} elements but value has {} (type `{}`)",
                        pats.len(), width, ty),
                span.clone(),
            );
        }
    }

    fn check_block(&mut self, block: &Block) -> TyType {
        self.env.push_scope();

        // If there's no explicit tail expr but the last stmt is a yielding
        // form (directive block, bare expr, stage, or if/match — the latter
        // two are expressions parsed as stmts when at the head of a block),
        // defer it so we can extract its type with body's bindings in scope.
        let n = block.stmts.len();
        let deferred_last = if block.tail_expr.is_none() && n > 0 {
            matches!(&block.stmts[n - 1],
                Stmt::DirectiveBlock { .. } | Stmt::Expr { assign: None, .. }
                | Stmt::Stage { .. } | Stmt::If(_) | Stmt::Match(_))
        } else { false };

        let stop = if deferred_last { n - 1 } else { n };
        for stmt in &block.stmts[..stop] {
            self.check_stmt(stmt);
        }

        let ty = if let Some(tail) = &block.tail_expr {
            self.check_expr(tail)
        } else if deferred_last {
            match &block.stmts[n - 1] {
                Stmt::DirectiveBlock { body, .. } => {
                    // Walk the body's stmts in the outer scope so lets are visible.
                    for s in &body.stmts {
                        self.check_stmt(s);
                    }
                    if let Some(t) = &body.tail_expr {
                        self.check_expr(t)
                    } else { TyType::Unit }
                }
                Stmt::Expr { lhs, .. } => self.check_expr(lhs),
                Stmt::Stage { body, .. } => self.check_stage_expr(body),
                Stmt::If(ie) => self.check_if(ie, true),   // block's trailing value
                Stmt::Match(me) => self.check_match(me),
                _ => unreachable!(),
            }
        } else if n > 0 && matches!(&block.stmts[n - 1],
            Stmt::Return { .. } | Stmt::Break(_) | Stmt::Continue(_)) {
            // A block whose final statement is `return` / `break` / `continue`
            // never falls through to a value — it diverges. Type it as ⊥
            // (Unknown) so an enclosing `if` arm or function body unifies with
            // any expected type.
            TyType::Unknown
        } else {
            TyType::Unit
        };
        self.env.pop_scope();
        ty
    }

    /// True when evaluating `e` has no observable side effect, so discarding its
    /// value is dead code. Deliberately CONSERVATIVE: it covers pure operators
    /// and leaves only. A call, index, field access, range, or any block-like
    /// form (if/match/block/struct-lit/fn-lit/arena/directive) is treated as
    /// potentially-effectful and returns false — so the unused-value lint never
    /// fires on a legitimate effect-for-discard statement (`print`, `map_set`,
    /// `write_file`, a user fn of unknown purity). Soundness (no false positives)
    /// over completeness.
    fn expr_is_effect_free(e: &Expr) -> bool {
        match e {
            Expr::Literal(..) | Expr::Ident(..) | Expr::Underscore(_) | Expr::Nil(_) => true,
            Expr::Tuple(es, _) | Expr::TensorLit(es, _) => es.iter().all(Self::expr_is_effect_free),
            Expr::BinOp { lhs, rhs, .. } => {
                Self::expr_is_effect_free(lhs) && Self::expr_is_effect_free(rhs)
            }
            Expr::UnOp { operand, .. } => Self::expr_is_effect_free(operand),
            Expr::Cast { expr, .. } => Self::expr_is_effect_free(expr),
            // Transpose (`m'`) is a pure read; other postfixes (call / index /
            // field / bracket-args / constructor) may carry effects → not pure.
            Expr::Postfix { expr, op: PostfixOp::Transpose, .. } => Self::expr_is_effect_free(expr),
            _ => false,
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(l) => { self.check_let(l); }
            Stmt::Expr { lhs, assign, span } => {
                // #403 (MEMORY §2): the write target of a plain `=` / `:=` /
                // `<-` is not a read — it is the write that initializes an
                // uninit binding. Suppress the read error for the target
                // during the LHS check, re-mark for the RHS so `t[0] = t[1]`
                // still reports the undefined read, then clear for good.
                // Compound ops (`+=`, …) read the target: no suppression.
                let uninit_write: Option<String> = match assign {
                    Some((AssignOp::Eq | AssignOp::ColonEq | AssignOp::StreamArrow, _)) =>
                        lhs_root_ident(lhs)
                            .filter(|r| self.env.is_uninit(r))
                            .map(str::to_string),
                    _ => None,
                };
                if let Some(r) = &uninit_write { self.env.clear_uninit(r); }
                let lhs_ty = self.check_expr(lhs);
                if let Some((op, rhs)) = assign {
                    if let Some(r) = &uninit_write { self.env.mark_uninit(r.clone()); }
                    let rhs_ty = self.check_expr(rhs);
                    if let Some(r) = &uninit_write { self.env.clear_uninit(r); }
                    // Re-assigning a fresh uninit allocation re-arms the flag.
                    if matches!(op, AssignOp::Eq | AssignOp::ColonEq) && is_uninit_ctor(rhs) {
                        if let Expr::Ident(n, _) = lhs {
                            self.env.mark_uninit(n.clone());
                        }
                    }
                    // #442 (MEMORY §3.1): mutating Vault data outside a
                    // `vault { … }` block is a cross-arena write. Mutations
                    // are element/field writes, compound assigns, and `<-`
                    // appends through a Vault-origin binding. A plain
                    // whole-`=` rebind is not a mutation (the old Vault data
                    // is untouched); it re-tags by the new value below.
                    let is_mutation = match op {
                        AssignOp::Eq => !matches!(lhs, Expr::Ident(..)),
                        AssignOp::ColonEq => false,
                        _ => true, // compound assigns and `<-` modify in place
                    };
                    // #442: the spec's hard error (MEMORY §3.1). Landed first
                    // as a safe-mode warning because the corpus freely mutated
                    // vault-allocated scratch outside `vault {}`; those 264
                    // sites were migrated to `forge.*` (they were per-run
                    // scratch, not long-lived data) before this promotion.
                    // Ungated by demon mode, matching the sibling §2
                    // uninit-read error — this is a spec violation, not a lint.
                    if is_mutation
                        && !matches!(self.arena_stack.last(), Some(ArenaKind::Vault))
                    {
                        if let Some(root) = lhs_root_ident(lhs) {
                            if self.env.is_vault_origin(root) {
                                self.error_with_hint(
                                    format!(
                                        "cross-arena write: `{root}` lives in the Vault; \
                                         mutating it belongs in an explicit `vault {{ … }}` \
                                         block"),
                                    span.clone(),
                                    Some(format!(
                                        "wrap the write: `vault {{ {root}[…] = … }}` — or \
                                         allocate with `forge.*` if this is per-step scratch")),
                                );
                            }
                        }
                    }
                    // Whole-`=`/`:=` rebinds re-tag the binding by the arena
                    // of its new value.
                    if matches!(op, AssignOp::Eq | AssignOp::ColonEq) {
                        if let Expr::Ident(n, _) = lhs {
                            if is_vault_ctor(rhs) {
                                self.env.mark_vault_origin(n.clone());
                            } else if matches!(op, AssignOp::ColonEq) {
                                self.env.unmark_vault_origin_here(n);
                            } else {
                                self.env.clear_vault_origin(n);
                            }
                        }
                    }
                    // Enforce immutability: plain `let x = ...` may not be written through.
                    // `:=` (shadow) and `<-` (stream-append) are exempt.
                    if !matches!(op, AssignOp::ColonEq | AssignOp::StreamArrow) {
                        if let Expr::Ident(name, _) = lhs {
                            if self.env.lookup(name).is_some()
                                && !self.env.is_mutable_binding(name)
                            {
                                self.error(
                                    format!(
                                        "cannot assign to immutable binding `{}`; \
                                         use `let !{}` to allow mutation (or `let mut {}` — both work, `let !` is idiomatic)",
                                        name, name, name
                                    ),
                                    span.clone(),
                                );
                            }
                        } else if let Expr::Postfix { op: PostfixOp::Index(_), .. } = lhs {
                            // #247 / #90: an element write `x[i] = ...` (or `x.f[i] = ...`,
                            // or a write through a non-`!` parameter) goes through the same
                            // immutability rule. Walk the index/field chain to the base
                            // identifier and reuse the immutable-binding check.
                            if let Some(base) = lhs_root_ident(lhs) {
                                // #403 (SPEC §4.8): a KV is append-only — `<-` is the
                                // only legal way to extend it, and element writes are
                                // rejected outright. Without this, `c[i] = v` surfaced
                                // only at runtime, as a misleading out-of-bounds on the
                                // `~` axis (which starts at 0).
                                if matches!(self.env.lookup(base), Some(TyType::KV(..))) {
                                    self.error(
                                        format!(
                                            "cannot element-assign `{base}[…]` — a `KV` is \
                                             append-only; use `{base} <- value` to extend \
                                             the stream (SPEC §4.8)"),
                                        span.clone(),
                                    );
                                } else if self.env.lookup(base).is_some()
                                    && !self.env.is_mutable_binding(base)
                                {
                                    self.error(
                                        format!(
                                            "cannot write to an element of immutable binding `{}`; \
                                             element writes need `let !{}` (or `!{}` on the parameter)",
                                            base, base, base
                                        ),
                                        span.clone(),
                                    );
                                }
                            }
                        }
                    }
                    // Footgun lint (#232): self-assignment `x = x` is dead code —
                    // and a signature of degenerate (repetition-collapsed) codegen.
                    if matches!(op, AssignOp::Eq) {
                        if let (Expr::Ident(ln, _), Expr::Ident(rn, _)) = (lhs, rhs) {
                            if ln == rn {
                                self.warn(
                                    format!("self-assignment `{} = {}` has no effect", ln, rn),
                                    span.clone(),
                                    Some("remove it — dead code (also the signature of degenerate codegen)".to_string()),
                                );
                            }
                        }
                    }
                    if matches!(op, AssignOp::ColonEq) {
                        // `:=` creates a shadow binding in the current scope, not a write-through.
                        // Warn when it shadows an outer `let !` binding — almost always a mistake.
                        if let Expr::Ident(name, _) = lhs {
                            if self.env.outer_scope_has_mutable(name) {
                                self.error(
                                    format!(
                                        "`:=` creates a new shadow binding `{}`, scoped to this block. \
                                         The outer `let !{}` will not be updated. \
                                         Did you mean `=` (assignment)?",
                                        name, name
                                    ),
                                    span.clone(),
                                );
                            }
                        }
                    } else if matches!(op, AssignOp::StreamArrow) {
                        // <- requires a KV/Tensor lhs (SPEC §3.6, §4.9 arena coherence).
                        if !matches!(lhs_ty, TyType::Unknown) && lhs_ty.as_tensor_like().is_none() {
                            self.error(
                                format!("`<-` stream-append requires a KV[~] tensor on the left; got `{}`", lhs_ty),
                                lhs.span_of(),
                            );
                        }
                    } else if !lhs_ty.compatible_with(&rhs_ty) {
                        // Conservative diagnostic — only flag clear type-class mismatches.
                        if matches!(&lhs_ty, TyType::Scalar(_)) != matches!(&rhs_ty, TyType::Scalar(_)) {
                            self.error(
                                format!("assignment type mismatch: {} ← {}", lhs_ty, rhs_ty),
                                lhs.span_of(),
                            );
                        }
                    }
                    // #295: `x = 300` where `x: i8` — a literal RHS must fit the
                    // existing binding's narrow type. Only plain `=` (not `:=`
                    // shadow, `<-` append, or compound ops) overwrites in place.
                    if matches!(op, AssignOp::Eq) {
                        self.check_int_literal_range(&lhs_ty, rhs);
                    }
                } else {
                    // Footgun lint (unused-value / `\>`-trap, PR #412): a bare
                    // expression-statement whose value
                    // is discarded AND whose evaluation has no side effect is dead
                    // code. This is the surface of the `\>` trap — `let m = a \> 0.0`
                    // has no infix parse, so it splits off a discarded `\>(0.0)`
                    // (= `relu(0.0)`), a pure UnOp — invisible today because nothing
                    // warns on it. Fires only when the expression is *definitely*
                    // effect-free (pure operators / leaves), so a bare effectful call
                    // (`print(..)`, `map_set(..)`, `write_file(..)`, or any user fn of
                    // unknown purity) never warns. Safe-mode lint (suppressed in demon
                    // mode). NOTE: this is a non-tail statement — a block's trailing
                    // value expression is handled in `check_block`, never here.
                    if !self.demon
                        && !matches!(lhs_ty, TyType::Unit | TyType::Unknown)
                        && Self::expr_is_effect_free(lhs)
                    {
                        self.warn(
                            format!(
                                "this expression computes a `{}` value that is then discarded — \
                                 the statement has no effect",
                                lhs_ty
                            ),
                            lhs.span_of(),
                            Some(
                                "bind it (`let x = …`), use it, or delete it. If you meant ReLU: \
                                 `\\>` is prefix-only (`\\>(x)`) — there is no infix `x \\> t`"
                                    .to_string(),
                            ),
                        );
                    }
                }
            }
            // A bare `if` statement — its value is discarded (#479).
            Stmt::If(if_expr) => { let _ = self.check_if(if_expr, false); }
            Stmt::Match(me) => { let _ = self.check_match(me); }
            Stmt::For { pattern, iter, body, span } => {
                let iter_ty = self.check_expr(iter);
                // #403 (MEMORY §9.1): `<-` appending to the value being
                // iterated is the mutate-while-iterating bug class. The spec
                // pins this rule to lexical scope deliberately — any `<-` on
                // the loop's iterable binding anywhere in the body errors,
                // regardless of branches — keeping the diagnostic
                // deterministic. Snapshot iteration (`let snap = c` then
                // `for v in snap { c <- … }`) stays legal because the
                // iterable names a different binding. The `<-` itself marks
                // the target as a stream, so no KV type lookup is needed.
                if let Expr::Ident(iter_name, _) = iter {
                    let mut appends = Vec::new();
                    collect_stream_appends_block(body, iter_name, &mut appends);
                    for hit in appends {
                        self.error(
                            format!(
                                "stream-iteration-aliasing: `{iter_name} <- …` inside a \
                                 `for` loop iterating `{iter_name}` — bind a snapshot \
                                 first (`let snap = {iter_name}`) and iterate that"),
                            hit,
                        );
                    }
                }
                // #204: maps are not iterable — `for … in <map>` type-checks but
                // fails at runtime ("cannot iterate over map"). Lint toward the
                // key/value accessors. Safe-mode lint (suppressed in demon mode).
                if matches!(iter_ty, TyType::Map) {
                    self.warn(
                        "`for … in <map>` — maps are not iterable in demoniC; this fails at runtime".to_string(),
                        span.clone(),
                        Some("iterate the keys or values: `for k in map_keys(m)` or `for v in map_vals(m)`".to_string()),
                    );
                }
                self.env.push_scope();
                // If iterating over a Range, the element type is always i64.
                let elem_ty = if matches!(iter, Expr::Range { .. }) {
                    TyType::Scalar(ScalarType::I64)
                } else {
                    TyType::Unknown
                };
                self.bind_pattern(pattern, &elem_ty);
                let _ = self.check_block(body);
                self.env.pop_scope();
            }
            Stmt::While { cond, body, .. } => {
                let _ = self.check_expr(cond);
                let _ = self.check_block(body);
            }
            Stmt::Loop { body, .. } => { let _ = self.check_block(body); }
            Stmt::Stage { body, .. } => { let _ = self.check_stage_expr(body); }
            Stmt::Directive { directives, inner, .. } => {
                self.check_directive_stmt(directives, inner);
            }
            Stmt::DirectiveBlock { directives, body, .. } => {
                self.lint_unimplemented_directives(directives);
                let _ = self.check_block(body);
            }
            Stmt::Break(_) | Stmt::Continue(_) => {}
            Stmt::Return { value, span } => {
                let ret_ty = if let Some(v) = value {
                    self.check_expr(v)
                } else {
                    TyType::Unit
                };
                // Compare against the enclosing function's declared return type.
                let declared = self.current_fn_ret.clone();
                if let Some(decl) = declared {
                    if !decl.compatible_with(&ret_ty) && !matches!(ret_ty, TyType::Unknown) {
                        self.error(
                            format!(
                                "`return` produces `{}` but function is declared to return `{}`",
                                ret_ty, decl
                            ),
                            span.clone(),
                        );
                    }
                    // #295: `return 300` from a `-> i8` function — literal out of range.
                    if let Some(v) = value {
                        self.check_int_literal_range(&decl, v);
                    }
                }
            }
        }
    }

    /// `value_pos` is false for an `if` written as a bare statement, whose
    /// value is discarded (#479). `if c { save() } else { }` calls something
    /// for its effect on one side and does nothing on the other; the branch
    /// "types" are incidental there and unifying them would reject legal code.
    /// Everywhere the value is actually used — a `let`, an argument, a block's
    /// trailing expression — the branches must agree.
    fn check_if(&mut self, if_expr: &IfExpr, value_pos: bool) -> TyType {
        let _ = self.check_expr(&if_expr.cond);
        let then_ty = self.check_block(&if_expr.then_branch);
        match &if_expr.else_branch {
            Some(ElseBranch::Block(b)) => {
                let else_ty = self.check_block(b);
                if !value_pos { return TyType::Unit; }
                self.unify_if_branches(then_ty, else_ty, &if_expr.span)
            }
            // #479: an `else if` chain must unify with the leading branch too.
            // This arm used to `return self.check_if(nested)` and DISCARD
            // `then_ty` outright, so `if a { 1 } else if b { "x" } else { 2 }`
            // reported no error at all — the first branch's type never met the
            // rest of the chain.
            Some(ElseBranch::If(nested)) => {
                let else_ty = self.check_if(nested, value_pos);
                if !value_pos { return TyType::Unit; }
                self.unify_if_branches(then_ty, else_ty, &if_expr.span)
            }
            // No `else`: the `if` yields no value on the false path, so the
            // expression is Unit regardless of what the then-branch produces.
            None => TyType::Unit,
        }
    }

    /// Unify the two branch types of an `if`/`else` (#479).
    ///
    /// `match` has done this since #244; `if` did not, and silently produced
    /// `Unit` on a mismatch. Two things followed. When the result was consumed,
    /// the diagnostic named `nil` and pointed at the CONSUMER, sending the
    /// reader to look for a missing return value several lines from the actual
    /// mistake. When it was not consumed by an annotated binding, nothing was
    /// reported at all: `if a > 0 { "x" } else { a }` type-checked clean with a
    /// `str` branch and an `i64` branch.
    ///
    /// Deliberately the SAME predicate `match` uses (`compatible_with`, with
    /// `Unknown` exempt as the diverging/bottom type), so the two forms cannot
    /// drift apart — `if c { v } else { panic(..) }` still yields `typeof v`.
    fn unify_if_branches(&mut self, then_ty: TyType, else_ty: TyType, span: &Span) -> TyType {
        if matches!(then_ty, TyType::Unknown) { return else_ty; }
        if matches!(else_ty, TyType::Unknown) { return then_ty; }
        if then_ty.compatible_with(&else_ty) {
            // Prefer the concrete side: an untyped literal adopts its context
            // (SPEC §"Untyped numeric literals"), so `if c { 0.0 } else { x_f32 }`
            // is an f32, not a float literal.
            return match (&then_ty, &else_ty) {
                (TyType::IntLit(_) | TyType::FloatLit(_), other)
                    if !matches!(other, TyType::IntLit(_) | TyType::FloatLit(_)) => else_ty,
                _ => then_ty,
            };
        }
        self.error(
            format!(
                "`if` branches yield incompatible types: the `if` branch is `{}`, \
                 the `else` branch is `{}`",
                then_ty, else_ty,
            ),
            span.clone(),
        );
        // Report the then-branch's type rather than Unit so a single mismatch
        // produces ONE diagnostic instead of cascading `nil` errors downstream.
        then_ty
    }

    fn check_match(&mut self, me: &MatchExpr) -> TyType {
        let scrut_ty = self.check_expr(&me.scrutinee);
        // #336: the scrutinee's enum variant set, if it is an enum — drives
        // variant validation and real (closed-set) exhaustiveness.
        let enum_variants: Option<Vec<String>> = match &scrut_ty {
            TyType::Enum(n) => self.env.enums.get(n).cloned(),
            _ => None,
        };
        let enum_name: Option<String> = match &scrut_ty {
            TyType::Enum(n) => Some(n.clone()),
            _ => None,
        };
        let mut result: Option<TyType> = None;
        let mut has_catchall = false;
        let mut covered_bools: std::collections::HashSet<bool> = std::collections::HashSet::new();
        let mut covered_variants: std::collections::HashSet<String> = std::collections::HashSet::new();
        for arm in &me.arms {
            // #393: shape patterns in `match` are parsed but the interpreter does
            // not match them — left silent they take the wrong arm. Reject loudly
            // until shape-pattern matching lands. (`x @ pat` binds are implemented.)
            if let Pattern::Shape(_, sp) = &arm.pattern {
                self.error(
                    "shape patterns in `match` are not yet implemented — they are \
                     parsed but never match at runtime; guard on `t.shape` instead",
                    sp.clone(),
                );
            }
            // #269: a bare-identifier arm with no guard BINDS (catch-all) — it
            // does NOT compare against a like-named constant. `match k { TOK_EQ
            // => ... }` matches everything and binds a fresh `TOK_EQ`; the author
            // almost certainly meant to compare. Fire only when the name already
            // resolves to an in-scope value (high-confidence: it's shadowing
            // something real), so genuine fresh catch-all binds stay quiet.
            if let Pattern::Ident(name, pspan) = &arm.pattern {
                if arm.guard.is_none() && self.env.lookup(name).is_some() {
                    self.warn(
                        format!("match arm `{}` binds the scrutinee (catch-all); it does not compare \
                                 against the constant `{}`", name, name),
                        pspan.clone(),
                        Some(format!("for a value comparison use a literal arm or a guard: \
                                      `x if x == {} => ...`", name)),
                    );
                }
                // #350: a bare-ident catch-all on an *enum* scrutinee whose name
                // is a variant of a DIFFERENT enum is almost always a mistake —
                // the author meant a variant pattern, but because it isn't a
                // variant of THIS enum it silently binds everything (catch-all),
                // shadowing later arms. Fire only when the name resolves to a
                // real variant elsewhere (high-confidence), so genuine fresh
                // catch-all binds stay quiet. (#269 above handles value shadows.)
                if arm.guard.is_none()
                    && name != "_"
                    && self.env.lookup(name).is_none()
                    && enum_variants.as_ref().is_some_and(|vs| !vs.contains(name))
                {
                    let mut owners: Vec<&String> = self.env.enums.iter()
                        .filter(|(_, vs)| vs.contains(name))
                        .map(|(en, _)| en)
                        .collect();
                    owners.sort();
                    if let Some(owner) = owners.first() {
                        let this = enum_name.clone().unwrap_or_default();
                        self.warn(
                            format!("match arm `{}` binds the scrutinee (catch-all), but `{}` is a \
                                     variant of enum `{}`, not `{}`", name, name, owner, this),
                            pspan.clone(),
                            Some(format!("did you mean a variant of `{}`? qualify it as `{}.<Variant>`, \
                                          or use `_` for a real catch-all", this, this)),
                        );
                    }
                }
            }
            // #336/#350: validate enum-variant patterns — qualified (`Shape.Circle`)
            // or the bare payload form (`Circle(r)`, empty `enum_name`, resolved
            // against the scrutinee) — and check payload-binding arity.
            if let Pattern::EnumVariant { enum_name: pen, variant, bindings, span } = &arm.pattern {
                let resolved = match &enum_name {
                    // Empty `pen` is the bare payload form — resolve against the
                    // scrutinee, like a bare ident.
                    Some(en) if pen.is_empty() || en == pen => {
                        if enum_variants.as_ref().is_some_and(|vs| vs.contains(variant)) {
                            true
                        } else {
                            self.error(format!("enum `{}` has no variant `{}`", en, variant), span.clone());
                            false
                        }
                    }
                    Some(en) => {
                        self.error(format!("pattern is `{}.{}` but the scrutinee is `{}`", pen, variant, en), span.clone());
                        false
                    }
                    None => {
                        let shown = if pen.is_empty() { variant.clone() } else { format!("{}.{}", pen, variant) };
                        self.error(format!("enum-variant pattern `{}` on a non-enum scrutinee `{}`", shown, scrut_ty), span.clone());
                        false
                    }
                };
                // #350 Part 2: payload-binding arity must match the variant's
                // declared field count (0 for a tag-only variant).
                if resolved {
                    if let Some(en) = &enum_name {
                        let field_count = self.env.enum_payloads.get(en)
                            .and_then(|m| m.get(variant))
                            .map_or(0, |f| f.len());
                        if bindings.len() != field_count {
                            self.error(format!(
                                "variant `{}.{}` carries {} field{}, but this pattern binds {}",
                                en, variant, field_count,
                                if field_count == 1 { "" } else { "s" }, bindings.len()),
                                span.clone());
                        }
                    }
                }
            }
            self.env.push_scope();
            // #350 Part 2: a payload variant binds its sub-patterns to the
            // variant's field types; everything else binds via the generic path.
            if let Pattern::EnumVariant { variant, bindings, .. } = &arm.pattern {
                if let Some(en) = enum_name.as_ref().filter(|_| !bindings.is_empty()) {
                    let raw: Vec<Type> = self.env.enum_payloads.get(en)
                        .and_then(|m| m.get(variant)).cloned().unwrap_or_default();
                    for (i, b) in bindings.iter().enumerate() {
                        let fty = raw.get(i).map_or(TyType::Unknown, |t| self.resolve_type(t));
                        self.bind_pattern(b, &fty);
                    }
                }
            } else {
                let bind_ty = if enum_name.is_some() { scrut_ty.clone() } else { TyType::Unknown };
                self.bind_pattern(&arm.pattern, &bind_ty);
            }
            if let Some(guard) = &arm.guard {
                let _ = self.check_expr(guard);
            }
            let arm_ty = self.check_expr(&arm.body);
            self.env.pop_scope();
            // Track coverage for exhaustiveness.
            match &arm.pattern {
                // A bare ident is a *variant* match when it names one (enum
                // scrutinee), else a catch-all binding.
                Pattern::Ident(name, _)
                    if enum_variants.as_ref().is_some_and(|vs| vs.contains(name)) => {
                    covered_variants.insert(name.clone());
                }
                Pattern::Wildcard(_) | Pattern::Ident(_, _) | Pattern::Rest(_) => { has_catchall = true; }
                Pattern::EnumVariant { variant, .. } => { covered_variants.insert(variant.clone()); }
                Pattern::Literal(Literal::Bool(b), _) => { covered_bools.insert(*b); }
                _ => {}
            }
            result = Some(match result {
                None => arm_ty,
                Some(prev) => {
                    // #244: all arms must yield compatible types.
                    // Unknown (diverging / bottom) is exempt — it unifies with anything.
                    if !matches!(arm_ty, TyType::Unknown)
                        && !matches!(prev, TyType::Unknown)
                        && !prev.compatible_with(&arm_ty)
                    {
                        self.error(
                            format!(
                                "match arms yield incompatible types: first arm is `{}`, this arm is `{}`",
                                prev, arm_ty
                            ),
                            arm.span.clone(),
                        );
                    }
                    if matches!(arm_ty, TyType::Unknown) { prev } else { arm_ty }
                }
            });
        }
        // Exhaustiveness: for bool scrutinee, both arms (or a catchall) are required.
        if matches!(scrut_ty, TyType::Scalar(ScalarType::Bool)) && !has_catchall {
            let missing: Vec<&str> = [true, false].iter()
                .filter(|&&b| !covered_bools.contains(&b))
                .map(|&b| if b { "true" } else { "false" })
                .collect();
            if !missing.is_empty() {
                self.error(
                    format!(
                        "match arm coverage incomplete: missing case{} for `{}`\n  \
                         scrutinee type: bool\n  \
                         hint: add `{} => ...` or a catch-all `_ => ...` arm",
                        if missing.len() == 1 { "" } else { "s" },
                        missing.join("`, `"),
                        missing.join("` or `"),
                    ),
                    me.scrutinee.span_of(),
                );
            }
        }
        // #336/#291.3: real exhaustiveness for an enum scrutinee — a closed
        // discriminant set, so every variant must be covered unless there's a
        // catch-all (`_` or a bare-ident bind).
        if let (Some(en), Some(variants)) = (&enum_name, &enum_variants) {
            if !has_catchall {
                let missing: Vec<String> = variants.iter()
                    .filter(|v| !covered_variants.contains(*v))
                    .map(|v| format!("{}.{}", en, v))
                    .collect();
                if !missing.is_empty() {
                    self.error(
                        format!(
                            "match arm coverage incomplete: missing case{} {}\n  \
                             scrutinee type: {}\n  \
                             hint: add the missing arm(s) or a catch-all `_ => ...`",
                            if missing.len() == 1 { "" } else { "s" },
                            missing.join(", "),
                            en,
                        ),
                        me.scrutinee.span_of(),
                    );
                }
            }
        }
        // #291.3: an *open* scalar scrutinee (`i64`, `str`, `float`, narrow
        // ints — anything but `bool`, whose two cases are handled above, and
        // enums, a closed set handled above) cannot be value-exhaustive, so it
        // must carry a catch-all `_` (or a bare-ident bind). Without one a
        // no-arm-matches case is a runtime panic; this makes it a compile error.
        if let TyType::Scalar(s) = &scrut_ty {
            if !matches!(s, ScalarType::Bool) && !has_catchall {
                self.error(
                    format!(
                        "match on `{}` is not exhaustive\n  \
                         scrutinee type: {}\n  \
                         hint: add a catch-all `_ => ...` arm (an open scalar can't be fully covered)",
                        scrut_ty, scrut_ty,
                    ),
                    me.scrutinee.span_of(),
                );
            }
        }
        result.unwrap_or(TyType::Unit)
    }

    // ── Expression typing ─────────────────────────────────────────────────

    fn check_expr(&mut self, expr: &Expr) -> TyType {
        use Expr::*;
        match expr {
            Literal(lit, sp) => {
                // #445 + #295: a suffix-typed int literal must fit its own
                // declared width (`300u8` is an error with or without an
                // annotation).
                // 64-bit suffixes are exempt: the lexer stores hex/binary
                // masks as i64 BIT PATTERNS (#282), so `0xffff…ffffu64` reads
                // as -1 here yet is a legitimate u64 value. Any 64-bit
                // pattern fits a 64-bit type by construction.
                if let crate::ast::Literal::Int(n, Some(st)) = lit {
                    if !matches!(st, crate::ast::ScalarType::I64 | crate::ast::ScalarType::U64) {
                    if let Some((lo, hi)) = int_scalar_range(st.clone()) {
                        let v = *n as i128;
                        if v < lo || v > hi {
                            self.error(format!(
                                "integer literal {} out of range for its `{}` suffix (valid range {}..={})",
                                n, TyType::Scalar(st.clone()), lo, hi), sp.clone());
                        }
                    }
                    }
                }
                self.lit_type(lit)
            }
            Nil(_) => TyType::Unit,
            Underscore(_) | Spread(_) => TyType::Unknown,
            Ident(name, span) => {
                // Locals first, then top-level fns, then arena keywords as opaque.
                if let Some(t) = self.env.lookup(name) {
                    let t = t.clone();
                    // #403 (MEMORY §2): definite assignment for uninit
                    // allocations — reading a `forge.uninit` binding before
                    // any write lands returns undefined bytes. Report at the
                    // first read, then clear: one bug, one report. Write
                    // targets and fill-style call arguments never reach here
                    // marked (suppressed/cleared at their sites).
                    if self.env.is_uninit(name) {
                        self.env.clear_uninit(name);
                        self.error(
                            format!(
                                "read of uninitialized memory: `{name}` comes from \
                                 `forge.uninit`/`vault.uninit` and nothing has been \
                                 written to it yet"),
                            span.clone(),
                        );
                    }
                    return t;
                }
                if let Some(sig) = self.env.functions.get(name) {
                    return TyType::Fn {
                        params: sig.params.iter().map(|(_, t)| t.clone()).collect(),
                        ret: Box::new(sig.ret.clone()),
                    };
                }
                if matches!(name.as_str(), "vault" | "forge" | "stream" | "~" | "self") {
                    return TyType::Unknown;
                }
                // Well-known math constants
                if matches!(name.as_str(), "pi" | "tau" | "e" | "inf" | "nan") {
                    return TyType::Scalar(ScalarType::F64);
                }
                // Activation operators used as values in pipe contexts (e.g. `\|> \>`).
                if matches!(name.as_str(), "\\>" | "\\<") {
                    return TyType::Unknown;
                }
                // Scalar type names appearing in value position (e.g. `D as f32`,
                // `vault.zeros[f32, [...]]`). Resolve to a type-marker, conservatively Unknown.
                if matches!(name.as_str(),
                    "i8" | "i16" | "i32" | "i64" |
                    "u8" | "u16" | "u32" | "u64" |
                    "int4" | "int8" |
                    "f16" | "bf16" | "tf32" | "f32" | "f64" |
                    "fp8_e4m3" | "fp8_e5m2" | "trit" |
                    "bool" | "str" | "nil"
                ) {
                    return TyType::Unknown;
                }
                // Built-in type constructors used in value position
                // (e.g. `Rng.seed(...)`, `Mesh[dp=8, tp=4]`). These are
                // type-level names; pre-alpha treats them as opaque.
                if matches!(name.as_str(),
                    "Tensor" | "View" | "KV" | "Mesh" | "Rng" | "Weights"
                ) {
                    return TyType::Unknown;
                }
                // Model names declared in the program are usable as values
                // (for `.load(...)` / `.new(...)` constructor patterns).
                if self.env.models.contains_key(name) {
                    return TyType::Unknown;
                }
                // Common mesh axis names — referenced as expressions inside
                // sharding/collective directives (`axis=tp`, `axis=dp`, etc.).
                // Treat as opaque values for pre-alpha.
                if matches!(name.as_str(), "dp" | "tp" | "pp" | "ep" | "sp") {
                    return TyType::Unknown;
                }
                // Shape params in scope are SymDim variables, not first-class values;
                // they have no runtime type. Accept silently as Unknown to avoid noise
                // in expression positions like `B as f32`.
                if self.env.shape_param_in_scope(name) {
                    return TyType::Unknown;
                }
                self.error(format!("undefined identifier `{}`", name), span.clone());
                TyType::Unknown
            }
            Tuple(elems, _) => {
                if elems.len() == 1 { return self.check_expr(&elems[0]); }
                TyType::Tuple(elems.iter().map(|e| self.check_expr(e)).collect())
            }
            TensorLit(elems, lit_span) => {
                // Infer shape: scalar elements → 1D [N]; tensor elements → prepend N.
                let n = elems.len() as i64;
                let elem_ty = if let Some(first) = elems.first() {
                    self.check_expr(first)
                } else {
                    TyType::Unknown
                };
                for e in elems.iter().skip(1) { let _ = self.check_expr(e); }
                let ty = match elem_ty {
                    TyType::Tensor(inner_elem, inner_shape) => {
                        // 2D+ literal: outer dim N, then inner shape dims.
                        let mut dims = vec![SymDim::Const(n)];
                        dims.extend(inner_shape.dims.clone());
                        TyType::Tensor(inner_elem, Shape::new(dims))
                    }
                    TyType::Scalar(s) => {
                        TyType::Tensor(Box::new(TyType::Scalar(s)), Shape::new(vec![SymDim::Const(n)]))
                    }
                    _ => TyType::Tensor(Box::new(TyType::Unknown), Shape::new(vec![SymDim::Const(n)])),
                };
                // #403 (SPEC §4.2): tensor literals are for small, human-legible
                // constants; the spec warns past 256 total elements and reserves
                // the right to make it an error. Leaf count comes from the full
                // inferred shape, so nested literals count all their scalars.
                if let TyType::Tensor(_, shape) = &ty {
                    let total = shape.dims.iter().try_fold(1i64, |acc, d| match d {
                        SymDim::Const(k) => Some(acc.saturating_mul(*k)),
                        _ => None,
                    });
                    if let Some(total) = total {
                        if total > 256 {
                            self.warn(
                                format!("tensor literal has {total} elements — literals are for \
                                         small, human-legible constants (past 256 elements)"),
                                lit_span.clone(),
                                Some("for bulk data use `forge.zeros/ones/uninit` and fill, \
                                      or `vault.load` for constants".to_string()),
                            );
                        }
                    }
                }
                ty
            }
            Block(b) => self.check_block(b),
            If(ie) => self.check_if(ie, true),
            Match(me) => self.check_match(me),
            FnLit(fl) => {
                self.env.push_scope();
                for p in &fl.params {
                    let ty = p.ty.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unknown);
                    self.env.bind(&p.name, ty);
                }
                let _ = self.check_block(&fl.body);
                self.env.pop_scope();
                let params = fl.params.iter().map(|p|
                    p.ty.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unknown)
                ).collect();
                let ret = Box::new(fl.ret_type.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unit));
                TyType::Fn { params, ret }
            }
            ArenaBlock(ab) => {
                // #442 (MEMORY §3): arena context is lexical — track the
                // innermost block for the cross-arena write check.
                self.arena_stack.push(ab.kind.clone());
                let t = self.check_block(&ab.body);
                self.arena_stack.pop();
                t
            }
            DirectiveBlock { directives, body, .. } => {
                self.lint_unimplemented_directives(directives);
                self.check_block(body)
            }
            StructLit { name, type_args, fields, span } => {
                let mut field_tys = Vec::new();
                for (fname, val) in fields {
                    field_tys.push((fname, self.check_expr(val)));
                }
                if let Some(info) = self.env.models.get(name).cloned() {
                    // Build shape-param substitution from constructor type args, e.g. M[3] binds N=3.
                    let shape_bindings: HashMap<String, SymDim> = info.shape_params.iter()
                        .zip(type_args.iter())
                        .map(|(param, arg_expr)| (param.clone(), SymDim::from_expr(arg_expr).simplify()))
                        .collect();
                    for (fname, fty) in &field_tys {
                        if let Some(declared_ty) = info.fields.get(*fname) {
                            let expected_ty = if shape_bindings.is_empty() {
                                declared_ty.clone()
                            } else {
                                substitute_shape_args(declared_ty.clone(), &shape_bindings)
                            };
                            if !expected_ty.compatible_with(fty) {
                                self.error(
                                    format!("mismatched type for field `{}` of model `{}`: expected `{}`, got `{}`", fname, name, expected_ty, fty),
                                    span.clone(),
                                );
                            }
                        } else {
                            self.error(
                                format!("model `{}` has no field named `{}`", name, fname),
                                span.clone(),
                            );
                        }
                    }
                    // #246: check that every declared field is present in the constructor.
                    let provided_names: std::collections::HashSet<&str> =
                        field_tys.iter().map(|(n, _)| n.as_str()).collect();
                    let mut missing: Vec<&str> = info.fields.keys()
                        .filter(|k| !provided_names.contains(k.as_str()))
                        .map(|k| k.as_str())
                        .collect();
                    if !missing.is_empty() {
                        missing.sort();
                        self.error(
                            format!(
                                "model `{}` constructor is missing required field{}: `{}`",
                                name,
                                if missing.len() == 1 { "" } else { "s" },
                                missing.join("`, `"),
                            ),
                            span.clone(),
                        );
                    }
                    TyType::Named { name: name.clone(), args: Vec::new() }
                } else {
                    self.error(format!("unknown model `{}`", name), span.clone());
                    TyType::Unknown
                }
            }
            BinOp { op, lhs, rhs, span } => self.check_binop(op.clone(), lhs, rhs, span.clone()),
            UnOp { operand, .. } => self.check_expr(operand),
            Postfix { expr, op, span } => self.check_postfix(expr, op, span.clone()),
            Cast { ty, .. } => self.resolve_type(ty),
            Range { .. } => TyType::Unknown,  // ranges have no concrete type yet
        }
    }

    fn check_stage_expr(&mut self, expr: &Expr) -> TyType {
        if let Expr::Postfix { expr: callee, op: PostfixOp::Call(args), .. } = expr {
            if let Expr::Ident(name, _) = callee.as_ref() {
                if is_pipeline_block_placeholder(name) {
                    for arg in args {
                        match arg {
                            CallArg::Positional(e) => { let _ = self.check_expr(e); }
                            CallArg::Named { value, .. } => { let _ = self.check_expr(value); }
                            CallArg::Spread(_) => {}
                        }
                    }
                    return TyType::Unknown;
                }
            }
        }
        self.check_expr(expr)
    }

    fn check_binop(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr, span: Span) -> TyType {
        let lt = self.check_expr(lhs);
        let rt = self.check_expr(rhs);
        use BinOp::*;
        // #248: a tensor-like operand shape, falling back to the *syntactic*
        // arena-constructor shape when the operand's reported type is `Unknown`
        // (constructors deliberately report `Unknown`). This lets the static
        // shape checks below fire on `forge.zeros[f32,[2,3]] @ forge.zeros[f32,[4,5]]`
        // — the flagship guarantee from SPEC §178/§758 — which the JIT already
        // statically rejects at lowering, without changing constructor-reported
        // types or loosening `Shape::same`.
        let l_shape: Option<(TyType, Shape)> = lt.as_tensor_like()
            .map(|(t, s)| (t.clone(), s.clone()))
            .or_else(|| ctor_tensor_ty(lhs).and_then(|t| t.as_tensor_like().map(|(e, s)| (e.clone(), s.clone()))))
            .or_else(|| match lhs {
                // #283: a binding whose RHS was a constructor (via the
                // side-table), mirroring the indexing path — a `let` must
                // not launder a static shape error into a runtime one.
                Expr::Ident(name, _) => self.env.lookup_ctor_shape(name)
                    .map(|s| (TyType::Unknown, s.clone())),
                _ => None,
            });
        let r_shape: Option<(TyType, Shape)> = rt.as_tensor_like()
            .map(|(t, s)| (t.clone(), s.clone()))
            .or_else(|| ctor_tensor_ty(rhs).and_then(|t| t.as_tensor_like().map(|(e, s)| (e.clone(), s.clone()))))
            .or_else(|| match rhs {
                Expr::Ident(name, _) => self.env.lookup_ctor_shape(name)
                    .map(|s| (TyType::Unknown, s.clone())),
                _ => None,
            });
        match op {
            Matmul => {
                match (&l_shape, &r_shape) {
                    (Some((lt_elem, ls)), Some((_, rs))) => {
                        match ls.matmul(rs) {
                            Ok(out) => TyType::Tensor(Box::new(lt_elem.clone()), out),
                            Err(ShapeError { msg }) => {
                                self.error(format!("matmul: {}", msg), span);
                                TyType::Unknown
                            }
                        }
                    }
                    _ if matches!(lt, TyType::Unknown) || matches!(rt, TyType::Unknown) => TyType::Unknown,
                    _ => {
                        self.error(format!("matmul `@` requires tensors; got {} and {}", lt, rt), span);
                        TyType::Unknown
                    }
                }
            }
            DotAdd | DotSub | DotMul | DotDiv | DotPow | DotPow2
            | DotGt | DotLt | DotGe | DotLe => {
                // Elementwise: broadcast-merge the shapes; element type from lhs
                // (comparisons keep the lhs element type — at this layer we don't
                // distinguish bool tensors; the spec uses 0/1 masks).
                match (&l_shape, &r_shape) {
                    (Some((lt_elem, ls)), Some((_, rs))) => {
                        match ls.broadcast(rs) {
                            Ok(out) => TyType::Tensor(Box::new(lt_elem.clone()), out),
                            Err(ShapeError { msg }) => {
                                self.error(format!("elementwise: {}", msg), span);
                                TyType::Unknown
                            }
                        }
                    }
                    // tensor `op` scalar: result has tensor's shape
                    (Some((lt_elem, ls)), None) if rt.is_numeric() => {
                        TyType::Tensor(Box::new(lt_elem.clone()), ls.clone())
                    }
                    (None, Some((rt_elem, rs))) if lt.is_numeric() => {
                        TyType::Tensor(Box::new(rt_elem.clone()), rs.clone())
                    }
                    (None, None) if lt.is_numeric() && rt.is_numeric() => {
                        lt.clone()
                    }
                    _ if matches!(lt, TyType::Unknown) || matches!(rt, TyType::Unknown) => TyType::Unknown,
                    _ => {
                        self.error(format!("elementwise `{:?}` requires tensors; got {} and {}", op, lt, rt), span);
                        TyType::Unknown
                    }
                }
            }
            Add | Sub | Mul | Div | Mod | Pow | StarStar => {
                // Footgun lint (#198): demoniC `%` is truncated (sign follows the
                // dividend), so `(0 - 7) % 3 == -1`. Python/Julia `%` is floored
                // (`-7 % 3 == 2`), so a literal port silently diverges on negative
                // dividends. High-confidence trigger only: flag when the dividend
                // is syntactically sign-bearing — a subtraction `(a - b) % n` or a
                // negation `-x % n` — the shapes where the sign actually bites.
                // Stays quiet on the common safe cases (`i % n`, `len(x) % k`).
                if matches!(op, Mod) && expr_may_be_negative(lhs) {
                    self.warn(
                        "`%` is truncated in demoniC (sign follows the dividend), not floored".to_string(),
                        span.clone(),
                        Some("for Python/Julia floored-mod semantics use `((a % b) + b) % b`".to_string()),
                    );
                }
                // Footgun lint (#231): no-effect arithmetic — an identity operand
                // (`+ 0`, `- 0`, `* 1`, `/ 1`) leaves the value unchanged. Never
                // intentional; on a loop-control variable (`i = i + 0`) the program
                // type-checks but loops forever. (`x ^ 0` is caught by the #194
                // `^`-is-XOR lint instead.)
                let no_effect = match op {
                    Add => is_num_literal(rhs, 0.0) || is_num_literal(lhs, 0.0),
                    Sub => is_num_literal(rhs, 0.0),
                    Mul => is_num_literal(rhs, 1.0) || is_num_literal(lhs, 1.0),
                    Div => is_num_literal(rhs, 1.0),
                    _ => false,
                };
                if no_effect {
                    self.warn(
                        "no-effect arithmetic: an identity operand leaves the value unchanged".to_string(),
                        span.clone(),
                        Some("remove it — a no-op step on a loop counter (`i = i + 0`) loops forever".to_string()),
                    );
                }
                // String concatenation: str + anything or anything + str via Add.
                if matches!(op, Add) {
                    match (&lt, &rt) {
                        (TyType::Scalar(ScalarType::Str), _) | (_, TyType::Scalar(ScalarType::Str)) => {
                            return TyType::Scalar(ScalarType::Str);
                        }
                        _ => {}
                    }
                }
                // Scalar arithmetic (tensor versions use the dotted ops).
                if lt.is_numeric() && rt.is_numeric() {
                    // A literal operand adopts the other (concrete) operand's
                    // type; two literals stay an untyped literal (#295). Otherwise
                    // result = lhs type (pre-alpha: no widening).
                    match (&lt, &rt) {
                        // Two untyped literals of the same kind stay that kind.
                        (TyType::IntLit(_), TyType::IntLit(_)) => lt,
                        (TyType::FloatLit(_), TyType::FloatLit(_)) => lt,
                        // Mixed untyped literals: the float kind wins (the int
                        // literal promotes to float).
                        (TyType::IntLit(_), TyType::FloatLit(_)) => rt,
                        (TyType::FloatLit(_), TyType::IntLit(_)) => lt,
                        // One untyped literal adopts the other (concrete) operand.
                        (TyType::IntLit(_), _) | (TyType::FloatLit(_), _) => rt,
                        // #473: two concrete floats of different widths — the
                        // WIDER one wins. Both backends compute `f32 + f64` in
                        // f64 (the f32 operand promotes; mixed float arithmetic
                        // never silently narrows), so typing it as the lhs made
                        // an expression whose runtime value IS an f64
                        // unbindable to an `f64` — `let w: f64 = a_f32 + b_f64`
                        // was rejected. Same-width pairs and integer arithmetic
                        // keep the lhs rule unchanged.
                        (TyType::Scalar(a), TyType::Scalar(b))
                            if is_f32_family(a) && matches!(b, ScalarType::F64) => rt,
                        (TyType::Scalar(a), TyType::Scalar(b))
                            if matches!(a, ScalarType::F64) && is_f32_family(b) => lt,
                        _ => lt,
                    }
                } else if matches!(lt, TyType::Unknown) || matches!(rt, TyType::Unknown) {
                    TyType::Unknown
                } else if lt.as_tensor_like().is_some() || rt.as_tensor_like().is_some() {
                    // Tensors with `+/-/*//` are a common mistake — should use `.+ .- .* ./`
                    self.error_with_hint(
                        format!("`{:?}` on tensors — did you mean the dotted form (e.g. `.+`)?", op),
                        span, Some("scalar operators don't broadcast over tensors".to_string()),
                    );
                    TyType::Unknown
                } else {
                    self.error(format!("`{:?}` on non-numeric operands: {} and {}", op, lt, rt), span);
                    TyType::Unknown
                }
            }
            And | Or => TyType::Scalar(ScalarType::Bool),
            Eq | NotEq | Lt | Gt | LtEq | GtEq => TyType::Scalar(ScalarType::Bool),
            Pipe | RShift => {
                // `x |> f` / `x >> f` pipe into a callable RHS. The placeholder
                // form (`x |> _ .+ b`) and bare activation stages (`\|> \>`,
                // which parse to an ident) are valid non-callable RHS forms.
                // Catch the footgun where the RHS is a concrete *value* — most
                // often `x >> 2` meant as a bitwise shift (#188): it would
                // type-check and then fail at runtime with "expected callable".
                let rhs_is_value = rt.is_numeric() || rt.as_tensor_like().is_some();
                if rhs_is_value && !expr_contains_underscore(rhs) {
                    self.error_with_hint(
                        format!("`>>`/`|>` pipe into a callable, but the right side is {rt}, not callable"),
                        span,
                        Some("`>>` is the pipe operator, not bitwise shift; for an elementwise stage use a `_` placeholder (e.g. `x |> _ .+ y`)".to_string()),
                    );
                }
                TyType::Unknown
            }
            // Bitwise operators require integer operands.
            BitAnd | BitOr | BitXor | BitShl | BitShr => {
                // Lookalike-operator lint (#194): `^` is XOR, not power. The
                // high-confidence shape is `<int> ^ <int literal>` — nearly
                // always `base ** exp` reached for with the wrong operator.
                if matches!(op, BitXor) && matches!(rhs, Expr::Literal(Literal::Int(..), _)) {
                    self.warn(
                        "`^` is bitwise XOR, not exponentiation".to_string(),
                        span.clone(),
                        Some("for a power use `**` (e.g. `2 ** 8`); `^` also binds looser than `/`".to_string()),
                    );
                }
                if matches!(lt, TyType::Unknown) || matches!(rt, TyType::Unknown) {
                    TyType::Unknown
                } else if lt.is_numeric() && rt.is_numeric() {
                    lt
                } else {
                    self.error(
                        format!("bitwise operator requires integer operands; got {} and {}", lt, rt),
                        span,
                    );
                    TyType::Unknown
                }
            }
            // StreamArrow is a stmt-level assignment op, not a binary expr op;
            // shouldn't reach here, but cover defensively.
        }
    }

    /// #350 Part 2: type-check a payload-variant constructor call
    /// `Enum.Variant(args)` against the variant's declared field types and
    /// yield the enum type. A tag-only variant has zero fields.
    fn check_enum_construction(&mut self, en: String, variant: String, args: &[CallArg], span: Span) -> TyType {
        let raw: Vec<Type> = self.env.enum_payloads.get(&en)
            .and_then(|m| m.get(&variant)).cloned().unwrap_or_default();
        let field_tys: Vec<TyType> = raw.iter().map(|t| self.resolve_type(t)).collect();
        let arg_tys: Vec<TyType> = args.iter().map(|a| match a {
            CallArg::Positional(e) => self.check_expr(e),
            CallArg::Named { value, .. } => self.check_expr(value),
            CallArg::Spread(_) => TyType::Unknown,
        }).collect();
        if arg_tys.len() != field_tys.len() {
            self.error(format!(
                "variant `{}.{}` carries {} field{}, but {} argument{} given",
                en, variant, field_tys.len(), if field_tys.len() == 1 { "" } else { "s" },
                arg_tys.len(), if arg_tys.len() == 1 { "" } else { "s" }),
                span);
        } else {
            for (i, (aty, fty)) in arg_tys.iter().zip(field_tys.iter()).enumerate() {
                if !matches!(aty, TyType::Unknown) && !matches!(fty, TyType::Unknown)
                    && !fty.compatible_with(aty)
                {
                    self.error(format!(
                        "variant `{}.{}` field {} expects `{}`, got `{}`",
                        en, variant, i, fty, aty),
                        span.clone());
                }
            }
        }
        TyType::Enum(en)
    }

    /// #397: reject `<tensor>.split[...]`. The `.split` Field access alone is
    /// caught in the `PostfixOp::Field` arm, but a trailing bracket (`[n, axis]`)
    /// makes the whole node a `BracketArgs`/`Index` that short-circuits before the
    /// Field is ever type-checked — so those arms must reject it too. Returns true
    /// if it rejected (the caller should then return `Unknown`).
    /// Type a tensor `.split[n, axis=k]` (SPEC §6.4): it yields an `n`-tuple of
    /// tensors. The `.split[...]` *bracket* form is unambiguously tensor-split —
    /// strings/lists use the `.split(...)` *call* form, a different node — so a
    /// `.split` field under a bracket is always this op, even when the receiver's
    /// element type inferred as `Unknown` (e.g. `forge.zeros[...]`). Returns the
    /// tuple type so destructure arity is checked; the piece shapes are not
    /// tracked symbolically here (typed `Unknown`), which the interp computes at
    /// runtime. Returns `None` when `base` is not a `.split`.
    fn tensor_split_type(&mut self, base: &Expr, op: &PostfixOp) -> Option<TyType> {
        let Expr::Postfix { expr: recv, op: PostfixOp::Field(fname), .. } = base else { return None; };
        if fname != "split" { return None; }
        let _ = self.check_expr(recv); // surface the receiver's own errors
        // Piece count `n` = the first literal-int bracket arg.
        let n = match op {
            PostfixOp::BracketArgs(args) => args.iter().find_map(|a| match a {
                CallArg::Positional(e) => lit_usize(e),
                _ => None,
            }),
            PostfixOp::Index(elems) => elems.iter().find_map(|el| match el {
                IndexElem::Expr(e) => lit_usize(e),
                _ => None,
            }),
            _ => None,
        };
        Some(match n {
            Some(k) if k > 0 => TyType::Tuple(vec![TyType::Unknown; k]),
            _ => TyType::Unknown,
        })
    }

    fn check_postfix(&mut self, expr: &Expr, op: &PostfixOp, span: Span) -> TyType {
        // #336: `Color.Red` — a qualified enum-variant value. Intercept before
        // the base is typed as a value (the enum name is not a binding).
        if let (Expr::Ident(base, _), PostfixOp::Field(variant)) = (expr, op) {
            if let Some(variants) = self.env.enums.get(base) {
                if variants.contains(variant) {
                    return TyType::Enum(base.clone());
                }
                self.error(format!("enum `{}` has no variant `{}`", base, variant), span);
                return TyType::Unknown;
            }
        }
        // #350 Part 2: `Shape.Circle(args)` — payload-variant construction.
        // Intercept before the method-call machinery reads it as a `.Circle`
        // method call on the enum value `Shape`.
        if let PostfixOp::Call(args) = op {
            if let Expr::Postfix { expr: base, op: PostfixOp::Field(variant), .. } = expr {
                if let Expr::Ident(en, _) = base.as_ref() {
                    if self.env.enums.get(en).is_some_and(|vs| vs.contains(variant)) {
                        return self.check_enum_construction(en.clone(), variant.clone(), args, span);
                    }
                }
            }
        }
        let recv = self.check_expr(expr);
        match op {
            PostfixOp::Transpose => {
                // Swap last two axes of the receiver's shape.
                if let Some((t, sh)) = recv.as_tensor_like() {
                    if sh.rank() >= 2 {
                        let mut dims = sh.dims.clone();
                        let n = dims.len();
                        dims.swap(n - 1, n - 2);
                        return TyType::Tensor(Box::new(t.clone()), Shape::new(dims));
                    }
                }
                recv
            }
            PostfixOp::Query => {
                // ? is only legal inside a function whose return type is (T, str) or (T, Err).
                // SPEC §4.9: "Legal only inside a function whose return type is also (_, Err)."
                let in_err_fn = match &self.current_fn_ret {
                    Some(TyType::Tuple(tys)) if tys.len() == 2 => {
                        matches!(&tys[1], TyType::Scalar(crate::ast::ScalarType::Str)
                                        | TyType::Unknown
                                        | TyType::Unit)
                        || (if let TyType::Named { name, .. } = &tys[1] { name == "Err" } else { false })
                    }
                    Some(TyType::Unknown) | None => true,  // conservative: don't flag Unknown context
                    _ => false,
                };
                if !in_err_fn {
                    self.error(
                        format!("`?` is only legal inside a function returning `(T, str)` or `(T, Err)`; this function returns `{}`",
                            self.current_fn_ret.as_ref().map(|t| t.to_string())
                                .unwrap_or_else(|| "unknown".into())),
                        span,
                    );
                }
                // ? unwraps the T from (T, str) or (T, Err); fall back to Unknown if recv isn't a known tuple.
                match recv {
                    TyType::Tuple(ref tys) if tys.len() == 2 => tys[0].clone(),
                    _ => TyType::Unknown,
                }
            }
            PostfixOp::Index(elems) => {
                if let Some(ty) = self.tensor_split_type(expr, op) { return ty; }
                // Type-check each index expr; result shape pruned per scalar indices.
                for e in elems {
                    if let IndexElem::Expr(e) = e {
                        let _ = self.check_expr(e);
                    } else if let IndexElem::Slice { start, end, step, .. } = e {
                        if let Some(e) = start.as_deref() { let _ = self.check_expr(e); }
                        if let Some(e) = end.as_deref()   { let _ = self.check_expr(e); }
                        if let Some(e) = step.as_deref()  { let _ = self.check_expr(e); }
                    }
                }
                // Only return element type for full scalar indexing (no slice elems).
                let all_scalar = elems.iter().all(|e| matches!(e, IndexElem::Expr(_)));
                if all_scalar {
                    // #248: fall back to the constructor shape when the receiver
                    // is a literal `forge.zeros[..]` or a binding whose RHS was one
                    // (via the side-table), so static OOB is caught without
                    // enriching (and contaminating) binding types.
                    let recv_shape = recv.as_tensor_like()
                        .map(|(t, s)| (t.clone(), s.clone()))
                        .or_else(|| ctor_tensor_ty(expr).and_then(|t|
                            t.as_tensor_like().map(|(e, s)| (e.clone(), s.clone()))))
                        .or_else(|| match expr {
                            Expr::Ident(name, _) => self.env.lookup_ctor_shape(name)
                                .map(|s| (TyType::Unknown, s.clone())),
                            _ => None,
                        });
                    if let Some((elem_ty, shape)) = recv_shape {
                        // #248: static out-of-bounds on a constant index into a
                        // constant axis is a compile-time error (SPEC §417).
                        // demoniC allows Python-style negatives, so the valid
                        // range is `-dim <= idx < dim`; only flag literals (and
                        // const-foldable expressions) provably outside it.
                        for (i, elem) in elems.iter().enumerate() {
                            if i >= shape.rank() { break; }
                            if let IndexElem::Expr(ie) = elem {
                                if let (SymDim::Const(idx), SymDim::Const(dim)) =
                                    (SymDim::from_expr(ie).simplify(), shape.dims[i].simplify())
                                {
                                    if idx >= dim || idx < -dim {
                                        self.error(
                                            format!("index {} out of bounds for axis {} of size {}", idx, i, dim),
                                            span.clone(),
                                        );
                                    }
                                }
                            }
                        }
                        // When the shape came only from the constructor side-table
                        // the element type is Unknown; keep the original Unknown
                        // result (don't synthesize a `Tensor[?, ..]` that could
                        // leak into compatibility checks).
                        if matches!(elem_ty, TyType::Unknown) {
                            return TyType::Unknown;
                        }
                        if elems.len() >= shape.rank() {
                            return elem_ty.clone();
                        }
                        let remaining = shape.dims[elems.len()..].to_vec();
                        return TyType::Tensor(Box::new(elem_ty.clone()), Shape::new(remaining));
                    }
                }
                TyType::Unknown
            }
            PostfixOp::Call(args) => {
                // Skip arity validation only for truly variadic builtins
                // (print, panic, argmax/argmin, allreduce, etc.).
                // Fixed-arity stdlib functions are checked normally.
                let is_builtin = if let Expr::Ident(name, _) = expr {
                    self.env.is_builtin(name)
                } else { false };

                // PORTS.md §5: no `@grad fn` may call a port. The call is an
                // effect boundary the tape cannot record, so a gradient through
                // it would be silently wrong — reject it at compile time.
                if self.in_grad_fn {
                    if let Expr::Ident(name, _) = expr {
                        if matches!(name.as_str(), "port_open" | "port_call" | "port_close") {
                            self.error(
                                format!("port-forbidden: `{}` is illegal inside a `@grad fn` \
                                         — a port call is an effect boundary the gradient \
                                         cannot cross (PORTS.md §5)", name),
                                span.clone(),
                            );
                        }
                    }
                }

                // Divergent builtins: `panic` / `exit` never return. They have
                // type ⊥ (bottom), modelled here as `Unknown` (compatible with
                // any expected type). This lets `if c { return x } else { panic(..) }`
                // and trailing `panic(..)` type-check against any declared return.
                // We still check the argument expressions for their own errors.
                let is_divergent = matches!(expr, Expr::Ident(name, _) if name == "panic" || name == "exit");
                if is_divergent {
                    for a in args {
                        match a {
                            CallArg::Positional(e) => { self.check_expr(e); }
                            CallArg::Named { value, .. } => { self.check_expr(value); }
                            CallArg::Spread(_) => {}
                        }
                    }
                    return TyType::Unknown;
                }

                // #403 (MEMORY §2): fill-vs-read for uninit bindings passed as
                // call arguments, decided BEFORE the args are type-checked
                // (the identifier check reports any still-marked binding as a
                // read). Bound to a `!` param of a known fn → the call is the
                // fill: clear silently. Bound to a plain param, or passed to a
                // builtin (builtins never fill their arguments) → a read:
                // leave the mark. Unknown callees (methods, fn values, module
                // paths) → clear silently rather than guess.
                for (i, a) in args.iter().enumerate() {
                    // Named args don't map to positional param indexes —
                    // treat them as fills rather than misindex.
                    let (arg_ident, positional) = match a {
                        CallArg::Positional(Expr::Ident(n, _)) => (n, true),
                        CallArg::Named { value: Expr::Ident(n, _), .. } => (n, false),
                        _ => continue,
                    };
                    if !self.env.is_uninit(arg_ident) { continue; }
                    let treat_as_read = positional && match expr {
                        Expr::Ident(fname, _) => match self.fn_mut_params.get(fname) {
                            // Known fn: `!` param fills, plain param reads.
                            Some(muts) => !muts.get(i).copied().unwrap_or(false),
                            // Builtins never fill their arguments — both the
                            // fixed-arity set (builtin_sig) and the variadic
                            // set (is_builtin). Anything else (merged-import
                            // fns lose their `!` flags) → assume fill.
                            None => crate::types::builtin_sig(fname).is_some()
                                || self.env.is_builtin(fname),
                        },
                        _ => false,
                    };
                    if !treat_as_read {
                        self.env.clear_uninit(arg_ident);
                    }
                }
                // Resolve callee.
                let mut arg_tys: Vec<TyType> = args.iter().map(|a| match a {
                    CallArg::Positional(e) => self.check_expr(e),
                    CallArg::Named { value, .. } => self.check_expr(value),
                    CallArg::Spread(_) => TyType::Unknown,
                }).collect();

                let mut is_method = false;
                if let Expr::Postfix { expr: base, op: PostfixOp::Field(method), .. } = expr {
                    // It's a method call! Prepend the receiver type to the argument types.
                    let base_ty = self.check_expr(base);
                    // Validate the @grad calling conventions (#243): when the
                    // receiver is a top-level user fn, `f.<method>(...)` only
                    // means something for the autodiff method set. Anything
                    // else used to fall through to generic field access and
                    // silently produce garbage at runtime (`dmc run`), or a
                    // confusing lowering error (`dmc jit`).
                    if let Expr::Ident(fn_name, _) = base.as_ref() {
                        if matches!(base_ty, TyType::Fn { .. }) {
                            if let Some(&grad_n) = self.grad_fn_counts.get(fn_name) {
                                let is_grad_name = matches!(method.as_str(),
                                    "fwd" | "grad" | "fwd_bwd" | "fwd_bwd_bwd");
                                if grad_n == 0 && is_grad_name {
                                    self.error(format!(
                                        "`{fn_name}.{method}` — `{fn_name}` is not a `@grad fn`; \
                                         mark it `@grad` to use the autodiff methods"),
                                        span.clone());
                                    return TyType::Unknown;
                                }
                                if grad_n >= 1 && !is_grad_name {
                                    let extra = if grad_n >= 2 { ", `.fwd_bwd_bwd`" } else { "" };
                                    self.error(format!(
                                        "unknown @grad method `.{method}` on fn `{fn_name}` — \
                                         valid methods: `.fwd`, `.grad`, `.fwd_bwd`{extra}"),
                                        span.clone());
                                    return TyType::Unknown;
                                }
                                if method == "fwd_bwd_bwd" && grad_n == 1 {
                                    self.error(format!(
                                        "`{fn_name}.fwd_bwd_bwd` (second-order autodiff) requires \
                                         stacked `@grad @grad` on fn `{fn_name}`"),
                                        span.clone());
                                    return TyType::Unknown;
                                }
                                // #392: a VALID @grad method call — synthesize its
                                // result type. The grad methods are not real fn
                                // signatures, so without this the call types as
                                // `Unknown` and the static tuple-arity check
                                // (check.rs, "tuple pattern has N elements") never
                                // fires — so `let (v,g1,g2) = f.fwd_bwd_bwd(w)`
                                // (value is a 2-tuple) silently binds all three to
                                // nil instead of erroring. `.grad` stays `Unknown`
                                // (a single Grads bundle, dynamic field access);
                                // `.fwd_bwd`/`.fwd_bwd_bwd` are `(loss, Grads)`.
                                if is_grad_name {
                                    let loss_ty = match &base_ty {
                                        TyType::Fn { ret, .. } => (**ret).clone(),
                                        _ => TyType::Unknown,
                                    };
                                    return match method.as_str() {
                                        "fwd" => loss_ty,
                                        "grad" => TyType::Unknown,
                                        _ => TyType::Tuple(vec![loss_ty, TyType::Unknown]),
                                    };
                                }
                            }
                        }
                    }
                    // #441: a call to a method not defined on the receiver's
                    // model is a check-time error. Without this, the field
                    // access falls through to `Unknown` and the call becomes a
                    // silent no-op at runtime (statement position) or an
                    // opaque value that errors only at its point of use (value
                    // position). Fields are allowed through so a fn-typed
                    // field stays callable; unknown `Named` types (not in
                    // `env.models`) stay quiet. All model methods are hoisted
                    // in pass 1, so forward references within and across
                    // models remain legal.
                    if let TyType::Named { name: model_name, .. } = &base_ty {
                        if let Some(info) = self.env.models.get(model_name) {
                            if !info.methods.contains_key(method)
                                && !info.fields.contains_key(method)
                            {
                                self.error(
                                    format!("no method `{method}` on model `{model_name}`"),
                                    span.clone(),
                                );
                                return TyType::Unknown;
                            }
                        }
                    }
                    if !matches!(base_ty, TyType::Module { .. }) {
                        // Footgun lint (#199 + #202): demoniC has no `.method()`
                        // call syntax. On a receiver whose type definitionally
                        // carries no methods, `recv.m(args)` type-checks but
                        // resolves to an *opaque* value at runtime — the failure
                        // this lint exists for, since Rust and Python habits make
                        // `recv.m(args)` the first thing a newcomer writes.
                        // `ty_has_no_methods` deliberately excludes models (`Named`)
                        // and `Unknown`, so genuine model methods and ambiguous
                        // receivers never false-positive.
                        if ty_has_no_methods(&base_ty) {
                            let is_str = matches!(base_ty, TyType::Scalar(ScalarType::Str));
                            let argll = if args.is_empty() { String::new() } else { ", …".to_string() };
                            if is_str && is_supported_str_method(method) {
                                // A real string method (split/replace/upper/len/…),
                                // implemented in interp::call_str_method. It works,
                                // so stay quiet — even when the name is also a global
                                // builtin (`len`), which used to false-positive.
                            } else if self.env.functions.contains_key(method) {
                                // #199: a builtin function written as a method
                                // (`x.floor()`). The author meant `floor(x)`.
                                self.warn(
                                    format!("`.{method}()` method-call syntax — demoniC has no methods; `{method}` is a builtin function"),
                                    span.clone(),
                                    Some(format!("call it as a function with the receiver first: `{method}(<receiver>{argll})`")),
                                );
                            } else if is_str {
                                // #202: an unsupported string method (`.to_lowercase()`,
                                // `.chars()`, …) — falls through to an opaque value.
                                self.warn(
                                    format!("`.{method}()` is not a demoniC string method — it resolves to an opaque value at runtime, not a real call"),
                                    span.clone(),
                                    Some("supported string methods: split, lines, trim/strip, upper, lower, replace, contains, starts_with, ends_with, find, index, count, len".to_string()),
                                );
                            } else {
                                // #202: a method call on a method-less, non-string
                                // type (scalar / tensor / tuple / …). No method
                                // dispatch exists, so it resolves to an opaque value.
                                self.warn(
                                    format!("`.{method}()` method-call syntax on `{base_ty}` — demoniC has no methods on this type; it resolves to an opaque value at runtime"),
                                    span.clone(),
                                    Some("demoniC has no method calls — use a function: `f(x)`, not `x.f()`".to_string()),
                                );
                            }
                        }
                        arg_tys.insert(0, base_ty);
                        is_method = true;
                    }
                }

                if let TyType::Fn { params, ret } = &recv {
                    let mut shape_bindings = HashMap::new();
                    for (pty, aty) in params.iter().zip(arg_tys.iter()) {
                        infer_call_shape_bindings(pty, aty, &mut shape_bindings);
                    }
                    if !is_builtin
                        && !args.iter().any(|a| matches!(a, CallArg::Spread(_)))
                        && !args.iter().any(|a| matches!(a, CallArg::Named { .. }))
                    {
                        let expected_len = params.len();
                        let actual_len = if is_method { args.len() + 1 } else { args.len() };
                        if actual_len != expected_len {
                            self.error(
                                format!("wrong number of args: expected {}, got {}", expected_len, actual_len),
                                span,
                            );
                            return TyType::Unknown;
                        }
                        for (i, (pty, aty)) in params.iter().zip(arg_tys.iter()).enumerate() {
                            let inferred_pty = substitute_shape_args(pty.clone(), &shape_bindings);
                            if !inferred_pty.compatible_with(aty) {
                                let report_idx = if is_method { i as isize - 1 } else { i as isize };
                                if report_idx >= 0 {
                                    self.error(
                                        format!("arg {}: expected {}, got {}", report_idx, inferred_pty, aty),
                                        span.clone(),
                                    );
                                } else {
                                    self.error(
                                        format!("receiver: expected {}, got {}", inferred_pty, aty),
                                        span.clone(),
                                    );
                                }
                            }
                            // #295: `f(300)` where the param is `i8` — literal arg
                            // out of range. `arg_tys` is the receiver-prepended list
                            // for a method, so map back to the positional `args`
                            // index (the i==0 receiver has no arg expr → skip).
                            let arg_idx = if is_method { i.wrapping_sub(1) } else { i };
                            if let Some(CallArg::Positional(e)) = args.get(arg_idx) {
                                self.check_int_literal_range(&inferred_pty, e);
                            }
                        }
                    }
                    return substitute_shape_args((**ret).clone(), &shape_bindings);
                }
                // Calling something that's not a function — defer to runtime checks.
                TyType::Unknown
            }
            PostfixOp::BracketArgs(_) => {
                if let Some(ty) = self.tensor_split_type(expr, op) { return ty; }
                // Generic instantiation or shape-literal call; conservatively Unknown.
                TyType::Unknown
            }
            PostfixOp::Constructor(fields) => {
                let mut model_name_opt = None;
                match expr {
                    Expr::Ident(name, _) => {
                        model_name_opt = Some(name.clone());
                    }
                    Expr::Postfix { expr: inner, op: PostfixOp::BracketArgs(_), .. } |
                    Expr::Postfix { expr: inner, op: PostfixOp::Index(_), .. } => {
                        if let Expr::Ident(name, _) = &**inner {
                            model_name_opt = Some(name.clone());
                        }
                    }
                    _ => {}
                }

                // Check all the fields in the constructor
                let mut field_tys = Vec::new();
                for (name, val) in fields {
                    field_tys.push((name, self.check_expr(val)));
                }

                if let Some(ref model_name) = model_name_opt {
                    if let Some(info) = self.env.models.get(model_name).cloned() {
                        for (fname, fty) in &field_tys {
                            if let Some(expected_ty) = info.fields.get(*fname) {
                                if !expected_ty.compatible_with(fty) {
                                    self.error(
                                        format!("mismatched type for field `{}` of model `{}`: expected `{}`, got `{}`", fname, model_name, expected_ty, fty),
                                        span.clone(),
                                    );
                                }
                            } else {
                                self.error(
                                    format!("model `{}` has no field named `{}`", model_name, fname),
                                    span.clone(),
                                );
                            }
                        }
                        // #246: check that every declared field is present in the constructor.
                        let provided_names: std::collections::HashSet<&str> =
                            field_tys.iter().map(|(n, _)| n.as_str()).collect();
                        let mut missing: Vec<&str> = info.fields.keys()
                            .filter(|k| !provided_names.contains(k.as_str()))
                            .map(|k| k.as_str())
                            .collect();
                        if !missing.is_empty() {
                            missing.sort();
                            self.error(
                                format!(
                                    "model `{}` constructor is missing required field{}: `{}`",
                                    model_name,
                                    if missing.len() == 1 { "" } else { "s" },
                                    missing.join("`, `"),
                                ),
                                span.clone(),
                            );
                        }
                        return TyType::Named { name: model_name.clone(), args: Vec::new() };
                    }
                }
                recv
            }
            PostfixOp::Field(name) => {
                if let TyType::Module { alias, .. } = &recv {
                    let qual_name = format!("{}.{}", alias, name);
                    if let Some(sig) = self.env.functions.get(&qual_name) {
                        return TyType::Fn {
                            params: sig.params.iter().map(|(_, t)| t.clone()).collect(),
                            ret: Box::new(sig.ret.clone()),
                        };
                    }
                    if self.env.models.contains_key(&qual_name) {
                        return TyType::Unknown;
                    }
                    if let Some(t) = self.env.lookup(&qual_name) {
                        return t.clone();
                    }
                    self.error(format!("undefined identifier `{}`", qual_name), span.clone());
                    return TyType::Unknown;
                }
                if let TyType::Named { name: model_name, .. } = &recv {
                    if let Some(info) = self.env.models.get(model_name) {
                        if let Some(t) = info.fields.get(name) { return t.clone(); }
                        if let Some(sig) = info.methods.get(name) {
                            return TyType::Fn {
                                params: sig.params.iter().map(|(_, t)| t.clone()).collect(),
                                ret: Box::new(sig.ret.clone()),
                            };
                        }
                    }
                }
                // Common shape/dtype field access on tensors — silently accept.
                if matches!(name.as_str(), "shape" | "dtype" | "rank" | "ndim" | "size") {
                    return TyType::Unknown;
                }
                TyType::Unknown
            }
        }
    }

    fn lit_type(&self, lit: &Literal) -> TyType {
        match lit {
            // #445: an explicit type suffix (`42u32`) is the most explicit
            // statement of intent a literal can carry — it types concretely,
            // exactly like a suffixed float. Only the bare literal stays
            // context-adopting.
            Literal::Int(n, suffix) => match suffix {
                Some(s) => { let _ = n; TyType::Scalar(s.clone()) }
                None => TyType::IntLit(*n),
            },
            Literal::Float(v, suffix) => match suffix {
                Some(s) => TyType::Scalar(s.clone()),
                // Untyped float literal: stays context-flexible (#284), adopting
                // a float annotation/return/arg type, defaulting to f32 when bound
                // unconstrained.
                None => TyType::FloatLit(v.to_bits()),
            },
            Literal::Str(_)   => TyType::Scalar(ScalarType::Str),
            Literal::Char(_)  => TyType::Scalar(ScalarType::U32),
            Literal::Bool(_)  => TyType::Scalar(ScalarType::Bool),
            Literal::Nil      => TyType::Unit,
        }
    }

    // ── Type resolution (ast::Type → TyType) ──────────────────────────────

    fn resolve_type(&mut self, ty: &Type) -> TyType {
        match ty {
            Type::Scalar(s, _) => TyType::Scalar(s.clone()),
            Type::Tensor(inner, sh, _) => {
                TyType::Tensor(Box::new(self.resolve_type(inner)), self.resolve_shape(sh))
            }
            Type::View(inner, sh, _) => {
                TyType::View(Box::new(self.resolve_type(inner)), self.resolve_shape(sh))
            }
            Type::KV(inner, sh, _) => {
                TyType::KV(Box::new(self.resolve_type(inner)), self.resolve_shape(sh))
            }
            Type::Mesh(axes, _) => {
                let resolved: Vec<(String, SymDim)> = axes.iter()
                    .map(|a| (a.name.clone(), SymDim::from_expr(&a.size).simplify()))
                    .collect();
                TyType::Mesh(resolved)
            }
            Type::Fn(args, ret, _) => TyType::Fn {
                params: args.iter().map(|t| self.resolve_type(t)).collect(),
                ret: Box::new(self.resolve_type(ret)),
            },
            Type::Tuple(elems, _) => TyType::Tuple(elems.iter().map(|t| self.resolve_type(t)).collect()),
            Type::Array(inner, size, _) => {
                TyType::Array(Box::new(self.resolve_type(inner)), SymDim::from_expr(size).simplify())
            }
            Type::RawPtr(inner, _) => TyType::RawPtr(Box::new(self.resolve_type(inner))),
            Type::Named { name, args, span } => {
                // `any` (#186): the dynamic escape hatch. Resolves to ⊥/Unknown,
                // which `compatible_with` treats as match-anything in both
                // directions — letting a value whose type varies at runtime cross
                // a function boundary (param or return). `any`-typed code is
                // interpreter-only; the JIT rejects it (see ty_from_ast in jit.rs).
                if name == "any" {
                    return TyType::Unknown;
                }
                // NOTE: a `m: map` annotation is deliberately NOT resolved to
                // TyType::Map. Map is match-anything in `compatible_with`, so a
                // `map`-typed parameter also accepts a list — the annotation can't
                // guarantee the runtime value is a real map, and trusting it made
                // the #204 for-in-map lint fire on a list passed as `map`. Only
                // the map-PRODUCING builtins (map/map_new/map_set/map_del) yield
                // TyType::Map, so the lint only fires where a map is guaranteed.
                if let Some(alias) = self.aliases.get(name).cloned() {
                    return self.resolve_type_alias(&alias, args, span.clone());
                }
                // #336: a bare enum name resolves to its nominal enum type.
                if self.env.enums.contains_key(name) {
                    return TyType::Enum(name.clone());
                }
                // Resolve type args; named types can carry type or expr args
                let resolved_args: Vec<TyType> = args.iter().filter_map(|a| match a {
                    TypeArg::Type(t) => Some(self.resolve_type(t)),
                    _ => None,
                }).collect();
                TyType::Named { name: name.clone(), args: resolved_args }
            }
        }
    }

    fn resolve_type_alias(&mut self, alias: &TypeAlias, args: &[TypeArg], span: Span) -> TyType {
        if self.resolving_aliases.iter().any(|name| name == &alias.name) {
            self.error(format!("recursive type alias `{}`", alias.name), span);
            return TyType::Unknown;
        }
        if args.len() != alias.shape_params.len() {
            self.error(
                format!(
                    "type alias `{}` expects {} shape args, got {}",
                    alias.name,
                    alias.shape_params.len(),
                    args.len(),
                ),
                span,
            );
            return TyType::Unknown;
        }

        let mut shape_args = HashMap::new();
        for (param, arg) in alias.shape_params.iter().zip(args) {
            match symdim_from_type_arg(arg) {
                Some(dim) => {
                    shape_args.insert(param.name.clone(), dim);
                }
                None => {
                    self.error(
                        format!("type alias `{}` expects shape arg `{}`", alias.name, param.name),
                        param.span.clone(),
                    );
                    return TyType::Unknown;
                }
            }
        }

        self.resolving_aliases.push(alias.name.clone());
        self.env.push_shape_scope();
        for param in &alias.shape_params {
            self.env.bind_shape_param(&param.name, None);
        }
        let resolved = self.resolve_type(&alias.ty);
        self.env.pop_shape_scope();
        self.resolving_aliases.pop();

        substitute_shape_args(resolved, &shape_args)
    }

    fn resolve_shape(&mut self, sh: &ShapeSpec) -> Shape {
        let dims = sh.elems.iter().map(|e| match e {
            ShapeElem::Wildcard(_) => SymDim::Wildcard,
            ShapeElem::Spread(_)   => SymDim::Wildcard,  // approximation
            ShapeElem::Streaming(_) => SymDim::Streaming,
            ShapeElem::Expr(e) => SymDim::from_expr(e).simplify(),
        }).collect();
        Shape::new(dims)
    }

    // ── Diagnostics ───────────────────────────────────────────────────────

    fn error(&mut self, msg: impl Into<String>, span: Span) {
        self.errors.push(TypeError { msg: msg.into(), span, hint: None });
    }

    fn error_with_hint(&mut self, msg: impl Into<String>, span: Span, hint: Option<String>) {
        self.errors.push(TypeError { msg: msg.into(), span, hint });
    }

    /// Emit a non-fatal lint (safe-mode diagnostic). Surfaced by the CLI but does
    /// not fail `--check`. Demon mode releases the restriction: when `self.demon`
    /// is set the lint is dropped on the floor — no collection, no print (#196).
    fn warn(&mut self, msg: impl Into<String>, span: Span, hint: Option<String>) {
        if self.demon { return; }
        self.warnings.push(TypeError { msg: msg.into(), span, hint });
    }
}

/// A receiver type that definitionally carries no methods — scalars, tensors,
/// tuples, etc. Models (`Named`) own methods; `Unknown`/`Module` are ambiguous
/// or already handled, so they are excluded. Used by the #199 method-call lint
/// to confirm `x.floor()` cannot resolve to any real method.
/// Walk an assignment LHS down through index (`x[i]`) and field (`x.f`) postfix
/// chains to the base identifier it ultimately writes through. Used by the #247
/// immutability check so that `x[i] = ...` / `x.f[i] = ...` are subject to the
/// same `let !` rule as a plain `x = ...`. Returns `None` if the chain isn't
/// rooted in a simple identifier (e.g. `foo()[i] = ...`).
fn lhs_root_ident(e: &Expr) -> Option<&str> {
    match e {
        Expr::Ident(n, _) => Some(n.as_str()),
        Expr::Postfix { expr, op: PostfixOp::Index(_), .. } => lhs_root_ident(expr),
        Expr::Postfix { expr, op: PostfixOp::Field(_), .. } => lhs_root_ident(expr),
        _ => None,
    }
}

/// #248: recognize a *literal* arena tensor constructor
/// `(forge|vault).(zeros|ones|uninit)[<elem>, [<dims…>]]` and derive the
/// concrete `Tensor` type it allocates, with dims as `SymDim`s.
///
/// This is deliberately a syntactic recognizer used at operator sites (and at
/// un-annotated `let` bindings), NOT a change to what `check_expr` reports for
/// a constructor expression — that stays `Unknown`. Keeping the constructor's
/// reported type `Unknown` is what preserves KV-seeding annotations
/// (`let k: KV[…] = forge.ones[…]`) and leaves `Shape::same` untouched, while
/// still letting the existing `Shape::matmul`/`Shape::broadcast` checks fire on
/// provably-incompatible *constant* dims (they return `Unknown` — no error —
/// for symbolic dims, so this never false-positives on shape-parametric code).
/// #403 (MEMORY §2): does this expression allocate uninitialized memory
/// (`forge.uninit[…]` / `vault.uninit[…]`)? Element-type-agnostic on purpose:
/// a model-element uninit (#181) is just as undefined as a scalar-element one,
/// so this matches looser than `ctor_tensor_ty`.
fn is_uninit_ctor(expr: &Expr) -> bool {
    let base = match expr {
        Expr::Postfix { expr: base, op: PostfixOp::Index(_), .. } => base.as_ref(),
        _ => return false,
    };
    matches!(base, Expr::Postfix { expr: inner, op: PostfixOp::Field(m), .. }
        if m == "uninit" && matches!(inner.as_ref(), Expr::Ident(a, _) if a == "forge" || a == "vault"))
}

/// #442 (MEMORY §3.1): does this expression produce a value living in the
/// Vault? True when the postfix spine (calls, index/bracket args) bottoms out
/// in a `vault.<method>` access (`vault.zeros[…]`, `vault.load[…](path)`, …)
/// or when it is a `vault { … }` block expression (whose allocations land in
/// the Vault by MEMORY §3 rule 1).
fn is_vault_ctor(expr: &Expr) -> bool {
    if matches!(expr, Expr::ArenaBlock(ab) if ab.kind == ArenaKind::Vault) {
        return true;
    }
    let mut e = expr;
    loop {
        match e {
            Expr::Postfix { expr: inner, op: PostfixOp::Field(_), .. } => {
                return matches!(inner.as_ref(), Expr::Ident(a, _) if a == "vault");
            }
            Expr::Postfix { expr: inner, .. } => e = inner.as_ref(),
            _ => return false,
        }
    }
}

fn ctor_tensor_ty(expr: &Expr) -> Option<TyType> {
    let (base, idxs) = match expr {
        Expr::Postfix { expr: base, op: PostfixOp::Index(idxs), .. } => (base.as_ref(), idxs),
        _ => return None,
    };
    let (arena, method) = match base {
        Expr::Postfix { expr: inner, op: PostfixOp::Field(method), .. } => match inner.as_ref() {
            Expr::Ident(arena, _) => (arena.as_str(), method.as_str()),
            _ => return None,
        },
        _ => return None,
    };
    if !matches!(arena, "forge" | "vault") { return None; }
    if !matches!(method, "zeros" | "ones" | "uninit") { return None; }
    // Must be exactly `[<elem>, <shape>]` — two plain index exprs, no slices.
    let args: Vec<&Expr> = idxs.iter().filter_map(|e| match e {
        IndexElem::Expr(x) => Some(x),
        _ => None,
    }).collect();
    if args.len() != 2 || args.len() != idxs.len() { return None; }
    // Element type must be a scalar type name. A model element type
    // (`forge.uninit[ModelName, [N]]`, #181) returns None — not a tensor.
    let elem = scalar_type_from_ident(args[0])?;
    let dims = match args[1] {
        Expr::TensorLit(elems, _) => elems.iter().map(SymDim::from_expr).collect(),
        other => vec![SymDim::from_expr(other)],
    };
    Some(TyType::Tensor(Box::new(TyType::Scalar(elem)), Shape::new(dims)))
}

/// Map a scalar type *name* used in value position (a constructor's element-type
/// argument, e.g. the `f32` in `forge.zeros[f32, […]]`) to its `ScalarType`.
/// Returns `None` for anything that isn't a known scalar type name.
fn scalar_type_from_ident(e: &Expr) -> Option<ScalarType> {
    let name = match e { Expr::Ident(n, _) => n.as_str(), _ => return None };
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

fn ty_has_no_methods(t: &TyType) -> bool {
    matches!(t,
        TyType::Scalar(_) | TyType::Tensor(..) | TyType::View(..)
        | TyType::KV(..) | TyType::Array(..) | TyType::Tuple(_)
        | TyType::RawPtr(_) | TyType::Mesh(_) | TyType::Unit)
}

/// The string methods demoniC's interpreter actually implements (the supported
/// `recv.m(args)` forms on a `str` receiver). Any other name on a `str` falls
/// through to an opaque value at runtime, so the #202 lint flags it. **Keep this
/// in sync with `call_str_method` in interp.rs.**
pub(crate) fn is_supported_str_method(name: &str) -> bool {
    matches!(name,
        "split" | "lines" | "split_lines" | "trim" | "strip"
        | "upper" | "lower" | "starts_with" | "ends_with" | "contains"
        | "replace" | "find" | "index" | "count" | "len")
}

/// Best-effort syntactic check: could this expression evaluate to a negative
/// value? Used by the #198 `%` lint to flag only sign-bearing dividends. We
/// deliberately keep this narrow (subtraction or unary negation) so the common
/// non-negative shapes — loop indices, lengths, literals — don't warn.
/// True if `e` is the numeric literal `v` (int or float), seeing through
/// grouping parens. Used by the #231 no-effect-arithmetic lint to spot identity
/// operands (`+ 0` / `* 1` / …).
fn is_num_literal(e: &Expr, v: f64) -> bool {
    match e {
        Expr::Literal(Literal::Int(n, _), _) => *n as f64 == v,
        Expr::Literal(Literal::Float(f, _), _) => *f == v,
        Expr::Tuple(elems, _) if elems.len() == 1 => is_num_literal(&elems[0], v),
        _ => false,
    }
}

/// The f32-family scalar types (#473) — f32-backed in both backends by the
/// #179 convention. Mirrors `interp::scalar_is_f32_family` / the JIT's copy.
fn is_f32_family(t: &ScalarType) -> bool {
    matches!(t, ScalarType::F32 | ScalarType::F16 | ScalarType::Bf16
                | ScalarType::Tf32 | ScalarType::Fp8E4M3 | ScalarType::Fp8E5M2)
}

fn expr_may_be_negative(e: &Expr) -> bool {
    match e {
        Expr::UnOp { op: UnOp::Neg, .. } => true,
        // `a - b` can go negative — but a constant decrement on an otherwise
        // non-negative value (`sum - 1`, `idx - 7`) is a common safe adjustment
        // and not the floored-mod footgun. Int literals are always non-negative
        // in the AST (a leading `-` is a `UnOp::Neg`), so a literal subtrahend
        // marks the decrement idiom; only flag a non-literal difference.
        Expr::BinOp { op: BinOp::Sub, rhs, .. } =>
            !matches!(rhs.as_ref(), Expr::Literal(Literal::Int(..), _)),
        // See through grouping parens (parser builds a 1-tuple for `(e)`).
        Expr::Tuple(elems, _) if elems.len() == 1 => expr_may_be_negative(&elems[0]),
        _ => false,
    }
}

fn is_pipeline_block_placeholder(name: &str) -> bool {
    let Some(suffix) = name.strip_prefix("block_") else {
        return false;
    };
    !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit())
}

fn directive_i64_arg(d: &Directive, name: &str) -> Option<i64> {
    d.args.iter().find_map(|arg| match arg {
        DArg::Named { name: arg_name, value, .. } if arg_name == name => {
            expr_i64(value)
        }
        _ => None,
    })
}

fn expr_i64(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Literal(Literal::Int(n, _), _) => Some(*n),
        Expr::UnOp { op: UnOp::Neg, operand, .. } => expr_i64(operand).map(|n| -n),
        _ => None,
    }
}

fn directive_mesh_axis_arg(d: &Directive) -> Option<String> {
    d.args.iter().find_map(|arg| match arg {
        DArg::Named { name, value, .. } if name == "mesh" => {
            if let Expr::Postfix { op: PostfixOp::Field(axis), .. } = value {
                Some(axis.clone())
            } else {
                None
            }
        }
        _ => None,
    })
}

fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let normalized = if axis < 0 {
        rank as i64 + axis
    } else {
        axis
    };
    if normalized < 0 || normalized >= rank as i64 {
        None
    } else {
        Some(normalized as usize)
    }
}

fn symdim_divided_by(dim: &SymDim, divisor: &str) -> bool {
    match dim {
        SymDim::Div(_, rhs) => symdim_mentions_var(rhs, divisor),
        _ => false,
    }
}

fn symdim_mentions_var(dim: &SymDim, name: &str) -> bool {
    match dim {
        SymDim::Var(var) => var == name,
        SymDim::Add(l, r)
        | SymDim::Sub(l, r)
        | SymDim::Mul(l, r)
        | SymDim::Div(l, r)
        | SymDim::Mod(l, r) => symdim_mentions_var(l, name) || symdim_mentions_var(r, name),
        SymDim::Neg(x) => symdim_mentions_var(x, name),
        SymDim::Const(_) | SymDim::Streaming | SymDim::Wildcard | SymDim::Unknown => false,
    }
}

fn stmt_kind(stmt: &Stmt) -> &'static str {
    match stmt {
        Stmt::Let(_) => "let",
        Stmt::Expr { .. } => "expr",
        Stmt::If(_) => "if",
        Stmt::Match(_) => "match",
        Stmt::For { .. } => "for",
        Stmt::While { .. } => "while",
        Stmt::Loop { .. } => "loop",
        Stmt::Stage { .. } => "stage",
        Stmt::Directive { .. } => "directive",
        Stmt::DirectiveBlock { .. } => "directive block",
        Stmt::Break(_) => "break",
        Stmt::Continue(_) => "continue",
        Stmt::Return { .. } => "return",
    }
}

fn symdim_from_type_arg(arg: &TypeArg) -> Option<SymDim> {
    match arg {
        TypeArg::Expr(e) | TypeArg::Named { value: e, .. } => {
            Some(SymDim::from_expr(e).simplify())
        }
        TypeArg::Type(Type::Named { name, args, .. }) if args.is_empty() => {
            Some(SymDim::Var(name.clone()))
        }
        _ => None,
    }
}

/// #295: inclusive `(min, max)` value range of a fixed-width integral scalar,
/// or `None` for types we don't literal-range-check. Returns `None` for the
/// quantization/exotic int kinds (`Int4`/`Int8`/`Trit`) and every non-integral
/// type, so a literal bound to one of those is left alone (conservative — only
/// the standard C-width ints get an overflow diagnostic). `i128` comfortably
/// holds every bound including `u64::MAX`.
fn int_scalar_range(s: ScalarType) -> Option<(i128, i128)> {
    use ScalarType::*;
    Some(match s {
        I8  => (i8::MIN as i128,  i8::MAX as i128),
        I16 => (i16::MIN as i128, i16::MAX as i128),
        I32 => (i32::MIN as i128, i32::MAX as i128),
        I64 => (i64::MIN as i128, i64::MAX as i128),
        U8  => (0, u8::MAX as i128),
        U16 => (0, u16::MAX as i128),
        U32 => (0, u32::MAX as i128),
        U64 => (0, u64::MAX as i128),
        _ => return None,
    })
}

/// Resolve any untyped integer literal in `ty` to its default `i64` (#295).
/// Applied when a literal is *bound* (a `let` without an integral annotation, a
/// tuple/array element) so it stops being context-flexible and becomes a
/// concrete value type — the standard literal-defaulting fallback.
fn concretize(ty: TyType) -> TyType {
    match ty {
        TyType::IntLit(_) => TyType::Scalar(ScalarType::I64),
        TyType::FloatLit(_) => TyType::Scalar(ScalarType::F64),
        TyType::Tuple(ts) => TyType::Tuple(ts.into_iter().map(concretize).collect()),
        TyType::Tensor(e, s) => TyType::Tensor(Box::new(concretize(*e)), s),
        TyType::View(e, s) => TyType::View(Box::new(concretize(*e)), s),
        TyType::KV(e, s) => TyType::KV(Box::new(concretize(*e)), s),
        TyType::Array(e, n) => TyType::Array(Box::new(concretize(*e)), n),
        other => other,
    }
}

fn substitute_shape_args(ty: TyType, args: &HashMap<String, SymDim>) -> TyType {
    match ty {
        TyType::Tensor(inner, shape) => {
            TyType::Tensor(Box::new(substitute_shape_args(*inner, args)), substitute_shape(shape, args))
        }
        TyType::View(inner, shape) => {
            TyType::View(Box::new(substitute_shape_args(*inner, args)), substitute_shape(shape, args))
        }
        TyType::KV(inner, shape) => {
            TyType::KV(Box::new(substitute_shape_args(*inner, args)), substitute_shape(shape, args))
        }
        TyType::Mesh(axes) => TyType::Mesh(
            axes.into_iter()
                .map(|(name, dim)| (name, substitute_symdim(dim, args)))
                .collect()
        ),
        TyType::Fn { params, ret } => TyType::Fn {
            params: params.into_iter().map(|p| substitute_shape_args(p, args)).collect(),
            ret: Box::new(substitute_shape_args(*ret, args)),
        },
        TyType::Tuple(elems) => {
            TyType::Tuple(elems.into_iter().map(|t| substitute_shape_args(t, args)).collect())
        }
        TyType::Array(inner, size) => {
            TyType::Array(Box::new(substitute_shape_args(*inner, args)), substitute_symdim(size, args))
        }
        TyType::Named { name, args: type_args } => TyType::Named {
            name,
            args: type_args.into_iter().map(|t| substitute_shape_args(t, args)).collect(),
        },
        TyType::RawPtr(inner) => TyType::RawPtr(Box::new(substitute_shape_args(*inner, args))),
        TyType::Scalar(_) | TyType::IntLit(_) | TyType::FloatLit(_) | TyType::Enum(_) | TyType::Unit | TyType::Map | TyType::Unknown | TyType::Module { .. } => ty,
    }
}

fn infer_call_shape_bindings(param: &TyType, arg: &TyType, out: &mut HashMap<String, SymDim>) {
    match (param, arg) {
        (TyType::Tensor(pt, ps), TyType::Tensor(at, ash))
        | (TyType::View(pt, ps), TyType::View(at, ash))
        | (TyType::KV(pt, ps), TyType::KV(at, ash)) => {
            infer_call_shape_bindings(pt, at, out);
            if ps.rank() == ash.rank() {
                for (pdim, adim) in ps.dims.iter().zip(&ash.dims) {
                    infer_symdim_binding(pdim, adim, out);
                }
            }
        }
        (TyType::Tuple(params), TyType::Tuple(args)) if params.len() == args.len() => {
            for (p, a) in params.iter().zip(args) {
                infer_call_shape_bindings(p, a, out);
            }
        }
        (TyType::Array(pt, psize), TyType::Array(at, asize)) => {
            infer_call_shape_bindings(pt, at, out);
            infer_symdim_binding(psize, asize, out);
        }
        (TyType::Fn { params: pp, ret: pr }, TyType::Fn { params: ap, ret: ar })
            if pp.len() == ap.len() =>
        {
            for (p, a) in pp.iter().zip(ap) {
                infer_call_shape_bindings(p, a, out);
            }
            infer_call_shape_bindings(pr, ar, out);
        }
        _ => {}
    }
}

fn infer_symdim_binding(param: &SymDim, arg: &SymDim, out: &mut HashMap<String, SymDim>) {
    if let SymDim::Var(name) = param {
        out.entry(name.clone()).or_insert_with(|| arg.clone());
    }
}

fn substitute_shape(shape: Shape, args: &HashMap<String, SymDim>) -> Shape {
    Shape::new(
        shape.dims.into_iter()
            .map(|dim| substitute_symdim(dim, args))
            .collect()
    )
}

fn substitute_symdim(dim: SymDim, args: &HashMap<String, SymDim>) -> SymDim {
    use SymDim::*;
    match dim {
        Var(name) => args.get(&name).cloned().unwrap_or(Var(name)),
        Add(l, r) => Add(
            Box::new(substitute_symdim(*l, args)),
            Box::new(substitute_symdim(*r, args)),
        ).simplify(),
        Sub(l, r) => Sub(
            Box::new(substitute_symdim(*l, args)),
            Box::new(substitute_symdim(*r, args)),
        ).simplify(),
        Mul(l, r) => Mul(
            Box::new(substitute_symdim(*l, args)),
            Box::new(substitute_symdim(*r, args)),
        ).simplify(),
        Div(l, r) => Div(
            Box::new(substitute_symdim(*l, args)),
            Box::new(substitute_symdim(*r, args)),
        ).simplify(),
        Mod(l, r) => Mod(
            Box::new(substitute_symdim(*l, args)),
            Box::new(substitute_symdim(*r, args)),
        ).simplify(),
        Neg(x) => Neg(Box::new(substitute_symdim(*x, args))).simplify(),
        Const(_) | Streaming | Wildcard | Unknown => dim,
    }
}

// ─── Expr span extension ─────────────────────────────────────────────────────
// Walk the Expr to find its top-level span for diagnostics.

/// Walk a Type collecting all identifier-style shape var references
/// (used for implicit shape param inference).
fn collect_shape_vars(ty: &Type, out: &mut std::collections::HashSet<String>) {
    use crate::ast::*;
    fn walk_expr(e: &Expr, out: &mut std::collections::HashSet<String>) {
        match e {
            Expr::Ident(name, _) => {
                // Heuristic: uppercase-starting single-word ident = likely shape var.
                if name.chars().next().map_or(false, |c| c.is_ascii_uppercase()) {
                    out.insert(name.clone());
                }
            }
            Expr::BinOp { lhs, rhs, .. } => { walk_expr(lhs, out); walk_expr(rhs, out); }
            Expr::UnOp { operand, .. } => walk_expr(operand, out),
            Expr::Tuple(es, _) => { for e in es { walk_expr(e, out); } }
            _ => {}
        }
    }
    fn walk_shape(sh: &ShapeSpec, out: &mut std::collections::HashSet<String>) {
        for elem in &sh.elems {
            if let ShapeElem::Expr(e) = elem { walk_expr(e, out); }
        }
    }
    match ty {
        Type::Tensor(inner, sh, _) | Type::View(inner, sh, _) | Type::KV(inner, sh, _) => {
            collect_shape_vars(inner, out);
            walk_shape(sh, out);
        }
        Type::Mesh(axes, _) => { for a in axes { walk_expr(&a.size, out); } }
        Type::Fn(args, ret, _) => {
            for a in args { collect_shape_vars(a, out); }
            collect_shape_vars(ret, out);
        }
        Type::Tuple(elems, _) => { for e in elems { collect_shape_vars(e, out); } }
        Type::Array(inner, size, _) => {
            collect_shape_vars(inner, out);
            walk_expr(size, out);
        }
        Type::Named { args, .. } => {
            for a in args {
                match a {
                    TypeArg::Type(t) => collect_shape_vars(t, out),
                    TypeArg::Expr(e) | TypeArg::Named { value: e, .. } => walk_expr(e, out),
                }
            }
        }
        Type::Scalar(_, _) | Type::RawPtr(_, _) => {}
    }
}

/// Extract a positive literal `usize` from an expression (a bare int literal) —
/// reads the `.split[n, ...]` piece count.
fn lit_usize(e: &crate::ast::Expr) -> Option<usize> {
    match e {
        crate::ast::Expr::Literal(crate::ast::Literal::Int(n, _), _) if *n > 0 => Some(*n as usize),
        _ => None,
    }
}

/// Collect every identifier *referenced* (read) anywhere in a block — used by
/// the #398 captured-mutable check. Conservative and read-only; it does not try
/// to track binding scopes (the caller filters against the type env instead).
fn collect_body_idents(block: &crate::ast::Block, out: &mut std::collections::HashSet<String>) {
    use crate::ast::*;
    fn walk_e(e: &Expr, out: &mut std::collections::HashSet<String>) {
        match e {
            Expr::Ident(name, _) => { out.insert(name.clone()); }
            Expr::BinOp { lhs, rhs, .. } => { walk_e(lhs, out); walk_e(rhs, out); }
            Expr::UnOp { operand, .. } => walk_e(operand, out),
            Expr::Cast { expr, .. } => walk_e(expr, out),
            Expr::Tuple(es, _) | Expr::TensorLit(es, _) => { for e in es { walk_e(e, out); } }
            Expr::Range { start, end, .. } => {
                if let Some(s) = start { walk_e(s, out); }
                if let Some(en) = end { walk_e(en, out); }
            }
            Expr::Postfix { expr, op, .. } => {
                walk_e(expr, out);
                match op {
                    PostfixOp::Call(args) | PostfixOp::BracketArgs(args) => {
                        for a in args {
                            match a {
                                CallArg::Positional(e) | CallArg::Named { value: e, .. } => walk_e(e, out),
                                CallArg::Spread(_) => {}
                            }
                        }
                    }
                    PostfixOp::Index(elems) => {
                        for el in elems {
                            if let IndexElem::Expr(e) = el { walk_e(e, out); }
                        }
                    }
                    PostfixOp::Constructor(fields) => { for (_, e) in fields { walk_e(e, out); } }
                    _ => {}
                }
            }
            Expr::StructLit { type_args, fields, .. } => {
                for t in type_args { walk_e(t, out); }
                for (_, v) in fields { walk_e(v, out); }
            }
            Expr::If(ie) => {
                walk_e(&ie.cond, out);
                walk_block(&ie.then_branch, out);
                match &ie.else_branch {
                    Some(ElseBranch::Block(b)) => walk_block(b, out),
                    Some(ElseBranch::If(i)) => walk_e(&Expr::If(i.clone()), out),
                    None => {}
                }
            }
            Expr::Match(m) => {
                walk_e(&m.scrutinee, out);
                for arm in &m.arms {
                    if let Some(g) = &arm.guard { walk_e(g, out); }
                    walk_e(&arm.body, out);
                }
            }
            Expr::Block(b) => walk_block(b, out),
            Expr::DirectiveBlock { body, .. } => walk_block(body, out),
            // #398: a closure BODY can still capture module-level mutable state
            // (`let !bias` referenced inside `fn() { sum(bias) }`). Recurse so the
            // captured-mutable check sees those free references, but exclude the
            // closure's OWN params — they shadow and are not captures. Skipping
            // the whole FnLit (as before) let a capture hide inside a closure and
            // slip past the check, silently yielding an opaque `grads.<name>`.
            Expr::FnLit(fl) => {
                let mut inner = std::collections::HashSet::new();
                walk_block(&fl.body, &mut inner);
                for p in &fl.params { inner.remove(&p.name); }
                out.extend(inner);
            }
            // An arena/vault/stream block feeding the loss can likewise carry a
            // captured reference — recurse into its body too.
            Expr::ArenaBlock(ab) => walk_block(&ab.body, out),
            Expr::Literal(..) | Expr::Nil(_) | Expr::Underscore(_) | Expr::Spread(_) => {}
        }
    }
    fn walk_block(b: &Block, out: &mut std::collections::HashSet<String>) {
        for s in &b.stmts { walk_stmt(s, out); }
        if let Some(t) = &b.tail_expr { walk_e(t, out); }
    }
    fn walk_stmt(s: &Stmt, out: &mut std::collections::HashSet<String>) {
        match s {
            Stmt::Let(l) => walk_e(&l.value, out),
            Stmt::Expr { lhs, assign, .. } => {
                walk_e(lhs, out);
                if let Some((_, rhs)) = assign { walk_e(rhs, out); }
            }
            Stmt::Return { value: Some(e), .. } => walk_e(e, out),
            Stmt::If(ie) => walk_e(&Expr::If(Box::new(ie.clone())), out),
            Stmt::Match(m) => walk_e(&Expr::Match(Box::new(m.clone())), out),
            Stmt::For { iter, body, .. } => { walk_e(iter, out); walk_block(body, out); }
            Stmt::While { cond, body, .. } => { walk_e(cond, out); walk_block(body, out); }
            Stmt::Loop { body, .. } => walk_block(body, out),
            Stmt::Stage { body, .. } => walk_e(body, out),
            Stmt::Directive { inner, .. } => walk_stmt(inner, out),
            Stmt::DirectiveBlock { body, .. } => walk_block(body, out),
            Stmt::Return { value: None, .. } | Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }
    walk_block(block, out);
}

impl Expr {
    pub fn span_of(&self) -> Span {
        match self {
            Expr::Literal(_, s) | Expr::Ident(_, s) | Expr::Underscore(s)
            | Expr::Spread(s)  | Expr::Nil(s) | Expr::Tuple(_, s)
            | Expr::TensorLit(_, s) | Expr::Range { span: s, .. }
            | Expr::BinOp { span: s, .. } | Expr::UnOp { span: s, .. }
            | Expr::Postfix { span: s, .. } | Expr::Cast { span: s, .. }
            | Expr::DirectiveBlock { span: s, .. }
            | Expr::StructLit { span: s, .. } => s.clone(),
            Expr::Block(b)  => b.span.clone(),
            Expr::ArenaBlock(ab) => ab.span.clone(),
            Expr::If(i)     => i.span.clone(),
            Expr::Match(m)  => m.span.clone(),
            Expr::FnLit(f)  => f.span.clone(),
        }
    }
}

fn prefix_types_in_ty(ty: TyType, prefix: &str, imported_models: &std::collections::HashSet<String>) -> TyType {
    match ty {
        TyType::Tensor(inner, shape) => {
            TyType::Tensor(Box::new(prefix_types_in_ty(*inner, prefix, imported_models)), shape)
        }
        TyType::View(inner, shape) => {
            TyType::View(Box::new(prefix_types_in_ty(*inner, prefix, imported_models)), shape)
        }
        TyType::KV(inner, shape) => {
            TyType::KV(Box::new(prefix_types_in_ty(*inner, prefix, imported_models)), shape)
        }
        TyType::Mesh(axes) => TyType::Mesh(axes),
        TyType::Fn { params, ret } => TyType::Fn {
            params: params.into_iter().map(|p| prefix_types_in_ty(p, prefix, imported_models)).collect(),
            ret: Box::new(prefix_types_in_ty(*ret, prefix, imported_models)),
        },
        TyType::Tuple(elems) => {
            TyType::Tuple(elems.into_iter().map(|t| prefix_types_in_ty(t, prefix, imported_models)).collect())
        }
        TyType::Array(inner, size) => {
            TyType::Array(Box::new(prefix_types_in_ty(*inner, prefix, imported_models)), size)
        }
        TyType::Named { name, args } => {
            let new_name = if imported_models.contains(&name) {
                format!("{}.{}", prefix, name)
            } else {
                name
            };
            TyType::Named {
                name: new_name,
                args: args.into_iter().map(|t| prefix_types_in_ty(t, prefix, imported_models)).collect(),
            }
        }
        other => other,
    }
}

// ─── Lint: dead write-back through a copied tensor element ────────────────────
//
// See `Checker::lint_writeback` for the rationale. This is a purely structural
// AST walk — no type information required — so it is robust against the parts
// of the type system that are still pre-alpha.

/// Per-function-scope accumulators for the write-back lint.
struct WbScope {
    /// Identifiers that appear in a read position anywhere in the scope.
    reads: HashSet<String>,
    /// Identifiers assigned with a plain `=` (the only form that could be a
    /// mistaken write-back), mapped to the first such assignment span.
    eq_assigns: HashMap<String, Span>,
    /// Mutable bindings initialized from a scalar tensor index: (name, span).
    idx_bindings: Vec<(String, Span)>,
}

impl WbScope {
    fn new() -> Self {
        WbScope { reads: HashSet::new(), eq_assigns: HashMap::new(), idx_bindings: Vec::new() }
    }
}

fn lint_item_writeback(item: &Item, out: &mut Vec<TypeError>) {
    match item {
        Item::Fn(f) => collect_writeback_warnings(&f.body, out),
        Item::Model(m) => {
            for mem in &m.members {
                if let ModelMember::Method(f) = mem {
                    collect_writeback_warnings(&f.body, out);
                }
            }
        }
        Item::Arena(a) => collect_writeback_warnings(&a.body, out),
        Item::Pub(inner) => lint_item_writeback(inner, out),
        Item::Directive { inner, .. } => lint_item_writeback(inner, out),
        Item::Let(_) | Item::TypeAlias(_) | Item::Enum(_) | Item::ExternFn(_) | Item::Use(_) => {}
    }
}

/// Analyse one function/method/arena body as a single scope.
fn collect_writeback_warnings(body: &Block, out: &mut Vec<TypeError>) {
    let mut sc = WbScope::new();
    scan_block_wb(body, &mut sc, out);
    for (name, let_span) in &sc.idx_bindings {
        if sc.eq_assigns.contains_key(name) && !sc.reads.contains(name) {
            out.push(TypeError {
                msg: format!(
                    "`{}` is bound to a copy of a tensor element, then assigned but never \
                     read; the assignment does not write back to the tensor (scalar indexing \
                     copies)",
                    name
                ),
                span: let_span.clone(),
                hint: Some(
                    "to mutate the tensor, assign through the index instead, e.g. `t[i] = ...`".to_string(),
                ),
            });
        }
    }
}

/// `t[i]` / `t[i, j]` — a scalar element read (every index is a point expr,
/// not a slice or full-axis). Slices yield views with their own CoW story and
/// are out of scope for this lint.
fn is_scalar_index(e: &Expr) -> bool {
    if let Expr::Postfix { op: PostfixOp::Index(elems), .. } = e {
        !elems.is_empty() && elems.iter().all(|el| matches!(el, IndexElem::Expr(_)))
    } else {
        false
    }
}

fn scan_block_wb(block: &Block, sc: &mut WbScope, out: &mut Vec<TypeError>) {
    for stmt in &block.stmts {
        scan_stmt_wb(stmt, sc, out);
    }
    if let Some(tail) = &block.tail_expr {
        scan_expr_wb(tail, sc, out);
    }
}

fn scan_stmt_wb(stmt: &Stmt, sc: &mut WbScope, out: &mut Vec<TypeError>) {
    match stmt {
        Stmt::Let(l) => {
            scan_expr_wb(&l.value, sc, out);
            if (l.mutating || l.is_mut) && is_scalar_index(&l.value) {
                if let Pattern::Ident(name, _) = &l.pattern {
                    sc.idx_bindings.push((name.clone(), l.span.clone()));
                }
            }
        }
        Stmt::Expr { lhs, assign, span } => match assign {
            Some((AssignOp::Eq, rhs)) => {
                scan_expr_wb(rhs, sc, out);
                if let Expr::Ident(name, _) = lhs {
                    // A pure write to a bare local — not a read of it.
                    sc.eq_assigns.entry(name.clone()).or_insert_with(|| span.clone());
                } else {
                    // `t[i] = ...`, `obj.f = ...` — the target sub-exprs are reads.
                    scan_expr_wb(lhs, sc, out);
                }
            }
            // Compound assigns and `<-` read the LHS as well as write it.
            Some((_, rhs)) => {
                scan_expr_wb(lhs, sc, out);
                scan_expr_wb(rhs, sc, out);
            }
            None => scan_expr_wb(lhs, sc, out),
        },
        Stmt::If(ifx) => scan_if_wb(ifx, sc, out),
        Stmt::Match(m) => scan_match_wb(m, sc, out),
        Stmt::For { iter, body, .. } => {
            scan_expr_wb(iter, sc, out);
            scan_block_wb(body, sc, out);
        }
        Stmt::While { cond, body, .. } => {
            scan_expr_wb(cond, sc, out);
            scan_block_wb(body, sc, out);
        }
        Stmt::Loop { body, .. } => scan_block_wb(body, sc, out),
        Stmt::Stage { body, .. } => scan_expr_wb(body, sc, out),
        Stmt::Directive { inner, .. } => scan_stmt_wb(inner, sc, out),
        Stmt::DirectiveBlock { body, .. } => scan_block_wb(body, sc, out),
        Stmt::Return { value, .. } => {
            if let Some(v) = value {
                scan_expr_wb(v, sc, out);
            }
        }
        Stmt::Break(_) | Stmt::Continue(_) => {}
    }
}

/// #403 (MEMORY §9.1): collect spans of `name <- …` statements anywhere in
/// `block`, lexically — nested loops, if/match arms, directive and arena
/// blocks included. `<-` exists only as a statement-level assign op, so
/// expressions need walking only where they carry blocks.
fn collect_stream_appends_block(block: &Block, name: &str, out: &mut Vec<Span>) {
    for stmt in &block.stmts {
        collect_stream_appends_stmt(stmt, name, out);
    }
    if let Some(tail) = &block.tail_expr {
        collect_stream_appends_expr(tail, name, out);
    }
}

fn collect_stream_appends_stmt(stmt: &Stmt, name: &str, out: &mut Vec<Span>) {
    match stmt {
        Stmt::Expr { lhs, assign, span } => {
            if matches!(assign, Some((AssignOp::StreamArrow, _)))
                && matches!(lhs, Expr::Ident(n, _) if n == name)
            {
                out.push(span.clone());
            } else if assign.is_none() {
                collect_stream_appends_expr(lhs, name, out);
            }
        }
        Stmt::If(ifx) => collect_stream_appends_if(ifx, name, out),
        Stmt::Match(m) => {
            for arm in &m.arms {
                collect_stream_appends_expr(&arm.body, name, out);
            }
        }
        Stmt::For { body, .. }
        | Stmt::While { body, .. }
        | Stmt::Loop { body, .. }
        | Stmt::DirectiveBlock { body, .. } => collect_stream_appends_block(body, name, out),
        Stmt::Directive { inner, .. } => collect_stream_appends_stmt(inner, name, out),
        Stmt::Let(_) | Stmt::Stage { .. } | Stmt::Break(_) | Stmt::Continue(_)
        | Stmt::Return { .. } => {}
    }
}

fn collect_stream_appends_if(ifx: &IfExpr, name: &str, out: &mut Vec<Span>) {
    collect_stream_appends_block(&ifx.then_branch, name, out);
    match &ifx.else_branch {
        Some(ElseBranch::Block(b)) => collect_stream_appends_block(b, name, out),
        Some(ElseBranch::If(inner)) => collect_stream_appends_if(inner, name, out),
        None => {}
    }
}

fn collect_stream_appends_expr(e: &Expr, name: &str, out: &mut Vec<Span>) {
    match e {
        Expr::Block(b) => collect_stream_appends_block(b, name, out),
        Expr::If(ifx) => collect_stream_appends_if(ifx, name, out),
        Expr::Match(m) => {
            for arm in &m.arms {
                collect_stream_appends_expr(&arm.body, name, out);
            }
        }
        Expr::ArenaBlock(ab) => collect_stream_appends_block(&ab.body, name, out),
        Expr::DirectiveBlock { body, .. } => collect_stream_appends_block(body, name, out),
        _ => {}
    }
}

fn scan_if_wb(ifx: &IfExpr, sc: &mut WbScope, out: &mut Vec<TypeError>) {
    scan_expr_wb(&ifx.cond, sc, out);
    scan_block_wb(&ifx.then_branch, sc, out);
    match &ifx.else_branch {
        Some(ElseBranch::Block(b)) => scan_block_wb(b, sc, out),
        Some(ElseBranch::If(inner)) => scan_if_wb(inner, sc, out),
        None => {}
    }
}

fn scan_match_wb(m: &MatchExpr, sc: &mut WbScope, out: &mut Vec<TypeError>) {
    scan_expr_wb(&m.scrutinee, sc, out);
    for arm in &m.arms {
        if let Some(g) = &arm.guard {
            scan_expr_wb(g, sc, out);
        }
        scan_expr_wb(&arm.body, sc, out);
    }
}

fn scan_expr_wb(e: &Expr, sc: &mut WbScope, out: &mut Vec<TypeError>) {
    match e {
        Expr::Ident(name, _) => {
            sc.reads.insert(name.clone());
        }
        Expr::Literal(..) | Expr::Underscore(_) | Expr::Spread(_) | Expr::Nil(_) => {}
        Expr::Tuple(es, _) | Expr::TensorLit(es, _) => {
            for x in es {
                scan_expr_wb(x, sc, out);
            }
        }
        Expr::Block(b) => scan_block_wb(b, sc, out),
        Expr::If(ifx) => scan_if_wb(ifx, sc, out),
        Expr::Match(m) => scan_match_wb(m, sc, out),
        // A closure is its own scope; analyse it independently so its reads
        // neither suppress nor trigger warnings in the enclosing function.
        Expr::FnLit(f) => collect_writeback_warnings(&f.body, out),
        Expr::ArenaBlock(a) => scan_block_wb(&a.body, sc, out),
        Expr::DirectiveBlock { body, .. } => scan_block_wb(body, sc, out),
        Expr::BinOp { lhs, rhs, .. } => {
            scan_expr_wb(lhs, sc, out);
            scan_expr_wb(rhs, sc, out);
        }
        Expr::UnOp { operand, .. } => scan_expr_wb(operand, sc, out),
        Expr::Postfix { expr, op, .. } => {
            scan_expr_wb(expr, sc, out);
            scan_postfix_wb(op, sc, out);
        }
        Expr::Cast { expr, .. } => scan_expr_wb(expr, sc, out),
        Expr::StructLit { type_args, fields, .. } => {
            for a in type_args {
                scan_expr_wb(a, sc, out);
            }
            for (_, v) in fields {
                scan_expr_wb(v, sc, out);
            }
        }
        Expr::Range { start, end, .. } => {
            if let Some(s) = start {
                scan_expr_wb(s, sc, out);
            }
            if let Some(en) = end {
                scan_expr_wb(en, sc, out);
            }
        }
    }
}

fn scan_postfix_wb(op: &PostfixOp, sc: &mut WbScope, out: &mut Vec<TypeError>) {
    match op {
        PostfixOp::Index(elems) => {
            for el in elems {
                match el {
                    IndexElem::Expr(e) => scan_expr_wb(e, sc, out),
                    IndexElem::Slice { start, end, step, .. } => {
                        for o in [start, end, step] {
                            if let Some(x) = o {
                                scan_expr_wb(x, sc, out);
                            }
                        }
                    }
                    IndexElem::FullSlice(_) => {}
                }
            }
        }
        PostfixOp::Call(args) | PostfixOp::BracketArgs(args) => {
            for a in args {
                match a {
                    CallArg::Positional(e) => scan_expr_wb(e, sc, out),
                    CallArg::Named { value, .. } => scan_expr_wb(value, sc, out),
                    CallArg::Spread(_) => {}
                }
            }
        }
        PostfixOp::Constructor(fields) => {
            for (_, v) in fields {
                scan_expr_wb(v, sc, out);
            }
        }
        PostfixOp::Transpose | PostfixOp::Query | PostfixOp::Field(_) => {}
    }
}
