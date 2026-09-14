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
use crate::shape::{Equiv, Shape, ShapeError, SymDim};
use crate::types::{Env, FnSig, ModelInfo, TyType};

// ─── Diagnostic ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TypeError {
    pub msg: String,
    pub span: Span,
    pub hint: Option<String>,
    /// For a shape error: the (expected, actual) shapes as data (#485), set
    /// where a mismatch compares two tensor-like types whose shapes disagree.
    /// The human rendering never reads this — `--json` is its only consumer.
    pub shapes: Option<(Shape, Shape)>,
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type error at {}:{}: {}", self.span.line, self.span.col, self.msg)?;
        if let Some(h) = &self.hint { write!(f, "\n  hint: {}", h)?; }
        Ok(())
    }
}

/// PORTS.md §5: the enclosing construct that makes a port call illegal.
/// A port call is an effect boundary; each of these promises something the
/// boundary breaks, so `port_open`/`port_call`/`port_close` are rejected in
/// their lexical extent (closures included) with a `port-forbidden` tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PortBan {
    /// `@grad fn` — the tape cannot record the call (#497).
    GradFn,
    /// `@fuse` block — no fusion crosses a port call.
    Fuse,
    /// `@deterministic` block — only a port with a determinism manifest may
    /// be called, and no port carries one yet.
    Deterministic,
}

impl PortBan {
    /// The construct as it appears in source, for the diagnostic.
    fn what(self) -> &'static str {
        match self {
            PortBan::GradFn => "`@grad fn`",
            PortBan::Fuse => "`@fuse` block",
            PortBan::Deterministic => "`@deterministic` block",
        }
    }

    /// Why this construct cannot contain the call.
    fn because(self) -> &'static str {
        match self {
            PortBan::GradFn =>
                "a port call is an effect boundary the gradient cannot cross",
            PortBan::Fuse =>
                "no fusion crosses a port call, so the block cannot be one kernel",
            PortBan::Deterministic =>
                "only a port with a determinism manifest may be called there, \
                 and no port carries one yet",
        }
    }

    /// #578: why this construct cannot contain an `extern fn` call. Same three
    /// constructs, different reasons — the gradient tape and the fusion pass
    /// fail on any opaque call, but `@deterministic` fails on a specific
    /// property of foreign code: its accumulation order is not this
    /// language's to fix.
    fn extern_because(self) -> &'static str {
        match self {
            PortBan::GradFn =>
                "a foreign call is a hard non-differentiable barrier — the tape \
                 cannot record what it does",
            PortBan::Fuse =>
                "no fusion crosses a foreign call, so the block cannot be one kernel",
            PortBan::Deterministic =>
                "foreign code accumulates in its own order, which this language \
                 does not fix and cannot reproduce",
        }
    }
}

/// PORTS.md §5: does this directive stack forbid port calls in what it wraps?
/// `@fuse` and `@deterministic` do; `@grad` is decided at the `fn` (it is a
/// `fn` directive, not a block one).
fn port_ban_of(directives: &[Directive]) -> Option<PortBan> {
    if directives.iter().any(|d| d.name == "fuse") {
        Some(PortBan::Fuse)
    } else if directives.iter().any(|d| d.name == "deterministic") {
        Some(PortBan::Deterministic)
    } else {
        None
    }
}

#[derive(Clone, Debug)]
pub struct ModuleEnv {
    pub env: Env,
    pub aliases: HashMap<String, TypeAlias>,
    /// #761: each `pub type` alias's dimensions, fully resolved against this
    /// module's own top-level `let` constants (a private `let` never crosses
    /// the module boundary as a name, SPEC.md §6.7 / #724 / #731 — but the
    /// alias's already-resolved SHAPE may, exactly as the JIT already treats
    /// an alias as transparent metadata, #622). An importer's checker never
    /// re-derives a name it cannot see; it reuses this cache instead.
    pub resolved_aliases: HashMap<String, TyType>,
    pub public_items: HashSet<String>,
    /// #724: every name this file's top-level `let`s bind. Never exported
    /// (`pub let` is a compile-time error, #616), so this never feeds
    /// `process_imports`'s reference-rewriting the way `public_items` does —
    /// it exists only so an importer's stray read of one of these names earns
    /// the specific accessor-rule diagnostic instead of a bare `undefined
    /// identifier` (`Checker::check_expr`'s `Expr::Ident` / qualified-`Field`
    /// arms).
    pub top_level_let_names: HashSet<String>,
    /// D4 (#724 follow-up): this file's OWN view of `import-let` names it can
    /// already name a module for — i.e. `Checker::import_private_lets` as it
    /// stood once this file finished checking, mapping each such name to the
    /// module that actually declares it (which may be a module this file
    /// never `use`s directly — see below). Only `top_level_let_names` is
    /// this file's OWN top-level `let`s; this field is every private `let`
    /// reachable through an UNALIASED `use` chain of any depth.
    ///
    /// Without this, a three-file chain (`top` `use`s `mid` `use`s `base`,
    /// `base` declares `let CAP`) has `top`'s `process_imports` only ever
    /// consult `mid`'s `top_level_let_names` — which is empty, `CAP` being
    /// `base`'s own binding, not `mid`'s — so `top`'s bare read of `CAP`
    /// (never legal: `SPEC.md §6.7`, a `let` never crosses even one module
    /// boundary) fell through to the generic `undefined identifier `CAP``
    /// instead of naming `base.dmc`. Threading this field through lets
    /// `process_imports` merge it in alongside `top_level_let_names` the
    /// same way `checked_modules` itself is threaded leaf-first — each
    /// file's `import_private_lets` already accumulates whatever its own
    /// unaliased imports contributed, so one file up the chain merging this
    /// field in (already keyed to the ORIGINAL declaring module, not the
    /// intermediate one) is transitive to any depth for free.
    pub transitive_import_lets: HashMap<String, String>,
}

// ─── Checker ─────────────────────────────────────────────────────────────────

pub struct Checker {
    pub env: Env,
    pub aliases: HashMap<String, TypeAlias>,
    /// #761: each alias's dimensions, resolved once against `const_dims` and
    /// its own shape parameters — see `resolve_alias_body` and `ModuleEnv`'s
    /// field of the same name.
    pub resolved_aliases: HashMap<String, TyType>,
    /// #761: module-level `let NAME = <int literal>` constants — the
    /// checker's analogue of the JIT's `ConstDim` table (#620). Consulted by
    /// `resolve_dim` wherever a shape position's name is not an in-scope
    /// shape parameter, so `Tensor[i64,[CAP]]` reads `CAP` as the dimension
    /// it names instead of a dangling free variable. Module-local only — an
    /// importer never gets a copy (a `let`, `pub` or not, never crosses a
    /// module boundary as a name, SPEC.md §6.7); what crosses for a `pub
    /// type` alias is the already-resolved shape in `resolved_aliases`.
    const_dims: HashMap<String, i64>,
    /// #766: every top-level `let` name this file declares — literal or
    /// not, mutable or not — recorded alongside `const_dims` purely so a
    /// name in an alias/annotation dimension position that ISN'T in
    /// `const_dims` can say precisely why instead of claiming no such
    /// top-level `let` exists at all: is it bound at all (this set), and if
    /// so is it mutable (`mutable_top_level_lets`) or just non-literal?
    all_top_level_let_names: std::collections::HashSet<String>,
    /// #766: the subset of `all_top_level_let_names` bound `let !`/`let
    /// mut` — checked first in `resolve_alias_body`'s diagnostic, since a
    /// mutable top-level binding is never a constant regardless of what its
    /// initializer looks like.
    mutable_top_level_lets: std::collections::HashSet<String>,
    resolving_aliases: Vec<String>,
    pub errors: Vec<TypeError>,
    /// Non-fatal lint diagnostics. Surfaced to the user but never block
    /// execution (`dmc run` proceeds even when warnings are present).
    pub warnings: Vec<TypeError>,
    /// Return type of the innermost function being checked; used for `?` context validation.
    current_fn_ret: Option<TyType>,
    /// #728: name of the innermost function or method being checked — `None`
    /// at module (top-level `let`) scope. Lets a diagnostic raised while
    /// resolving a param/return annotation name its enclosing function
    /// without threading it through every call.
    current_fn_name: Option<String>,
    /// #728: set to the enclosing model's name while `check_fn` runs for one
    /// of its methods (via `check_model`), so a diagnostic can say "method
    /// `f` of model `M`" instead of just "function `f`". `None` for a
    /// free function.
    current_model_name: Option<String>,
    /// #728 rework: the innermost function's own declared shape-parameter
    /// names (`f.shape_params`, e.g. `["N"]` for `fn cap[N](...)`) — restored
    /// alongside `current_fn_name` at the end of `check_fn`. Deliberately
    /// NOT the enclosing model's shape params too: `check_model`'s own
    /// "model-level shape params are not automatically in scope inside
    /// methods" diagnostic (the `unknown shape param` error a few lines
    /// above where `current_model_name` is set) means a method's own
    /// `shape_params` is genuinely the only thing in scope — reusing the
    /// model's names here would recommend a spelling that doesn't check.
    /// Consulted by the type-arg-less generic-model hint (`type_arg_less_
    /// hint`) at every position but the field one (which reads the
    /// enclosing model's `shape_params` directly, since that IS the
    /// declaring scope there) to decide whether the fix is "wrap with the
    /// name already in scope" or "declare a new one".
    current_fn_shape_params: Vec<String>,
    /// Top-level user fns → number of stacked `@grad` directives (0 = plain
    /// fn). Drives check-time validation of the autodiff calling conventions
    /// (`f.fwd_bwd(...)` etc.): an unknown method name on a `@grad fn` used
    /// to fall through to generic field access and silently produce garbage
    /// at runtime.
    grad_fn_counts: HashMap<String, usize>,
    /// #398: per-fn identifier sets (reads, closure reads, bound names) from
    /// `collect_body_idents`, recorded for every registered fn. The
    /// captured-mut rule follows calls out of a `@grad fn` body through this
    /// map: the tape does not trace a call, so a capture read inside a callee
    /// would leave the returned gradient silently short of that path.
    fn_body_refs: HashMap<String, BodyRefs>,
    /// #398: the same scan for **model methods**, keyed by method name —
    /// several models may define a method of the same name, so each entry
    /// carries every `(model, refs)` that answers to it. A method call in a
    /// `@grad fn` body names only the method (`h.contrib()`), and by the time
    /// the captured-mut rule runs the body's locals have left the env, so the
    /// receiver's model usually cannot be pinned down. The walk therefore
    /// enters every candidate: over-reporting a capture that is genuinely read
    /// by *some* `contrib` is a diagnostic the author can act on, where
    /// missing it is a wrong gradient nobody sees.
    method_body_refs: HashMap<String, Vec<(String, BodyRefs)>>,
    /// #403 (MEMORY §2): per-fn param mutability (`!` markers), by position.
    /// Lets call sites treat an uninit binding passed to a `!` param as the
    /// fill that initializes it, and one passed to a plain param as a read.
    fn_mut_params: HashMap<String, Vec<bool>>,
    /// #838 (option C, §3.4.1): per-method param mutability (`!` markers),
    /// by position, keyed `(model name, method name)` — the same role
    /// `fn_mut_params` plays for a free fn, kept separate because a model
    /// method's `FnSig` (`env.models[..].methods`) carries only param
    /// *types*, and two models may declare a method of the same name.
    /// Position 0 is the method's own `self`, mirroring the receiver
    /// `check_postfix` prepends to `arg_tys` for a method call.
    method_mut_params: HashMap<(String, String), Vec<bool>>,
    /// #442 (MEMORY §3.1): the lexically enclosing arena blocks, innermost
    /// last. Empty = the default Forge context (MEMORY §3 rule 3).
    arena_stack: Vec<ArenaKind>,
    /// #674 (the arena-budget design's rule for entered budgets):
    /// the lexically enclosing `forge.enter(b) { … }` extents, innermost
    /// last. `forge.enter` selects a *budget*, not an arena (`SPEC.md §6.3`),
    /// so this is a SEPARATE stack from `arena_stack` and must never be
    /// pushed an `ArenaKind` — pushing `ArenaKind::Forge` there would
    /// silently re-open #442 for a Vault write nested inside an entered
    /// extent (probes `a1`/`a2`). Each entry is the `enter`'s own operand
    /// span, kept for a future diagnostic rather than a real runtime
    /// descriptor — no budget TABLE exists at check time (`--forge` is a
    /// runtime flag, §2.3); R3 (`budget-arena`) fires on this stack being
    /// non-empty, the same lexical test #442 uses on `arena_stack`.
    budget_stack: Vec<Span>,
    /// #674 slice 3a: set only while checking the CTOR half of an
    /// `Expr::AllocIn` node, to the EXACT span of that ctor-spine node (not a
    /// bare `bool`, since `check_expr(ctor)` recurses through everything
    /// underneath it, including any index/shape argument that could itself
    /// contain an unrelated `.kv` spine — a `bool` held for the whole
    /// recursion would wrongly suppress the ambient check for THAT nested
    /// node too). Suppresses the AMBIENT (`budget_stack`-based) instance of
    /// R3(i) inside `check_postfix`'s `Index` arm, but ONLY when the node
    /// being checked there is this exact one (span equality) — without
    /// this, `forge.kv[…] in b` lexically inside `forge.enter(b) { … }`
    /// double-reports: once from `AllocIn`'s own unconditional check (which
    /// does not depend on `budget_stack`) and once more when that same
    /// `check_expr(ctor)` call recurses into the identical `Index` node.
    checking_alloc_in_ctor: Option<Span>,
    /// #474: every model name declared in this program, hoisted ahead of
    /// signature registration. `resolve_type` needs to know a name is a model
    /// before `env.models` has it — `Outer`'s `!surf: Inner[H, W]` field is
    /// resolved while registering `Outer`, which may precede `Inner`.
    model_names: std::collections::HashSet<String>,
    /// #474: set while typing the callee of a call, and consumed by the one
    /// postfix level it applies to. A shape bracket between a method name and
    /// its arguments (`b.blit![2, 2](src)`) is legal only there; the same
    /// bracket standing on its own is a reference to a method, which is not a
    /// value this language has.
    typing_callee: bool,
    /// PORTS.md §5: the innermost construct forbidding port calls, if any —
    /// a `@grad fn` body (#497), a `@fuse` block, or a `@deterministic` block,
    /// closures they enclose included. `port_open`/`port_call`/`port_close`
    /// are rejected here with a `port-forbidden` diagnostic. Outermost wins:
    /// the diagnostic names the construct the call first escaped.
    port_ban: Option<PortBan>,
    /// #578: the names declared `extern fn`, so a call to one can be rejected
    /// in the three constructs the spec's `extern fn` rules forbid — the same
    /// three `port_ban` already tracks, for the same reason: a foreign call is
    /// an effect boundary. `env.functions` cannot answer this; it holds every
    /// signature under one key space.
    extern_fns: std::collections::HashSet<String>,
    pub checked_modules: HashMap<PathBuf, ModuleEnv>,
    /// #724: name → the `use` path string as written, for every top-level
    /// `let` a directly (unaliased) `use`d module declares. `SPEC.md §6.7`:
    /// a top-level `let` never crosses a module boundary — only `pub fn`,
    /// `pub model` and `pub type` do — so `process_imports` never merges
    /// these into `env`. Recorded here anyway so a bare read of one of these
    /// names in the importer earns the specific accessor-rule diagnostic
    /// (`` `CAP` is a top-level `let` of `util.dmc` `` — `check_expr`'s
    /// `Expr::Ident` arm) rather than a plain `undefined identifier`.
    /// `pub(crate)`, not private: D4 threads this out into `ModuleEnv`
    /// (`transitive_import_lets`) once this file finishes checking, so an
    /// importer of THIS file can name the original declaring module for a
    /// `let` this file only saw through one of its own unaliased imports.
    pub(crate) import_private_lets: HashMap<String, String>,
    /// Same, for a `use "..." as alias` import — keyed by `(alias, name)` so
    /// `alias.CAP` earns the same diagnostic at the qualified-field-access
    /// site (`PostfixOp::Field` on a `TyType::Module`).
    import_private_lets_qualified: HashMap<(String, String), String>,
    /// Demon mode: the Control Art Restriction released. When set, the
    /// safe-mode lint family (`warn()`) is suppressed entirely — raw, full
    /// speed, no guardrails (issue #196). Hard type errors still fire.
    pub demon: bool,
    /// #463: the file-size lint dial — `lints.max_file_lines` from the
    /// nearest `demoni.json` (PACKAGES.md §4). `None` means unconfigured,
    /// and unconfigured means the lint does not fire: the threshold is a
    /// per-project dial, not a number the compiler picks. Resolved from the
    /// source path in `check_program` unless a caller (tests) set it first.
    pub max_file_lines: Option<usize>,
    /// #785: spans already reported by `check_tensor_element_scalar`
    /// (`(start, end, line, col)` of the offending element's own AST node).
    /// `resolve_type` runs more than once over the same written annotation —
    /// a fn's params/return resolve once in `register_fn` and again in
    /// `check_fn`, an aliased annotation resolves through `resolve_type_
    /// alias` at every use site — so without this a single bad `Tensor[Buf,
    /// [4]]` would print the same error two or more times. Keyed by span
    /// rather than suppressed some other way so two DIFFERENT bad
    /// annotations (even of the same offending model) each still get their
    /// own diagnostic.
    reported_element_type_spans: std::collections::HashSet<(usize, usize, usize, usize)>,
    /// #799: `(name, span)` pairs already reported by `resolve_dim_at`'s
    /// mutable-top-level-`let`-as-dimension diagnostic. `resolve_dim_at`
    /// runs once per dimension *slot* at a shape annotation, but a MODEL
    /// method's shape-carrying param (`Tensor[f32,[N]]` etc.) is itself
    /// re-resolved once per call site by `resolve_dim_at`'s own caller —
    /// same span, same offending name, one real mistake — so without this a
    /// single `let !N = 4` used in one method param would print the
    /// identical diagnostic once per call to that method. Keyed by name
    /// alongside the span (not span alone) so two DIFFERENT mutable names
    /// that happen to land on the same span (an arithmetic dimension like
    /// `[N + M]` with both `N` and `M` mutable) still each get reported.
    reported_mutable_dim_spans: std::collections::HashSet<(String, usize, usize, usize, usize)>,
}

impl Checker {
    pub fn new() -> Self {
        Self {
            env: Env::new(),
            aliases: HashMap::new(),
            resolved_aliases: HashMap::new(),
            const_dims: HashMap::new(),
            all_top_level_let_names: std::collections::HashSet::new(),
            mutable_top_level_lets: std::collections::HashSet::new(),
            resolving_aliases: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            current_fn_ret: None,
            current_fn_name: None,
            current_model_name: None,
            current_fn_shape_params: Vec::new(),
            grad_fn_counts: HashMap::new(),
            fn_body_refs: HashMap::new(),
            method_body_refs: HashMap::new(),
            fn_mut_params: HashMap::new(),
            method_mut_params: HashMap::new(),
            arena_stack: Vec::new(),
            budget_stack: Vec::new(),
            checking_alloc_in_ctor: None,
            model_names: std::collections::HashSet::new(),
            typing_callee: false,
            port_ban: None,
            extern_fns: std::collections::HashSet::new(),
            checked_modules: HashMap::new(),
            import_private_lets: HashMap::new(),
            import_private_lets_qualified: HashMap::new(),
            demon: false,
            max_file_lines: None,
            reported_element_type_spans: std::collections::HashSet::new(),
            reported_mutable_dim_spans: std::collections::HashSet::new(),
        }
    }

    /// #474: is `name` a model type? Local declarations are hoisted into
    /// `model_names` before any signature is resolved; imported ones are
    /// already in `env.models` by then (`process_imports` runs first).
    fn is_model_name(&self, name: &str) -> bool {
        self.model_names.contains(name) || self.env.models.contains_key(name)
    }

    /// #728 (ratified
    /// 2026-09-09): true when `ty` is a bare `Named` type naming a model that
    /// declares at least one shape parameter, with no argument list of its
    /// own — the spelling `SPEC.md §3.10` already says "an empty argument
    /// list is never a wildcard" about. A model with no shape parameters is
    /// unaffected: its bare name is its only legal spelling, so this never
    /// matches one.
    ///
    /// Only ever consulted from pass 2 (`check_fn`/`check_let`/`check_expr`),
    /// after pass 1 has registered every model in the file — `env.models` is
    /// complete regardless of declaration order by then, unlike during
    /// registration itself (see `model_names`'s own doc comment, and
    /// `register_model`, which resolves field types before every model in
    /// the file is necessarily registered — the reason the field half of
    /// this lint runs as its own post-registration pass instead of inline
    /// there).
    fn bare_generic_model<'a>(&'a self, name: &str, args_empty: bool) -> Option<&'a [String]> {
        if !args_empty { return None; }
        let info = self.env.models.get(name)?;
        if info.shape_params.is_empty() { return None; }
        Some(&info.shape_params)
    }

    /// #728: describes the innermost function or method currently being
    /// checked, for a diagnostic naming it — `None` at module scope (a
    /// top-level `let` or allocator call outside any `fn`).
    fn current_fn_desc(&self) -> Option<String> {
        let fname = self.current_fn_name.as_ref()?;
        Some(match &self.current_model_name {
            Some(m) => format!("method `{fname}` of model `{m}`"),
            None => format!("function `{fname}`"),
        })
    }

    /// #728 rework (reviewer defect 1): the hint text for the type-arg-less
    /// generic-model warning, tailored to the annotation position instead of
    /// one "write it explicitly" line for every site.
    ///
    /// `ty` is the WHOLE annotation (a param's declared type, a field's,
    /// …) and `match_span` is the specific bare-model occurrence's own span
    /// within it — which may be `ty` itself (`b: Buf`) or nested inside an
    /// array/tuple/fn-type wrapper found by `find_bare_generic_models`
    /// (`arr: [Buf; 4]`). The hint is built by re-rendering `ty` with just
    /// that one occurrence bracketed (`render_type_for_hint`), so a wrapper
    /// is never silently dropped from the suggested spelling — an earlier
    /// version of this hint said `arr: Surface[H, W]` for a `[Surface; 4]`
    /// param, which is not the parameter's type at all.
    ///
    /// `declared` is the shape-parameter names already in scope at the
    /// annotation: the enclosing fn's own `current_fn_shape_params` for a
    /// parameter, return, `let` binding, allocator or alias body, or the
    /// enclosing model's own `shape_params` for a field (model-level shape
    /// params ARE the declaring scope for a field's own type, unlike inside
    /// a method body — see `current_fn_shape_params`'s doc comment). When
    /// every one of `params` is already a name in `declared`, the model's
    /// own shape params can be spelled directly (`Buf[N]`) — the escape
    /// hatch the `type_arg_less_generic_model_silent_with_explicit_brackets`
    /// test pins (`fn cap[N](b: Buf[N])`).
    ///
    /// `param_site` is `Some((fn_name, param_name))` only in parameter
    /// position — the one spot the reviewer's audit found "write it
    /// explicitly" is actively wrong advice: `-> Buf[N]`, `let x: Buf[N]`,
    /// `inner: Buf[N]` and `forge.uninit[Surface[H,W],[4]]` are hard type
    /// errors (`undefined identifier N`/`H`) naming a shape variable with
    /// nothing binding it, so a reader who follows the hint hits a second
    /// error immediately — but `fn cap(b: Buf[N])` (no matching `[N]` on the
    /// function itself) *check-passes*, silencing this lint forever, and the
    /// JIT refuses the resulting call identically to the un-bracketed form
    /// either way. The only spelling that is both accepted by `--check` and
    /// lowered by `dmc jit` is declaring the shape parameter on the function
    /// itself: `fn cap[N](b: Buf[N])` (verified below and pinned by
    /// `type_arg_less_hint_names_the_blessed_spelling_in_param_position`).
    ///
    /// The "declare it on the function" and "reuse what's already declared"
    /// branches below fire ONLY for a DIRECT match — `match_span` equal to
    /// `ty`'s own span, i.e. the annotation IS the bare model (`b: Buf`),
    /// not one nested inside an array/tuple/fn-type wrapper
    /// (`find_bare_generic_models`'s other positions, defect 2). Verified by
    /// probe, not assumed: `fn f[H, W](arr: [Surface[H, W]; 4])` and
    /// `fn cap[N](p: (Buf[N], i64))` both `--check` clean, but calling either
    /// with a real argument fails at RUNTIME — `` shape param `H`/`N` cannot
    /// be inferred from any argument `` — because shape-param inference
    /// only ever looks at a DIRECT model/tensor-typed parameter, never one
    /// nested inside a wrapper type (the same gap the ratified decision's
    /// own §8 tracks separately as #739, "a model-array parameter's shape
    /// param binds on neither backend" — confirmed here to extend to a
    /// tuple- and fn-type-nested parameter too, not just an array one). A
    /// CONCRETE bracket at a nested position carries no such risk (verified
    /// the same way: `[Surface[4, 4]; 4]`/`(Buf[4], i64)` both check AND run
    /// correctly, no inference needed since nothing is left to infer), so a
    /// nested match always gets that hint regardless of position — including
    /// in parameter position, where it would otherwise get the "declare on
    /// the function" advice.
    fn type_arg_less_hint(
        &self,
        ty: &Type,
        model: &str,
        params: &[String],
        match_span: &Span,
        declared: &[String],
        param_site: Option<(&str, &str)>,
        siblings: &[(String, Vec<String>, Span)],
    ) -> String {
        let is_direct = matches!(ty, Type::Named { span, .. } if span.start == match_span.start);
        let already_in_scope = is_direct && params.iter().all(|p| declared.iter().any(|d| d == p));
        let declare_on_fn = is_direct && !already_in_scope && param_site.is_some();
        let bracket_names: Vec<String> = if already_in_scope || declare_on_fn {
            // Reusing what's already in scope, or (in direct param position)
            // the model's own shape-param names, freshly declared on the
            // function below — either way the bracket names `params` itself.
            params.to_vec()
        } else {
            // A nested match, or a direct match with nothing already in
            // scope and no function to declare on: a concrete bracket is a
            // legal, complete spelling on its own — and, at a nested
            // position, the ONLY verified-safe one (see this fn's own doc
            // comment).
            params.iter().map(|_| "4".to_string()).collect()
        };
        let replacement = format!("{model}[{}]", bracket_names.join(", "));
        // An annotation can carry more than one bare-model occurrence (`y:
        // ([Buf; 2], (Buf, i64))`) — each gets its own warning, one per
        // `find_bare_generic_models` match. If each warning's hint bracketed
        // only ITS OWN occurrence, neither hint alone would silence the
        // lint: applying it leaves the sibling occurrence still bare, which
        // still warns. So every OTHER match found in this same `ty` gets
        // bracketed too, with the same concrete placeholder a nested match
        // gets on its own (verified safe at any nested position — see this
        // fn's own doc comment) — bracketing every match in the rendered
        // spelling so that applying any ONE of the annotation's hints fixes
        // the whole annotation and converges to zero warnings in one edit.
        let mut replacements: Vec<(usize, String)> = vec![(match_span.start, replacement)];
        for (sib_model, sib_params, sib_span) in siblings {
            if sib_span.start == match_span.start { continue; }
            let sib_bracket = sib_params.iter().map(|_| "4".to_string()).collect::<Vec<_>>().join(", ");
            replacements.push((sib_span.start, format!("{sib_model}[{sib_bracket}]")));
        }
        let rendered = render_type_for_hint(ty, &replacements);
        if already_in_scope {
            return format!("write the shape arguments explicitly: `{rendered}`");
        }
        if declare_on_fn {
            let (fname, pname) = param_site.expect("declare_on_fn implies param_site is Some");
            // The function's own shape-param list, plus whichever of the
            // model's shape params it doesn't already declare, appended —
            // so a function that already declares an unrelated shape param
            // (`fn f[M](...)`) gets `fn f[M, N](...)`, not a conflicting
            // redeclaration of `M`.
            let merged: Vec<&str> = declared.iter().map(String::as_str)
                .chain(params.iter().filter(|p| !declared.contains(p)).map(String::as_str))
                .collect();
            return format!(
                "declare the shape parameter{} on the function: `fn {fname}[{}]({pname}: {rendered})`",
                if params.len() == 1 { "" } else { "s" },
                merged.join(", "),
            );
        }
        format!("write a concrete shape argument: `{rendered}`")
    }

    /// #728: the warning itself. `site` names the annotation position
    /// ("parameter `b` of function `cap`", "field `t` of model `Buf`", …).
    /// A warning at `0.0.5-draft`; becomes a compile-time error at
    /// `0.0.6-draft` (the ratified decision's follow-up PR (4), issue #728).
    ///
    /// #728 rework: `ty`, `declared` and `param_site` steer the hint text —
    /// see `type_arg_less_hint`. `span` is the specific bare-model
    /// occurrence's own span (may be nested inside `ty`); it is both the
    /// diagnostic's span and `type_arg_less_hint`'s substitution target.
    /// `siblings` is every OTHER bare-model match found in this same `ty`
    /// (i.e. `find_bare_generic_models`'s full result minus this one) — so
    /// the rendered hint can bracket them too (see `type_arg_less_hint`).
    fn warn_type_arg_less_generic_model(
        &mut self,
        ty: &Type,
        model: &str,
        params: &[String],
        site: String,
        span: Span,
        declared: &[String],
        param_site: Option<(&str, &str)>,
        siblings: &[(String, Vec<String>, Span)],
    ) {
        let hint = self.type_arg_less_hint(ty, model, params, &span, declared, param_site, siblings);
        self.warn(
            format!(
                "bare-generic-model: {site} names `{model}` with no argument list — \
                 `{model}` is shape-parameterized, and an empty argument list is never \
                 a wildcard (SPEC.md §3.10); this warning becomes a compile-time error \
                 at 0.0.6-draft"
            ),
            span,
            Some(hint),
        );
    }

    /// #728 rework (reviewer defect 2): walk `ty` looking for every bare
    /// generic-model occurrence, not just a top-level `Named` type —
    /// recursing into a fixed-size array's element type (`[Surface; 4]`,
    /// SPEC.md §3.10 calls this a field/parameter type same as any other),
    /// a tuple's element types, and a fn-type's param/return types. That is
    /// every arm `resolve_type` (below, its `Type::Array`/`Type::Tuple`/
    /// `Type::Fn`/`Type::Tensor`/`Type::View`/`Type::KV`/`Type::RawPtr`
    /// arms) itself recurses into, except two: `RawPtr` and, as of #785,
    /// `Tensor`/`View`/`KV` (see below). Verified escapes at both param and
    /// field position before this fix (probe d11, `[Surface; 4]` as a param
    /// type; the equivalent field case), and by direct probe for a tuple
    /// element type and a fn-type parameter (a model named bare inside
    /// `(Buf, i64)` or `fn(Buf) -> i64` check-passed identically, with no
    /// warning, before this rework).
    ///
    /// `Tensor`/`View`/`KV`'s own element type WAS believed already refused
    /// as a hard type error, on the theory that the checker's tensor-literal
    /// typing (`ctor_tensor_ty`) requires a *scalar* element name there —
    /// but that function only reads a *constructor call*'s own element-type
    /// argument (`forge.zeros[f32, […]]`); it had no bearing on a `Tensor`/
    /// `View`/`KV` written as a param, field, return or binding *annotation*.
    /// Measured false: `Tensor[Buf, [4]]` (also `View[Buf, [4]]`,
    /// `KV[Buf, [4]]`) as a param or field type parsed and `--check`-passed
    /// silently, with no warning and no error — filed and fixed as #785.
    /// An interim revision of this lint (between #785's filing and its own
    /// fix landing) recursed into these three arms too, on the premise that
    /// #785 was still open and a bare generic model there was the ONLY
    /// diagnostic a reader would get. That premise is gone now that
    /// `resolve_type`'s own `Tensor`/`View`/`KV` arms hard-refuse ANY
    /// non-scalar element (`check_tensor_element_scalar`) — bracketed or
    /// not, model or enum or nested tensor — so recursing here again would
    /// print the #785 hard error and this #728 warning for the identical
    /// span (verified: `fn f(t: Tensor[Surface, [4]])` for a shape-
    /// parameterized `Surface` printed both before this revision). Pulled
    /// back out, exactly the same call `RawPtr` already made below, for the
    /// same reason: a stricter, more specific diagnostic already covers this
    /// position, so this lint has nothing left to add there.
    ///
    /// `RawPtr`'s inner type is the other nested position deliberately
    /// unwalked: it is the `extern fn` boundary (§3.12), and
    /// `check_extern_boundary`'s `extern_boundary_ty_ok` refuses any `*T`
    /// whose `T` isn't a scalar (or `nil`) as a hard type error
    /// unconditionally (verified: `extern fn f(b: *Buf)` for a model `Buf`
    /// is refused as `extern-boundary`) — a model there is already refused
    /// before this lint would ever see it. `Mesh` has no nested `Type` at
    /// all (`Vec<MeshAxis>`, each just a name and a size expr).
    fn find_bare_generic_models(&self, ty: &Type, out: &mut Vec<(String, Vec<String>, Span)>) {
        match ty {
            Type::Named { name, args, span } => {
                if let Some(params) = self.bare_generic_model(name, args.is_empty()) {
                    out.push((name.clone(), params.to_vec(), span.clone()));
                }
            }
            Type::Array(inner, _, _) => self.find_bare_generic_models(inner, out),
            Type::Tuple(elems, _) => {
                for e in elems { self.find_bare_generic_models(e, out); }
            }
            Type::Fn(params, ret, _) => {
                for p in params { self.find_bare_generic_models(p, out); }
                self.find_bare_generic_models(ret, out);
            }
            // #785: NOT recursed into — `resolve_type`'s own arms for these
            // three now hard-refuse any non-scalar element (including a bare
            // generic model), same reasoning as `RawPtr` just below.
            Type::Tensor(..) | Type::View(..) | Type::KV(..) => {}
            Type::Scalar(..) | Type::Mesh(..) | Type::RawPtr(..) => {}
        }
    }

    /// #728 rework: `find_bare_generic_models` on `ty`, then warn once per
    /// match found — all with the same `site`/`declared`/`param_site`, since
    /// those describe the *annotation* (the parameter, the field, …), not
    /// which nesting depth inside it the match was found at. Each warning
    /// also gets the full match list as its `siblings`, so a multi-match
    /// annotation's hints bracket every match, not just their own (see
    /// `type_arg_less_hint`).
    fn lint_bare_generic_models_in(
        &mut self,
        ty: &Type,
        site: &str,
        declared: &[String],
        param_site: Option<(&str, &str)>,
    ) {
        let mut matches = Vec::new();
        self.find_bare_generic_models(ty, &mut matches);
        for (model, params, span) in matches.clone() {
            self.warn_type_arg_less_generic_model(ty, &model, &params, site.to_string(), span, declared, param_site, &matches);
        }
    }

    pub fn check_program(&mut self, program: &Program, path: Option<&Path>) {
        self.process_imports(program, path);
        self.pass1_collect_signatures(program);
        self.pass2_check_types(program);
        self.pass3_arena_coherence(program);
        self.lint_writeback(program);
        // #728: a model field's type is resolved once, during registration
        // (pass 1) — `check_model` never revisits it — so this runs as its
        // own pass, after pass 1 has finished registering every model in the
        // file. Doing the check inline in `register_model` would miss a
        // forward-referenced model's shape parameter count (`model_names`
        // is hoisted ahead of registration; `env.models`'s `ModelInfo` is
        // not).
        self.lint_type_arg_less_generic_model_fields(program);
        // #728 rework (reviewer defect 5): the alias hatch — `type B = Buf`
        // then `fn cap(b: B)` — carried no warning even after the field
        // fix, because `B`'s own body is never itself a field, param,
        // return, or `let` annotation; it's a fifth position the ratified
        // decision's principle covers all the same ("every position [a
        // shape-generic model] can be written" — SPEC.md §3.10, extended by
        // #728 §8 to "every annotation position"), so a bare-model alias
        // body warns here rather than at every one of its (unbounded) call
        // sites. `type S = Buf[4]` (probe g34) stays silent — the blessed
        // hatch a bracketed alias body already is.
        self.lint_type_arg_less_generic_model_alias_bodies(program);
        // #463: resolve the file-size dial from the nearest demoni.json,
        // unless a caller pinned it already (tests, future CLI overrides).
        if self.max_file_lines.is_none() {
            if let Some(p) = path {
                self.max_file_lines = crate::manifest::max_file_lines_for(p);
            }
        }
        self.lint_file_size(program);
        // #728 rework (reviewer defect 4): `lint_type_arg_less_generic_
        // model_fields` and `lint_type_arg_less_generic_model_alias_bodies`
        // both run as trailing passes after pass 2 (see their own doc
        // comments for why — a model/alias's body is resolved once, during
        // registration, and a forward reference isn't known to be a model
        // yet at that point), so a field or alias warning always lands after
        // every param/return/let/allocator warning from the whole file,
        // regardless of which line it's actually on. Stable-sort by source
        // position so the file's own line order is what a reader sees,
        // matching every other diagnostic stream in this compiler (`--json`
        // and the plain-text printer alike just iterate `warnings` in
        // order — see `print_warnings`, `main.rs`) and what
        // `type_arg_less_generic_model_field_before_later_param_warns_in_
        // source_order` pins. Stable so two warnings that happen to share a
        // span (none of this lint's do, in practice) keep their relative
        // pass order rather than depending on sort implementation details.
        self.warnings.sort_by_key(|w| (w.span.start, w.span.line, w.span.col));
    }

    /// #463: safe-mode file-size lint. Warns — never errors — when a source
    /// file exceeds the per-project dial `lints.max_file_lines` from the
    /// nearest `demoni.json` (PACKAGES.md §4). Off unless configured: the
    /// threshold is a dial each codebase sets for itself, not a number the
    /// compiler picks. A lint, not a parser cap: a hard limit real programs
    /// could hit would make the two implementations disagree and break the
    /// differential gate that holds the interpreter and JIT to the same
    /// parsed AST. Released by
    /// `--demon` like the rest of the family, via `warn()`.
    fn lint_file_size(&mut self, program: &Program) {
        let Some(max) = self.max_file_lines else { return };
        let lines = program.source_lines;
        if lines <= max { return; }
        // Anchor the diagnostic at the end of the file — the excess is there,
        // not at line 1.
        let span = Span {
            start: program.span.end,
            end: program.span.end,
            line: lines,
            col: 1,
        };
        self.warn(
            format!("file is {lines} lines, over the {max}-line limit set by demoni.json (lints.max_file_lines)"),
            span,
            Some("split the file — a source a model reads whole is a source it edits correctly (SPEC axiom 5)".to_string()),
        );
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

    /// #728: warn on
    /// a model field annotated with a shape-parameterized model's bare name
    /// (`inner: Buf` on `model Buf[N] { .. }`) — one of the ruling's four
    /// annotation positions (the others are the parameter, return and
    /// binding positions, checked in `check_fn`/`check_let`; the allocator's
    /// own model type argument is the sixth, checked in `check_expr`).
    fn lint_type_arg_less_generic_model_fields(&mut self, program: &Program) {
        for item in &program.items {
            self.lint_type_arg_less_generic_model_fields_item(item);
        }
    }

    fn lint_type_arg_less_generic_model_fields_item(&mut self, item: &Item) {
        match item {
            Item::Model(m) => {
                // #728 rework: the model's own shape params ARE the
                // declaring scope for a field's type (`model Outer[K] {
                // inner: Buf[K] }` is exactly how a field spells "reuse the
                // enclosing model's own shape param" — unlike inside a
                // method body, where `check_model`'s "not automatically in
                // scope" diagnostic means only the method's own
                // `shape_params` count).
                let declared: Vec<String> = m.shape_params.iter().map(|sp| sp.name.clone()).collect();
                for member in &m.members {
                    if let ModelMember::Field { name, ty, .. } = member {
                        let site = format!("field `{name}` of model `{}`", m.name);
                        self.lint_bare_generic_models_in(ty, &site, &declared, None);
                    }
                }
            }
            Item::Directive { inner, .. } => self.lint_type_arg_less_generic_model_fields_item(inner),
            Item::Pub(inner) => self.lint_type_arg_less_generic_model_fields_item(inner),
            _ => {}
        }
    }

    /// #728 rework (reviewer defect 5): warn on a `type` alias whose body is
    /// itself a bare generic-model annotation (`type B = Buf`, probe g35) —
    /// the alias-hatch escape the field fix alone doesn't close, since
    /// `fn cap(b: B)` never re-states `Buf` at all; the ratified decision's
    /// principle ("an empty argument list is never a wildcard", extended to
    /// "every annotation position" by #728 §8) covers the alias body as a
    /// sixth position for the same reason it covers the other five: the
    /// bare name is still meaningless there, wildcard or not. A bracketed
    /// alias body (`type S = Buf[4]`, probe g34) stays silent — this is
    /// exactly the hatch #728's design brief called out (§4/g34) as the
    /// blessed way to pin a geometry once and reuse it, not an escape.
    ///
    /// Runs as its own trailing pass for the same reason the field lint
    /// does: an alias's body is resolved once, during `pass1`'s alias
    /// registration (`collect_alias_item`), which may run before every
    /// model in the file is itself registered.
    fn lint_type_arg_less_generic_model_alias_bodies(&mut self, program: &Program) {
        for item in &program.items {
            self.lint_type_arg_less_generic_model_alias_bodies_item(item);
        }
    }

    fn lint_type_arg_less_generic_model_alias_bodies_item(&mut self, item: &Item) {
        match item {
            Item::TypeAlias(alias) => {
                // The alias's own shape params (`type Wrap[N] = Buf[N]`) are
                // the declaring scope for its body, same reasoning as a
                // model's own shape params for a field.
                let declared: Vec<String> = alias.shape_params.iter().map(|sp| sp.name.clone()).collect();
                let site = format!("the body of type alias `{}`", alias.name);
                self.lint_bare_generic_models_in(&alias.ty, &site, &declared, None);
            }
            Item::Directive { inner, .. } => self.lint_type_arg_less_generic_model_alias_bodies_item(inner),
            Item::Pub(inner) => self.lint_type_arg_less_generic_model_alias_bodies_item(inner),
            _ => {}
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
                        // #724: `SPEC.md §6.7` — a top-level `let` never
                        // crosses a module boundary, so these names are never
                        // merged into `self.env`/`self.aliases` below. Captured
                        // here, before either branch below moves pieces of
                        // `imported_env`, purely so a stray read of one of
                        // them in this file earns the accessor-rule
                        // diagnostic instead of a bare `undefined identifier`.
                        let import_let_names = imported_env.top_level_let_names.clone();
                        if let Some(alias) = &us.alias {
                            self.env.bind(alias.clone(), TyType::Module { alias: alias.clone(), path: canonical_path.clone() });
                            for name in &import_let_names {
                                self.import_private_lets_qualified
                                    .entry((alias.clone(), name.clone()))
                                    .or_insert_with(|| us.path.clone());
                            }

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
                                let qualified_name = format!("{}.{}", alias, name);
                                // #766: `resolve_alias_body` caches by
                                // `alias.name` (`self.resolved_aliases.get(
                                // &alias.name)`) — if a qualified type
                                // reference (`d.Row`) ever parses (it does
                                // NOT today: `parse_type`'s generic-named-
                                // type arm consumes a single ident token and
                                // never a following `.`, so `self.aliases`
                                // is only ever looked up by a bare name —
                                // this whole branch is latent), a
                                // `TypeAlias` stored under the qualified key
                                // here but still carrying its original bare
                                // `.name` ("Row") would cache-miss on
                                // `resolved_aliases.get("Row")`, silently
                                // re-derive the alias's shape against THIS
                                // file's own `const_dims` (wrong — the
                                // declaring module's, not this one's, is
                                // what the already-resolved value below is
                                // for), and cache that wrong re-derivation
                                // under the bare "Row" — poisoning a
                                // same-named unqualified alias if this file
                                // ever declares or imports one too.
                                // Renaming the stored alias's own `.name` to
                                // the qualified key keeps the two maps' keys
                                // in lockstep, so the cache can only ever
                                // hit correctly whenever this does become
                                // reachable.
                                let mut qualified_alias = type_alias.clone();
                                qualified_alias.name = qualified_name.clone();
                                self.aliases.insert(qualified_name.clone(), qualified_alias);
                                // #761: the alias's dimensions, already resolved
                                // against the DECLARING module's own top-level
                                // `let` constants (which this checker instance
                                // cannot see — SPEC.md §6.7 / #724 / #731) —
                                // carried across the boundary as the alias's
                                // already-resolved shape, the same way a
                                // model's resolved field types are prefixed
                                // just above.
                                if let Some(resolved) = imported_env.resolved_aliases.get(name) {
                                    self.resolved_aliases.insert(
                                        qualified_name,
                                        prefix_types_in_ty(resolved.clone(), alias, &imported_model_names),
                                    );
                                }
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
                                    // #761: carry the alias's already-resolved
                                    // dimensions across the boundary too — see
                                    // the aliased-import branch above.
                                    if let Some(resolved) = imported_env.resolved_aliases.get(&name) {
                                        self.resolved_aliases.insert(name.clone(), resolved.clone());
                                    }
                                    self.aliases.insert(name, type_alias);
                                }
                            }
                            for name in &import_let_names {
                                self.import_private_lets
                                    .entry(name.clone())
                                    .or_insert_with(|| us.path.clone());
                            }
                            // D4: also inherit whatever `imported_env` itself
                            // could already name a module for — this file's
                            // own `top_level_let_names` insert above wins on
                            // a name clash (`or_insert_with` on an already-
                            // populated entry is a no-op), so a name this
                            // file re-declares still reports against ITS
                            // OWN file first.
                            for (name, module) in &imported_env.transitive_import_lets {
                                self.import_private_lets
                                    .entry(name.clone())
                                    .or_insert_with(|| module.clone());
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
        // #761: the const-dim table has to exist before anything resolves a
        // type — including the alias-caching loop right below — so it comes
        // first, as its own order-independent pre-pass (a forward-referenced
        // constant must resolve exactly like a backward one).
        self.collect_const_dims(program);
        for item in &program.items {
            self.collect_alias_item(item);
        }
        // #761: resolve every alias's dimensions now, against this module's
        // own `const_dims` and its own shape parameters, and cache the
        // result. Separate from the loop above: an alias may reference
        // another alias declared later in the file, and `collect_alias_item`
        // has to finish hoisting every name (mirroring how enums/models are
        // hoisted before anything resolves against them) before any of them
        // can be resolved.
        for item in &program.items {
            self.cache_resolved_alias_item(item);
        }
        for item in &program.items {
            self.register_item_signatures(item);
        }
    }

    /// #761: collect every module-level `let NAME = <int literal>` (optionally
    /// negated) into `const_dims`, the way `collect_alias_item` hoists alias
    /// names — order-independent, and before anything is resolved against it.
    /// A mutable top-level binding (`let !`/`let mut`) is not a constant and
    /// is left out; so is anything whose value isn't (transitively, see
    /// below) a literal int, since that's the same shape the JIT's own
    /// `ConstDim` table restricts to (#620) and is exactly what makes a name
    /// usable as a dimension rather than an arbitrary (and here,
    /// unevaluated) expression.
    ///
    /// #766: a top-level `let` whose initializer is itself a bare name of
    /// another top-level literal constant (`let A = 4  let B = A`) folds
    /// too — `dmc run` already treats `B` as `4` (the JIT's `ConstDim` table
    /// folds a name-to-name chain the same way), so a program like this
    /// being check-clean on `main` must not become a hard check error just
    /// because this collector previously only ever looked at literal *int*
    /// initializers. The fold has to be order-independent, exactly like the
    /// direct-literal case (a forward reference is as valid as a backward
    /// one) — so this collects every name-chain candidate in one pass over
    /// the program, then closes over `const_dims` to a fixed point: each
    /// full pass that resolves at least one more candidate against what
    /// `const_dims` now holds repeats, and a pass that resolves nothing
    /// means every remaining candidate's source is neither a literal nor
    /// (transitively) a name that bottoms out in one.
    fn collect_const_dims(&mut self, program: &Program) {
        let mut chains: Vec<(String, String)> = Vec::new();
        for item in &program.items {
            self.collect_const_dim_item(item, &mut chains);
        }
        let mut changed = true;
        while changed {
            changed = false;
            for (name, source) in &chains {
                if self.const_dims.contains_key(name) {
                    continue;
                }
                if let Some(&v) = self.const_dims.get(source) {
                    self.const_dims.insert(name.clone(), v);
                    changed = true;
                }
            }
        }
    }

    fn collect_const_dim_item(&mut self, item: &Item, chains: &mut Vec<(String, String)>) {
        match item {
            Item::Let(l) => {
                if let Pattern::Ident(name, _) = &l.pattern {
                    // #766: recorded regardless of what kind of `let` this
                    // is — see the fields' own doc comments for why.
                    self.all_top_level_let_names.insert(name.clone());
                    if l.mutating || l.is_mut {
                        self.mutable_top_level_lets.insert(name.clone());
                        return;
                    }
                    if let Some(n) = expr_i64(&l.value) {
                        self.const_dims.insert(name.clone(), n);
                    } else if let Expr::Ident(source, _) = &l.value {
                        chains.push((name.clone(), source.clone()));
                    }
                }
            }
            Item::Directive { inner, .. } => self.collect_const_dim_item(inner, chains),
            Item::Pub(inner) => self.collect_const_dim_item(inner, chains),
            _ => {}
        }
    }

    fn collect_alias_item(&mut self, item: &Item) {
        match item {
            Item::TypeAlias(alias) => {
                // #674 slice 3a (§2.2): `Budget` is reserved — `model Budget`,
                // `enum Budget` and `type Budget = …` are name errors at
                // registration, not one of the three `budget-*` refusals.
                if self.reject_reserved_budget_name(&alias.name, &alias.span) { return; }
                self.aliases.insert(alias.name.clone(), alias.clone());
            }
            // #336: register enums in this first pass1 loop so a later
            // model field or fn signature can resolve an enum-typed annotation.
            Item::Enum(e) => {
                if self.reject_reserved_budget_name(&e.name, &e.span) { return; }
                self.register_enum(e);
            }
            // #474: the *name* only, and this early, so that resolving any
            // annotation — including a field of a model declared further up —
            // knows to read `Inner[H, W]`'s args as dims.
            Item::Model(m) => {
                if self.reject_reserved_budget_name(&m.name, &m.span) { return; }
                self.model_names.insert(m.name.clone());
            }
            Item::Directive { inner, .. } => self.collect_alias_item(inner),
            Item::Pub(inner) => self.collect_alias_item(inner),
            _ => {}
        }
    }

    /// #674 slice 3a (§2.2, §6 Q7): `Budget` is a reserved builtin type name —
    /// `model Budget { … }`, `enum Budget { … }` and `type Budget = …` are all
    /// refused here, at first registration, as a name error (probe `s6`: all
    /// three ran clean before this fix; 0 corpus files use the name). Returns
    /// `true` (and has already reported) when `name` is reserved, so the
    /// caller skips its own registration — `register_model` mirrors this
    /// check so a model named `Budget` is never inserted into `env.models`
    /// either, even though `register_model` runs in a later pass.
    fn reject_reserved_budget_name(&mut self, name: &str, span: &Span) -> bool {
        if name != "Budget" { return false; }
        self.error(
            "`Budget` is a reserved type name (#674) and cannot be redeclared \
             by a `model`, `enum` or `type` alias".to_string(),
            span.clone(),
        );
        true
    }

    /// #761: eagerly resolve+cache a `type` alias's dimensions (see
    /// `resolve_alias_body`). Recurses through `Item::Pub`/`Item::Directive`
    /// the same way `collect_alias_item` does; every other item is a no-op
    /// here (`register_item_signatures` handles the rest, afterward).
    fn cache_resolved_alias_item(&mut self, item: &Item) {
        match item {
            Item::TypeAlias(alias) => {
                self.resolve_alias_body(alias);
            }
            Item::Directive { inner, .. } => self.cache_resolved_alias_item(inner),
            Item::Pub(inner) => self.cache_resolved_alias_item(inner),
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
        // #766/#793: precedence between a fn's shape params and a top-level
        // constant of the same name. This pass only ever binds `f`'s
        // DECLARED shape params (the loop just above): an explicit
        // `fn f[N](x: Tensor[i32,[N]])` shape param always wins (`resolve_
        // dim`'s `shape_param_in_scope` check is true, so `N` stays a `Var`
        // here exactly as it does in `check_fn`) — but a name that is only
        // an IMPLICIT shape param (never declared in `[...]`, e.g. `fn f(x:
        // Tensor[f32,[S]])` with no `[S]` on `f`) is NOT bound here, so if
        // it also happens to name a top-level literal `let` constant,
        // `resolve_dim` folds it to that constant for THIS signature — the
        // one call sites actually check args against — the same way a type
        // alias's own dimensions fold (`resolve_alias_body`).
        //
        // `check_fn` applies the SAME rule to its own copy of these types
        // (its implicit-shape-param loop, just above the analogous point in
        // that fn) — as of #793 it skips binding an implicit name that is
        // also a top-level constant, so it folds there too instead of
        // staying a free `Var`. Before #793 the two passes disagreed here:
        // `register_fn` folded the constant into the registered signature
        // while `check_fn` left the body's copy of the same type as `[N]`,
        // so a value produced by one folded signature (e.g. another fn's
        // return type, or `forge.zeros[f32,[N]]`) failed to check against a
        // parameter of the same folded type — `fn outer(c: Tensor[f32,[N]])
        // { inner(c) }` with both `inner` and `outer` taking `Tensor[f32,
        // [N]]`. The two passes are symmetric again now; keep them that way.
        //
        // This is why `let N = 4  fn f(x: Tensor[f32,[N]])` called with an
        // `[8]` argument is a check error: `dmc run`'s own `ConstDim` table
        // folds `N` to `4` for the same reason (#620), so it refuses `[8]`
        // too — this checker was simply the more permissive of the two
        // before this fix, not the correct one. Do not add a guard here to
        // suppress the fold: that would just resurrect the check/run gap.
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
        // #398: the identifiers each fn body reads/binds, so the captured-mut
        // rule can follow calls out of a `@grad fn` body. Only the name sets are
        // kept, not the AST.
        let mut refs = collect_body_idents(&f.body);
        for p in &f.params { refs.bound.insert(p.name.clone()); }
        self.fn_body_refs.insert(f.name.clone(), refs);
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
        self.extern_fns.insert(e.name.clone());
        self.fn_mut_params.insert(
            e.name.clone(),
            e.params.iter().map(|p| p.mutating).collect(),
        );
    }

    fn register_model(&mut self, m: &ModelDecl) {
        // #674 slice 3a: the reservation error is already reported by
        // `collect_alias_item` (the earlier pass); skip re-registering
        // `Budget` as a model here so `TyType::Budget` (`resolve_type`) has
        // no `env.models` entry to be shadowed by or confused with.
        if m.name == "Budget" { return; }
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
                    self.method_mut_params.insert(
                        (m.name.clone(), f.name.clone()),
                        f.params.iter().map(|p| p.mutating).collect(),
                    );
                    methods.insert(f.name.clone(), FnSig { shape_params: m_shape, params, ret });
                    // #398: keep the method body's identifier sets alongside the
                    // plain fns', so the captured-mut call-graph walk can follow
                    // `h.contrib()` into `contrib`'s body. Registered here, in
                    // the registration pass, so the walk does not depend on
                    // whether the model is declared above or below the
                    // `@grad fn` that calls into it.
                    let mut mrefs = collect_body_idents(&f.body);
                    for p in &f.params { mrefs.bound.insert(p.name.clone()); }
                    mrefs.bound.insert("self".to_string());
                    self.method_body_refs
                        .entry(f.name.clone())
                        .or_default()
                        .push((m.name.clone(), mrefs));
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

    /// Like `check_item`, but threads whether the item is under a `pub`.
    /// `extern fn` cares (SPEC §9 makes `pub extern fn` a compile-time error)
    /// and so does `let` (#596: SPEC.md §6.7 makes `pub let` a compile-time
    /// error, since a top-level binding never crosses a module boundary).
    fn check_item_vis(&mut self, item: &Item, is_public: bool) {
        match item {
            Item::Fn(f)       => self.check_fn(f),
            Item::ExternFn(e) => {
                // SPEC.md §6.8 / §9: an `extern fn` is always exported in the
                // linkage sense (foreign symbols are a process-wide resource),
                // so the `pub` keyword on it is meaningless — and a compile-time
                // error. An extern decl has no body; what remains to check is
                // the boundary itself.
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
                self.check_extern_boundary(e);
            }
            Item::Model(m)    => self.check_model(m),
            Item::TypeAlias(_) => {}
            Item::Enum(_) => {} // registered in pass 1; nothing to check in a body
            Item::Arena(a) => {
                self.arena_stack.push(a.kind.clone());
                let _ = self.check_block(&a.body);
                self.arena_stack.pop();
            }
            Item::Let(l)   => {
                // #596: SPEC.md §6.7 exports only `pub fn` / `pub model` /
                // `pub type` — a top-level binding never crosses a module
                // boundary, `pub` or not (`process_imports` only ever pulls
                // functions and models out of an imported file's env). Before
                // this, `pub let` type-checked clean and the `pub` was a
                // silent no-op: the failure surfaced at the *importer*, as an
                // "undefined identifier", with nothing pointing back at the
                // declaration that dropped it. Reject it here instead, at the
                // declaration, so `--check` on the library file alone catches
                // it.
                if is_public {
                    // Mutability lives on the `LetStmt` (`mutating` for `!`,
                    // `is_mut` for `mut`), not on the pattern — a bare
                    // `Pattern::Ident` match never sees the `!`, so
                    // `pub let !LIMIT` rendered as `` `pub let LIMIT` ``, and
                    // a `pub let (a, b) = ...` tuple pattern isn't a
                    // `Pattern::Ident` at all, so it rendered as the bare
                    // `` `pub let` `` fallback.
                    let rendered = pattern_for_diagnostic(&l.pattern).map(|pat| {
                        let bang = if l.mutating || l.is_mut { "!" } else { "" };
                        format!(" {bang}{pat}")
                    });
                    self.error(
                        format!(
                            "pub-let: `pub let{}` does not export — bindings cross a module \
                             boundary as a `pub fn` accessor, not a `pub let` (SPEC.md §6.7)",
                            rendered.unwrap_or_default()),
                        l.span.clone(),
                    );
                }
                self.check_let(l);
            }
            Item::Use(_)   => {}
            Item::Directive { directives, inner, .. } => {
                self.check_directive_item(directives, inner, is_public);
            }
            Item::Pub(inner) => self.check_item_vis(inner, true),
        }
    }

    /// #578: enforce the spec's `extern fn` boundary-type rule. Parameter and
    /// return types are *restricted to* scalar types, raw pointer types `*T`,
    /// and `nil`; the same section names `Tensor`, `View`, `KV`, `Mesh`,
    /// `Port`, `model`, tuple and `fn` illegal there. Until this, the rule was
    /// prose only: `extern fn f(t: Tensor[f32, [4]]) -> str` checked clean and
    /// JIT-compiled clean, which at a boundary that performs no shape,
    /// alignment or aliasing check is a segfault waiting for a call site.
    ///
    /// The **positive** rule is what is enforced, not the list of examples: the
    /// first is exhaustive and the second is not. `str` is a scalar type, so it
    /// is admitted here and refused by the JIT instead (`jit-extern`) — an
    /// implementation-capability refusal belongs in the JIT subset, which
    /// `STABILITY.md §4` marks explicitly not-stable, rather than in the
    /// checker, where narrowing past the spec's own rule would be a change to
    /// shipped surface under `STABILITY.md §3`.
    fn check_extern_boundary(&mut self, e: &ExternFnDecl) {
        // A shape-generic extern decl is refused by the JIT; here the shape
        // scope only has to exist so an annotation naming a shape param
        // resolves to a dim rather than cascading an "undefined" error.
        self.env.push_shape_scope();
        for sp in &e.shape_params {
            self.env.bind_shape_param(&sp.name, None);
        }
        for p in &e.params {
            if let Some(ast_ty) = &p.ty {
                let ty = self.resolve_type(ast_ty);
                if !extern_boundary_ty_ok(&ty) {
                    self.error(
                        format!(
                            "extern-boundary: parameter `{}` of `extern fn {}` has type `{}` — \
                             an extern boundary is restricted to scalar types, raw pointers \
                             `*T`, and `nil`; a tensor crosses as its data pointer (`*f32`) \
                             plus separately passed extents (see the spec's `extern fn` rules)",
                            p.name, e.name, render_ty(&ty)),
                        p.span.clone(),
                    );
                }
            }
        }
        if let Some(ret_ty) = &e.ret_type {
            let ty = self.resolve_type(ret_ty);
            if !extern_boundary_ty_ok(&ty) {
                self.error(
                    format!(
                        "extern-boundary: `extern fn {}` returns `{}` — an extern boundary is \
                         restricted to scalar types, raw pointers `*T`, and `nil` \
                         (see the spec's `extern fn` rules)",
                        e.name, render_ty(&ty)),
                    e.span.clone(),
                );
            }
        }
        self.env.pop_shape_scope();
    }

    /// #369: directives the compiler actually acts on. Any `@ident` parses, so
    /// a documented-but-unimplemented directive (`@inplace`, `@recompute`) or
    /// a typo is otherwise a silent no-op — warn instead of quietly doing
    /// nothing. (`@host` stays effective via `@host match`.)
    ///
    /// #505 removed `@comptime` from the unimplemented set: `comptime.rs`
    /// folds it before this checker runs, so the warning would now be false —
    /// the directive has an effect, and on a folded block there is no longer a
    /// directive here to warn about at all.
    fn lint_unimplemented_directives(&mut self, directives: &[Directive]) {
        const EFFECTIVE: &[&str] =
            &["grad", "cast", "shard", "tp", "pp", "fuse", "deterministic", "host", "comptime"];
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

    /// DIRECTIVES.md §3 "illegal stacks": a stack the spec rejects outright.
    /// `stack` is the directive list written on one construct, outermost
    /// first; `nested` is every directive on the chain of lone directive
    /// constructs it wraps, so `@cast(f32) { @cast(bf16) { x } }` reads as the
    /// same stack as `@cast(f32) @cast(bf16) { x }` and is rejected the same
    /// way — and so does `@cast(f32) { @fuse { @cast(bf16) { x } } }`, where
    /// the two casts are separated by an intervening directive level.
    ///
    /// Two of the six §3 entries are enforced here (`@cast @cast`,
    /// `@fuse @fuse`); `@inplace`'s attachment rule is
    /// `check_inplace_target`, `@shard`'s is `check_sharding_directives`,
    /// `fuse-infeasible` is `check_fuse_feasible` (#503), and
    /// `comptime-non-static` is `comptime.rs`, which runs before this checker
    /// and seeds its diagnostics into `self.errors` (#505). All six are
    /// enforced.
    fn check_illegal_stack(&mut self, stack: &[Directive], nested: &[&Directive]) {
        for name in ["cast", "fuse"] {
            let mut hits = stack.iter().chain(nested.iter().copied()).filter(|d| d.name == name);
            let (Some(outer), Some(inner)) = (hits.next(), hits.next()) else { continue };
            let (msg, hint) = if name == "cast" {
                (format!(
                    "illegal directive stack: {} inside {} — the inner dtype \
                     wins, so the outer cast is dead (DIRECTIVES.md §3)",
                    render_directive(inner), render_directive(outer),
                ), format!("drop {}; a cast scope has one dtype", render_directive(outer)))
            } else {
                (format!(
                    "illegal directive stack: {} inside {} — fusion is \
                     idempotent, so the second is redundant (DIRECTIVES.md §3)",
                    render_directive(inner), render_directive(outer),
                ), "drop one `@fuse`; one fused unit is one kernel".to_string())
            };
            self.error_with_hint(msg, inner.span.clone(), Some(hint));
        }
    }

    /// SPEC.md §7.7 / DIRECTIVES.md §3: the fuse-infeasible analysis. `@fuse`
    /// promises the block lowers as one kernel with no materialized
    /// intermediates. The set the JIT's fused kernel actually collapses today
    /// is small — a single elementwise chain (`.+ .- .* ./ .^ .** .< .> .<=
    /// .>=`, `\>` ReLU) over f32 tensor operands sharing one shape, with
    /// float-scalar broadcasts (tensors do not broadcast against each
    /// other) — and the contract is about not lying: a body containing
    /// anything outside that set is refused here, at the same compile stage
    /// as the §3 stacking rejections, naming the offending op. Port calls in
    /// the block are the other half, already gated as `port-forbidden`
    /// (PORTS.md §5). The JIT's lowering keeps its own per-op refusals as a
    /// backstop; this check is what makes both backends refuse the same
    /// program at the same stage (SPEC.md §7.6).
    fn check_fuse_feasible(&mut self, directives: &[Directive], body: &Block, span: &Span) {
        if !directives.iter().any(|d| d.name == "fuse") {
            return;
        }
        // A statement inside the block materializes its result — the single
        // pass the directive promises cannot carry it. The fused unit is one
        // expression; intermediates are hoisted above the block.
        if let Some(stmt) = body.stmts.first() {
            self.error(
                format!("fuse-infeasible: a `{}` statement inside `@fuse` materializes \
                         an intermediate — the block collapses a single elementwise \
                         expression (DIRECTIVES.md §3)", stmt_kind(stmt)),
                stmt_span(stmt).clone(),
            );
            return;
        }
        let Some(tail) = body.tail_expr.as_deref() else {
            self.error(
                "fuse-infeasible: the `@fuse` block is empty — there is nothing \
                 to collapse (DIRECTIVES.md §3)".to_string(),
                span.clone(),
            );
            return;
        };
        self.check_fuse_unit(tail);
    }

    /// The fused unit is one expression; run the walk over it and refuse a
    /// scalar yield. Shared by the three block paths (`@fuse { … }` as
    /// statement, expression, and trailing block). The statement attachment
    /// (`@fuse let x = …`) never gets here — it is an attachment error in
    /// `check_directive_stmt`, since the JIT refuses every statement-attached
    /// directive and the catalog's attachment is block / expr only.
    fn check_fuse_unit(&mut self, e: &Expr) {
        match self.fuse_expr(e) {
            Err((msg, espan)) => self.error(msg, espan),
            // A scalar-valued unit has no lanes: the fused kernel is a loop
            // that writes a tensor, and there is no loop to emit here.
            Ok(FuseLane::Scalar) => self.error(
                "fuse-infeasible: the fused expression yields a scalar — a fused \
                 kernel writes a tensor, one lane per element (DIRECTIVES.md §3)".to_string(),
                e.span_of(),
            ),
            Ok(FuseLane::Tensor(_)) => {}
        }
    }

    /// The fusable-expression walk for `check_fuse_feasible`. `Ok(Tensor)` —
    /// the subtree yields a tensor lane, carrying its shape when the checker
    /// knows it (`None` for an unresolved name — the normal checker reports
    /// that; this walk stays quiet). `Ok(Scalar)` — a float-scalar broadcast.
    /// `Err` carries the first offender found, as (message, span). Mirrors the
    /// JIT's `fuse_infer_ty`/`fuse_emit_elem` support exactly: what this walk
    /// admits, that kernel lowers.
    fn fuse_expr(&self, e: &Expr) -> Result<FuseLane, (String, Span)> {
        match e {
            // A one-element tuple is parenthesization; look through it.
            Expr::Tuple(elems, _) if elems.len() == 1 => self.fuse_expr(&elems[0]),
            Expr::BinOp { op, lhs, rhs, span } => match op {
                BinOp::DotAdd | BinOp::DotSub | BinOp::DotMul | BinOp::DotDiv
                | BinOp::DotPow
                | BinOp::DotLt | BinOp::DotGt | BinOp::DotLe | BinOp::DotGe => {
                    let l = self.fuse_expr(lhs)?;
                    let r = self.fuse_expr(rhs)?;
                    match (l, r) {
                        (FuseLane::Scalar, FuseLane::Scalar) => Err((format!(
                            "fuse-infeasible: `{}` has no tensor operand here — a fused \
                             lane reads at least one tensor (DIRECTIVES.md §3)",
                            crate::fmt::binop_str(op)), span.clone())),
                        // Two tensor operands pair elements one-to-one: the
                        // fused kernel reads every leaf at one lane offset, so
                        // tensor-tensor broadcast has no single pass to ride —
                        // the JIT's kernel refuses unequal shapes, and so does
                        // this walk.
                        (FuseLane::Tensor(Some(a)), FuseLane::Tensor(Some(b))) => {
                            if !fuse_shapes_agree(&a, &b) {
                                return Err((format!(
                                    "fuse-infeasible: `{}` reads tensors of shapes {} \
                                     and {} — fused lanes pair elements one-to-one, and \
                                     only float scalars broadcast (DIRECTIVES.md §3)",
                                    crate::fmt::binop_str(op), a, b), span.clone()));
                            }
                            Ok(FuseLane::Tensor(Some(a)))
                        }
                        (FuseLane::Tensor(a), FuseLane::Tensor(b)) =>
                            Ok(FuseLane::Tensor(a.or(b))),
                        (FuseLane::Tensor(s), FuseLane::Scalar)
                        | (FuseLane::Scalar, FuseLane::Tensor(s)) =>
                            Ok(FuseLane::Tensor(s)),
                    }
                }
                _ => Err((format!(
                    "fuse-infeasible: `{}` is not elementwise — only `.+ .- .* ./ .^ \
                     .** .< .> .<= .>=` and `\\>` collapse into one kernel (DIRECTIVES.md §3)",
                    crate::fmt::binop_str(op)), span.clone())),
            },
            Expr::UnOp { op: UnOp::ReLU, operand, span } => {
                match self.fuse_expr(operand)? {
                    FuseLane::Tensor(s) => Ok(FuseLane::Tensor(s)),
                    FuseLane::Scalar =>
                        Err(("fuse-infeasible: `\\>` (ReLU) needs a tensor operand inside \
                              `@fuse` (DIRECTIVES.md §3)".to_string(), span.clone())),
                }
            }
            Expr::UnOp { op, span, .. } => {
                let tok = match op {
                    UnOp::Neg => "-", UnOp::Not => "!", UnOp::Deref => "*",
                    UnOp::GeLU => "\\<", UnOp::BitNot => "~", UnOp::ReLU => unreachable!(),
                };
                Err((format!(
                    "fuse-infeasible: unary `{}` is outside the fusable set — `\\>` \
                     (ReLU) is its only unary op (DIRECTIVES.md §3)", tok), span.clone()))
            }
            Expr::Ident(name, span) => {
                // An unresolved or unknown-typed name is the normal checker's
                // report to make; stay quiet rather than cascade.
                let Some(ty) = self.env.lookup(name) else {
                    return Ok(FuseLane::Tensor(None));
                };
                match ty {
                    TyType::Unknown => Ok(FuseLane::Tensor(None)),
                    TyType::Tensor(elem, shape) | TyType::View(elem, shape) => {
                        if !matches!(elem.as_ref(),
                                     TyType::Scalar(ScalarType::F32) | TyType::FloatLit(_)) {
                            return Err((format!(
                                "fuse-infeasible: `{}` is a {} — the fused kernel \
                                 computes in f32 (DIRECTIVES.md §3)", name, ty), span.clone()));
                        }
                        if shape.dims.iter().any(|d| matches!(d, SymDim::Streaming)) {
                            return Err((format!(
                                "fuse-infeasible: `{}` carries a streaming `~` extent — \
                                 a fused lane needs a static shape (DIRECTIVES.md §3)",
                                name), span.clone()));
                        }
                        Ok(FuseLane::Tensor(Some(shape.clone())))
                    }
                    t if t.is_float() => Ok(FuseLane::Scalar),
                    _ => Err((format!(
                        "fuse-infeasible: `{}` is a {} — only f32 tensors and float \
                         scalars fuse (DIRECTIVES.md §3)", name, ty), span.clone())),
                }
            }
            Expr::Literal(Literal::Float(..), _) => Ok(FuseLane::Scalar),
            Expr::Literal(_, span) => Err((
                "fuse-infeasible: only float literals broadcast into a fused kernel \
                 (DIRECTIVES.md §3)".to_string(), span.clone())),
            Expr::Postfix { expr, op: PostfixOp::Call(_), span } => {
                let callee = match expr.as_ref() {
                    Expr::Ident(n, _) => format!("`{}`", n),
                    Expr::Postfix { op: PostfixOp::Field(f), .. } => format!("`{}`", f),
                    _ => "a function".to_string(),
                };
                Err((format!(
                    "fuse-infeasible: a call to {} does not collapse — calls are \
                     outside the elementwise set (DIRECTIVES.md §3)", callee), span.clone()))
            }
            Expr::Postfix { op: PostfixOp::Transpose, span, .. } => Err((
                "fuse-infeasible: `'` (transpose) is not elementwise — it reorders \
                 lanes (DIRECTIVES.md §3)".to_string(), span.clone())),
            Expr::Postfix { op: PostfixOp::Index(_), span, .. } => Err((
                "fuse-infeasible: indexing does not collapse into a fused lane \
                 (DIRECTIVES.md §3)".to_string(), span.clone())),
            Expr::Cast { span, .. } => Err((
                "fuse-infeasible: an `as` cast is outside the fusable set — cast \
                 above the block (DIRECTIVES.md §3)".to_string(), span.clone())),
            Expr::TensorLit(_, span) => Err((
                "fuse-infeasible: a tensor literal does not fuse — bind it to a `let` \
                 above the block (DIRECTIVES.md §3)".to_string(), span.clone())),
            Expr::DirectiveBlock { directives, span, .. } => {
                let name = directives.first()
                    .map(|d| format!("`@{}`", d.name))
                    .unwrap_or_else(|| "directive".to_string());
                Err((format!(
                    "fuse-infeasible: a {} block inside `@fuse` does not collapse — \
                     wrap `@fuse` in it, not the other way (DIRECTIVES.md §3)", name),
                    span.clone()))
            }
            Expr::If(_) | Expr::Match(_) => Err((
                "fuse-infeasible: control flow inside `@fuse` breaks the single-pass \
                 contract (DIRECTIVES.md §3)".to_string(), e.span_of())),
            other => Err((
                "fuse-infeasible: this expression is outside the fusable set — only \
                 elementwise ops over f32 tensor operands collapse (DIRECTIVES.md §3)"
                    .to_string(), other.span_of())),
        }
    }

    /// DIRECTIVES.md §3: `@inplace` attaches to an assignment statement and
    /// nothing else — the write that could copy-on-write is the whole point
    /// (`MEMORY.md §4.3`). Anywhere else it has no write to guard.
    fn check_inplace_target(&mut self, directives: &[Directive], what: &str) {
        for d in directives.iter().filter(|d| d.name == "inplace") {
            self.error_with_hint(
                format!("illegal directive stack: `@inplace` attaches to an assignment \
                         statement, not to {} — there is no write here that could \
                         copy-on-write (DIRECTIVES.md §3, MEMORY.md §4.3)", what),
                d.span.clone(),
                Some("move `@inplace` onto the assignment itself, e.g. `@inplace x += bias`"
                    .to_string()),
            );
        }
    }

    /// Run `body` under the port-call ban this directive stack imposes
    /// (PORTS.md §5). Outermost wins: a `@fuse` block inside a `@grad fn`
    /// keeps the `@grad fn` diagnostic — that is the restriction the
    /// programmer has to satisfy first, and both forbid the same call.
    fn with_port_ban<R>(
        &mut self,
        directives: &[Directive],
        body: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let outer = self.port_ban;
        if outer.is_none() {
            self.port_ban = port_ban_of(directives);
        }
        let r = body(self);
        self.port_ban = outer;
        r
    }

    fn check_directive_item(&mut self, directives: &[Directive], inner: &Item, is_public: bool) {
        self.lint_unimplemented_directives(directives);
        if !matches!(inner, Item::Pub(_)) {
            self.check_illegal_stack(directives, &[]);
        }
        match inner {
            Item::Let(l) => {
                self.check_inplace_target(directives, "a `let` binding");
                let ty = self.with_port_ban(directives, |c| c.check_let(l));
                self.check_sharding_directives(directives, &ty, l.span.clone());
            }
            Item::Pub(inner_inner) => self.check_directive_item(directives, inner_inner, true),
            // The parser only builds `Item::Directive` around a `let` (or a
            // `pub let`); this arm is defensive.
            _ => {
                self.check_inplace_target(directives, "a top-level item");
                self.with_port_ban(directives, |c| c.check_item_vis(inner, is_public));
            }
        }
    }

    fn check_fn(&mut self, f: &FnDecl) {
        // #728: the innermost function/method name, for the type-arg-less
        // generic-model warning (param/return positions, below) — restored
        // at the end of this fn so a caller (a call-site check, another
        // fn's body) doesn't inherit it. `current_model_name` is set by
        // `check_model` around its call to `check_fn` per method, and is
        // NOT touched here — a free fn always checks with it `None`.
        let outer_fn_name = self.current_fn_name.replace(f.name.clone());
        // #728 rework: this fn's own declared shape params — the "already
        // in scope" half of the type-arg-less generic-model hint (see
        // `current_fn_shape_params`'s own doc comment for why the model's
        // shape params, for a method, are deliberately excluded). Restored
        // alongside `current_fn_name`, same reasoning.
        let outer_fn_shape_params = std::mem::replace(
            &mut self.current_fn_shape_params,
            f.shape_params.iter().map(|sp| sp.name.clone()).collect(),
        );
        self.lint_unimplemented_directives(&f.directives);
        self.check_illegal_stack(&f.directives, &[]);
        self.check_inplace_target(&f.directives, "a `fn` declaration");
        // DIRECTIVES.md §1: `@fuse` attaches to a block or expression. On a
        // `fn` declaration both backends ignore it — an unaudited promise,
        // the class #503 exists to close.
        for d in f.directives.iter().filter(|d| d.name == "fuse") {
            self.error_with_hint(
                "illegal directive stack: `@fuse` attaches to a block or \
                 expression, not to a `fn` declaration — both backends ignore \
                 it here (DIRECTIVES.md §1)".to_string(),
                d.span.clone(),
                Some("wrap the body's fused expression instead: `@fuse { … }`".to_string()),
            );
        }
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
            if self.env.shape_param_in_scope(name) {
                continue;
            }
            // #793 (regression from #766): an implicit shape-param name that
            // also names a top-level literal constant folds to that constant
            // here too, not just in `register_fn`'s registered signature —
            // see `register_fn`'s #766 comment for the full precedence
            // rule. Binding it as a free `Var` here (the old, unconditional
            // behavior) left the body's own copy of a param/return type
            // resolving to `[N]` while `register_fn` had already folded the
            // SAME name to `[4]` in the signature callers actually check
            // args against — so a value whose type was itself produced by
            // this same fold (e.g. another fn's return type, or `forge.
            // zeros[f32,[N]]`) mismatched by construction (#793). Skipping
            // the bind here means `resolve_type`, just below, folds it via
            // `resolve_dim` exactly like `register_fn` does, so the two
            // sides agree. An EXPLICIT `fn f[N](...)` shape param still
            // shadows the constant: it was bound in the loop above, so
            // `shape_param_in_scope` is already true and this branch never
            // runs for it (SPEC.md §6.6: a declared shape param shadows a
            // same-named top-level constant).
            if self.const_dims.contains_key(name) {
                continue;
            }
            self.env.bind_shape_param(name, None);
        }
        for p in &f.params {
            // #728: a type-arg-less annotation of a shape-parameterized model
            // in parameter position — checked against the AST type directly,
            // ahead of `resolve_type`, since `resolve_type` folds the model's
            // own args down to `TyType::Named { args: [] }` for a bracketed
            // reference with expr args too, losing the "was a bracket
            // written at all" distinction this lint needs.
            //
            // #728 rework: walks into `[M; N]`/tuple/fn-type nesting
            // (defect 2), and passes `param_site` so the hint names the
            // ruling's blessed spelling — declare the shape parameter on
            // the function — rather than a bracket that would be a hard
            // type error here (defect 1).
            if let Some(ty) = &p.ty {
                let site = format!(
                    "parameter `{}` of {}",
                    p.name,
                    self.current_fn_desc().unwrap_or_else(|| "this function".to_string()),
                );
                let declared = self.current_fn_shape_params.clone();
                self.lint_bare_generic_models_in(ty, &site, &declared, Some((&f.name, &p.name)));
            }
            let ty = p.ty.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unknown);
            if p.mutating {
                self.env.bind_mutable(p.name.clone(), ty);
            } else {
                self.env.bind(&p.name, ty);
            }
        }
        // Track the current fn's return type for ? context validation (pass 2 / arena coherence).
        // #728: same lint, return position — checked against the AST type
        // ahead of the fold into `TyType`, same reason as the param loop.
        // #728 rework: array/tuple/fn-type recursion (defect 2); no
        // `param_site` — a concrete bracket (or the fn's own shape param, if
        // it already declares one under the same name) is a complete,
        // legal spelling in return position, unlike param position.
        if let Some(ty) = &f.ret_type {
            let site = format!(
                "the return type of {}",
                self.current_fn_desc().unwrap_or_else(|| "this function".to_string()),
            );
            let declared = self.current_fn_shape_params.clone();
            self.lint_bare_generic_models_in(ty, &site, &declared, None);
        }
        let declared_ret = f.ret_type.as_ref().map(|t| self.resolve_type(t)).unwrap_or(TyType::Unit);
        let outer_fn_ret = self.current_fn_ret.replace(declared_ret.clone());
        // PORTS.md §5: port calls are an effect boundary a gradient cannot
        // cross. Track the `@grad fn` body (and any closures it checks) so the
        // call site can reject them. `@fuse`/`@deterministic` on the `fn` ban
        // them for the same reason the block form does.
        let outer_ban = self.port_ban;
        self.port_ban = if f.directives.iter().any(|d| d.name == "grad") {
            Some(PortBan::GradFn)
        } else {
            port_ban_of(&f.directives)
        };
        let body_ty = self.check_block(&f.body);
        self.port_ban = outer_ban;
        self.current_fn_ret = outer_fn_ret;
        self.current_fn_name = outer_fn_name;
        self.current_fn_shape_params = outer_fn_shape_params;
        // #398: a captured mutable binding (a module-level `let !`) read in a
        // `@grad fn` is a differentiable input — the interpreter records it as a
        // tape input and returns its adjoint in `Grads` under the binding's own
        // name (AUTODIFF.md §2). Two shapes stay compile-time errors, because the
        // tape genuinely cannot produce a gradient for them and an absent (or
        // zero) field would be a silent wrong answer:
        //   * a capture read only from inside a closure literal — closure bodies
        //     are not traced;
        //   * a capture whose type carries no gradient (ints, bools, strings, …).
        // This runs after `check_block`, so body-local bindings are already popped
        // from the env: a mutable binding here that is neither a param nor a local
        // can only be a captured module-level `let !` / `let mut`.
        if self.grad_fn_counts.get(&f.name).copied().unwrap_or(0) >= 1 {
            let params: std::collections::HashSet<&str> =
                f.params.iter().map(|p| p.name.as_str()).collect();
            let refs = collect_body_idents(&f.body);
            // A name the body BINDS itself (`let cap = …`, a `for` pattern) is
            // a local, not a capture: the module binding of that name is
            // shadowed and never read. Filtering by `reads_free` fixes the rule
            // in both directions — no phantom `g.cap` for a shadowed binding,
            // and no rejection of a local `let counter = 2.0f32` merely because
            // the module's unread `counter` happens to be an `i64`.
            let mut captured: Vec<&String> = refs.direct.iter().chain(refs.in_closure.iter())
                .filter(|n| !params.contains(n.as_str())
                            && refs.reads_free(n)
                            && self.env.is_mutable_binding(n))
                .collect();
            captured.sort();
            captured.dedup();
            for name in captured {
                if refs.in_closure.contains(name) {
                    self.error_with_hint(
                        format!("captured mutable binding `{}` is not differentiable inside a \
                                 closure — the gradient tape does not enter closure bodies, so \
                                 its gradient would be silently absent", name),
                        f.body.span.clone(),
                        Some(format!("read `{}` directly in the differentiated body (a direct \
                                      capture does get a gradient), or pass it as a `!` parameter",
                                     name)),
                    );
                    continue;
                }
                let ty = self.env.lookup(name).cloned().unwrap_or(TyType::Unknown);
                if !capture_type_is_differentiable(&ty) {
                    self.error_with_hint(
                        format!("captured mutable binding `{}` has non-differentiable type `{}` \
                                 inside a `@grad fn`", name, ty),
                        f.body.span.clone(),
                        Some("only float scalars and float tensors carry gradients; a captured \
                              integer, bool, or string binding never enters the tape".to_string()),
                    );
                    continue;
                }
                // The tape does not trace a call, so a capture the body reads
                // *and* a called fn reads would come back with a gradient that
                // silently omits the callee's path. Name the callee.
                if let Some(callee) = self.capture_read_in_callee(&f.name, &refs, name) {
                    self.error_with_hint(
                        format!("captured mutable binding `{}` is also read inside fn `{}`, \
                                 which this `@grad fn` calls — the tape does not trace calls, \
                                 so `{}`'s gradient would silently omit that path",
                                name, callee, name),
                        f.body.span.clone(),
                        Some(format!("read `{}` only in the differentiated body (inline what \
                                      `{}` does with it), or stop reading it here so the \
                                      gradient is not half-computed", name, callee)),
                    );
                }
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
                self.error_mismatch(
                    format!("fn `{}` returns `{}` but body produces `{}`", f.name, declared_ret, effective_body_ty),
                    f.body.span.clone(),
                    Some(format!("declared return type at line {}", f.span.line)),
                    &declared_ret,
                    &effective_body_ty,
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

    /// #398: does any user fn reachable from this `@grad fn` body read the
    /// captured mutable `name` freely? Returns the first such fn.
    ///
    /// The tape records only what the differentiated body itself evaluates; a
    /// plain fn call is executed concretely and contributes no nodes. So when
    /// the body reads a capture *and* a callee reads the same capture, the
    /// adjoint we return covers only the body's half — a silently partial
    /// gradient. Names are resolved through `fn_body_refs`, and a callee that
    /// binds the name itself (its own param or `let`) is not a reader of the
    /// module binding.
    fn capture_read_in_callee(
        &self,
        grad_fn: &str,
        body: &BodyRefs,
        name: &str,
    ) -> Option<String> {
        // A callee is either a plain fn (keyed by its name in `fn_body_refs`)
        // or one model's method (keyed by `model` + `method` in
        // `method_body_refs`). Both are walked; both can read a capture.
        #[derive(PartialEq, Eq, Hash, Clone)]
        enum Callee {
            Fn(String),
            /// (model, method) — reported as `Model.method`.
            Method(String, String),
        }
        impl Callee {
            fn label(&self) -> String {
                match self {
                    Callee::Fn(n) => n.clone(),
                    Callee::Method(m, f) => format!("{}.{}", m, f),
                }
            }
        }
        let mut seen: std::collections::HashSet<Callee> = std::collections::HashSet::new();
        seen.insert(Callee::Fn(grad_fn.to_string()));
        let mut queue: Vec<Callee> = Vec::new();
        // `enqueue` expands one body's outgoing edges: identifiers that name a
        // registered fn (conservative — a bare fn reference passed as a value
        // counts too), plus every model method matching a `recv.m()` call.
        let enqueue = |refs: &BodyRefs, queue: &mut Vec<Callee>| {
            for n in refs.direct.iter().chain(refs.in_closure.iter()) {
                if self.fn_body_refs.contains_key(n) {
                    queue.push(Callee::Fn(n.clone()));
                }
            }
            for m in &refs.method_calls {
                let Some(cands) = self.method_body_refs.get(m) else { continue };
                for (model, _) in cands {
                    queue.push(Callee::Method(model.clone(), m.clone()));
                }
            }
        };
        enqueue(body, &mut queue);
        let mut found: Option<String> = None;
        while let Some(callee) = queue.pop() {
            if !seen.insert(callee.clone()) {
                continue;
            }
            let refs = match &callee {
                Callee::Fn(n) => self.fn_body_refs.get(n),
                Callee::Method(model, m) => self.method_body_refs.get(m)
                    .and_then(|cands| cands.iter()
                        .find(|(owner, _)| owner == model)
                        .map(|(_, r)| r)),
            };
            let Some(refs) = refs else { continue };
            if refs.reads_free(name) {
                // Report the alphabetically first reader, so the diagnostic is
                // stable no matter how the traversal ordered the graph.
                let label = callee.label();
                if found.as_deref().map_or(true, |f| label.as_str() < f) {
                    found = Some(label);
                }
            }
            enqueue(refs, &mut queue);
        }
        found
    }

    /// AUTODIFF.md §2: a `@grad fn` that captures **no** mut bindings and has
    /// **no** mut (`!`) parameters is a compile-time error — there is nothing
    /// for the backward to produce. The JIT enforces the parameter half in
    /// `declare_grad_fn` (jit.rs); mirror it here so `--check` and the
    /// tree-walking `run` reject it too, not just `--jit`. Since #398 a
    /// directly-read captured mut binding *is* a differentiable input in the
    /// interpreter, so it satisfies this rule on its own — which is what the
    /// spec sentence has always said. (The JIT can't reach that case: a
    /// module-level non-const `let` is already unsupported at lowering, so
    /// such a program runs on the interpreter.)
    fn check_grad_fn_differentiable(&mut self, f: &FnDecl) {
        let is_grad = f.directives.iter().any(|d| d.name == "grad");
        if !is_grad {
            return;
        }
        if f.params.iter().any(|p| p.mutating) {
            return;
        }
        let params: std::collections::HashSet<&str> =
            f.params.iter().map(|p| p.name.as_str()).collect();
        let refs = collect_body_idents(&f.body);
        let has_captured_mut = refs.direct.iter().any(|n| {
            // Same shadowing rule as the capture scan itself (#398): a body-local
            // `let cap` is not a capture, so it cannot be the differentiable
            // input that saves an otherwise input-less `@grad fn`.
            !params.contains(n.as_str())
                && refs.reads_free(n)
                && self.env.is_mutable_binding(n)
                && capture_type_is_differentiable(
                    self.env.lookup(n).unwrap_or(&TyType::Unknown))
        });
        if !has_captured_mut {
            self.error(
                "`@grad fn` with no `!` (mut) parameter and no captured `mut` binding \
                 has nothing to differentiate (see the spec on `@grad`)",
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
        self.check_illegal_stack(&m.directives, &[]);
        self.check_inplace_target(&m.directives, "a `model` declaration");
        self.env.push_shape_scope();
        for sp in &m.shape_params {
            self.env.bind_shape_param(&sp.name, None);
        }
        let model_shape_names: std::collections::HashSet<String> =
            m.shape_params.iter().map(|sp| sp.name.clone()).collect();
        for member in &m.members {
            // SPEC §3.11: a port handle "cannot appear inside tensor element
            // types, model fields, or Vault constants". Nothing enforced the
            // model-field half, and the two backends disagreed about what a
            // program that did it even meant: `dmc run` accepted it, while
            // `dmc jit` reported `cannot convert \`Port\` to \`Port\`` — a
            // message that reads like a compiler bug and half was: the field's
            // declared type had resolved to a *model* named `Port`, so the two
            // sides rendered the same and neither was what the author wrote.
            // Rejecting it here means the same answer under both backends,
            // named at the field, which is what SPEC already promised.
            if let ModelMember::Field { name, ty, span, .. } = member {
                if crate::ports::is_port_type(ty) {
                    self.error_with_hint(
                        format!("model field `{}.{}` is a port handle — a handle cannot \
                                 appear in a model field (SPEC §3.11)", m.name, name),
                        span.clone(),
                        Some("a handle does not outlive its run and owns no demoniC \
                              memory; pass it as a parameter instead (`Port[L]` is \
                              writable in parameter and return positions)".to_string()),
                    );
                }
            }
            if let ModelMember::Method(f) = member {
                // Detect shape vars used in method signature that are not declared on the
                // model or the method itself. Emit a targeted diagnostic before body checking
                // so users aren't left wondering where an unfamiliar symbol came from.
                //
                // #798: an undeclared name that names a top-level immutable
                // `let` constant is not "unknown" — SPEC.md §6.6 says a
                // named dimension in ANY type annotation that is not a
                // declared shape parameter resolves against a top-level
                // literal `let`, and a method's parameter/return annotation
                // is such an annotation (`check_fn` already applies this to
                // a free fn, #793/#797). Excluding it here leaves it to fold
                // normally, the same way, when `check_fn` runs just below —
                // register_model's own signature registration already
                // folds it via `resolve_dim`/`const_dims`, so the two agree.
                // A name that names a MUTABLE top-level `let` (`let
                // !`/`let mut`) is excluded too, for a different reason:
                // it is not "unknown" either, but it is not a usable
                // constant — `resolve_dim_at` (via `check_fn`'s own
                // `resolve_type` call on this same signature, just below)
                // gives it the precise, located "a mutable binding is never
                // a constant dimension" diagnostic (#799) instead of this
                // one, which would otherwise misname the problem and, via
                // its hint, recommend `fn m[N](self, ...)` — a spelling
                // that would silently shadow the constant.
                let method_shape_names: std::collections::HashSet<String> =
                    f.shape_params.iter().map(|sp| sp.name.clone()).collect();
                let mut all_used = std::collections::HashSet::new();
                for p in &f.params {
                    if let Some(t) = &p.ty { collect_shape_vars(t, &mut all_used); }
                }
                if let Some(rt) = &f.ret_type { collect_shape_vars(rt, &mut all_used); }
                // #798: a third category reaches here alongside "genuinely
                // undeclared" and "mutable top-level `let`" (excluded just
                // above): an IMMUTABLE top-level `let` that is not a
                // literal constant (`let N = side()`, or a chain that never
                // bottoms out in one) — not in `const_dims` (literal-only),
                // not in `mutable_top_level_lets` either, so without this it
                // fell straight through to "unknown shape param" plus the
                // shadowing hint below, the exact complaint #798 was filed
                // on. `all_top_level_let_names` names it precisely instead,
                // the same wording `resolve_alias_body`'s own second branch
                // uses for the identical case in an alias body.
                let mut undeclared: Vec<String> = Vec::new();
                let mut non_literal_lets: Vec<String> = Vec::new();
                for v in all_used {
                    if method_shape_names.contains(&v)
                        || model_shape_names.contains(&v)
                        || self.const_dims.contains_key(&v)
                        || self.mutable_top_level_lets.contains(&v)
                    {
                        continue;
                    }
                    if self.all_top_level_let_names.contains(&v) {
                        non_literal_lets.push(v);
                    } else {
                        undeclared.push(v);
                    }
                }
                non_literal_lets.sort();
                for name in &non_literal_lets {
                    self.error(
                        format!(
                            "method `{}` of model `{}` names shape dimension `{}`, \
                             but `{}` is a top-level `let` in this module that is \
                             not a literal constant (nor a chain of top-level \
                             `let`s that bottoms out in one) (SPEC.md §6.6)",
                            f.name, m.name, name, name),
                        f.span.clone(),
                    );
                }
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
                             model-level shape params [{}] are already in scope \
                             inside methods (SPEC.md §3.10) — `{}` isn't declared \
                             as one of them",
                            f.name,
                            undeclared.join(", "),
                            model_params.join(", "),
                            undeclared.join(", "),
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
                // #728: names the enclosing model in the type-arg-less
                // generic-model warning ("method `f` of model `M`") while
                // `check_fn` resolves this method's own param/return
                // annotations.
                let outer_model_name = self.current_model_name.replace(m.name.clone());
                self.check_fn(f);
                self.current_model_name = outer_model_name;
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
        // #597: a *bare* literal (any radix, any suffix) bound to a 64-bit
        // target is a bit pattern by construction — `0xffffffffffffffffu64`
        // folds to `-1` through `expr_i64` below but is a legitimate `u64`
        // value; both `i64` and `u64` are exactly 64 bits, so no bit pattern
        // is ever out of range for either. This mirrors the identical
        // carve-out for a literal's own suffix, elsewhere in this file. An
        // *explicit* leading `-` (`-1u64`) is not a bare literal — it is the
        // program writing down a negative value for an unsigned type, which
        // stays an error, so the check below still runs for it.
        if matches!(s, ScalarType::I64 | ScalarType::U64)
            && matches!(value_expr, Expr::Literal(Literal::Int(..), _))
        {
            return;
        }
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
        // #728: a type-arg-less annotation of a shape-parameterized model
        // on a `let` binding — the fourth of the ratified decision's
        // annotation positions. #728 rework: array/tuple/fn-type recursion
        // (defect 2); no `param_site` (this isn't a parameter).
        if let Some(ty) = &l.ty {
            let pat = pattern_for_diagnostic(&l.pattern).unwrap_or_else(|| "binding".to_string());
            let site = match self.current_fn_desc() {
                Some(desc) => format!("let binding `{pat}` in {desc}"),
                None => format!("let binding `{pat}`"),
            };
            let declared_shape_params = self.current_fn_shape_params.clone();
            self.lint_bare_generic_models_in(ty, &site, &declared_shape_params, None);
        }
        let declared = l.ty.as_ref().map(|t| self.resolve_type(t));
        let final_ty = if let Some(d) = declared {
            if !d.compatible_with(&value_ty) {
                self.error_mismatch(
                    format!("let binding has type {} but value has type {}", d, value_ty),
                    l.span.clone(),
                    None,
                    &d,
                    &value_ty,
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
            let ctor_ty = if l.ty.is_none() { ctor_tensor_ty(&l.value) } else { None };
            let ctor_shape = ctor_ty.as_ref().and_then(|t| match t {
                TyType::Tensor(_, s) => Some(s.clone()),
                _ => None,
            });
            self.env.set_ctor_shape(name.clone(), ctor_shape);
            // #575: same fact, element-type half — see `Env::set_ctor_elem`.
            // `embed`'s `ids` argument check needs to see e.g. `f32` on a
            // bound `let ids = forge.zeros[f32, [2]]` even though `ids`
            // itself types as `Unknown` (constructors deliberately report
            // `Unknown`, #248 above), the same way its shape survives.
            let ctor_elem = ctor_ty.and_then(|t| match t {
                TyType::Tensor(e, _) => match *e {
                    TyType::Scalar(s) => Some(s),
                    _ => None,
                },
                _ => None,
            });
            self.env.set_ctor_elem(name.clone(), ctor_elem);
            // #533 (PORTS.md §3.2): remember a `forge.trit[K, N]` RHS. The
            // constructor has no element-type argument, so nothing else on this
            // path records that the binding holds a packed ternary tensor — and
            // whether a tensor has a copy-mode wire dtype is a static property
            // of its element type. Cleared on a rebind to anything else.
            let is_trit = l.ty.is_none() && is_trit_ctor(&l.value);
            self.env.set_trit_origin(name.clone(), is_trit);
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
            // #476 (MEMORY §2): reach one level into the model being built.
            // `Holder { cells: vault.uninit[Cell, [3]] }` leaves the field just
            // as undefined as the local spelling, but the array is a field, not
            // a binding, so the check above never saw it — the read sailed
            // through `--check` and surfaced at runtime as a complaint about
            // `opaque`. Any earlier marks under this name are stale.
            self.env.clear_uninit_fields_under(name);
            if let Expr::StructLit { name: model, fields, .. } = &l.value {
                for (fname, fexpr) in fields {
                    // Model-array fields only — see `is_uninit_field`.
                    let is_model_array = self.env.models.get(model)
                        .and_then(|m| m.fields.get(fname))
                        .is_some_and(|t| matches!(t, TyType::Array(..)));
                    if is_model_array && is_uninit_ctor(fexpr) {
                        self.env.mark_uninit(format!("{name}.{fname}"));
                    }
                }
            }
            // #442 (MEMORY §3.1): tag bindings whose value lives in the Vault,
            // so mutating them outside a `vault { … }` block can error.
            if is_vault_ctor(&l.value) {
                self.env.mark_vault_origin(name.clone());
            } else {
                self.env.unmark_vault_origin_here(name);
            }
            // #792 rework: record an immutable local `let` bound to an
            // integer literal (or a chain to another already-known int
            // constant, local or top-level) as a usable dimension
            // constant — mirrors `collect_const_dims`'s top-level fold
            // (#761/#766), scoped to this binding. Consulted by
            // `resolve_dim` so a bracketed shape argument like `let k = 4
            // mk[k]()` resolves `k` to `Const(4)` instead of staying an
            // unbound `Var` that a concrete dim can only ever fail to
            // match (`SymDim::equivalent`'s `Unknown`).
            if !(l.mutating || l.is_mut) {
                let const_val = expr_i64(&l.value).or_else(|| match &l.value {
                    Expr::Ident(source, _) =>
                        self.env.int_const(source).or_else(|| self.const_dims.get(source).copied()),
                    _ => None,
                });
                match const_val {
                    Some(v) => self.env.mark_int_const(name.clone(), v),
                    None => self.env.unmark_int_const_here(name),
                }
            } else {
                self.env.unmark_int_const_here(name);
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
        self.check_illegal_stack(directives, &[]);
        // No block here, so no source type to vet — only the model's fields.
        self.check_cast_model_overlay(directives, None);
        // DIRECTIVES.md §3: `@inplace` attaches to an assignment statement and
        // nothing else — that is the only construct whose write could CoW.
        match inner {
            Stmt::Expr { assign: Some(_), .. } => {}
            Stmt::Let(_) => self.check_inplace_target(directives, "a `let` binding"),
            Stmt::Expr { .. } => self.check_inplace_target(directives, "a bare expression"),
            other => {
                let what = format!("a `{}` statement", stmt_kind(other));
                self.check_inplace_target(directives, &what);
            }
        }
        // DIRECTIVES.md §1: `@fuse` attaches to a block or expression — a
        // statement is neither. The JIT refuses every statement-attached
        // directive before any fuse analysis runs, so admitting the form here
        // (even with a feasible body) would split the backends on a program
        // the expression spelling handles. Same refusal as the `fn` form.
        for d in directives.iter().filter(|d| d.name == "fuse") {
            self.error_with_hint(
                format!("illegal directive stack: `@fuse` attaches to a block or \
                         expression, not to a `{}` statement (DIRECTIVES.md §1)",
                        stmt_kind(inner)),
                d.span.clone(),
                Some("write the fused unit as an expression: `let x = @fuse { … }`"
                    .to_string()),
            );
        }
        if let Stmt::Let(l) = inner {
            let ty = self.with_port_ban(directives, |c| c.check_let(l));
            self.check_sharding_directives(directives, &ty, l.span.clone());
        } else {
            self.with_port_ban(directives, |c| c.check_stmt(inner));
        }
    }

    /// DIRECTIVES.md §3: `@shard`/`@tp` name a mesh axis the sharded value's
    /// type has to be able to accept. `ty` is the type the directive attaches
    /// to — the bound type for the `let` form, the body's type for the
    /// expression/block form. Both forms run this, so `@shard(axis=0,
    /// mesh=mesh.dp) { s }` on an `f32` is the same hard error as the `let`
    /// spelling of it.
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

    /// #598: `@cast(Model) { bytes }` overlays a raw byte tensor onto a
    /// model's fields (DIRECTIVES.md §4.1: the overlay casts "are
    /// value-transforming by design"). The only byte-layout guarantee the spec
    /// makes for a model is field ORDER — SPEC.md §3.10's "Deterministic field
    /// order. Wire layout matches source order." — so the overlay reaches
    /// exactly as far as each field's own type fixes a byte extent: a scalar
    /// of known width, or a tensor of such scalars whose every dim is a static
    /// constant (element count × element width, row-major, per the spec's own
    /// tensor convention). A `str` field's dynamic, arena-backed text, a
    /// tensor at a symbolic shape, a nested model, and a `View`/`KV`
    /// descriptor over someone else's arena have no extent to state.
    ///
    /// Before this, the interpreter's overlay materializer (`interp.rs`'s
    /// `cast_to_struct`) treated an undefined field the same as a zero-width
    /// scalar — `str` had a byte size of 0 — so a `str` field silently
    /// became `Int(0)` instead of text, with `--check` reporting nothing. The
    /// corruption surfaced only lazily, at the first read of that field
    /// (#598's `cannot iterate over opaque`). This refuses the program at the
    /// `@cast` site instead, naming every field the overlay cannot lay out.
    ///
    /// `source` is the type of the block the directive wraps, where the
    /// checker knows it. The overlay reads its source ONE ELEMENT PER BYTE, so
    /// a source of wider elements would silently contribute one byte each and
    /// run short — refused here where the source's element type is known (a
    /// typed parameter or declared binding). A source built locally by
    /// `forge.zeros[f32, …]` carries no element type through the checker yet,
    /// so it is not caught here; that inference gap is tracked separately.
    fn check_cast_model_overlay(&mut self, directives: &[Directive], source: Option<&TyType>) {
        let Some(cast_dir) = directives.iter().find(|d| d.name == "cast") else { return };
        let Some(DArg::Positional(Expr::Ident(ty_name, _))) = cast_dir.args.first() else { return };
        let Some(info) = self.env.models.get(ty_name).cloned() else { return };
        // `ModelInfo::fields` is a `HashMap`: sort by name so the diagnostics a
        // program gets never depend on hash iteration order.
        let mut fields: Vec<(&String, &TyType)> = info.fields.iter().collect();
        fields.sort_by(|a, b| a.0.cmp(b.0));
        for (fname, fty) in fields {
            if let Some(reason) = cast_overlay_field_refusal(fty) {
                self.error(
                    format!(
                        "@cast({}): model field `{}: {}` has no overlay layout — {} \
                         (DIRECTIVES.md §4.1, SPEC.md §3.10)",
                        ty_name, fname, render_ty(fty), reason,
                    ),
                    cast_dir.span.clone(),
                );
            }
        }
        // The overlay's source is the `bytes` of `@cast(Model) { bytes }`: a
        // tensor read one byte per ELEMENT. `u8` is the spelling that says so;
        // `i64` is admitted because that is the byte-stream shape `read_bytes`
        // hands back — "one i64 per byte (0–255) … Feeds `@cast(Model){bytes}`"
        // (STDLIB.md). Any other element type is a tensor of values, not of
        // bytes, and would be silently truncated one element at a time.
        if let Some((elem, _)) = source.and_then(|t| t.as_tensor_like()) {
            if let TyType::Scalar(st) = elem {
                if !matches!(st, ScalarType::U8 | ScalarType::I64) {
                    self.error(
                        format!(
                            "@cast({}): the overlay source is a byte tensor read one byte per \
                             element, but this one has `{}` elements — pass a `u8` tensor or \
                             view (DIRECTIVES.md §4.1)",
                            ty_name, render_ty(elem),
                        ),
                        cast_dir.span.clone(),
                    );
                }
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
            self.error(format!("{} requires a tensor-like value; got `{}`", directive, ty), span);
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
        let ty = self.check_block_body(block);
        self.env.pop_scope();
        ty
    }

    /// The statements and value of `block`, checked in the CURRENT scope.
    /// `check_block` wraps this in a scope of its own; the trailing
    /// directive-block arm below calls it directly so the inner `let`s stay
    /// visible to the enclosing block, and so a body that ends in an `if` /
    /// `match` STATEMENT (keyword-led, so parsed as a stmt rather than a tail
    /// expr) yields that statement's value exactly as a plain block does.
    fn check_block_body(&mut self, block: &Block) -> TyType {
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

        if let Some(tail) = &block.tail_expr {
            self.check_expr(tail)
        } else if deferred_last {
            match &block.stmts[n - 1] {
                Stmt::DirectiveBlock { directives, body, span: db_span } => {
                    // This path bypasses `check_stmt`, so the directive checks
                    // that arm runs have to happen here too — a trailing
                    // `@fuse { … }` is still a `@fuse` block.
                    self.lint_unimplemented_directives(directives);
                    self.check_illegal_stack(directives, &nested_directive_stack(body));
                    self.check_inplace_target(directives, "a block");
                    self.check_fuse_feasible(directives, body, db_span);
                    // Walk the body in the outer scope so its lets are visible,
                    // and take its value the way any block does — a body that
                    // ends in an `if` / `match` statement yields that value.
                    let ty = self.with_port_ban(directives, |c| c.check_block_body(body));
                    self.check_cast_model_overlay(directives, Some(&ty));
                    self.check_sharding_directives(directives, &ty, db_span.clone());
                    ty
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
        }
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
                // #476: the same suppression one level in. `h.cells[i] = Cell { .. }`
                // is the write that initializes the field, not a read of it —
                // and a whole-`=` rebind of the root drops every mark under it.
                let uninit_field_write: Option<(String, String)> = match assign {
                    Some((AssignOp::Eq | AssignOp::ColonEq | AssignOp::StreamArrow, _)) =>
                        lhs_field_path(lhs)
                            .filter(|(r, f)| self.env.is_uninit_field(r, f)),
                    _ => None,
                };
                if let Some((r, f)) = &uninit_field_write { self.env.clear_uninit_field(r, f); }
                if matches!(assign, Some((AssignOp::Eq | AssignOp::ColonEq, _))) {
                    if let Expr::Ident(n, _) = lhs { self.env.clear_uninit_fields_under(n); }
                }
                let lhs_ty = self.check_expr(lhs);
                if let Some((op, rhs)) = assign {
                    if let Some(r) = &uninit_write { self.env.mark_uninit(r.clone()); }
                    // #476: re-arm for the RHS so `h.cells[0] = h.cells[1]`
                    // still reports the undefined read it performs.
                    if let Some((r, f)) = &uninit_field_write {
                        self.env.mark_uninit(format!("{r}.{f}"));
                    }
                    let rhs_ty = self.check_expr(rhs);
                    if let Some((r, f)) = &uninit_field_write { self.env.clear_uninit_field(r, f); }
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
                // #724: the `Range` arm in `check_expr` now recurses into
                // `start`/`end` itself, so this one `check_expr(iter)` call
                // covers a `for k in 0..(EXPR)` iterable's bounds too — no
                // need to special-case `Expr::Range` here (and doing so
                // anyway would double-report the same error, once from this
                // call and once from the `Range` arm it delegates to).
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
            Stmt::DirectiveBlock { directives, body, span } => {
                self.lint_unimplemented_directives(directives);
                self.check_illegal_stack(directives, &nested_directive_stack(body));
                self.check_inplace_target(directives, "a block");
                self.check_fuse_feasible(directives, body, span);
                let ty = self.with_port_ban(directives, |c| c.check_block(body));
                self.check_cast_model_overlay(directives, Some(&ty));
                self.check_sharding_directives(directives, &ty, span.clone());
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
                // #350 (S9 ruling on #501): in a match whose scrutinee is
                // enum-typed, a bare identifier that does NOT name a variant of
                // that enum is an *irrefutable* catch-all binding (SPEC §4.5) —
                // it swallows every variant left, shadows the arms below it, and
                // silences the exhaustiveness check that would otherwise report
                // the variant the author forgot. A typo'd variant name reads as
                // working code. Warn on every such arm; the hint says which of
                // the two intents to write explicitly.
                //
                // Bare idents that DO resolve to a variant are genuine variant
                // patterns and stay silent — the corpus spells arms both ways on
                // purpose (`examples/enum_traffic.dmc`, `examples/enum_shape.dmc`).
                // A guarded arm states its own intent, and a name already bound
                // in scope is #269's case above; both are left to those paths.
                if arm.guard.is_none()
                    && name != "_"
                    && self.env.lookup(name).is_none()
                    && enum_variants.as_ref().is_some_and(|vs| !vs.contains(name))
                {
                    let this = enum_name.clone().unwrap_or_default();
                    let variants = enum_variants.clone().unwrap_or_default();
                    // A variant of some *other* enum is the highest-confidence
                    // read: name the enum that actually owns it.
                    let owner = {
                        let mut owners: Vec<&String> = self.env.enums.iter()
                            .filter(|(_, vs)| vs.contains(name))
                            .map(|(en, _)| en)
                            .collect();
                        owners.sort();
                        owners.first().map(|o| (*o).clone())
                    };
                    let (msg, hint) = if let Some(owner) = owner {
                        (format!("match arm `{}` binds the scrutinee (catch-all), but `{}` is a \
                                  variant of enum `{}`, not `{}`", name, name, owner, this),
                         format!("did you mean a variant of `{}`? qualify it as `{}.<Variant>`, \
                                  or use `_` for a real catch-all", this, this))
                    } else if let Some(near) = closest_variant(name, &variants) {
                        // Near-miss on this enum's own variants: a typo or a
                        // case slip, not a binding anyone wanted.
                        (format!("match arm `{}` binds the scrutinee (catch-all); enum `{}` has no \
                                  variant `{}`", name, this, name),
                         format!("did you mean `{}.{}`? a bare non-variant ident is an irrefutable \
                                  catch-all — write `_` if that is what you meant", this, near))
                    } else {
                        (format!("match arm `{}` binds the scrutinee (catch-all); enum `{}` has no \
                                  variant `{}`", name, this, name),
                         format!("write `_` for a catch-all, or qualify the variant you meant as \
                                  `{}.<Variant>` (variants: {})", this, variants.join(", ")))
                    };
                    self.warn(msg, pspan.clone(), Some(hint));
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
                // #724: `SPEC.md §6.7` — this name is a top-level `let` of a
                // module this file `use`s, unaliased. It never resolved
                // through `process_imports` (top-level `let`s are never
                // merged), so without this arm the fallback below would call
                // it a plain `undefined identifier` — true, but it hides the
                // actual rule: a constant does not cross a module boundary at
                // all, `pub` or not.
                if let Some(module) = self.import_private_lets.get(name) {
                    self.error(
                        format!(
                            "import-let: `{}` is a top-level `let` of `{}`; a constant \
                             crosses a module boundary as a `pub fn` accessor, not a \
                             bare reference (SPEC.md §6.7)",
                            name, module),
                        span.clone(),
                    );
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
                    // Unsuffixed float elements (`[1.0, 2.0]` — the ordinary
                    // spelling): both backends build f32 lanes from float
                    // leaves (the interpreter tags tensor literals `DType::F32`,
                    // the JIT's literal lowering picks f32 on any float leaf),
                    // so the checker says f32 too instead of `Unknown`. The
                    // scalar f64 default (SPEC §2) is for a *bare* literal
                    // bound without context, not for tensor data.
                    TyType::FloatLit(_) => TyType::Tensor(
                        Box::new(TyType::Scalar(ScalarType::F32)),
                        Shape::new(vec![SymDim::Const(n)]),
                    ),
                    _ => TyType::Tensor(Box::new(TyType::Unknown), Shape::new(vec![SymDim::Const(n)])),
                };
                // #403/#501 (SPEC §4.2, TOKENIZER §8a): tensor literals are for
                // small, human-legible constants. Past 256 total elements this
                // was a lint that TOKENIZER §8a promised to promote; #501 (the
                // pre-0.1.0 sweep) collects that promise. Like the §3.1
                // cross-arena write it is a spec violation rather than a lint,
                // so it goes through `error_with_hint` and demon mode does not
                // suppress it. Leaf count comes from the full inferred shape,
                // so nested literals count all their scalars.
                if let TyType::Tensor(_, shape) = &ty {
                    let total = shape.dims.iter().try_fold(1i64, |acc, d| match d {
                        SymDim::Const(k) => Some(acc.saturating_mul(*k)),
                        _ => None,
                    });
                    if let Some(total) = total {
                        if total > 256 {
                            self.error_with_hint(
                                format!("tensor literal has {total} elements — the limit is 256 \
                                         (SPEC §4.2); literals are for small, human-legible constants"),
                                lit_span.clone(),
                                Some("write it as `forge.zeros[T, S]`, `forge.ones[T, S]` or \
                                      `forge.uninit[T, S]` and fill, or `vault.load[T, S](\"file.bin\")` \
                                      for constant data".to_string()),
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
            BudgetBlock(bb) => {
                // #674 slice 3a: `forge.enter(b) { … }` selects a budget, not
                // an arena — a SEPARATE lexical stack from `arena_stack`
                // (§2.3's own note: this must never push an `ArenaKind`).
                let bty = self.check_expr(&bb.operand);
                if !matches!(bty, TyType::Budget | TyType::Unknown) {
                    self.error(
                        format!("`forge.enter` expects a `Budget`, found `{bty}`"),
                        bb.operand.span_of(),
                    );
                }
                self.budget_stack.push(bb.span.clone());
                let t = self.check_block(&bb.body);
                self.budget_stack.pop();
                t
            }
            AllocIn { expr: ctor, budget, span } => {
                let bty = self.check_expr(budget);
                if !matches!(bty, TyType::Budget | TyType::Unknown) {
                    self.error(
                        format!("`in` expects a `Budget`, found `{bty}`"),
                        budget.span_of(),
                    );
                }
                // R3 (`budget-arena`, `DIAGNOSTICS.md §5.1`, #674 §2.3): one
                // class, two instances — bytes that no budget can own,
                // written as though a budget owned them. An explicit `in b`
                // override can only ever name Forge bytes: a `vault.*` spine
                // is a region of Forge only by NOT being named `vault.*`
                // (`MEMORY.md §3.2`), and a `.kv` spine always lands in the
                // Stream arena regardless of prefix (`MEMORY.md §9`,
                // `interp.rs:try_arena_constructor`) — so both are refused
                // here unconditionally, independent of `budget_stack`.
                if let Expr::Postfix { expr: base, op: PostfixOp::Index(_), .. } = ctor.as_ref() {
                    if let Some((arena, method)) = arena_ctor_receiver(base) {
                        // #674 slice 3a fix: `.kv` always lands in the Stream
                        // arena regardless of prefix (`MEMORY.md §9`), so
                        // `vault.kv` must hit the kv message, not the vault
                        // one — test `method == "kv"` before the `arena ==
                        // "vault"` arm, or `vault.kv […] in b` reports the
                        // Vault instance and hints `forge.kv … in b`, which
                        // the kv arm itself refuses.
                        if method == "kv" {
                            self.error(
                                format!(
                                    "budget-arena: `{arena}.kv` allocates in the Stream \
                                     arena, which no budget owns; allocate the cache \
                                     before entering the budget"),
                                span.clone(),
                            );
                        } else if arena == "vault" {
                            self.error(
                                format!(
                                    "budget-arena: `vault.{method}` allocates in the Vault, \
                                     which no budget owns; spell it `forge.` to put it \
                                     inside the budget, or drop the `in b` and let it \
                                     charge the root"),
                                span.clone(),
                            );
                        }
                    }
                }
                // Suppress the ambient `budget_stack` instance of R3(i)
                // (`check_postfix`'s `Index` arm) ONLY for `ctor`'s own span
                // while checking it — otherwise a `.kv` spine overridden with
                // `in b` inside a `forge.enter` extent reports twice. Scoped
                // to this exact span, not a bare flag, so a DIFFERENT `.kv`
                // spine nested inside `ctor`'s own index/shape arguments
                // still trips the ambient check below.
                let prev = std::mem::replace(&mut self.checking_alloc_in_ctor, Some(ctor.span_of()));
                let ty = self.check_expr(ctor);
                self.checking_alloc_in_ctor = prev;
                ty
            }
            DirectiveBlock { directives, body, span } => {
                self.lint_unimplemented_directives(directives);
                self.check_illegal_stack(directives, &nested_directive_stack(body));
                self.check_inplace_target(directives, "a block");
                self.check_fuse_feasible(directives, body, span);
                let ty = self.with_port_ban(directives, |c| c.check_block(body));
                self.check_cast_model_overlay(directives, Some(&ty));
                self.check_sharding_directives(directives, &ty, span.clone());
                ty
            }
            StructLit { name, type_args, fields, span } => {
                let mut field_tys = Vec::new();
                for (fname, val) in fields {
                    field_tys.push((fname, self.check_expr(val)));
                }
                if let Some(info) = self.env.models.get(name).cloned() {
                    // Build shape-param substitution from constructor type args, e.g. M[3] binds N=3.
                    let mut shape_bindings: HashMap<String, SymDim> = info.shape_params.iter()
                        .zip(type_args.iter())
                        .map(|(param, arg_expr)| (param.clone(), SymDim::from_expr(arg_expr).simplify()))
                        .collect();
                    // #474: a literal written without a bracket is not silent
                    // about its shape — its fields say it. Recover the args
                    // from them, so both the literal's own field check and the
                    // type it hands back speak in dims rather than in the
                    // model's unbound parameter names.
                    let literal_args: Vec<TyType> = if type_args.is_empty() {
                        let args = self.infer_literal_shape_args(&info, &field_tys);
                        for (param, arg) in info.shape_params.iter().zip(&args) {
                            if let TyType::Dim(d) = arg {
                                shape_bindings.insert(param.clone(), d.clone());
                            }
                        }
                        args
                    } else {
                        type_args.iter()
                            .map(|e| TyType::Dim(SymDim::from_expr(e).simplify()))
                            .collect()
                    };
                    for (fname, fty) in &field_tys {
                        if let Some(declared_ty) = info.fields.get(*fname) {
                            let expected_ty = if shape_bindings.is_empty() {
                                declared_ty.clone()
                            } else {
                                substitute_shape_args(declared_ty.clone(), &shape_bindings)
                            };
                            let val = fields.iter()
                                .find(|(k, _)| k == *fname).map(|(_, v)| v);
                            if !expected_ty.compatible_with(fty) {
                                self.field_type_mismatch(
                                    fname, name, &expected_ty, fty, val, span.clone());
                            } else if let Some(hint) =
                                unproven_model_literal(&expected_ty, fty, val)
                            {
                                self.error_with_hint(
                                    format!("field `{}` of model `{}` expects `{}`, and the \
                                             literal given does not say what shape it is",
                                            fname, name, expected_ty),
                                    span.clone(), Some(hint));
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
                    // #474: the literal keeps the shape args it was written
                    // with. `Inner[4, 5] { … }` used to type as a bare `Inner`,
                    // throwing away the two numbers standing right there, so it
                    // could never unify with an `Inner[H, W]` field or
                    // parameter — the information needed was present at the
                    // construction site and discarded by the type.
                    //
                    // A literal written *without* a bracket still has its
                    // fields, and they say the same thing: a `!px` holding a
                    // 4x5 tensor pins `H` and `W` as surely as `Inner[4, 5]`
                    // does. Unify them, so a bare literal carries a real claim
                    // instead of an absent one that anything would satisfy.
                    TyType::Named { name: name.clone(), args: literal_args }
                } else {
                    self.error(format!("unknown model `{}`", name), span.clone());
                    TyType::Unknown
                }
            }
            BinOp { op, lhs, rhs, span } => self.check_binop(op.clone(), lhs, rhs, span.clone()),
            UnOp { operand, .. } => self.check_expr(operand),
            Postfix { expr, op, span } => self.check_postfix(expr, op, span.clone()),
            // #547: the cast's target type is a new *expectation* on the
            // outer expression, but it says nothing about the operand
            // underneath — an `as` cast doesn't change what the two sides of
            // an inner binop are, only what the result is reinterpreted as.
            // This arm used to resolve `ty` and stop, never recursing into
            // `expr` at all, so every check that lives in `check_expr` —
            // `check_binop`'s #295/#538/#539 operand-position checks among
            // them — silently never ran on a cast's operand, at any nesting
            // depth. Recurse for the side effects (errors/lints), same as
            // any other operand position.
            //
            // #550: the cast ITSELF is checked here too. This arm used to
            // resolve `ty` and return it on faith, so `s as i64` on a `str`
            // reported `Check OK` and `dmc run` then handed the *str* back out
            // of a function declared `-> i64` — the value's runtime kind did
            // not match the static type the checker had assigned it. `#549`
            // fixed the recursion into the operand and deliberately left the
            // cast alone; `check_cast` is the missing half.
            Cast { ty, expr, span } => {
                let mut from = self.check_expr(expr);
                // #248/#283's trick, reused: an arena constructor reports
                // `Unknown` on purpose, and so does a `let` bound to one, but
                // the side-table still knows the value is a tensor. Without
                // this the operand looks like ⊥ and no legality question about
                // it can be answered — which is exactly #550's second repro,
                // `let t = forge.zeros[f32, [2]]  t as i64`.
                if matches!(from, TyType::Unknown) {
                    if let Some(t) = self.ctor_tensor_fallback(expr) { from = t; }
                }
                let to = self.resolve_type(ty);
                self.check_cast(&from, &to, span.clone())
            }
            // #724: a bare `Expr::Range` has no concrete type of its own, but
            // `start`/`end` are still expressions that need checking — this
            // arm used to return `Unknown` without recursing into them, so an
            // undefined or import-private identifier inside *any* range
            // (a `for` bound, a value-position range, a call argument) sailed
            // through `--check` silently unless the range happened to sit at
            // a slice site (`t[a..b]`, #562), which checks its bounds via
            // `classify_index_axis` instead of `check_expr` and so never
            // reaches this arm at all. Recursing here — rather than only at
            // the `for`-statement's iterable, which is one position among
            // many a range can occupy — closes the leak everywhere at once.
            //
            // #744: #724 only closed the recursion; it left every bound's
            // *type* unconstrained, so `for k in 0..("x")` and `0..(1.5)`
            // type-checked clean and ran as silent zero-iteration loops
            // (`SPEC.md §4.3` is the range/slice grammar; a bound is an
            // `index_int`-checked integer expression there and everywhere
            // else a range appears — see the worked example at §7.4b,
            // `for i in 0..len(t)`). `check_range_bound` enforces that here,
            // for every position a
            // bare `Expr::Range` can occupy (value, call argument, and a
            // `for`'s iterable, which reaches this arm through its own
            // `check_expr(iter)` call — see the `Stmt::For` comment above).
            // The slice site (`t[a..b]`) is fixed separately, in
            // `PostfixOp::Index` below, because it never reaches this arm at
            // all (same reason #724's recursion had to be added there too).
            Range { start, end, .. } => {
                if let Some(s) = start { self.check_range_bound(s, "range start"); }
                if let Some(e) = end   { self.check_range_bound(e, "range end"); }
                TyType::Unknown
            }
        }
    }

    /// #744: a range/slice bound must have an integer type. `SPEC.md §4.3`
    /// is the range/slice grammar and treats `..`/`..=` and the colon slice
    /// form as the same grammar — "both spellings are permanent surface" —
    /// so the same constraint holds wherever a bound is written: a `for`'s
    /// iterable, a slice axis, a bare range value, or a call argument (an
    /// index bound is `index_int`-checked in `interp.rs`, matching `SPEC.md
    /// §4.3`'s own grammar for the other range positions). `is_integral_shape_arg`
    /// (#474) is reused rather than duplicated: an untyped integer literal, any
    /// integral scalar width (`i8`..`u64`, `int4`/`int8`), and a mixed-width
    /// pair (`0u8..4i64`) all pass unremarked — the interpreter iterates a
    /// range by value regardless of the two bounds' widths, confirmed
    /// against `dmc run` (`0u8..4u8`, `0..4u8`, `0u8..4i64` all run 0..4
    /// today) — and so does `Unknown` (a `map_get` result, a shape parameter
    /// reference, or any expression the checker can't pin down: refusing
    /// those here would reject working programs over a gap elsewhere, same
    /// reasoning as #474's own doc comment).
    ///
    /// One `Unknown`-typed case is deliberately excluded from that leniency:
    /// a bound that is itself syntactically a range (`(0..2)..3`). Its own
    /// `check_expr(Expr::Range)` call above types it `Unknown` — the same
    /// bottom `is_integral_shape_arg` waves through for a genuinely
    /// undecidable expression — but a nested range is not undecidable, it is
    /// a known wrong shape, and letting it through left `dmc run` to fail
    /// later instead (`t[(0..2)..3]` → `--check` OK, run panics "range start
    /// must be integer, got range"; `for i in (0..4)..8` checked clean and
    /// silently iterated `0..8`; both confirmed against the release binary).
    /// So it is caught here, syntactically, before the `Unknown` bottom can
    /// absorb it. `check_expr(e)` still runs first so a range-shaped bound's
    /// own inner bounds still get their #724 name-resolution recursion — only
    /// the type verdict on `e` itself is overridden. The syntactic test looks
    /// through parens: `parse_primary`'s `(expr)` parses to a one-element
    /// `Expr::Tuple` (there is no dedicated paren node), and `check_expr`'s own
    /// `Tuple` arm unwraps exactly that shape before typing it — `unwrap_parens`
    /// mirrors it so `((0..2))..3` and `(0..2)..3` are caught the same way.
    ///
    /// Otherwise, only a *known* non-integer type — `str`, a float width,
    /// `bool`, `nil`, a model — is refused, matching wording `interp.rs`'s
    /// `index_int` already uses for the same failure at run time (`"{what}
    /// must be integer, got {}"`) so a bound the checker misses (an
    /// `Unknown`-typed dynamic value) still fails the same way at the index
    /// site, just later (the `for`/value site is different: see
    /// `eval_range_bound`, which absorbs a non-int to 0 with no diagnostic —
    /// a pre-existing gap this fix does not close).
    fn check_range_bound(&mut self, e: &Expr, what: &str) {
        let ty = self.check_expr(e);
        if matches!(unwrap_parens(e), Expr::Range { .. }) {
            self.error(format!("{what} must be integer, got range"), e.span_of());
            return;
        }
        if !is_integral_shape_arg(&ty) {
            self.error(format!("{} must be integer, got {}", what, ty), e.span_of());
        }
    }

    /// #533: is this argument statically a packed `trit` tensor?
    ///
    /// Three spellings reach the port primitives, and all three are decidable
    /// at check time: an annotated binding or parameter (`Tensor[trit, …]`,
    /// which lands in `ty`), the constructor written inline
    /// (`port_tensor_encode(forge.trit[2, 2])`), and a `let` bound to one —
    /// which types as `Unknown`, because `forge.trit` carries no element-type
    /// argument, and so needs the side-table.
    fn is_trit_tensor(&self, e: &Expr, ty: Option<&TyType>) -> bool {
        if let Some(t) = ty {
            if let Some((elem, _)) = t.as_tensor_like() {
                if matches!(elem, TyType::Scalar(ScalarType::Trit)) { return true; }
            }
        }
        if is_trit_ctor(e) { return true; }
        matches!(e, Expr::Ident(name, _) if self.env.is_trit_binding(name))
    }

    /// The tensor type an expression carries when its *reported* type is
    /// `Unknown` — a literal arena constructor, or a simple binding whose RHS
    /// was one. The element type is only known on the syntactic form; through
    /// the side-table only the shape survives, which is enough to know the
    /// value is a tensor and not a scalar.
    fn ctor_tensor_fallback(&self, e: &Expr) -> Option<TyType> {
        ctor_tensor_ty(e).or_else(|| match e {
            Expr::Ident(name, _) => self.env.lookup_ctor_shape(name)
                .map(|s| TyType::Tensor(Box::new(TyType::Unknown), s.clone())),
            _ => None,
        })
    }

    /// #575: the tensor type `embed`'s `ids` argument carries, seen through
    /// the same `Unknown`-reporting gap `ctor_tensor_fallback` covers for the
    /// general case — but recovering the *element* half too, where
    /// `ctor_tensor_fallback` deliberately leaves it `Unknown` (its own doc
    /// comment: "through the side-table only the shape survives"). The
    /// element half comes from `Env::lookup_ctor_elem`, #575's counterpart to
    /// `lookup_ctor_shape`. Returns `Some` only when the element type is
    /// actually known; a `None` here means "the checker can't tell", never
    /// "not an integer" — the caller must leave those alone.
    fn embed_ids_ty(&self, e: &Expr, ty: Option<&TyType>) -> Option<TyType> {
        if let Some(t) = ty {
            if let Some((elem, shape)) = t.as_tensor_like() {
                if !matches!(elem, TyType::Unknown) {
                    return Some(TyType::Tensor(Box::new(elem.clone()), shape.clone()));
                }
            }
        }
        if let Some(t @ TyType::Tensor(_, _)) = ctor_tensor_ty(e) {
            return Some(t);
        }
        if let Expr::Ident(name, _) = e {
            if let (Some(elem), Some(shape)) =
                (self.env.lookup_ctor_elem(name), self.env.lookup_ctor_shape(name))
            {
                return Some(TyType::Tensor(Box::new(TyType::Scalar(elem.clone())), shape.clone()));
            }
        }
        None
    }

    /// #550: type an `expr as Type`, rejecting a cast the language defines no
    /// conversion for. Reuses the JIT's wording — `cannot convert `str` to
    /// `i64`` — because the JIT already refuses exactly these programs and the
    /// two backends should not describe the same bad program differently.
    fn check_cast(&mut self, from: &TyType, to: &TyType, span: Span) -> TyType {
        if !cast_is_legal(from, to) {
            self.error(
                format!("cannot convert `{}` to `{}`", render_ty(from), render_ty(to)),
                span,
            );
            return TyType::Unknown;
        }
        cast_result_ty(from, to)
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
            DotAdd | DotSub | DotMul | DotDiv | DotPow
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
                // #538/#539: literal adoption+range-check and the
                // integral-mismatch check are shared with comparisons and
                // bitwise/shift below — see `adopt_and_check_operand_types`.
                if lt.is_numeric() && rt.is_numeric() {
                    self.adopt_and_check_operand_types(&lt, &rt, lhs, rhs, &span)
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
            Eq | NotEq | Lt | Gt | LtEq | GtEq => {
                // #538/#539: a comparison's *result* is always `bool` — never
                // adopted from an operand — but its operands are still an
                // untyped/suffixed literal's use site (SPEC §3.1) and are
                // just as subject to #295's range check and #284's
                // strict-typing rule as an arithmetic operand is. Run the
                // shared check for its side effects only; a non-numeric
                // comparison (str == str, enum == enum, bool == bool, …)
                // skips it exactly as before.
                if lt.is_numeric() && rt.is_numeric() {
                    self.adopt_and_check_operand_types(&lt, &rt, lhs, rhs, &span);
                }
                TyType::Scalar(ScalarType::Bool)
            }
            Pipe | RShift => {
                // `x \|> f` pipes into a callable RHS. The placeholder form
                // (`x |> _ .+ b`) and bare activation stages (`\|> \>`, which
                // parse to an ident) are valid non-callable RHS forms. Catch
                // the footgun where the RHS is a concrete *value*: it would
                // type-check and then fail at runtime with "expected callable".
                // (#188's `x >> 2` shape no longer reaches here at all: `>>`
                // is the right shift since #530 and types in the bitwise arm
                // below. `RShift` is unreachable — the token parses to
                // `BitShr`, and nothing constructs this variant.)
                let rhs_is_value = rt.is_numeric() || rt.as_tensor_like().is_some();
                if rhs_is_value && !expr_contains_underscore(rhs) {
                    self.error_with_hint(
                        format!("`\\|>` pipes into a callable, but the right side is {rt}, not callable"),
                        span,
                        Some("for an elementwise stage use a `_` placeholder (e.g. `x \\|> _ .+ y`)".to_string()),
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
                    // #538/#539: same shared check as arithmetic and
                    // comparisons — a literal operand (`a & 5000000000`) is
                    // range-checked against what it adopts, and a
                    // differently-typed suffix or declared width (`a & 2i64`)
                    // disagrees, matching the JIT's single `lower_binop`
                    // guard across every scalar binop uniformly.
                    self.adopt_and_check_operand_types(&lt, &rt, lhs, rhs, &span)
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

    /// #538/#539: the operand-position half of #295's literal range check
    /// and #445's suffix-conflict rule, shared by every scalar binop that
    /// reaches this point with two already-numeric operands — arithmetic,
    /// comparison, and bitwise/shift alike. Mirrors the JIT's single
    /// `adopt_int_literal_kind` guard in `lower_binop`, which enforces both
    /// rules uniformly across every scalar binop; before this helper existed
    /// `check_binop` only enforced them in the arithmetic arm, so `a & 2i64`
    /// or `a < 5000000000` (a: i32) checked clean and only the JIT refused
    /// them — the three-way `--check`/`run`/`jit` split #538 and #539 exist
    /// to close.
    ///
    /// Returns the type an untyped/float-promoted operand *adopts* — the
    /// arithmetic and bitwise/shift callers use this as their own result
    /// type (matching the pre-#538/#539 behavior when both operands already
    /// agree). A comparison's result is always `bool`, never this return
    /// value — its caller calls this purely for the check side effects and
    /// discards what comes back.
    ///
    /// Precondition (checked by every caller before calling this):
    /// `lt.is_numeric() && rt.is_numeric()`.
    fn adopt_and_check_operand_types(
        &mut self,
        lt: &TyType,
        rt: &TyType,
        lhs: &Expr,
        rhs: &Expr,
        span: &Span,
    ) -> TyType {
        // A literal operand adopts the other (concrete) operand's type;
        // two literals stay an untyped literal (#295). Otherwise result =
        // lhs type (pre-alpha: no widening).
        match (lt, rt) {
            // Two untyped literals of the same kind stay that kind.
            (TyType::IntLit(_), TyType::IntLit(_)) => lt.clone(),
            (TyType::FloatLit(_), TyType::FloatLit(_)) => lt.clone(),
            // Mixed untyped literals: the float kind wins (the int
            // literal promotes to float).
            (TyType::IntLit(_), TyType::FloatLit(_)) => rt.clone(),
            (TyType::FloatLit(_), TyType::IntLit(_)) => lt.clone(),
            // #538: an untyped int literal adopts the other (concrete)
            // operand's type in either operand order — and, like the
            // annotation/return/param contexts (#295), its magnitude must
            // fit what it adopts. `check_int_literal_range` no-ops when the
            // adopted target isn't a range-checked integral scalar (e.g.
            // the other operand is itself untyped, or float), so this is
            // safe to call unconditionally here.
            (TyType::IntLit(_), _) => { self.check_int_literal_range(rt, lhs); rt.clone() }
            (_, TyType::IntLit(_)) => { self.check_int_literal_range(lt, rhs); lt.clone() }
            // An untyped float literal adopts the other (concrete) operand;
            // floats aren't range-checked (#295 is int-only).
            (TyType::FloatLit(_), _) => rt.clone(),
            (_, TyType::FloatLit(_)) => lt.clone(),
            // #473: two concrete floats of different widths — the WIDER one
            // wins. Both backends compute `f32 + f64` in f64 (the f32
            // operand promotes; mixed float arithmetic never silently
            // narrows), so typing it as the lhs made an expression whose
            // runtime value IS an f64 unbindable to an `f64` — `let w: f64 =
            // a_f32 + b_f64` was rejected. Same-width pairs and integer
            // arithmetic keep the lhs rule unchanged.
            (TyType::Scalar(a), TyType::Scalar(b))
                if is_f32_family(a) && matches!(b, ScalarType::F64) => rt.clone(),
            (TyType::Scalar(a), TyType::Scalar(b))
                if matches!(a, ScalarType::F64) && is_f32_family(b) => lt.clone(),
            // #539: two concrete *integral* scalars — one may be a
            // suffix-typed literal, which types exactly as concretely as a
            // declared variable (SPEC §3.1) — of different kinds never
            // silently combine. This is the same #284 strict-typing rule
            // already enforced at the annotation/return/param positions,
            // reaching the operand position. Deliberately narrower than
            // `compatible_with`: scoped to the *integral* family only, so
            // `f64 ** i64` (a float base with an integer exponent — `**`
            // always computes in f64 regardless of the exponent's declared
            // width, both backends agree, and it's an established idiom —
            // see `examples/translations/adamw_step.dmc`) keeps checking
            // clean; only same-family, different-width integers (or an
            // integer suffix conflicting with one) are flagged.
            _ => {
                if lt.is_integral() && rt.is_integral() && lt != rt {
                    self.error(
                        format!("operand types disagree: {} vs {} (no implicit cast)", lt, rt),
                        span.clone(),
                    );
                }
                lt.clone()
            }
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

    /// Type an expression that has already been typed once on this path, and
    /// drop whatever it reports. Used where a sub-expression's type is needed
    /// a second time (a bracketed method's receiver): re-checking is cheap and
    /// idempotent, but re-reporting would duplicate every diagnostic inside it.
    fn quiet_ty(&mut self, e: &Expr) -> TyType {
        let (errs, warns) = (self.errors.len(), self.warnings.len());
        let t = self.check_expr(e);
        self.errors.truncate(errs);
        self.warnings.truncate(warns);
        t
    }

    /// #474: type `recv.method![shape args]` — the shape bracket standing
    /// between a model method and its call.
    ///
    /// `b.blit![2, 2](src)` parses as `Call(Index(Field(b, "blit!"), [2, 2]),
    /// [src])`, so the method-call resolution in the `Call` arm never sees a
    /// method: the callee typed as `Unknown` and the call inherited it. No
    /// arity check, no argument check, and a result that unified with
    /// anything — `let ok: bool = b.blit![2, 2](src)` on a `-> i64` method
    /// passed `--check`. The interpreter dispatches this spelling for real
    /// (`call_bracketed_method`); the checker has to see the same signature.
    ///
    /// The returned `Fn` drops the receiver parameter — the call site does not
    /// write it — and has every shape argument substituted: the model's own,
    /// read off the receiver's type, then the method's, read off the bracket.
    /// The bracket shadows a reused name, which is the only reading that makes
    /// writing it mean anything, and is what the interpreter does.
    ///
    /// `None` leaves the call to the path that owns it: a non-model receiver,
    /// a *field* of that name being indexed (`b.handlers[i](x)`), or a name
    /// that is no member at all — the last is #441's undefined method, whose
    /// error the caller raises.
    ///
    /// Called only from the `Call` arm. A bracket that is not called is not a
    /// working value (see the note there), so it keeps typing as `Unknown`.
    /// #474: `b.generic![2, 2]` with no call after it.
    ///
    /// The bracket is the shape-argument half of a method call, and a method
    /// call is the only thing it can be part of — there is no value in this
    /// language that means "the method `generic!` of `b`, with `SH` and `SW`
    /// fixed". Left alone it typed as `Unknown`, passed `--check`, and at run
    /// time evaluated to an opaque and did nothing: a statement that looks
    /// exactly like the call the author meant to write, and silently is not
    /// one. That is the same ghost the bracketed dispatch was written to kill,
    /// wearing its last spelling, so it is refused for the same reason.
    fn reject_uncalled_method_bracket(&mut self, expr: &Expr, op: &PostfixOp, span: &Span) {
        if !is_shape_bracket(op) { return; }
        let Expr::Postfix { expr: base, op: PostfixOp::Field(method), .. } = expr else { return };
        let TyType::Named { name: model, .. } = self.quiet_ty(base) else { return };
        let Some(info) = self.env.models.get(&model) else { return };
        // A field of that name being indexed is ordinary indexing, not this.
        if info.fields.contains_key(method) || !info.methods.contains_key(method) { return; }
        self.error_with_hint(
            format!("`{model}.{method}` is a method, and a shape bracket on it is \
                     part of calling it — `{method}[…]` on its own is not a value"),
            span.clone(),
            Some(format!("call it: `{method}[…](args)`")),
        );
    }

    fn bracketed_method_ty(&mut self, expr: &Expr, op: &PostfixOp, span: &Span) -> Option<TyType> {
        let bracket = shape_bracket_pairs(op)?;
        let Expr::Postfix { expr: base, op: PostfixOp::Field(method), .. } = expr else {
            return None;
        };
        let TyType::Named { name: model, args: model_args } = self.quiet_ty(base) else {
            return None;
        };
        let info = self.env.models.get(&model)?.clone();
        if info.fields.contains_key(method) { return None; }
        let sig = info.methods.get(method)?.clone();
        let owner = format!("{model}.{method}");

        let mut binds: HashMap<String, SymDim> = info.shape_params.iter()
            .zip(model_args.iter())
            .filter_map(|(p, a)| match a {
                TyType::Dim(d) => Some((p.clone(), d.clone())),
                _ => None,
            })
            .collect();
        let mut pos = 0usize;
        let mut bound: Vec<String> = Vec::new();
        for (name, value) in &bracket {
            let pname = match name {
                None => match sig.shape_params.get(pos) {
                    Some(p) => { pos += 1; p.clone() }
                    None => {
                        self.error(format!(
                            "`{}` declares {} shape parameter(s), got more bracket args",
                            owner, sig.shape_params.len()), span.clone());
                        return Some(TyType::Unknown);
                    }
                },
                Some(n) => {
                    if !sig.shape_params.iter().any(|p| p == n) {
                        self.error(format!(
                            "`{}` is not a shape parameter of `{}` (declared: {})",
                            n, owner, sig.shape_params.join(", ")), span.clone());
                        return Some(TyType::Unknown);
                    }
                    (*n).to_string()
                }
            };
            if bound.contains(&pname) {
                self.error(format!(
                    "shape parameter `{}` of `{}` bound twice", pname, owner), span.clone());
                return Some(TyType::Unknown);
            }
            // The positional spelling reaches here through the `Index` arm,
            // which has already typed these; the named one does not, and its
            // arguments would otherwise never be checked at all.
            let vty = if matches!(op, PostfixOp::Index(_)) {
                self.quiet_ty(value)
            } else {
                self.check_expr(value)
            };
            // #792 rework: a bracket arg spelled as a bare scalar-type
            // keyword (`mk[i64, 4]()`) is a TYPE, not a shape dimension —
            // `check_expr`/`quiet_ty` resolve it to the opaque type-marker
            // `Unknown` (see their own `Ident` arm), which `is_integral_
            // shape_arg` waves through below same as any other `Unknown`,
            // so it silently became a `SymDim::Var`/`Wildcard` standing in
            // for a type. `substitute_shape_args` only ever substitutes
            // dims, never a `Named` type's own name, so nothing downstream
            // could have replaced it either way — refused here, naming the
            // position, rather than left to surface as a confusing later
            // mismatch (e.g. a fn's return type "producing" the bracket
            // name itself).
            if let Expr::Ident(id_name, _) = value {
                if is_scalar_type_keyword(id_name) {
                    self.error(format!(
                        "shape argument `{}` of `{}` is a type (`{}`), not a \
                         dimension — demoniC's shape brackets bind integer \
                         dims; a type parameter in this position is not \
                         supported", pname, owner, id_name), span.clone());
                    return Some(TyType::Unknown);
                }
            }
            // A shape argument is a dim. `b.generic!["x", 2](s)` used to reach
            // the interpreter and die there, while the arity, the names and
            // the duplicate check all reported at `--check` — the one member
            // of the family that waited until run time for no reason.
            if !is_integral_shape_arg(&vty) {
                self.error(format!(
                    "shape argument `{}` of `{}` must be an integer, got {}",
                    pname, owner, vty), span.clone());
                return Some(TyType::Unknown);
            }
            binds.insert(pname.clone(), self.resolve_bracket_dim(value));
            bound.push(pname);
        }

        let drop_self = usize::from(sig.params.first().is_some_and(|(n, _)| n == "self"));
        Some(TyType::Fn {
            params: sig.params.iter().skip(drop_self)
                .map(|(_, t)| substitute_shape_args(t.clone(), &binds))
                .collect(),
            ret: Box::new(substitute_shape_args(sig.ret.clone(), &binds)),
        })
    }

    /// #792 rework: fold an explicit shape-bracket argument's expression to
    /// a dim, the way `resolve_dim` folds any other dim slot — a local
    /// `let` constant or a top-level `const_dims` name becomes `Const`
    /// (`let k = 4  mk[k]()` must bind `N` to `Const(4)`, not leave it an
    /// unbound `Var` that a concrete dim can only ever fail to match). A
    /// name that survives resolution as a `Var` and is NOT an in-scope
    /// shape parameter is a genuinely unanalyzable expression (a runtime
    /// value, not a constant) — such a dim is `Wildcard` (matches
    /// anything) rather than a hard mismatch, since the checker has no
    /// basis to refuse it and refusing would only false-positive. An
    /// in-scope shape parameter stays `Var`: it is meaningful to whatever
    /// unifies against it later (`infer_call_shape_bindings`,
    /// `substitute_shape_args`), not merely unresolved.
    fn resolve_bracket_dim(&self, value: &Expr) -> SymDim {
        let dim = self.resolve_dim(SymDim::from_expr(value).simplify());
        match dim {
            SymDim::Var(ref name) if !self.env.shape_param_in_scope(name) => SymDim::Wildcard,
            other => other,
        }
    }

    /// #792: `mk[8]()` — an explicit shape bracket between a *free* function's
    /// name and its call, instantiating the function's own shape parameters
    /// (SPEC §3.5's `fn name[shape_vars](...)`, and the §3.10 `area[2, 3](b)`
    /// spelling for the case argument-based inference has nothing to bind
    /// from — a shape param of a zero-arg fn, or one that reuses a lowercase
    /// name inference never binds). Mirrors `bracketed_method_ty` above for
    /// the free-function case: no receiver to drop, and the callee may take
    /// no value parameters at all (`mk[N]()`), which is exactly the case this
    /// bracket exists for.
    ///
    /// A shape parameter this bracket does not bind is left as its own
    /// `SymDim::Var` — the ordinary argument-inference pass in the `Fn` arm
    /// below (`infer_call_shape_bindings`) still runs against the substituted
    /// signature and fills it in from a value argument when one is present,
    /// so a partial explicit bracket and inference compose the same way they
    /// do for a bracketed method call.
    ///
    /// SPEC §3.10: "inference binds uppercase-initial names only" — and, per
    /// `jit_shape_arg_inference_732_733.rs`'s
    /// `a_lowercase_shape_param_spelled_at_the_call_is_never_checked_against_the_argument`,
    /// a lowercase name spelled explicitly at the call is not merely exempt
    /// from inference; the interpreter and JIT never check its agreement
    /// with the argument either — `cap[8](a)` on a `Buf[4]` answers rather
    /// than refusing, "in flat contradiction of the instance". A *parameter*
    /// type this bracket substitutes therefore uses `SymDim::Wildcard`
    /// (matches any dim in `SymDim::equivalent`) for a lowercase-bound name
    /// instead of the literal value, so that axis of the argument comparison
    /// is skipped exactly where the two backends already skip it — while an
    /// uppercase-bound name, and the *return* type either way, substitute the
    /// real value and stay checked.
    ///
    /// `None` when `expr` is not a bare identifier, is a local binding that
    /// shadows a top-level fn of the same name (a plain `Fn`-typed value —
    /// the ordinary call path handles it), or names a function with no shape
    /// parameters at all (an ordinary index/tensor-slice bracket keeps its
    /// existing meaning). Called only from the `Call` arm, same as
    /// `bracketed_method_ty` — a bracket that is never called stays `Unknown`,
    /// same as it always has for a free function.
    fn fn_bracket_ty(&mut self, expr: &Expr, op: &PostfixOp, span: &Span) -> Option<TyType> {
        let bracket = shape_bracket_pairs(op)?;
        let Expr::Ident(fn_name, _) = expr else { return None; };
        // A local binding of the same name shadows the top-level fn — leave
        // this to the ordinary indexing/call path.
        if self.env.lookup(fn_name).is_some() { return None; }
        let sig = self.env.functions.get(fn_name)?.clone();
        if sig.shape_params.is_empty() { return None; }

        let mut binds: HashMap<String, SymDim> = HashMap::new();
        let mut pos = 0usize;
        let mut bound: Vec<String> = Vec::new();
        for (name, value) in &bracket {
            let pname = match name {
                None => match sig.shape_params.get(pos) {
                    Some(p) => { pos += 1; p.clone() }
                    None => {
                        self.error(format!(
                            "`{}` declares {} shape parameter(s), got more bracket args",
                            fn_name, sig.shape_params.len()), span.clone());
                        return Some(TyType::Unknown);
                    }
                },
                Some(n) => {
                    if !sig.shape_params.iter().any(|p| p == n) {
                        self.error(format!(
                            "`{}` is not a shape parameter of `{}` (declared: {})",
                            n, fn_name, sig.shape_params.join(", ")), span.clone());
                        return Some(TyType::Unknown);
                    }
                    (*n).to_string()
                }
            };
            if bound.contains(&pname) {
                self.error(format!(
                    "shape parameter `{}` of `{}` bound twice", pname, fn_name), span.clone());
                return Some(TyType::Unknown);
            }
            // Same split as `bracketed_method_ty`: the positional spelling's
            // values are already typed by the `Index` arm's own indexing
            // loop, so re-checking them would duplicate diagnostics; the
            // named spelling (`BracketArgs`) is never typed elsewhere.
            let vty = if matches!(op, PostfixOp::Index(_)) {
                self.quiet_ty(value)
            } else {
                self.check_expr(value)
            };
            // #792 rework: see `bracketed_method_ty`'s identical guard —
            // `mk[i64, 4]()` binds a TYPE where a dim is expected.
            if let Expr::Ident(id_name, _) = value {
                if is_scalar_type_keyword(id_name) {
                    self.error(format!(
                        "shape argument `{}` of `{}` is a type (`{}`), not a \
                         dimension — demoniC's shape brackets bind integer \
                         dims; a type parameter in this position is not \
                         supported", pname, fn_name, id_name), span.clone());
                    return Some(TyType::Unknown);
                }
            }
            if !is_integral_shape_arg(&vty) {
                self.error(format!(
                    "shape argument `{}` of `{}` must be an integer, got {}",
                    pname, fn_name, vty), span.clone());
                return Some(TyType::Unknown);
            }
            binds.insert(pname.clone(), self.resolve_bracket_dim(value));
            bound.push(pname);
        }

        // Parameter-position substitution only: a lowercase-bound name
        // becomes a wildcard dim rather than its literal value, so it stays
        // permissive against whatever the caller's argument actually is —
        // see the doc comment above.
        let param_binds: HashMap<String, SymDim> = binds.iter()
            .map(|(name, dim)| {
                if name.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
                    (name.clone(), SymDim::Wildcard)
                } else {
                    (name.clone(), dim.clone())
                }
            })
            .collect();

        Some(TyType::Fn {
            params: sig.params.iter()
                .map(|(_, t)| substitute_shape_args(t.clone(), &param_binds))
                .collect(),
            ret: Box::new(substitute_shape_args(sig.ret.clone(), &binds)),
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
        // #476 (MEMORY §2): reading a model-array field that is still straight
        // out of `uninit`. Reported here, at the read, rather than left to the
        // runtime — which used to hand back an `Opaque` and fail later at the
        // first *use*, naming neither the field nor initialization.
        if let (Expr::Ident(root, _), PostfixOp::Field(fname)) = (expr, op) {
            if self.env.is_uninit_field(root, fname) {
                self.env.clear_uninit_field(root, fname); // one bug, one report
                self.error(
                    format!(
                        "read of uninitialized memory: `{root}.{fname}` comes from \
                         `forge.uninit`/`vault.uninit` and nothing has been written to \
                         it yet. Filling it through a copy (`let !cs = {root}.{fname}`) \
                         does not reach the field — write the elements through \
                         `{root}.{fname}[i]`, or build the array as a local and store \
                         it at construction"),
                    span.clone(),
                );
            }
        }
        // #476: a model instance aliases through its `Rc`, so anything handed
        // the binding may fill the field. Give up the marks rather than risk a
        // false report: on a method call through the receiver, and on any bare
        // identifier passed as an argument.
        if let PostfixOp::Call(args) = op {
            if let Expr::Postfix { expr: recv, op: PostfixOp::Field(_), .. } = expr {
                if let Expr::Ident(r, _) = recv.as_ref() {
                    self.env.clear_uninit_fields_under(r);
                }
            }
            for a in args {
                if let CallArg::Positional(Expr::Ident(n, _)) = a {
                    self.env.clear_uninit_fields_under(n);
                }
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
        // #474: the marker applies to exactly one postfix level. Take it on
        // entry so nothing nested inherits it, then re-arm it for this node's
        // own callee if this node is a call.
        let is_callee = std::mem::take(&mut self.typing_callee);
        if matches!(op, PostfixOp::Call(_)) { self.typing_callee = true; }
        let recv = self.check_expr(expr);
        self.typing_callee = false;
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
                if !is_callee { self.reject_uncalled_method_bracket(expr, op, &span); }
                // R3(i) (`budget-arena`, #674 §2.3): `forge.kv`/`stream.kv`
                // lexically inside a `forge.enter(b) { … }` extent, with NO
                // `in` override — B3 would otherwise charge this ambiently to
                // the entered budget, but a `.kv` allocation always lands in
                // the Stream arena regardless of prefix (`MEMORY.md §9`), so
                // no budget can own it. The explicit-override twin of this
                // check lives in `Expr::AllocIn`'s own arm (unconditional of
                // `budget_stack`, since an override doesn't care about lexical
                // nesting). A bare `vault.*` ctor is deliberately NOT flagged
                // here — `MEMORY.md §3.2`/§3.5: it charges the root, unchanged.
                // `checking_alloc_in_ctor` suppresses this ambient instance
                // ONLY when `span` is the exact node `Expr::AllocIn` is
                // checking as its own ctor half (span equality, not a bare
                // flag — see that field's own doc comment), so an overridden
                // `.kv` spine reports once but a DIFFERENT `.kv` spine nested
                // inside that ctor's own index/shape arguments still reports.
                if !self.budget_stack.is_empty()
                    && self.checking_alloc_in_ctor.as_ref() != Some(&span)
                {
                    if let Some((arena, method)) = arena_ctor_receiver(expr) {
                        if method == "kv" && matches!(arena, "forge" | "stream") {
                            self.error(
                                format!(
                                    "budget-arena: `{arena}.kv` allocates in the Stream \
                                     arena, which no budget owns; allocate the cache \
                                     before entering the budget"),
                                span.clone(),
                            );
                        }
                    }
                }
                // #728: a builtin's own model type argument, bare —
                // `forge.uninit[Surface, [N]]` — the sixth annotation
                // position the ratified decision's predicate applies to.
                // Detected structurally rather than via `resolve_type`: this
                // slot is parsed as an *expression* (`Surface[2, 3]` indexes
                // the value `Surface`), so there is no `ast::Type` here to
                // read the "was a bracket written" bit off — only the shape
                // of `elems[0]` says it. A bracketed argument parses as a
                // nested `Postfix { .., op: Index(_) }`, not a bare `Ident`,
                // so it never matches the pattern below.
                //
                // #728 rework (reviewer defect 3): `model_array_alloc_arena_
                // of` matches the full model-array allocator set (`zeros`/
                // `ones`/`uninit` — the same set `ctor_tensor_ty` matches for
                // a scalar element, `MODEL_ARRAY_ALLOCATOR_METHODS`), not
                // `uninit` alone — `forge.zeros[Surface, [4]]` and
                // `forge.ones[Surface, [4]]` allocate a model array exactly
                // as `forge.uninit[Surface, [4]]` does and escaped this lint
                // before this fix.
                if let Some((arena, method)) = model_array_alloc_arena_of(expr) {
                    if let Some(IndexElem::Expr(Expr::Ident(mname, mspan))) = elems.first() {
                        if let Some(shape_params) = self.bare_generic_model(mname, true) {
                            let shape_params = shape_params.to_vec();
                            let site = match self.current_fn_desc() {
                                Some(desc) => format!("the element type of `{arena}.{method}` in {desc}"),
                                None => format!("the element type of `{arena}.{method}`"),
                            };
                            let (mname, mspan) = (mname.clone(), mspan.clone());
                            let declared = self.current_fn_shape_params.clone();
                            // This position has no `ast::Type` at all (the
                            // model argument is parsed as an expression, per
                            // the doc comment above) — synthesize a bare
                            // `Type::Named` standing for exactly what was
                            // written, so `type_arg_less_hint`'s renderer has
                            // something to substitute into. Since its own
                            // span IS `match_span`, the substitution reduces
                            // to the bracketed replacement outright — same
                            // hint text a bare top-level `Named` annotation
                            // would get.
                            let synth_ty = Type::Named { name: mname.clone(), args: Vec::new(), span: mspan.clone() };
                            self.warn_type_arg_less_generic_model(&synth_ty, &mname, &shape_params, site, mspan, &declared, None, &[]);
                        }
                    }
                }
                // #799 follow-up (PR #824 review): the allocator's own
                // shape argument (`forge.ones[f32, [N]]`, `forge.zeros
                // [Surface, [N]]`) is refused the same way every other
                // annotation position is when a dimension names a mutable
                // top-level `let` — SPEC.md §6.6 says the rule applies to
                // "any [type] annotation", and an allocator's own type
                // argument bracket IS one, exactly as #728's type-arg-
                // less-model ruling just above already treats this same
                // bracket. Structural, like that check and `ctor_tensor_
                // ty`/`model_array_alloc_arena_of`: this position is
                // parsed as an expression (the shape bracket indexes the
                // allocator call), so there is no `ast::Type` here for
                // `resolve_type`/`resolve_dim_at`'s usual annotation walk
                // to reach — `mutable_let_dim_name` is applied directly to
                // each dimension expression's `SymDim::from_expr` reading
                // instead, sharing `resolve_dim_at`'s own wording and its
                // `reported_mutable_dim_spans` dedup set, so a shape
                // re-examined by both this check and a later annotation
                // walk over the SAME expression still reports once.
                if model_array_alloc_arena_of(expr).is_some() {
                    let args: Vec<&Expr> = elems.iter().filter_map(|e| match e {
                        IndexElem::Expr(x) => Some(x),
                        _ => None,
                    }).collect();
                    if args.len() == 2 {
                        let dim_exprs: Vec<&Expr> = match args[1] {
                            Expr::TensorLit(dims, _) => dims.iter().collect(),
                            other => vec![other],
                        };
                        for de in dim_exprs {
                            let dim = SymDim::from_expr(de);
                            if let Some(name) = self.mutable_let_dim_name(&dim) {
                                let sp = de.span_of();
                                let key = (name.clone(), sp.start, sp.end, sp.line, sp.col);
                                if self.reported_mutable_dim_spans.insert(key) {
                                    self.error(
                                        format!(
                                            "dimension `{}` names a mutable top-level \
                                             `let` (`let !`/`let mut`) in this module \
                                             — a mutable binding is never a constant \
                                             dimension (SPEC.md §6.6)",
                                            name),
                                        sp,
                                    );
                                }
                            }
                        }
                    }
                }
                if let Some(ty) = self.tensor_split_type(expr, op) { return ty; }
                // Type-check each index expr; result shape pruned per scalar
                // indices. #562: `t[a..b]` parses as `IndexElem::Expr(Expr::Range)`
                // (see parser.rs's `parse_index_elem_or_arg`, and jit.rs/interp.rs's
                // own "not `IndexElem::Slice`" notes), so its bounds must be
                // checked explicitly here too — `classify_index_axis` unpacks
                // `start`/`end` before this loop ever calls `check_expr` on
                // the whole bracket element, so neither spelling reaches the
                // `Range` arm in `check_expr` (its own #744 fix) at all.
                //
                // #744: `check_range_bound` (below `check_expr`) is the same
                // integer constraint the `Range` arm now applies elsewhere —
                // a `str`/float/bool/… start, end, or step here used to type
                // clean and panic at run time instead. The label must match
                // which of the two `SPEC.md §4.3` spellings this element is:
                // `interp.rs`'s `eval_index_elems` names a `..`/`..=` bound
                // ("range start"/"range end", no step — that spelling has
                // none) and a colon-form bound ("slice start"/"slice
                // end"/"slice step") differently, but `classify_index_axis`
                // deliberately collapses both into one `IndexAxis::Slice` (it
                // exists to decide "scalar vs. slice", not to keep the two
                // spellings apart), so the label has to be read off `e`
                // itself, before that collapse, or `t[0..("x")]` reports
                // "slice end must be integer" for a bound the interpreter
                // would have called a "range end" (confirmed mismatched
                // against the release binary).
                for e in elems {
                    let is_range = matches!(e, IndexElem::Expr(Expr::Range { .. }));
                    match classify_index_axis(e) {
                        IndexAxis::Scalar(e) => { let _ = self.check_expr(e); }
                        IndexAxis::Full => {}
                        IndexAxis::Slice { start, end, step, .. } => {
                            let (w_start, w_end) = if is_range {
                                ("range start", "range end")
                            } else {
                                ("slice start", "slice end")
                            };
                            if let Some(e) = start { self.check_range_bound(e, w_start); }
                            if let Some(e) = end   { self.check_range_bound(e, w_end); }
                            if let Some(e) = step  { self.check_range_bound(e, "slice step"); }
                        }
                    }
                }
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
                // Only return element type for full scalar indexing (no slice
                // elems anywhere in the bracket) — `..` and `:` are two
                // spellings of the same axis-slice and must agree (#562).
                let all_scalar = elems.iter().all(|e| matches!(classify_index_axis(e), IndexAxis::Scalar(_)));
                if all_scalar {
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
                    return TyType::Unknown;
                }
                // #562: at least one elem is a slice/full-slice — the result is
                // a tensor, never the element type. Derive the real sliced shape
                // when every slice axis' extent is expressible in the SymDim
                // algebra (handles both `x[0..S]` and the derived `x[0..S/2]`);
                // fall back to `Unknown` rather than assert a shape that might
                // be wrong (e.g. a stepped slice, or a negative-literal bound
                // that needs Python-style from-the-end resolution).
                if let Some((elem_ty, shape)) = recv_shape {
                    if !matches!(elem_ty, TyType::Unknown) {
                        if let Some(mut dims) = derive_slice_shape(elems, &shape) {
                            if elems.len() < shape.rank() {
                                dims.extend(shape.dims[elems.len()..].iter().cloned());
                            }
                            return TyType::Tensor(Box::new(elem_ty.clone()), Shape::new(dims));
                        }
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

                // PORTS.md §5: no `@grad fn`, `@fuse` block, or `@deterministic`
                // block may call a port. The call is an effect boundary — the
                // tape cannot record it, fusion cannot cross it, and no port yet
                // carries the manifest determinism needs. Reject at compile time
                // rather than silently break the enclosing promise.
                //
                // The list is exactly the three that reach a runtime.
                // `port_tensor_encode` / `port_tensor_decode` share the prefix
                // but not the property: they are pure value transforms with no
                // runtime on the other end (SPEC.md §4.11), so they stay legal
                // here. `check_tests::the_copy_mode_primitives_are_not_port_calls`
                // pins that, because the distinction lives only in this list.
                if let Some(ban) = self.port_ban {
                    if let Expr::Ident(name, _) = expr {
                        if matches!(name.as_str(), "port_open" | "port_call" | "port_close") {
                            self.error(
                                format!("port-forbidden: `{}` is illegal inside a {} — {} \
                                         (PORTS.md §5)", name, ban.what(), ban.because()),
                                span.clone(),
                            );
                        }
                        // #578: the spec's `extern fn` rules forbid a call from
                        // exactly these three constructs, for the same reason
                        // the port rule above does — a foreign call is an
                        // effect boundary. `@comptime` is the fourth, already
                        // enforced by its own total ban on calls
                        // (`comptime-non-static`), so it is not repeated here.
                        //
                        // The `@deterministic` row is what makes "no foreign
                        // accumulation order inside `@deterministic`" a
                        // property of the language rather than of one
                        // kernel-selection arm (#578's BLAS fast path).
                        else if self.extern_fns.contains(name) {
                            self.error(
                                format!("extern-context: `extern fn {}` is illegal inside a {} \
                                         — {} (see the spec's `extern fn` rules)",
                                        name, ban.what(), ban.extern_because()),
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

                // #533 (PORTS.md §3.2): "`trit` has no wire dtype. A packed
                // ternary weight is a demoniC storage format, not a portable
                // element type." Both backends refuse it, but only at run time,
                // so a program that cannot run passed `--check` clean. The set
                // of encodable element types is a property of the *argument's
                // type*, never of any value, which makes this a compile-time
                // error by the same reasoning as AGENTS.md §2.5. Same words as
                // the two backends — `interp.rs`'s `port_tensor_encode` arm and
                // the JIT's `port_tensor_wire_ty`, pinned by #512's
                // `jit_a_trit_tensor_is_refused_the_way_the_interpreter_refuses_it`.
                if let Expr::Ident(name, _) = expr {
                    let trit_arg = match name.as_str() {
                        // `port_tensor_encode(t)`
                        "port_tensor_encode" => Some(0),
                        // `port_tensor_decode(s, like)` — `like` declares the
                        // payload buffer, so its dtype is the one being asked for.
                        "port_tensor_decode" => Some(1),
                        _ => None,
                    };
                    if let Some(i) = trit_arg {
                        if let Some(CallArg::Positional(e)) = args.get(i) {
                            if self.is_trit_tensor(e, arg_tys.get(i)) {
                                self.error(format!(
                                    "{}: a `trit` tensor has no copy-mode wire dtype \
                                     (PORTS.md §3.2)", name),
                                    span.clone());
                            }
                        }
                    }
                }

                // #575: `embed(vocab, ids)` declares `ids: Tensor[i64, [...B]]`
                // (STDLIB.md §3.6), but nothing enforced it — a float `ids`
                // tensor passed `--check` clean. The two backends then
                // disagreed at run time: the interpreter truncated each float
                // id and gathered the row, while the JIT refused with `` `embed`:
                // ids must be an integer tensor ``. That split is the bug, not
                // the JIT's refusal (deliberately not reclassified as a JIT gap
                // in #577's sweep) — an index has no defined meaning as a
                // float, so this is a hole in the checker's acceptance, and the
                // fix is to close it here rather than teach the JIT to accept
                // what the interpreter was wrong to allow. Scoped to the case
                // the checker can actually decide: `ids` typed as a tensor with
                // a known, non-integral element type. An `Unknown` element (an
                // opaque or already-errored expression) is left alone rather
                // than risk a cascade.
                if let Expr::Ident(name, _) = expr {
                    if name == "embed" {
                        if let Some(CallArg::Positional(ids_expr)) = args.get(1) {
                            if let Some(ids_full_ty) = self.embed_ids_ty(ids_expr, arg_tys.get(1)) {
                                let elem_is_integral = ids_full_ty.as_tensor_like()
                                    .is_some_and(|(e, _)| e.is_integral());
                                if !elem_is_integral {
                                    self.error(format!(
                                        "embed-index-type: `embed`'s `ids` argument must be an \
                                         integer tensor (STDLIB.md §3.6), got {}", ids_full_ty),
                                        span.clone());
                                }
                            }
                        }
                    }
                }

                // #745: a keyed map builtin's arg 0 must be a map. `TyType::Map`
                // is deliberately match-anything in `compatible_with` (#204), so
                // giving `map_set` et al. a `Map`-typed arg-0 slot in
                // `types.rs::builtin_sig` catches nothing on its own — a
                // dedicated check is needed, the same shape as the `embed`
                // check just above. Only fires when the callee is still the
                // builtin — a user-declared top-level fn of the same name
                // shadows it, the same "the thing the program actually
                // declared outranks a builtin spelling" rule the JIT's #720
                // review (F2) applies to `map`/`map_new`: `fn map_get(a: i64,
                // b: i64) -> i64 { a + b }` + `map_get(1, 2)` is a real,
                // `--check`-clean, `dmc run`-answering program (answers `3`),
                // and this check must not refuse it as a bad map arg. Gated on
                // `self.fn_mut_params` rather than `self.env.functions` —
                // `env.functions` is seeded with every builtin's own stub
                // signature by `install_builtins` (`Env::new`, before any
                // source is read), so `contains_key` there is true for
                // `map_set` etc. even with no shadowing fn in sight;
                // `fn_mut_params` is populated only by `register_fn`/
                // `register_extern_fn` from a real top-level `FnDecl`/
                // `ExternFnDecl` in this program, exactly the same signal the
                // uninit-fill check just above (`self.fn_mut_params.get(fname)`)
                // already uses to tell "a known user fn" from "everything
                // else". And only fires when the argument's static type is a
                // *known, concrete*
                // non-map. Three shapes are deliberately left alone, all real
                // and common in `examples/`:
                //   - `Unknown` (a `map_get` result, #742/#743's shape, or any
                //     other expression the checker can't pin down).
                //   - `Named { name: "map", .. }` — an `m: map`-annotated
                //     parameter (`demonic_lexer.dmc`'s `fn tok_kind(t: map) ->
                //     str { map_get(t, "k") }`), which `resolve_type`
                //     deliberately does NOT resolve to `TyType::Map` (see its
                //     own comment, and `for_in_map_param_does_not_warn`): `map`
                //     is match-anything there too, precisely so a real map
                //     passed through it isn't mistaken for a guaranteed one —
                //     that reasoning applies here exactly as it does to the
                //     #204 lint.
                //   - `Unit` (nil) — back on this list, but for a narrower
                //     reason than the one #753 removed it for. `compatible_with`'s
                //     new `(Unit, Map)` carve-out (types.rs ~146) means the
                //     general call-arg mechanism (`inferred_pty.compatible_with`,
                //     `PostfixOp::Call`'s `TyType::Fn` arm below) now refuses a
                //     `Unit`-typed arg 0 against `map_set`'s `Map`-typed param
                //     all on its own — so this dedicated check must NOT also
                //     flag it, or `map_set(nil, "k", 1)` reports the same span
                //     twice (`map_set: first arg must be map, got nil` here,
                //     `arg 0: expected map, got nil` from the general
                //     mechanism). The general mechanism's message is what a
                //     caller now sees for this case; this check still catches
                //     every OTHER concrete non-map (int, bool, tensor, ...)
                //     since none of those get a `compatible_with` carve-out.
                if let Expr::Ident(name, _) = expr {
                    if matches!(name.as_str(),
                        "map_set" | "map_get" | "map_has" | "map_contains" | "map_del")
                        && !self.fn_mut_params.contains_key(name) {
                        if let Some(map_ty) = arg_tys.first() {
                            let is_map_annotation =
                                matches!(map_ty, TyType::Named { name, .. } if name == "map");
                            if !matches!(map_ty, TyType::Map | TyType::Unknown | TyType::Unit)
                                && !is_map_annotation {
                                self.error(format!(
                                    "{name}: first arg must be map, got {map_ty}"),
                                    span.clone());
                            }
                        }
                        // #753 (sibling finding): gating this on the key's
                        // static TYPE (`Unit`) rather than the key EXPRESSION
                        // over-fires. A no-`-> T` fn's return type defaults to
                        // `Unit` regardless of its body's real tail
                        // (`f.ret_type.unwrap_or(Unit)`), so `fn key(i: i64) {
                        // "k" + to_str(i) }` — a working helper the checker
                        // silently accepts, because `compatible_with`'s
                        // `(Unit, Str)` arm (the same `(T, Err)`-convention
                        // exemption that lets an `Err`-typed fn return plain
                        // `nil`) masks its own declared-vs-body mismatch — typed
                        // the call `key(1)` as `Unit`. `map_set(m, key(1), 5)`
                        // is a real program with a genuine `str` key at
                        // runtime, and the old type-based check refused it,
                        // naming a type (`nil`) the value never actually has.
                        // Fixing that helper's own declaration is a separate,
                        // not-yet-filed rule (a str-tailed fn needing `-> str`,
                        // the sibling of #753's map-tailed-fn-needs-`-> map`
                        // fix) — it is not this call site's job to infer a
                        // callee's real return type from its body. Scoped
                        // instead to the one shape that is unambiguously a bad
                        // key: a literal `nil` at the call site itself
                        // (`Expr::Nil`), not merely a `Unit`-typed expression.
                        if let Some(CallArg::Positional(Expr::Nil(_))) = args.get(1) {
                            self.error(format!(
                                "{name}: key must be str, got nil"),
                                span.clone());
                        }
                    }
                }

                // #674 ("well-formedness stays check-time"): `forge.carve`'s
                // first argument must type as `Budget`, its second as an
                // integer, and a literal (not merely typed) non-positive
                // `bytes` is refused too — the one static slice of R1/R2 a
                // checker with no arena state can still decide (R1/R2 proper,
                // past the parent's free capacity or a live child, are
                // runtime refusals, `MEMORY.md §3.2`). `forge.reap`'s sole
                // argument must type as `Budget`. `forge.enter`'s own operand
                // and `in b`'s budget operand are already typed this way
                // (`Expr::BudgetBlock`'s and `Expr::AllocIn`'s own arms,
                // above) — this closes the same hole for the two calls
                // neither of those arms sees.
                if let Expr::Postfix { expr: inner, op: PostfixOp::Field(method), .. } = expr {
                    if let Expr::Ident(arena, _) = inner.as_ref() {
                        if arena == "forge" && method == "carve" {
                            // PR #814 review: `forge.carve()` and
                            // `forge.carve(root)` both "checked clean" before
                            // this — `arg_tys.first()`/`.get(1)` below are
                            // `Option`s that silently no-op on a missing
                            // argument, so a call with too FEW arguments hit
                            // no type check at all (only a present-but-wrong-
                            // typed argument was ever caught). Pin the arity
                            // itself first, independent of whether the
                            // arguments present are well-typed.
                            if args.len() != 2 {
                                self.error(format!(
                                    "forge.carve: expects 2 arguments (parent, bytes), found {}",
                                    args.len()),
                                    span.clone());
                            }
                            if let Some(parent_ty) = arg_tys.first() {
                                if !matches!(parent_ty, TyType::Budget | TyType::Unknown) {
                                    self.error(format!(
                                        "forge.carve: parent must be a `Budget`, found `{parent_ty}`"),
                                        span.clone());
                                }
                            }
                            if let Some(bytes_ty) = arg_tys.get(1) {
                                if !bytes_ty.is_integral() && !matches!(bytes_ty, TyType::Unknown) {
                                    self.error(format!(
                                        "forge.carve: bytes must be an integer, found `{bytes_ty}`"),
                                        span.clone());
                                } else if let Some(CallArg::Positional(bytes_expr)) = args.get(1) {
                                    if let Some(n) = expr_i64(bytes_expr) {
                                        if n <= 0 {
                                            self.error(format!(
                                                "forge.carve: bytes must be positive, found `{n}`"),
                                                span.clone());
                                        }
                                    }
                                }
                            }
                        } else if arena == "forge" && method == "reap" {
                            // Same gap as `forge.carve` above: `forge.reap()`
                            // checked clean with no argument to type at all.
                            if args.len() != 1 {
                                self.error(format!(
                                    "forge.reap: expects 1 argument (budget), found {}",
                                    args.len()),
                                    span.clone());
                            }
                            if let Some(budget_ty) = arg_tys.first() {
                                if !matches!(budget_ty, TyType::Budget | TyType::Unknown) {
                                    self.error(format!(
                                        "forge.reap: expects a `Budget`, found `{budget_ty}`"),
                                        span.clone());
                                }
                            }
                        }
                    }
                }

                // #474: `b.blit![2, 2](src)` puts a shape bracket between the
                // method name and the call, so the Field-shaped resolution
                // below never sees a method — and #441's check-time "no such
                // method" gate did not fire on this spelling either, leaving
                // exactly the hole #441 closed. Both are handled on the peeled
                // callee: an undefined name is that error, and a defined one
                // hands the machinery below the method's real signature
                // instead of the `Unknown` this used to type as. The receiver
                // is re-typed quietly so nothing is reported twice.
                //
                // Deliberately here, at the call, and not in the bracket's own
                // postfix arm: `let f = b.blit![2, 2]` is not a working value
                // — a method bound without being called evaluates to an opaque
                // at run time, for shape-bracketed and plain methods alike —
                // so handing the bracket a callable type on its own would be
                // the checker endorsing something that does not run.
                let mut recv = recv;
                if let Expr::Postfix { expr: bracketed, op: bop, .. } = expr {
                    if is_shape_bracket(bop) {
                        if let Expr::Postfix { expr: base, op: PostfixOp::Field(method), .. } =
                            bracketed.as_ref()
                        {
                            if let TyType::Named { name: model_name, .. } = self.quiet_ty(base) {
                                let unknown = self.env.models.get(&model_name).is_some_and(|info| {
                                    !info.methods.contains_key(method)
                                        && !info.fields.contains_key(method)
                                });
                                if unknown {
                                    self.error(
                                        format!("no method `{method}` on model `{model_name}`"),
                                        span.clone(),
                                    );
                                    return TyType::Unknown;
                                }
                            }
                        }
                        if let Some(t) = self.bracketed_method_ty(bracketed, bop, &span) {
                            recv = t;
                        } else if let Some(t) = self.fn_bracket_ty(bracketed, bop, &span) {
                            recv = t;
                        }
                    }
                }

                // #838 (option C): the callee's own `!`-marked param
                // positions, for a *direct* call only — populated below for
                // `fname(...)` and `receiver.method(...)`, never for a call
                // through a bound fn value (a separate, already-refused
                // shape, #780's fourth rule). `None` for a `@grad fn`
                // callee: there `!` marks a differentiable input, not a
                // mutation (AUTODIFF.md §1), and `.fwd`/`.fwd_bwd`/
                // `.fwd_bwd_bwd`/`.grad` never reach this far at all — see
                // the early returns in the method-call branch below.
                let mut call_mut_flags: Option<Vec<bool>> = None;
                let mut is_method = false;
                if let Expr::Postfix { expr: base, op: PostfixOp::Field(method), .. } = expr {
                    // It's a method call! Prepend the receiver type to the argument types.
                    let base_ty = self.check_expr(base);
                    if let TyType::Named { name: model_name, .. } = &base_ty {
                        call_mut_flags = self.method_mut_params
                            .get(&(model_name.clone(), method.clone()))
                            .cloned();
                    }
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
                            } else if matches!(base_ty,
                                TyType::Unit | TyType::Scalar(ScalarType::Nil)) {
                                // `nil` used to take the same forward-compat
                                // path as any other method-less receiver, so
                                // this warning told the author their call would
                                // "resolve to an opaque value at runtime". It
                                // no longer does: `nil` has no methods and both
                                // backends now say so at the call (SPEC §4.10).
                                // Describing the old behaviour would send the
                                // reader looking for a value that is never
                                // produced.
                                self.warn(
                                    format!("`.{method}()` on a `nil` receiver — `nil` has no methods; this raises at runtime rather than producing a value"),
                                    span.clone(),
                                    Some("guard the receiver before calling: `if e != nil { … }` (the `(_, Err)` convention, SPEC §3.9)".to_string()),
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
                } else if let Expr::Ident(fname, _) = expr {
                    // A direct call `fname(args)` of a known top-level fn.
                    // `grad_fn_counts` is populated only from a real
                    // top-level `FnDecl` (`register_fn`), same as
                    // `fn_mut_params` — a `@grad fn` callee is exempt, so a
                    // shadowing plain fn of the same name (impossible today,
                    // but the same guard the uninit-fill check above uses)
                    // is never mistakenly exempted.
                    if self.grad_fn_counts.get(fname).copied().unwrap_or(0) == 0 {
                        call_mut_flags = self.fn_mut_params.get(fname).cloned();
                    }
                }

                if let TyType::Fn { params, ret } = &recv {
                    if let Some(mut_flags) = &call_mut_flags {
                        self.check_mut_call_args(params, args, mut_flags, is_method);
                    }
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
                            if !inferred_pty.compatible_with(aty)
                                && !model_arg_binds_at_runtime(&inferred_pty, aty)
                            {
                                let report_idx = if is_method { i as isize - 1 } else { i as isize };
                                if report_idx >= 0 {
                                    self.error_mismatch(
                                        format!("arg {}: expected {}, got {}", report_idx, inferred_pty, aty),
                                        span.clone(),
                                        None,
                                        &inferred_pty,
                                        aty,
                                    );
                                } else {
                                    self.error_mismatch(
                                        format!("receiver: expected {}, got {}", inferred_pty, aty),
                                        span.clone(),
                                        None,
                                        &inferred_pty,
                                        aty,
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
                if !is_callee { self.reject_uncalled_method_bracket(expr, op, &span); }
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
                                    let expected_ty = expected_ty.clone();
                                    let val = fields.iter()
                                        .find(|(k, _)| k == *fname).map(|(_, v)| v);
                                    self.field_type_mismatch(
                                        fname, model_name, &expected_ty, fty, val, span.clone());
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
                    // #724: same rule as the bare-identifier arm in
                    // `check_expr`, for the qualified spelling (`use "..." as
                    // alias` then `alias.CAP`) — a top-level `let` never
                    // crosses a module boundary either way.
                    if let Some(module) = self.import_private_lets_qualified.get(&(alias.clone(), name.clone())) {
                        self.error(
                            format!(
                                "import-let: `{}` is a top-level `let` of `{}`; a constant \
                                 crosses a module boundary as a `pub fn` accessor, not a \
                                 bare reference (SPEC.md §6.7)",
                                name, module),
                            span.clone(),
                        );
                        return TyType::Unknown;
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

    /// Ruling on #838, option C (`SPEC.md §3.4.1` ¶Argument form):
    /// a `!` argument of **tensor or view** type must be a place whose root
    /// binding is mutable — a `let !`/`let mut` local, a `!` parameter, or a
    /// field path whose own root binding is one of those. Reuses
    /// `lhs_root_ident` and `is_mutable_binding`, the same two helpers the
    /// element-write rule (`x[i] = ...`, above) already applies to a plain
    /// `let` binding, so `bump(m.w)` on a plain `let m` is refused for the
    /// same reason `m.w[0] = 42.0` already is; `let !m = M { … } bump(m.w)`
    /// stays legal, since `m`'s own root is mutable. A model or `KV`
    /// argument is untouched here: the caller passes the same instance
    /// either way (SPEC §3.4.1), so this never runs for those positions —
    /// `mut_flags[i]` is checked against `params[i]`'s type, not merely
    /// against the `!` marker.
    ///
    /// `mut_flags` is the callee's own per-position `!` markers
    /// (`fn_mut_params` for a free fn, `method_mut_params` for a method) and
    /// is only ever populated for a **direct** call (`Expr::Ident` or
    /// `receiver.method`) of a known callee — never for a call through a
    /// bound fn value, which is a separate, already-refused shape (#780's
    /// fourth rule). Callers exempt a `@grad fn` callee before handing this
    /// a `Some(_)`; `.fwd`/`.fwd_bwd`/`.fwd_bwd_bwd`/`.grad` never reach
    /// this method at all — `check_postfix` returns their synthesized
    /// result type before the general call-argument loop runs
    /// (`AUTODIFF.md §1`).
    ///
    /// For a method call, position 0 of both `params` and `mut_flags` is
    /// the method's own `self` — its declared type is always `Unknown`
    /// (never `Tensor`/`View`), so it never trips the type gate below, and
    /// `args` (the caller's raw, receiver-less argument list) is indexed
    /// one lower via `offset`.
    fn check_mut_call_args(
        &mut self,
        params: &[TyType],
        args: &[CallArg],
        mut_flags: &[bool],
        is_method: bool,
    ) {
        let offset = if is_method { 1 } else { 0 };
        for (i, is_mut_param) in mut_flags.iter().enumerate() {
            if !*is_mut_param {
                continue;
            }
            if !matches!(params.get(i), Some(TyType::Tensor(..) | TyType::View(..))) {
                continue;
            }
            let Some(arg_idx) = i.checked_sub(offset) else { continue };
            let Some(CallArg::Positional(arg_expr)) = args.get(arg_idx) else { continue };
            if let Some(root) = lhs_root_ident(arg_expr) {
                if self.env.lookup(root).is_some() && !self.env.is_mutable_binding(root) {
                    self.error(
                        format!(
                            "passing `{root}` to a `!` parameter writes through it; \
                             `{root}` needs `let !{root}`"
                        ),
                        arg_expr.span_of(),
                    );
                }
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
                let elem = self.resolve_type(inner);
                self.check_tensor_element_scalar("Tensor", inner, &elem);
                TyType::Tensor(Box::new(elem), self.resolve_shape(sh))
            }
            Type::View(inner, sh, _) => {
                let elem = self.resolve_type(inner);
                self.check_tensor_element_scalar("View", inner, &elem);
                TyType::View(Box::new(elem), self.resolve_shape(sh))
            }
            Type::KV(inner, sh, _) => {
                let elem = self.resolve_type(inner);
                self.check_tensor_element_scalar("KV", inner, &elem);
                TyType::KV(Box::new(elem), self.resolve_shape(sh))
            }
            Type::Mesh(axes, _) => {
                let resolved: Vec<(String, SymDim)> = axes.iter()
                    .map(|a| (a.name.clone(), self.resolve_dim_at(SymDim::from_expr(&a.size), a.span.clone())))
                    .collect();
                TyType::Mesh(resolved)
            }
            Type::Fn(args, ret, _) => TyType::Fn {
                params: args.iter().map(|t| self.resolve_type(t)).collect(),
                ret: Box::new(self.resolve_type(ret)),
            },
            Type::Tuple(elems, _) => TyType::Tuple(elems.iter().map(|t| self.resolve_type(t)).collect()),
            Type::Array(inner, size, array_span) => {
                TyType::Array(Box::new(self.resolve_type(inner)), self.resolve_dim_at(SymDim::from_expr(size), array_span.clone()))
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
                // #674: `Budget` is a builtin nominal type, resolved BEFORE
                // the alias / enum / model lookups below — the name is reserved
                // (registration refuses `model`/`enum`/`type Budget`, `collect_alias_item`),
                // so nothing could shadow it here even without this early
                // return, but the ordering is the documented contract.
                if name == "Budget" {
                    if !args.is_empty() {
                        self.error(
                            "`Budget` takes no type arguments".to_string(),
                            span.clone(),
                        );
                        return TyType::Unknown;
                    }
                    return TyType::Budget;
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
                // #474: a model's arguments are shape arguments — `Box[H, W]`,
                // `Inner[4, 5]` — so resolve them as dims. Before this the
                // expression form (`Inner[4, 5]`) was dropped outright and the
                // identifier form (`Inner[H, W]`) resolved to a *type* named
                // `H`, which no substitution could ever replace with 4. Both
                // halves of the field-position unification hole.
                if self.is_model_name(name) {
                    let resolved_args: Vec<TyType> = args.iter()
                        .map(|a| match symdim_from_type_arg(a) {
                            Some(d) => TyType::Dim(self.resolve_dim_at(d, span.clone())),
                            None => TyType::Unknown,
                        })
                        .collect();
                    return TyType::Named { name: name.clone(), args: resolved_args };
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

    /// #785: `Tensor`/`View`/`KV`'s element type is restricted to a scalar
    /// type (SPEC.md §3.1's list, the rule itself stated at §3.2) —
    /// `RawPtr`'s pointee already enforced the sibling
    /// rule at the `extern fn` boundary (`extern_boundary_ty_ok`, the
    /// `extern-boundary` diagnostic), but these three `resolve_type` arms did
    /// not, so `Tensor[Buf, [4]]` / `View[Buf, [4]]` / `KV[Buf, [4]]` for a
    /// model `Buf` (also an enum, a tuple, another tensor, a raw pointer, …)
    /// passed `--check` silently: a model is not a scalar element type, and
    /// nothing downstream (the interpreter's `DType`, the JIT's `ScalarKind`)
    /// has a representation to build such a value in. Filed as #785.
    ///
    /// `kind` is the written keyword (`"Tensor"` / `"View"` / `"KV"`),
    /// `inner` is the *unwritten-through* AST element type — read only for
    /// its own span, since the resolved `TyType` carries none — and `elem`
    /// is what `inner` resolved to.
    ///
    /// `resolve_type` runs more than once over the same written annotation
    /// (a fn's params/return resolve once in `register_fn` and again in
    /// `check_fn`; an aliased annotation resolves through `resolve_type_
    /// alias` at every use site), so this dedupes on the element's own span
    /// via `reported_element_type_spans` rather than printing the same
    /// mistake two or more times.
    fn check_tensor_element_scalar(&mut self, kind: &str, inner: &Type, elem: &TyType) {
        // A `Named` naming something this checker never registered as a
        // model, enum or alias (an outright undefined type name) reaches
        // `resolve_type`'s `Type::Named` fallthrough arm the same shape as a
        // genuine model reference does — `is_model_name` is the only way to
        // tell them apart, and only this method (not the free `non_scalar_
        // tensor_element_reason`) has `self` to ask it. Read: "`Foo` is a
        // model" would be simply wrong for a `Foo` nobody declared.
        //
        // Three more `Named` shapes reach here that `is_model_name` alone
        // cannot tell from an undefined name either, and each is a real,
        // in-scope thing — the generic "names no model, enum or alias in
        // scope" reading would be false for all three, not merely imprecise:
        // `map` (a real §3.1 Map type — see the `TyType::Map` NOTE above,
        // it deliberately never resolves to that variant, so it reaches
        // this function as `Named { name: "map", .. }`), `Port` (SPEC.md
        // §3.11, refused here for a different reason than "undefined"),
        // and a function's own shape parameter (`env.shape_param_in_scope`
        // — a dim, not a type, but very much declared). Each gets its own
        // reason ahead of the generic fallthrough.
        let reason = match elem {
            TyType::Named { name, .. } if name == "map" => {
                "`map` is not a scalar type".to_string()
            }
            TyType::Named { name, .. } if name == "Port" => {
                "a `Port` handle is not a scalar type (SPEC.md §3.11)".to_string()
            }
            TyType::Named { name, .. } if self.env.shape_param_in_scope(name) => {
                format!("`{name}` is a shape parameter, not a scalar type")
            }
            TyType::Named { name, .. } if !self.is_model_name(name) => {
                format!("`{name}` names no model, enum or alias in scope — not a scalar type")
            }
            _ => match non_scalar_tensor_element_reason(elem) {
                Some(r) => r,
                None => return,
            },
        };
        let span = ast_type_span(inner);
        let key = (span.start, span.end, span.line, span.col);
        if self.reported_element_type_spans.insert(key) {
            self.error(
                format!(
                    "tensor-element-type: `{kind}`'s element type must be one of \
                     the scalar types (SPEC.md §3.2: i8/i16/i32/i64, u8/u16/u32/u64, \
                     int4/int8, f16/bf16/tf32/f32/f64, fp8_e4m3/fp8_e5m2, trit, \
                     bool, str, nil) — {reason}; nothing downstream can build a \
                     `{kind}` over it"
                ),
                span,
            );
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
                    shape_args.insert(param.name.clone(), self.resolve_dim_at(dim, param.span.clone()));
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

        let resolved = self.resolve_alias_body(alias);
        substitute_shape_args(resolved, &shape_args)
    }

    /// #761: resolve `alias`'s body exactly once, against this module's own
    /// `const_dims` and its own shape parameters, and cache the result in
    /// `resolved_aliases` — the fact that makes an importer's `resolve_type_
    /// alias` never need to re-derive a name it cannot see (a private `let`
    /// never crosses the module boundary, SPEC.md §6.7 / #724 / #731; the
    /// alias's already-resolved shape does, matching how the JIT already
    /// treats an alias as transparent metadata, #622). `pass1_collect_
    /// signatures` primes this cache for every alias up front
    /// (`cache_resolved_alias_item`); the memoization here also makes an
    /// alias chain (`type A = B  type B = Tensor[..]`) resolve each link
    /// once regardless of which one is primed first, and keeps a caller that
    /// reaches this without going through pass1 (some unit tests) correct.
    ///
    /// A name in the body that is neither one of the alias's own shape
    /// parameters nor a resolvable module constant is reported here, once,
    /// naming it: a type alias carries no "implicit shape parameter"
    /// convention for it to be one instead (unlike a `fn` signature's
    /// kvcache-style uppercase-name pattern, which this deliberately leaves
    /// alone — see `resolve_dim`).
    fn resolve_alias_body(&mut self, alias: &TypeAlias) -> TyType {
        if let Some(cached) = self.resolved_aliases.get(&alias.name) {
            return cached.clone();
        }
        // A cycle reachable only from *this* priming pass (not through a
        // fresh `resolve_type_alias` use, which reports "recursive type
        // alias" itself): resolve to Unknown rather than recursing forever.
        if self.resolving_aliases.iter().any(|n| n == &alias.name) {
            return TyType::Unknown;
        }
        self.resolving_aliases.push(alias.name.clone());
        self.env.push_shape_scope();
        for param in &alias.shape_params {
            self.env.bind_shape_param(&param.name, None);
        }
        let resolved = self.resolve_type(&alias.ty);
        self.env.pop_shape_scope();
        self.resolving_aliases.pop();
        self.resolved_aliases.insert(alias.name.clone(), resolved.clone());

        let own_params: std::collections::HashSet<&str> =
            alias.shape_params.iter().map(|p| p.name.as_str()).collect();
        let mut unresolved = std::collections::BTreeSet::new();
        collect_unresolved_dim_vars(&resolved, &own_params, &mut unresolved);
        for name in unresolved {
            // #766: a name that fell through to here is undefined as a
            // dimension, but "undefined" has three distinct causes, and only
            // one of them is "no such top-level `let` at all" — the other
            // two point the author at a real top-level binding that just
            // isn't usable as a constant, so the message says which.
            let msg = if self.mutable_top_level_lets.contains(&name) {
                format!(
                    "type alias `{}` names shape dimension `{}`, but `{}` is a \
                     mutable top-level `let` (`let !`/`let mut`) in this module \
                     — a mutable binding is never a constant dimension \
                     (SPEC.md §6.6)",
                    alias.name, name, name)
            } else if self.all_top_level_let_names.contains(&name) {
                format!(
                    "type alias `{}` names shape dimension `{}`, but `{}` is a \
                     top-level `let` in this module that is not a literal \
                     constant (nor a chain of top-level `let`s that bottoms \
                     out in one) (SPEC.md §6.6)",
                    alias.name, name, name)
            } else {
                format!(
                    "type alias `{}` names an undefined shape dimension `{}` — \
                     it is neither a shape parameter of `{}` nor a top-level \
                     `let` constant in this module (SPEC.md §6.6)",
                    alias.name, name, alias.name)
            };
            self.error(msg, alias.span.clone());
        }
        resolved
    }

    fn resolve_shape(&mut self, sh: &ShapeSpec) -> Shape {
        let dims = sh.elems.iter().map(|e| match e {
            ShapeElem::Wildcard(_) => SymDim::Wildcard,
            ShapeElem::Spread(_)   => SymDim::Wildcard,  // approximation
            ShapeElem::Streaming(_) => SymDim::Streaming,
            ShapeElem::Expr(e) => self.resolve_dim_at(SymDim::from_expr(e), sh.span.clone()),
        }).collect();
        Shape::new(dims)
    }

    /// #761: fold a dimension's free names against what can make one a
    /// dimension rather than a dangling variable. An in-scope shape
    /// parameter (its own fn/model/enum/alias, bound via `bind_shape_param`)
    /// is left as `Var` — call-site unification (`infer_call_shape_bindings`
    /// / `substitute_shape_args`) still substitutes it, unchanged. A
    /// top-level literal `let` constant is folded to `Const`, mirroring the
    /// JIT's `ConstDim` table (#620) — this is the fix: before it, `CAP` in
    /// `Tensor[i64,[CAP]]` stayed a `Var` forever, and a `Var` compared
    /// against a concrete dim is neither provably equal nor provably
    /// unequal (`SymDim::equivalent`'s `Unknown`), so a real mismatch never
    /// surfaced. Anything else is left as `Var` too, deliberately: an ad hoc
    /// generic dimension that is never declared in a `[...]` list (the
    /// kvcache pattern `check_fn`'s "implicit shape params" step
    /// recognizes) is legitimate, and this must not start erroring on it —
    /// that call is only safe to make for a type alias's own body, where no
    /// such convention exists (`resolve_alias_body`'s own check, after this
    /// runs).
    fn resolve_dim(&self, dim: SymDim) -> SymDim {
        match dim {
            SymDim::Var(name) => {
                if self.env.shape_param_in_scope(&name) {
                    SymDim::Var(name)
                } else if let Some(v) = self.env.int_const(&name) {
                    // #792 rework: a local `let` constant, e.g. `let k = 4`
                    // — see `Env::int_const`.
                    SymDim::Const(v)
                } else if let Some(&v) = self.const_dims.get(&name) {
                    SymDim::Const(v)
                } else {
                    SymDim::Var(name)
                }
            }
            SymDim::Add(l, r) => SymDim::Add(Box::new(self.resolve_dim(*l)), Box::new(self.resolve_dim(*r))),
            SymDim::Sub(l, r) => SymDim::Sub(Box::new(self.resolve_dim(*l)), Box::new(self.resolve_dim(*r))),
            SymDim::Mul(l, r) => SymDim::Mul(Box::new(self.resolve_dim(*l)), Box::new(self.resolve_dim(*r))),
            SymDim::Div(l, r) => SymDim::Div(Box::new(self.resolve_dim(*l)), Box::new(self.resolve_dim(*r))),
            SymDim::Mod(l, r) => SymDim::Mod(Box::new(self.resolve_dim(*l)), Box::new(self.resolve_dim(*r))),
            SymDim::Neg(x) => SymDim::Neg(Box::new(self.resolve_dim(*x))),
            other => other,
        }.simplify()
    }

    /// #799: find a `Var` inside `dim` that names a mutable top-level `let`
    /// and is not shadowed by anything that would make it legitimately
    /// resolvable — an in-scope shape parameter, a local `let` constant, or
    /// an immutable `const_dims` entry (checked in that order, mirroring
    /// `resolve_dim`'s own precedence). Recurses through the same
    /// arithmetic `dim` can carry; returns the first match found, which is
    /// enough to name the problem in a diagnostic.
    fn mutable_let_dim_name(&self, dim: &SymDim) -> Option<String> {
        match dim {
            SymDim::Var(name) => {
                if self.env.shape_param_in_scope(name)
                    || self.env.int_const(name).is_some()
                    || self.const_dims.contains_key(name)
                {
                    None
                } else if self.mutable_top_level_lets.contains(name) {
                    Some(name.clone())
                } else {
                    None
                }
            }
            SymDim::Add(l, r) | SymDim::Sub(l, r) | SymDim::Mul(l, r)
            | SymDim::Div(l, r) | SymDim::Mod(l, r) => {
                self.mutable_let_dim_name(l).or_else(|| self.mutable_let_dim_name(r))
            }
            SymDim::Neg(x) => self.mutable_let_dim_name(x),
            _ => None,
        }
    }

    /// #766: like `resolve_dim`, but for a dimension SLOT with a source
    /// location behind it (a shape bracket, a mesh axis, a model/alias
    /// shape arg) — refuses right here when the fully-resolved dimension is
    /// a non-positive constant, instead of letting it ride silently into a
    /// `Tensor`/`View`/`KV`/array/mesh shape and surface far from the real
    /// problem. Before this, `let CAP = -1` used as a dimension folded to
    /// `Const(-1)` (`expr_i64` admits negatives, same as the JIT's own
    /// `ConstDim` does — that part is correct) and the shape carrying it
    /// passed straight through to wherever it was first indexed, reporting
    /// something like `index 0 out of bounds for axis 0 of size -1` — true,
    /// but nowhere near the alias/annotation that actually named `CAP`.
    ///
    /// Deliberately NOT folded into `resolve_dim` itself: `resolve_dim`
    /// recurses through arithmetic (`Add`/`Sub`/...), and an intermediate
    /// sub-expression can be a negative constant on its way to a positive
    /// final one (nothing in this corpus does that today, but the algebra
    /// allows it) — checking only the fully-resolved dimension at each call
    /// site avoids flagging a value that was never the actual dimension.
    ///
    /// #799: also refuses here when a `Var` in `dim` names a *mutable*
    /// top-level `let` (`let !`/`let mut`) that is not shadowed by an
    /// in-scope shape parameter, a local `let` constant, or an immutable
    /// `const_dims` entry — `resolve_dim` correctly leaves such a name
    /// unresolved (only a literal, immutable top-level `let` is eligible to
    /// fold), but before this fix an unresolved `Var` here fell through
    /// silently to an ordinary implicit generic shape parameter: `let !N =
    /// 4  fn f(x: Tensor[f32,[N]])` type-checked as fully shape-generic
    /// with no diagnostic, one character away from the concrete, folded `N`
    /// an immutable `let N = 4` produces (#793/#766). Mirrors the type-alias
    /// path's own "a mutable binding is never a constant dimension" error
    /// (`resolve_alias_body`) at the annotation site instead of the alias
    /// declaration, so a fn or method annotation gets the same located
    /// diagnostic — suppressed while resolving an alias's OWN body
    /// (`resolving_aliases` non-empty), since that path already reports
    /// this, by name, with alias-specific wording.
    fn resolve_dim_at(&mut self, dim: SymDim, span: Span) -> SymDim {
        if self.resolving_aliases.is_empty() {
            if let Some(name) = self.mutable_let_dim_name(&dim) {
                let key = (name.clone(), span.start, span.end, span.line, span.col);
                if self.reported_mutable_dim_spans.insert(key) {
                    self.error(
                        format!(
                            "dimension `{}` names a mutable top-level `let` (`let \
                             !`/`let mut`) in this module — a mutable binding is \
                             never a constant dimension (SPEC.md §6.6)",
                            name),
                        span.clone(),
                    );
                }
            }
        }
        let resolved = self.resolve_dim(dim);
        if let SymDim::Const(v) = resolved {
            if v <= 0 {
                self.error(
                    format!(
                        "dimension `{}` is not a positive size — a `Tensor`, \
                         `View`, `KV`, `[T; N]`, or `Mesh` dimension must \
                         resolve to a positive integer",
                        v),
                    span,
                );
            }
        }
        resolved
    }

    // ── Diagnostics ───────────────────────────────────────────────────────

    fn error(&mut self, msg: impl Into<String>, span: Span) {
        self.errors.push(TypeError { msg: msg.into(), span, hint: None, shapes: None });
    }

    fn error_with_hint(&mut self, msg: impl Into<String>, span: Span, hint: Option<String>) {
        self.errors.push(TypeError { msg: msg.into(), span, hint, shapes: None });
    }

    /// An expected-vs-actual mismatch. Same diagnostic as `error_with_hint`,
    /// plus the structured shape pair (#485) when the two types are tensor-like
    /// and it is the *shapes* that disagree — an element-type mismatch over
    /// equal shapes carries none, so the payload never points away from the fix.
    fn error_mismatch(
        &mut self,
        msg: impl Into<String>,
        span: Span,
        hint: Option<String>,
        expected: &TyType,
        actual: &TyType,
    ) {
        let shapes = match (expected.as_tensor_like(), actual.as_tensor_like()) {
            (Some((_, e)), Some((_, a))) if !e.same(a) => Some((e.clone(), a.clone())),
            _ => None,
        };
        self.errors.push(TypeError { msg: msg.into(), span, hint, shapes });
    }

    /// #474: recover a bare model literal's shape args from the fields it was
    /// actually given. `Inner { px: <a 4x5 tensor>, n: 0 }` says `H = 4, W = 5`
    /// just as plainly as `Inner[4, 5] { … }` does — the declared field type
    /// names the parameters and the actual field type supplies the dims, which
    /// is the same unification a call site performs on its arguments.
    ///
    /// All or nothing: the args are returned only when *every* shape parameter
    /// is pinned. A literal that pins some and not others has not made a claim
    /// anyone can check, and half a claim must not read as a whole one. The
    /// empty result is what the field and argument checks treat as "this
    /// literal proves nothing", not as "this literal matches anything".
    fn infer_literal_shape_args(
        &self,
        info: &crate::types::ModelInfo,
        field_tys: &[(&String, TyType)],
    ) -> Vec<TyType> {
        if info.shape_params.is_empty() { return Vec::new(); }
        let mut bindings: HashMap<String, SymDim> = HashMap::new();
        for (fname, actual) in field_tys {
            let Some(declared) = info.fields.get(fname.as_str()) else { continue };
            infer_call_shape_bindings(declared, actual, &mut bindings);
        }
        let mut args = Vec::with_capacity(info.shape_params.len());
        for p in &info.shape_params {
            match bindings.get(p) {
                // A dim that resolved to nothing useful is not a binding.
                Some(SymDim::Unknown) | None => return Vec::new(),
                Some(d) => args.push(TyType::Dim(d.clone())),
            }
        }
        args
    }

    /// #476: a model-array field initialized with a bracket literal. The bare
    /// mismatch ("expected `[Cell; 3]`, got `Tensor[?, [3]]`") sends the author
    /// off to fix the literal's element type, which is not the problem — a
    /// bracket literal builds a tensor, and no bracket literal of any element
    /// type builds a model array. Point at the idiom that does.
    fn field_type_mismatch(
        &mut self,
        fname: &str, model: &str,
        expected: &TyType, got: &TyType,
        value: Option<&Expr>, span: Span,
    ) {
        let hint = match (expected, got) {
            (TyType::Array(elem, n), TyType::Tensor(..))
                if matches!(value, Some(Expr::TensorLit(..))) =>
            {
                Some(format!(
                    "a bracket literal builds a tensor, not a model array. Build \
                     `[{elem}; {n}]` with `vault.uninit[{elem}, [{n}]]`, fill every \
                     slot with a whole model literal, then store it: \
                     `let !cs = vault.uninit[{elem}, [{n}]]  \
                     vault {{ for i in 0..{n} {{ cs[i] = {elem} {{ .. }} }} }}  \
                     {model} {{ {fname}: cs }}` (MEMORY §2)"))
            }
            _ => None,
        };
        self.error_mismatch(
            format!("mismatched type for field `{}` of model `{}`: expected `{}`, got `{}`",
                    fname, model, expected, got),
            span, hint, expected, got,
        );
    }

    /// Emit a non-fatal lint (safe-mode diagnostic). Surfaced by the CLI but does
    /// not fail `--check`. Demon mode releases the restriction: when `self.demon`
    /// is set the lint is dropped on the floor — no collection, no print (#196).
    fn warn(&mut self, msg: impl Into<String>, span: Span, hint: Option<String>) {
        if self.demon { return; }
        self.warnings.push(TypeError { msg: msg.into(), span, hint, shapes: None });
    }
}

/// #350: the variant of `variants` that `name` most plausibly misspells, if
/// any. Case-insensitive edit distance, with a budget of one edit per three
/// characters (at least one) — tight enough that an ordinary binding name
/// (`other`, `rest`, `x`) suggests nothing, loose enough to catch `Gren` for
/// `Green` and the all-lowercase `green`. Ties break toward the earlier
/// (declaration-order) variant, so the suggestion is stable.
fn closest_variant<'a>(name: &str, variants: &'a [String]) -> Option<&'a String> {
    let lower = name.to_lowercase();
    let mut best: Option<(usize, &String)> = None;
    for v in variants {
        let d = edit_distance(&lower, &v.to_lowercase());
        let better = match best { None => true, Some((bd, _)) => d < bd };
        if better { best = Some((d, v)); }
    }
    let (d, v) = best?;
    if d <= (name.chars().count() / 3).max(1) { Some(v) } else { None }
}

/// Levenshtein distance over `char`s, two rolling rows.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur: Vec<usize> = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(sub);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
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

/// #476: the `binding.field` an assignment LHS writes through, peeling any
/// trailing index chain — `h.cells[i] = ..` and `h.cells = ..` both yield
/// `("h", "cells")`. Returns `None` for anything deeper or not rooted in a
/// simple identifier; the #476 check reaches exactly one level.
fn lhs_field_path(e: &Expr) -> Option<(String, String)> {
    match e {
        Expr::Postfix { expr, op: PostfixOp::Index(_), .. } => lhs_field_path(expr),
        Expr::Postfix { expr, op: PostfixOp::Field(f), .. } => match expr.as_ref() {
            Expr::Ident(root, _) => Some((root.clone(), f.clone())),
            _ => None,
        },
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

/// #728 rework (reviewer defect 3): is `expr` the *receiver* of a
/// `(forge|vault).<zeros|ones|uninit>[…]` allocator call — the model-array
/// allocator set, matching `MODEL_ARRAY_ALLOCATOR_METHODS`/`ctor_tensor_ty`
/// below, not just `uninit` (the original, narrower form of this function
/// keyed on `uninit` alone, letting `forge.zeros[Surface, [4]]` and
/// `forge.ones[Surface, [4]]` — both legal model-array allocators — escape
/// the type-arg-less generic-model lint). Same shape `is_uninit_ctor`
/// matches, minus the index wrapper (the one caller here, `check_expr`'s
/// `PostfixOp::Index` arm, already knows it's inside one; `expr` there is
/// that wrapper's own receiver). Returns the arena and the method name,
/// since the allocator diagnostic names both (`the element type of
/// `forge.zeros``, not always `` `forge.uninit` ``).
///
/// #674 slice 3a: re-gated on `ast::arena_ctor_receiver` — the same
/// receiver-recognition `is_arena_ctor_spine` uses — rather than a
/// hand-rolled `Ident` match, so this can never answer a different arena
/// name than the spine's own definition does. `stream` is on the spine but
/// never a model-array arena, hence the explicit narrowing below.
fn model_array_alloc_arena_of(expr: &Expr) -> Option<(&'static str, &str)> {
    let (arena, method) = arena_ctor_receiver(expr)?;
    if !MODEL_ARRAY_ALLOCATOR_METHODS.contains(&method) { return None; }
    match arena {
        "forge" => Some(("forge", method)),
        "vault" => Some(("vault", method)),
        _ => None,
    }
}

/// #728 rework (reviewer defect 1; multi-match fix): render `ty` back to
/// demoniC surface type syntax, replacing EVERY occurrence named in
/// `replacements` (a list of `(span.start, replacement text)` pairs) with
/// its own replacement text verbatim — used only to build the
/// type-arg-less generic-model hint, so it shows the annotation's own
/// array/tuple/fn-type structure with the bare model(s) bracketed
/// (`[Surface[H, W]; 4]`, `(Buf[N], i64)`, `fn(Buf[N]) -> i64`) instead of
/// naming a model alone and silently dropping whatever it was nested
/// inside (the wrapper `find_bare_generic_models`, defect 2, recurses
/// through).
///
/// An annotation with more than one bare-model occurrence
/// (`y: ([Buf; 2], (Buf, i64))`) gets one warning per occurrence, each
/// built by a separate `type_arg_less_hint` call — but `replacements`
/// there always covers ALL of them, not just the one this particular
/// warning is about: bracketing only the warning's own occurrence would
/// leave every hint incomplete (applying any single one still leaves a
/// sibling bare, which still warns), so `type_arg_less_hint` includes every
/// sibling match's own concrete bracket alongside the primary replacement.
/// One `replacements` entry per match found in `ty`, keyed by span start
/// (each occurrence's span is distinct, so this can't collide).
///
/// Reconstructs from the AST rather than slicing the original source text:
/// the checker never retains source (only the parsed `Program`), so this is
/// the only route to a rendered spelling at all. Every dim/size/shape-axis
/// expr renders through `SymDim`'s own `Display` (already the compiler's
/// one canonical "print an arithmetic dim back out" path — see its uses
/// elsewhere in this file), not a hand-rolled expression printer.
fn render_type_for_hint(ty: &Type, replacements: &[(usize, String)]) -> String {
    if let Type::Named { span, .. } = ty {
        if let Some((_, replacement)) = replacements.iter().find(|(start, _)| *start == span.start) {
            return replacement.clone();
        }
    }
    match ty {
        Type::Scalar(s, _) => crate::fmt::scalar_type_str(s).to_string(),
        Type::Tensor(e, sh, _) => format!(
            "Tensor[{}, {}]", render_type_for_hint(e, replacements), render_shape_for_hint(sh),
        ),
        Type::View(e, sh, _) => format!(
            "View[{}, {}]", render_type_for_hint(e, replacements), render_shape_for_hint(sh),
        ),
        Type::KV(e, sh, _) => format!(
            "KV[{}, {}]", render_type_for_hint(e, replacements), render_shape_for_hint(sh),
        ),
        Type::Mesh(axes, _) => {
            let parts: Vec<String> = axes.iter()
                .map(|a| format!("{}: {}", a.name, SymDim::from_expr(&a.size)))
                .collect();
            format!("Mesh[{}]", parts.join(", "))
        }
        Type::Fn(params, ret, _) => {
            let ps: Vec<String> = params.iter()
                .map(|p| render_type_for_hint(p, replacements))
                .collect();
            format!("fn({}) -> {}", ps.join(", "), render_type_for_hint(ret, replacements))
        }
        Type::Tuple(elems, _) => {
            let parts: Vec<String> = elems.iter()
                .map(|e| render_type_for_hint(e, replacements))
                .collect();
            format!("({})", parts.join(", "))
        }
        Type::Array(e, size, _) => format!(
            "[{}; {}]", render_type_for_hint(e, replacements), SymDim::from_expr(size),
        ),
        Type::RawPtr(e, _) => format!("*{}", render_type_for_hint(e, replacements)),
        Type::Named { name, args, .. } => {
            if args.is_empty() {
                name.clone()
            } else {
                let parts: Vec<String> = args.iter()
                    .map(|a| render_type_arg_for_hint(a, replacements))
                    .collect();
                format!("{name}[{}]", parts.join(", "))
            }
        }
    }
}

fn render_type_arg_for_hint(arg: &TypeArg, replacements: &[(usize, String)]) -> String {
    match arg {
        TypeArg::Type(t) => render_type_for_hint(t, replacements),
        TypeArg::Expr(e) => SymDim::from_expr(e).to_string(),
        TypeArg::Named { name, value, .. } => format!("{name}={}", SymDim::from_expr(value)),
    }
}

fn render_shape_for_hint(ss: &ShapeSpec) -> String {
    let parts: Vec<String> = ss.elems.iter().map(|e| match e {
        ShapeElem::Wildcard(_) => "?".to_string(),
        ShapeElem::Spread(_) => "..".to_string(),
        ShapeElem::Streaming(_) => "~".to_string(),
        ShapeElem::Expr(e) => SymDim::from_expr(e).to_string(),
    }).collect();
    format!("[{}]", parts.join(", "))
}

/// #533: is this expression a `forge.trit[K, N]` / `vault.trit[K, N]`
/// constructor? Unlike `forge.zeros[f32, […]]` it carries no element-type
/// argument — the dims are bare integers — so `ctor_tensor_ty` cannot read a
/// `trit` element off it and the binding types as `Unknown`. The port
/// primitives need the element type, so the constructor is recognised by shape.
fn is_trit_ctor(expr: &Expr) -> bool {
    let base = match expr {
        Expr::Postfix { expr: base, op: PostfixOp::Index(_), .. } => base.as_ref(),
        _ => return false,
    };
    matches!(base, Expr::Postfix { expr: inner, op: PostfixOp::Field(m), .. }
        if m == "trit" && matches!(inner.as_ref(), Expr::Ident(a, _) if a == "forge" || a == "vault"))
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

/// #562: a single index-bracket element, classified for shape-typing.
/// `t[a..b]` (an `IndexElem::Expr(Expr::Range)`) and `t[a:b]` (an
/// `IndexElem::Slice`) are two parses of the same axis-slice concept —
/// jit.rs's `classify_index`/`classify_index_static` and interp.rs's
/// `eval_index_elems` already unify them (see the "#276" / "not
/// `IndexElem::Slice`" comments there). The checker didn't: `PostfixOp::Index`
/// used to test `matches!(e, IndexElem::Expr(_))` to decide "is this whole
/// bracket a plain scalar index", which is also true of `IndexElem::Expr(Range)`
/// — so `x[0..S]` read as a single scalar index and collapsed the result to
/// the tensor's element type instead of a sliced tensor. This is the one
/// place that decides "scalar vs. slice" for the checker; every caller in
/// this file goes through it so the two spellings can't drift apart again.
enum IndexAxis<'a> {
    /// A single scalar index — drops the axis.
    Scalar(&'a Expr),
    /// `..` alone (full axis) — keeps the axis at its current extent.
    Full,
    /// `a..b` / `a..=b` / `a:b` / `a:b:c` — keeps the axis at a derived extent.
    Slice { start: Option<&'a Expr>, end: Option<&'a Expr>, step: Option<&'a Expr>, inclusive: bool },
}

fn classify_index_axis(e: &IndexElem) -> IndexAxis<'_> {
    match e {
        IndexElem::Expr(Expr::Range { start, end, inclusive, .. }) => IndexAxis::Slice {
            start: start.as_deref(), end: end.as_deref(), step: None, inclusive: *inclusive,
        },
        IndexElem::Expr(e) => IndexAxis::Scalar(e),
        IndexElem::FullSlice(_) => IndexAxis::Full,
        IndexElem::Slice { start, end, step, .. } => IndexAxis::Slice {
            start: start.as_deref(), end: end.as_deref(), step: step.as_deref(), inclusive: false,
        },
    }
}

/// True iff `d` simplifies to a negative constant. A negative slice bound
/// is Python-style "from the end" (SPEC), not a plain offset from zero —
/// `derive_slice_shape` can't model that arithmetic, so it bails to
/// `Unknown` rather than compute a wrong extent (e.g. `dim - (-2)` instead
/// of the real extent `2`).
fn is_negative_literal(d: &SymDim) -> bool {
    matches!(d, SymDim::Const(n) if *n < 0)
}

/// #562: derive the sliced output shape of an index bracket that contains
/// at least one slice/full-slice element (mixed with scalar elems is fine —
/// scalar axes are dropped, same as plain scalar indexing). Returns `None`
/// when any slice axis' extent isn't expressible in the `SymDim` algebra:
/// a stepped slice (`a:b:c`, extent needs a ceiling division this doesn't
/// model), a negative-literal bound, or a bound this can't reduce past
/// `SymDim::Unknown`. The caller falls back to `TyType::Unknown` in that
/// case — a wrong shape is worse than the current "can't say" answer.
fn derive_slice_shape(elems: &[IndexElem], shape: &Shape) -> Option<Vec<SymDim>> {
    let mut out = Vec::with_capacity(shape.rank());
    for (i, elem) in elems.iter().enumerate() {
        if i >= shape.rank() { return None; }
        let axis_dim = shape.dims[i].clone();
        match classify_index_axis(elem) {
            IndexAxis::Scalar(_) => {} // dropped
            IndexAxis::Full => out.push(axis_dim),
            IndexAxis::Slice { start, end, step, inclusive } => {
                if step.is_some() { return None; }
                let start_dim = match start {
                    Some(e) => SymDim::from_expr(e).simplify(),
                    None => SymDim::Const(0),
                };
                if is_negative_literal(&start_dim) { return None; }
                let end_dim = match end {
                    Some(e) => SymDim::from_expr(e).simplify(),
                    None => axis_dim,
                };
                if is_negative_literal(&end_dim) { return None; }
                let mut extent = SymDim::Sub(Box::new(end_dim), Box::new(start_dim)).simplify();
                if inclusive {
                    extent = SymDim::Add(Box::new(extent), Box::new(SymDim::Const(1))).simplify();
                }
                if matches!(extent, SymDim::Unknown) { return None; }
                out.push(extent);
            }
        }
    }
    Some(out)
}

/// #728 rework (reviewer defect 3): `MODEL_ARRAY_ALLOCATOR_METHODS` — the
/// builtin allocator method names that can build a *model* array
/// (`arena.<method>[Model, [dims]]`), the same set `ctor_tensor_ty` matches
/// for a *scalar*-element allocator — moved to `ast.rs` beside
/// `is_arena_ctor_spine` in #674 slice 3a (see that const's doc comment for
/// why it is a DIFFERENT, narrower set than the spine's own).
fn ctor_tensor_ty(expr: &Expr) -> Option<TyType> {
    let (base, idxs) = match expr {
        Expr::Postfix { expr: base, op: PostfixOp::Index(idxs), .. } => (base.as_ref(), idxs),
        _ => return None,
    };
    // #674 slice 3a: re-gated on `ast::arena_ctor_receiver` (see
    // `model_array_alloc_arena_of`'s twin comment) rather than a
    // hand-rolled `Ident` match, so this can never answer a different arena
    // name than the spine's own definition does.
    let (arena, method) = arena_ctor_receiver(base)?;
    if !matches!(arena, "forge" | "vault") { return None; }
    if !MODEL_ARRAY_ALLOCATOR_METHODS.contains(&method) { return None; }
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
///
/// Delegates to `ScalarType::from_ctor_name` (#834) — the same table
/// `interp.rs`'s `arena_ctor_dtype` now resolves a bracket name through, so
/// this checker and the interpreter can never again disagree about which
/// names this position admits.
fn scalar_type_from_ident(e: &Expr) -> Option<ScalarType> {
    let name = match e { Expr::Ident(n, _) => n.as_str(), _ => return None };
    ScalarType::from_ctor_name(name)
}

/// #785: `ty`'s own span — read directly off whichever `ast::Type` variant
/// it is, since a resolved `TyType` carries none of its own. Mirrors
/// `jit.rs`'s private `type_span` (kept separate: that one is jit.rs's own
/// and not exported).
fn ast_type_span(ty: &Type) -> Span {
    match ty {
        Type::Scalar(_, s)
        | Type::Tensor(_, _, s)
        | Type::View(_, _, s)
        | Type::KV(_, _, s)
        | Type::Mesh(_, s)
        | Type::Fn(_, _, s)
        | Type::Tuple(_, s)
        | Type::Array(_, _, s)
        | Type::RawPtr(_, s)
        | Type::Named { span: s, .. } => s.clone(),
    }
}

/// #785: is `ty` legal as a `Tensor`/`View`/`KV` element type — one of
/// SPEC.md §3.1's scalar types? `Unknown` (from `any`, or an already-errored
/// expression) is deliberately admitted, not flagged: the same "don't
/// cascade onto an already-⊥ type" rationale as every other check in this
/// file that special-cases `Unknown` (e.g. the `embed`-ids-must-be-integer
/// check, above) — it exists to suppress a second error, not manufacture
/// one. Returns `None` when `ty` is legal, `Some(reason)` naming what it
/// actually is otherwise, for the diagnostic.
fn non_scalar_tensor_element_reason(ty: &TyType) -> Option<String> {
    match ty {
        TyType::Scalar(_) | TyType::Unknown => None,
        TyType::Enum(name) => Some(format!("`{name}` is an enum, not a scalar type")),
        TyType::Named { name, .. } => Some(format!("`{name}` is a model, not a scalar type")),
        TyType::Tensor(..) => Some("a `Tensor` is not a scalar type".to_string()),
        TyType::View(..) => Some("a `View` is not a scalar type".to_string()),
        TyType::KV(..) => Some("a `KV` is not a scalar type".to_string()),
        TyType::Mesh(..) => Some("a `Mesh` is not a scalar type".to_string()),
        TyType::Fn { .. } => Some("a function type is not a scalar type".to_string()),
        TyType::Tuple(..) => Some("a tuple type is not a scalar type".to_string()),
        TyType::Array(..) => Some("a fixed-size array type is not a scalar type".to_string()),
        TyType::RawPtr(..) => Some("a raw pointer type is not a scalar type".to_string()),
        // No explicit `TyType::Map` arm: `check_tensor_element_scalar`'s own
        // `Named` match (its "map"/"Port"/shape-param special cases) intercepts
        // a `map` annotation before this function ever sees it — `resolve_type`
        // never produces `TyType::Map` for a written annotation (see the NOTE
        // at its `Type::Named` arm), so this function's caller never passes
        // one. The generic `other` arm below renders identically for it
        // anyway (`TyType::Map`'s `Display` is `"map"`), so there is nothing
        // an explicit arm would add.
        other => Some(format!("`{other}` is not a scalar type")),
    }
}

// ─── `as` cast legality (#550) ───────────────────────────────────────────────

/// Render a type the way the JIT's diagnostics do: scalars in their *source*
/// spelling (`i64`, not the `I64` that `TyType`'s `Display` derives from the
/// `ScalarType` variant name). The cast refusal below has to read exactly like
/// `dmc jit`'s, so it cannot go through `Display`.
fn render_ty(t: &TyType) -> String {
    match t {
        TyType::Scalar(s) => crate::fmt::scalar_type_str(s).to_string(),
        TyType::Tensor(e, sh) => format!("Tensor[{}, {}]", render_ty(e), sh),
        TyType::View(e, sh)   => format!("View[{}, {}]", render_ty(e), sh),
        TyType::KV(e, sh)     => format!("KV[{}, {}]", render_ty(e), sh),
        TyType::Tuple(ts) => format!(
            "({})",
            ts.iter().map(render_ty).collect::<Vec<_>>().join(", "),
        ),
        TyType::Array(e, n) => format!("[{}; {}]", render_ty(e), n),
        TyType::RawPtr(e) => format!("*{}", render_ty(e)),
        TyType::Unit => "nil".to_string(),
        other => other.to_string(),
    }
}

/// #598: why `@cast(Model) { bytes }` cannot lay this field out, or `None`
/// when it can. The spec's only wire-layout guarantee for a model is field
/// order (SPEC.md §3.10), so a field is admitted exactly when its own type
/// fixes a byte extent: a scalar of known width
/// (`ScalarType::overlay_byte_width`), or a tensor of such scalars at a fully
/// static shape — element count × element width, row-major. Everything else
/// (nested models, `View`/`KV` descriptors, model arrays, function types) has
/// no byte representation to read. See `check_cast_model_overlay`.
///
/// `Unknown` is admitted: it is the checker's conservative ⊥ and exists to
/// suppress cascade errors, not to generate one.
fn cast_overlay_field_refusal(ty: &TyType) -> Option<String> {
    match ty {
        TyType::Unknown => None,
        TyType::Scalar(s) => s.overlay_refusal_reason().map(str::to_string),
        TyType::Tensor(elem, shape) => {
            let TyType::Scalar(es) = &**elem else {
                return Some(format!(
                    "a tensor field overlays only with a scalar element type, not `{}`",
                    render_ty(elem)));
            };
            if let Some(reason) = es.overlay_refusal_reason() {
                return Some(format!("its element type has no byte form — {}", reason));
            }
            if let Some(d) = shape.dims.iter().find(
                |d| !matches!(d.simplify(), SymDim::Const(n) if n >= 0))
            {
                return Some(format!(
                    "a tensor field overlays only at a static shape, and `{}` is not a \
                     constant dimension", d));
            }
            None
        }
        TyType::View(..) | TyType::KV(..) => Some(
            "a `View`/`KV` field is a descriptor over another arena, not bytes of its own"
                .to_string()),
        TyType::Named { name, .. } => Some(format!(
            "a nested model field (`{}`) has no overlay layout of its own", name)),
        TyType::Array(inner, n) => Some(format!(
            "a model-array field (`[{}; {}]`) has no overlay layout", render_ty(inner), n)),
        other => Some(format!("`{}` has no byte layout", render_ty(other))),
    }
}

/// The scalar kinds an *elementwise tensor cast* is defined for — exactly the
/// targets `interp::apply_cast`'s tensor arm maps every element through. A
/// `bool`, `str`, `nil` or `trit` target falls through that arm and hands the
/// tensor back untouched, which is not a conversion.
fn scalar_is_elementwise_cast_target(s: &ScalarType) -> bool {
    use ScalarType::*;
    matches!(s,
        I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | Int4 | Int8 |
        F16 | Bf16 | Tf32 | F32 | F64 | Fp8E4M3 | Fp8E5M2)
}

/// A scalar type with a numeric (or `bool`) value that `as` can convert.
/// `str`, `nil` and the never-a-value kinds are excluded: they have no numeric
/// reading, which is why `str as i64` is the #550 hole and not a conversion.
fn ty_is_convertible_scalar(t: &TyType) -> bool {
    match t {
        TyType::IntLit(_) | TyType::FloatLit(_) => true,
        TyType::Scalar(s) => matches!(s, ScalarType::Bool) || {
            use ScalarType::*;
            matches!(s,
                I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | Int4 | Int8 |
                F16 | Bf16 | Tf32 | F32 | F64 | Fp8E4M3 | Fp8E5M2 | Trit)
        },
        _ => false,
    }
}

/// Types the checker deliberately does not model well enough to rule on a
/// cast. `Unknown` is ⊥ (and what `any` resolves to), `Dim` is a shape
/// parameter in value position (`D as f32`), `Named` covers models, `Port[L]`
/// and unresolved aliases, and the rest are opaque handles that reach `as`
/// only at boundaries the spec leaves to the backends. Casting one is not
/// endorsed — it is simply not *refused* on evidence the checker doesn't have.
fn cast_ty_is_opaque(t: &TyType) -> bool {
    matches!(t,
        TyType::Unknown | TyType::Dim(_) | TyType::Named { .. } | TyType::Module { .. } |
        TyType::Map | TyType::Mesh(_) | TyType::RawPtr(_) | TyType::Fn { .. } |
        TyType::Array(..))
}

/// #550: the conversions `as` is licensed to perform.
///
/// SPEC §3.1 makes `as` the only way one concrete scalar becomes another —
/// "All implicit numeric conversions are forbidden. Casts are explicit: `x as
/// f32`" — and does not license it between arbitrary types. The complete set:
///
/// 1. **scalar ↔ scalar** inside the numeric + `bool` families (SPEC §3.1:
///    "`i64 → i32`, `f64 → f32`, `f64 → i8` all require an `as` cast"),
///    including the narrowing two's-complement wrap of #540 (`5000000000 as
///    i32` is `705032704`) and OPERATORS.md §7's float→int truncate/saturate.
/// 2. **`x as str`** — SPEC §3.1, "Integer-to-string conversion: `n as str`";
///    both backends render `bool` and float sources through the same path.
/// 3. **the elementwise tensor cast** — `Tensor[E, S] as N` for a numeric `N`,
///    which retags or truncates every element (DIRECTIVES.md §4.1 documents
///    the `@cast(i64) { t }` block analog). The JIT declines to lower it as
///    *unsupported*, not illegal, and points at `dmc run`.
/// 4. **enum ↔ ordinal** — SPEC §3.1's enum paragraph: "An enum value is its
///    variant's `i64` ordinal … `Token.Eq as i64 == 1`", and the explicit
///    `n as Light` the JIT lowers back the other way.
/// 5. the **identity** cast, and any cast touching a type the checker does not
///    model (see `cast_ty_is_opaque`).
///
/// Everything else is a type error: `str as i64`, `nil as i64`, a tuple cast
/// to a scalar, a tensor cast to another tensor type. Each of those is already
/// refused by `dmc jit` and silently hands the *unconverted* value back under
/// `dmc run`, which is the soundness hole #550 was filed over.
fn cast_is_legal(from: &TyType, to: &TyType) -> bool {
    if from == to { return true; }
    if cast_ty_is_opaque(from) || cast_ty_is_opaque(to) { return true; }
    match to {
        TyType::Scalar(t) => match t {
            // (2) `x as str`.
            ScalarType::Str => matches!(from,
                TyType::Scalar(_) | TyType::IntLit(_) | TyType::FloatLit(_) |
                TyType::Enum(_) | TyType::Unit),
            // `x as nil` discards a value that is already `nil`; nothing else
            // has a `nil` reading.
            ScalarType::Nil => matches!(from, TyType::Unit),
            // (1) + (3) + (4).
            _ => ty_is_convertible_scalar(from)
                || matches!(from, TyType::Enum(_))
                || (from.as_tensor_like().is_some() && scalar_is_elementwise_cast_target(t)),
        },
        // (4), the reverse direction: an integer ordinal back to its enum.
        TyType::Enum(_) => from.is_integral(),
        _ => false,
    }
}

/// The type an accepted cast produces.
///
/// Almost always the target type — but an **elementwise tensor cast** yields a
/// *tensor* of the target element type, not a bare scalar. Typing `t as i64`
/// as `i64` was the other half of the #550 hole: `dmc run` maps every element
/// and hands back a `Tensor`, so a `-> i64` function could return one with the
/// checker's blessing.
fn cast_result_ty(from: &TyType, to: &TyType) -> TyType {
    if let TyType::Scalar(t) = to {
        if scalar_is_elementwise_cast_target(t) {
            match from {
                TyType::Tensor(_, sh) => return TyType::Tensor(Box::new(to.clone()), sh.clone()),
                TyType::View(_, sh)   => return TyType::View(Box::new(to.clone()), sh.clone()),
                TyType::KV(_, sh)     => return TyType::KV(Box::new(to.clone()), sh.clone()),
                _ => {}
            }
        }
    }
    to.clone()
}

/// #578: is `t` a type an `extern fn` boundary admits? The spec restricts a
/// parameter or return type there to scalar types, raw pointer types `*T`, and
/// `nil`. A raw pointer's pointee is itself restricted to a scalar type or
/// `nil` — `*Tensor[f32, [4]]` is not a pointer form the language has.
///
/// The literal types are admitted because a defaulted literal concretizes to a
/// scalar, and `Unknown` is ⊥: an annotation the checker could not resolve has
/// already produced its own diagnostic, and a second one here would be a
/// cascade rather than information.
fn extern_boundary_ty_ok(t: &TyType) -> bool {
    match t {
        TyType::Scalar(_) | TyType::IntLit(_) | TyType::FloatLit(_)
        | TyType::Unit | TyType::Unknown => true,
        TyType::RawPtr(inner) => matches!(**inner,
            TyType::Scalar(_) | TyType::Unit | TyType::Unknown),
        _ => false,
    }
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

/// Every directive reachable from `body` by descending through blocks that
/// consist of a lone directive construct and nothing else — the `@fuse` and
/// `@cast(bf16)` in `{ @fuse { @cast(bf16) { … } } }`. Braces around a lone
/// directive block are the stacked form written long-hand, so DIRECTIVES.md §3
/// reads the stack across them; the descent continues through intervening
/// levels so `@cast(f32) { @fuse { @cast(bf16) { … } } }` is the same stack as
/// `@cast(f32) @fuse @cast(bf16) { … }` and is rejected the same way. A block
/// holding anything besides that lone construct is a real block, not a
/// long-hand stack, and ends the descent.
fn nested_directive_stack(body: &Block) -> Vec<&Directive> {
    let mut out = Vec::new();
    let mut cur = body;
    loop {
        let (directives, next) = match (cur.stmts.as_slice(), &cur.tail_expr) {
            ([], Some(tail)) => match tail.as_ref() {
                Expr::DirectiveBlock { directives, body, .. } => (directives, body),
                _ => break,
            },
            ([Stmt::DirectiveBlock { directives, body, .. }], None) => (directives, body),
            _ => break,
        };
        out.extend(directives.iter());
        cur = next;
    }
    out
}

/// The directive as written, for a diagnostic: `` `@cast(bf16)` `` when every
/// argument is a bare identifier, `` `@cast` `` otherwise. An illegal stack
/// names both of its directives, and for `@cast @cast` the dtype is the only
/// thing telling them apart.
fn render_directive(d: &Directive) -> String {
    let args: Vec<String> = d.args.iter().filter_map(|a| match a {
        DArg::Positional(Expr::Ident(n, _)) => Some(n.clone()),
        DArg::Named { name, value: Expr::Ident(n, _), .. } => Some(format!("{}={}", name, n)),
        _ => None,
    }).collect();
    if args.is_empty() || args.len() != d.args.len() {
        format!("`@{}`", d.name)
    } else {
        format!("`@{}({})`", d.name, args.join(", "))
    }
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

/// One operand's role in the fusable walk (`Checker::fuse_expr`): a tensor
/// lane carrying its shape when the checker knows it (`None` — an unresolved
/// name), or a float-scalar broadcast.
enum FuseLane {
    Scalar,
    Tensor(Option<Shape>),
}

/// Do two tensor operands of a fused elementwise op agree, lane for lane?
/// The JIT's `fuse_infer_ty` refuses unequal concrete shapes at
/// monomorphization, so every dim pair must be *provably* equal here —
/// `equivalent()` ruling `Equal`, which admits the `?` wildcard. An
/// `Unknown` pair (`N` against `M` across fn params) is refused: some
/// monomorphization can differ, and the promise `@fuse` makes is the
/// kernel's, not the optimist's. This is deliberately stricter than
/// `Shape::matmul`'s Unknown-tolerant stance — a matmul that mismatches
/// fails at runtime on both backends, while a fused lane pair that
/// mismatches would check clean and then split the backends.
fn fuse_shapes_agree(a: &Shape, b: &Shape) -> bool {
    a.rank() == b.rank()
        && a.dims.iter().zip(&b.dims)
            .all(|(x, y)| matches!(x.equivalent(y), Equiv::Equal))
}

/// The source span of a statement, for anchoring a diagnostic at the
/// statement itself rather than at its enclosing block.
fn stmt_span(stmt: &Stmt) -> &Span {
    match stmt {
        Stmt::Let(l) => &l.span,
        Stmt::If(i) => &i.span,
        Stmt::Match(m) => &m.span,
        Stmt::Expr { span, .. } | Stmt::For { span, .. } | Stmt::While { span, .. }
        | Stmt::Loop { span, .. } | Stmt::Stage { span, .. }
        | Stmt::Directive { span, .. } | Stmt::DirectiveBlock { span, .. }
        | Stmt::Return { span, .. } => span,
        Stmt::Break(span) | Stmt::Continue(span) => span,
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

/// #474: this postfix bracket read as shape arguments — `[2, 2]`, `[SH=2]` —
/// one `(name?, value)` pair per argument. `None` when the bracket cannot be
/// shape arguments at all (a slice, `..`, a spread, or an empty bracket), which
/// keeps it on the indexing path. Mirrors `interp::shape_bracket_args`.
fn shape_bracket_pairs(op: &PostfixOp) -> Option<Vec<(Option<&str>, &Expr)>> {
    let pairs: Option<Vec<_>> = match op {
        PostfixOp::Index(elems) => elems.iter().map(|e| match e {
            IndexElem::Expr(e) => Some((None, e)),
            _ => None,
        }).collect(),
        PostfixOp::BracketArgs(args) => args.iter().map(|a| match a {
            CallArg::Positional(e) => Some((None, e)),
            CallArg::Named { name, value, .. } => Some((Some(name.as_str()), value)),
            CallArg::Spread(_) => None,
        }).collect(),
        _ => None,
    };
    pairs.filter(|p| !p.is_empty())
}

/// #474: does this postfix bracket read as shape arguments rather than as
/// indexing?
fn is_shape_bracket(op: &PostfixOp) -> bool {
    shape_bracket_pairs(op).is_some()
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

/// #761: walk a resolved `TyType`'s dimension positions for a `SymDim::Var`
/// name that isn't in `known` (a type alias's own shape parameters), and
/// record each one — used to report an alias body that names a dimension it
/// cannot possibly mean (not its own parameter, not a module constant
/// `resolve_dim` could find). Deduplicated and ordered (`BTreeSet`) so the
/// diagnostics don't depend on traversal order.
fn collect_unresolved_dim_vars(
    ty: &TyType,
    known: &std::collections::HashSet<&str>,
    out: &mut std::collections::BTreeSet<String>,
) {
    match ty {
        TyType::Tensor(inner, sh) | TyType::View(inner, sh) | TyType::KV(inner, sh) => {
            collect_unresolved_dim_vars(inner, known, out);
            for d in &sh.dims {
                collect_unresolved_symdim_vars(d, known, out);
            }
        }
        TyType::Mesh(axes) => {
            for (_, d) in axes {
                collect_unresolved_symdim_vars(d, known, out);
            }
        }
        TyType::Fn { params, ret } => {
            for p in params {
                collect_unresolved_dim_vars(p, known, out);
            }
            collect_unresolved_dim_vars(ret, known, out);
        }
        TyType::Tuple(elems) => {
            for e in elems {
                collect_unresolved_dim_vars(e, known, out);
            }
        }
        TyType::Array(inner, size) => {
            collect_unresolved_dim_vars(inner, known, out);
            collect_unresolved_symdim_vars(size, known, out);
        }
        TyType::RawPtr(inner) => collect_unresolved_dim_vars(inner, known, out),
        TyType::Named { args, .. } => {
            for a in args {
                collect_unresolved_dim_vars(a, known, out);
            }
        }
        TyType::Dim(d) => collect_unresolved_symdim_vars(d, known, out),
        _ => {}
    }
}

fn collect_unresolved_symdim_vars(
    d: &SymDim,
    known: &std::collections::HashSet<&str>,
    out: &mut std::collections::BTreeSet<String>,
) {
    match d {
        SymDim::Var(name) => {
            if !known.contains(name.as_str()) {
                out.insert(name.clone());
            }
        }
        SymDim::Add(l, r) | SymDim::Sub(l, r) | SymDim::Mul(l, r)
        | SymDim::Div(l, r) | SymDim::Mod(l, r) => {
            collect_unresolved_symdim_vars(l, known, out);
            collect_unresolved_symdim_vars(r, known, out);
        }
        SymDim::Neg(x) => collect_unresolved_symdim_vars(x, known, out),
        _ => {}
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
        // #474: a model's shape argument substitutes like any other dim, which
        // is the whole point of holding it as one — `Inner[H, W]` under
        // `H=4, W=5` becomes `Inner[4, 5]` and meets the literal.
        TyType::Dim(d) => TyType::Dim(substitute_symdim(d, args)),
        // #758: `OneOf`'s kinds are fixed labels (`Tensor`/`Str`/`Map`/`Tuple`),
        // not shape-bearing — no dim inside it could ever need substitution,
        // so it passes through like every other shape-free leaf below.
        TyType::Scalar(_) | TyType::IntLit(_) | TyType::FloatLit(_) | TyType::Enum(_) | TyType::Unit
        | TyType::Map | TyType::Budget | TyType::Unknown | TyType::Module { .. }
        | TyType::OneOf(_) => ty,
    }
}

/// #474 (parameter position): a model argument is typed bare — a model literal
/// discards the shape args it was written with — so a parameterized model
/// parameter (`Box[H, W]`) has nothing here to unify against and the call is
/// rejected, forcing every call site to spell out `f![2, 2](b, …)`. The dims
/// are not lost, though: the instance carries them, and the interpreter's
/// shape-param harvest binds them from the argument itself. So let this pair
/// through and leave the binding to the call.
///
/// Deliberately narrow — a bare model against the *same* model with args, and
/// nothing else. It is not a general "parameterized and bare are the same
/// type" rule, and in particular it is not reached from the model-field check,
/// which is a separate hole.
/// #474: a model literal that cannot show it fits a parameterized slot.
///
/// `compatible_with` lets a `Named` with no args past a `Named` with args —
/// it has to, because plenty of legitimate values type as a bare model name
/// and the parameter position leans on it deliberately (see
/// `model_arg_binds_at_runtime`, where the interpreter's harvest supplies the
/// dims from the instance at the call). In a *field* slot there is no later
/// binding step to save it: the field says `Inner[4, 5]` and whatever goes in
/// is that shape forever after. So a bare literal there was accepted on the
/// strength of its name alone, and `Inner { px: <a 7x7 buffer> }` sat in a
/// 4x5 field with `--check` clean — the emptiness read as a wildcard.
///
/// It is a unification opportunity instead: a literal whose fields pin its
/// shape args carries them (`infer_literal_shape_args`) and is compared like
/// any other. Only one that pins nothing lands here, and it is refused —
/// nothing about it can be checked, now or later.
///
/// Returns the hint for that refusal, or `None` when this pair is not it.
fn unproven_model_literal(expected: &TyType, got: &TyType, value: Option<&Expr>) -> Option<String> {
    if !matches!(value, Some(Expr::StructLit { .. })) { return None; }
    let (TyType::Named { name: e, args: eargs }, TyType::Named { name: g, args: gargs }) =
        (expected, got) else { return None };
    // Only a concrete expectation is worth refusing over. An expectation still
    // written in shape variables has nothing to contradict yet.
    if e != g || gargs.len() == eargs.len() || eargs.is_empty() { return None; }
    if !eargs.iter().all(|a| matches!(a, TyType::Dim(SymDim::Const(_)))) { return None; }
    let dims: Vec<String> = eargs.iter().map(|a| a.to_string()).collect();
    Some(format!(
        "write the shape args on the literal — `{}[{}] {{ … }}` — or give it a \
         field whose own type pins them; a bare `{} {{ … }}` makes no claim the \
         checker can compare against `{}[{}]`",
        e, dims.join(", "), e, e, dims.join(", ")))
}

/// #768: look through the parens `parse_primary` leaves behind. `(expr)` has
/// no dedicated AST node — it parses to a one-element `Expr::Tuple` (see
/// `parse_primary`'s `LParen` arm) — so a syntactic match against a bare
/// `Expr::Range` misses a parenthesized one (`(0..2)`) entirely unless it
/// unwraps that shape first, the same way `check_expr`'s own `Tuple` arm does
/// before typing it. Recurses so redundant nested parens (`((0..2))`) unwrap
/// too.
fn unwrap_parens(e: &Expr) -> &Expr {
    match e {
        Expr::Tuple(elems, _) if elems.len() == 1 => unwrap_parens(&elems[0]),
        _ => e,
    }
}

/// #792 rework: the bare scalar-type keywords `check_expr`'s own `Ident` arm
/// resolves to the opaque type-marker `Unknown` (same list, kept in sync by
/// hand — see that arm's comment). Used to catch a shape-bracket argument
/// that names a TYPE rather than a dimension (`mk[i64, 4]()`) before it
/// silently passes as an unanalyzable `Unknown`/`Var`.
fn is_scalar_type_keyword(name: &str) -> bool {
    matches!(name,
        "i8" | "i16" | "i32" | "i64" |
        "u8" | "u16" | "u32" | "u64" |
        "int4" | "int8" |
        "f16" | "bf16" | "tf32" | "f32" | "f64" |
        "fp8_e4m3" | "fp8_e5m2" | "trit" |
        "bool" | "str" | "nil")
}

/// #474: may this type stand as a shape argument? A dim is an integer, so an
/// int literal or any integral scalar qualifies. `Unknown` qualifies too — an
/// expression the checker could not type is not evidence of a wrong one, and
/// refusing it here would reject working programs over a gap elsewhere.
fn is_integral_shape_arg(ty: &TyType) -> bool {
    use crate::ast::ScalarType::*;
    match ty {
        TyType::IntLit(_) | TyType::Unknown => true,
        TyType::Scalar(s) => matches!(s,
            I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | Int4 | Int8),
        _ => false,
    }
}

fn model_arg_binds_at_runtime(param: &TyType, arg: &TyType) -> bool {
    matches!(
        (param, arg),
        (TyType::Named { name: p, args: pargs }, TyType::Named { name: a, args: aargs })
            if p == a && !pargs.is_empty() && aargs.is_empty()
    )
}

fn infer_call_shape_bindings(param: &TyType, arg: &TyType, out: &mut HashMap<String, SymDim>) {
    match (param, arg) {
        (TyType::Tensor(pt, ps), TyType::Tensor(at, ash))
        | (TyType::View(pt, ps), TyType::View(at, ash))
        | (TyType::KV(pt, ps), TyType::KV(at, ash))
        // #792 rework: `compatible_with` (types.rs ~223) accepts a `Tensor`
        // or `KV` argument for a `View`-typed parameter (one-directional —
        // see the comment there). Inference has to mirror that, or a call
        // that only compiles through the one-directional widening never
        // gets to bind its shape vars in the first place: `fn total(v: View[u8,
        // [N]]) -> i64` called as `total(t)` for a `Tensor[u8, [4]]` t would
        // otherwise refuse with "expected View[U8, [N]], got Tensor[U8, [4]]"
        // even though the same call spelled out as `total[4](t)` (which
        // bypasses inference by supplying the shape args explicitly) already
        // passed.
        | (TyType::View(pt, ps), TyType::Tensor(at, ash))
        | (TyType::View(pt, ps), TyType::KV(at, ash)) => {
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
        // #474 (parameter position): a model argument carries the shape args
        // it was constructed with, so `!b: Box[H, W]` binds H and W from the
        // instance exactly as a tensor parameter binds them from a shape. The
        // alternative — every call site spelling `f![2, 3](b, …)` — is what
        // the downstream corpus carries today. An argument written without
        // its args (an unannotated `let`, a bare literal) simply binds
        // nothing; the pair still compares, since `compatible_with` reads a
        // missing argument list as "no claim".
        (TyType::Named { name: pn, args: pa }, TyType::Named { name: an, args: aa })
            if pn == an && pa.len() == aa.len() =>
        {
            for (p, a) in pa.iter().zip(aa) {
                if let (TyType::Dim(pd), TyType::Dim(ad)) = (p, a) {
                    infer_symdim_binding(pd, ad, out);
                }
            }
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

/// Render a `let` pattern for the `pub-let` diagnostic (#596): the two shapes
/// a top-level binding actually takes, `Pattern::Ident` and a tuple of them
/// (`let (a, b) = ...`), print as their source would. Anything more exotic
/// returns `None` so the caller falls back to the bare `` `pub let` `` text
/// rather than rendering something misleading.
fn pattern_for_diagnostic(pat: &Pattern) -> Option<String> {
    match pat {
        Pattern::Ident(name, _) => Some(name.clone()),
        Pattern::Tuple(pats, _) => {
            let mut parts = Vec::with_capacity(pats.len());
            for p in pats {
                parts.push(pattern_for_diagnostic(p)?);
            }
            Some(format!("({})", parts.join(", ")))
        }
        _ => None,
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

/// Can a captured `mut` binding of this type carry a gradient (AUTODIFF.md §2,
/// #398)? Float scalars and float-element tensors do; integers, bools, strings,
/// enums, models and the rest do not — the tape has no adjoint for them.
///
/// `Unknown` (a type the checker could not resolve) is admitted rather than
/// rejected: the interpreter tapes a capture only when its *runtime value* is a
/// float scalar or float tensor, so an unresolved binding degrades to "no
/// gradient field", never to a wrong gradient — and a false error here would
/// reject working programs.
fn capture_type_is_differentiable(ty: &TyType) -> bool {
    match ty {
        TyType::Unknown => true,
        TyType::Tensor(elem, _) | TyType::View(elem, _) => elem.is_float(),
        other => other.is_float(),
    }
}

/// The identifiers a `@grad fn` body reads, split by whether the tape can see
/// the read (#398). Read-only and conservative: it does not track binding
/// scopes, so the caller filters the names against the type env (checker) or
/// the module's mutable bindings (interpreter).
#[derive(Debug, Default)]
pub(crate) struct BodyRefs {
    /// Identifiers read *directly* in the differentiated body. These are the
    /// reads the interpreter's tape records, so a captured mutable among them
    /// becomes a real tape input with a real adjoint.
    pub direct: std::collections::HashSet<String>,
    /// Identifiers read from inside a closure literal in the body. The tape
    /// does not trace closure bodies, so a captured mutable reached this way
    /// gets no adjoint and stays a compile-time error.
    pub in_closure: std::collections::HashSet<String>,
    /// Names the body *binds* — `let` idents, `for` patterns, closure params.
    /// A read of one of these is local, not a reference to a module-level
    /// binding of the same name. Used when scanning a *called* fn for reads of
    /// a captured mutable, so a callee's own local `b` is not mistaken for the
    /// module's `!b`.
    pub bound: std::collections::HashSet<String>,
    /// Method names invoked on some receiver in this body (`h.contrib()` →
    /// `contrib`). A plain call is discoverable from `direct` alone, because
    /// the callee's name *is* an `Expr::Ident`; a method call's name lives in
    /// a `PostfixOp::Field` and would otherwise be invisible to the call-graph
    /// walk — which is how a method reading a capture used to slip past the
    /// captured-mut rule and return a silent half-gradient.
    pub method_calls: std::collections::HashSet<String>,
}

impl BodyRefs {
    /// Does this body read `name` as a free (non-local) identifier, anywhere —
    /// closure literals included?
    pub fn reads_free(&self, name: &str) -> bool {
        !self.bound.contains(name)
            && (self.direct.contains(name) || self.in_closure.contains(name))
    }
}

/// Collect every identifier *referenced* (read) anywhere in a block, keeping
/// closure-literal reads separate — used by the #398 captured-mutable rule
/// (`Checker::check_fn`) and by the interpreter's capture set (`call_grad`),
/// so the diagnostic and the implementation agree on one scan.
pub(crate) fn collect_body_idents(block: &crate::ast::Block) -> BodyRefs {
    use crate::ast::*;
    type Set = std::collections::HashSet<String>;
    fn walk_e(e: &Expr, out: &mut Set, closures: &mut Set, bound: &mut Set, methods: &mut Set) {
        match e {
            Expr::Ident(name, _) => { out.insert(name.clone()); }
            Expr::BinOp { lhs, rhs, .. } => { walk_e(lhs, out, closures, bound, methods); walk_e(rhs, out, closures, bound, methods); }
            Expr::UnOp { operand, .. } => walk_e(operand, out, closures, bound, methods),
            Expr::Cast { expr, .. } => walk_e(expr, out, closures, bound, methods),
            Expr::Tuple(es, _) | Expr::TensorLit(es, _) => { for e in es { walk_e(e, out, closures, bound, methods); } }
            Expr::Range { start, end, .. } => {
                if let Some(s) = start { walk_e(s, out, closures, bound, methods); }
                if let Some(en) = end { walk_e(en, out, closures, bound, methods); }
            }
            Expr::Postfix { expr, op, .. } => {
                walk_e(expr, out, closures, bound, methods);
                // `recv.m(...)` parses as Call(Field(recv, "m")): the callee's
                // name is a `PostfixOp::Field`, never an `Expr::Ident`, so the
                // plain-call discovery in `direct` cannot see it. Record the
                // method name so the captured-mut call-graph walk can enter the
                // method body the same way it enters a called fn's.
                if matches!(op, PostfixOp::Call(_)) {
                    if let Expr::Postfix { op: PostfixOp::Field(mname), .. } = &**expr {
                        methods.insert(mname.clone());
                    }
                }
                match op {
                    PostfixOp::Call(args) | PostfixOp::BracketArgs(args) => {
                        for a in args {
                            match a {
                                CallArg::Positional(e) | CallArg::Named { value: e, .. } => walk_e(e, out, closures, bound, methods),
                                CallArg::Spread(_) => {}
                            }
                        }
                    }
                    PostfixOp::Index(elems) => {
                        for el in elems {
                            if let IndexElem::Expr(e) = el { walk_e(e, out, closures, bound, methods); }
                        }
                    }
                    PostfixOp::Constructor(fields) => { for (_, e) in fields { walk_e(e, out, closures, bound, methods); } }
                    _ => {}
                }
            }
            Expr::StructLit { type_args, fields, .. } => {
                for t in type_args { walk_e(t, out, closures, bound, methods); }
                for (_, v) in fields { walk_e(v, out, closures, bound, methods); }
            }
            Expr::If(ie) => {
                walk_e(&ie.cond, out, closures, bound, methods);
                walk_block(&ie.then_branch, out, closures, bound, methods);
                match &ie.else_branch {
                    Some(ElseBranch::Block(b)) => walk_block(b, out, closures, bound, methods),
                    Some(ElseBranch::If(i)) => walk_e(&Expr::If(i.clone()), out, closures, bound, methods),
                    None => {}
                }
            }
            Expr::Match(m) => {
                walk_e(&m.scrutinee, out, closures, bound, methods);
                for arm in &m.arms {
                    pat_names(&arm.pattern, bound);
                    if let Some(g) = &arm.guard { walk_e(g, out, closures, bound, methods); }
                    walk_e(&arm.body, out, closures, bound, methods);
                }
            }
            Expr::Block(b) => walk_block(b, out, closures, bound, methods),
            Expr::DirectiveBlock { body, .. } => walk_block(body, out, closures, bound, methods),
            // #398: a closure BODY can still capture module-level mutable state
            // (`let !bias` referenced inside `fn() { sum(bias) }`). Recurse so the
            // captured-mutable rule sees those free references — but book them as
            // `in_closure`: the tape never enters a closure body, so a capture
            // reached only this way is not differentiable. Exclude the closure's
            // OWN params — they shadow and are not captures. Reads nested inside
            // a further closure are in-closure too.
            Expr::FnLit(fl) => {
                let mut inner = Set::new();
                let mut inner_closures = Set::new();
                // The closure's own bindings stay inside it — they must not
                // mask an outer read of the same name in `bound`.
                let mut inner_bound = Set::new();
                walk_block(&fl.body, &mut inner, &mut inner_closures, &mut inner_bound, methods);
                inner.extend(inner_closures);
                for p in &fl.params { inner.remove(&p.name); }
                closures.extend(inner);
            }
            // An arena/vault/stream block feeding the loss can likewise carry a
            // captured reference — recurse into its body too.
            Expr::ArenaBlock(ab) => walk_block(&ab.body, out, closures, bound, methods),
            // #674 slice 3a: `forge.enter(b) { … }` and `<ctor> in b` are new
            // `Expr` shapes this walker must not silently skip — its own
            // arm exists precisely because `variants.sh` shows this fn is
            // one of the 7 files a new `Expr` variant breaks the build in.
            Expr::BudgetBlock(bb) => {
                walk_e(&bb.operand, out, closures, bound, methods);
                walk_block(&bb.body, out, closures, bound, methods);
            }
            Expr::AllocIn { expr, budget, .. } => {
                walk_e(expr, out, closures, bound, methods);
                walk_e(budget, out, closures, bound, methods);
            }
            Expr::Literal(..) | Expr::Nil(_) | Expr::Underscore(_) | Expr::Spread(_) => {}
        }
    }
    fn walk_block(b: &Block, out: &mut Set, closures: &mut Set, bound: &mut Set, methods: &mut Set) {
        for s in &b.stmts { walk_stmt(s, out, closures, bound, methods); }
        if let Some(t) = &b.tail_expr { walk_e(t, out, closures, bound, methods); }
    }
    /// Names a pattern binds (`let (a, b)`, `for i in …`, a match arm's
    /// payload binders). Recorded so a callee's own local of the same name is
    /// not read as a reference to a module-level binding.
    fn pat_names(p: &Pattern, bound: &mut Set) {
        match p {
            Pattern::Ident(n, _) if n != "_" => { bound.insert(n.clone()); }
            Pattern::Tuple(ps, _) => { for sub in ps { pat_names(sub, bound); } }
            Pattern::Bind(a, b, _) => { pat_names(a, bound); pat_names(b, bound); }
            Pattern::EnumVariant { bindings, .. } => {
                for sub in bindings { pat_names(sub, bound); }
            }
            _ => {}
        }
    }
    fn walk_stmt(s: &Stmt, out: &mut Set, closures: &mut Set, bound: &mut Set, methods: &mut Set) {
        match s {
            Stmt::Let(l) => { pat_names(&l.pattern, bound); walk_e(&l.value, out, closures, bound, methods) }
            Stmt::Expr { lhs, assign, .. } => {
                walk_e(lhs, out, closures, bound, methods);
                if let Some((_, rhs)) = assign { walk_e(rhs, out, closures, bound, methods); }
            }
            Stmt::Return { value: Some(e), .. } => walk_e(e, out, closures, bound, methods),
            Stmt::If(ie) => walk_e(&Expr::If(Box::new(ie.clone())), out, closures, bound, methods),
            Stmt::Match(m) => walk_e(&Expr::Match(Box::new(m.clone())), out, closures, bound, methods),
            Stmt::For { pattern, iter, body, .. } => {
                pat_names(pattern, bound);
                walk_e(iter, out, closures, bound, methods);
                walk_block(body, out, closures, bound, methods);
            }
            Stmt::While { cond, body, .. } => { walk_e(cond, out, closures, bound, methods); walk_block(body, out, closures, bound, methods); }
            Stmt::Loop { body, .. } => walk_block(body, out, closures, bound, methods),
            Stmt::Stage { body, .. } => walk_e(body, out, closures, bound, methods),
            Stmt::Directive { inner, .. } => walk_stmt(inner, out, closures, bound, methods),
            Stmt::DirectiveBlock { body, .. } => walk_block(body, out, closures, bound, methods),
            Stmt::Return { value: None, .. } | Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }
    let mut refs = BodyRefs::default();
    walk_block(block, &mut refs.direct, &mut refs.in_closure, &mut refs.bound, &mut refs.method_calls);
    refs
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
            | Expr::StructLit { span: s, .. }
            | Expr::AllocIn { span: s, .. } => s.clone(),
            Expr::Block(b)  => b.span.clone(),
            Expr::ArenaBlock(ab) => ab.span.clone(),
            Expr::BudgetBlock(bb) => bb.span.clone(),
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
                shapes: None,
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
        // #826: an assign's RHS (and, for anything but a bare `<-` onto
        // `name` itself, its LHS too) is a general `Expr` that can carry a
        // block — `x = { c <- … }`, `other <- { c <- … }` — and the previous
        // `else if assign.is_none()` shape dropped that RHS on the floor
        // whenever `assign` was `Some(..)`, silently, for every case except
        // the exact `name <- …` hit. Structured like `scan_stmt_wb`'s twin
        // arm below: the direct hit is still recorded by span, but every
        // operand is now walked regardless of which case matched.
        Stmt::Expr { lhs, assign, span } => match assign {
            Some((AssignOp::StreamArrow, rhs)) => {
                if matches!(lhs, Expr::Ident(n, _) if n == name) {
                    out.push(span.clone());
                } else {
                    collect_stream_appends_expr(lhs, name, out);
                }
                collect_stream_appends_expr(rhs, name, out);
            }
            Some((_, rhs)) => {
                collect_stream_appends_expr(lhs, name, out);
                collect_stream_appends_expr(rhs, name, out);
            }
            None => collect_stream_appends_expr(lhs, name, out),
        },
        Stmt::If(ifx) => collect_stream_appends_if(ifx, name, out),
        Stmt::Match(m) => collect_stream_appends_match(m, name, out),
        // #826: `iter`/`cond` are evaluated on every pass of the *outer*
        // loop this walker is scanning (once per outer iteration, or once
        // per inner iteration attempt), so a block hiding there is just as
        // reachable as one in `body` — `scan_stmt_wb`'s twin arms below walk
        // both for the same reason.
        Stmt::For { iter, body, .. } => {
            collect_stream_appends_expr(iter, name, out);
            collect_stream_appends_block(body, name, out);
        }
        Stmt::While { cond, body, .. } => {
            collect_stream_appends_expr(cond, name, out);
            collect_stream_appends_block(body, name, out);
        }
        Stmt::Loop { body, .. } | Stmt::DirectiveBlock { body, .. } => {
            collect_stream_appends_block(body, name, out)
        }
        Stmt::Directive { inner, .. } => collect_stream_appends_stmt(inner, name, out),
        // #826: `let`'s value, `@pp`'s per-`Stage` body, and `return`'s value
        // are all a general `Expr` that can carry a block
        // (`let x = { c <- … }`, `return { c <- … }`) — the previous blanket
        // `=> {}` for this whole group dropped them silently along with the
        // two control-flow-only variants that genuinely carry nothing.
        Stmt::Let(l) => collect_stream_appends_expr(&l.value, name, out),
        Stmt::Stage { body, .. } => collect_stream_appends_expr(body, name, out),
        Stmt::Return { value, .. } => {
            if let Some(v) = value {
                collect_stream_appends_expr(v, name, out);
            }
        }
        Stmt::Break(_) | Stmt::Continue(_) => {}
    }
}

fn collect_stream_appends_if(ifx: &IfExpr, name: &str, out: &mut Vec<Span>) {
    // #826: `cond` is a general `Expr` too (`if { c <- … } { … }` parses,
    // same as `scan_if_wb`'s twin arm below) — the previous version only
    // walked the branches and silently dropped it.
    collect_stream_appends_expr(&ifx.cond, name, out);
    collect_stream_appends_block(&ifx.then_branch, name, out);
    match &ifx.else_branch {
        Some(ElseBranch::Block(b)) => collect_stream_appends_block(b, name, out),
        Some(ElseBranch::If(inner)) => collect_stream_appends_if(inner, name, out),
        None => {}
    }
}

fn collect_stream_appends_match(m: &MatchExpr, name: &str, out: &mut Vec<Span>) {
    // #826: the scrutinee and each arm's guard are general `Expr`s that can
    // carry a block (`match ({ c <- … }) { … }`) — the previous version only
    // walked arm bodies, mirroring `scan_match_wb`'s twin below.
    collect_stream_appends_expr(&m.scrutinee, name, out);
    for arm in &m.arms {
        if let Some(g) = &arm.guard {
            collect_stream_appends_expr(g, name, out);
        }
        collect_stream_appends_expr(&arm.body, name, out);
    }
}

/// #826: made exhaustive over `Expr` (no trailing `_ => {}`) so that a new
/// `Expr` variant that can carry a block is a compile error here, not a
/// silent skip — the same shape #819 fixed for this walker's `BudgetBlock`/
/// `AllocIn` arms, and the same shape `scan_expr_wb` below (the #398
/// dead-write-back lint's twin walker) already has. Variants with no
/// sub-expression, or whose sub-expressions are grammar-restricted to a form
/// that can never carry a block, get an explicit no-op arm instead of being
/// silently swallowed.
fn collect_stream_appends_expr(e: &Expr, name: &str, out: &mut Vec<Span>) {
    match e {
        Expr::Literal(..) | Expr::Ident(..) | Expr::Underscore(_) | Expr::Spread(_)
        | Expr::Nil(_) => {}
        Expr::Tuple(es, _) | Expr::TensorLit(es, _) => {
            for x in es {
                collect_stream_appends_expr(x, name, out);
            }
        }
        Expr::Block(b) => collect_stream_appends_block(b, name, out),
        Expr::If(ifx) => collect_stream_appends_if(ifx, name, out),
        Expr::Match(m) => collect_stream_appends_match(m, name, out),
        // #826: a closure literal's body is lexically nested text just like
        // a nested loop or if/match arm (this walker's own doc comment on
        // `collect_stream_appends_block` above) — `g(fn() -> f32 { c <- … })`
        // is reachable during the outer loop the same way a directly nested
        // loop body is, so it gets walked rather than treated as an isolated
        // scope the way `scan_expr_wb`'s own `FnLit` arm does for the
        // differently-shaped #398 lint.
        Expr::FnLit(f) => collect_stream_appends_block(&f.body, name, out),
        Expr::ArenaBlock(ab) => collect_stream_appends_block(&ab.body, name, out),
        // #674 slice 3a: this walker's own `_ => {}` tail would otherwise
        // silently walk past a stream append written inside
        // `forge.enter(b) { … }` (the arena-budget design's "dangerous walker"
        // note) — so it gets an explicit arm rather than
        // falling through the wildcard below.
        // #819: the operand half is ALSO a `postfix_expr` whose `primary`
        // alternative includes `block` (§2.1 GRAMMAR.ebnf), so
        // `forge.enter({ c <- x  b }) { … }` genuinely parses with a `Block`
        // in the operand position — the same shape the `Expr::AllocIn` arm
        // below was fixed for. Walk it too.
        Expr::BudgetBlock(bb) => {
            collect_stream_appends_expr(&bb.operand, name, out);
            collect_stream_appends_block(&bb.body, name, out);
        }
        // #819: `Expr::AllocIn`'s `expr` (ctor) half is always a
        // `postfix_expr` (§2.1 GRAMMAR.ebnf) with no nested block to
        // collect in practice — but its `budget` half is ALSO a
        // `postfix_expr`, and `postfix_expr`'s own `primary` alternative
        // includes `block`, so `forge.zeros[…] in { c <- x  b }` genuinely
        // parses with a `Block` there. The previous `Expr::AllocIn { .. } =>
        // {}` arm dropped both operands silently, so a `<-` written inside
        // the `in` operand's block went unreported by the #403 lint. Walk
        // both operands unconditionally (mirroring `scan_expr_wb`'s twin
        // arm below) rather than special-casing which operand can carry a
        // block — that reasoning is exactly what went stale once already.
        Expr::AllocIn { expr, budget, .. } => {
            collect_stream_appends_expr(expr, name, out);
            collect_stream_appends_expr(budget, name, out);
        }
        Expr::DirectiveBlock { body, .. } => collect_stream_appends_block(body, name, out),
        Expr::BinOp { lhs, rhs, .. } => {
            collect_stream_appends_expr(lhs, name, out);
            collect_stream_appends_expr(rhs, name, out);
        }
        Expr::UnOp { operand, .. } => collect_stream_appends_expr(operand, name, out),
        // #826: the walker's actual bug report — a call's arguments (and a
        // method receiver, index operands, generic/bracket args, struct
        // constructor field values) are general `Expr`s too, and a block
        // literal is a valid `postfix_expr` primary in any of those
        // positions (§2.1 GRAMMAR.ebnf). Mirrors `scan_postfix_wb` below.
        Expr::Postfix { expr, op, .. } => {
            collect_stream_appends_expr(expr, name, out);
            collect_stream_appends_postfix(op, name, out);
        }
        Expr::Cast { expr, .. } => collect_stream_appends_expr(expr, name, out),
        Expr::StructLit { type_args, fields, .. } => {
            for a in type_args {
                collect_stream_appends_expr(a, name, out);
            }
            for (_, v) in fields {
                collect_stream_appends_expr(v, name, out);
            }
        }
        Expr::Range { start, end, .. } => {
            if let Some(s) = start {
                collect_stream_appends_expr(s, name, out);
            }
            if let Some(en) = end {
                collect_stream_appends_expr(en, name, out);
            }
        }
    }
}

/// #826: twin of `scan_postfix_wb` below, for the same reason — a call's
/// positional/named argument values, an index's expression/slice operands,
/// and a constructor literal's field values can all carry a block.
fn collect_stream_appends_postfix(op: &PostfixOp, name: &str, out: &mut Vec<Span>) {
    match op {
        PostfixOp::Index(elems) => {
            for el in elems {
                match el {
                    IndexElem::Expr(e) => collect_stream_appends_expr(e, name, out),
                    IndexElem::Slice { start, end, step, .. } => {
                        for o in [start, end, step] {
                            if let Some(x) = o {
                                collect_stream_appends_expr(x, name, out);
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
                    CallArg::Positional(e) => collect_stream_appends_expr(e, name, out),
                    CallArg::Named { value, .. } => collect_stream_appends_expr(value, name, out),
                    CallArg::Spread(_) => {}
                }
            }
        }
        PostfixOp::Constructor(fields) => {
            for (_, v) in fields {
                collect_stream_appends_expr(v, name, out);
            }
        }
        PostfixOp::Transpose | PostfixOp::Query | PostfixOp::Field(_) => {}
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
        // #674 slice 3a: see `walk_e`'s twin arm — a new `Expr` shape this
        // exhaustive walker must not silently skip.
        Expr::BudgetBlock(bb) => {
            scan_expr_wb(&bb.operand, sc, out);
            scan_block_wb(&bb.body, sc, out);
        }
        Expr::AllocIn { expr, budget, .. } => {
            scan_expr_wb(expr, sc, out);
            scan_expr_wb(budget, sc, out);
        }
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
