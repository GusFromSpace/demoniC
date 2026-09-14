//! The model-methods design's slices 1 and 2 (#634), at the command
//! line.
//!
//! A `fn` declared inside `model M { … }` with a `self` receiver is lowered as
//! the free function `M$$method`, its receiver an ordinary first parameter, and
//! `m.method(a)` is rewritten to `M$$method(m, a)` in `lower_postfix`'s `Call`
//! arm. On a **shape-generic** model (slice 2) that declaration is a template
//! instead, and the symbol called is one instantiation of it —
//! `Buf$$N4$$fill!`, one per distinct receiver `type_args` tuple.
//! `tools/jit_probes.py` gates the cases where both backends *answer*; what
//! belongs here is everything the battery cannot say, because `jit_probes.py`
//! scores a refusal as a warning and has no way to assert which refusal a probe
//! got (design §4).
//!
//! Five things, then:
//!
//! 1. **Aliasing.** One instance reached through two `!` names is one instance
//!    (p17). The battery would score a wrong answer here as a divergence, but
//!    only once both backends run — and the property is the one #475 got wrong
//!    silently, so it is pinned where the assertion is explicit.
//! 2. **The refusals that must stay refusals**, with class *and* message: a
//!    receiver `model_of_expr` cannot type (p22's array element, p23's call
//!    result), a type-arg-less generic receiver (slice 2 refuses rather than
//!    infers, #706/#709's posture at the call), a method's own bracket left
//!    unspelled, a method call inside a `@grad fn` body (§5's sharpest risk),
//!    and a cross-file UFCS chain (#627). Refused is the correct answer for
//!    each; *silently mis-typed* is what a bug here would look like.
//! 3. **The `$$` separator's non-collision** (design's N2). A file `M.dmc` with
//!    an unrelated private `fn method` merges to `M$method` — the exact string
//!    a single-`$` method mangling would have handed model `M`'s `method`. Both
//!    must survive one merge, and both must be callable.
//! 4. **A divergence slice 1 makes reachable but did not introduce** — p11's
//!    returned field tensor. See its test for the measurement and why it is
//!    pinned rather than fixed here.
//! 5. **Slice 2's monomorphization, asserted as symbols and not only as
//!    answers** (p16). #474 was a shape-generic method that silently never ran;
//!    one body shared across two instantiations is the same failure one layer
//!    down. `two_instantiations_get_two_symbols_p16` asserts the answers `8`
//!    and `16` *and* that `Buf$$N4$$cap` and `Buf$$N8$$cap` are two declared
//!    symbols, so a lowering that happened to answer right by luck would still
//!    fail.

use std::path::PathBuf;
use std::process::{Command, Output};

fn dmc_binary() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_dmc") {
        return PathBuf::from(path);
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    #[cfg(debug_assertions)]
    let profile = "debug";
    #[cfg(not(debug_assertions))]
    let profile = "release";
    crate_dir.join("target").join(profile).join("dmc")
}

/// A directory of `.dmc` files; the first is the entry.
struct Tree {
    dir: tempfile::TempDir,
    entry: String,
}

impl Tree {
    fn new(files: &[(&str, &str)]) -> Tree {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, src) in files {
            std::fs::write(dir.path().join(name), src).expect("write");
        }
        Tree { dir, entry: files[0].0.to_string() }
    }

    fn one(src: &str) -> Tree {
        Tree::new(&[("e.dmc", src)])
    }

    fn dmc(&self, args: &[&str]) -> Output {
        Command::new(dmc_binary())
            .args(args)
            .arg(self.dir.path().join(&self.entry))
            .output()
            .expect("spawn dmc")
    }

    fn out_of(&self, o: &Output) -> String {
        let mut s = String::from_utf8_lossy(&o.stdout).to_string();
        s.push_str(&String::from_utf8_lossy(&o.stderr));
        s.trim().to_string()
    }

    fn run_output(&self) -> String {
        let out = self.dmc(&["run"]);
        assert!(out.status.success(), "dmc run failed:\n{}", self.out_of(&out));
        self.out_of(&out)
    }

    fn jit_output(&self) -> String {
        let out = self.dmc(&["jit"]);
        assert!(out.status.success(), "dmc jit failed:\n{}", self.out_of(&out));
        self.out_of(&out)
    }

    /// Every `--json` diagnostic `dmc jit` emits, the summary excluded.
    fn jit_json(&self) -> Vec<String> {
        let out = self.dmc(&["jit", "--json"]);
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .filter(|l| l.starts_with('{') && !l.contains("\"kind\":\"summary\""))
            .map(|l| l.to_string())
            .collect()
    }

    /// Every `--json --all` diagnostic — the summary excluded. `--all` reports
    /// every refusal in the program rather than stopping at the first (#684 /
    /// #690), which is what makes one refusal *per declared symbol* visible.
    fn jit_json_all(&self) -> Vec<String> {
        let out = self.dmc(&["jit", "--json", "--all"]);
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .filter(|l| l.starts_with('{') && !l.contains("\"kind\":\"summary\""))
            .map(|l| l.to_string())
            .collect()
    }

    /// The one `jit-ineligible` diagnostic, as `(code, message)`.
    fn one_refusal(&self) -> (String, String) {
        let json = self.jit_json();
        let ineligible: Vec<&String> =
            json.iter().filter(|l| l.contains("\"kind\":\"jit-ineligible\"")).collect();
        assert_eq!(ineligible.len(), 1, "expected exactly one refusal, got {:#?}", json);
        (field_of(ineligible[0], "code"), field_of(ineligible[0], "message"))
    }

    /// #747/#755: every `jit-ineligible` diagnostic under `--json --all`, raw
    /// JSON lines, in emission order (Pass 1's roots before any Pass 2/3/4
    /// cascade). `--all` is what reaches Pass 2 past a Pass-1 item that
    /// raised a non-defect refusal and continued (`one_refusal`'s plain
    /// `--json` stops at the first refusal, which for these cases is often
    /// the very declare-time one this collect path is built to get past).
    /// Does not assert there is exactly one — since #755 D3, a model
    /// method's own declare-time refusal is recorded once as a root
    /// alongside every call site's cascade, so most of these regressions
    /// need to see more than one line.
    fn all_refusals_all(&self) -> Vec<String> {
        self.jit_json_all().into_iter()
            .filter(|l| l.contains("\"kind\":\"jit-ineligible\""))
            .collect()
    }
}

/// A `--json` line's string field. The messages here contain no escapes, so a
/// scan to the next `"` is enough and pulls in no JSON dependency.
fn field_of(line: &str, key: &str) -> String {
    let needle = format!("\"{key}\":\"");
    let start = line.find(&needle).unwrap_or_else(|| panic!("no `{key}` in {line}"))
        + needle.len();
    let rest = &line[start..];
    let end = rest.find('"').expect("unterminated field");
    rest[..end].to_string()
}

// ─── what lowers ─────────────────────────────────────────────────────────────

/// p17, in the spelling slice 1 reaches: `fn two(!a: C, !b: C)` called `two(c,
/// c)`, each parameter mutating through a method. One instance, two names, and
/// the mutations must compound — `11`, not `1` or `10`.
///
/// A `!` model parameter is the bare i64 pointer to the instance's field struct
/// on both backends, and `method_fn_decl` puts `!self` on that same path, so
/// this is the property falling out of option A rather than one it adds. It is
/// pinned because the failure mode is a *wrong answer*: #475 was a `!` tensor
/// out-parameter lost through `self`, silently, and nothing refused.
///
/// The field is a tensor element rather than the design probe's scalar so the
/// case does not also depend on the whole-field scalar store (#635, slice 0);
/// the aliasing property under test is the same either way.
#[test]
fn one_instance_through_two_mut_names_is_one_instance_p17() {
    let t = Tree::one(
        "model C {\n\
         \x20   !t: Tensor[i64, [1]]\n\
         \x20   fn bump!(!self, k: i64) -> i64 { self.t[0] = self.t[0] + k  self.t[0] }\n\
         }\n\
         fn two(!a: C, !b: C) -> i64 { a.bump!(1)  b.bump!(10) }\n\
         fn main() -> i64 {\n\
         \x20   let !c = C { t: forge.zeros[i64, [1]] }\n\
         \x20   two(c, c)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 11");
    assert_eq!(t.jit_output(), "=> 11", "the JIT must agree with `dmc run`");
}

/// p10: a method and a free function of the same name, both applicable. The
/// method wins — `SPEC.md §4.12` lists genuine model methods first among UFCS's
/// exceptions and `desugar.rs`'s guard suppresses the rewrite for any name a
/// model declares, so the free `fn tag(m: M)` is reachable only when written as
/// one. The rewrite must not widen that: it fires on the postfix spelling only.
#[test]
fn a_method_wins_over_a_same_named_free_fn_p10() {
    let t = Tree::one(
        "model M {\n\
         \x20   !n: i64\n\
         \x20   fn tag(self) -> i64 { 1 }\n\
         }\n\
         fn tag(m: M) -> i64 { 2 }\n\
         fn main() -> i64 {\n\
         \x20   let !m = M { n: 0 }\n\
         \x20   m.tag() * 10 + tag(m)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 12");
    assert_eq!(t.jit_output(), "=> 12");
}

/// The design's **N2**: `M.dmc` declares `model M` with a method `method`, and
/// beside it an unrelated *private* `fn method`. `merge.rs` renames the private
/// fn to `M$method` — the identical string a single-`$` method mangling would
/// have produced for the model's `method`, both landing in `Jit::fns`, one flat
/// name map. Under `M$$method` they cannot meet.
///
/// The assertion is the answer, not the absence of a crash: `707` is `7` from
/// the method plus `700` from the private fn, so neither shadowed the other. A
/// single-`$` spelling would have hit `declare_fn`'s "duplicate function" check
/// — which is to say it would have refused two legal, unrelated files once
/// merged.
#[test]
fn a_method_symbol_cannot_collide_with_a_merge_renamed_private_fn_n2() {
    let t = Tree::new(&[
        (
            "e.dmc",
            "use \"M.dmc\"\n\
             fn main() -> i64 {\n\
             \x20   let !m = M { n: 7 }\n\
             \x20   m.method() + via_private(m)\n\
             }\n",
        ),
        (
            "M.dmc",
            "pub model M {\n\
             \x20   !n: i64\n\
             \x20   fn method(self) -> i64 { self.n }\n\
             }\n\
             fn method(m: M) -> i64 { m.n * 100 }\n\
             pub fn via_private(m: M) -> i64 { method(m) }\n",
        ),
    ]);
    assert_eq!(t.run_output(), "=> 707");
    assert_eq!(t.jit_output(), "=> 707");
}

/// A refusal raised inside an *imported private model's* method must report the
/// file that declared it and quote no merged name — the rule
/// `jit_multifile_merge.rs` states for every other item, now that a method has
/// a symbol of its own (`lib$Node$$sq`).
///
/// It also exercises `Jit::demangle`'s and `decl_file_of`'s backoff *through*
/// the `$$` pair with no change to either: the first `rfind('$')` lands
/// mid-separator, nothing matches the trailing-`$` remainder, and the loop's
/// own backoff steps one `$` further left, where the merge's real prefix
/// boundary is.
#[test]
fn a_refusal_inside_an_imported_methods_body_names_its_file_and_no_symbol() {
    let t = Tree::new(&[
        ("e.dmc", "use \"lib.dmc\"\nfn main() -> i64 { lib_val() }\n"),
        (
            "lib.dmc",
            "model Node {\n\
             \x20   v: i64\n\
             \x20   fn sq(self) -> i64 { self.v ** 2 }\n\
             }\n\
             fn mk() -> Node { Node { v: 3 } }\n\
             pub fn lib_val() -> i64 { let n = mk()  n.sq() }\n",
        ),
    ]);
    assert_eq!(t.run_output(), "=> 9");
    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{json:#?}");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-operand-type");
    assert!(!message.contains('$'), "a symbol reached the user in {message:?}");
    assert!(
        field_of(&json[0], "file").ends_with("lib.dmc"),
        "the refusal belongs to the declaring file, got {:?}",
        field_of(&json[0], "file"),
    );
}

// ─── the price of eager declaration ──────────────────────────────────────────

/// Pass 2 (`compile_program_inner`, `jit.rs:3068-3109`) `define_fn`s every
/// declared `FnDecl` — including the one Pass 1 emits per `ModelMember::Method`
/// — whether or not `main` ever calls it. A method whose body contains a
/// construct the JIT does not lower therefore refuses the **whole program**,
/// not merely the call: `main` never reaches `never`, `dmc run` answers `3`,
/// and `dmc jit` refuses at the method body regardless.
///
/// This is not new eagerness slice 1 introduces — a free fn is declared and
/// defined the same way in the same two passes, and refuses identically when
/// never called (verified alongside the model case below, on a binary built
/// from this same tree). Coordinator decision: keep eager declaration. It
/// matches the design (the model-methods design's slice 1 rule: "one `FnDecl` per
/// method") and it matches how a free fn is already treated — deferring a
/// method's definition to first call would be a new, method-only mechanism
/// with no free-fn analog, not a fix. The price: a model with an unlowerable,
/// never-called method cannot be JIT-run at all today, same as a file with an
/// unlowerable, never-called free fn.
#[test]
fn an_uncalled_methods_unlowered_construct_refuses_the_whole_program() {
    let method = Tree::one(
        "model M {\n\
         \x20   !n: i64\n\
         \x20   fn never(self, s: str) -> i64 {\n\
         \x20       let b = s.trim()\n\
         \x20       self.n\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !m = M{n:3}\n\
         \x20   m.n\n\
         }\n",
    );
    assert_eq!(method.run_output(), "=> 3", "`never` is never called");
    let (code, message) = method.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(message.contains(".trim()"), "{message:?}");

    // The free-fn analog: no model, no method, an uncalled `fn` with the same
    // unlowered construct. Same verdict, same class, confirming the price is
    // not new.
    let free_fn = Tree::one(
        "fn never(s: str) -> i64 {\n\
         \x20   let b = s.trim()\n\
         \x20   0\n\
         }\n\
         fn main() -> i64 { 3 }\n",
    );
    assert_eq!(free_fn.run_output(), "=> 3", "`never` is never called");
    let (code, message) = free_fn.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(message.contains(".trim()"), "{message:?}");
}

/// The same price, paid through slice 0's field-store refusal instead of a
/// str-method refusal: an uncalled method rebinding a `Tensor`-typed field
/// (which slice 0 of the model-methods design leaves refused) takes
/// down the whole program under `dmc jit`, never called or not.
#[test]
fn an_uncalled_methods_tensor_field_rebind_refuses_the_whole_program() {
    let t = Tree::one(
        "model M {\n\
         \x20   !n: i64\n\
         \x20   !t: Tensor[i64, [3]]\n\
         \x20   fn never!(!self) -> i64 {\n\
         \x20       self.t = forge.zeros[i64, [3]]\n\
         \x20       self.n\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !m = M{n:3, t: forge.zeros[i64,[3]]}\n\
         \x20   m.n\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 3", "`never!` is never called");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-assign-form");
    assert!(message.contains('t'), "the message names the field: {message:?}");
}

/// **And the price a shape-generic model does NOT pay** — the asymmetry slice 2
/// introduces, recorded because it is real and deliberate rather than
/// discovered later.
///
/// A generic model's method is a *template*, so it is declared only when a call
/// site supplies a shape tuple. An uncalled one with the identical unlowered
/// body is therefore never declared and never defined, and the program lowers.
/// That is not a special case for methods: it is exactly how an uncalled free
/// template behaves today, asserted alongside. The two rules are the same rule
/// — "a template is instantiated at a call site, a monomorphic fn is declared
/// up front" — applied to a method for the first time.
#[test]
fn an_uncalled_generic_models_method_is_never_declared() {
    let t = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn never(self, s: str) -> i64 {\n\
         \x20       let b = s.trim()\n\
         \x20       N\n\
         \x20   }\n\
         \x20   fn cap(self) -> i64 { N * 2 }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !b = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   b.cap()\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 8");
    assert_eq!(t.jit_output(), "=> 8", "`never` is never instantiated, so never lowered");

    // The free-template analog: an uncalled shape-generic `fn` with the same
    // unlowered construct lowers identically, which is what makes the rule one
    // rule rather than a method-only exemption.
    let free_template = Tree::one(
        "fn never[K](s: str, t: Tensor[i64, [K]]) -> i64 {\n\
         \x20   let b = s.trim()\n\
         \x20   K\n\
         }\n\
         fn main() -> i64 { 8 }\n",
    );
    assert_eq!(free_template.run_output(), "=> 8");
    assert_eq!(free_template.jit_output(), "=> 8");
}

/// p12's shape on a shape-generic model: a method that recurses through `self`
/// while mutating a scalar field, at two instantiations in one program. Two
/// things at once — the synthetic declaration must be in `self.fns` **before**
/// its own body is lowered or the self-call cannot resolve (design §5's first
/// ordering hazard, one instantiation deep), and the two instantiations must
/// not share state.
#[test]
fn a_generic_models_method_recurses_and_instantiations_stay_separate() {
    let t = Tree::one(
        "model Buf[N] {\n\
         \x20   !n: i64\n\
         \x20   fn down!(!self, k: i64) -> i64 {\n\
         \x20       if k == 0 { self.n } else { self.n = self.n + 1  self.down!(k - 1) }\n\
         \x20   }\n\
         }\n\
         fn main() -> nil {\n\
         \x20   let !a = Buf[4] { n: 0 }\n\
         \x20   let !b = Buf[8] { n: 100 }\n\
         \x20   print(to_str(a.down!(5)))\n\
         \x20   print(to_str(b.down!(3)))\n\
         \x20   nil\n\
         }\n",
    );
    assert_eq!(t.run_output(), "5\n103");
    assert_eq!(t.jit_output(), "5\n103");
}

/// §5's `@grad` rule holds for a generic receiver too. Slice 2 puts one more
/// symbol — `S$N2$$twice` — within `ad_lower_call`'s reach, and the refusal
/// keys on `model_of_expr`, which answers for a generic receiver exactly as it
/// does for a plain one. Without this the JIT would differentiate through a
/// method the interpreter's tape cannot cross: silent wrong gradients rather
/// than a refusal.
#[test]
fn a_generic_models_method_inside_a_grad_fn_still_refuses_as_jit_grad() {
    let t = Tree::one(
        "model S[N] {\n\
         \x20   !k: f32\n\
         \x20   fn twice(self, x: f32) -> f32 { x * 2.0 }\n\
         }\n\
         @grad fn loss(!w: Tensor[f32, [2]], s: S[2]) -> f32 { s.twice(sum(w .* w)) }\n\
         fn main() -> nil {\n\
         \x20   let !w = forge.zeros[f32, [2]]\n\
         \x20   w[0] = 2.0\n\
         \x20   let (l, g) = loss.fwd_bwd(w, S[2]{ k: 1.0 })\n\
         \x20   print(to_str(l))\n\
         \x20   nil\n\
         }\n",
    );
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-grad");
    assert!(message.contains(".twice()"), "{message:?}");
    assert!(message.contains("`S`"), "the message names the model: {message:?}");
}

// ─── what must stay refused ──────────────────────────────────────────────────

/// p22: the receiver is an **array element**. `model_of_expr` does not type an
/// `Index`, so the rewrite does not fire and the call keeps refusing. That is
/// the correct answer rather than merely a convenient one — `model_of_expr`
/// answering `None` is the same guard that keeps a cross-file UFCS chain out
/// (below), so widening it is not free. The design's corpus census finds zero
/// `Index` receivers.
///
/// Class *and* the method name are asserted: a slice-1 bug that lowered this
/// against the wrong receiver would answer a number, and a bug that lowered it
/// against the array's base pointer would answer a plausible one. The message
/// must also **not** recite the str-method list — `model_of_expr` answering
/// `None` used to fall all the way through to `lower_call`'s str-method
/// fallback, which named the wrong family for a receiver that was never a
/// string; `model_of_expr_diag_hint` widens only that message, naming the
/// model instead.
#[test]
fn an_array_element_receiver_still_refuses_p22() {
    let t = Tree::one(
        "model M {\n\
         \x20   !t: Tensor[i64, [1]]\n\
         \x20   fn bump!(!self) -> i64 { self.t[0] = self.t[0] + 1  self.t[0] }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !arr = forge.uninit[M, [2]]\n\
         \x20   arr[0] = M { t: forge.zeros[i64, [1]] }\n\
         \x20   arr[1] = M { t: forge.zeros[i64, [1]] }\n\
         \x20   let a = arr[0].bump!()\n\
         \x20   a * 100 + arr[0].t[0]\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 101");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(message.contains(".bump!()"), "the message names the method: {message:?}");
    assert!(message.contains("`M`"), "the message names the model: {message:?}");
    assert!(!message.contains("str"), "the old str-method message is gone: {message:?}");
}

/// p23: the receiver is a **call result**, `m.bump!().bump!()`. Same guard,
/// same verdict — and the case matters beyond the census because a method
/// returning `self` returns the caller's own instance (`dmc run` answers `22`),
/// so a lowering that guessed at the receiver here would be wrong about
/// aliasing, not merely about types. Same str-message guard as p22: the
/// message must name the model, not the str-method list.
#[test]
fn a_call_result_receiver_still_refuses_p23() {
    let t = Tree::one(
        "model M {\n\
         \x20   !t: Tensor[i64, [1]]\n\
         \x20   fn bump!(!self) -> M { self.t[0] = self.t[0] + 1  self }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !m = M { t: forge.zeros[i64, [1]] }\n\
         \x20   let r = m.bump!().bump!()\n\
         \x20   r.t[0] * 10 + m.t[0]\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 22");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(message.contains(".bump!()"), "the message names the method: {message:?}");
    assert!(message.contains("`M`"), "the message names the model: {message:?}");
    assert!(!message.contains("str"), "the old str-method message is gone: {message:?}");
}

/// The same guard as p22/p23, through a different receiver shape: a plain
/// free fn returning a model (`mk()`), called immediately (`mk().get()`).
/// `model_of_expr` cannot type a call result either way, but the callee here
/// is an ordinary `fn`, not a model method — `model_of_expr_diag_hint`'s
/// `Call` arm has to look the return type up in `self.fns` directly rather
/// than through `model_methods`, a separate lookup path from p23's.
#[test]
fn a_free_fn_call_result_receiver_still_refuses_naming_the_model() {
    let t = Tree::one(
        "model M {\n\
         \x20   !n: i64\n\
         \x20   fn get(self) -> i64 { self.n }\n\
         }\n\
         fn mk() -> M { M { n: 5 } }\n\
         fn main() -> i64 {\n\
         \x20   mk().get()\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 5");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(message.contains(".get()"), "the message names the method: {message:?}");
    assert!(message.contains("`M`"), "the message names the model: {message:?}");
    assert!(!message.contains("str"), "the old str-method message is gone: {message:?}");
}

/// p5, slice 2: a method on a **shape-generic** model lowers, and the model's
/// shape parameter is bound from the receiver's own instantiation — `N` is `4`
/// inside `cap` because the receiver is a `Buf[4]`, `SPEC.md §3.10`'s "bind
/// from the instance" read off `TyKind::Model("Buf", [4])`.
#[test]
fn a_method_on_a_shape_generic_model_lowers_p5() {
    let t = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn cap(self) -> i64 { N * 2 }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !b = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   b.cap()\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 8");
    assert_eq!(t.jit_output(), "=> 8");
}

/// **p16 — the #474 regression gate, and the reason slice 2 is its own slice.**
///
/// The same model at two instantiations in one program. #474's failure was a
/// shape-generic method that *silently never ran*; one shared body across two
/// instantiations is the same failure one layer down, and it answers `8` and
/// `8` where the correct answer is `8` and `16`. Neither backend would have
/// said a word.
///
/// The behaviour is asserted first, then the mechanism underneath it: **two
/// distinct symbols are declared**, one per `type_args` tuple. A refusal inside
/// the method's body is what makes the symbol names observable at the command
/// line — `--json --all` reports each refusing item by name (#690 collects
/// every one rather than stopping at the first) — so the second program below
/// is the first with an unlowerable construct added to the shared body, and it
/// reports `Buf$$N4$$cap` and `Buf$$N8$$cap`, not one name twice.
#[test]
fn two_instantiations_get_two_symbols_p16() {
    let t = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn cap(self) -> i64 { N * 2 }\n\
         }\n\
         fn main() -> nil {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   print(to_str(a.cap()))\n\
         \x20   print(to_str(b.cap()))\n\
         \x20   nil\n\
         }\n",
    );
    assert_eq!(t.run_output(), "8\n16");
    assert_eq!(t.jit_output(), "8\n16", "not `8` and `8` — one body per instantiation");

    // The same two call sites, with a construct the JIT does not lower added to
    // the shared body, so each instantiation reports its own symbol.
    let observable = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn cap(self, s: str) -> i64 { s.trim()  N * 2 }\n\
         }\n\
         fn main() -> nil {\n\
         \x20   let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n\
         \x20   print(to_str(a.cap(\"x\")))\n\
         \x20   print(to_str(b.cap(\"y\")))\n\
         \x20   nil\n\
         }\n",
    );
    let items: Vec<String> = observable
        .jit_json_all()
        .iter()
        .filter(|l| l.contains("\"kind\":\"jit-ineligible\""))
        .map(|l| field_of(l, "item"))
        .collect();
    assert!(items.iter().any(|i| i == "Buf$$N4$$cap"), "declared at [4]: {items:?}");
    assert!(items.iter().any(|i| i == "Buf$$N8$$cap"), "declared at [8]: {items:?}");
}

/// A shape-generic model reached through a **type-arg-less** receiver — `fn
/// f(b: Buf)` for `model Buf[N]` — has no honest instantiation key, and slice 2
/// does not invent one. q6 measures that the JIT cannot infer a shape argument
/// from a model argument and q9 that the call cannot spell the model's bracket,
/// so the receiver's own written type is the only source, and it is empty here.
///
/// Refused as `jit-type-form` with #706/#709's wording — the same class, and
/// the same actionable spelling, that a *field read* through this identical
/// receiver already carries. Inventing a binding instead is exactly the shape
/// of #474.
#[test]
fn a_type_arg_less_generic_receiver_refuses_at_the_method_call() {
    let t = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn cap(self) -> i64 { N * 2 }\n\
         }\n\
         fn f(b: Buf) -> i64 { b.cap() }\n\
         fn main() -> i64 {\n\
         \x20   let !b = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   f(b)\n\
         }\n",
    );
    // #728: `f`'s own `b: Buf` is now a checker warning too (a type-arg-less
    // annotation of a shape-generic model) — `dmc run` prints it alongside
    // the answer, which this asserts unchanged. #728 rework (reviewer
    // defect 1): param position gets the "declare it on the function"
    // hint, not "write it explicitly" (which would be a hard
    // `undefined identifier N` type error in this position) — see
    // `type_arg_less_hint`'s doc comment in check.rs.
    assert_eq!(
        t.run_output(),
        "=> 8\n\
         warning at 5:9: bare-generic-model: parameter `b` of function `f` names \
         `Buf` with no argument list — `Buf` is shape-parameterized, and an empty \
         argument list is never a wildcard (SPEC.md §3.10); this warning becomes a \
         compile-time error at 0.0.6-draft\n\
         \x20 hint: declare the shape parameter on the function: `fn f[N](b: Buf[N])`",
        "dmc run binds N from the instance",
    );
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-type-form");
    assert!(message.contains(".cap()"), "the message names the method: {message:?}");
    assert!(message.contains("shape arguments are not known here"), "{message:?}");
    assert!(message.contains("`m: Buf[4]`"), "the actionable spelling: {message:?}");
}

/// A **nested** shape-generic model receiver, `outer.inner.get()`, where the
/// inner model's shape argument is spelled in the outer's own shape param
/// (`inner: Kernel[N]` inside `model Wrap[N]`). `model_layouts` is built with an
/// empty env, so reading the field's layout type alone hands back `Kernel[]` —
/// a type-arg-less receiver, which would refuse a program whose shape arguments
/// are perfectly well known. `model_ty_of_expr` re-resolves the field's AST type
/// against the outer instantiation first, the same #267 step the field *read*
/// already takes.
#[test]
fn a_nested_generic_model_receiver_resolves_through_the_outer_instantiation() {
    let t = Tree::one(
        "model Kernel[K] {\n\
         \x20   !w: Tensor[i64, [K]]\n\
         \x20   fn width(self) -> i64 { K }\n\
         }\n\
         model Wrap[N] {\n\
         \x20   !inner: Kernel[N]\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !k = Kernel[3] { w: forge.zeros[i64, [3]] }\n\
         \x20   let !o = Wrap[3] { inner: k }\n\
         \x20   o.inner.width()\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 3");
    assert_eq!(t.jit_output(), "=> 3");
}

/// A method that declares a bracket of **its own** (`fn blit![SH](self, ..)`),
/// in both spellings. Slice 2 lowers the bracketed one and refuses the
/// bracket-less one, and the split is a deliberate line rather than an
/// accident of the parser.
///
/// **Bracketed** (`m.blit![4](t)`) parses as `Index` over `Field`, which is why
/// it was `jit-indirect-call` before this slice: `lower_call`'s `Index` arm
/// wants a bare `Ident` under the bracket and a method's receiver is never one.
/// `lower_model_method_call` now takes that shape too, binds `SH` from the
/// bracket, and instantiates. #671 (landed as #679) is what makes this safe to
/// lower: before it, `call_bracketed_method` never ran `apply_writebacks`, so a
/// correct JIT lowering of the bracketed form would have *diverged* from the
/// interpreter rather than closed a gap.
///
/// **Bracket-less** (`m.blit!(t)`) keeps refusing. `dmc run` infers `SH` from
/// the argument's own tensor type and answers `4`, so this is a program the
/// reference backend accepts and the JIT declines — a located gap, not a
/// divergence. The reason it is not inferred here is mechanical: the ordinary
/// arguments are lowered inside `lower_call`, and lowering them a second time
/// up front, to unify against, would emit every argument's side effects twice.
#[test]
fn a_methods_own_bracket_lowers_when_spelled_and_refuses_when_not() {
    let bracketed = Tree::one(
        "model M {\n\
         \x20   !n: i64\n\
         \x20   fn blit![SH](self, t: Tensor[i64, [SH]]) -> i64 { SH }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !m = M { n: 1 }\n\
         \x20   let t = forge.zeros[i64, [4]]\n\
         \x20   m.blit![4](t)\n\
         }\n",
    );
    assert_eq!(bracketed.run_output(), "=> 4");
    assert_eq!(bracketed.jit_output(), "=> 4");

    let t = Tree::one(
        "model M {\n\
         \x20   !n: i64\n\
         \x20   fn blit![SH](self, t: Tensor[i64, [SH]]) -> i64 { SH }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !m = M { n: 1 }\n\
         \x20   let t = forge.zeros[i64, [4]]\n\
         \x20   m.blit!(t)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 4", "SH is inferable from t; dmc run succeeds");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(message.contains(".blit!()"), "{message:?}");
    assert!(message.contains("`M`"), "the message names the model: {message:?}");
    assert!(message.contains("`SH`"), "the message names the unbound param: {message:?}");
    assert!(message.contains(".blit![SH](..)"), "the actionable spelling: {message:?}");
    assert!(!message.contains("str"), "the old str-method message is gone: {message:?}");
}

/// Both brackets at once: a method with its own shape parameter, on a
/// shape-generic model. The model's half binds from the receiver and the
/// method's from the call, in that order, and the symbol spells both —
/// `Buf$$N4$$scale!$K3`.
#[test]
fn a_generic_models_method_with_its_own_bracket_binds_both_halves() {
    let t = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn scale![K](self, u: Tensor[i64, [K]]) -> i64 { N * 100 + K }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !b = Buf[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   let u = forge.zeros[i64, [3]]\n\
         \x20   b.scale![3](u)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 403");
    assert_eq!(t.jit_output(), "=> 403");
}

/// The model-methods design: a method call inside a `@grad fn` body must be
/// **refused**, not differentiated. The checker permits methods under `@grad`
/// (#398 scans every same-named method across every model), but the
/// interpreter's tape does not cross the call — p18's method body is `x * 2.0`,
/// no field read at all, and `dmc run` still fails with "must produce a scalar
/// derived from a differentiable input", while the same expression inlined
/// answers on both backends. Slice 1 brings `M$$method` within reach of the
/// forward pass, so without this arm the JIT would differentiate what `dmc run`
/// cannot: silent wrong gradients rather than a refusal.
///
/// `jit-grad` — the class its neighbours p19/p20 already carry — and the
/// message names the method and the model rather than "indirect calls", which
/// is not what the program wrote.
#[test]
fn a_method_call_inside_a_grad_fn_refuses_as_jit_grad() {
    let t = Tree::one(
        "model S {\n\
         \x20   !k: f32\n\
         \x20   fn twice(self, x: f32) -> f32 { x * 2.0 }\n\
         }\n\
         @grad fn loss(!w: Tensor[f32, [2]], s: S) -> f32 { s.twice(sum(w .* w)) }\n\
         fn main() -> nil {\n\
         \x20   let !w = forge.zeros[f32, [2]]\n\
         \x20   w[0] = 2.0\n\
         \x20   let s = S { k: 3.0 }\n\
         \x20   let (l, g) = loss.fwd_bwd(w, s)\n\
         \x20   print(to_str(l as i64))\n\
         \x20   nil\n\
         }\n",
    );
    // `dmc run` cannot trace it either — the reference backend fails at
    // runtime, so the JIT refusing is the stricter of two non-answers, never a
    // capability being taken away.
    assert!(!t.dmc(&["run"]).status.success(), "the interpreter cannot trace it either");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-grad");
    assert!(message.contains(".twice()"), "{message:?}");
    assert!(message.contains("`S`"), "the message names the model: {message:?}");
}

/// #627: a UFCS call onto an **imported** `pub fn` never desugars, because
/// `desugar_ufcs` builds its function set from the one file it is handed. It
/// reaches the backend as a postfix `Field` then `Call` — the same shape a
/// model method call has. The rewrite must key on `model_of_expr` answering
/// `Some`, never on the chain's shape, or slice 1 would silently lower a
/// program `merge.rs` deliberately leaves refused.
///
/// The receiver here is an `i64`, so `model_of_expr` answers `None` and the
/// call falls through untouched.
#[test]
fn a_cross_file_ufcs_chain_still_refuses_627() {
    let t = Tree::new(&[
        (
            "e.dmc",
            "use \"lib.dmc\"\nfn main() -> i64 {\n    let v = 21\n    v.dbl()\n}\n",
        ),
        ("lib.dmc", "pub fn dbl(v: i64) -> i64 { v * 2 }\n"),
    ]);
    let json = t.jit_json();
    let ineligible: Vec<&String> =
        json.iter().filter(|l| l.contains("\"kind\":\"jit-ineligible\"")).collect();
    assert_eq!(ineligible.len(), 1, "{json:#?}");
    assert_eq!(field_of(ineligible[0], "code"), "jit-unknown-fn");
    assert!(
        field_of(ineligible[0], "message").contains(".dbl()"),
        "{:?}",
        field_of(ineligible[0], "message"),
    );
}

// ─── a divergence slice 1 makes reachable but did not introduce ──────────────

/// p11, and the finding underneath it.
///
/// The design predicted `dmc run`'s `0` — a returned **field** does not alias,
/// because `SPEC.md §3.4`'s live-alias rule is a property of the *binding form*
/// `let !x = m.field`, not of the tensor's provenance — and expected the JIT to
/// agree once the call lowered. It does not: `dmc jit` answers `6`.
///
/// The cause is not the method, the field, or slice 1. The second half of this
/// test is the same program with **no model and no method** — a plain tensor
/// passed to a free function and returned — and it diverges identically on a
/// binary built before slice 1. `dmc run` copies a tensor out of a call; the
/// JIT hands back the same pointer. The design's §2 baseline table has no `q`
/// row for a *returned* tensor, which is why it went unmeasured.
///
/// Pinned rather than fixed: which side is correct is the #650 question in a
/// new position (`SPEC.md §3.4`'s copy rule against the JIT's pointer
/// representation), it is not a model-methods change, and fixing it in the JIT
/// would alter every function returning a tensor. Slice 1 only makes it
/// *reachable* through a method, so p11 stays out of `tools/jit_probes.py` —
/// the battery scores a divergence as a failure with no allowlist on that
/// branch — and lives here, where the two answers can be stated out loud. The
/// same posture `jit.rs`'s `a_mut_tensor_arg_that_is_not_an_identifier_still_diverges`
/// takes for #636.
///
/// COORDINATOR WAIVER (PR #705): filed as #707, the return-side clause of the
/// #650 decision — #650 rules on a tensor passed IN through a `!` param;
/// #707 asks the same question of a tensor returned OUT. Waived rather than
/// blocked on: the divergence predates this PR (the `via_free_fn` half below
/// runs unchanged on `main`), so slice 1 did not introduce it, only made it
/// reachable through a receiver.
#[test]
fn a_returned_tensor_diverges_and_did_so_before_methods_p11() {
    let via_method = Tree::one(
        "model B {\n\
         \x20   !t: Tensor[i64, [3]]\n\
         \x20   fn peek(self) -> Tensor[i64, [3]] { self.t }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !b = B { t: forge.zeros[i64, [3]] }\n\
         \x20   let !v = b.peek()\n\
         \x20   v[0] = 6\n\
         \x20   b.t[0]\n\
         }\n",
    );
    assert_eq!(via_method.run_output(), "=> 0", "the interpreter copies the returned tensor");
    assert_eq!(via_method.jit_output(), "=> 6", "the JIT returns the same pointer");

    // The same divergence with no model, no field and no method anywhere: it is
    // a property of returning a tensor, and predates slice 1 entirely.
    let via_free_fn = Tree::one(
        "fn id(t: Tensor[i64, [3]]) -> Tensor[i64, [3]] { t }\n\
         fn main() -> i64 {\n\
         \x20   let !a = forge.zeros[i64, [3]]\n\
         \x20   let !v = id(a)\n\
         \x20   v[0] = 6\n\
         \x20   a[0]\n\
         }\n",
    );
    assert_eq!(via_free_fn.run_output(), "=> 0");
    assert_eq!(via_free_fn.jit_output(), "=> 6");
}

/// A second shape of the SAME #636 root p14 already pins, found reviewing
/// slice 1: a method forwarding its **own tensor field** to another fn's `!`
/// param, one call deeper than `main`. `self.t` is a `Postfix{Field}`
/// expression, not a bare `Expr::Ident` — exactly the shape
/// `interp.rs`'s `apply_writebacks` drops (`:7623`) — so the callee's write
/// reaches the JIT's pointer but never makes it back through the
/// interpreter's tape: `dmc run` answers `0`, `dmc jit` answers `909`.
///
/// Not method-specific, same as p11 above: the second half is the identical
/// shape with **no model method at all** — a plain top-level `fn go!(!m: M)`
/// forwarding `m.t` — and it diverges identically. Slice 1 only makes the
/// method-body spelling reachable; the root is #636, unchanged by this PR.
/// Pinned here rather than in `tools/jit_probes.py`'s `MUST_LOWER` for the
/// same reason as p14: a divergence has no allowlist branch in that battery,
/// so a name here would score a red battery on a case the interpreter, not
/// the JIT, gets wrong.
#[test]
fn a_methods_own_tensor_field_forwarded_to_a_mut_param_diverges_636() {
    let via_method = Tree::one(
        "fn fill(!t: Tensor[i64, [1]]) -> nil { t[0] = 909  nil }\n\
         model M {\n\
         \x20   !t: Tensor[i64, [1]]\n\
         \x20   fn go!(!self) -> nil { fill(self.t)  nil }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !m = M { t: forge.zeros[i64, [1]] }\n\
         \x20   m.go!()\n\
         \x20   m.t[0]\n\
         }\n",
    );
    assert_eq!(via_method.run_output(), "=> 0", "the interpreter drops the non-ident writeback");
    assert_eq!(via_method.jit_output(), "=> 909", "the JIT writes back through the pointer");

    // The same shape with no model method anywhere: `go!` is an ordinary
    // top-level fn taking a `!` model param and forwarding its field. Predates
    // slice 1 entirely — nothing here is a synthesized `M$$method`.
    let via_free_fn = Tree::one(
        "fn fill(!t: Tensor[i64, [1]]) -> nil { t[0] = 909  nil }\n\
         model M { !t: Tensor[i64, [1]] }\n\
         fn go!(!m: M) -> nil { fill(m.t)  nil }\n\
         fn main() -> i64 {\n\
         \x20   let !m = M { t: forge.zeros[i64, [1]] }\n\
         \x20   go!(m)\n\
         \x20   m.t[0]\n\
         }\n",
    );
    assert_eq!(via_free_fn.run_output(), "=> 0");
    assert_eq!(via_free_fn.jit_output(), "=> 909");
}

// ─── the rework: a method shape param that shadows a model one ───────────────

/// **D1 — the shadow, and the divergence class it belongs to (#474).**
///
/// `model Buf[N] { !t: Tensor[i64,[N]]  fn f![N](self, x: Tensor[i64,[N]]) }`
/// is a legal program: the method declares a shape parameter that *shadows* the
/// model's. Two different bindings under one spelling, and `dmc run` keeps them
/// apart — the receiver's `N` types `self` and its fields, the call's bracket
/// types the method's own names, and in the body a bare `N` resolves
/// method-first. `sum(self.t) * 10 + N` on a `Buf[4]` called `f![2]` prints
/// `42` there: `self.t` is `[4]`, `N` is `2`.
///
/// One `HashMap` for both let the bracket overwrite the model's binding, and
/// then the receiver, its fields and the model half of the mangled symbol all
/// spoke about the method's value: a `Buf[4]` receiver resolved as a `Buf[2]`
/// and landed on `Buf$$N2$$f!$N2`. Measured before the fix, with `--check` OK on
/// every row: `sum(self.t)` answered `4` under `dmc run` and `2` under
/// `dmc jit`.
///
/// Row 1 is that program. Row 2 is the same shadow one axis over — the model's
/// *second* parameter shadowed, so a wrong binding cannot be masked by the two
/// dimensions happening to agree.
#[test]
fn a_shadowing_method_bracket_keeps_the_receivers_own_instantiation() {
    let one_param = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn f![N](self, x: Tensor[i64, [N]]) -> i64 { sum(self.t) as i64 }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.ones[i64, [4]] }\n\
         \x20   let x2 = forge.zeros[i64, [2]]\n\
         \x20   a.f![2](x2)\n\
         }\n",
    );
    assert_eq!(one_param.run_output(), "=> 4", "self.t is the receiver's [4]");
    assert_eq!(one_param.jit_output(), "=> 4", "not `2` — the bracket is not the receiver");

    let second_param = Tree::one(
        "model M[R, C] {\n\
         \x20   !t: Tensor[i64, [C]]\n\
         \x20   fn f![C](self, x: Tensor[i64, [C]]) -> i64 { sum(self.t) as i64 }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !a = M[2, 6] { t: forge.ones[i64, [6]] }\n\
         \x20   let x = forge.zeros[i64, [2]]\n\
         \x20   a.f![2](x)\n\
         }\n",
    );
    assert_eq!(second_param.run_output(), "=> 6");
    assert_eq!(second_param.jit_output(), "=> 6", "not `2`");
}

/// The same shadow, read as a **memory** question rather than an arithmetic
/// one. The body walks `0..N` — the method's `N`, `8` — over `self.t`, which is
/// the receiver's `[4]`. `dmc run` errors on the fifth element; the JIT has to
/// error there too, which it does only if the receiver's type carried `[4]`
/// into the bounds check. Before the fix it carried `[8]` and read four `i64`
/// past the end of the buffer, answering a number.
///
/// "Both error" is the assertion, not "both answer": this is a program neither
/// backend should complete.
#[test]
fn a_shadowing_bracket_does_not_widen_the_receivers_bounds() {
    let t = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn f![N](self, x: Tensor[i64, [N]]) -> i64 {\n\
         \x20       let mut s = 0\n\
         \x20       for i in 0..N { s = s + self.t[i] }\n\
         \x20       s\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.ones[i64, [4]] }\n\
         \x20   let z8 = forge.zeros[i64, [8]]\n\
         \x20   a.f![8](z8)\n\
         }\n",
    );
    let run = t.dmc(&["run"]);
    assert!(!run.status.success(), "dmc run must not complete: {}", t.out_of(&run));
    assert!(t.out_of(&run).contains("index 4 out of bounds for axis 0 of size 4"),
            "{}", t.out_of(&run));
    let jit = t.dmc(&["jit"]);
    assert!(!jit.status.success(), "dmc jit must not complete either: {}", t.out_of(&jit));
    assert!(t.out_of(&jit).contains("index 4 out of bounds for axis 0 of size 4"),
            "the bounds check runs against the receiver's own [4]: {}", t.out_of(&jit));
}

/// **Two receivers, one bracket, two symbols.** `Buf[4]` and `Buf[2]` both
/// called `f![2]`: the model half of the symbol follows the receiver and the
/// method half the bracket, so they are `Buf$$N4$$f!$N2` and `Buf$$N2$$f!$N2`
/// — distinct. Sharing one symbol is #474's failure with the shadow as its
/// trigger, and before the fix it was not even a wrong *answer*: the two
/// signatures disagreed and `dmc jit` died with an uncoded
/// `operands disagree: f64 vs i64`.
///
/// Answers first — `42` and `22`, the same as `dmc run` — then the mechanism,
/// through `--json --all`'s per-item names, the same way `p16` reads them.
#[test]
fn two_receivers_at_one_shadowing_bracket_get_two_symbols() {
    let t = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn f![N](self, x: Tensor[i64, [N]]) -> i64 { (sum(self.t) as i64) * 10 + N }\n\
         }\n\
         fn main() -> nil {\n\
         \x20   let !a = Buf[4] { t: forge.ones[i64, [4]] }\n\
         \x20   let !b = Buf[2] { t: forge.ones[i64, [2]] }\n\
         \x20   let x2 = forge.zeros[i64, [2]]\n\
         \x20   print(to_str(a.f![2](x2)))\n\
         \x20   print(to_str(b.f![2](x2)))\n\
         \x20   nil\n\
         }\n",
    );
    assert_eq!(t.run_output(), "42\n22");
    assert_eq!(t.jit_output(), "42\n22", "self.t follows the receiver, N the bracket");

    // The same two call sites with an unlowered construct in the shared body,
    // so each instantiation reports its own symbol.
    let observable = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn f![N](self, x: Tensor[i64, [N]], s: str) -> i64 {\n\
         \x20       s.trim()\n\
         \x20       (sum(self.t) as i64) * 10 + N\n\
         \x20   }\n\
         }\n\
         fn main() -> nil {\n\
         \x20   let !a = Buf[4] { t: forge.ones[i64, [4]] }\n\
         \x20   let !b = Buf[2] { t: forge.ones[i64, [2]] }\n\
         \x20   let x2 = forge.zeros[i64, [2]]\n\
         \x20   print(to_str(a.f![2](x2, \"p\")))\n\
         \x20   print(to_str(b.f![2](x2, \"q\")))\n\
         \x20   nil\n\
         }\n",
    );
    let items: Vec<String> = observable
        .jit_json_all()
        .iter()
        .filter(|l| l.contains("\"kind\":\"jit-ineligible\""))
        .map(|l| field_of(l, "item"))
        .collect();
    assert!(items.iter().any(|i| i == "Buf$$N4$$f!$N2"), "the [4] receiver: {items:?}");
    assert!(items.iter().any(|i| i == "Buf$$N2$$f!$N2"), "the [2] receiver: {items:?}");
}

/// The shadow's fourth face: the bracket **omitted**. The method's own `N` is
/// unspelled, so the call is the refusal `a_methods_own_bracket_lowers_when_..`
/// already pins — but with one map the model's `N` sat in it and pre-satisfied
/// the unbound check, so the call instantiated at the *receiver's* value and
/// then failed downstream on an argument type instead: `jit-conversion`,
/// "cannot convert `Tensor[i64, [2]]` to `Tensor[i64, [4]]`", which describes
/// neither what the program did wrong nor what to write. Reading the method's
/// own map alone is what puts it back on the arm that names the parameter.
#[test]
fn an_omitted_bracket_shadowing_a_model_param_refuses_with_its_own_arm() {
    let t = Tree::one(
        "model Buf[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn f![N](self, x: Tensor[i64, [N]]) -> i64 { sum(self.t) as i64 }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !a = Buf[4] { t: forge.ones[i64, [4]] }\n\
         \x20   let x2 = forge.zeros[i64, [2]]\n\
         \x20   a.f!(x2)\n\
         }\n",
    );
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(message.contains(".f!()"), "{message:?}");
    assert!(message.contains("`Buf`"), "the message names the model: {message:?}");
    assert!(message.contains("`N` is not spelled at the call"), "{message:?}");
    assert!(!message.contains("cannot convert"), "not the downstream failure: {message:?}");
}

/// **D2 — the mono suffix's separator, one namespace over from the method
/// separator's own N2.**
///
/// `merge.rs` renames a private top-level **model** to `<file-prefix>$<name>`,
/// so a private `model N4` in `A.dmc` merges to `A$N4` and its method `f` to
/// `A$N4$$f`. `model A[N]` at `N = 4` wrote the identical string under a
/// single-`$` mono suffix. Both land in `Jit::fns`, one flat name map. Measured
/// before the fix: this exact three-file program, which `dmc run` answers
/// `7008` for, was refused `jit-conversion` — "cannot convert `A` to `N4`",
/// naming a model the call site never mentions.
///
/// Under `A$$N4$$f` they cannot meet, and for the same reason `M$$method`
/// cannot meet `M$method`: every merge-produced prefix is one `$` immediately
/// followed by a bare user identifier, which contains no `$` at all, so two
/// consecutive `$` never appear anywhere the merge writes. `7008` is `7000`
/// from the generic model's method, `8` from its shape, and the private model's
/// own method underneath — so the assertion is that all three ran, not that
/// nothing crashed.
#[test]
fn a_monomorphized_method_symbol_cannot_collide_with_a_merge_renamed_model() {
    let t = Tree::new(&[
        ("e.dmc", "use \"A.dmc\"\nfn main() -> i64 { entry() }\n"),
        (
            "A.dmc",
            "use \"B.dmc\"\n\
             model N4 {\n\
             \x20   !q: i64\n\
             \x20   fn f(self) -> i64 { self.q }\n\
             }\n\
             pub fn entry() -> i64 {\n\
             \x20   let !a = A[4] { t: forge.ones[i64, [4]] }\n\
             \x20   let !n = N4 { q: 1000 }\n\
             \x20   a.f() * 1000 + n.f() + 8\n\
             }\n",
        ),
        (
            "B.dmc",
            "pub model A[N] {\n\
             \x20   !t: Tensor[i64, [N]]\n\
             \x20   fn f(self) -> i64 { N + 2 }\n\
             }\n",
        ),
    ]);
    assert_eq!(t.run_output(), "=> 7008");
    assert_eq!(t.jit_output(), "=> 7008", "a legal program, not a `cannot convert A to N4`");
}

/// **D3 — a bracket that was written but is not a dimension.**
///
/// The unbound-parameter refusal says "is not spelled at the call" and tells
/// the user to write it. For `b.blit![0..2](x)` that is advice for a mistake
/// the program did not make: the bracket *is* there. What it is not is a
/// compile-time dimension — a literal or a foldable constant — and that is
/// what the message has to say. A range is not an integer in this position on
/// either backend ("shape argument `SH` of `M.blit!` must be an integer, got
/// range" under `dmc run`), so this stays a refusal regardless of anything
/// `let`-related.
///
/// `b.blit![k](x)` with `let k = 3` used to refuse the same way (this test's
/// own doc used to predict "a local `let` literal may become foldable" and
/// close it) — #712 landed exactly that: a local `let`'s literal initializer
/// is now as foldable a shape dimension as a module constant everywhere
/// `scope_const_dims` is consulted, this call-site bracket included. See
/// `a_local_lets_own_bracket_now_folds_and_lowers_712` for that case, now a
/// gap CLOSED rather than a refusal to pin.
#[test]
fn a_bracket_that_is_not_a_compile_time_dimension_says_so() {
    let via_range = Tree::one(
        "model M {\n\
         \x20   !n: i64\n\
         \x20   fn blit![SH](self, t: Tensor[i64, [SH]]) -> i64 { SH }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !b = M { n: 1 }\n\
         \x20   let t = forge.zeros[i64, [2]]\n\
         \x20   b.blit![0..2](t)\n\
         }\n",
    );
    let (code, message) = via_range.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(message.contains("`SH` is not a compile-time dimension at the call"), "{message:?}");
    assert!(!message.contains("not spelled at the call"), "it WAS spelled: {message:?}");
}

/// #712 closed the gap `a_bracket_that_is_not_a_compile_time_dimension_says_so`
/// used to pin as a refusal: `b.blit![k](t)` with `let k = 3` now lowers on
/// both backends and agrees, because a method call's own bracket argument is
/// resolved through `resolve_shape_expr`/`scope_const_dims` — the exact table
/// #712 extended to fold a local `let`'s literal initializer, the same as a
/// module constant. No special-casing was needed in the method-bracket path
/// itself; it already deferred to the shared const-dim table.
#[test]
fn a_local_lets_own_bracket_now_folds_and_lowers_712() {
    let via_local = Tree::one(
        "model M {\n\
         \x20   !n: i64\n\
         \x20   fn blit![SH](self, t: Tensor[i64, [SH]]) -> i64 { SH }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !b = M { n: 1 }\n\
         \x20   let k = 3\n\
         \x20   let t = forge.zeros[i64, [3]]\n\
         \x20   b.blit![k](t)\n\
         }\n",
    );
    assert_eq!(via_local.run_output(), "=> 3");
    assert_eq!(via_local.jit_output(), "=> 3", "the #712 fold must reach a method's own bracket too");
}

// ─── #708: `m(x)` desugars to `m.forward(x)` (SPEC.md §3.10) ────────────────
//
// Neither slice 1 nor slice 2's design text covers the call-as-function
// spelling: `m.forward(x)` arrives as `Field`-then-`Call` and was already
// lowered by `lower_model_method_call`, but `m(x)` arrives as a bare `Ident`
// under `Call` and fell all the way through `lower_call` to the generic
// "unknown function `m`" refusal, naming the binding as though it were an
// unresolved symbol rather than a model instance. The fix is a rewrite at the
// top of `lower_call`'s callee match: an `Ident` callee whose static type is a
// model declaring `forward` is handed to `lower_model_method_call` unchanged,
// via a synthesized `<recv>.forward(..)` callee expression, so every decision
// slice 1/2 already made — non-generic vs. shape-generic instantiation, `self`
// vs. `!self`, an unresolved receiver's shape arguments — is inherited rather
// than re-implemented.

/// The issue's own probe: a non-generic model, `fn forward(self, x: i64) -> i64
/// { x * 3 }`, called `l(7)`. `dmc run` already answers `21` (the interpreter's
/// desugar predates this fix); `dmc jit` refused `jit-unknown-fn: unknown
/// function \`l\`` before it.
#[test]
fn call_desugars_to_forward_on_a_non_generic_model_708() {
    let t = Tree::one(
        "model Lin {\n\
         \x20   fn forward(self, x: i64) -> i64 { x * 3 }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let l = Lin {}\n\
         \x20   l(7)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 21");
    assert_eq!(t.jit_output(), "=> 21", "the JIT must agree with `dmc run`");
}

/// The same desugar on a SHAPE-GENERIC model (slice 2's territory): `forward`
/// reads the model's own shape parameter, and the call-as-function spelling
/// must key the instantiation on the receiver's `type_args` the same way
/// `l.forward(x)` already does — `model_ty_of_expr` answers the same
/// `(model, type_args)` pair for the bare-`Ident` callee that
/// `lower_model_method_call` already consumes for the `Field`-then-`Call`
/// shape, so this is inheritance, not new logic.
#[test]
fn call_desugars_to_forward_on_a_shape_generic_model_708() {
    let t = Tree::one(
        "model Lin[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn forward(self, x: i64) -> i64 { x * N }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !l = Lin[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   l(7)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 28");
    assert_eq!(t.jit_output(), "=> 28");
}

/// A `!` model PARAMETER binding, called as a function inside the callee: `fn
/// step(!m: M, x: i64) -> i64 { m(x) }`. `model_ty_of_expr` answers for a
/// parameter the same way it does for a local `let` — nothing about the
/// rewrite is local-binding-specific — but this is the shape the corpus's own
/// `method_call_via_mut_param` probe (p8) exists to pin one layer up, so its
/// call-as-function twin is pinned here rather than assumed.
#[test]
fn call_desugars_to_forward_through_a_mut_param_binding_708() {
    let t = Tree::one(
        "model M {\n\
         \x20   !n: i64\n\
         \x20   fn forward(self, x: i64) -> i64 { x + self.n }\n\
         }\n\
         fn step(!m: M, x: i64) -> i64 { m(x) }\n\
         fn main() -> i64 {\n\
         \x20   let !m = M { n: 5 }\n\
         \x20   step(m, 7)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 12");
    assert_eq!(t.jit_output(), "=> 12");
}

/// `forward` declared `!self`: the call-as-function spelling must mutate
/// through the same receiver an explicit `.forward(x)` would, not a copy.
/// `a * 10 + m.n` folds the return value and the caller-visible mutation into
/// one scalar, the same trick `method_mutate_scalar_field` (p2) uses one layer
/// up.
#[test]
fn call_desugars_to_a_mut_self_forward_708() {
    let t = Tree::one(
        "model M {\n\
         \x20   !n: i64\n\
         \x20   fn forward(!self, x: i64) -> i64 { self.n = self.n + x  self.n }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !m = M { n: 0 }\n\
         \x20   let a = m(7)\n\
         \x20   a * 10 + m.n\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 77");
    assert_eq!(t.jit_output(), "=> 77");
}

/// A binding that is a plain `fn`-typed value, not a model — the pre-existing
/// indirect-call path just below the new rewrite in `lower_call`. Nothing
/// about #708 touches this: `model_ty_of_expr` answers `None` for a non-model
/// `Ident`, so the rewrite never fires and the ordinary call goes through
/// unchanged.
#[test]
fn a_plain_function_call_is_unchanged_by_the_forward_rewrite_708() {
    let t = Tree::one(
        "fn plain(x: i64) -> i64 { x * 2 }\n\
         fn main() -> i64 { plain(7) }\n",
    );
    assert_eq!(t.run_output(), "=> 14");
    assert_eq!(t.jit_output(), "=> 14");
}

/// A model with NO `forward` method, called as a function: `n(7)` on `model
/// NoFwd { field1: i64 }`. `--check` is untyped for SPEC §3.10's desugar (it
/// passes this program — measured, not assumed: `dmc --check` answers "no type
/// errors" on this exact source), so both backends only discover the missing
/// `forward` at the call. `dmc run` answers a *runtime* error, `cannot call
/// struct value (no \`forward\` method found)`, already true before this fix.
/// The JIT must refuse with a located, classed refusal of its own —
/// `jit-unknown-fn`, naming the model and the missing method — rather than
/// falling through to the generic "unknown function `n`" message the bare
/// `Ident` callee would otherwise hit, which would misname a real model
/// instance as an unresolved symbol.
#[test]
fn a_model_with_no_forward_called_as_a_function_is_a_located_refusal_708() {
    let t = Tree::one(
        "model NoFwd {\n\
         \x20   field1: i64\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let n = NoFwd { field1: 1 }\n\
         \x20   n(7)\n\
         }\n",
    );
    let checked = t.dmc(&["--check"]);
    assert!(checked.status.success(), "{}", t.out_of(&checked));
    assert!(t.out_of(&checked).contains("no type errors"),
            "SPEC §3.10's desugar is untyped at --check: {}", t.out_of(&checked));

    let run = t.dmc(&["run"]);
    assert!(!run.status.success(), "dmc run must not complete: {}", t.out_of(&run));
    assert!(t.out_of(&run).contains("cannot call struct value (no `forward` method found)"),
            "{}", t.out_of(&run));

    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(message.contains("`NoFwd`"), "names the model: {message:?}");
    assert!(message.contains("no `forward` method"), "{message:?}");
    assert!(!message.contains("unknown function `n`"),
            "must not misname the model instance as an unresolved symbol: {message:?}");
}

/// A SHAPE-GENERIC model whose receiver's type carries no shape arguments
/// (`fn f(l: Lin) -> i64 { l(7) }` on `model Lin[N]`), called as a function.
/// `lower_model_method_call` already refuses the `Field`-then-`Call` spelling
/// of this exact receiver shape ("the receiver's shape arguments are not
/// known here") — #706/#709's posture, reused unchanged since the
/// call-as-function rewrite hands off to that same function. Asserted here so
/// a future change to the rewrite's guard cannot silently widen it into
/// guessing a shape argument.
#[test]
fn call_desugar_on_a_type_arg_less_generic_receiver_refuses_like_forward_does_708() {
    let t = Tree::one(
        "model Lin[N] {\n\
         \x20   !t: Tensor[i64, [N]]\n\
         \x20   fn forward(self, x: i64) -> i64 { x * N }\n\
         }\n\
         fn f(l: Lin) -> i64 { l(7) }\n\
         fn main() -> i64 {\n\
         \x20   let !l = Lin[4] { t: forge.zeros[i64, [4]] }\n\
         \x20   f(l)\n\
         }\n",
    );
    // #728: `f`'s own `l: Lin` is now a checker warning too (a type-arg-less
    // annotation of a shape-generic model) — `dmc run` prints it alongside
    // the answer, which this asserts unchanged. #728 rework (reviewer
    // defect 1): see the sibling assertion above for why the hint declares
    // the shape parameter on the function rather than "write it explicitly".
    assert_eq!(
        t.run_output(),
        "=> 28\n\
         warning at 5:9: bare-generic-model: parameter `l` of function `f` names \
         `Lin` with no argument list — `Lin` is shape-parameterized, and an empty \
         argument list is never a wildcard (SPEC.md §3.10); this warning becomes a \
         compile-time error at 0.0.6-draft\n\
         \x20 hint: declare the shape parameter on the function: `fn f[N](l: Lin[N])`",
        "the interpreter infers N from the argument",
    );
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-type-form");
    assert!(message.contains("the receiver's shape arguments are not known here"), "{message:?}");
}

// ─── PR #727 review: arity, receiver breadth, and the bracketed spelling ────

/// #734: `--check` does not type the desugared call, so `m(7, 8)` on a
/// `forward` that takes one argument passes `--check` clean — the explicit
/// `m.forward(7, 8)` spelling is a check error, but the desugared one is not
/// checked at all — and only fails at `dmc run`'s runtime arity check.
/// Before this fix the JIT reached the shared "resolve callee" arity check
/// (the same one an ordinary free-function call goes through) on the
/// SYNTHESIZED `.forward(..)` call, which raised an uncoded `jit-error`
/// naming the mangled symbol (`Lin$$forward`) — a `kind` a program should
/// never diverge into (`DIAGNOSTICS.md §5.2`: every refusal the JIT itself
/// declines to lower is `jit-ineligible` with a class). The fix catches the
/// mismatch in the desugar's own arm and reports a located, classed
/// `jit-unknown-fn` refusal naming the model, `forward`, and both arities,
/// citing #734.
#[test]
fn call_desugar_with_wrong_arity_is_a_classed_refusal_not_an_uncoded_error_734() {
    let t = Tree::one(
        "model Lin {\n\
         \x20   fn forward(self, x: i64) -> i64 { x * 3 }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let l = Lin {}\n\
         \x20   l(7, 8)\n\
         }\n",
    );
    let checked = t.dmc(&["--check"]);
    assert!(checked.status.success(), "{}", t.out_of(&checked));
    assert!(t.out_of(&checked).contains("no type errors"),
            "#734: --check does not type the desugared call's arity: {}", t.out_of(&checked));

    let run = t.dmc(&["run"]);
    assert!(!run.status.success(), "dmc run must not complete: {}", t.out_of(&run));

    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-unknown-fn", "must be a classed refusal, not an uncoded jit-error");
    assert!(message.contains("`Lin`"), "names the model: {message:?}");
    assert!(message.contains("forward"), "names the method: {message:?}");
    assert!(message.contains('2') && message.contains('3'),
            "states both the declared and the supplied arity: {message:?}");
}

/// #708 review: the interpreter's forward desugar is value-directed
/// (`interp.rs`'s callee-`Value` match, ~6189) — it dispatches on whatever
/// the callee EXPRESSION evaluated to, never on the expression's own syntax
/// — so `o.inner(7)` desugars exactly like `o.inner.forward(7)` already did.
/// Before this fix the JIT's rewrite gated on a bare `Ident` callee and this
/// field-receiver spelling fell through to a refusal that (falsely) called
/// `o` an unresolvable receiver; `model_ty_of_expr` covers a field-postfix
/// callee already (it is the same function `lower_model_method_call` uses
/// for `recv` on the `Field`-then-`Call` spelling), so dropping the `Ident`
/// gate is what makes this free.
#[test]
fn call_desugar_through_a_field_receiver_lowers_708() {
    let t = Tree::one(
        "model Inner {\n\
         \x20   fn forward(self, x: i64) -> i64 { x * 3 }\n\
         }\n\
         model Outer {\n\
         \x20   inner: Inner\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let o = Outer { inner: Inner {} }\n\
         \x20   o.inner(7)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 21");
    assert_eq!(t.jit_output(), "=> 21", "the JIT must agree with `dmc run`");
}

/// The receiver forms `model_ty_of_expr` does NOT cover, by design (its own
/// doc comment): a call result. `mk()(7)` still refuses on the JIT, same as
/// before this fix, but the fix must not have changed WHICH class or turned
/// it into a misnamed message — it is the ordinary indirect-call refusal, the
/// callee not being `Field`/`Ident`-shaped at all.
#[test]
fn call_desugar_through_a_call_result_receiver_still_refuses() {
    let t = Tree::one(
        "model M {\n\
         \x20   fn forward(self, x: i64) -> i64 { x * 3 }\n\
         }\n\
         fn mk() -> M { M {} }\n\
         fn main() -> i64 {\n\
         \x20   mk()(7)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 21", "the interpreter's desugar is value-directed");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-indirect-call");
    assert!(message.contains("indirect"), "{message:?}");
}

/// A model ARRAY element receiver, `ms[0](7)`: `model_ty_of_expr` answers
/// `None` for an `Index` receiver by design, so this still refuses — but
/// before this review it refused as `unknown function \`ms\``, because the
/// bracket was read as a generic function's explicit shape-args bracket
/// (`f[4](..)`) rather than as an array index. `ms` is not a function; the
/// message must name the array and the model, not recite a symbol that was
/// never unresolved.
#[test]
fn call_desugar_through_a_model_array_element_still_refuses_named_correctly() {
    let t = Tree::one(
        "model M {\n\
         \x20   fn forward(self, x: i64) -> i64 { x * 3 }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !ms = forge.uninit[M, [2]]\n\
         \x20   ms[0] = M {}\n\
         \x20   ms[1] = M {}\n\
         \x20   ms[0](7)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 21", "the interpreter's desugar is value-directed");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(!message.contains("unknown function `ms`"),
            "must not misname the array binding as an unresolved symbol: {message:?}");
    assert!(message.contains("[M; N]") && message.contains("array"), "{message:?}");
}

/// #735: `m[3](7)` on a model whose `forward` takes a shape parameter of its
/// own (`fn forward[K]`) — a bracketed instance call. Before this review the
/// JIT read the bracket as a generic function's explicit shape args and
/// refused `unknown function \`m\``, naming `m` as though it were an
/// unresolved symbol rather than a model instance. It is also not a spelling
/// SPEC.md §3.10 gives any meaning to: `dmc run` does not dispatch
/// `forward[3]` here either — it indexes `m` as if it had element type and
/// prints the opaque `<opaque index(...)>`, filed as #735. The fix must name
/// the model and say the bracketed call-as-function spelling is not lowered
/// — not claim the binding is unresolved.
#[test]
fn call_desugar_bracketed_instance_call_names_the_model_735() {
    let t = Tree::one(
        "model M {\n\
         \x20   fn forward[K](self, x: i64) -> i64 { x * K }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let m = M {}\n\
         \x20   m[3](7)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> <opaque index(...)>",
               "#735: the interpreter does not dispatch this spelling either");

    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-unknown-fn");
    assert!(!message.contains("unknown function `m`"),
            "must not misname the model instance as an unresolved symbol: {message:?}");
    assert!(message.contains("`M`"), "names the model: {message:?}");
    assert!(message.contains("bracket"), "{message:?}");
}

/// The best divergence this PR closes, pinned by name: a free `fn l` and a
/// model instance bound to the same name `l` in an inner scope. Measured
/// against the pre-#708 binary (`HEAD~1`): `l(7)` under the JIT fell through
/// the callee match — no rewrite existed yet — into the ordinary
/// free-function call path and silently called the WRONG `l`, answering
/// `707` (the free fn's `7 * 100 + 7`), not the model's `forward`, `21` —
/// with no refusal at all: a silent wrong answer, not a decline. `dmc run`
/// always resolved the shadow correctly because the interpreter's desugar
/// dispatches on the callee's runtime `Value`, never on a name lookup that
/// could still see the free function past its shadow.
#[test]
fn call_desugar_wins_over_a_free_fn_of_the_same_name_708() {
    let t = Tree::one(
        "model M {\n\
         \x20   fn forward(self, x: i64) -> i64 { x * 3 }\n\
         }\n\
         fn l(x: i64) -> i64 { x * 100 + 7 }\n\
         fn main() -> i64 {\n\
         \x20   let l = M {}\n\
         \x20   l(7)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 21", "the shadow's model instance, not the free fn's 107");
    assert_eq!(t.jit_output(), "=> 21", "the JIT must agree with `dmc run`, not silently call the free fn");
}

// ─── p11 / p14: the two pre-existing divergences `m(x)` also reaches ────────
//
// Both are the SAME divergences `tools/jit_probes.py`'s prose and the tests
// above already pin through the `.forward(..)`/free-fn spellings (#707,
// #636) — the call-as-function spelling reaches the identical root, not a
// new one, because `lower_call`'s rewrite hands off to the SAME
// `lower_model_method_call` / ordinary-call machinery those spellings already
// go through. Pinned here rather than added to `MUST_LOWER`, for the same
// reason the originals are not there: a divergence has no allowlist branch in
// that battery, so a name there would score a red battery on a case the
// interpreter, not the JIT, gets wrong.

/// p11 reached through `m(x)`: `let !v = b(0)` where `forward` returns
/// `self.t` (a field tensor). Same root as
/// `a_returned_tensor_diverges_and_did_so_before_methods_p11` above — a
/// RETURNED tensor does not alias under `dmc run` (`SPEC.md §3.4`'s
/// live-alias rule is a property of the binding form, not the tensor's
/// provenance) but does under `dmc jit` (same pointer handed back). Waived as
/// #707, the return-side clause of the #650 decision.
#[test]
fn call_desugar_reaches_the_returned_tensor_divergence_p11_707() {
    let t = Tree::one(
        "model B {\n\
         \x20   !t: Tensor[i64, [3]]\n\
         \x20   fn forward(self, x: i64) -> Tensor[i64, [3]] { self.t }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !b = B { t: forge.zeros[i64, [3]] }\n\
         \x20   let !v = b(0)\n\
         \x20   v[0] = 6\n\
         \x20   b.t[0]\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 0", "the interpreter copies the returned tensor");
    assert_eq!(t.jit_output(), "=> 6", "the JIT returns the same pointer — #707");
}

/// p14 reached through `m(x)`: `m(g.heat)`, a `!` tensor argument that is NOT
/// a bare identifier. Same root as
/// `a_mut_tensor_arg_that_is_not_an_identifier_still_diverges` (`jit.rs`) —
/// `interp.rs`'s `apply_writebacks` only writes back a bare `Expr::Ident`
/// argument, so `g.heat`'s write is dropped there, while the JIT's writeback
/// targets the pointer it was actually given. `forward` returns the constant
/// `1` (not the written value) so the divergence shows up ONLY in
/// `g.heat[0]`, the same isolation trick the free-fn pin uses. Tracked as
/// #636.
#[test]
fn call_desugar_reaches_the_non_ident_mut_arg_divergence_p14_636() {
    let t = Tree::one(
        "model Graph { !heat: Tensor[i64, [3]] }\n\
         model M {\n\
         \x20   fn forward(self, !t: Tensor[i64, [3]]) -> i64 { t[0] = t[0] + 9  1 }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !g = Graph { heat: forge.zeros[i64, [3]] }\n\
         \x20   let m = M {}\n\
         \x20   let a = m(g.heat)\n\
         \x20   a * 100 + g.heat[0]\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 100", "the interpreter drops the non-ident writeback — #636");
    assert_eq!(t.jit_output(), "=> 109", "the JIT writes back through the pointer");
}

// ─── #747: a bracket-less call whose method's OWN signature can't declare ───

/// #747's exact report: a non-generic model's method takes a `View[u8,
/// [128]]` param — a type `ty_from_ast`'s catch-all already refuses in ANY
/// signature (`a_free_fns_view_param_refuses_the_same_honest_way_747` pins the
/// free-fn baseline below) — and is called with no bracket at all,
/// `net.handle_rx!(pkt)`.
///
/// Before the fix, `method_fn_decl`'s synthesized decl failing to declare
/// (inside the `Item::Model` member loop) propagated out via `?` and got
/// attributed to the MODEL as a whole; `lower_model_method_call` then found
/// the method's mangled name in neither `self.fns` nor `self.templates` and
/// fell into the arm written for a *different* unreachable case — a bracket
/// on a method that declares no shape params — so a call with NO bracket at
/// all was told it had "a shape bracket it does not declare". Now the
/// method's own declare-time error is stored (`Jit::model_method_errors`) and
/// re-surfaced verbatim: the same honest `jit-type-form` refusal a free fn
/// with the identical param gets, not a fabricated one about a bracket.
#[test]
fn a_bracket_less_view_param_method_call_refuses_honestly_not_as_own_bracket_747() {
    let t = Tree::one(
        "model NetworkDriver {\n\
         \x20   !count: i64\n\
         \x20   fn handle_rx!(!self, raw_packet: View[u8, [128]]) -> nil {\n\
         \x20       self.count = self.count + 1\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !net = NetworkDriver { count: 0 }\n\
         \x20   let !buf = forge.zeros[u8, [128]]\n\
         \x20   net.handle_rx!(buf)\n\
         \x20   net.count\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 1", "dmc run accepts a View param; this is a gap, not a divergence");

    // #755 D3 rework: Pass 1 now records `handle_rx!`'s own declare-time
    // refusal ONCE, as a root at the model item — whether or not it is ever
    // called — so `--all` reports TWO sites for this program: the root, and
    // this one call site's cascade. (Before D3, the call site registered as
    // its own independent, unattributed blocker; `one_refusal_all`'s
    // "exactly one" no longer holds here.)
    let all = t.all_refusals_all();
    assert_eq!(all.len(), 2, "the model's root plus one cascade: {:#?}", all);

    let root = &all[0];
    assert_eq!(field_of(root, "code"), "jit-type-form");
    assert_eq!(field_of(root, "item"), "NetworkDriver", "the root belongs to the model, not the call site");
    assert!(!root.contains("cascade_of"), "the root itself carries no cascade_of: {root}");
    assert!(field_of(root, "message").contains("this type in a signature"), "{root:?}");

    let cascade = &all[1];
    assert_eq!(field_of(cascade, "code"), "jit-type-form", "the true reason, not `jit-unknown-fn`");
    assert_eq!(field_of(cascade, "item"), "main", "the call site, once Pass 1 no longer aborts the whole model on it");
    assert!(cascade.contains("\"cascade_of\":0"), "must cascade off the model's root: {cascade}");
    let message = field_of(cascade, "message");
    assert!(!message.contains("shape bracket"), "the call wrote no bracket: {message:?}");
    assert!(!message.contains("does not declare"), "the fabricated wording must be gone: {message:?}");
    assert!(message.contains("this type in a signature"), "the true `ty_from_ast` refusal: {message:?}");
}

/// The free-fn baseline `a_bracket_less_view_param_method_call_refuses_honestly_..`
/// above must match: a `View` param is refused the same honest way whether it
/// sits on a method or an ordinary function, because both route through the
/// same `declare_fn` → `ty_from_ast`. This is #747's OTHER open question
/// ("is a `View` param itself lowerable on a free fn?") settled: no, and the
/// class and wording are already correct there — #747 was never about this
/// message being wrong, only about a method call reaching a different, wrong
/// one instead of it.
#[test]
fn a_free_fns_view_param_refuses_the_same_honest_way_747() {
    let t = Tree::one(
        "fn handle_rx(raw_packet: View[u8, [128]]) -> i64 { raw_packet[0] as i64 }\n\
         fn main() -> i64 {\n\
         \x20   let !buf = forge.zeros[u8, [128]]\n\
         \x20   handle_rx(buf)\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 0");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-type-form");
    assert!(message.contains("this type in a signature"), "{message:?}");
}

/// The same call, `Tensor[u8, [128]]` in place of `View`: a slice-1 shape the
/// JIT already declares and lowers eagerly (#634 slice 1), so a bracket-less
/// method call over it must keep answering, not regress alongside the `View`
/// fix above.
#[test]
fn a_bracket_less_tensor_param_method_call_lowers_747() {
    let t = Tree::one(
        "model NetworkDriver {\n\
         \x20   !count: i64\n\
         \x20   fn handle_rx!(!self, raw_packet: Tensor[u8, [128]]) -> nil {\n\
         \x20       self.count = self.count + 1\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !net = NetworkDriver { count: 0 }\n\
         \x20   let !buf = forge.zeros[u8, [128]]\n\
         \x20   net.handle_rx!(buf)\n\
         \x20   net.count\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 1");
    assert_eq!(t.jit_output(), "=> 1", "the JIT must agree with `dmc run`");
}

/// The scalar control: an ordinary `i64` param, bracket-less, on a
/// non-generic model — the case slice 1 was built for and the one #747's
/// mis-fire must never touch.
#[test]
fn a_bracket_less_scalar_param_method_call_lowers_747() {
    let t = Tree::one(
        "model NetworkDriver {\n\
         \x20   !count: i64\n\
         \x20   fn handle_rx!(!self, x: i64) -> nil {\n\
         \x20       self.count = self.count + x\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !net = NetworkDriver { count: 0 }\n\
         \x20   net.handle_rx!(5)\n\
         \x20   net.count\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 5");
    assert_eq!(t.jit_output(), "=> 5");
}

/// The same `View` param, on a SHAPE-GENERIC model this time (a receiver
/// bracket in play, `NetworkDriver[64]`). This shape never went through the
/// fallback #747 patches — a shape-generic method is filed as a *template*
/// unconditionally (`method_fn_decl` gives it a non-empty `shape_params` from
/// the model's own), so the `View` param is only type-checked at
/// instantiation (`declare_template_signature`).
///
/// #811 update: before that issue's fix, instantiation raised the same
/// honest `ty_from_ast` catch-all every eager `View` param above does ("this
/// type in a signature"), which is what this test used to pin. #811 gave
/// `declare_template_signature`'s own parameter conversion (only — see
/// `view_param_ty_for_template`'s doc comment in `jit.rs`) a `View`→`Tensor`
/// rewrite, since the JIT has no runtime representation for a View distinct
/// from a contiguous `Tensor`; a shape-generic model's method is filed as a
/// template regardless of whether ITS OWN `View` param's shape mentions the
/// model's shape parameter (it doesn't, here — `[128]` is a literal, `N` is
/// unrelated), so `handle_rx!`'s `View[u8, [128]]` now lowers exactly like
/// `Tensor[u8, [128]]` does. This is the one #747/#755 fixture that touches
/// the template path; every EAGER (non-generic) `View`-param fixture in this
/// file is unaffected, because `declare_fn` never calls that rewrite.
#[test]
fn a_shape_generic_models_view_param_method_call_lowers_811() {
    let t = Tree::one(
        "model NetworkDriver[N] {\n\
         \x20   !count: i64\n\
         \x20   scratch: Tensor[u8, [N]]\n\
         \x20   fn handle_rx!(!self, raw_packet: View[u8, [128]]) -> nil {\n\
         \x20       self.count = self.count + 1\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !net = NetworkDriver[64] { count: 0, scratch: forge.zeros[u8, [64]] }\n\
         \x20   let !buf = forge.zeros[u8, [128]]\n\
         \x20   net.handle_rx!(buf)\n\
         \x20   net.count\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 1");
    assert_eq!(t.jit_output(), "=> 1", "the JIT must agree with `dmc run` (#811)");
}

/// The structural half of the fix: one method's own signature failing to
/// declare must not take its SIBLINGS down with it. Before #747's fix, the
/// `Item::Model` member loop `?`-propagated the first method's error straight
/// out of the whole model's registration, so `bump!` — an ordinary `i64`
/// method that has nothing to do with `handle_rx!`'s `View` param and is the
/// only one actually called here — was refused too, with the SAME fabricated
/// "shape bracket it does not declare" wording, naming `bump!` of all things.
///
/// #755 D3 rework: what the isolation buys is revised. An adversarial review
/// of the original fix found it went one step further than intended — a
/// broken method that is NEVER called produced NO diagnostic at all under
/// plain `dmc jit` (this test used to assert `t.jit_output() == "=> 5"`,
/// i.e. success), which is silent acceptance of a program with a method
/// that never compiles, not merely "isolated". The posture decision keeps
/// #705's eager stance instead: plain mode is all-or-nothing, same as a
/// free fn's own declare-time refusal always has been, so `handle_rx!`
/// still refuses the WHOLE PROGRAM here even though only `bump!` is ever
/// called. `--all` is where the isolation actually pays off — `bump!`
/// itself carries no refusal of its own (Pass 2 defines it cleanly), and
/// the program's only site is `handle_rx!`'s one root, uncalled and so
/// uncascaded.
#[test]
fn a_broken_sibling_method_no_longer_blocks_the_rest_of_the_model_747() {
    let t = Tree::one(
        "model NetworkDriver {\n\
         \x20   !count: i64\n\
         \x20   fn handle_rx!(!self, raw_packet: View[u8, [128]]) -> nil {\n\
         \x20       self.count = self.count + 1\n\
         \x20   }\n\
         \x20   fn bump!(!self, n: i64) -> nil {\n\
         \x20       self.count = self.count + n\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !net = NetworkDriver { count: 0 }\n\
         \x20   net.bump!(5)\n\
         \x20   net.count\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 5");

    // Plain mode: all-or-nothing. `handle_rx!`'s own signature never
    // declares — called or not — and that still costs the whole compile.
    let out = t.dmc(&["jit"]);
    assert!(!out.status.success(),
        "dmc jit must refuse: handle_rx!'s own signature never declares, called or \
         not — bump! being callable on its own is not silent acceptance (D3)");

    // `--all`: the isolation's real payoff. `bump!` registers and defines
    // fine (no site of its own); the program's one site is `handle_rx!`'s
    // root, with no cascade since nothing calls it.
    let all = t.all_refusals_all();
    assert_eq!(all.len(), 1, "handle_rx!'s one root, and nothing else: {:#?}", all);
    assert_eq!(field_of(&all[0], "item"), "NetworkDriver");
    assert_eq!(field_of(&all[0], "code"), "jit-type-form");
    assert!(!all[0].contains("cascade_of"), "never called, so no cascade: {}", all[0]);
}

// ─── #755 rework of #747: D1 (isolation must not swallow a defect), D2
// (re-surfaced errors must name the DECLARING file), D3 (declare-time
// posture — a root at the model item, cascades at every call site, plain
// mode all-or-nothing) ───────────────────────────────────────────────────

/// D3: a broken method that is NEVER called — no sibling, no call site at
/// all — must still refuse in BOTH modes. Before this rework the declare-
/// time gap #747/D1 isolates was invisible everywhere a broken method was
/// never reached: no diagnostic in `--all` (nothing ever re-surfaces it)
/// and a `dmc jit` that ran to completion despite a method sitting in the
/// model that never compiles.
#[test]
fn a_never_called_broken_method_refuses_the_program_on_both_modes_755() {
    let t = Tree::one(
        "model NetworkDriver {\n\
         \x20   !count: i64\n\
         \x20   fn handle_rx!(!self, raw_packet: View[u8, [128]]) -> nil {\n\
         \x20       self.count = self.count + 1\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !net = NetworkDriver { count: 0 }\n\
         \x20   net.count\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 0", "`handle_rx!` is never called");

    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-type-form");
    assert!(message.contains("this type in a signature"), "{message:?}");

    let all = t.all_refusals_all();
    assert_eq!(all.len(), 1, "one root, never called: {:#?}", all);
    assert_eq!(field_of(&all[0], "item"), "NetworkDriver");
    assert_eq!(field_of(&all[0], "code"), "jit-type-form");
    assert!(!all[0].contains("cascade_of"), "{}", all[0]);
}

/// D3: the same broken method called from TWO sites collects as ONE root
/// (the model item, recorded once at Pass 1) plus TWO cascades — not two
/// independent, unattributed blockers, which is what re-surfacing the
/// stored error with no cascade linkage at all used to produce.
#[test]
fn a_broken_method_called_from_two_sites_gets_one_root_two_cascades_755() {
    let t = Tree::one(
        "model NetworkDriver {\n\
         \x20   !count: i64\n\
         \x20   fn handle_rx!(!self, raw_packet: View[u8, [128]]) -> nil {\n\
         \x20       self.count = self.count + 1\n\
         \x20   }\n\
         }\n\
         fn call_a() -> nil {\n\
         \x20   let !net = NetworkDriver { count: 0 }\n\
         \x20   let !buf = forge.zeros[u8, [128]]\n\
         \x20   net.handle_rx!(buf)\n\
         }\n\
         fn call_b() -> nil {\n\
         \x20   let !net = NetworkDriver { count: 0 }\n\
         \x20   let !buf = forge.zeros[u8, [128]]\n\
         \x20   net.handle_rx!(buf)\n\
         }\n\
         fn main() -> i64 { 0 }\n",
    );
    assert_eq!(t.run_output(), "=> 0");

    let all = t.all_refusals_all();
    assert_eq!(all.len(), 3, "1 root + 2 cascades: {:#?}", all);
    assert_eq!(field_of(&all[0], "item"), "NetworkDriver");
    assert!(!all[0].contains("cascade_of"), "the root carries no cascade_of: {}", all[0]);
    for (i, item) in [(1, "call_a"), (2, "call_b")] {
        assert_eq!(field_of(&all[i], "item"), item);
        assert!(all[i].contains("\"cascade_of\":0"), "site {i} must cascade off the root: {}", all[i]);
    }
}

/// D1: the catch that isolates one method's OWN declare-time REFUSAL
/// (`JitErrorKind::Unsupported`) from its siblings (#747) must not also
/// swallow a genuine DEFECT (`JitErrorKind::Error`). Two `bump!` methods
/// with the same signature on one model collide in `declare_fn` as
/// "duplicate function `B.bump!`" — a hard compiler error, not a refusal —
/// and that must still stop the whole compile on BOTH `dmc jit` and
/// `--json --all`, the same as any other `jit-error` ever has. Before this
/// fix, the per-method catch in the `Item::Model` arm caught EVERY error
/// class, including this one, and silently recorded it into
/// `model_method_errors` instead of propagating — `status: ok` for a
/// program with two colliding methods.
#[test]
fn two_same_named_methods_on_one_model_still_hard_stop_755() {
    let t = Tree::one(
        "model B {\n\
         \x20   !n: i64\n\
         \x20   fn bump!(!self, x: i64) -> nil {\n\
         \x20       self.n = self.n + 1\n\
         \x20   }\n\
         \x20   fn bump!(!self, x: i64) -> nil {\n\
         \x20       self.n = self.n + x\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !b = B { n: 0 }\n\
         \x20   b.bump!(5)\n\
         \x20   b.n\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 5", "the interpreter dispatches by name and takes the last one");

    // `dmc jit`, no --json: must refuse, not silently `status: ok`.
    let out = t.dmc(&["jit"]);
    assert!(!out.status.success(), "two `bump!` methods must hard-stop, not vanish");

    // `--json`: a `jit-error` (defect), not `jit-ineligible` (refusal).
    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{:#?}", json);
    assert!(json[0].contains("\"kind\":\"jit-error\""), "{}", json[0]);
    assert!(json[0].contains("duplicate function"), "{}", json[0]);

    // `--json --all`: the earlier bug caught EVERY error class here,
    // including this one, and recorded it into `model_method_errors`
    // instead of propagating. Must still abort collection with the same
    // `jit-error`.
    let all = t.dmc(&["jit", "--json", "--all"]);
    let stderr = String::from_utf8_lossy(&all.stderr).to_string();
    assert!(stderr.contains("\"kind\":\"jit-error\""), "{stderr}");
    assert!(stderr.contains("duplicate function"), "{stderr}");
}

/// D2: the re-surfaced error's line/col are the DECLARATION's own (correct
/// before this fix too), but the FILE must also be the declaration's own —
/// not whichever file the CALLING body happens to live in. Before this fix,
/// `Jit::attribute` re-stamped `error_file` from the caller's
/// `current_decl_file`, so a merged, multi-file build pointed a method's
/// declare-time refusal at the wrong file entirely (here, `main.dmc:3:...`
/// for a method actually declared in `driver.dmc`).
#[test]
fn a_multi_file_methods_declare_time_refusal_names_the_declaring_file_755() {
    let t = Tree::new(&[
        ("main.dmc", "use \"driver.dmc\"\n\
                      fn main() -> i64 {\n\
                      \x20   let !net = NetworkDriver { count: 0 }\n\
                      \x20   let !buf = forge.zeros[u8, [128]]\n\
                      \x20   net.handle_rx!(buf)\n\
                      \x20   net.count\n\
                      }\n"),
        ("driver.dmc", "pub model NetworkDriver {\n\
                        \x20   !count: i64\n\
                        \x20   fn handle_rx!(!self, raw_packet: View[u8, [128]]) -> nil {\n\
                        \x20       self.count = self.count + 1\n\
                        \x20   }\n\
                        }\n"),
    ]);
    assert_eq!(t.run_output(), "=> 1");

    let json = t.jit_json();
    assert_eq!(json.len(), 1, "{:#?}", json);
    let file = field_of(&json[0], "file");
    assert!(file.ends_with("driver.dmc"), "must name the declaring file, not the caller's: {file:?}");
    assert!(json[0].contains("\"line\":3"), "handle_rx!'s own declaration line in driver.dmc: {}", json[0]);

    // `--all` must agree: the root belongs to `driver.dmc` too, and the
    // cascade at `main`'s call site still carries its OWN file (unaffected —
    // #755 only touches the re-surfaced error's file, not `main`'s).
    let all = t.all_refusals_all();
    assert_eq!(all.len(), 2, "{:#?}", all);
    let root_file = field_of(&all[0], "file");
    assert!(root_file.ends_with("driver.dmc"), "{root_file:?}");
    let cascade_file = field_of(&all[1], "file");
    assert!(cascade_file.ends_with("driver.dmc"),
        "the cascade's own file field is the DECLARATION's too, per #755 D2 — \
         only main.dmc's now-fixed misattribution is what this pins: {cascade_file:?}");
}

/// D5: #727 (`m(x)` desugaring to `m.forward(x)` in the JIT) is a separate,
/// still-open PR — not yet on `main` as of this rework, so the desugared
/// CALL form (`net(buf)`) cannot be exercised here; `dmc jit` refuses it
/// today as an ordinary unknown-function call regardless of `forward`'s own
/// signature (`lower_call` has no model-instance dispatch yet). What IS
/// exercisable now, and pinned here as the closest available regression:
/// a method literally named `forward`, called EXPLICITLY as `.forward(..)`,
/// gets the same D1–D3 treatment as any other method name — its own
/// declare-time refusal re-surfaces honestly, naming the true reason and
/// the declaring file, not a fabricated one. Once #727 lands, the desugared
/// call reaches the very same `lower_model_method_call` this pins, and this
/// test's assertions should still hold for `net(buf)` unchanged.
#[test]
fn a_forward_named_methods_own_signature_refuses_honestly_755() {
    let t = Tree::one(
        "model NetworkDriver {\n\
         \x20   !count: i64\n\
         \x20   fn forward(!self, raw_packet: View[u8, [128]]) -> nil {\n\
         \x20       self.count = self.count + 1\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !net = NetworkDriver { count: 0 }\n\
         \x20   let !buf = forge.zeros[u8, [128]]\n\
         \x20   net.forward(buf)\n\
         \x20   net.count\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 1");
    let (code, message) = t.one_refusal();
    assert_eq!(code, "jit-type-form", "the true reason, not a bracket or unknown-fn message");
    assert!(message.contains("this type in a signature"), "{message:?}");
}

/// D3 follow-up: a model with TWO `View`-param methods, neither ever called,
/// must surface TWO roots — one per broken method — not just the first in
/// member order. `first_model_method_error` used to return only the first
/// match, so `handle_tx!` never appeared anywhere in `--all`'s output while
/// its summary still claimed `truncated:false` — a completeness contract
/// violated silently, since nothing in the run actually aborted collection.
#[test]
fn two_broken_methods_on_one_model_are_two_roots_755() {
    let t = Tree::one(
        "model NetworkDriver {\n\
         \x20   !count: i64\n\
         \x20   fn handle_rx!(!self, raw_packet: View[u8, [128]]) -> nil {\n\
         \x20       self.count = self.count + 1\n\
         \x20   }\n\
         \x20   fn handle_tx!(!self, raw_packet: View[u8, [64]]) -> nil {\n\
         \x20       self.count = self.count + 1\n\
         \x20   }\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let !net = NetworkDriver { count: 0 }\n\
         \x20   net.count\n\
         }\n",
    );
    assert_eq!(t.run_output(), "=> 0", "neither broken method is ever called");

    let out = t.dmc(&["jit", "--json", "--all"]);
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let lines: Vec<&str> = stderr.lines().filter(|l| l.starts_with('{')).collect();

    let roots: Vec<&str> = lines.iter().copied()
        .filter(|l| l.contains("\"kind\":\"jit-ineligible\"") && !l.contains("cascade_of"))
        .collect();
    assert_eq!(roots.len(), 2, "one root per broken method, not just the first: {:#?}", lines);
    for root in &roots {
        assert_eq!(field_of(root, "item"), "NetworkDriver");
        assert_eq!(field_of(root, "code"), "jit-type-form");
    }

    let summary = lines.last().expect("summary line");
    assert!(summary.contains("\"kind\":\"summary\""), "{summary}");
    assert!(summary.contains("\"truncated\":false"),
        "neither refusal is a defect, so collection must not report truncated: {summary}");
}
