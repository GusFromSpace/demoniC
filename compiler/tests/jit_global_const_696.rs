//! #696: a fn body reading a top-level `let` constant (a str constant, or a
//! numeric constant whose initializer isn't a compile-time literal) used to
//! fail with an uncoded `jit error: unknown binding X (JIT has no globals)`
//! — the issue-#480 defect class — instead of the `jit-item` refusal the
//! refusal table already reserves for a top-level non-literal `let` (the
//! #657 family; DIAGNOSTICS.md §5.2: `jit-item` is "a top-level item form").
//!
//! `collect_fns`'s `Item::Let` arm already refuses the LET ITEM ITSELF as
//! `jit-item` when its initializer isn't a compile-time literal (#268 only
//! ever admits `let NAME = <literal>` into `global_consts`). What #696 R1
//! fixes is the SEPARATE site: `Jit::lower_ident`'s fallback, hit when a
//! function BODY reads that same name and finds it in none of locals / shape
//! params / functions / `global_consts`. That fallback used to raise a bare
//! `err(...)` — an uncoded `jit-error` that aborts `--all` collection
//! (DIAGNOSTICS.md §5.5) before anything past it is walked.
//!
//! The rework's R1 also covers `pi`/`tau`/`e`/`inf`/`nan`: an *adversarial*
//! review of the first pass found these five reaching that exact same
//! fallback and being refused as `jit-item` too — but a built-in math
//! constant names no top-level item at all (DIAGNOSTICS.md §5.2's `jit-item`
//! is "a top-level item form"), and the refusal message ("the JIT has no
//! globals to bind it to") was false on a value that needs no globals at
//! all. The fix actually LOWERS them, as an `f64const` immediate, the same
//! way #268's own literal top-level constants already do — no refusal, no
//! code, just a value.
//!
//! R2 covers a body's read of an already-refused top-level `let`: it used to
//! count as an independent ROOT blocker rather than a *cascade* of the
//! `let`'s own declare-time refusal (`cascade_candidate_name` didn't
//! recognize the read-site message's shape). Fixed by giving that message
//! the same "first backtick-quoted token is the item's own name" shape the
//! recognizer already expects from `jit-unknown-fn`, and teaching the
//! recognizer this new shape.
//!
//! R3 covers the sibling WRITE site (assigning to a top-level `let !`, e.g.
//! `M = 4` for a top-level `let !M = 3`): the exact same class of gap, fixed
//! the exact same way.
//!
//! Each fixture below drives the real `dmc` binary. `dmc run` is checked
//! alongside `dmc jit` on every fixture that has a `dmc run` answer, to
//! confirm each refusal is real JIT-ineligibility (or, for the five
//! constants, a real JIT capability) and not a defect the language itself
//! has no meaning for.

use std::path::PathBuf;
use std::process::{Command, Output};

/// Path to the `dmc` binary built by cargo for this crate.
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

struct Probe {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Probe {
    fn new(src: &str) -> Probe {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("probe.dmc");
        std::fs::write(&path, src).expect("write probe");
        Probe { _dir: dir, path }
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(dmc_binary())
            .args(args)
            .arg(&self.path)
            .output()
            .expect("spawn dmc")
    }

    /// The path exactly as the diagnostics report it: `dmc` canonicalizes the
    /// input before anything else, so that is what lands in `"file"`.
    fn reported(&self) -> String {
        self.path.canonicalize().expect("canonicalize probe path")
            .to_string_lossy().to_string()
    }
}

/// The JSON Lines `--json` writes to stderr, split into lines.
fn lines(out: &Output) -> Vec<String> {
    let e = String::from_utf8_lossy(&out.stderr).to_string();
    if e.is_empty() { return Vec::new(); }
    e.trim_end_matches('\n').split('\n').map(|s| s.to_string()).collect()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

/// A str constant read from a fn body (the `examples/doom_fire.dmc:21:17`
/// `PALETTE` shape). Two `jit-item` sites: the `let` item itself (a str
/// initializer is never a compile-time literal, #268), and the read inside
/// `read_s` — the site #696 R1 fixes, now `cascade_of` the item's own root
/// (R2) instead of a second independent blocker.
#[test]
fn a_str_top_level_constant_read_in_a_body_refuses_jit_item() {
    let src = "let S = \"hi\"\n\
               \n\
               fn read_s() -> i64 {\n\
               \x20   len(S)\n\
               }\n\
               \n\
               fn main() -> nil {\n\
               \x20   print(read_s())\n\
               }\n";
    let probe = Probe::new(src);

    // The interpreter has a perfectly good meaning for this: it is not a
    // defect in the language, only a gap in the JIT's lowered subset.
    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(stdout(&run), "2\n");

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 3, "expected item refusal + read refusal + summary:\n{:#?}", ls);

    assert_eq!(
        ls[0],
        format!(
            "{{\"schema\":1,\"kind\":\"jit-ineligible\",\"severity\":\"error\",\"code\":\"jit-item\",\
             \"message\":\"top-level `let` bindings (only `let NAME = <literal>` constants are \
             supported); use `dmc run` for full semantics\",\"file\":\"{}\",\"line\":1,\"col\":1,\
             \"index\":0,\"item\":\"S\"}}",
            probe.reported(),
        ),
        "declare-time refusal for the `let S` item itself changed shape",
    );
    assert_eq!(
        ls[1],
        format!(
            "{{\"schema\":1,\"kind\":\"jit-ineligible\",\"severity\":\"error\",\"code\":\"jit-item\",\
             \"message\":\"`S` is a top-level `let`; the JIT has no globals for it; \
             use `dmc run` for full semantics\",\
             \"file\":\"{}\",\"line\":4,\"col\":9,\"index\":1,\"item\":\"read_s\",\"cascade_of\":0}}",
            probe.reported(),
        ),
        "the #696 R1 site (reading `S` from `read_s`'s body), now R2's cascade of index 0",
    );
    assert_eq!(
        ls[2],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"failed\",\
         \"errors\":2,\"warnings\":0,\"refusals\":2,\"blockers\":1,\"cascades\":1,\
         \"classes\":{\"jit-item\":1},\"truncated\":false,\"exit\":1}",
        "must not carry a jit-error, must not truncate collection, and (R2) the read is a \
         cascade, not a second root — 1 blocker, not 2",
    );
}

/// #696 R2: a top-level `let` read from TWO independent bodies. Before R2,
/// `cascade_candidate_name` didn't recognize the read-site message's shape,
/// so both reads counted as independent root blockers (2 blockers, 0
/// cascades) instead of one root (the declare-time refusal) and two cascades
/// of it.
#[test]
fn a_top_level_constant_read_from_two_bodies_is_one_blocker_and_two_cascades() {
    let src = "let S = \"hi\"\n\
               \n\
               fn read_s_a() -> i64 {\n\
               \x20   len(S)\n\
               }\n\
               \n\
               fn read_s_b() -> i64 {\n\
               \x20   len(S)\n\
               }\n\
               \n\
               fn main() -> nil {\n\
               \x20   print(read_s_a() + read_s_b())\n\
               }\n";
    let probe = Probe::new(src);

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 4, "expected item refusal + 2 read refusals + summary:\n{:#?}", ls);

    assert_eq!(ls[0].contains("\"item\":\"S\""), true, "root: {}", ls[0]);
    assert!(!ls[0].contains("cascade_of"), "the root site itself carries no cascade_of: {}", ls[0]);
    for (i, item) in [(1, "read_s_a"), (2, "read_s_b")] {
        assert!(ls[i].contains(&format!("\"item\":\"{}\"", item)), "site {}: {}", i, ls[i]);
        assert!(ls[i].contains("\"cascade_of\":0"), "site {} must cascade off the root: {}", i, ls[i]);
    }
    assert_eq!(
        ls[3],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"failed\",\
         \"errors\":3,\"warnings\":0,\"refusals\":3,\"blockers\":1,\"cascades\":2,\
         \"classes\":{\"jit-item\":1},\"truncated\":false,\"exit\":1}",
        "one root, two cascades — not three independent blockers",
    );
}

/// #696 R3: the sibling WRITE site. A top-level `let !M = 3`, mutated in a
/// body (`M = 4`), used to reach `lower_assign`'s unknown-binding fallback
/// and raise the uncoded `jit error: assignment to unknown binding M` —
/// truncating `--all` collection, same as R1's read-site defect did before
/// its fix. Reclassified the same way: `M` names a top-level `let`, so the
/// gap is `jit-item`, not an untriaged defect — and (same mechanism as R2)
/// the write is a cascade of `M`'s own declare-time root.
#[test]
fn a_top_level_mutable_let_written_in_a_body_refuses_jit_item() {
    let src = "let !M = 3\n\
               \n\
               fn bump() -> nil {\n\
               \x20   M = 4\n\
               }\n\
               \n\
               fn main() -> nil {\n\
               \x20   bump()\n\
               \x20   print(1)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(stdout(&run), "1\n");

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 3, "expected item refusal + write refusal + summary:\n{:#?}", ls);

    assert_eq!(
        ls[0],
        format!(
            "{{\"schema\":1,\"kind\":\"jit-ineligible\",\"severity\":\"error\",\"code\":\"jit-item\",\
             \"message\":\"top-level `let` bindings (only `let NAME = <literal>` constants are \
             supported); use `dmc run` for full semantics\",\"file\":\"{}\",\"line\":1,\"col\":1,\
             \"index\":0,\"item\":\"M\"}}",
            probe.reported(),
        ),
        "declare-time refusal for the `let !M` item itself",
    );
    assert_eq!(
        ls[1],
        format!(
            "{{\"schema\":1,\"kind\":\"jit-ineligible\",\"severity\":\"error\",\"code\":\"jit-item\",\
             \"message\":\"`M` is a top-level `let`; the JIT has no globals for it; \
             use `dmc run` for full semantics\",\
             \"file\":\"{}\",\"line\":4,\"col\":5,\"index\":1,\"item\":\"bump\",\"cascade_of\":0}}",
            probe.reported(),
        ),
        "the #696 R3 site: writing `M` from `bump`'s body, a cascade of index 0",
    );
    assert_eq!(
        ls[2],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"failed\",\
         \"errors\":2,\"warnings\":0,\"refusals\":2,\"blockers\":1,\"cascades\":1,\
         \"classes\":{\"jit-item\":1},\"truncated\":false,\"exit\":1}",
        "no jit-error, no truncation, one root and one cascade",
    );
}

/// #696 R1: `pi`/`tau`/`e`/`inf`/`nan` are compile-time literals, exactly
/// like any other numeric constant (#268), and now LOWER as an `f64const`
/// immediate instead of refusing — a `jit-item` refusal named no top-level
/// item at all, and its message ("the JIT has no globals to bind it to") was
/// false on a value that needs no globals. `dmc jit` must now succeed on all
/// five, with stdout byte-identical to `dmc run` (the `cosine_schedule.dmc:
/// 8:37` / `fft_radix2.dmc:38:28` `pi` shape).
#[test]
fn the_five_math_constants_lower_and_match_dmc_run_byte_for_byte() {
    let src = "fn f_pi() -> f64 { pi }\n\
               fn f_tau() -> f64 { tau }\n\
               fn f_e() -> f64 { e }\n\
               fn f_inf() -> f64 { inf }\n\
               fn f_nan() -> f64 { nan }\n\
               \n\
               fn main() -> nil {\n\
               \x20   print(f_pi())\n\
               \x20   print(f_tau())\n\
               \x20   print(f_e())\n\
               \x20   print(f_inf())\n\
               \x20   print(f_nan())\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    let run_out = stdout(&run);
    // Pin the exact values: std::f64::consts::{PI,TAU,E} and
    // f64::{INFINITY,NAN}, printed as interp.rs/jit.rs render them.
    assert_eq!(
        run_out,
        "3.141592653589793\n6.283185307179586\n2.718281828459045\ninf\nNaN\n",
        "dmc run's own constant table changed",
    );

    let jit = probe.run(&["jit"]);
    assert!(
        jit.status.success(),
        "dmc jit must now lower all five constants instead of refusing: {}",
        String::from_utf8_lossy(&jit.stderr),
    );
    assert_eq!(stdout(&jit), run_out, "jit and run must agree byte-for-byte on all five constants");
}

/// #696 R1, `nan` specifically: `nan != nan` is the one universal,
/// backend-independent check for "this really is IEEE NaN", not merely a
/// value that PRINTS as `NaN` — belt and suspenders alongside the previous
/// test's printed-form pin.
#[test]
fn nan_is_unequal_to_itself_on_both_backends() {
    let src = "fn main() -> nil {\n\
               \x20   print(nan != nan)\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(stdout(&run), "true\n");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit.stderr));
    assert_eq!(stdout(&jit), "true\n");
}

/// Precedence regression: `lower_ident` checks `self.locals` (a body-local
/// `let`) long before it ever reaches `global_consts`, and checks
/// `global_consts` (a top-level `let`) one line above the #696 built-in
/// math-constant arm added at jit.rs ~8981 (`"pi" => Some(std::f64::consts::PI), ...`).
/// Both orderings mean a user binding named `pi` must always beat the
/// built-in — the reverse would silently shadow real code with the
/// constant. Pins both levels so a future reorder of that arm (or of the
/// `self.locals` check above it) breaks a test here instead of silently
/// changing which `pi` a program gets.
#[test]
fn a_body_local_let_beats_the_built_in_math_constant() {
    let src = "fn f() -> i64 {\n\
               \x20   let pi = 3\n\
               \x20   pi\n\
               }\n\
               \n\
               fn main() -> nil {\n\
               \x20   print(f())\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    let run_out = stdout(&run);
    assert_eq!(run_out, "3\n",
        "a body-local `pi` must shadow the built-in math constant, not the reverse");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit.stderr));
    assert_eq!(stdout(&jit), run_out,
        "jit and run must agree: `lower_ident` checks `self.locals` before it ever reaches \
         the #696 built-in-constant arm");
}

/// Precedence regression, the other level: a top-level `let pi` (landing in
/// `global_consts`) must also beat the built-in constant — `lower_ident`
/// checks `global_consts` one line above the #696 built-in math-constant arm
/// (jit.rs ~8981), and a future reorder of those two checks would silently
/// swap which `pi` every caller of `f` below gets.
#[test]
fn a_top_level_let_beats_the_built_in_math_constant() {
    let src = "let pi = 3.0\n\
               \n\
               fn f() -> f64 {\n\
               \x20   pi\n\
               }\n\
               \n\
               fn main() -> nil {\n\
               \x20   print(f())\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    let run_out = stdout(&run);
    assert_eq!(run_out, "3\n",
        "a top-level `let pi` must shadow the built-in math constant, not the reverse");

    let jit = probe.run(&["jit"]);
    assert!(jit.status.success(), "dmc jit failed: {}", String::from_utf8_lossy(&jit.stderr));
    assert_eq!(stdout(&jit), run_out,
        "jit and run must agree: `lower_ident` checks `global_consts` one line above the \
         #696 built-in-constant arm (jit.rs ~8981) — a future reorder would silently break \
         this");
}

/// A numeric top-level constant whose initializer is a computed expression
/// rather than a literal (the `reel_to_reel.dmc:259:33` `HP_A =
/// exp(-6.28318530718 * 20.0 / SAMPLE_RATE)` shape): still not admitted to
/// `global_consts` by #268's `is_const_literal_expr`, so it takes the same
/// two-site shape the str fixture does — and (R2) the read is a cascade too.
#[test]
fn a_non_literal_numeric_top_level_constant_read_in_a_body_refuses_jit_item() {
    let src = "let A = 3\n\
               let B = exp(A as f64)\n\
               \n\
               fn read_b() -> f64 {\n\
               \x20   B\n\
               }\n\
               \n\
               fn main() -> nil {\n\
               \x20   print(read_b())\n\
               }\n";
    let probe = Probe::new(src);

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(stdout(&run), "20.085536923187668\n");

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 3, "expected item refusal + read refusal + summary:\n{:#?}", ls);

    assert_eq!(
        ls[0],
        format!(
            "{{\"schema\":1,\"kind\":\"jit-ineligible\",\"severity\":\"error\",\"code\":\"jit-item\",\
             \"message\":\"top-level `let` bindings (only `let NAME = <literal>` constants are \
             supported); use `dmc run` for full semantics\",\"file\":\"{}\",\"line\":2,\"col\":1,\
             \"index\":0,\"item\":\"B\"}}",
            probe.reported(),
        ),
        "declare-time refusal for the `let B` item itself (A, a literal, is unaffected)",
    );
    assert_eq!(
        ls[1],
        format!(
            "{{\"schema\":1,\"kind\":\"jit-ineligible\",\"severity\":\"error\",\"code\":\"jit-item\",\
             \"message\":\"`B` is a top-level `let`; the JIT has no globals for it; \
             use `dmc run` for full semantics\",\
             \"file\":\"{}\",\"line\":5,\"col\":5,\"index\":1,\"item\":\"read_b\",\"cascade_of\":0}}",
            probe.reported(),
        ),
        "the #696 R1 site (reading `B` from `read_b`'s body), now R2's cascade of index 0",
    );
    assert_eq!(
        ls[2],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"failed\",\
         \"errors\":2,\"warnings\":0,\"refusals\":2,\"blockers\":1,\"cascades\":1,\
         \"classes\":{\"jit-item\":1},\"truncated\":false,\"exit\":1}",
    );
}

/// Must-not-regress: a plain literal top-level constant (#268's own case) is
/// still lowered as an immediate, not refused — #696 only touches the
/// fall-through past `global_consts`, never the lookup itself.
#[test]
fn a_literal_top_level_constant_read_in_a_body_still_lowers() {
    let src = "let K = 3\n\
               \n\
               fn f() -> i64 { K }\n\
               \n\
               fn main() -> nil { print(f()) }\n";
    let probe = Probe::new(src);

    let out = probe.run(&["jit"]);
    assert!(out.status.success(), "jit failed: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(stdout(&out), "3\n");
}

/// Must-not-regress, the other direction: a name that reaches the SAME
/// `lower_ident` fallback for a completely unrelated reason — here,
/// `forge.identity[...]` has no lowering, and its indexing chain decomposes
/// down to lowering the bare ident `forge` — must keep raising the plain
/// uncoded `jit-error` it always has, not be swept into `jit-item` by #696's
/// fix. `forge` backs no top-level `let` and is none of the five math
/// constants, so `top_level_let_names` and the math-constant match both miss
/// it, exactly as they should — the discovery that motivated adding
/// `top_level_let_names` in the first place rather than reclassifying every
/// name this fallback ever sees (found via `examples/translations/\
/// rls_ridge.dmc`'s own `forge.identity[f32, 3]`, same shape, real corpus
/// site).
#[test]
fn an_unrelated_name_reaching_the_same_fallback_still_raises_the_uncoded_jit_error() {
    let src = "fn f() -> Tensor[f32, [3, 3]] {\n\
               \x20   forge.identity[f32, 3]\n\
               }\n\
               \n\
               fn main() -> nil {\n\
               \x20   let t = f()\n\
               \x20   print(t[0, 0])\n\
               }\n";
    let probe = Probe::new(src);

    let checked = probe.run(&["--check"]);
    assert!(checked.status.success(), "fixture does not type-check: {}",
        String::from_utf8_lossy(&checked.stderr));

    let run = probe.run(&["run"]);
    assert!(run.status.success(), "dmc run failed: {}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(stdout(&run), "1\n");

    let out = probe.run(&["jit", "--json", "--all"]);
    assert_eq!(out.status.code(), Some(1));
    let ls = lines(&out);
    assert_eq!(ls.len(), 2, "expected one jit-error + summary:\n{:#?}", ls);

    assert_eq!(
        ls[0],
        format!(
            "{{\"schema\":1,\"kind\":\"jit-error\",\"severity\":\"error\",\
             \"message\":\"unknown binding `forge` (JIT has no globals)\",\
             \"file\":\"{}\",\"line\":2,\"col\":5,\"index\":0,\"item\":\"f\"}}",
            probe.reported(),
        ),
        "an unrelated defect must not be renamed jit-item",
    );
    assert_eq!(
        ls[1],
        "{\"schema\":1,\"kind\":\"summary\",\"command\":\"jit\",\"status\":\"failed\",\
         \"errors\":1,\"warnings\":0,\"refusals\":1,\"blockers\":1,\"cascades\":0,\
         \"classes\":{\"jit-error\":1},\"truncated\":true,\"exit\":1}",
        "still truncates: a root jit-error is a genuine, untriaged defect",
    );
}
