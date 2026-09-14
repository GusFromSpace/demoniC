//! #640 — a flat `let` of an arena constructor lands the tensor ONCE.
//!
//! `forge.ones[f32, [4096]]` parses as `Postfix{Index}` over the `forge.ones`
//! receiver, so `jit.rs:is_aliasing_place_expr` used to call a FRESH
//! allocation a place expression and `let t = forge.ones[…]` snapshotted it:
//! the tensor was materialized twice, and the flat form needed double the
//! `--forge` of the same allocation written `let t = forge { forge.ones[…] }`
//! (the arena-budgets design's rule that a fresh allocation is not a place
//! expression). The predicate is now gated on `ast::is_arena_ctor_spine`.
//!
//! These tests drive the real `dmc` binary, because the property is a
//! *bisected budget ceiling* on a command line: the minimum `--forge` at
//! which a program completes. The second half of the file is the other side
//! of the same classification — a real place expression (a row view) must
//! still snapshot, and is still charged for it.

use std::path::PathBuf;
use std::process::Command;

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

/// One `[4096]` f32 tensor's declared element bytes — the unit every figure
/// in this file is a multiple of.
const T: u64 = 4096 * 4;

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

    fn runs_under(&self, mode: &str, forge: u64) -> bool {
        Command::new(dmc_binary())
            .arg(mode)
            .arg(&self.path)
            .arg(format!("--forge={}", forge))
            .output()
            .expect("spawn dmc")
            .status
            .success()
    }

    fn stdout(&self, mode: &str) -> String {
        let out = Command::new(dmc_binary())
            .arg(mode)
            .arg(&self.path)
            .output()
            .expect("spawn dmc");
        assert!(out.status.success(), "`dmc {}` failed:\n{}",
                mode, String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).to_string()
    }
}

/// The smallest `--forge` under which `src` runs to completion on `mode`,
/// by doubling then binary search — the same procedure the arena-budgets
/// design's own probe suite uses by hand. `None` if the program
/// does not complete even at the 16 MiB cap (it should never get there: the
/// programs here allocate at most 48 KiB).
fn min_forge(mode: &str, src: &str) -> Option<u64> {
    let probe = Probe::new(src);
    let mut hi = 1024u64;
    while hi <= 16 * 1024 * 1024 {
        if probe.runs_under(mode, hi) { break; }
        hi *= 2;
    }
    if hi > 16 * 1024 * 1024 { return None; }
    let mut lo = hi / 2;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if probe.runs_under(mode, mid) { hi = mid; } else { lo = mid; }
    }
    Some(hi)
}

const FLAT: &str = r#"
fn main() -> nil {
    let t = forge.ones[f32, [4096]]
    print(to_str(t[0]))
    nil
}
"#;

const BLOCK: &str = r#"
fn main() -> nil {
    let t = forge { forge.ones[f32, [4096]] }
    print(to_str(t[0]))
    nil
}
"#;

const MUT_FLAT: &str = r#"
fn main() -> nil {
    let !t = forge.ones[f32, [4096]]
    print(to_str(t[0]))
    nil
}
"#;

const MODEL_LIT: &str = r#"
model M { w: Tensor[f32, [4096]] }
fn main() -> nil {
    let m = M { w: forge.ones[f32, [4096]] }
    print(to_str(m.w[0]))
    nil
}
"#;

/// The headline. Before #640 the flat form bisected to 32 832 B against the
/// block form's 16 448 B — one whole extra tensor. They must now agree, and
/// both must sit under two tensors' worth of budget.
#[test]
fn a_flat_let_of_a_constructor_needs_the_same_forge_as_the_block_form() {
    let flat = min_forge("jit", FLAT).expect("flat let never completed");
    let block = min_forge("jit", BLOCK).expect("block form never completed");
    assert_eq!(
        flat, block,
        "`let t = forge.ones[…]` bisects to {} B but the same allocation \
         through `forge {{ … }}` bisects to {} B — the flat form is landing \
         the tensor more than once again (#640)",
        flat, block,
    );
    assert!(
        flat < 2 * T,
        "the flat `let` needs {} B for one {} B tensor — at or above two \
         tensors' worth, which is #640's double materialization (#640)",
        flat, T,
    );
}

/// The `!` spelling shares the `let` lowering's classification, so it shared
/// the bug: `let !t = ctor` owns the fresh allocation outright and has
/// nothing to snapshot it from.
#[test]
fn a_mutating_flat_let_of_a_constructor_lands_the_tensor_once() {
    let mutf = min_forge("jit", MUT_FLAT).expect("`let !t = ctor` never completed");
    let block = min_forge("jit", BLOCK).expect("block form never completed");
    assert_eq!(
        mutf, block,
        "`let !t = forge.ones[…]` bisects to {} B against the block form's \
         {} B (#640)",
        mutf, block,
    );
}

/// `lower_struct_lit` reads the same predicate, so a model literal whose
/// field is initialized straight from a constructor was double-charged too.
#[test]
fn a_model_literal_field_from_a_constructor_lands_the_tensor_once() {
    let lit = min_forge("jit", MODEL_LIT).expect("model literal never completed");
    assert!(
        lit < 2 * T,
        "`M {{ w: forge.ones[…] }}` needs {} B for one {} B tensor — the \
         constructor is still being snapshotted into the field (#640)",
        lit, T,
    );
}

/// The interpreter is the reference and #640 did not touch it: both `let`
/// forms charge the constructor once there, before and after.
#[test]
fn the_interpreter_charges_the_constructor_once_either_way() {
    let flat = min_forge("run", FLAT).expect("flat let never completed");
    let block = min_forge("run", BLOCK).expect("block form never completed");
    assert_eq!(flat, block, "`dmc run`'s own charging moved (it must not)");
    assert!(flat < 2 * T, "`dmc run` needs {} B for one {} B tensor", flat, T);
}

// ── the other side of the classification: a real place expression ─────────

const ROW_VIEW: &str = r#"
fn main() -> nil {
    let !g = forge.ones[f32, [2, 4096]]
    let !r = g[0]
    r[0] = 9.0
    print(to_str(g[0, 0]))
    nil
}
"#;

/// A row view is a genuine alias of the parent's storage, so `let r = g[0]`
/// must still snapshot — the narrowing in `is_aliasing_place_expr` exempts
/// the arena-constructor spine and nothing else. Charged: 2 × T for `g` plus
/// T for the copy.
#[test]
fn an_indexed_place_is_still_snapshotted_and_still_charged() {
    let rows = min_forge("jit", ROW_VIEW).expect("row view never completed");
    assert!(
        rows >= 3 * T,
        "`let !r = g[0]` over a [2, 4096] tensor bisects to {} B — under the \
         3 × {} B the parent plus its snapshot cost, so the row view is no \
         longer being copied (#249)",
        rows, T,
    );
}

/// And the copy is load-bearing, not just charged: writing through the bound
/// row must not reach the grid, identically on both backends.
#[test]
fn a_write_through_a_bound_row_does_not_reach_the_grid() {
    let probe = Probe::new(ROW_VIEW);
    let run = probe.stdout("run");
    let jit = probe.stdout("jit");
    assert_eq!(run.trim(), "1", "`dmc run` saw the write through the row view");
    assert_eq!(jit.trim(), run.trim(), "run={:?} jit={:?}", run, jit);
}

/// A plain rebind of an existing binding is still a value copy on the JIT —
/// #640 narrowed the constructor case only. By the arena-budgets design's
/// own rule, the interpreter makes this copy too, on the Rust heap where no
/// `--forge` meter sees it, so eliding it here would make a rebind a live
/// alias and diverge from `dmc run`.
#[test]
fn a_plain_rebind_is_still_a_value_copy() {
    let src = r#"
fn main() -> nil {
    let !x = forge.ones[f32, [4]]
    let y = x
    x[0] = 5.0
    print(to_str(y[0]))
    nil
}
"#;
    let probe = Probe::new(src);
    let run = probe.stdout("run");
    let jit = probe.stdout("jit");
    assert_eq!(run.trim(), "1", "`dmc run` made the rebind an alias");
    assert_eq!(jit.trim(), run.trim(), "run={:?} jit={:?}", run, jit);
}
