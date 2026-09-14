//! #674 slice 3d — `forge.enter(b) { … }` and `<ctor> in b` on the JIT
//! (the arena-budgets design's hole-table rows B3 and B5, slice 3d). Sibling of
//! `jit_budgets_3c.rs`, and for the same reason: the failure mode of a
//! mis-charged allocation is a located, process-exit runtime error, so the
//! cases that must FAIL have to run in their own process rather than inside
//! the test binary.
//!
//! What lives where, so nothing is tested twice and nothing falls between:
//!
//!   * `jit.rs`'s own unit tests read the budget table's meters directly
//!     after a run — the strongest form of "this allocation charged that
//!     budget", but only for cases that stay inside every ceiling, since an
//!     exhaustion there would take the test binary down with it.
//!   * THIS file drives the CLI, so it covers the refusals: a leaked
//!     register mis-charging until something exhausts, and a stale handle
//!     at `enter` / `in b`.
//!   * `tools/jit_probes.py` runs the same five edges against `dmc run` as
//!     well, which is the cross-backend half neither of the above can be.

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

    fn jit(&self, args: &[&str]) -> Output {
        self.jit_env(args, &[])
    }

    /// #674 slice 3d: `jit` with extra environment. The one caller is the
    /// `DMC_BUDGET_POISON=1` case below — §3.4's debug policy, which retains
    /// a reaped budget's chunks and `mprotect`s them `PROT_NONE`, so a stale
    /// read faults at the access site instead of silently reading whatever
    /// took the storage.
    fn jit_env(&self, args: &[&str], env: &[(&str, &str)]) -> Output {
        let mut cmd = Command::new(dmc_binary());
        cmd.arg("jit").arg(&self.path).args(args);
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.output().expect("spawn dmc")
    }
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

// ─── The five exit edges (B5), as refusal boundaries ──────────────────────
//
// Each source below carves a 4 KiB budget, runs the block, leaves it by ONE
// edge, and then allocates 16 KiB. 16 KiB cannot fit a 4 KiB budget, so a
// register that did not come back turns the program into a located `forge
// arena exhausted` naming "the entered budget's 4 KiB extent" — the exact
// sentence `assert_edge_pops` refuses to see.
//
// `--forge=1M` is comfortably above what the root needs, so the ONLY way
// these fail is a missed pop.

fn assert_edge_pops(src: &str, edge: &str) {
    let probe = Probe::new(src);
    let out = probe.jit(&["--forge=1M"]);
    let err = stderr(&out);
    assert!(
        !err.contains("the entered budget's 4 KiB extent"),
        "the `{edge}` edge did not pop the budget register — the allocation after \
         the block was charged to the 4 KiB budget the block entered:\n{err}",
    );
    assert!(out.status.success(), "the `{edge}` edge case failed:\n{err}");
    assert!(stdout(&out).contains("done"), "stdout: {}", stdout(&out));
}

#[test]
fn the_normal_edge_pops_the_budget_register() {
    assert_edge_pops(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4096)
    forge.enter(b) {
        let u = forge.zeros[f32, [64]]
    }
    let t = forge.zeros[f32, [4096]]
    print("done")
    nil
}
"#,
        "normal",
    );
}

#[test]
fn the_return_edge_pops_the_budget_register() {
    assert_edge_pops(
        r#"
fn f(b: Budget) -> i64 {
    forge.enter(b) {
        let u = forge.zeros[f32, [64]]
        return 7
    }
    3
}
fn main() -> nil {
    let b = forge.carve(forge.root, 4096)
    let k = f(b)
    let t = forge.zeros[f32, [4096]]
    print("done " + to_str(k))
    nil
}
"#,
        "return",
    );
}

#[test]
fn the_break_edge_pops_the_budget_register() {
    assert_edge_pops(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4096)
    for i in 0..3 {
        forge.enter(b) {
            let u = forge.zeros[f32, [64]]
            break
        }
    }
    let t = forge.zeros[f32, [4096]]
    print("done")
    nil
}
"#,
        "break",
    );
}

#[test]
fn the_continue_edge_pops_the_budget_register() {
    // Three iterations: a `continue` that failed to pop would also nest
    // three pushes deep, so this is the edge where a leak compounds.
    assert_edge_pops(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4096)
    for i in 0..3 {
        forge.enter(b) {
            let u = forge.zeros[f32, [64]]
            continue
        }
    }
    let t = forge.zeros[f32, [4096]]
    print("done")
    nil
}
"#,
        "continue",
    );
}

#[test]
fn the_query_edge_pops_the_budget_register() {
    // The edge the design names as the trap (§5 slice 3d): a `?` early
    // return leaves the extent AND the function.
    assert_edge_pops(
        r#"
fn g() -> (i64, Err) {
    (0, "boom")
}
fn f(b: Budget) -> (i64, Err) {
    forge.enter(b) {
        let u = forge.zeros[f32, [64]]
        let y = g()?
        let v = forge.zeros[f32, [64]]
    }
    (5, nil)
}
fn main() -> nil {
    let b = forge.carve(forge.root, 4096)
    let (v, e) = f(b)
    let t = forge.zeros[f32, [4096]]
    print("done")
    nil
}
"#,
        "?",
    );
}

// ─── The control: the same 16 KiB INSIDE the extent must be refused ───────

#[test]
fn the_edge_cases_are_discriminating_sixteen_kib_cannot_fit_four() {
    // Without this, every `assert_edge_pops` above would pass on a lowering
    // that never entered the budget at all. Same shapes, same flag, the
    // allocation moved inside the block: it must be refused at the entered
    // budget's own ceiling, naming that ceiling rather than `--forge`.
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4096)
    forge.enter(b) {
        let t = forge.zeros[f32, [4096]]
    }
    print("done")
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=1M"]);
    assert!(!out.status.success(), "16 KiB was admitted into a 4 KiB budget");
    let err = stderr(&out);
    assert!(
        err.contains("forge arena exhausted") && err.contains("the entered budget's 4 KiB extent"),
        "expected PRE-A at the ENTERED budget's ceiling, got:\n{err}",
    );
}

// ─── Nesting: an inner exit restores the OUTER budget, not the root ───────

#[test]
fn a_nested_enter_restores_the_outer_budget_not_the_root() {
    // The root is left with 4 KiB free after the two carves, so the 16 KiB
    // allocation between the inner block's exit and the outer block's can
    // only be served by `outer`. A pop-to-root would refuse it; a pop that
    // never happened (still `inner`, 4 KiB) would refuse it too. The slack
    // is 4 KiB rather than zero because the JIT charges a handful of small
    // incidental allocations to the root that the interpreter does not —
    // and 4 KiB is still a quarter of what the assertion needs, so nothing
    // about the discrimination is softened by it.
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let outer = forge.carve(forge.root, 262144)
    let inner = forge.carve(forge.root, 4096)
    forge.enter(outer) {
        forge.enter(inner) {
            let a = forge.zeros[f32, [64]]
        }
        let big = forge.zeros[f32, [4096]]
    }
    print("done")
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=270336"]);
    assert!(out.status.success(), "the outer budget did not serve it:\n{}", stderr(&out));
    assert!(stdout(&out).contains("done"));

    // The control that makes it discriminating: the root has only 4 KiB free
    // after the two carves, so the same 16 KiB allocation outside both
    // extents is refused.
    let control = Probe::new(
        r#"
fn main() -> nil {
    let outer = forge.carve(forge.root, 262144)
    let inner = forge.carve(forge.root, 4096)
    let big = forge.zeros[f32, [4096]]
    print("done")
    nil
}
"#,
    );
    let out = control.jit(&["--forge=270336"]);
    assert!(!out.status.success(), "the root had free capacity it should not have had");
    assert!(stderr(&out).contains("forge arena exhausted"), "{}", stderr(&out));
}

// ─── `<ctor> in b` — B3, the per-constructor override ─────────────────────

#[test]
fn alloc_in_charges_the_named_budget_not_the_entered_one() {
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let small = forge.carve(forge.root, 4096)
    let big = forge.carve(forge.root, 262144)
    forge.enter(small) {
        let t = forge.zeros[f32, [4096]] in big
    }
    print("done")
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=1M"]);
    assert!(out.status.success(), "the `in big` override did not take:\n{}", stderr(&out));
    assert!(stdout(&out).contains("done"));
}

#[test]
fn alloc_in_restores_the_entered_budget_after_the_constructor() {
    // The override is for ONE constructor (B3): the next allocation is back
    // on the entered budget, which cannot hold it.
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let small = forge.carve(forge.root, 4096)
    let big = forge.carve(forge.root, 262144)
    forge.enter(small) {
        let t = forge.zeros[f32, [4096]] in big
        let u = forge.zeros[f32, [4096]]
    }
    print("done")
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=1M"]);
    assert!(!out.status.success(), "the second constructor did not charge the entered budget");
    assert!(
        stderr(&out).contains("the entered budget's 4 KiB extent"),
        "expected PRE-A at `small`'s ceiling, got:\n{}", stderr(&out),
    );
}

// ─── `budget-stale` at the two new sites (§2.3's table) ───────────────────

#[test]
fn entering_a_reaped_budget_is_a_located_budget_stale() {
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4M)
    forge.reap(b)
    forge.enter(b) {
        let u = forge.zeros[f32, [64]]
    }
    nil
}
"#,
    );
    let out = probe.jit(&[]);
    assert!(!out.status.success(), "a reaped handle was admitted at `forge.enter`");
    let err = stderr(&out);
    assert!(err.starts_with("runtime error at "), "must be located: {err}");
    assert!(err.contains("budget-stale:"), "{err}");
}

#[test]
fn allocating_in_a_reaped_budget_is_a_located_budget_stale() {
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4M)
    forge.reap(b)
    let t = forge.zeros[f32, [4]] in b
    nil
}
"#,
    );
    let out = probe.jit(&[]);
    assert!(!out.status.success(), "a reaped handle was admitted at `in b`");
    let err = stderr(&out);
    assert!(err.starts_with("runtime error at "), "must be located: {err}");
    assert!(err.contains("budget-stale:"), "{err}");
}

// ─── `vault.*` inside an entered extent charges the root (§3.5) ────────────
//
// The arena-budgets design's ruled default:
// "`vault.<ctor>[…]` INSIDE an entered extent, no `in b`: charges the ROOT.
// The spelling says Vault; no budget was named." `dmc run` has always done
// it — Vault is a flat `vault_used` meter a budget never owns, and
// `interp_tests.rs`'s `hole_table_674_row19_…` pins the stamp half of it.
// The JIT lowers both spellings into the ONE arena (`MEMORY.md §1.1`), so
// the distinction has to be made at the emit site, and before this slice it
// was not made at all: the buffer went into `b`'s storage.
//
// These run in their OWN PROCESS and not as `jit.rs` unit tests, because the
// behaviour under test used to be a SIGSEGV — the exact reason this file
// exists. The meters-read-directly form of the same claim is
// `jit.rs::tests::a_vault_ctor_inside_enter_charges_the_root_not_the_entered_budget_674`.

/// The use-after-free itself: the vault tensor outlives `forge.reap(b)`, a
/// 64 KiB `forge.zeros` takes the reaped storage, and the program then reads
/// the tensor. Before the fix this exited 139.
const VAULT_SURVIVES_REAP: &str = r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 65536)
    let w = forge.enter(b) { vault.ones[f32, [4]] }
    forge.reap(b)
    let pad = forge.zeros[f32, [16384]]
    print("read " + to_str(w[0] + w[3]))
    nil
}
"#;

#[test]
fn a_vault_ctor_inside_enter_survives_the_reap() {
    let probe = Probe::new(VAULT_SURVIVES_REAP);
    let out = probe.jit(&["--forge=1M"]);
    assert!(
        out.status.success(),
        "the vault tensor did not survive `reap(b)` — it was placed in `b`'s storage \
         (exit {:?}):\n{}",
        out.status.code(), stderr(&out),
    );
    assert!(
        stdout(&out).contains("read 2"),
        "the vault tensor read back wrong after the reap and the 64 KiB that followed \
         it — stdout: {}", stdout(&out),
    );
}

#[test]
fn a_vault_ctor_inside_enter_survives_the_reap_under_poison() {
    // §3.4's `DMC_BUDGET_POISON=1`: a reaped budget's chunks are retained,
    // filled and `PROT_NONE`d, so a read of storage that WAS in `b` faults at
    // the access site rather than returning whatever moved in. A vault tensor
    // was never in `b`, so it must read back exactly as above — this is the
    // mode that distinguishes "routed to the root" from "got lucky".
    let probe = Probe::new(VAULT_SURVIVES_REAP);
    let out = probe.jit_env(&["--forge=1M"], &[("DMC_BUDGET_POISON", "1")]);
    assert!(
        out.status.success(),
        "under `DMC_BUDGET_POISON=1` the read faulted (exit {:?}), so the vault tensor \
         had been placed in the reaped budget's chunks:\n{}",
        out.status.code(), stderr(&out),
    );
    assert!(stdout(&out).contains("read 2"), "stdout: {}", stdout(&out));
}

#[test]
fn a_vault_ctor_inside_enter_draws_on_the_root_extent_not_the_entered_one() {
    // The metering half, as a minimal pair one token apart. `b` is 4 KiB and
    // the constructor wants 16 KiB, so one of the two ceilings has to refuse;
    // WHICH ceiling the refusal names is the whole assertion. Run at
    // `--forge=20479`, one byte under the 4 096 + 16 384 the ROOT needs, so
    // the `vault` spelling refuses too — but against the root's `--forge`
    // budget, which is what says the bytes were being taken from entry 0.
    const SRC: &str = r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4096)
    forge.enter(b) {
        @@[f32, [4096]]
    }
    nil
}
"#;
    let vault = Probe::new(&SRC.replace("@@", "vault.zeros"));
    let err = stderr(&vault.jit(&["--forge=20479"]));
    assert!(
        err.contains("--forge budget"),
        "the `vault.*` constructor's 16 KiB was not drawn from the ROOT — the refusal \
         names something else:\n{err}",
    );
    assert!(
        !err.contains("entered budget"),
        "the `vault.*` constructor was charged to the entered budget:\n{err}",
    );

    let forge = Probe::new(&SRC.replace("@@", "forge.zeros"));
    let err = stderr(&forge.jit(&["--forge=20479"]));
    assert!(
        err.contains("the entered budget's 4 KiB extent"),
        "the `forge.*` control did not charge the entered budget, so the pair above \
         is not discriminating:\n{err}",
    );
}
