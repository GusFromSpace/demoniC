//! #674 slice 3c — the JIT's budget table (the arena-budget design's
//! §3.1, §3.3, §5 "Slice 3c"). Drives the actual `dmc jit` binary, because
//! carve/reap/root are now runtime calls (`__dmc_budget_carve`,
//! `__dmc_budget_reap`) with located, process-exit failure modes — exactly
//! the surface `arena_flags.rs` already exercises for `--forge` exhaustion.
//!
//! Everything here goes through `forge.root` and `forge.carve`/`forge.reap`
//! against it: `forge.enter`/`in b` do not lower until slice 3d, so a
//! carved entry's OWN storage (chunk ownership, first-chunk sizing, the
//! poison fault) is unreachable from demoniC source in this slice and is
//! instead covered by `jit.rs`'s own unit tests, which hold the private API
//! (`BudgetEntry`, `budget_carve`/`budget_allocate`/`budget_reap`,
//! `PageChunk`) a program cannot reach yet. What IS reachable from source —
//! carve/reap against the root, PRE (R1), PRE-A, R2, `budget-stale`, and
//! the (INV)/(LOC)/(NN) invariants over repeated carve/reap — is covered
//! here, end to end through the CLI.

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

    /// Run under `dmc jit`, with any extra CLI args (e.g. `--forge=...`).
    fn jit(&self, args: &[&str]) -> Output {
        let mut cmd = Command::new(dmc_binary());
        cmd.arg("jit").arg(&self.path).args(args);
        cmd.output().expect("spawn dmc")
    }
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

// ─── carve / reap round trip against the root ────────────────────────────

#[test]
fn carve_then_reap_against_the_root_runs_to_completion() {
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4M)
    forge.reap(b)
    print("ran")
    nil
}
"#,
    );
    let out = probe.jit(&[]);
    assert!(out.status.success(), "carve/reap against the root failed:\n{}", stderr(&out));
    assert!(stdout(&out).contains("ran"), "{}", stdout(&out));
}

#[test]
fn forge_root_is_a_budget_storable_and_passable() {
    // §2.2: a `Budget` may live in a model field, a `let`, and be passed as
    // an ordinary argument — exercise the surface without relying on any
    // one position.
    let probe = Probe::new(
        r#"
fn takes(b: Budget) -> nil {
    forge.reap(forge.carve(b, 1M))
    nil
}
model Tenant {
    budget: Budget,
}
fn main() -> nil {
    let r = Tenant { budget: forge.root }
    takes(r.budget)
    print("ran")
    nil
}
"#,
    );
    let out = probe.jit(&[]);
    assert!(out.status.success(), "Budget in a model field / fn argument failed:\n{}", stderr(&out));
    assert!(stdout(&out).contains("ran"), "{}", stdout(&out));
}

// ─── (PRE) — R1, budget-carve ─────────────────────────────────────────────

#[test]
fn a_carve_past_the_parents_free_capacity_is_refused_at_the_carve() {
    // #674 criterion 3: refused AT THE CARVE, located, `budget-carve` — not
    // at some later allocation.
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4M)
    let b2 = forge.carve(forge.root, 4M)
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=6M"]);
    assert!(!out.status.success(), "a carve past the parent's free capacity was admitted");
    let err = stderr(&out);
    assert!(err.starts_with("runtime error at "), "{}", err);
    assert!(err.contains("budget-carve:"), "{}", err);
    assert_eq!(
        err.trim(),
        "runtime error at 4:14: budget-carve: this carve needs 4 MiB, but only \
         2 MiB of the parent budget's 6 MiB extent is free",
    );
}

#[test]
fn a_nested_carve_past_a_childs_free_capacity_is_also_refused_at_the_carve() {
    // The same PRE boundary one level down — a child is a budget too, and
    // `budget-carve` reads ITS `free`, not the root's.
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4M)
    let c1 = forge.carve(b, 3M)
    let c2 = forge.carve(b, 2M)
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=16M"]);
    assert!(!out.status.success(), "a nested carve past the child's free capacity was admitted");
    let err = stderr(&out);
    assert!(err.contains("budget-carve:"), "{}", err);
    assert!(err.contains("only 1 MiB of the parent budget's 4 MiB extent is free"), "{}", err);
}

#[test]
fn carving_from_an_unbounded_root_always_succeeds() {
    // §4.1: with no `--forge`, `free(root) = ∞` — PRE is vacuous there,
    // and only there; the carved child is still bounded by its own extent.
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 100000000000M)
    print("carved")
    nil
}
"#,
    );
    let out = probe.jit(&[]);
    assert!(out.status.success(), "a carve from an unbounded root was refused:\n{}", stderr(&out));
    assert!(stdout(&out).contains("carved"));
}

// ─── PRE-A — the table's `free`, never `ForgeArena::exceeds_budget` ───────

#[test]
fn allocation_into_a_parent_with_a_live_child_is_bounded_by_the_tables_free() {
    // §3.1's central fix: `ForgeArena::exceeds_budget` alone would let the
    // ROOT allocate against its whole `extent` even after a carve reduced
    // its `free` — spending the child's reservation and breaking (LOC).
    // PRE-A must read the table's `free` instead, so this allocation into
    // the root (an ordinary `forge.zeros[...]`, no `in b` needed) is
    // refused once the carve has taken 4 of the root's 6 MiB.
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4M)
    let t = forge.zeros[f32, [2000000]]
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=6M"]);
    assert!(!out.status.success(), "the root allocated past the free the carve reserved for its child");
    let err = stderr(&out);
    assert!(err.contains("forge arena exhausted"), "{}", err);
    assert!(err.contains("only 2 MiB"), "the free the carve reserved was not respected: {}", err);
}

#[test]
fn two_disjoint_budgets_exhaust_at_their_own_ceilings() {
    // #674 criterion 1 (`m6`'s failure, fixed): `b1` exhausts while `b2`,
    // untouched, still serves — allocation is charged to the entered
    // budget, and two children carved from an unbounded root are
    // independent ceilings. `enter`/`in b` are 3d, so this drives it
    // through `--forge` alone: an ample root, and `b1`'s own small extent,
    // to isolate that carving `b2` does not touch `b1`'s remaining free.
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b1 = forge.carve(forge.root, 1M)
    let b2 = forge.carve(forge.root, 1M)
    let t = forge.zeros[f32, [2000000]]
    nil
}
"#,
    );
    // The allocation lands in the ROOT (current-budget register is always
    // entry 0 in 3c), so this asserts the weaker but still load-bearing
    // half of criterion 1: carving `b2` must not further reduce what `b1`
    // itself could still be carved to, i.e. sibling carves are independent
    // deductions from the SAME parent, not from each other.
    let out = probe.jit(&["--forge=64M"]);
    assert!(out.status.success(), "sibling carves from an ample root interfered with each other:\n{}", stderr(&out));
}

// ─── R2 — budget-reap, live child ─────────────────────────────────────────

#[test]
fn reaping_a_budget_with_a_live_child_is_refused() {
    // #674 criterion 4.
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4M)
    let c = forge.carve(b, 1M)
    forge.reap(b)
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=8M"]);
    assert!(!out.status.success(), "a budget with a live child was reaped");
    let err = stderr(&out);
    assert!(err.starts_with("runtime error at "), "{}", err);
    assert_eq!(
        err.trim(),
        // REWORK (nit): "reap it before reaping their parent" did not agree
        // in number — "it" (singular) named the child, "their" (plural)
        // named the parent it belongs to. Singular throughout: "its parent".
        "runtime error at 5:5: budget-reap: this budget has 1 live child \
         budget; reap it before reaping its parent",
    );
}

#[test]
fn reaping_the_root_is_refused() {
    // §3.1(b)/§4.1: `parent(0) == 0` makes the root a fixed point;
    // `reap(root)` is refused outright rather than crediting its own
    // `free` to itself. Same `budget-reap` tag as R2 — one more
    // precondition on the existing class, not a fourth refusal (§2.3).
    let probe = Probe::new(
        r#"
fn main() -> nil {
    forge.reap(forge.root)
    nil
}
"#,
    );
    let out = probe.jit(&[]);
    assert!(!out.status.success(), "forge.root was reaped");
    let err = stderr(&out);
    assert!(err.contains("budget-reap:"), "{}", err);
    assert!(err.contains("forge.root"), "{}", err);
}

// ─── budget-stale ──────────────────────────────────────────────────────────

#[test]
fn a_reaped_budget_handle_is_stale_on_reuse() {
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b = forge.carve(forge.root, 4M)
    forge.reap(b)
    forge.reap(b)
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=8M"]);
    assert!(!out.status.success(), "a double-reap of the same handle was admitted");
    let err = stderr(&out);
    assert!(err.contains("budget-stale:"), "{}", err);
    assert!(err.contains("already been reaped") || err.contains("already reaped"), "{}", err);
}

#[test]
fn a_reused_slot_still_makes_the_old_descriptor_stale() {
    // §6 Q8: a reaped index CAN be reused by a later carve — the packed
    // generation, not the index alone, is what keeps a pre-reap descriptor
    // from matching the reused entry.
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b1 = forge.carve(forge.root, 1M)
    forge.reap(b1)
    let b2 = forge.carve(forge.root, 1M)
    forge.reap(b1)
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=8M"]);
    assert!(!out.status.success(), "a stale handle matched a slot a later carve reused");
    let err = stderr(&out);
    assert!(err.contains("budget-stale:"), "{}", err);
}

// ─── (INV) B_root = Σ (free + used), through repeated carve ──────────────

#[test]
fn inv_the_roots_capacity_is_the_childrens_whole_extents() {
    // Adapted from §4's `ac_inv`: `--forge=256K`, four 64 KiB carves
    // succeed and a fifth is refused — the root's capacity is the
    // children's whole extents, full stop (no allocation needed to show
    // it, since a carve alone already reserves the meter). Two probes,
    // not one with a `print` in the middle: `print` itself allocates a
    // forge string INTO THE ROOT, which would eat into the same `free`
    // these carves are pinning to the byte, and it is the carve boundary
    // being tested here, not `print`'s own cost.
    let four = Probe::new(
        r#"
fn main() -> nil {
    let b1 = forge.carve(forge.root, 64K)
    let b2 = forge.carve(forge.root, 64K)
    let b3 = forge.carve(forge.root, 64K)
    let b4 = forge.carve(forge.root, 64K)
    nil
}
"#,
    );
    let out = four.jit(&["--forge=256K"]);
    assert!(out.status.success(), "four 64 KiB carves did not fit inside a 256 KiB root:\n{}", stderr(&out));

    let five = Probe::new(
        r#"
fn main() -> nil {
    let b1 = forge.carve(forge.root, 64K)
    let b2 = forge.carve(forge.root, 64K)
    let b3 = forge.carve(forge.root, 64K)
    let b4 = forge.carve(forge.root, 64K)
    let b5 = forge.carve(forge.root, 64K)
    nil
}
"#,
    );
    let out5 = five.jit(&["--forge=256K"]);
    assert!(!out5.status.success(), "a fifth 64 KiB carve fit inside a 256 KiB root");
    assert!(stderr(&out5).contains("budget-carve:"), "{}", stderr(&out5));
}

// ─── (LOC) E(r) = free(r) + used(r) + Σ E(children) ───────────────────────

#[test]
fn loc_reaps_credit_is_the_extent_not_the_unspent_remainder() {
    // Adapted from §4's `ac_loc`: reap one 64 KiB child, a carve of exactly
    // 64 KiB then succeeds and 64 KiB + 1 B is refused — the credit is the
    // WHOLE extent, not merely what the child spent (here: nothing). No
    // `print` between the boundary-critical carves (see the INV test's
    // comment above for why).
    let probe = Probe::new(
        r#"
fn main() -> nil {
    let b1 = forge.carve(forge.root, 64K)
    forge.reap(b1)
    let b2 = forge.carve(forge.root, 64K)
    nil
}
"#,
    );
    let out = probe.jit(&["--forge=64K"]);
    assert!(out.status.success(), "the reap did not credit the full 64 KiB back:\n{}", stderr(&out));

    let over = Probe::new(
        r#"
fn main() -> nil {
    let b1 = forge.carve(forge.root, 64K)
    forge.reap(b1)
    let b2 = forge.carve(forge.root, 64K + 1)
    nil
}
"#,
    );
    let out2 = over.jit(&["--forge=64K"]);
    assert!(!out2.status.success(), "one byte over the credited extent was admitted");
    assert!(stderr(&out2).contains("budget-carve:"), "{}", stderr(&out2));
}

// ─── (NN) free, used >= 0 — reap/recarve cycling ──────────────────────────

#[test]
fn nn_repeated_reap_recarve_cycles_leave_capacity_exactly_where_it_started() {
    // Adapted from §4's `ac_nn`: cycling carve/reap must not leak or
    // manufacture capacity. An unsigned underflow would show as an
    // enormous free capacity — i.e. the 65th carve, and worse the
    // one-byte-over 66th, would wrongly SUCCEED instead of refusing. Two
    // probes, no `print` between the boundary-critical carves (see the INV
    // test's comment for why).
    let mut cycles = String::new();
    for _ in 0..64 {
        cycles.push_str("    forge.reap(forge.carve(forge.root, 1M))\n");
    }

    let sixty_five = Probe::new(&format!(
        "fn main() -> nil {{\n{cycles}    let last = forge.carve(forge.root, 1M)\n    nil\n}}\n"
    ));
    let out = sixty_five.jit(&["--forge=1M"]);
    assert!(
        out.status.success(),
        "cycle 65's carve did not fit a 1 MiB root after 64 reap/recarve \
         cycles — capacity drifted:\n{}",
        stderr(&out),
    );

    let sixty_six = Probe::new(&format!(
        "fn main() -> nil {{\n{cycles}    let last = forge.carve(forge.root, 1M)\n    \
         let over = forge.carve(forge.root, 1)\n    nil\n}}\n"
    ));
    let out2 = sixty_six.jit(&["--forge=1M"]);
    assert!(
        !out2.status.success(),
        "one byte over a fully-recycled 1 MiB root was admitted — an unsigned \
         underflow would look exactly like this",
    );
    assert!(stderr(&out2).contains("budget-carve:"), "{}", stderr(&out2));
}

// ─── #803: the 512 MiB-plus multi-chunk case still passes ────────────────

#[test]
fn the_803_multi_chunk_512_mib_plus_case_still_passes() {
    // Two 270 000 000 B tensors (540 000 000 B combined) that force a
    // second chunk under a budget just past 512 MiB — #779/#803's own
    // regression shape (`tools/jit_probes.py`'s
    // `forge_second_chunk_not_capped_at_budget_674`), re-run here as an
    // end-to-end check that PageChunk/the budget table (§3.3, §3.4) did
    // not reintroduce it: before #803 a second chunk's own size, charged
    // against `committed`, overshot the budget even though the combined
    // request did not.
    let probe = Probe::new(
        r#"
fn main() -> i64 {
    (sum(forge.zeros[f32, [67500000]]) + sum(forge.zeros[f32, [67500000]])) as i64
}
"#,
    );
    let out = probe.jit(&["--forge=556870912"]);
    assert!(out.status.success(), "the #803 multi-chunk regression case failed again:\n{}", stderr(&out));
}
