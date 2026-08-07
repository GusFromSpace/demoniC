//! Spec-coverage probe runner.
//!
//! Walks `compiler/tests/spec_probes/` and runs every `.dmc` file through
//! the full `dmc` pipeline (lex + parse + check + run). Every probe must
//! exit successfully and produce "Run OK" in its output. A failing probe
//! is a regression in a spec-promised behavior — see the probe's header
//! comment for the spec section being exercised.
//!
//! Probes that don't (yet) cleanly pass are documented in
//! `spec_probes/PENDING.md` and not run here.

use std::path::PathBuf;
use std::process::Command;

/// Path to the `dmc` binary built by cargo for this crate.
/// Falls back to `target/release/dmc` if `CARGO_BIN_EXE_dmc` is unset
/// (which can happen under certain `cargo test` invocations).
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

/// Run every `.dmc` probe in `tests/spec_probes/` against the `dmc`
/// binary. Each probe must succeed (full pipeline through run).
#[test]
fn all_probes_pass() {
    let probes_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("spec_probes");

    assert!(
        probes_dir.is_dir(),
        "spec_probes directory missing: {}",
        probes_dir.display()
    );

    let binary = dmc_binary();
    assert!(
        binary.exists(),
        "dmc binary not built at {} — run `cargo build --release` first",
        binary.display()
    );

    let mut probes: Vec<PathBuf> = std::fs::read_dir(&probes_dir)
        .expect("read spec_probes dir")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("dmc"))
        .collect();
    probes.sort();

    assert!(
        !probes.is_empty(),
        "no .dmc probes found in {}",
        probes_dir.display()
    );

    let mut failures: Vec<(PathBuf, String)> = Vec::new();
    for probe in &probes {
        let output = Command::new(&binary)
            .arg(probe)
            .output()
            .unwrap_or_else(|e| panic!("failed to invoke dmc on {}: {e}", probe.display()));

        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );

        if !output.status.success() || !combined.contains("Run OK") {
            failures.push((probe.clone(), combined));
        }
    }

    if !failures.is_empty() {
        let mut msg = format!(
            "\n{} of {} spec probe(s) failed:\n",
            failures.len(),
            probes.len()
        );
        for (probe, out) in &failures {
            msg.push_str(&format!("\n  ─── {} ───\n", probe.display()));
            for line in out.lines().take(8) {
                msg.push_str(&format!("    {line}\n"));
            }
        }
        panic!("{msg}");
    }
}
