/// Usage-banner tests — the banner and the argument parser must agree.
///
/// `dmc` with no arguments is the only in-binary description of the CLI, and
/// it had drifted: `--gpu` and `--blas` were parsed in `real_main`, wired into
/// the JIT and named in the README, while the banner listed neither. The fix
/// is not a longer banner alone — it is a test, per #534: a rule whose
/// behavior was never demonstrated is not a rule.
///
/// So these tests harvest the accepted flags from the parser's OWN source
/// rather than from a hand-kept list. A new `else if a == "--foo"` branch, or
/// a new `"--foo" =>` match arm in a subcommand's option loop, fails here
/// until `USAGE_LINES` names it — and, in the other direction, the banner
/// cannot advertise a flag nothing accepts.

use super::USAGE_LINES;

const MAIN_SRC: &str = include_str!("main.rs");
/// `--vault` / `--forge` are recognized by `arena::flag_arena`, not by the
/// pre-pass in `real_main`, so the scan reads that file too.
const ARENA_SRC: &str = include_str!("arena.rs");

/// Every `--flag` string literal the source *dispatches on*: either compared
/// (`a == "--gpu"`) or used as a match arm (`"--seed" => …`). A literal in any
/// other position — a message, a `val(args, &mut i, "--seed")` label — is not
/// a flag the parser accepts, and is skipped.
fn accepted_flags(src: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while let Some(rel) = src[i..].find("\"--") {
        let open = i + rel;
        let Some(len) = src[open + 1..].find('"') else { break };
        let flag = &src[open + 1..open + 1 + len];
        let after = open + 2 + len;
        i = after;
        // Flag-shaped names only: `"--"` itself, and any literal carrying
        // punctuation or a format placeholder, is not a flag name.
        if flag.len() < 3 || !flag[2..].chars().all(|c| c.is_ascii_lowercase() || c == '-') {
            continue;
        }
        let compared = src[..open].trim_end().ends_with("==");
        let arm = src[after..].trim_start().starts_with("=>");
        if compared || arm {
            out.push(flag.to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Does the banner name `flag` as a whole token? A bare `contains` would let
/// `--check` stand in for a `--check-all`, so the character on each side must
/// not extend the name.
fn banner_names(flag: &str) -> bool {
    USAGE_LINES.iter().any(|line| {
        line.match_indices(flag).any(|(at, _)| {
            let before = line[..at].chars().next_back();
            let after = line[at + flag.len()..].chars().next();
            !matches!(before, Some(c) if c.is_ascii_alphanumeric() || c == '-')
                && !matches!(after, Some(c) if c.is_ascii_alphanumeric() || c == '-')
        })
    })
}

/// Every `--flag` the banner mentions, tokenized out of its prose (`--vault=16G`
/// yields `--vault`; backticks and trailing punctuation are not part of a name).
fn flags_named_in_banner() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in USAGE_LINES {
        let mut i = 0;
        while let Some(rel) = line[i..].find("--") {
            let at = i + rel;
            let name: String = line[at + 2..]
                .chars()
                .take_while(|c| c.is_ascii_lowercase() || *c == '-')
                .collect();
            i = at + 2 + name.len().max(1);
            if !name.is_empty() {
                out.push(format!("--{}", name));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Flags the banner names that are deliberately *not* `dmc` flags: cargo's
/// `--features gpu`, which the `--gpu` entry cites as build advice.
const FOREIGN_FLAGS: &[&str] = &["--features"];

fn all_accepted_flags() -> Vec<String> {
    let mut flags = accepted_flags(MAIN_SRC);
    flags.extend(accepted_flags(ARENA_SRC));
    flags.sort();
    flags.dedup();
    flags
}

/// The scan is the load-bearing half of the drift test: if it silently matched
/// nothing, every other assertion here would pass vacuously.
#[test]
fn the_flag_scan_actually_finds_the_parser_flags() {
    let flags = all_accepted_flags();
    for anchor in ["--blas", "--check", "--demon", "--gpu", "--jit", "--json", "--vault"] {
        assert!(flags.contains(&anchor.to_string()), "scan missed {} in {:?}", anchor, flags);
    }
    assert!(flags.len() >= 20, "scan found only {:?}", flags);
    // A literal that is neither compared nor a match arm is not a flag.
    assert_eq!(accepted_flags(r#"eprintln!("try --wombat");"#), Vec::<String>::new());
    assert_eq!(accepted_flags(r#"if a == "--wombat" { x = true; }"#), vec!["--wombat"]);
    assert_eq!(accepted_flags(r#""--wombat" => x = true,"#), vec!["--wombat"]);
}

#[test]
fn banner_names_every_flag_the_parser_accepts() {
    let missing: Vec<String> = all_accepted_flags()
        .into_iter()
        .filter(|f| !banner_names(f))
        .collect();
    assert!(
        missing.is_empty(),
        "`dmc` with no arguments accepts these flags but never mentions them: {:?} \
         — add them to USAGE_LINES",
        missing,
    );
}

#[test]
fn banner_names_no_flag_the_parser_rejects() {
    let accepted = all_accepted_flags();
    let invented: Vec<String> = flags_named_in_banner()
        .into_iter()
        .filter(|f| !accepted.contains(f) && !FOREIGN_FLAGS.contains(&f.as_str()))
        .collect();
    assert!(invented.is_empty(), "usage banner advertises unparsed flags: {:?}", invented);
}

/// The two entries this test file exists for carry the caveats that decide
/// whether a caller can rely on them: both are `dmc jit`-only, `--gpu` needs a
/// macOS `--features gpu` build, and `--blas` is opt-in, inert without
/// `cblas_sgemm`, and never chosen inside `@deterministic`. Each string is the
/// one a user would grep for, including the two diagnostic codes.
#[test]
fn gpu_and_blas_entries_keep_their_caveats() {
    let banner = USAGE_LINES.join("\n");
    for needle in [
        "--features gpu",
        "macOS",
        "gpu-unavailable",
        "cblas_sgemm",
        "blas-unavailable",
        "@deterministic",
    ] {
        assert!(banner.contains(needle), "usage banner lost `{}`", needle);
    }
}
