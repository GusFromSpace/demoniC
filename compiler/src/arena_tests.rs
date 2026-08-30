/// Arena sizing flag tests — `MEMORY.md §1.1`.
/// Parsing and rendering only; the metering itself is covered in
/// `interp_tests.rs` and `tests/arena_flags.rs`.

use super::arena::{flag_arena, fmt_bytes, parse_size, Arena, ArenaLimits};

#[test]
fn bare_byte_counts() {
    assert_eq!(parse_size("1"), Ok(1));
    assert_eq!(parse_size("4096"), Ok(4096));
    assert_eq!(parse_size("4096B"), Ok(4096));
    assert_eq!(parse_size("4096b"), Ok(4096));
}

#[test]
fn binary_units() {
    assert_eq!(parse_size("1K"), Ok(1024));
    assert_eq!(parse_size("1KiB"), Ok(1024));
    assert_eq!(parse_size("1kb"), Ok(1024));
    assert_eq!(parse_size("2M"), Ok(2 * 1024 * 1024));
    assert_eq!(parse_size("16G"), Ok(16 * 1024 * 1024 * 1024));
    assert_eq!(parse_size("1t"), Ok(1024u64 * 1024 * 1024 * 1024));
}

#[test]
fn zero_is_rejected() {
    let e = parse_size("0").unwrap_err();
    assert!(e.contains("zero"), "{}", e);
    let e = parse_size("0G").unwrap_err();
    assert!(e.contains("zero"), "{}", e);
}

#[test]
fn non_numeric_is_rejected() {
    for bad in ["", "G", "abc", "-1", "-1G", "1.5G", "1 G", "+2M"] {
        assert!(parse_size(bad).is_err(), "`{}` should not parse", bad);
    }
}

#[test]
fn unknown_unit_names_itself() {
    let e = parse_size("16Q").unwrap_err();
    assert!(e.contains("`Q`"), "{}", e);
    assert!(e.contains("B, K, M, G, or T"), "{}", e);
}

#[test]
fn overflow_is_rejected_not_wrapped() {
    // 2^24 TiB is exactly 2^64 bytes — one past a u64 byte count.
    let e = parse_size("16777216T").unwrap_err();
    assert!(e.contains("overflows"), "{}", e);
    // Just under it still parses, so the boundary is the multiply, not a guess.
    assert_eq!(parse_size("8388608T"), Ok(1u64 << 63));
    // Digits alone can overflow before any unit is applied.
    let e = parse_size("99999999999999999999").unwrap_err();
    assert!(e.contains("overflows"), "{}", e);
}

#[test]
fn flags_map_to_their_arena() {
    assert_eq!(flag_arena("--vault"), Some(Arena::Vault));
    assert_eq!(flag_arena("--vault=16G"), Some(Arena::Vault));
    assert_eq!(flag_arena("--forge"), Some(Arena::Forge));
    assert_eq!(flag_arena("--forge=2G"), Some(Arena::Forge));
    assert_eq!(flag_arena("--profile"), None);
    assert_eq!(flag_arena("--vaultish"), None);
    assert_eq!(flag_arena("run"), None);
}

#[test]
fn limits_start_unbounded() {
    let mut l = ArenaLimits::default();
    assert_eq!(l.forge, None);
    assert_eq!(l.vault, None);
    l.set(Arena::Forge, 1024);
    assert_eq!(l.forge, Some(1024));
    assert_eq!(l.vault, None, "sizing one arena must not bound the other");
}

#[test]
fn byte_rendering_round_trips_through_the_parser() {
    assert_eq!(fmt_bytes(512), "512 B");
    assert_eq!(fmt_bytes(1024), "1 KiB");
    assert_eq!(fmt_bytes(2 * 1024 * 1024), "2 MiB");
    assert_eq!(fmt_bytes(16 * 1024 * 1024 * 1024), "16 GiB");
    assert_eq!(fmt_bytes(1536 * 1024 * 1024), "1.5 GiB");
    // Exact renderings are valid flag values again.
    assert_eq!(parse_size("16GiB"), Ok(16 * 1024 * 1024 * 1024));
}
