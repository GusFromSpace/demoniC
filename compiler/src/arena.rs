//! Arena sizing flags — `MEMORY.md §1.1`.
//!
//! `dmc run model.dmc --vault=16G --forge=2G` puts a hard byte budget on the
//! Vault and Forge arenas. This module owns the two pieces both backends need:
//! parsing a size off the command line, and printing one back in a diagnostic.
//!
//! Sizes are integers with an optional binary unit suffix. No floats: `1536M`,
//! not `1.5G`. The brutalist reading — a byte count is a byte count.

/// Byte budgets for the two sized arenas. `None` is "unbounded", which is what
/// you get when the flag is absent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArenaLimits {
    pub vault: Option<u64>,
    pub forge: Option<u64>,
}

impl ArenaLimits {
    pub fn set(&mut self, which: Arena, bytes: u64) {
        match which {
            Arena::Vault => self.vault = Some(bytes),
            Arena::Forge => self.forge = Some(bytes),
        }
    }
}

/// The arena an allocation is charged against. The Stream arena (`MEMORY.md §9`)
/// has no sizing flag — a `KV` carries its own `capacity` — so it is not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arena {
    Vault,
    Forge,
}

impl Arena {
    pub fn name(self) -> &'static str {
        match self {
            Arena::Vault => "vault",
            Arena::Forge => "forge",
        }
    }

    /// The flag that sizes this arena, for use in diagnostics.
    pub fn flag(self) -> &'static str {
        match self {
            Arena::Vault => "--vault",
            Arena::Forge => "--forge",
        }
    }
}

/// Recognize `--vault`, `--vault=<size>`, `--forge`, `--forge=<size>`.
/// Returns the arena the flag sizes, or `None` for any other argument.
pub fn flag_arena(arg: &str) -> Option<Arena> {
    match arg.split('=').next().unwrap_or(arg) {
        "--vault" => Some(Arena::Vault),
        "--forge" => Some(Arena::Forge),
        _ => None,
    }
}

/// Parse an arena size: decimal digits plus an optional binary unit suffix
/// (`B`, `K`/`KiB`, `M`, `G`, `T`, case-insensitive). Returns the byte count.
///
/// Every rejection is a message the CLI can print verbatim: an empty value, a
/// non-numeric one, an unknown suffix, a count that overflows 64 bits, and zero
/// (an arena you cannot allocate from is a typo, not a configuration).
pub fn parse_size(text: &str) -> Result<u64, String> {
    if text.is_empty() {
        return Err("expected a size, e.g. `2G`".to_string());
    }
    let split = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
    let (digits, unit) = text.split_at(split);
    if digits.is_empty() {
        return Err(format!(
            "`{}` is not a size — expected digits with an optional \
             B/K/M/G/T suffix (e.g. `16G`)",
            text
        ));
    }
    let scale: u64 = match unit.to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1 << 10,
        "m" | "mb" | "mib" => 1 << 20,
        "g" | "gb" | "gib" => 1 << 30,
        "t" | "tb" | "tib" => 1 << 40,
        _ => {
            return Err(format!(
                "`{}`: `{}` is not a size unit — use B, K, M, G, or T \
                 (binary multiples; `16G` is 16 GiB)",
                text, unit
            ))
        }
    };
    let n: u64 = digits
        .parse()
        .map_err(|_| format!("`{}` overflows a 64-bit byte count", text))?;
    let bytes = n
        .checked_mul(scale)
        .ok_or_else(|| format!("`{}` overflows a 64-bit byte count", text))?;
    if bytes == 0 {
        return Err(format!("`{}` is zero — an arena needs a nonzero size", text));
    }
    Ok(bytes)
}

/// Render a byte count the way the sizing flags accept it back: binary units,
/// one decimal place, no decimal when the value is exact.
pub fn fmt_bytes(n: u64) -> String {
    const UNITS: [(u64, &str); 4] = [
        (1 << 40, "TiB"),
        (1 << 30, "GiB"),
        (1 << 20, "MiB"),
        (1 << 10, "KiB"),
    ];
    for (scale, name) in UNITS {
        if n >= scale {
            if n % scale == 0 {
                return format!("{} {}", n / scale, name);
            }
            return format!("{:.1} {}", n as f64 / scale as f64, name);
        }
    }
    format!("{} B", n)
}
