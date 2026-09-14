use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use crate::ast::Program;
use crate::lexer::{Lexer, Span};
use crate::parser::Parser;

/// Why resolution stopped.
///
/// Was a bare `String` (#485): the lex and parse cases already carried a file
/// and a line/col, and flattening them to prose meant `dmc --check --json`
/// could only report a parse error as `unstructured` while `dmc jit` — which
/// parses in `main` and so still had the `ParseError` — reported it as `parse`.
/// Same failure, two encodings, decided by which command you ran.
///
/// The import cases went the same way next: a `use` that names a file which
/// does not exist, or that closes a cycle, is raised while the resolver is
/// standing on the `use` item — it has the item's span and the importing
/// file's path — and flattening that to prose threw the location away. `Import`
/// and `Cycle` keep it.
///
/// `Display` reproduces the old strings byte for byte for `Lex` and `Parse`
/// (unaffected by #855), and now prints `Import`/`Cycle` located — `resolve
/// error at L:C: <msg>`, the same shape `lex error at`/`parse error at`/`type
/// error at` already use — instead of the old bare, unlocated `msg` (#855:
/// `dmc --check`, `run` and `jit` printed the sentence with no line/col at
/// all). `msg` itself no longer names the importing file for `Import`: the
/// span already does, and the importing file's absolute path in the message
/// body was both redundant and the thing tripping `check_pii.py` on any test
/// that quoted the rendered sentence. `Cycle`'s `msg` still names every file
/// in the cycle — that is the diagnostic's whole content, not redundant with
/// the span, which only points at the `use` that closed it.
#[derive(Debug)]
pub enum ResolveError {
    /// A located lexer failure in one file.
    Lex { file: PathBuf, msg: String, line: usize, col: usize },
    /// A located parser failure in one file.
    Parse { file: PathBuf, msg: String, line: usize, col: usize },
    /// A `use` whose target could not be resolved or read. `file` is the
    /// importing file and `line`/`col`/`start`/`end` the `use` item in it;
    /// `msg` is the human sentence, naming the import string and the OS's
    /// reason but not the importing file — the span names it instead.
    Import { file: PathBuf, msg: String, line: usize, col: usize, start: usize, end: usize },
    /// A `use` that closes an import cycle. `file` is the file whose `use`
    /// closes the cycle and `line`/`col`/`start`/`end` that `use`; `msg` is
    /// the human sentence, which names the whole cycle.
    Cycle { file: PathBuf, msg: String, line: usize, col: usize, start: usize, end: usize },
    /// Everything with no span: a per-file I/O failure under `dmc test`,
    /// hit reading a file the resolver was asked to treat as an entry point
    /// (a test file in the directory), where there is no importing `use` to
    /// blame it on. An I/O failure on the actual command-line entry file
    /// never reaches here: `main` reads and canonicalizes that file itself,
    /// before the resolver runs, and reports it as `cli`/`input-unreadable`
    /// or `input-unresolvable`.
    Other(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::Lex { file, msg, line, col } =>
                write!(f, "lex error in {:?}: lex error at {}:{}: {}", file, line, col, msg),
            ResolveError::Parse { file, msg, line, col } =>
                write!(f, "parse error in {:?}: parse error at {}:{}: {}", file, line, col, msg),
            ResolveError::Import { msg, line, col, .. } | ResolveError::Cycle { msg, line, col, .. } =>
                write!(f, "resolve error at {}:{}: {}", line, col, msg),
            ResolveError::Other(s) => write!(f, "{}", s),
        }
    }
}

impl ResolveError {
    /// Whether the error carries a file and a line/col.
    pub fn is_located(&self) -> bool {
        !matches!(self, ResolveError::Other(_))
    }
}

pub struct Resolver {
    pub files: HashMap<PathBuf, Program>,
    pub sorted_paths: Vec<PathBuf>,
}

impl Resolver {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            sorted_paths: Vec::new(),
        }
    }

    pub fn resolve_all(&mut self, main_path: &Path) -> Result<(), ResolveError> {
        let mut visited = HashSet::new();
        let mut path_stack = Vec::new();
        let canonical_main = main_path.canonicalize()
            .map_err(|e| ResolveError::Other(
                format!("error resolving path {:?}: {}", main_path, e)))?;

        self.dfs(&canonical_main, None, &mut visited, &mut path_stack)?;
        Ok(())
    }

    /// Resolve `path` and, depth-first, everything it imports.
    ///
    /// `origin` is the `use` that brought us here — the importing file and the
    /// item's span — or `None` for the entry file. It is what locates a read
    /// failure on an imported file: the failure is the `use`'s, not the
    /// missing file's.
    fn dfs(
        &mut self,
        path: &PathBuf,
        origin: Option<(&Path, &Span)>,
        visited: &mut HashSet<PathBuf>,
        path_stack: &mut Vec<PathBuf>,
    ) -> Result<(), ResolveError> {
        if visited.contains(path) {
            return Ok(());
        }

        let program = match parse_file(path) {
            Ok(p) => p,
            Err(ResolveError::Other(msg)) if origin.is_some() => {
                let (file, span) = origin.unwrap();
                return Err(ResolveError::Import {
                    file: file.to_path_buf(), msg,
                    line: span.line, col: span.col,
                    start: span.start, end: span.end,
                });
            }
            Err(e) => return Err(e),
        };

        path_stack.push(path.clone());

        let parent_dir = path.parent().unwrap_or(Path::new(""));
        for item in &program.items {
            if let crate::ast::Item::Use(us) = item {
                let import_path = parent_dir.join(&us.path);
                let canonical_import = import_path.canonicalize()
                    .map_err(|e| ResolveError::Import {
                        file: path.clone(),
                        msg: format!("cannot resolve import {:?}: {}", us.path, e),
                        line: us.span.line,
                        col: us.span.col,
                        start: us.span.start,
                        end: us.span.end,
                    })?;
                // A cycle is detected at the `use` that closes it, so the
                // diagnostic can point at that `use`. The stack holds every
                // file on the path from the entry file to this one.
                if let Some(cycle_start) = path_stack.iter().position(|p| *p == canonical_import) {
                    let mut cycle = path_stack[cycle_start..].to_vec();
                    cycle.push(canonical_import);
                    let cycle_strs: Vec<String> = cycle.iter()
                        .map(|p| format!("{:?}", p))
                        .collect();
                    return Err(ResolveError::Cycle {
                        file: path.clone(),
                        msg: format!("circular import detected: {}", cycle_strs.join(" -> ")),
                        line: us.span.line,
                        col: us.span.col,
                        start: us.span.start,
                        end: us.span.end,
                    });
                }
                self.dfs(&canonical_import, Some((path, &us.span)), visited, path_stack)?;
            }
        }

        path_stack.pop();
        visited.insert(path.clone());
        self.files.insert(path.clone(), program);
        self.sorted_paths.push(path.clone());

        Ok(())
    }
}

fn parse_file(path: &Path) -> Result<Program, ResolveError> {
    let src = std::fs::read_to_string(path)
        .map_err(|e| ResolveError::Other(format!("error reading {:?}: {}", path, e)))?;
    let tokens = Lexer::new(&src)
        .tokenize()
        .map_err(|e| ResolveError::Lex {
            file: path.to_path_buf(), msg: e.msg, line: e.line, col: e.col,
        })?;
    Parser::new(tokens)
        .parse_program()
        .map_err(|e| ResolveError::Parse {
            file: path.to_path_buf(), msg: e.msg, line: e.line, col: e.col,
        })
}
