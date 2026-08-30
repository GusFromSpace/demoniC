use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use crate::ast::Program;
use crate::lexer::Lexer;
use crate::parser::Parser;

/// Why resolution stopped.
///
/// Was a bare `String` (#485): the lex and parse cases already carried a file
/// and a line/col, and flattening them to prose meant `dmc --check --json`
/// could only report a parse error as `unstructured` while `dmc jit` — which
/// parses in `main` and so still had the `ParseError` — reported it as `parse`.
/// Same failure, two encodings, decided by which command you ran.
///
/// `Display` reproduces the old strings byte for byte, so the human renderer is
/// unchanged.
#[derive(Debug)]
pub enum ResolveError {
    /// A located lexer failure in one file.
    Lex { file: PathBuf, msg: String, line: usize, col: usize },
    /// A located parser failure in one file.
    Parse { file: PathBuf, msg: String, line: usize, col: usize },
    /// Everything with no span: I/O, a bad import path, a cycle.
    Other(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::Lex { file, msg, line, col } =>
                write!(f, "lex error in {:?}: lex error at {}:{}: {}", file, line, col, msg),
            ResolveError::Parse { file, msg, line, col } =>
                write!(f, "parse error in {:?}: parse error at {}:{}: {}", file, line, col, msg),
            ResolveError::Other(s) => write!(f, "{}", s),
        }
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

        self.dfs(&canonical_main, &mut visited, &mut path_stack)?;
        Ok(())
    }

    fn dfs(
        &mut self,
        path: &PathBuf,
        visited: &mut HashSet<PathBuf>,
        path_stack: &mut Vec<PathBuf>,
    ) -> Result<(), ResolveError> {
        if path_stack.contains(path) {
            let cycle_start = path_stack.iter().position(|p| p == path).unwrap();
            let mut cycle = path_stack[cycle_start..].to_vec();
            cycle.push(path.clone());
            let cycle_strs: Vec<String> = cycle.iter()
                .map(|p| format!("{:?}", p))
                .collect();
            return Err(ResolveError::Other(
                format!("circular import detected: {}", cycle_strs.join(" -> "))));
        }

        if visited.contains(path) {
            return Ok(());
        }

        let program = parse_file(path)?;

        path_stack.push(path.clone());

        let parent_dir = path.parent().unwrap_or(Path::new(""));
        for item in &program.items {
            if let crate::ast::Item::Use(us) = item {
                let import_path = parent_dir.join(&us.path);
                let canonical_import = import_path.canonicalize()
                    .map_err(|e| ResolveError::Other(format!(
                        "error resolving import {:?} in file {:?}: {}",
                        us.path, path, e
                    )))?;
                self.dfs(&canonical_import, visited, path_stack)?;
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
