# demoniC Package Manifests

**Companion to:** `docs/SPEC.md §6.6` (Modules and imports). Normative.

**Status:** Draft. This is a manifest format, not a package manager.

---

## 1. Purpose

A package manifest gives tools one static place to find a small demoniC
program graph.

It does not change language semantics. `use "path.dmc"` remains the
only import mechanism. The compiler shall not fetch dependencies, search
registries, infer modules from directories, or rewrite import paths from
the manifest.

## 2. File Name

The manifest file is named:

```text
demoni.json
```

It lives at a package root. Tooling may accept an explicit manifest path,
but repository checks look for files named `demoni.json`.

## 3. Required Fields

```json
{
  "schema": "demoni.package.v0",
  "name": "example-pack",
  "version": "0.0.1",
  "root": ".",
  "entry": "examples/package_manifest.dmc",
  "modules": {
    "ring_buffer": "examples/ring_buffer.dmc"
  }
}
```

`schema` must be exactly `demoni.package.v0`.

`name` is a lowercase package name. It may contain ASCII letters,
digits, `_`, and `-`. It must start with a lowercase letter.

`version` is an opaque release string matching `MAJOR.MINOR.PATCH`.

`root` is a relative directory path from the manifest file. It must
exist. `.` is the normal value.

`entry` is a relative `.dmc` file path from `root`. It must exist.

`modules` maps module aliases to relative `.dmc` file paths from `root`.
Each path must exist.

## 4. Optional Fields

`exports` is a list of module aliases from `modules`.

```json
{
  "exports": ["ring_buffer", "text_pipeline"]
}
```

If `exports` is omitted, tooling treats every module as internal. This
does not affect `pub`; it only gives packaging tools a public surface to
display.

## 5. Path Rules

All manifest paths must be relative. Paths must not contain `..`.

This keeps manifests vendorable by copy. A package may refer to files
inside a checked-in `vendor/` directory, but it must name them by path.
There is no package registry in this format.

## 6. Validation

Run the validator directly:

```
python3 tools/validate_manifest.py
```

It validates every `demoni.json` in the tree. It also runs in CI.
The validator rejects:

- missing required fields;
- unknown `schema`;
- absolute paths;
- parent-directory escapes;
- non-existent `root`, `entry`, or module files;
- module aliases outside `[A-Za-z_][A-Za-z0-9_]*`;
- exported names absent from `modules`.
