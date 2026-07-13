# Architecture docs index

Status: Current-state code maps
Last updated: 2026-07-13

These pages describe the current workspace and code layout. They are
descriptive references, not target architecture rules or implementation plans.
The normative Phase C2 contract is `../tracing_runtime_rewrite_spec.md`.
These maps describe the landed implementation while its immutable Granite gate
is still pending.

## Maps

| File | Scope |
|---|---|
| `workspace_map.md` | Two-repo workspace roles and integration boundaries |
| `project_code_map.md` | Main project package, CLI, artifacts, and safe command map |
| `sibling_circuit_tracer_chunked_code_map.md` | Sibling library package/API/phase/optimization map |

## Usage notes

- Use these maps to orient current-state work and review integration points.
- Keep workflow decisions in the owning docs; do not treat these pages as
  policy or a target design.
