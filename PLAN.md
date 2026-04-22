# Current Implementation Plan — Library Merge + Worktree-Pair Refresh

## Problem statement

The project repo has already been consolidated, but the sibling fork library has
not.

Current situation:

- main project workspace is now the canonical merged branch,
- main sibling library `../circuit-tracer_chunked` is still on
  `exact-trace-hidden-knobs@62f5271` with uncommitted local edits,
- optimization sibling library `../worktrees_opt/circuit-tracer_chunked` is on
  `exact-trace-hidden-knobs-opt@e99647c`, which is two commits ahead of the base,
- the old optimization project worktree `../worktrees_opt/nlp_research_project`
  is also stale and still carries unique local/untracked content.

That means the **project repo and library are out of sync across workspaces**.
Because the runtime always imports the sibling library from the current
workspace-relative path, this is not just cleanup — it directly affects what code
our SLURM jobs are running.

So the next task is:

1. carefully merge the sibling library states,
2. establish one validated main project+library baseline,
3. carefully refresh the old optimization project+library pair from that
   baseline,
4. preserve unique operational files in the old optimization pair.

## Scope

This phase covers:

- comparing main vs optimization library checkouts,
- merging committed optimization-branch library work into the main sibling
  library while preserving main-side debug/checkpoint behavior,
- validating the merged library with lightweight local checks,
- refreshing the old optimization project+library pair together,
- preserving still-useful unique untracked material in the optimization pair.

## Non-goals

- do not touch or rewrite `midpoint_checkin_draft.md`,
- do not run GPU tracing on login nodes,
- do not treat generated artifacts as authoritative source material,
- do not blindly delete or recreate the old optimization worktree pair,
- do not open new optimization or cross-cluster investigation changes until the
  library merge is settled,
- do not rely on fp32 collapse as a "feature"; the permanent fix must remain
  semantics-preserving.

## Current branch relationship

### Project repo

- main project workspace: merged and validated on `exact-trace-bench-harness`
- old optimization project workspace: `exact-trace-bench-opt` with local docs
  drift and useful untracked operational files

### Sibling library

- main library workspace: `exact-trace-hidden-knobs@62f5271` plus local edits
- optimization library workspace: `exact-trace-hidden-knobs-opt@e99647c`
- optimization branch is ahead by two commits from the shared base:
  1. `b01ca8b` — `Add bounded refresh chunk caching`
  2. `e99647c` — `Add exact trace internal dtype control`

## Merge goals by side

### Preserve from main library workspace

- float64 / stable exact-trace behavior currently used by the main project repo,
- cross-cluster debug/checkpoint telemetry scaffolding,
- richer debug summary payloads,
- main-side local edits in:
  - `circuit_tracer/attribution/context_nnsight.py`
  - `circuit_tracer/replacement_model/replacement_model_nnsight.py`
  - `tests/test_chunked_decoder_optimizations.py`

### Preserve from optimization library workspace

- bounded refresh chunk caching,
- explicit exact-trace internal dtype control,
- `graph.py` compute-dtype plumbing,
- optimization-branch tests around caching / telemetry / partial influences,
- the optimization-oriented implementation direction already chosen in the spec.

## Highest-risk merge hotspots

These need explicit manual review rather than blind restore/cherry-pick:

- `circuit_tracer/attribution/attribute_nnsight.py`
- `circuit_tracer/graph.py`
- `tests/test_partial_influences.py`

These are important but likely more one-sided:

- `circuit_tracer/attribution/context_nnsight.py`
- `circuit_tracer/replacement_model/replacement_model_nnsight.py`
- `circuit_tracer/attribution/attribute.py`
- `tests/test_attribute_nnsight_telemetry.py`
- `tests/test_chunked_decoder_optimizations.py`

## Proposed approach

## Workstream A — Create one canonical library baseline in the main pair

### Goal

Make the main project workspace plus `../circuit-tracer_chunked` the canonical,
validated baseline for both tracks.

### Strategy

Use the **main library checkout as the behavioral base**, then reapply the
optimization-branch commits carefully on top of it.

Why:

- main already matches the newly consolidated project repo more closely,
- main carries the cross-cluster debug/checkpoint work we want to preserve,
- optimization branch contributes important committed work we do not want to lose
  (dtype control + bounded refresh chunk caching).

### Merge order inside the library

1. inspect and snapshot current main library dirty state,
2. ingest optimization-only files/ideas,
3. manually reconcile the hotspot files,
4. unify/expand tests,
5. run lightweight library-local validation.

## Workstream B — Library file-by-file reconciliation

### 1. `circuit_tracer/attribution/attribute_nnsight.py`

Target state:

- preserve main-side debug/checkpoint and summary schema,
- preserve whatever wiring the project repo now expects for cross-cluster debug,
- incorporate optimization branch support for:
  - bounded refresh chunk caching,
  - exact trace internal dtype control,
  - any related counters/telemetry,
- avoid reintroducing older semantics or silently dropping main-side debug data.

Acceptance:

- canonical debug artifacts still work,
- optimization cache + dtype control are present,
- the file remains consistent with the merged project repo interface.

### 2. `circuit_tracer/graph.py`

Target state:

- keep explicit compute-dtype control,
- keep semantics aligned with the chosen exact normalization contract,
- preserve any main-side safety changes,
- do not regress partial influence correctness.

### 3. `circuit_tracer/attribution/context_nnsight.py`

Target state:

- preserve main-side edits,
- confirm they remain compatible with optimization-branch caching/dtype changes,
- make only minimal merge-driven edits if needed.

### 4. `circuit_tracer/replacement_model/replacement_model_nnsight.py`

Target state:

- preserve main-side edits,
- verify they still match the merged tracing path expectations.

### 5. Tests

Target state:

- preserve optimization-branch tests:
  - `tests/test_attribute_nnsight_telemetry.py`
  - relevant additions in `tests/test_partial_influences.py`
- preserve main-side tests:
  - `tests/test_chunked_decoder_optimizations.py`
- update tests so they reflect the merged debug + dtype + cache behavior.

## Workstream C — Tie the merged library back to the project repo

### Goal

Confirm the merged project repo and merged main library actually agree.

### Required checks

1. project repo uses the canonical public knob `exact_trace_internal_dtype`,
2. library still exposes the behavior expected by the project repo,
3. cross-cluster debug artifacts still flow end-to-end,
4. requested dtype / resolved dtype data still appear consistently,
5. no project-side launcher/config assumptions are broken by the library merge.

## Workstream D — Refresh the old optimization pair carefully

### Goal

Make the old optimization pair operational again without wiping still-useful
files.

Target pair:

- project: `../worktrees_opt/nlp_research_project`
- library: `../worktrees_opt/circuit-tracer_chunked`

### Refresh principle

Refresh the **pair together** from the validated main baseline.

Do not refresh only the project or only the library.

### Required pre-refresh inspection

Before replacing anything, inventory:

- local tracked modifications,
- untracked docs/spec notes,
- fixture catalogs,
- generated scenario inputs,
- any other operational files needed to keep the optimization worktree usable.

### Refresh goal

After refresh, the optimization pair should:

- inherit the validated merged baseline,
- still contain any unique operational files worth keeping,
- be ready for the permanent overflow fix implementation.

## Workstream E — Use the chosen overflow-fix direction

### Goal

Make sure the optimization track begins from the already chosen spec direction,
not from an open-ended redesign.

Chosen direction from `docs/phase4_refresh_optimization_spec.md`:

- preferred first permanent fix: **scaled row-L1 computation** (or equivalent
  exact stable normalization representation)

Why this remains first:

- fp32 collapse on `828_base` and `361_base` is not acceptable as a stable
  solution,
- fp64 is the current safe default, but it is not the permanent numerical fix,
- the permanent fix should preserve exact semantics while removing raw
  row-abs-sum overflow as a failure mode.

## Safe validation plan

Only run lightweight local validation during the merge itself.

### Allowed

- `uv run ruff check ...`
- lightweight targeted unit tests
- `uv run python -m pytest <small test selection>` if cheap enough

### Not allowed here

- no direct GPU tracing on login nodes,
- no heavy benchmark reruns until the library merge is stable.

### Validation focus

1. partial influence correctness,
2. cache behavior / counters,
3. dtype-control behavior,
4. telemetry/debug artifact compatibility,
5. project-repo integration assumptions.

## Sequencing

### Phase 1 — Main library merge

1. capture current main library state,
2. compare hotspot files against optimization library,
3. merge and validate the main library,
4. only then treat the main pair as canonical.

### Phase 2 — Optimization pair refresh plan

1. inspect unique files in the old optimization project+library pair,
2. identify what must be preserved,
3. define exact refresh actions before applying them.

### Phase 3 — Optimization pair refresh

1. refresh old optimization project worktree from the validated main project
   baseline,
2. refresh old optimization library worktree from the validated main library
   baseline,
3. re-apply preserved unique files intentionally,
4. verify the pair is still operational.

### Phase 4 — Resume two-track work

After the pair refresh:

- main pair resumes cross-cluster investigation,
- optimization pair begins the permanent overflow fix,
- both tracks use the standard run placement convention:
  `ascend|cardinal` × `fast|anomaly|long_eval`.

## Acceptance criteria

This merge phase is complete when all of the following are true:

1. main project repo and main sibling library are intentionally merged and locally
   validated as a pair,
2. the main pair preserves:
   - canonical cross-cluster debug behavior,
   - explicit dtype control,
   - optimization-branch cache work,
3. old optimization project+library worktrees are refreshed together from that
   validated baseline,
4. useful untracked operational files from the optimization pair are preserved,
5. the optimization pair is ready to begin the permanent overflow fix from the
   chosen spec direction,
6. future runs can no longer accidentally mix project and library state without
   us noticing.

## Risks and open questions

### Risks

1. **Hotspot merge risk in `attribute_nnsight.py`**
   - easiest place to silently lose either debug schema or optimization logic.

2. **Project/library contract drift**
   - the project repo may assume interfaces or payloads that a careless library
     merge breaks.

3. **Optimization pair refresh damage**
   - blindly resetting the old worktree pair could lose useful docs/fixtures.

4. **False confidence from project-only validation**
   - the repo merge looked good, but jobs still depend on the sibling library.

### Open questions

- which files in the old optimization project worktree are genuinely unique and
  must survive the refresh?
- which files in the old optimization library worktree, if any, contain local
  uncommitted but still-useful material beyond the committed branch state?
- after the library merge, do we want a dedicated lightweight integration test at
  the project level that asserts the expected debug payload fields against the
  sibling library behavior?
