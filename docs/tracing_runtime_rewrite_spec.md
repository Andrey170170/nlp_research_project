# Tracing Runtime Rewrite Spec

Status: Phase C2 implementation in progress; closure pending login-safe and immutable Granite gates
Last updated: 2026-07-13
Scope: sibling tracing runtime plus the project execution path that invokes it

This document defines the target architecture for Phase C2, informally
"cleanup strikes again." It is normative. The descriptive maps under
`docs/architecture/` describe the currently landed implementation. They must
not describe the C2 Granite gate as closed until the immutable comparisons are
adjudicated.

Phase D is closed. Its accepted compact artifacts, incremental telemetry,
bounded-storage evidence, and failure behavior are frozen in
`docs/architecture/pre_c2_trace_contract.md` as the C2 comparison oracle.

## 1. Decision

The Phase C1 extraction reduced file size and separated several algorithms,
but it did not produce a comprehensible tracing runtime. The project no longer
needs to preserve upstream or legacy runtime API compatibility. Phase C2 will:

1. replace the tracing path with one canonical typed runtime;
2. model coherent domain concepts rather than wrap old argument lists in bags;
3. migrate sibling and project callers atomically;
4. delete obsolete entry points, translators, re-export namespaces, and tests;
5. preserve scientific semantics and retained artifact contracts through
   characterization and immutable GPU gates, not through API compatibility.

There is no compatibility window and no dual runtime. Retained research
artifacts may need explicit versioned readers, but old Python call signatures do
not constrain the new design.

## 2. Current Evidence

The visible problem is broader than `attribute_nnsight.py`.

| Surface | Current size / shape | Problem |
|---|---:|---|
| sibling `attribute_nnsight.py` | 2,254 lines; about 240 imported bindings | policy, validation, lifecycle, telemetry, compatibility, and execution aggregation |
| `_attribute_impl` | 375 lines; 90 parameters | duplicated public-to-private relay |
| `_run_attribution` | 1,360 lines; 92 parameters | roughly 760 lines before Phase 0 execution begins |
| sibling `runtime.py` | 510 lines | reflects the legacy signature, translates kwargs, and routes back into the old implementation |
| sibling `context_nnsight.py` | 1,715 lines; `compute_batch` 315 lines | graph capture, replay, buffers, attribution math, and diagnostics share one mutable object |
| sibling NNSight replacement model | 1,412 lines; setup method 308 lines | model adaptation and tracing procedure are interleaved |
| project `trace_pipeline_chunked.py` | 5,198 lines; completion path 2,549 lines | library execution, generation loop, artifact packaging, diagnostics, and project policy are mixed |
| project experiment runner | 1,151 lines; command builder 392 lines | scenarios are converted back into a flat CLI argument surface |
| project CLI | 2,768 lines; parser builder 1,678 lines | unrelated command families share one registration function |

TransformerLens attribution is a useful readability reference, not a size
target. NNSight legitimately owns more mechanisms. The requirement is that a
programmer can read the orchestration without first understanding storage,
replay, telemetry schemas, scheduler internals, or every provider implementation.

## 3. Engineering Principles

### 3.1 Complexity must have an owner

Memory management, replay, NNSight graph capture, provider topology, frontier
expansion, and telemetry remain complex. Each belongs to a named subsystem with
an explicit contract. The top-level runner must not reconstruct subsystem
internals through imports of private helpers.

### 3.2 Objects represent concepts, not displaced signatures

A meaningful object:

- has a name that explains why its fields belong together;
- establishes invariants when constructed;
- is produced or owned by one subsystem;
- is consumed as one coherent fact or capability;
- reduces reconstruction and repeated validation in consumers.

A dataclass that contains dozens of unrelated old arguments and is unpacked once
inside one phase is still an argument bag. Names such as `Context`, `Services`,
`State`, or `PhaseInputs` are insufficient unless the object has a precise
domain meaning and lifecycle.

Functions may take several arguments. Prefer direct, meaningful arguments over
an artificial wrapper. Introduce an object only when the values form a real
domain concept, result, policy, capability, or ownership boundary.

### 3.3 One canonical execution path

`trace_one`, `trace_batch`, and `open_session` use the same runner and backend
contracts. The project invokes those APIs directly. No internal call may route
through `attribute(...)`, `attribute_nnsight.py`, a legacy kwargs translator, or
a subprocess CLI argument mirror.

### 3.4 Resolve before executing

Request normalization, provider capability discovery, semantic validation,
physical-plan resolution, admission, and artifact-path validation happen before
the runner creates execution resources. Phases consume a resolved plan; they do
not reinterpret raw user fields or choose fallback behavior independently.

### 3.5 Resource ownership is explicit

Every NNSight graph, row store, replay ledger, cache, offload handle, observer,
and sink has one owner and one cleanup contract. Ownership transfer is explicit
in return types. Cleanup uses one run scope and preserves the primary exception.

### 3.6 Observability observes domain events

Tracing code emits typed lifecycle and domain events. It does not build JSON
dictionaries, sample resources, assign sequence numbers, flush files, render
human messages, or mutate result payloads to attach telemetry. Observability
adapters own those mechanics.

### 3.7 Dependency direction is enforced

The intended direction is:

```text
project campaign/harness
        -> sibling tracing API
        -> request + planning + admission
        -> runner
        -> backend phases
        -> provider/model/storage mechanisms

backend phases -> typed observability interface -> sinks/renderers
```

Lower layers do not import the runner, project harness, CLI, or campaign types.
The governor depends on provider profiles and mechanism catalogs, not concrete
phase implementations or model/checkpoint names.

## 4. Canonical Domain Model

Names may change during implementation, but each concept and ownership boundary
must remain recognizable.

| Concept | Meaning and invariant |
|---|---|
| `AttributionProblem` | model, prompt/token input, target selection, and graph objective; contains scientific intent, not execution tuning |
| `TraceSemantics` | choices allowed to affect the mathematical result; complete and fingerprintable |
| `TraceRequest` | public request composing one attribution problem, trace semantics, execution constraints, and named evidence; validates that each choice has one owner |
| `ResourceEnvelope` | operator/SLURM resource limits and provenance |
| `ExecutionConstraints` | explicit physical restrictions or mechanism requirements; no semantic fields |
| `ResolvedTracePlan` | validated phase plans, storage strategy, and admitted resource estimates; every field has one policy owner |
| `PreparedAttribution` | tokenized input, resolved targets, provider capabilities, dimensions, and immutable pre-run facts |
| `ActiveFeatureUniverse` | ordered active-feature identity, shape, membership fingerprint, and lookup operations |
| `ForwardTraceSession` | owned NNSight saved-forward capability with capacity and deterministic rebuild/close contract |
| `AttributionRowStorage` | interface for full-file, tiled, or recipe-ledger rows; exposes capabilities rather than backend flags |
| `SeedAttributions` | Phase-3 output: logit rows, denominators, initial frontier evidence, and stable hashes |
| `FrontierExpansion` | Phase-4 output: selected features, committed rows, frontier history, and final influence state |
| `GraphComponents` | backend-neutral graph nodes, edges, weights, tokens, and packaging metadata |
| `TraceResult` | output plus semantic/execution fingerprints, plan/admission report, and observability summary |

The plan composes domain-owned policies such as `SessionPlan`,
`RowStoragePlan`, `ReplayPlan`, `NumericalPolicy`, and `FrontierExpansionPlan`.
It must not become a flat replacement for the old 90-argument surface.

## 5. Target Runtime Flow

The final runner should expose the pipeline directly:

```python
def run_trace(
    problem: AttributionProblem,
    plan: ResolvedTracePlan,
    observer: TraceObserver,
) -> TraceResult:
    prepared = prepare_attribution(problem, plan.preparation, observer)

    with open_forward_session(prepared, plan.session, observer) as forward:
        active = discover_active_features(prepared, forward, plan.discovery, observer)
        rows = open_row_storage(active, plan.storage, observer)
        seeds = compute_seed_attributions(prepared, forward, rows, plan.phase3, observer)
        expansion = expand_frontier(prepared, forward, rows, seeds, plan.phase4, observer)
        components = assemble_graph(prepared, active, seeds, expansion)

    return package_trace_result(problem, plan, components, observer.summary())
```

This sketch is not a mandate for these exact function names or one physical
forward session. It is a readability contract: the algorithm and ownership
transitions are visible. Policy resolution, request normalization, telemetry
serialization, and cleanup bookkeeping do not obscure the flow.

## 6. Target Module Ownership

### 6.1 Sibling library

Proposed package shape:

```text
circuit_tracer/
  tracing/
    api.py                 # trace_one, trace_batch, open_session
    problem.py             # AttributionProblem and semantic request objects
    plan.py                # resolved execution-plan composition
    planning.py            # capability discovery, governor/admission invocation
    runner.py              # small backend-neutral coordinator
    session.py             # sequence/window session ownership
    result.py              # TraceResult and fingerprints
  attribution/
    targets.py
    nnsight/
      backend.py           # NNSight backend implementation of runner contracts
      preparation.py
      active_features.py
      seed_attribution.py
      frontier_expansion.py
      graph_assembly.py
      forward_session.py
      storage/
        protocol.py
        full_file.py
        tiled.py
        recipe_ledger.py
      policies/
        numerics.py
        sessions.py
        frontier.py
        replay.py
  observability/
    events.py
    observer.py
    run_scope.py
    recorder.py
    resources.py
    sinks.py
    human_logs.py
```

Existing good mechanisms may move rather than be rewritten. The package shape
is a responsibility map, not a demand to create one file per noun. Merge
modules when a split would only add pass-through boilerplate.

`attribute_nnsight.py`, its private-helper re-exports, `_attribute_impl`, the
legacy translator in `runtime.py`, and the generic flat `attribute(...)`
dispatch path are deleted after callers move. TransformerLens becomes another
backend behind the canonical tracing API; it is not a second public runtime.

### 6.2 Project harness

The project continues to own scenarios, campaigns, fixtures, SLURM policy,
immutable snapshots, output layout, comparisons, and scientific interpretation.
It does not implement attribution.

Target execution path:

```text
scenario -> project request builder -> sibling trace API -> TraceResult
         -> project artifact writer -> extraction/comparison
```

`trace_pipeline_chunked.py` is retired or reduced to a temporary migration
script and then deleted. Generation loops use `open_session` or `trace_batch`.
Compact graph conversion that is generally useful belongs in the sibling result
or graph API; experiment-specific artifact naming and sidecars remain project
owned.

The 392-line subprocess command builder must not remain the canonical typed
boundary. Scenario parsing builds domain requests directly. SLURM scripts may
still invoke a project CLI, but the CLI calls Python APIs rather than rebuilding
the old flat argument surface.

Split only the trace-related project CLI registration required by this path in
C2. Unrelated analysis commands are outside scope unless touched by the same
dependency problem.

## 7. Phase and Mechanism Contracts

Phase numbers remain useful in telemetry and scientific discussion, but code
contracts should use domain terms. Each operation must document:

1. required immutable inputs;
2. resource/capability ownership;
3. semantic invariants;
4. deterministic ordering requirements;
5. emitted domain events;
6. output and ownership transfer;
7. failure and cleanup behavior.

Storage and replay variants implement one capability protocol. Callers do not
branch on strings such as `column_tiled_v1` after plan resolution. The resolved
plan selects an implementation, and phases call its operations.

Provider differences enter through typed capabilities and provider operations.
No phase, runner, or governor branch may depend on `gemma`, `clt`, `plt`, a
checkpoint name, or repository naming convention.

## 8. Lifecycle and Observability

`TraceRunScope` owns the observer, recorder, sinks, resource sampling, terminal
event, exception attachment, and cleanup ledger. It does not own attribution
state or become a general run-context bag.

Resource owners register independent cleanup actions with the scope or use
nested context managers. Cleanup order is explicit. A cleanup failure cannot
skip later actions or mask the primary trace failure.

Phases receive a narrow `TraceObserver` protocol with domain methods or spans.
They never import JSONL sinks, schema serializers, human renderers, cgroup/CUDA
samplers, or exception-export helpers.

## 9. Migration Work Packages

### C2.0 - Freeze and characterize (complete)

- Adjudicate the final D/E outputs.
- Pin successful A--E compact artifacts, fingerprints, telemetry, failure
  behavior, and resource measurements as pre-rewrite references.
- Inventory every project and sibling caller of `attribute`,
  `attribute_nnsight`, `runtime._translate`, and private re-exports.
- Record current module/function dependency and ownership maps.

### C2.1 - Canonical domain API (implemented; gate pending)

- Define the domain objects in section 4 with construction-time validation.
- Remove `legacy_kwargs` and signature reflection from `TraceRequest`.
- Define backend, storage, observer, and provider capability protocols.
- Add architecture tests for dependency direction and canonical API ownership.

### C2.2 - Planning and admission boundary (implemented; gate pending)

- Move semantic validation, capability discovery, physical resolution, and
  admission into planning modules.
- Compose the existing Phase B governor contracts and Phase D mechanisms into a
  resolved plan without enabling automatic governor decisions yet.
- Ensure phases cannot consume raw requests or silently resolve fallback modes.

### C2.3 - NNSight runner rewrite (implemented; gate pending)

- Build the readable runner and migrate preparation, active-feature discovery,
  forward-session ownership, seed attribution, frontier expansion, and graph
  assembly behind explicit contracts.
- Refactor `context_nnsight.py`, replacement-model setup, and transcoder math
  where their responsibilities prevent those contracts from being honest.
- Keep algorithms and canonical ordering fixed while changing ownership.

### C2.4 - Lifecycle and observability rewrite (implemented; gate pending)

- Introduce the run scope and typed domain events.
- Remove telemetry schema construction and cleanup/result mutation from the
  runner and phases.
- Preserve incremental failure evidence and terminal-event guarantees.

### C2.5 - Atomic caller migration (implemented; final stale-surface audit pending)

- Migrate `trace_one`, `trace_batch`, `open_session`, TransformerLens routing,
  sibling tests, project scenario execution, full-answer sequencing, and compact
  artifact writing to the canonical API.
- Replace project subprocess argument assembly with typed request construction.
- Split trace command registration from the project CLI mega-parser.

### C2.6 - Delete old paths (in progress)

- Delete `attribute_nnsight.py`, `_attribute_impl`, legacy kwargs translation,
  generic flat attribution routing, private helper re-exports, obsolete project
  trace pipelines, and compatibility-only tests.
- Fail repository checks on stale imports or references.
- Update descriptive architecture maps only after deletion lands.

### C2.7 - Validation and closure (pending)

- Run focused subsystem, failure-injection, type, lint, and architecture tests.
- Run the broad login-safe sibling and project suites, with documented external
  test exclusions only.
- Create immutable project+sibling snapshots and run `361_base` 1B CLT/PLT
  through canonical full-retention and bounded mechanisms.
- Compare against the pinned pre-C2 references before Phase E starts.

## 10. Acceptance Criteria

Phase C2 closes only when all of the following hold:

1. There is one canonical tracing runtime for one, batch, and session execution.
2. Project tracing calls the sibling domain API directly; no flat subprocess
   knob relay is the canonical boundary.
3. `attribute_nnsight.py`, legacy translators, reflected legacy signatures,
   and compatibility re-export namespaces are absent.
4. The top-level runner visibly expresses preparation, active-feature
   discovery, forward-session ownership, row storage, seed attribution,
   frontier expansion, and graph assembly without policy or telemetry mechanics.
5. No canonical library function exposes the current flat 90-field surface.
   Multi-argument functions are accepted when every argument is a meaningful
   domain concept or capability.
6. Domain objects enforce documented invariants and are not unpacked argument
   bags. Architecture review explicitly audits this criterion.
7. Cross-domain validation happens once before execution; phases consume a
   resolved plan and cannot reinterpret request fields.
8. Resource ownership and cleanup are explicit, independently attempted, and
   primary-exception preserving.
9. Phases do not import sink, JSONL, resource-sampling, or human-log machinery.
10. Provider/model differences enter through capabilities, not family-name
    branches.
11. Existing hotspots (`context_nnsight`, replacement-model setup, transcoder
    attribution methods, project trace pipeline, command builder, trace CLI
    registration) are either decomposed by honest ownership or have a written
    justification for remaining cohesive. Moving code behind imports is not a
    pass condition.
12. Semantic and execution fingerprints remain independently stable.
13. Immutable Granite `361_base` CLT/PLT results meet the accepted graph parity,
    telemetry, lifecycle, and resource criteria against pre-C2 references.
14. Phase E consumes only the C2 canonical plan/runner boundary; no governor
    integration lands in a transitional runtime.

Line count is an audit signal, not the objective. Reviewers should be able to
explain each module's owner, each long function's single responsibility, and
the complete trace flow without chasing private imports across the package.
