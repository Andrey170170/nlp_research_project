# Current Project Roadmap

Status: Current scratch roadmap
Last updated: 2026-07-10

## Active Priority

The active work is the memory-governor and reusable tracing-runtime program in
`plans/2026-07-03_governor_rearch.md`, targeting the contract in
`docs/memory_governor_rearchitecture_spec.md`.

Phase A is closed for implementation purposes:

- Cardinal A4 supplied the scoped semantics-caste and session evidence needed
  to freeze the API. The delayed Ascend reproduction is optional and does not
  block implementation.
- Granite H200 strict baselines completed for Gemma 3 1B CLT, 1B PLT, 4B PLT,
  and 12B PLT across `828_base`, `361_base`, and `94_base`.
- The 12B rerun completed from an immutable project+sibling snapshot with live
  telemetry. Its three traces took 5.8--6.4 hours at b64/c4096. Retain the 600G
  host request until the 38--244 GiB fixture-dependent MaxRSS spread is repeated
  and explained.
- Snapshot-by-default launch enforcement and incremental telemetry are durable
  infrastructure, not pending roadmap tasks.

The pre-Phase-B roadmap was archived at
`docs/history/current_project_roadmap_pre_phase_b_2026-07-10.md`.

## Phase B - Complete

Phase B completed as login-safe contract and arithmetic work. It does not load
models or run attribution.

### B1. Freeze the taxonomy and schemas

1. Complete `docs/knob_api_taxonomy.md` with tier, demand class, bytes-cost
   formula, caste, validation scope, and ownership for every governor-relevant
   field.
2. Keep semantic choices scenario/provider-owned in `strict`; memory pressure
   may select only separately represented physical mechanisms.
3. Record the coupled NNSight trace-capacity family explicitly:
   `max(source_batch_size, feature_batch_size, logit_batch_size)`.
4. Version provider calibration profiles separately from semantic-relaxation
   evidence. Granite data calibrates cost/resources; Cardinal A4 remains the
   only current validated-relaxed evidence scope.

### B2. Land governor v0 in the sibling

Implement in `../circuit-tracer_chunked`:

- immutable `TraceSemantics`, `ResourceEnvelope`, provider-profile,
  `TracePlan`, demand-estimate, and admission-report value objects;
- stable semantic, profile, envelope, and execution fingerprints;
- SLURM/cgroup host-budget discovery with explicit provenance;
- a pure provider-agnostic resolver with no model loading or model-family
  branches;
- rigid VRAM/host admission, elastic file-cache planning, spill selection,
  walltime reporting, trace-capacity binding diagnostics, and actionable
  refusal reasons;
- synthetic cross-layer, same-layer, and approximate/top-k fixtures plus
  versioned Granite 1B/4B/12B calibration fixtures.

Completion record (2026-07-10):

- sibling branch/commit: `phase-b-governor-contract` / `0ce3f96`;
- B1 taxonomy frozen in `docs/knob_api_taxonomy.md`;
- B2 contracts, trusted-evidence registry, nested cgroup/Slurm discovery,
  provider profiles, pure resolver, fingerprints, and admission reports landed;
- 38 focused governor tests and 51 existing telemetry/provider regressions pass;
- Ruff and uv-based Pyright pass.

Resolver outputs remain advisory until Phase C consumes them and Granite parity
validates runtime integration. The trusted validation-evidence registry is
intentionally empty until transferred A4 artifacts are reproduced and reviewed.

## Phase C - Active Next

The sibling owns the tracing runtime and public API. Land it in reviewable
slices, preserving strict compact-output parity after each behavioral change:

1. split the attribution mega-module mechanically around runtime phases;
2. add `TraceRequest`, `TraceResult`, `trace_one`, `trace_batch`, and
   `open_session`, retaining `attribute(...)` as a compatibility facade;
3. translate legacy conflated knobs into separate logical semantics and
   physical execution fields with versioned deprecation telemetry;
4. make phases consume governor grants and add semantics-preserving residency,
   row-store, replay, prefetch, and bounded-execution ladders;
5. validate each runtime landing on Granite from immutable two-repo snapshots.

## Phase D - After the Sibling API Stabilizes

Keep experiment policy in this repository. Adapt the harness once to the stable
sibling seam:

- map scenarios and CHPC allocations into sibling requests/envelopes;
- persist streamed telemetry and both fingerprints;
- keep fixtures, campaigns, SLURM policy, snapshots, artifact layout,
  extraction, comparison, and scientific interpretation project-side;
- consolidate launches through the packaged CLI and archive compatibility-only
  wrappers.

Do not begin broad Phase D refactors while Phase C contracts are still moving.

## Validation and Provenance

- Login nodes: pure resolver tests, lightweight unit tests, `ruff`, and `ty`
  only. Model/provider loading and attribution remain SLURM-only.
- Every SLURM run executes from an immutable read-only project+sibling snapshot
  unless an explicitly justified live-workspace exception is recorded.
- Every governed run records both repository states, snapshot manifest, output
  root, allocation, profile/evidence versions, semantic fingerprint, execution
  fingerprint, and job IDs.
- Baseline-changing decisions go to `EXPERIMENTS.md`; verbose events go to
  `experiments/logs/YYYY-MM.jsonl`.

## Deferred, Non-Blocking Follow-ups

- transfer the Cardinal A4 source artifacts from OSC and regenerate the derived
  report before shipping any validated-relaxed profile as a runtime default;
- optionally reproduce A4 on Ascend for historical environment comparison;
- repeat Granite calibration where needed to explain resource variance and
  tighten conservative envelopes;
- revisit dynamic post-Phase-0 spending only after Phase C mechanisms prove
  fixed-semantics parity.
