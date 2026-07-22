# Memory governor + rearchitecture execution plan

Status: superseded as an execution checklist; retained for implementation history
Date: 2026-07-03; last updated 2026-07-21
Scope: sibling library `../circuit-tracer_chunked` rewrite + project harness
restructure + validation campaigns

Target design (the "how it is supposed to be" document):
`docs/memory_governor_rearchitecture_spec.md`. This plan is the "how we get
there" companion. Rearchitecture evidence base: the two scout reports,
`reports/circuit_tracer_architecture_scout.md` and
`reports/harness_architecture_scout.md` (both 2026-06-30, pre-merge — see
Phase R0).

The active work state now lives in `docs/current_project_roadmap.md`. Fidelity,
calibration, and solver language below that still refers to `strict` or
`validated_relaxed` is historical. The current contract is the four-policy
`exact` / `bounded` / `best_effort` / `research` design in the target spec.

## Problem statement

Two intertwined problems, one plan:

1. The exact-trace optimization surface is a flat bag of per-mechanism knobs
   that cannot trade resources across the VRAM / host-RAM / file-backed tiers.
   The target design replaces it with a budget-driven memory governor
   (invariant, castes, plan epochs, degradation ladders — see the spec).
2. At the start of C2, the first sibling decomposition had moved phase algorithms and observability
   mechanics, but left an accidental aggregation point: `attribute_nnsight.py`
   is still 2,254 lines, imports about 240 bindings, and relays a 90-field
   surface into a 1,360-line runner. The project still mirrors that surface
   through a 5,198-line trace pipeline and a 392-line command builder. Phase C2
   replaces this path with one coherent domain runtime; compatibility with the
   old Python API is explicitly not a constraint.

The governor is not an add-on: it is the scout report's "exact-trace
policy/config seam" (sibling candidate #2) with a concrete job description,
so both efforts share one migration.

Architecture coverage invariant: the rework is **provider-agnostic**. The memory governor
and speedups must apply to any model/transcoder pair already supported by the
exact/chunked provider contract, not only the Gemma/GemmaScope2 combinations
used for initial evidence. Gemma 3 12B + PLT, GPT-OSS 20B + PLT, and Llama 3.1
8B + a top-k transcoder should all use the same planner if their providers
declare the required capabilities/metadata. The governor may branch on provider
capabilities and topology; it must not branch on model-family or checkpoint-name
special cases.

Validation-scope caveat: the A4 caste decision below is intentionally narrow.
It applies only to corrected-hook Gemma-3-1B + GemmaScope2 PLT-small on
Cardinal in fp32 for `t01_361_s1002_g300` (and the explicit 298--300 window)
under the listed configurations. It is not evidence for 4B, 12B, CLT, another
cluster, another dtype, or another provider regime.

## Step 0 — Merge PLT parity + hook fix into main (DONE 2026-07-03)

- Project: PR #3 merged as `e020a34` (PLT optimization parity harness); PR #2
  merged as `1cc5370` (circuit stability analysis pipeline).
- Sibling: PR #2 merged as `3abdc98`, including post-review fix `d77f5cc`
  (preserve exact provider metadata in caches) and other automated-review
  fixes.
- Verified on merged main: `transcoder_config.py` defaults
  `feature_input_hook="mlp.hook_in"` for both CLT and PLT presets — the
  wrong-hook (`hook_resid_mid`) contamination is fixed at the preset level
  for both architectures.
- Residual caveat: all pre-fix GemmaScope2 artifacts (CLT especially) remain
  contaminated provenance; nothing that touched `hook_resid_mid` counts as a
  baseline or as knob-caste validation evidence.

## Phase R0 — Scouting refresh (DONE 2026-07-06, login-safe, small)

The scout reports predate the PLT parity merge. Before Phase C1 relies on
their line references and module inventories:

1. Re-verify the sibling map deltas: `transcoder/provider.py` (new, ~160
   lines), PLT provider paths in `single_layer_transcoder.py` (~960 lines),
   caching/hf_utils provider-metadata changes from the merge and review
   fixes.
2. Update or annotate `reports/circuit_tracer_architecture_scout.md` line
   spans where drifted; confirm the deepening-candidate list is unchanged in
   substance (expected: yes — the mega-module is untouched by the merge).
3. Same light pass over the harness scout for `transcoder_config.py`,
   `cli.py`, `runner.py` growth from the PLT work.

Done when: scout reports carry a post-merge delta note and Phase C1 tasks can
cite current line references.

R0 result: `reports/circuit_tracer_architecture_scout.md` and
`reports/harness_architecture_scout.md` now carry post-merge delta notes. The
deepening order is unchanged in substance; the main delta is the new
provider/capability/cache-metadata seam from PLT parity, which broadens the
transcoder/HF cleanup notes without changing the attribution-first ordering.

## Phase A — Corrected-regime validation + calibration (SLURM, before rewrite)

Goal: establish clean baselines and resolve the knob-caste questions the
governor design forks on. No library rewrite until A is read out.

### A1. Selection-margin telemetry (small library addition, output-invariant)

Add Phase-4 selection diagnostics to the existing telemetry payloads: did
the `max_feature_nodes` cap bind; relative k vs k+1 score gap; count of
features within epsilon of the boundary. Login-safe tests for the payload;
no behavior change. This must land before A3 so probe runs record *why*
outputs move, not just whether.

### A2. Corrected-hook CLT re-baseline

- Re-run canonical CLT gates (`828_base`, `361_base`, `94_base`, fast tier)
  with corrected hooks on main.
- Record new baselines; mark all `hook_resid_mid`-era CLT references as
  contaminated provenance in `EXPERIMENTS.md`.
- Sanity gate on every run: `active_features ≈ tokens x layers x trained L0`.

A2 result (2026-07-07): complete. Cardinal job `12242778_[0-2]` and Ascend job
`6242253_[0-2]` all completed with `ExitCode=0:0`. The corrected-hook CLT fast
artifacts for `828_base`, `361_base`, and `94_base` are under:

- `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/phase-a2-corrected-clt-fast-cardinal-20260706`
- `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/ascend/fast/phase-a2-corrected-clt-fast-ascend-20260706`

Use these as the current corrected-hook CLT fast baselines. Older
`hook_resid_mid` CLT/PLT artifacts remain contaminated provenance only.

### A3. FP-sensitivity + cost-model probe campaign (1B/4B)

One campaign, two axes, per the spec section 10 step 2:

- Sensitivity axis: per (model, prefix size): `decoder_chunk_size` sweep
  {2048, 4096, 8192} with everything else pinned + same-config repeat run as
  determinism control; diff compact outputs; read selection margins.
- Calibration axis: prefix-length spread yields nnz-vs-tokens curves, unit
  decoder-chunk/encoder-row timings, rigid-vs-elastic memory curves
  (cgroup anon vs file, confirmed visible in Grafana). Record calibration data
  against provider profile fields (topology, dimensions, dtype, checkpoint bytes,
  capability flags), not as GemmaScope2-only constants.

Decision gate out of Phase A:

- `decoder_chunk_size` caste in the corrected regime: performance-only
  (governor gets its biggest VRAM lever) or semantics-caste
  (scenario-pinned). Record the verdict with provenance in
  `docs/knob_api_taxonomy.md` and `EXPERIMENTS.md`.
- Whether PLT ~10:1 selection binding shows FP sensitivity at all, and
  whether tolerance-aware tie-breaking needs to enter the restructure scope.

A3 interim result (2026-07-07): partial, not sufficient for the final Phase-A
decision gate. The Cardinal matrix `12242814`--`12242853` used the launch
snapshot recorded on 2026-07-06. It produced `8` successful traces and `32` CUDA
OOM failures:

- success scope: only `t00_828_det_g000` (`prefix_token_count=73`) succeeded for
  both `plt_1b_small` and `plt_4b_small` across the chunk sweep plus the
  provider-default repeat;
- failure scope: every target with `prefix_token_count>=424` failed with
  `torch.cuda.OutOfMemoryError` during the Gemma MLP forward, across both
  providers and all requested chunks/repeats. These failures used aggressive
  memory-for-speed knobs (`cross_batch_decoder_cache_bytes` 16--32 GiB,
  prepared refresh cache 8--16 GiB, replay windows 8--16, and active pinned CPU
  encoder residency), so they need lower-memory reruns before being interpreted
  as inherent long-prefix infeasibility;
- shortest-prefix cost signal: larger chunks were much faster (`plt_1b_small`
  roughly `918s -> 438s -> 247s` for `2048 -> 4096 -> 8192`; `plt_4b_small`
  roughly `2639s -> 1683s -> 991s`), but compact artifacts drifted across chunk
  sizes.

Instrumentation caveat: the A1 ranker-frontier selection-margin telemetry is
present in ordinary `trace_pipeline_chunked.py` telemetry output, but the A3
full-answer runner did not persist generic `telemetry_events` into the per-token
`trace_results.jsonl`/`trace.json` artifacts. Project commit `620e781` fixed the
compact success path by writing per-token `telemetry.jsonl` sidecars and
recording `telemetry_event_count` / `telemetry_events_path` in the trace row.
The first post-fix pilot still failed before a compact result was available, so
error-path telemetry persistence is still not proven.

Survival pilot result (2026-07-07): the two-job Cardinal pilot for failed target
`t01_361_s1002_g300` finished, still with CUDA OOM. Initial 400G/4h submissions
`12255831` and `12255832` were canceled while pending; replacement 250G/2h jobs
used lower memory-for-speed knobs (`cross_batch_decoder_cache_bytes=0`, prepared
refresh cache `0`, replay window `4`, prefetch `2`, lazy encoder residency, no
row-store preallocation):

- `12255846` (`plt_1b_small`, chunk `8192`) failed `FAILED|1:0` after `00:02:01`
  with MaxRSS `66124448K`; the trace row captured an `OutOfMemoryError`
  allocating `5.59 GiB`, Phase-0 active features `227051`, precompute `60.21s`,
  and CUDA peak allocated `91.14 GiB`.
- `12255847` (`plt_4b_small`, chunk `4096`) failed `FAILED|1:0` after `00:04:14`
  with MaxRSS `189595184K`; the trace row captured an `OutOfMemoryError`
  allocating `4.14 GiB`, Phase-0 active features `313100`, precompute `172.59s`,
  and CUDA peak allocated `89.65 GiB`.

The pilot ruled out the aggressive cache/residency knobs as the only blocker, but
it did not exercise the existing Phase-1 trace-batch cap: both jobs used
`phase1_trace_batch_policy=legacy` with effective source batch `1024` (1B) or
`512` (4B). Next pilot should keep the survival profile and add
`phase1_trace_batch_policy=cap_effective_batches`, starting with
`phase1_trace_batch_size_max=128` for 1B and `64` for 4B, before any broad A3
relaunch. This survival-v2 pilot was submitted on 2026-07-07 as Cardinal jobs
`12260794` (1B c8192 cap128) and `12260795` (4B c4096 cap64), using immutable
dirty-workspace snapshots with the error-path telemetry fix and full-answer
Phase-1 cap plumbing.

Survival-v2 phase1cap result (2026-07-07): both jobs still failed with CUDA OOM
during Phase 1 forward, but the cap plumbing and error-path telemetry persistence
were validated on real failed GPU artifacts:

- `12260794` (`plt_1b_small`, chunk `8192`, cap128) applied the source batch cap
  (`1024 -> 128`) and failed `FAILED|1:0` after `00:01:55`, MaxRSS `66121944K`,
  `OutOfMemoryError` allocating `5.59 GiB`; Phase-0 active features `227051`,
  precompute `62.45s`, CUDA peak allocated `91.14 GiB`, token telemetry sidecar
  persisted 4 events.
- `12260795` (`plt_4b_small`, chunk `4096`, cap64) applied the source batch cap
  (`512 -> 64`) and failed `FAILED|1:0` after `00:04:11`, MaxRSS `189633676K`,
  `OutOfMemoryError` allocating `4.14 GiB`; Phase-0 active features `313100`,
  precompute `175.01s`, CUDA peak allocated `89.65 GiB`, token telemetry sidecar
  persisted 4 events.

Resulting A3 guidance: Phase-1 source-batch capping alone is not enough for this
long-prefix PLT target. If A3 is continued before the governor rewrite, the next
bounded pilot should lower the whole Phase-1 pressure surface: smaller source cap
plus lower feature/logit/attribution batch sizes. Do not relaunch the broad A3
failed matrix yet.

Survival-v3 allbatch launch (2026-07-07): committed the telemetry/cap plumbing
fixes, then submitted the less-drastic first bracket with all main batches
coherently lowered while keeping `max_feature_nodes` and `update_interval`
unchanged:

- `12263597` (`plt_1b_small`, chunk `8192`): `attribution_batch_size=256`,
  `feature_batch_size=256`, `logit_batch_size=256`,
  `phase1_trace_batch_size_max=256`.
- `12263598` (`plt_4b_small`, chunk `4096`): `attribution_batch_size=128`,
  `feature_batch_size=128`, `logit_batch_size=128`,
  `phase1_trace_batch_size_max=128`.

Launch provenance is under
`/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/manual_scenarios/phase-a3-survival-v3-allbatch-lessdrastic-20260707/manifest.json`.

Survival-v3 allbatch result (2026-07-08): both coherent all-main-batch jobs
completed and saved compact graphs for the failed long-prefix t01 target:

- `12263597` (`plt_1b_small`, c8192, all main batches `256`) completed `0:0`
  after `00:15:05`, MaxRSS `79193196K`, trace time `857.36s`, telemetry events
  `4623`, graph artifact `9.80 MiB`; trace capacity was `256` bound equally by
  source/feature/logit, Phase-0 active features `227051`, Phase-1 forward
  `1.23s`, forward CUDA peak allocated/reserved `79.69/85.90 GiB`.
- `12263598` (`plt_4b_small`, c4096, all main batches `128`) completed `0:0`
  after `01:34:05`, MaxRSS `211938236K`, trace time `5557.93s`, telemetry events
  `16551`, graph artifact `9.92 MiB`; trace capacity was `128` bound equally by
  source/feature/logit, Phase-0 active features `313100`, Phase-1 forward
  `1.44s`, forward CUDA peak allocated/reserved `65.30/75.35 GiB`.

Runtime breakdown: Phase 1 should be treated as an admission/survival gate, not
the optimization target. The completed traces are dominated by Phase 3/4:

- 1B b256: Phase-3 logit attribution `44.63s`, Phase-4 feature attribution
  `744.66s`, together about `92%` of trace time; Phase 1 `1.23s`.
- 4B b128: Phase-3 logit attribution `147.25s`, Phase-4 feature attribution
  `5220.87s`, together about `97%` of trace time; Phase 1 `1.44s`.

Phase-3/4 sampled CUDA allocation stayed around `23.0 GiB` for 1B and `33.2 GiB`
for 4B, while PyTorch retained the earlier forward-pass reserve. Governor work
should therefore use Phase-1 peaks as admission constraints but optimize Phase-3
and especially Phase-4 throughput, microbatch count, and refresh cadence.

Guidance update: coherent all-main-batch reduction is sufficient for this t01
long-prefix PLT target, while source-only caps are not. Treat this as survival
and Phase-3/4 throughput evidence, not broad PLT chunk-tolerance evidence. Before
relaunching A3 broadly, choose an upward bracket or a bounded subset using the
observed Phase-3/4 timings, refresh counts, MaxRSS, and trace-capacity margins.

Historical A3 decision status (superseded by A4 on 2026-07-09): provisional;
keep `decoder_chunk_size` scenario-pinned. A3 survival-v3 supplied the A4
anchor and proved success/error telemetry persistence, but did not itself clear
any semantics-sensitive knob.

### A4. Cardinal caste evidence (SUFFICIENT TO PROCEED 2026-07-09)

A4 is split into three independent evidence tracks so environment reproduction
cannot block the architecture decision:

Source status: the comparison metrics below were transcribed from the OSC-side
Codex readout supplied by the user. The Cardinal artifacts and generated campaign
files still need transfer to CHPC for path verification and report regeneration.
That provenance follow-up does not block interface/mechanism implementation, but
no validation profile may be promoted as a shipped runtime default until its
source artifacts and derived report are locally reproducible.

1. **Execution-caste evidence (complete, decision-bearing).** Corrected-hook
   Gemma-3-1B + GemmaScope2 PLT-small, Cardinal, fp32,
   `t01_361_s1002_g300` and the indicated 298--300 window/configurations.
2. **Environment reproducibility (deferred, non-blocking).** Ascend job
   `6260319` is optional follow-up for historical OSC environment comparison.
   It is not required to proceed and cannot broaden the Cardinal caste result.
3. **Session regression (complete for the tested window).** Per-token and
   window-reuse results below validate the first session implementation target.

Recorded A4 Cardinal metrics use the order **feature overlap / edge overlap /
weighted-edge similarity**, followed by **top64 / top256 / top1024 / top5000**:

| Comparison | Feature / edge / weighted | Top64 / 256 / 1024 / 5000 |
|---|---:|---:|
| chunk c4096/b256 vs A3 c8192/b256 | .994643 / .988269 / .987517 | 1 / .996 / .993 / .992 |
| batch c8192/b128 vs A3 c8192/b256 | .990040 / .987479 / .984281 | .969 / .984 / .989 / .991 |
| per-token g300 vs A3 | .998292 / .992627 / .992163 | 1 / 1 / .995 / .995 |
| window-reuse g300 vs A3 | 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| window-reuse vs per-token g298 | 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| window-reuse vs per-token g299 | .984737 / .973165 / .976543 | 1 / .992 / .992 / .991 |
| window-reuse vs per-token g300 | .998292 / .992627 / .992163 | 1 / 1 / .995 / .995 |

Decision: Cardinal A4 is sufficient to begin the sibling implementation. The
observed drift is validation evidence, not a guarantee of invariance and not a
license for unconstrained runtime adaptation. Frontier-margin telemetry remains
a validation diagnostic and warning only; free memory or a measured frontier
margin must never select runtime semantics.

The implementation therefore exposes three fidelity modes:

- `strict` (default): preserve the request's logical semantics and refuse with
  an actionable admission report if no semantics-preserving plan fits.
- `validated_relaxed`: permit only named semantic substitutions on an explicit
  allowlist, each tied to versioned validation evidence whose scope includes the
  request. No inference from “small drift” or frontier margins is allowed.
- `research`: explicit per-request overrides, fully fingerprinted and labeled;
  no claim of equivalence.

### A5. Promote validation evidence and calibration profiles

Project-side tooling generates calibration and comparison artifacts. Promote
approved cost/resource calibration into versioned provider profiles consumed by
the sibling; keep the A4 caste evidence separately versioned because it governs
fidelity allowlists, not cost. CHPC Granite/H200 is the current calibration
target. Cardinal/Ascend OSC records are historical; future CHPC baselines may
calibrate cost and resource envelopes but do not become matched A4 caste
evidence without a dedicated validation campaign.

A5 calibration result (2026-07-10): Granite H200 strict baselines completed for
the 1B CLT, 1B PLT, 4B PLT, and 12B PLT stack across the three canonical base
fixtures. The immutable 12B rerun completed at b64/c4096 in 5.8--6.4 hours with
38--244 GiB observed MaxRSS. Promote these observations as versioned arithmetic
fixtures, retain 600G as the conservative 12B envelope, and keep resource
repeatability analysis as a post-B2 calibration task rather than an
implementation gate.

## Phase B — Taxonomy + sibling resolver contract (COMPLETE 2026-07-10)

### B1. Knob taxonomy extension (COMPLETE)

Extend `docs/knob_api_taxonomy.md` with per-knob columns: tier, bytes-cost
formula, caste, validated-under (regime/provider topology/model family where
relevant/scenario family), and ownership (provider-declared vs scenario-declared
vs governor-derived). Populate formulas from code reading; populate caste from
Phase A evidence only. This is the governor's requirements document and the
guardrail that keeps provider semantics knobs out of memory policy.

Must capture the NNSight batch-coupling invariant from the A3 survival-v2/v3
pilots: current trace capacity is `max(source_batch_size, feature_batch_size,
logit_batch_size)`, and Phase-3/Phase-4 `compute_batch` calls reuse that cached
forward trace capacity. The taxonomy should therefore classify
`attribution_batch_size` / source batch, `feature_batch_size`,
`logit_batch_size`, and `phase1_trace_batch_size_max` as a coupled planning
family, not independent dials. Also record that `feature_batch_size` influences
Phase-4 refresh/frontier cadence, so lowering it may be semantics-sensitive
until validated under the target scenario.

### B2. Governor v0 as a pure resolver in the sibling (COMPLETE)

After A4, implement the pure resolver directly in `../circuit-tracer_chunked`;
do not build a disposable project-side resolver next to `transcoder_config.py`.
The pure function maps `TraceSemantics`, provider profile, and
`ResourceEnvelope` to a `TracePlan`. Project-generated calibration enters only
through promoted, versioned provider profiles. Fixtures:

- must reproduce the hand-tuned 1B/4B/12B stress presets within tolerance,
- must beat them where Phase A data shows they were too conservative
  (12B pilot: ~35 GiB idle VRAM, replay window 4, 8 GiB cache),
- must include synthetic provider fixtures covering cross-layer, same-layer, and
  top-k/approximate provider semantics so the resolver cannot hardcode
  Gemma/GemmaScope2/CLT/PLT names,
- must reproduce the recorded Granite baseline batch/chunk/resource envelopes
  and report the observed 12B 5.8--6.4 hour walltime range without treating the
  CHPC runs as semantic-equivalence evidence,
- must emit an explicit derived `trace_capacity` with binding reason
  (`source`, `feature`, or `logit`) and reject or warn on plans that claim to
  lower Phase-1 memory while a larger feature/logit batch still binds,
- host budget auto-discovery from the SLURM/cgroup limit,
- admission-style plan output (predicted per-tier rigid/elastic demand,
  walltime estimate) as a printable report even before anything consumes it.

Validate v0 arithmetically with synthetic and recorded profiles, then against a
small SLURM matrix before promoting resolver outputs as launch defaults. Such
coverage validates planning and resource calibration; it does not broaden A4's
semantic-caste scope.

B2 completion record: sibling `phase-b-governor-contract@0ce3f96` implements
the immutable contracts, separate semantic/execution/evidence fingerprints,
empty package-owned trusted-evidence registry, provider/dimension/cost profiles,
nested cgroup plus Slurm host discovery, per-tier estimates, coupled-batch
diagnostics, capacity-driven row-store planning, and deterministic advisory
admission reports. Validation passed 38 focused tests, 51 existing
telemetry/provider regressions, Ruff, and uv-based Pyright. No runtime path
consumes the plan yet; staged runtime integration remains a Phase E gate after
Phase D mechanism parity and the Phase C2 runtime rewrite.

## Phase C1 — First behavior-preserving sibling cleanup (COMPLETE)

Structural implementation completed at sibling `phase-b-governor-contract@0d65fba`.
Initial immutable Granite jobs `1613072` (1B CLT) and `1613073` (1B PLT)
completed with exact compact parity, unchanged peak VRAM, and improved
walltime, but their manual scenarios omitted the live incremental telemetry
flag. Project commit `fed889d` corrects the scenarios; jobs `1613108` and
`1613109` reran them from immutable snapshot
`workspace_20260710_232642_phase_c_gate_live_telemetry_rerun`. Both produced
byte-identical compact NPZs, complete matching live/final telemetry, closed
zero-error sinks, and unchanged peak VRAM. PLT met its timing threshold. CLT
reached its `32G` host-memory request and slowed materially; this is explicitly
accepted as an allocation headroom outlier, with `64G` now required for future
1B CLT validation. Phase C1 is complete and Phase D may begin.

Goal: make the tracing runtime safe to change before changing how it executes.
Phase C1 is structural only: no governor consumption, no new memory mechanism,
no changed defaults, and no semantic/physical compatibility migration. Every
landing keeps the login-safe rails green; the phase ends with an immutable
Granite parity gate.

### C1a. Attribution mega-module decomposition

Split `attribute_nnsight.py` into deep modules with cohesive ownership:

- phase orchestration and phase-specific algorithms;
- runtime lifecycle and cleanup;
- row-store access behind its existing behavior;
- replay/prefix validation and session state;
- decoder/cache access;
- provider-facing helpers;
- result assembly and artifact boundaries.

This first pass intentionally kept `attribute(...)` and current execution paths
while moving code mechanically. Phase C2 supersedes that compatibility choice
and judges the result by comprehensibility and ownership, not extraction alone.

### C1b. Deep observability modules

Remove logging/telemetry mechanics from algorithm code. Core tracing modules
should make a small number of typed, domain-level calls or use lifecycle spans;
they must not assemble event dictionaries, timestamps, memory snapshots, JSONL
records, flush policies, and duplicate human log messages inline.

The observability subsystem owns:

- versioned event schemas and typed phase/batch facades;
- monotonic sequencing, timestamps, sanitization, and bounded retention;
- phase/batch begin/done/error/cancel/cleanup spans;
- CUDA and cgroup memory sampling;
- incremental JSONL sinks and terminal flushing;
- human-readable logging adapters derived from structured events;
- no-op, collecting, and failure-injection observers for tests.

Preserve current incremental telemetry compatibility and error survival. Avoid a
generic global event bus: phase code depends on narrow observability interfaces,
while sinks depend only on event schemas.

The project supplies run-owned artifact paths and may consume emitted events;
the sibling sink is the sole owner of canonical sequencing, serialization,
incremental flushing, and terminal records. The harness must not build a second
canonical JSONL stream from the same events.

### C1c. Supporting module cleanup

Extract transcoder decoder-cache, diagnostics, fingerprints, and loaders from
math objects; keep provider topology details in provider adapters. As capacity
allows, follow with the strong small scout items (`hf_utils` pure parsing,
replacement-model adapter boundaries, graph value-object/algorithm split).
None may introduce behavior changes during the Phase C1 gate.

### Phase C1 validation gate — structural parity

After login-safe tests, Ruff, and type checks pass, create one immutable
project+sibling snapshot and run `361_base` on Granite H200 for:

- Gemma 3 1B + GemmaScope2 CLT medium;
- Gemma 3 1B + GemmaScope2 PLT small.

Use strict baseline parameters. Require exact compact graph/artifact parity,
required telemetry event families and lifecycle ordering, valid incremental
JSONL termination, and no unexplained peak-VRAM or walltime regression over 10%
versus the recorded baseline. Rerun an exceeded metric before classifying it as
a regression. Event counts or prose log lines need not be byte-identical. Phase
D does not start until this gate is recorded.

## Phase D — Explicit controls and memory mechanisms

Goal: implement and validate mechanisms before any governor selects them.
Every mechanism is directly selectable by tests/operators; the Phase B resolver
remains advisory and is not applied automatically.

Phase D is the final mechanism and knob-boundary pass before governor
integration. It has two required end-to-end outcomes:

1. eliminate the oversized Phase-1 NNSight capacity spike by allowing logical
   Phase-3/4 work to execute through a smaller physical session and physical
   microbatches; and
2. remove mandatory retained `K x N` dense storage from the extreme-case path,
   where `K` is selected rows and `N` is the active-feature universe.

This does not make tracing independent of `N`: influence, visited, rank, and
frontier vectors remain `O(N)`. The requirement is to bound dense working sets
and retained row storage, not to hide that irreducible linear floor.

### D0. Lifecycle failure integrity

Before adding mechanism branches, close the failure-path gaps exposed by the
Phase C1 review:

- perform row-store, context, terminal-event, and sink cleanup independently so
  one failure cannot skip the remaining cleanup;
- preserve and re-raise the primary tracing exception, while attaching or
  reporting cleanup failures; raise an `ExceptionGroup` only when cleanup is the
  sole failure;
- perform pure request/config validation before observer and sink creation, and
  report rejection at the API boundary without starting a run lifecycle; place
  all later work under one terminal lifecycle boundary;
- add injected failures for row-store cleanup, context cleanup, terminal-event
  emission, recorder/sink closure, cancellation, and preflight rejection.

This is the reliability prelude to Phase D, not a new phase and not permission
to change tracing semantics.

### D1. Logical/physical control split and canonical controls

- Separate decoder reduction tile/order from physical fetch/cache chunking.
- Separate frontier refresh stride/checkpoints from physical compute
  microbatches.
- Represent source, feature, and logit execution batches independently.
- Represent logical and physical fields separately in the canonical request and
  plan. The temporary Phase-D translator is test scaffolding only and is
  deleted by C2 rather than becoming a supported API.
- Preserve separate semantic and execution fingerprints.

Freeze and test these boundaries before mechanism implementation:

- logical decoder reduction tiles/order versus physical decoder fetch/cache
  chunks and contraction row tiles;
- logical Phase-4 reference frontier batches and refresh checkpoints versus
  Phase-4 physical backward microbatches;
- logical source/logit grouping versus Phase-3 physical backward microbatches;
- per-phase and per-resource replay, prefetch, residency, and cache controls.

Use direct runtime values rather than umbrella policy names:

- `nnsight_session_capacity`;
- `phase3_compute_microbatch_max_rows` and
  `phase4_execution_batch_max_rows`;
- `phase3_gradient_replay_window_layers` and
  `phase4_gradient_replay_window_layers`;
- `decoder_contraction_row_tile_size`;
- `feature_row_column_tile_size`, `influence_row_tile_size`, and
  `influence_column_tile_size`;
- `feature_row_retention = full_file | none_recompute`;
- independent feature-row cache bytes, after-write page dropping, after-read
  page dropping, storage root, and preallocation controls.

During Phase D validation, temporary mixed knobs retain a documented mapping so
the mechanisms can be compared. C2 removes those knobs and mappings. A provider
that cannot separate a requested logical and physical control reports that
capability limitation; it must not silently reinterpret the control.

### D2. Phase-1 peak-VRAM mechanisms

Make NNSight session capacity explicit and independent of logical batch sizes.
Every backward microbatch must be no larger than the session capacity. Phase 3
partitions ordered logit work into physical microbatches. Phase 4 keeps the
logical/reference frontier batches and refresh cadence, but may split an
oversized semantic batch or coalesce consecutive semantic batches into one
physical execution batch. It commits rows in canonical order and never crosses
a prepared refresh frontier. Session capacity bounds the physical execution
batch rather than the individual semantic batch.

Refactor row computation so temporary buffers use active lanes rather than the
cached session width. Prefer one reusable bounded session initially. Add
separate per-phase rebuild controls only if that cannot satisfy both phases;
any rebuild reports extra forwards, capacity, and observed forward peak.

### D3. Bounded Phase-3/4 mechanisms

- Refine Phase 4 incrementally as mechanisms touch it. The main loop should
  orchestrate cohesive operations: initialize frontier, plan refresh, plan
  batch, execute batch, commit rows, update frontier, and finalize.
- Hold evolving counters, buffers, frontier membership, and owned resources in
  one explicit `Phase4RuntimeState`-style object. Extract an operation only when
  it owns a real invariant or mechanism boundary; avoid one-method classes and
  pass-through wrappers added solely to reduce line count.
- Keep scheduler, row executor, reduction, and storage variants behind the
  corresponding operation boundary rather than multiplying branches in the
  orchestrator.
- Treat the existing file-backed `K x N` memmap as the full-retention reference.
  Page-cache controls may bound RSS, but this backend does not satisfy the
  no-full-retention gate because its logical/file shape remains `K x N`.
- Add canonical column-tiled row production and a two-dimensional influence
  solver with a `(row_start, row_end, column_start, column_end)` reader. Bound
  dense workspace by configured row and column tiles while preserving canonical
  row/column and accumulation order.
- Add a `RowRecipeLedger` no-retention backend. It retains source recipes,
  row-to-node order, denominator/fingerprint metadata, and bounded optional
  cache state, but not full feature rows. Influence refreshes replay requested
  row/column tiles; finalization replays selected rows and projects directly to
  final selected feature and non-feature columns.
- Reject bounded execution before load when a provider cannot produce ordered
  exact row tiles or deterministic replay. Never silently fall back to a full
  row allocation or full retained file.
- Generalize transfer overlap into explicit per-phase prefetch controls.
- Keep full materialization as the fast path for small fitting problems.
- Report working-set sizes and actual VRAM/host/file/disk use for each path.

### D4. Explicit sibling runtime API

Introduce provisional `TraceRequest`, `TraceResult`, `trace_one`, `trace_batch`,
and `open_session` over the explicit mechanisms so Phase D can validate them.
Phase C2 replaces their legacy-reflecting internals with the canonical domain
runtime and deletes `attribute(...)`; no dual runtime survives C2.

Typing is opportunistic and boundary-driven during D: replace `Any` when a
touched ownership or mechanism contract benefits from a protocol or concrete
type, but do not run a separate typing cleanup campaign or block mechanism work
on internal annotation completeness.

### Phase D validation gate — mechanism parity

From one immutable project+sibling snapshot, run `361_base` for both 1B CLT
medium and 1B PLT small on Granite H200:

| Case | Session and microbatches | Dense path | Required result |
|---|---|---|---|
| A | legacy-derived values | current full-file backend | reproduce the Phase C1 reference |
| B | explicit values equal to legacy | full-file backend | reference parity and correct split fingerprints |
| C | reduced session with split Phase 3/4 | full-file backend | unchanged logical checkpoints/output and lower Phase-1 peak |
| D | same reduced session | 2D column-tiled full retention | no full-width transient and parity with C |
| E | same reduced session | `none_recompute` replay | no `K x N` file/allocation and parity with C |

Also require synthetic very-large-shape tests proving D/E never attempt a
`K x N` tensor allocation and E never creates a `K x N` file, plus
fault-injected tile/replay interruption tests. Record semantic/execution
fingerprints, row/denominator/frontier hashes, compact artifact hash, per-phase
CUDA peaks, MaxRSS, maximum tile dimensions, apparent and allocated file bytes,
replay/forward/backward counts, and walltime.

- reference/default explicit configuration still matches the Phase C1 baseline;
- explicit selectors force each implemented mechanism instead of silently
  taking the fast path; envelope-driven selection remains disabled;
- Phase-1 reduced-capacity execution matches the reference and lowers both peak
  allocated and reserved VRAM, or passes a predeclared cap at least 10% below a
  repeated reference peak while the reference fails that cap;
- column-tiled execution bounds every dense transient, and no-retention replay
  avoids both full `K x N` allocation and file creation while matching the full
  path;
- semantic fingerprints remain fixed while execution fingerprints distinguish
  mechanisms;
- injected cleanup failures preserve the primary exception, attempt every
  cleanup action, group cleanup-only errors, and close terminal telemetry when
  the sink remains usable;
- `trace_one`, mixed-shape `trace_batch`, and `open_session` pass sequence,
  explicit reuse, cleanup, cancellation, and failure-recovery tests.

Bitwise compact parity is the initial target. If deterministic replay or a
canonical tiled reduction cannot be bitwise-identical, Phase D stops for an
explicit scientific decision and records row/frontier/graph drift; it does not
silently weaken the gate. Phase E does not start until both required mechanisms
are stable, directly controllable, and parity-proven under the accepted
criterion.

## Phase C2 — Cleanup Strikes Again: Atomic Tracing-Pipeline Rewrite

Phase D is closed and its accepted D/E artifacts are the frozen C2 oracle.
Canonical API, caller migration, ownership decomposition, observability
ownership, and stale-path deletion are complete. Sibling CPU-safe validation,
project Ruff, and all 181 project login-safe tests pass. Phase C2 remains open
until the immutable Granite comparison is adjudicated. Slurm state alone cannot
close it.

Goal: make the complete trace path understandable from project scenario to
sibling result before the governor is connected. This is a replacement, not
another compatibility-preserving extraction pass.

The normative design and work packages are in
`docs/tracing_runtime_rewrite_spec.md`. Binding requirements:

1. one canonical typed runtime owns `trace_one`, `trace_batch`, and
   `open_session` for every backend;
2. meaningful objects represent attribution intent, semantics, resource
   envelope, resolved plans, active features, forward sessions, row storage,
   seed attributions, frontier expansion, graph components, and results;
3. objects enforce domain invariants and must not merely contain the old flat
   arguments under a new name;
4. planning/admission, execution, mechanisms, lifecycle, observability, and
   project artifact ownership have enforceable dependency directions;
5. the project constructs typed sibling requests directly instead of invoking a
   5k-line local tracing implementation through a flat subprocess command;
6. callers and tests migrate atomically, then `attribute_nnsight.py`,
   `_attribute_impl`, generic flat `attribute(...)` routing, legacy translators,
   private helper re-exports, and obsolete project trace pipelines are deleted;
7. context, replacement-model, transcoder, project command-builder, and trace
   CLI hotspots are audited and refactored where their current responsibility
   boundaries prevent honest runtime contracts. Hiding them behind imports does
   not satisfy C2.

### Phase C2 validation gate — canonical runtime parity

- login-safe subsystem, failure-injection, architecture/dependency, lint, type,
  sibling, and project tests pass;
- repository checks find no stale project or sibling imports of removed tracing
  entry points, translators, or private re-export namespaces;
- the top-level runner visibly expresses the trace pipeline without inline
  policy resolution, telemetry serialization, or cleanup bookkeeping;
- immutable Granite H200 `361_base` 1B CLT/PLT runs through canonical
  full-retention and bounded mechanisms match the pinned pre-C2 semantic,
  graph, lifecycle, and accepted resource criteria;
- Phase E has exactly one stable plan/runner boundary to consume.

## Phase E — Staged constrained governor integration

Goal: make the governor coordinate validated Phase D mechanisms through the
Phase C2 canonical runtime, not define their behavior. Both D and C2 gates must
pass before implementation begins.

The governor is a staged constrained optimizer. It does not select one final
configuration before execution and then merely verify it. Every epoch inherits:

- logical semantics, which are never optimizer variables in strict mode;
- hard user requirements such as a forced row-store policy, cache size, batch
  bound, placement, or residency mode;
- decisions frozen by work that has already executed; and
- the latest measured resource and throughput evidence.

It re-optimizes every still-free physical mechanism. A hard requirement removes
that variable from the search domain; it does not disable optimization of the
remaining variables. If no candidate satisfies all hard constraints, refuse
with the conflicting constraints and nearest rejected candidates.

Use a deterministic lexicographic objective: (1) preserve strict semantics and
satisfy every hard constraint, (2) fit all resource and walltime budgets with
explicit safety margins, (3) minimize predicted completion time, and (4) break
ties by lower peak pressure, lower I/O amplification, and stable fingerprint
order. Do not optimize for minimum memory once a faster candidate safely fits.

1. **Pre-execution admission:** enumerate provisional mechanism families from
   closed-form estimates and hard constraints. Refuse actionably when no
   supported semantics-preserving family can fit. In the current API the
   model/provider is already constructed, so this is not a true pre-load gate.
   True pre-load refusal requires a future typed load specification at the
   model-loader boundary; the runtime must not claim that capability meanwhile.
2. **Loaded-state optimization:** measure permanent model VRAM plus
   representative encoder/decoder allocation and throughput, replace the
   corresponding estimates, then re-run the constrained search. Freeze only
   controls whose runtime state is created before or during Phase 0.
3. **Post-Phase-0 optimization:** replace estimated feature-universe size and
   distribution with the observed values. Recompute row-store, replay, and
   dense working-set costs, then re-optimize storage, encoder residency, and
   independent Phase-3/4 physical microbatches inside the remaining headroom.
4. **Phase-entry optimization:** after each preceding phase releases its grant,
   consume measured peak/throughput evidence and re-optimize still-free
   phase-local controls. Never alter a control that would invalidate existing
   state, logical frontier checkpoints, reduction order, or semantic identity.
5. **Phase transitions:** phases declare working sets, receive grants, and
   release phase/transient reservations on exit. Record selected and rejected
   candidates, binding constraints, objective score, predicted demand/time,
   actual demand/time, and prediction error for model refinement.

Freeze matrix:

| Control family | Last optimization epoch |
|---|---|
| provider/checkpoint/load placement | pre-load boundary when available |
| decoder fetch/cache ownership, source/Phase-1 scheduling, replay window, prefetch | loaded-state, before Phase 0 |
| row-store policy/placement, encoder residency | post-Phase-0, before row production |
| Phase-3 microbatch and phase-local buffers | Phase-3 entry |
| Phase-4 execution batch, row/influence tiles, phase-local buffers | Phase-4 entry |

Session capacity, Phase-1 source scheduling, Phase-3 physical microbatch size,
and Phase-4 physical execution-batch size are independent variables with
independent profile bounds and demand formulas. No umbrella physical-batch cap
may silently drive all four.

Logical semantics remain fixed. The governor may choose only mechanisms that
passed the Phase D gate; otherwise strict admission refuses.

Phase E adds the sibling-owned `ResourceEnvelope` runtime contract and makes
`trace_one`, `trace_batch`, and `open_session` consume governed plans. Phase F
owns project scenario/CLI adoption of that contract; it does not defer the
runtime envelope or governor execution itself.

Initial implementation state (2026-07-13): sibling commits `7cde998` and `5e6ec0d`
execute the first three ordered decision epochs, phase grants/releases, measured
resource samples, explicit refusal lifecycle, load-time decoder validation,
one-shot/session decoder-cache ownership, and effective storage/decoder
fingerprints. Project commit preparation carries named profiles/envelopes and
explicit physical requirements into that runtime. Checkpoint page-cache policy
is admission evidence only in the current already-loaded API; it is not labeled
as an effective live mechanism.

Gate readout (2026-07-14): full and tiled CLT/PLT arms passed graph parity, but
both forced-recompute arms timed out while still computing. The readout exposed
that the resolver is a single-candidate admission calculator rather than the
optimizer specified above, its walltime model is row-store agnostic, its bridge
unconditionally couples source capping to Phase-1 execution, and its profile
uses one physical-batch cap across independent phases. Phase E is reopened;
extending walltime without correcting selection/costing is not an acceptance
path.

Correction implementation state (2026-07-14): candidate search now covers all
distinct logical-step breakpoints down to one row, admits against concurrent
rather than summed phase-local peaks, and uses measured C2 row-policy runtime
ratios. The runtime adds Phase-3 and Phase-4 entry replans, feeds excess observed
live VRAM back as reduced phase headroom, separates user constraints from frozen
decisions, and records effective execution identity revisions. Source
microbatching is excluded from optimization until a real sequenced executor
consumes it; this avoids charging or rewarding a non-causal control.

Governor-v0.3 correction (2026-07-15): replace the remaining single-observation
profile assumptions before accepting Phase E. Profiles must separate loose
implementation safety limits from observed calibration support and evidence.
The optimizer independently searches session, Phase-1 schedule, Phase-3/4
microbatches, decoder cache, replay/prefetch, replay-tile cache, row policy,
residency, and placement, subject to hard requirements and freeze points.
Replay-tile cache is zero outside recompute; source microbatch remains
explicit/derived only because it has no sequenced executor.

Admission uses additive per-phase memory peaks and additive phase walltime.
Observed completed time replaces the matching predicted prefix; row-policy and
cache effects apply only to owning phase components. Each phase grant resets
CUDA peak statistics and release records peak allocated/reserved, ending live
allocation, duration, and operation/cache counters. The active run never refits
its own profile.

### Phase E validation gate — governed parity

For the v0.3 correction, run one roomy unconstrained 1B CLT trace and one roomy
unconstrained 1B PLT trace on Granite and require:

- strict compact artifacts match the original corrected-hook Granite baseline,
  not merely a C2-produced reference;
- runtime comparisons use the closed C2 A/D/E executions as the immediate
  performance baseline and report faster/slower ratios by phase;
- an unconstrained roomy envelope selects the fastest predicted fitting plan,
  not the lowest-memory plan;
- selected configurations are inspected against available H200 VRAM, host RAM,
  disk, and walltime, and telemetry demonstrates that safe headroom was used
  rather than left idle by an unrelated cap;
- planned and actual allocations plus every re-plan epoch are recorded;
- candidate sets, rejected reasons, binding constraints, objective scores,
  frozen/free variables, and prediction error are recorded at every epoch;
- failures/refusals preserve incremental telemetry and actionable reports.

The checked-in campaign contract is
`experiments/exact_trace_bench/phase_e_governor_gate.json`. It pins the roomy
auto envelopes and the original baseline artifact roots used as scientific
references. C2 roots remain runtime context only.

After the two-run correction gate, execute the 38-row calibration matrix in
`docs/governor_calibration_matrix.md`. It isolates the 1B session, Phase-1/3/4,
decoder-cache, replay-window, prefetch, replay-cache, row-policy, residency,
placement, and repeatability relations, then checks session scaling and
tiled-policy transfer on 4B/12B PLT. Promote a profile only after fitting causal
rows and validating held-out/larger-model rows; values inside safety limits but
outside those support ranges remain extrapolated.

## Phase F — Governed Harness Consolidation and Final Validation

The project continues to own scenarios, fixtures, campaigns, SLURM/CHPC
resource policy, immutable snapshots, artifact layout, extraction, comparison,
and scientific interpretation. Phase C2 has already migrated trace execution to
the canonical sibling API; Phase F adopts governed envelopes and consolidates
the remaining harness workflow after Phase E.

### F1. Governed scenario adapter

Map project scenarios and CHPC allocation facts into the sibling-owned
`ResourceEnvelope` and governed runtime added in Phase E. Consume streamed
events/fingerprints without duplicating sibling-owned sequencing,
serialization, and terminal flushing.

### F2. CLI and launch consolidation

Split CLI command families by domain while keeping `exact-trace-bench` stable.
Consolidate launches through the packaged CLI; demote stale scripts/templates to
compatibility wrappers or history without weakening snapshot enforcement.

### F3. Harness-local cleanup

Separate launch policy from rendering, snapshot creation from verification, and
scenario/wave policy where this reduces demonstrated complexity. Do not move
experiment interpretation into the sibling.

### Final Phase F validation gate — complete stack

First run a `361_base` smoke across 1B CLT, 1B PLT, 4B PLT, and 12B PLT. Then
run the canonical `828_base`/`361_base`/`94_base` baseline matrix when promoting
the integrated runtime. Require graph/artifact parity, complete telemetry,
semantic/execution fingerprints, plan-versus-actual resource reports,
batching/session behavior,
and immutable two-repo provenance. No default migration is complete until this
gate passes.

## Cross-cutting rules

- Every SLURM run that executes project code uses an immutable read-only
  project+sibling snapshot by default. Packaged launchers create or reuse it;
  direct `sbatch` launches must receive verified snapshot roots explicitly.
  Any live-workspace exception is labeled, justified, provenance-recorded, and
  blocks edits to both runtime checkouts until termination.
- Two-repo provenance on every SLURM run: both SHAs, dirty files, snapshot
  manifest/roots, output root, and job IDs, per `AGENTS.md`.
- Baseline-changing decisions -> `EXPERIMENTS.md`; verbose records ->
  `experiments/logs/2026-MM.jsonl`.
- No GPU/model work on login nodes; login-safe rails are the scout-listed
  pytest sets plus `ruff`/`ty`.
- Contaminated-era evidence never counts as caste validation
  (validated-under column is mandatory, not decorative).
- No model-family/checkpoint-name special cases in governor policy. New
  supported model/transcoder pairs must enter through provider metadata,
  capability flags, and mechanism rungs.
- Granite/H200 is current operational policy; OSC Cardinal/Ascend material is
  historical evidence. Ascend `6260319` is optional and non-blocking.
- A4 validation never generalizes beyond its named 1B PLT-small Cardinal fp32
  scope. Validation evidence and cost/resource calibration have separate
  versions and promotion paths.
- No git submodule. Use the editable sibling checkout locally and immutable
  project+sibling snapshots for experiment provenance.

## Migration acceptance gates

1. **C1 structural gate:** existing entry points, compact outputs, artifacts,
   and telemetry semantics survive the module split. Telemetry streams
   incrementally through success and failure, and consumers detect dropped or
   truncated streams.
2. **D control gate:** explicit logical and physical controls are independently
   selectable for mechanism validation. Physical changes alter only the
   execution fingerprint in `strict`.
3. **D mechanism gate:** Phase-1 reduced-peak and Phase-3/4 bounded paths match
   explicit reference paths, including tests that force each fallback.
4. **D API gate:** `trace_one` parity is followed by mixed-shape `trace_batch`
   and `open_session` tests for independent steps, explicit state reuse,
   298--300 window reuse, cleanup, cancellation, and failure recovery. Reuse is
   never inferred from free memory.
5. **C2 architecture gate:** one canonical typed runtime replaces the old
   argument-bag path across both repositories, removed paths have no stale
   imports, immutable CLT/PLT parity passes, and the project calls the canonical
   runtime directly.
6. **E governor gate:** governed runs match equivalent explicit D runs. Strict
   admission demonstrates a semantics-preserving degraded plan when a proven
   rung fits and actionable pre-load refusal when none fits.
7. **E evidence gate:** `validated_relaxed` accepts only named, versioned,
   scope-matched evidence; `research` overrides are explicit in request,
   result, and provenance.
8. **All GPU gates:** Granite runs record both repo SHAs/dirty state or immutable
   snapshot IDs, allocation/resource envelope, fingerprints, profile/evidence
   versions, job IDs, and output roots.
9. **F completion gate:** pass the `361_base` smoke for 1B CLT, 1B
   PLT, 4B PLT, and 12B PLT, then the canonical
   `828_base`/`361_base`/`94_base` matrix before promoting launch defaults.

## Ordering rationale and risks

- A before B: rewriting around unvalidated caste assumptions would bake the
  contaminated-era worldview into the new architecture.
- B before C: governor arithmetic and contracts are proven as a pure sibling
  resolver without coupling structural cleanup to runtime policy.
- C1 before D: the attribution mega-module and inline observability make memory
  mechanism edits difficult to review and easy to implement incompletely.
- D before C2 execution: D finishes mechanism characterization so the rewrite
  has complete behavior and failure references.
- C2 before E: the governor integrates once with a comprehensible canonical
  plan/runner boundary, not a transitional compatibility layer.
- E before F: governed project consolidation happens after sibling planning is
  validated.
- Main risks: (1) Phase 3 on 12B may be compute-bound — governor telemetry
  must be able to say so rather than promising memory wins (walltime
  estimation begins in B2 and is recalibrated in E); (2) mechanical splits touching
  12.6k-line modules risk silent behavior drift — parity runs after every
  landing where login-safe fixtures can cover it, plus Granite runs at phase
  gates; (3) retained schema/artifact readability for Track-A replay machinery
  requires explicit versioning where formats change, but old runtime APIs are
  not preserved;
  (4) overfitting the governor to the initial Gemma/GemmaScope2 evidence — use
  provider-contract fixtures and capability-based formulas from B onward.
