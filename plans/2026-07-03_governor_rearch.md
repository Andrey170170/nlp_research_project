# Memory governor + rearchitecture execution plan

Status: active execution plan  
Date: 2026-07-03  
Scope: sibling library `../circuit-tracer_chunked` rewrite + project harness
restructure + validation campaigns

Target design (the "how it is supposed to be" document):
`docs/memory_governor_rearchitecture_spec.md`. This plan is the "how we get
there" companion. Rearchitecture evidence base: the two scout reports,
`reports/circuit_tracer_architecture_scout.md` and
`reports/harness_architecture_scout.md` (both 2026-06-30, pre-merge — see
Phase R0).

## Problem statement

Two intertwined problems, one plan:

1. The exact-trace optimization surface is a flat bag of per-mechanism knobs
   that cannot trade resources across the VRAM / host-RAM / file-backed tiers.
   The target design replaces it with a budget-driven memory governor
   (invariant, castes, plan epochs, degradation ladders — see the spec).
2. The sibling library's core modules are shallow and entangled
   (`attribute_nnsight.py` ~12.6k lines post-merge; knob-bag `attribute()`
   interface), and the harness mirrors those internals. The scout reports
   identify the deepening candidates; the harness scout explicitly says the
   sibling library must be cleaned first.

The governor is not an add-on: it is the scout report's "exact-trace
policy/config seam" (sibling candidate #2) with a concrete job description,
so both efforts share one migration.

Coverage invariant: the rework is **provider-agnostic**. The memory governor
and speedups must apply to any model/transcoder pair already supported by the
exact/chunked provider contract, not only the Gemma/GemmaScope2 combinations
used for initial evidence. Gemma 3 12B + PLT, GPT-OSS 20B + PLT, and Llama 3.1
8B + a top-k transcoder should all use the same planner if their providers
declare the required capabilities/metadata. The governor may branch on provider
capabilities and topology; it must not branch on model-family or checkpoint-name
special cases.

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

The scout reports predate the PLT parity merge. Before Phase C relies on
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

Done when: scout reports carry a post-merge delta note and Phase C tasks can
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

Phase-A decision status: provisional. Keep `decoder_chunk_size` scenario-pinned
for the next rerun wave; do not treat it as a governor-derived performance-only
VRAM lever yet. The survival-v2 Phase-1 source cap pilot did not survive, so
there are still no long-prefix PLT compact outputs for tolerance-aware chunk
comparison. Persisted error-path telemetry is now proven and should be kept as
validation infrastructure.

## Phase B — Taxonomy + governor v0 (login-safe, project-side)

### B1. Knob taxonomy extension

Extend `docs/knob_api_taxonomy.md` with per-knob columns: tier, bytes-cost
formula, caste, validated-under (regime/provider topology/model family where
relevant/scenario family), and ownership (provider-declared vs scenario-declared
vs governor-derived). Populate formulas from code reading; populate caste from
Phase A evidence only. This is the governor's requirements document and the
guardrail that keeps provider semantics knobs out of memory policy.

### B2. Governor v0 as a pure resolver

Pure function (model config, provider profile/capabilities, scenario, hardware)
-> current knob values, next to `transcoder_config.py`. No sibling changes.
Fixtures:

- must reproduce the hand-tuned 1B/4B/12B stress presets within tolerance,
- must beat them where Phase A data shows they were too conservative
  (12B pilot: ~35 GiB idle VRAM, replay window 4, 8 GiB cache),
- must include synthetic provider fixtures covering cross-layer, same-layer, and
  top-k/approximate provider semantics so the resolver cannot hardcode
  Gemma/GemmaScope2/CLT/PLT names,
- host budget auto-discovery from the SLURM/cgroup limit,
- admission-style plan output (predicted per-tier rigid/elastic demand,
  walltime estimate) as a printable report even before anything consumes it.

Validate v0 against a small SLURM matrix (1B/4B PLT + corrected CLT) before
promoting resolver outputs as launch defaults.

## Phase C — Sibling library restructure (the main rewrite)

Order follows the sibling scout's strong recommendations, with the governor
landing as the policy seam. Every landing preserves compact outputs on
canonical prompts (parity runs) and keeps the login-safe test rails green
(scout report lists them). `attribute()` stays a compatibility facade
throughout.

### C1. Attribution mega-module split (scout #1)

Split `attribute_nnsight.py` around phase/runtime concepts: phase policy
resolvers, row-store module, replay/prefix validation, telemetry builders,
full-sequence session. Mechanical extraction first; no behavior change.

### C2. Policy seam = governor lands library-side (scout #2 + spec §5, §9)

- Ledger (tier x demand-class x lifetime), plan epochs 1-2 (admission with
  closed-form estimates + post-load calibration, profile-then-claim
  headroom pool).
- Phases become governor consumers: declare working sets, receive grants.
- Existing knobs become overrides of derived values.
- Provider runtime profiles and cost formulas consumed by capability/topology,
  not by CLT/PLT/Gemma special cases, building on the parity-merge provider
  contract.

### C3. Row-store + replay locality and degradation ladders (scout #3 + spec §6)

- Row store behind a narrow module with the ladder: file-backed full
  (today) -> tiled/windowed -> recompute-on-demand; spill roots
  tmp -> scratch (capacity-first, bandwidth-second).
- All rungs bitwise-identical outputs; write-time truncation explicitly out
  of governor authority.
- Generalize prefetch into one double-buffered mechanism configured
  per-phase.

### C4. Transcoder runtime helpers (scout #4)

Extract decoder cache, diagnostics, fingerprints, loaders out of
`cross_layer_transcoder.py`; keep math objects central; same treatment for
PLT and other provider implementations where applicable. The provider adapter,
not the governor, owns topology-specific details such as cross-layer vs
same-layer vs top-k semantics.

### C5. Smaller strong/worth-exploring items (scout #5-#7)

As capacity allows, in this order: hf_utils pure-parsing split (strong,
small), replacement-model adapter deepening, graph value-object vs
algorithms split. None block the governor.

## Phase D — Project harness restructure (after C stabilizes)

Per the harness scout (its own priority note: sibling first):

### D1. Runner runtime adapter (harness scout #1)

Deepen "run one trace spec against the sibling runtime" out of
`full_answer/runner.py`; shard orchestration stays the caller. Do this
against the post-C sibling seam so it adapts in one place.

### D2. CLI command-family modules (harness scout #2)

Split `cli.py` parser/handler groups by domain; keep the
`exact-trace-bench` entrypoint stable.

### D3. Single launch path (harness scout #3)

Consolidate launches through the packaged CLI; demote
`experiments/run_sparsification_experiment.py` and stale sbatch templates
to compatibility wrappers or archive them. Preserve run metadata and
snapshot safety.

### D4. Worth-exploring items (harness scout #4-#6)

Launch policy vs rendering split, workspace snapshot/verification split,
scenario/wave policy locality — opportunistic.

## Phase E — Dynamic epoch-3 spending

Enable post-Phase-0 re-planning (spend measured headroom on cache bytes,
replay window, prefetch depth), gated per knob on Phase A parity proofs.
Then extend to the ladder rungs (tiled row store engaged automatically when
projected size exceeds tmp budget). Acceptance criteria: spec section 11.

## Cross-cutting rules

- Two-repo provenance on every SLURM run: both SHAs, dirty files, snapshot
  status, per `AGENTS.md`.
- Baseline-changing decisions -> `EXPERIMENTS.md`; verbose records ->
  `experiments/logs/2026-MM.jsonl`.
- No GPU/model work on login nodes; login-safe rails are the scout-listed
  pytest sets plus `ruff`/`ty`.
- Contaminated-era evidence never counts as caste validation
  (validated-under column is mandatory, not decorative).
- No model-family/checkpoint-name special cases in governor policy. New
  supported model/transcoder pairs must enter through provider metadata,
  capability flags, and mechanism rungs.

## Ordering rationale and risks

- A before C: rewriting around unvalidated caste assumptions would bake the
  contaminated-era worldview into the new architecture.
- B2 before C2: the governor's arithmetic gets proven as a harmless pure
  resolver before it owns real allocations.
- C before D: the harness scout's own conclusion; harness cleanup first
  would bake in current sibling internals.
- Main risks: (1) Phase 3 on 12B may be compute-bound — governor telemetry
  must be able to say so rather than promising memory wins (walltime
  estimator is part of B2/C2); (2) mechanical splits touching
  12.6k-line modules risk silent behavior drift — parity runs after every
  landing, not just at phase ends; (3) schema/artifact compatibility for
  Track-A replay machinery — keep loaders compatible or version explicitly;
  (4) overfitting the governor to the initial Gemma/GemmaScope2 evidence — use
  provider-contract fixtures and capability-based formulas from B onward.
