# Memory governor + rearchitecture execution plan

Status: active execution plan; Phase B complete, Phase C next
Date: 2026-07-03; last updated 2026-07-10
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
consumes the plan yet; Granite integration/parity remains a Phase C gate.

## Phase C — Sibling library restructure (the main rewrite)

Order follows the sibling scout's strong recommendations, with the governor
landing as the policy seam. Every landing preserves compact outputs on
canonical prompts (parity runs) and keeps the login-safe test rails green
(scout report lists them).

The sibling owns the tracing runtime and its public contract:

- value objects: `TraceRequest`, `TraceSemantics`, `ResourceEnvelope`,
  `TracePlan`, and `TraceResult`;
- provider loading, provider profiles, and promoted profile-version checks;
- the pure resolver, governor policy, execution mechanisms, and streaming
  telemetry;
- first-class `trace_one`, `trace_batch`, and `open_session` APIs, with session
  operations for sequence tracing and explicit window reuse.

`attribute(...)` remains a compatibility facade over this API; it is not the
new architectural center. Do not use a git submodule to bind the repos. Keep the
editable sibling package for development and immutable two-repo snapshots for
runs. A third shared package or plugin system is deferred until duplication is
demonstrated after this boundary lands.

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
- Logical semantics and physical execution are distinct in the request and plan:
  decoder reduction tile/order is logical while fetch/cache chunking is
  physical; frontier refresh stride/checkpoints are logical while physical
  microbatch size is execution-only. The planner may vary only the latter.
- Frontier margins are emitted as warnings/validation telemetry. They never
  create a free-memory-dependent semantic gate.
- “Never die; degrade” applies only across semantics-preserving mechanism
  rungs. In `strict`, admission may refuse actionably instead of silently
  changing logical semantics.

### C2a. Compatibility migration for conflated knobs

The current knobs conflate logical and physical behavior. Migrate them in a
versioned compatibility layer:

1. Inventory each legacy knob and split it into logical `TraceSemantics` fields
   and physical `ResourceEnvelope`/`TracePlan` fields. At minimum separate
   decoder reduction tile/order from decoder fetch/cache chunk, and frontier
   refresh stride/checkpoints from execution microbatch.
2. Translate legacy arguments deterministically into both fields and emit a
   structured deprecation event containing the translation and compatibility
   schema version. Conflicting old/new fields fail before model load.
3. Preserve the legacy translation for one documented compatibility window;
   make `attribute(...)` call the same resolver/runtime as the new APIs.
4. Compare a **semantic fingerprint** (logical fields, provider/checkpoint/hook,
   dtype, fidelity mode and evidence version) separately from an **execution
   fingerprint** (plan/profile version, physical chunks, caches, microbatches,
   ladder rungs and environment). Never use an execution fingerprint as an
   equivalence claim.

### C3. Row-store + replay locality and degradation ladders (scout #3 + spec §6)

- Row store behind a narrow module with the ladder: file-backed full
  (today) -> tiled/windowed -> recompute-on-demand; spill roots
  tmp -> scratch (capacity-first, bandwidth-second).
- All rungs bitwise-identical outputs; write-time truncation explicitly out
  of governor authority.
- Generalize prefetch into one double-buffered mechanism configured
  per-phase.
- Phase-3/Phase-4 scale target: avoid mandatory materialization of dense matrices
  whose size grows as selected rows x active features / prompt length. Large
  prompts with tens of millions of active features must degrade into bounded
  tiled/lazy/recompute modes instead of attempting TB-scale allocations. The
  governor can only choose these modes after the row-store module exposes them.

### C4. Phase-1 admission + Phase-3/4 bounded execution rewrites

Implement the mechanisms that the governor will select as it moves from a pure
resolver to a real allocator. Treat Phase 1 and Phase 3/4 differently:

- **Phase 1 admission:** reduce or split the NNSight forward trace-capacity peak
  when a prompt/provider would otherwise fail admission. Slower is acceptable;
  Phase 1 is seconds while Phase 4 dominates wall time. Candidate mechanisms:
  split/rebuilt trace sessions, narrower source/logit/feature trace families,
  and explicit fallback plans that trade extra forward work for lower peak
  reserve.
- **Phase 3/4 throughput and scale:** optimize the actual runtime bottleneck.
  The governor should choose microbatch sizes, row-store rung, prefetch depth,
  and physical fetch/cache behavior based on measured/predicted VRAM, host RAM,
  file-backed cache, and walltime. A4 shows that the legacy
  `feature_batch_size`/refresh coupling can drift; keep the legacy field pinned
  and separate logical refresh checkpoints from physical microbatch before the
  governor controls the latter.
- **Bounded dense operators:** Phase-4 refresh/frontier planning and influence
  matmuls must have streaming/tiled implementations. Full dense materialization
  can stay as the fast rung for small problems, but large projected working sets
  must automatically choose bounded modes.
- **Telemetry contract:** every bounded mode reports working-set estimates,
  actual peak allocated/reserved VRAM, cgroup anon/file pressure, row-store bytes,
  refresh counts, and walltime so the governor can recalibrate rather than act as
  a static preset table.

### C5. Transcoder runtime helpers (scout #4)

Extract decoder cache, diagnostics, fingerprints, loaders out of
`cross_layer_transcoder.py`; keep math objects central; same treatment for
PLT and other provider implementations where applicable. The provider adapter,
not the governor, owns topology-specific details such as cross-layer vs
same-layer vs top-k semantics.

### C6. Smaller strong/worth-exploring items (scout #5-#7)

As capacity allows, in this order: hf_utils pure-parsing split (strong,
small), replacement-model adapter deepening, graph value-object vs
algorithms split. None block the governor.

## Phase D — Project harness restructure (after C stabilizes)

The project owns experiment policy and interpretation: scenarios, fixtures,
campaign definitions, SLURM/CHPC resource policy, workspace snapshots and
two-repo provenance, experiment layout, extraction, comparison, and scientific
interpretation. It consumes the sibling API and streaming telemetry but does not
reimplement provider loading, planning, tracing mechanisms, or sessions.

### D1. Runner runtime adapter (harness scout #1)

Deepen "run one trace spec against the sibling runtime" out of
`full_answer/runner.py`; shard orchestration stays the caller. Do this
against the post-C sibling seam so it adapts in one place. The adapter maps
scenario plus CHPC allocation into `TraceRequest` and `ResourceEnvelope`, calls
`trace_one`/`trace_batch`/`open_session`, and persists streamed events without
buffering the full event history in memory.

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

1. New and legacy entry points produce the same semantic fingerprint for the
   same logical request; compatibility translation is deterministic, versioned,
   warned, and rejects conflicts.
2. Physical plan changes alter only the execution fingerprint in `strict`;
   canonical parity tests cover decoder fetch/cache chunking and microbatching
   independently from logical decoder reduction and frontier refresh semantics.
3. `validated_relaxed` accepts only named, versioned, scope-matched evidence;
   `research` overrides are explicit in request, result, and provenance.
4. Telemetry streams incrementally for success and failure and includes request,
   semantic, execution, provider-profile, evidence, and event-schema versions.
   Consumers tolerate unknown additive events and detect dropped/truncated
   streams.
5. `trace_one` parity is followed by mixed-shape `trace_batch` tests and
   `open_session` tests for independent sequence steps, explicit state reuse,
   298--300 window reuse, cleanup, cancellation, and failure recovery. Reuse is
   never inferred from free memory.
6. Granite SLURM smoke/baseline runs record both repo SHAs/dirty state or
   immutable snapshot IDs, allocation/resource envelope, semantic and execution
   fingerprints, profile/evidence versions, job IDs, and output roots.
7. Strict admission demonstrates both outcomes: a semantics-preserving degraded
   plan when a rung fits, and an actionable pre-load refusal when none fits.

## Ordering rationale and risks

- A before C: rewriting around unvalidated caste assumptions would bake the
  contaminated-era worldview into the new architecture.
- B2 is the first C2 landing: the governor's arithmetic is proven as a pure
  sibling resolver before it owns real allocations, avoiding a disposable
  project-side implementation.
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
