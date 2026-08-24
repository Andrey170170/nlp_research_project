# Exact-trace performance optimization loop: short-prefix promotion and scaling

Status: complete through LS5. SP0-SP5 and LS0-LS3 are complete; LS4 closed
through the frozen 4B/1,024 development point and 828/512 holdout; LS5 completed
the 12B ladder through a cross-node-repeat-qualified 1,024-token selected
circuit. The qualified library point is handed to CIE. Evidence hardening and
further 12B optimization continue under the successor plan
`plans/2026-08-21_exact_trace_optimization_and_refactoring.md`.

Date: 2026-07-29

Branches: `perf/exact-trace-loop` in the project and sibling worktrees

Scope: exact-trace runtime performance only. Governor calibration, response
models, plan fitting, fidelity-scope authorization, baseline-registry changes,
and launch-default promotion are explicitly out of scope.

The canonical fidelity definitions and retention/promotion policy are in
`plans/2026-07-31_exact_trace_canonical_fidelity_taxonomy.md`. They supersede
older binary exact/bounded wording retained in the historical execution notes
below. In particular, `exact` permits promotion consideration but does not
select or mutate a default without comparative runtime, memory, compatibility,
and prompt-length evidence. Likewise, `bounded` permits code-retention review;
it does not require every passing experiment to remain selectable. Dominated
implementations are archived as patches and removed from runtime code.

Predecessor plan: `plans/2026-07-27_large_model_optimizations.md`

Starting results: `reports/2026-07-27_large_model_optimization_results.md`

Workflow reference: `docs/performance_optimization_loop.md`

Idea/code-quality registry: `docs/exact_trace_optimization_registry.md`

This plan has two ordered campaigns:

1. finish, prove, and package the short-prefix work, including the current
   approximately ten-minute 12B result; and
2. measure and improve scaling across longer prefixes, new prompts, and then
   larger models, using 1B as the fast development case.

The first campaign must produce a stable exact finalist and a mechanism-by-
mechanism evidence dossier before the second campaign changes the workload.
The scaling campaign then separates prefix-length effects from prompt effects
and transfers only accepted mechanisms from 1B to 4B and 12B.

The stage prefixes are descriptive:

| Prefix | Meaning | Current state |
|---|---|---|
| `SP` | canonical short-prefix completion and promotion evidence | SP0-SP5 complete; exact candidates admitted, selection deferred |
| `LS` | prefix-length, prompt, and model-size scaling | LS0-LS3 complete; LS4 closed through 4B/1,024 plus holdout; LS5 repeat-qualified the frozen 12B/1,024 selected circuit |

Within each campaign the number is execution order, not a governor phase or
calibration wave.

## 1. Desired outcomes

At completion, the performance workstream should be able to answer:

1. Which current short-prefix mechanisms are exact and independently
   promotable?
2. Is active-CPU encoder residency exact, repairably exact, or necessarily a
   bounded/research profile?
3. Can an exact GPU-resident feature-row tier remove the dominant Phase-4
   refresh traffic?
4. What is the fastest reproducible exact profile for the canonical short
   prefix at 1B, 4B, and 12B?
5. How do total and phase times scale with prefix tokens, active features,
   feature-row-store bytes, and model size?
6. Does the result transfer to prompts that were not used for optimization?
7. Where does full GPU row residency stop fitting, and what exact hybrid or
   file-backed fallback should take over?

The deliverables are:

- accepted sibling runtime mechanisms with focused tests;
- project-owned candidate profiles, probes, comparison gates, and campaign
  manifests;
- immutable-run evidence under declared cache and resource conditions;
- a short-prefix promotion report;
- a length/prompt/model scaling report and machine-readable measurements;
- append-only experiment-log records for accepted or rejected mechanisms; and
- a compact update to `EXPERIMENTS.md` only when an accepted result changes the
  current performance interpretation.

## 2. Boundaries and non-goals

This is not a governor campaign. Do not:

- fit or calibrate governor response models;
- promote calibration observations or fidelity scopes;
- change launch defaults as a side effect of selecting a fast candidate;
- turn measured model-specific breakpoints into hard-coded runtime branches;
- use governor stage names for the work below; or
- claim that a selected plan is a default-ready profile without a separate
  reviewed default-promotion action.

Also out of scope until new profiling changes the conclusion:

- multi-GPU execution;
- larger contraction-tile sweeps;
- ranking or frontier-planner optimization;
- Phase-5 optimization;
- revival of the old exact-range prepared-refresh cache;
- larger decoder chunks as an "exact physical" optimization; and
- broad batch-size sweeps that change endogenous frontier behavior.

The old prepared-refresh cache was rejected because exact-range reuse was poor,
miss preparation was expensive, and 0/8/32/64 GiB variants did not produce a
robust improvement. A new GPU row tier must retain canonical row data or use a
chunk-aligned/windowed ownership design. It must not reintroduce the rejected
exact-range LRU under a new name.

## 3. Starting state

### 3.1 Accepted and unresolved mechanisms

| Mechanism | Current evidence | Starting disposition |
|---|---|---|
| Provider-owned mapped selective decoder rows | byte-exact values/order in focused gates; repeated signed 4B exact gate; 12B traffic reduced from 95.316 GB to 0.910 GB; no Phase-4 decoder loads | accepted exact physical mechanism; formal packaging remains |
| Checkpoint page/working-set lifecycle | first 12B Phase-4 transition stall removed | strong engineering evidence; formal matched cold three-batch gate remains |
| Phase-scoped telemetry | detailed attribution available | implemented; paired `<2%` overhead gate remains |
| Lazy mapped b64/c4096 | repeated 12B reference | reproducible exact reference |
| Active-CPU encoder residency | fastest measured 12B exact candidate; not strict_exact | currently retained/selectable; eligible for comparative promotion after broader evidence |
| Active pinned CPU | slower than active CPU, no memory or fidelity advantage | rejected; runtime/profile branches removed and restoration patches archived |
| b128/b256 execution envelopes | slower and substantially more HBM than active CPU | rejected for the short-prefix finalist |
| Larger FP32 contraction tiles | no wall-time improvement | rejected |
| Source-layer fusion | not run after the compute proxy missed | deferred; profile again only if accumulation becomes material |

Mechanisms are promoted independently. Failure of active CPU must not block
packaging the mapped-row source or checkpoint-lifecycle result.

### 3.2 Current 12B timings

All cases below use the mapped selective-row source, b64/c4096, internal FP32,
`max_feature_nodes=8192`, `361_base`, and one H200.

| Configuration | Subprocess | Completion | Phase 0 | Phase 4 | Peak CUDA reserved |
|---|---:|---:|---:|---:|---:|
| lazy repeat 1 | 586.30s | 555.62s | 63.43s | 458.42s | 38.82 GiB |
| lazy repeat 2 | 563.27s | 539.42s | 62.59s | 443.74s | 38.82 GiB |
| active CPU repeat 1 | 509.80s | 478.01s | 86.56s | 359.68s | 38.82 GiB |
| active CPU repeat 2 | 508.67s | 483.19s | 90.67s | 359.56s | 38.82 GiB |

Active CPU removes about 83-85 seconds of lazy encoder materialization while
adding about 24-28 seconds to Phase 0. It is therefore a major performance
candidate, but one repeat differs from the lazy reference and executes 130
instead of 129 Phase-4 batches. It cannot yet be called exact.

### 3.3 Localized Phase-4 cost

The active-CPU repeats show:

| Component | Repeat 1 | Repeat 2 |
|---|---:|---:|
| Phase 4 | 359.83s | 359.72s |
| refresh total | 223.42s | 229.55s |
| partial influence solver | 213.01s | 219.90s |
| feature-row reads | 71.56s | 75.63s |
| transfer/cast/absolute value | 103.54s | comparable dominant share |
| direct accumulation/matmul | 26.58s | 26.64s |
| normalization | 10.02s | 10.71s |
| feature execution batches | 109.87s | 105.12s |
| encoder materialization inside batches | 0.14s | 0.12s |
| ranking | 0.57s | 0.54s |
| frontier planning | 0.39s | 0.38s |

The first active-CPU repeat performed 33 refreshes and:

- 35,443 direct-accumulation row subranges;
- 3,166,810 row touches;
- about 1.616 TB of cumulative row bytes;
- 35,443 feature-row read calls;
- 2,281 hits and 33,162 misses in the current 256 MiB CPU read cache; and
- only about 4.18 GB of retained logical feature-row data.

The dominant remaining short-prefix problem is repeated file/CPU row traffic
and CPU-to-GPU preparation during refresh. Ranking, planning, and finalization
are not meaningful optimization targets at this point.

### 3.4 Active-CPU first-divergence evidence

At the first differing refresh in the inspected lazy/active pair:

- the pending-frontier hash still matched;
- selected membership still matched;
- selected order differed;
- cutoff scores differed by only a few billionths; and
- later membership and execution-count differences compounded from there.

The diagnostic order is therefore raw encoder materialization, produced Phase-4
rows, refresh score inputs, and only then ranking/tie behavior. Do not begin by
changing the scheduler or adding arbitrary tie tolerances.

## 4. Campaign-wide invariants

### 4.1 Semantic pins

Within a control/candidate comparison, freeze:

- model, provider, checkpoint content, and tokenizer;
- prompt text, tokenized prefix, prefix-token count, target position, and target
  token;
- `max_feature_nodes=8192`;
- exact internal dtype FP32;
- stable row-L1 denominator behavior;
- semantic batch boundaries and refresh checkpoints;
- retention policy and compact-graph construction;
- decoder chunk regime;
- feature ordering and duplicate reconstruction; and
- all random seeds and deterministic settings.

Any candidate that changes these belongs to a separately named bounded regime,
not the exact physical comparison.

### 4.2 Exactness evidence levels

Do not use serialized NPZ byte identity as the only exactness test. Equal graph
entries may exchange serialization order even between reference repeats.
Record three independent evidence layers:

1. **Raw mechanism evidence**
   - provider values are byte-exact;
   - requested indices, duplicates, and reconstruction order are exact;
   - stored dtypes and shapes match;
   - denominators and other semantic fingerprints match.
2. **Execution evidence**
   - semantic batches and refresh checkpoints match;
   - prepared frontier membership and order match;
   - produced row hashes match in canonical row order;
   - no extra fallback, load, or recomputation path is used.
3. **Graph evidence**
   - aligned completion and target-token identity;
   - canonicalized labeled feature and edge support;
   - signed edge values or signed raw intermediates where the mechanism can
     affect sign;
   - compact-strict metrics against the declared same-regime control; and
   - explicit reporting of harmless serialization-order differences.

The existing CLI `exact` thresholds remain a compact-strict graph gate. They do
not, by themselves, prove raw or signed internal exactness.

### 4.3 Comparison scopes

Every candidate report declares both scopes when available:

- **Mechanism control:** the current mapped lazy exact profile with the same
  prompt, chunk regime, resource envelope, and code state.
- **Historical anchor:** the frozen scientific or performance artifact, used
  only when a valid artifact exists for that model/prompt/workload.

A new long-prefix or new-prompt workload will initially have only a newly
created current-source mechanism control. Do not imply historical parity where
no historical artifact exists.

### 4.4 Timing and repetition

Separate:

- setup or sidecar preparation;
- subprocess/startup;
- completion;
- Phase 0, Phase 1, Phase 3, Phase 4, and Phase 5;
- refresh planning, row read, transfer/preparation, accumulation, and
  normalization;
- feature execution and encoder materialization;
- GPU kernel/event time;
- host I/O and page/refault evidence; and
- teardown/reporting.

Use paired control/candidate ordering where feasible and reverse the order on a
second pair. Record cache state as controlled cold, controlled warm, or
uncontrolled. Never label an uncontrolled run cold.

Minimum formal repeats:

- 1B: three timing-eligible repeats for a finalist;
- 4B: two repeats;
- 12B: two repeats;
- exactness diagnostics: repeat until the first-divergence location is stable,
  with at least two completed comparisons.

A speed result is promotional only when its exactness gate passes and the
improvement exceeds noise. Use 10% Phase-4 improvement as the continuation
threshold for a new Phase-4 mechanism. Report smaller stable wins, but do not
automatically promote them as added complexity.

### 4.5 Workspace and cluster policy

All GPU/model runs are SLURM-only. Formal performance evidence uses Granite
H200 and an immutable read-only snapshot containing both the project and
sibling checkouts. Reuse one verified snapshot across runs that share an exact
source state.

Before launch, record:

- both branches, commits, and dirty-file lists;
- snapshot container, project root, sibling root, and manifest;
- model/provider/prompt and all semantic/physical controls;
- Slurm job, node, GPU, CPU, host-memory, HBM, and walltime envelope;
- cache-state protocol;
- output root; and
- probe or termination conditions.

A live-workspace launch is an explicit exception only. It must record the
reason and both dirty states, and neither runtime checkout may be edited until
the job terminates. Live runs do not become formal promotion evidence.

### 4.6 Scaling failures are architecture evidence

Every longer-prefix, larger-model, and harder-prompt rung is a deliberate
stress test of the implementation. A failure is not merely a launch obstacle:
it identifies the next missing general mechanism. Do not close a rung by
introducing a prompt/model/length-specific cap, accepting an avoidable fallback,
or widening the resource/walltime request until the point happens to pass.

Small batches, reduced session capacities, extra resources, and diagnostic
stops remain valid for localization and safe fallback. They count as a fix only
when they are part of a capability/shape-based runtime mechanism with explicit
semantic-versus-physical identity, general selection/admission behavior, and
evidence at the stress point that exposed the limitation. Stop the scaling
ladder until that mechanism is implemented and classified; do not skip the
failed rung.

## 5. Campaign SP — finish and promote the short-prefix work

`SP` means short prefix. These stages are performance-campaign stages, not
governor phases.

### SP0 — Freeze the incumbent and mechanism claim ledger

Purpose: make the current ten-minute 12B result durable and prevent one
unresolved candidate from obscuring already accepted mechanisms.

SP0 is report/schema work. Populate it from retained July 27 reports and
telemetry; it does not require a new GPU run.

Tasks:

- [x] Register the repeated mapped lazy b64/c4096 12B result as this campaign's
      same-regime short-prefix control without overwriting the frozen scientific
      baseline registry.
- [x] Add a machine-readable mechanism claim ledger to the campaign report.
- [x] Give every mechanism a stable evidence identifier and disposition:
      `exact_promotion_candidate`, `exact_opt_in`, `bounded_research`, or
      `rejected`.
- [x] Extract the current 1B/4B/12B short-prefix metrics into one compact table.
- [x] Record the unresolved telemetry-overhead and cold-transition gates rather
      than silently treating the July 27 campaign as formally complete.
- [x] Verify that the performance report distinguishes mechanism selection from
      default or baseline promotion.

Required claim-ledger fields:

| Category | Fields |
|---|---|
| Identity | mechanism ID, implementation commit, candidate profile |
| Applicability | provider capabilities, topology, dtype, shape/resource conditions |
| Semantics | scenario fingerprint, chunk regime, feature cap, refresh contract |
| Correctness | raw/order, execution, signed graph, compact comparison |
| Performance | paired timings, repeat count, cache state, speedup |
| Resources | host/HBM peaks, owned bytes, fallback/admission |
| Disposition | accepted/rejected status, rationale, remaining evidence |
| Promotion separation | mechanism, baseline, default, governor all recorded separately |

**Gate SP0:** a reader can identify exactly what is already accepted, what is
only bounded, and which formal evidence is missing without reading raw Slurm
logs.

### SP1 — Resolve active-CPU exactness

Purpose: determine whether the fastest current candidate can become the exact
short-prefix encoder placement.

Add diagnostic-only evidence at the following boundaries:

1. encoder row request:
   - requested index sequence and duplicate-aware hash;
   - source residency mode;
   - dtype, shape, device, and canonical byte hash;
2. selected encoder materialization:
   - post-selection/post-reorder index and value hashes;
   - transfer/cast operation and stream/synchronization metadata;
3. Phase-4 row production:
   - semantic batch and canonical output-row range;
   - signed FP32 row hash before row-store append;
   - denominator/normalization input hash;
4. refresh input:
   - active-row ranges and order;
   - pending-frontier hash;
   - complete score-vector hash or a collision-resistant chunked digest;
   - selected membership hash and selected-order hash;
   - cutoff score, gap, and tie count.

Diagnostics that force synchronization or hash large tensors are not timing
evidence. The harness must label diagnostic runs and exclude them from
performance comparisons.

Run the localization ladder:

1. focused synthetic active-CPU versus lazy row-selection tests;
2. 1B PLT `361_base` diagnostic comparison;
3. 1B repeat to confirm the first divergence boundary;
4. repair the earliest differing boundary only;
5. 1B exact end-to-end comparison;
6. 4B exact transfer gate; and
7. 12B exact repeat only after the smaller gates pass.

Decision branches:

- If raw encoder rows differ, repair provider materialization, dtype, ordering,
  or transfer synchronization.
- If raw rows match but produced Phase-4 rows differ, isolate the first
  computation/reduction boundary.
- If produced rows match but refresh score inputs differ, inspect row-store read
  order, accumulation grouping, and normalization.
- If score vectors match but order differs, inspect deterministic ranking/tie
  behavior without adding approximate score tolerances.
- If exactness requires removing the measured speed benefit, retain active CPU
  as `bounded_research` and keep lazy mapped as the exact finalist input.

**Gate SP1:** repeated 1B and transferred 4B evidence identifies and repairs the
first differing boundary, followed by two 12B comparisons with identical
semantic batches, refresh membership/order, signed canonical graph, and target
token. Otherwise active CPU remains bounded/research with a documented first
divergence and no exact-promotion claim.

### SP2 — Exact GPU-resident feature-row tier

Purpose: remove repeated file/CPU-to-GPU traffic from the dominant Phase-4
refresh path while preserving the canonical file-backed contract.

#### SP2.1 Design

Add a sibling-owned GPU-resident hot tier around the existing feature-row
store:

```text
Phase-4 row commit
  -> canonical signed FP32 row and canonical row index
  -> durable/reference file-backed append
  -> admitted GPU-resident append

Refresh read
  -> GPU tier when the complete requested canonical range is resident
  -> otherwise exact file-backed path
  -> identical returned values, shape, and requested order
```

The first implementation should be full-residency, not an LRU:

- calculate required bytes as canonical rows times active-feature width times
  element size, plus explicitly measured allocator/metadata overhead;
- admit only under an explicit experimental HBM budget and safety margin;
- allocate from actual shape/capability information, never a model-name branch;
- retain signed canonical FP32 values;
- preserve append and read order, including partial final ranges;
- keep file backing for durability, recovery, and exact fallback;
- rebuild or refuse the GPU tier explicitly after resume;
- never silently mix incomplete GPU data with file reads; and
- record admission, owned bytes, high-water mark, read hits, fallbacks, and
  refusal reason.

The performance campaign may use an explicit byte budget, but it must not call
that budget a calibrated governor policy.

#### SP2.2 Selectable execution and bounded-HBM scaling

The 12B SP2 characterization established two useful fidelity lanes and one
scaling limit. Preserve them as explicit, inspectable modes over the same
canonical signed row-store contract:

```text
canonical signed file/host rows
  -> cpu_exact
  -> cpu_prepared
  -> cuda_full
  -> cuda_windowed
```

- `cpu_exact` preserves the incumbent signed CPU preparation, layout, and
  reduction path.
- `cpu_prepared` prepares absolute host rows once and is bounded until an exact
  layout-preserving pipeline passes signed graph evidence.
- `cuda_full` retains the complete signed matrix in HBM and performs absolute
  value, normalization, and influence accumulation on CUDA.
- `cuda_windowed` retains canonical signed rows on the host, stages
  chunk-aligned bounded windows into HBM, and uses the same CUDA influence
  consumer as `cuda_full`.

Keep mode selection separate from byte admission. `cuda_full` must retain the
existing complete-allocation budget and safety gate. `cuda_windowed` must use
an explicit staging-byte budget, allocate a bounded number of complete row
windows, and fall back atomically when even one legal window cannot fit. An
`auto` selector may choose `cuda_full`, then `cuda_windowed`, then `cpu_exact`,
but it must report the resolved mode and reason.

The first windowed implementation may use synchronous transfers to establish
correctness and scaling. The performance implementation should use reusable
pinned host staging and two CUDA buffers so transfer of window `N+1` can
overlap computation of window `N`. Coalesce stable canonical ranges rather
than caching exact active-row requests. Preserve fixed processing order and
measure any change caused by window geometry under the bounded signed-graph
gate.

Required mode telemetry:

- requested and resolved execution mode;
- full-residency and window-staging required/admitted bytes;
- host, H2D, resident-hit, and fallback bytes;
- staging window count, rows, calls, transfer time, and synchronization time;
- peak owned HBM and pinned-host bytes; and
- CPU preparation, CUDA preparation, normalization, and accumulation time.

At the current 12B shape, full residency owns about 4.18 GB. Required bytes
scale as row capacity times active-feature width times element size: roughly
linearly with prompt length at a fixed output-node cap, and multiplicatively
when both axes grow. Test `cuda_windowed` on the current 12B workload by
artificially constraining staging/residency budgets before requiring a longer
prompt.

#### SP2.3 Focused validation

- [ ] append/read equality for empty, partial, full, duplicate, non-monotonic,
      and cross-tile ranges;
- [ ] exact canonical row order;
- [ ] exact direct and two-dimensional influence-solver results;
- [ ] capacity refusal before allocation;
- [ ] allocation-failure cleanup and exact fallback;
- [ ] checkpoint/resume rebuild or explicit refusal;
- [ ] no leaked GPU allocations after normal completion or injected failure;
- [ ] telemetry/schema coverage for admitted, hit, fallback, and refused paths;
- [ ] file-backed reference path unchanged when the feature is disabled.

#### SP2.4 Performance ladder

1. login-safe synthetic solver benchmark using generated tensors only;
2. 1B PLT `361_base` exact full run;
3. repeated 1B control/candidate timing;
4. 4B exact transfer run;
5. 12B transition probe with a few refreshes;
6. completed 12B exact run; and
7. repeated 12B finalist timing.

Use lazy encoder placement first to isolate row-store behavior. Compose with
active CPU only after SP1 independently passes exactness.

Required telemetry:

- logical and physically allocated row-store bytes;
- row count and active-feature width;
- GPU-tier admission budget, safety margin, and result;
- append bytes/time;
- file mirror bytes/time;
- refresh ranges/bytes served from GPU and file;
- avoided CPU reads and avoided H2D bytes;
- residual absolute-value, normalization, and accumulation time;
- CUDA allocated/reserved and external framebuffer peaks; and
- cleanup/rebuild/fallback outcomes.

Continuation thresholds:

- exact raw, execution, and signed graph evidence;
- at least 10% Phase-4 improvement on repeated 1B runs;
- no material regression in row production or Phase 0;
- no unsafe HBM pressure or unexplained allocator growth; and
- a projected 12B store that fits with the declared safety margin.

**Gate SP2:** the accepted candidate is exact, capability/shape-based, falls
back exactly, and produces a repeated Phase-4 speedup. A short-prefix 12B
promotion additionally requires two complete artifacts and an end-to-end
improvement outside paired timing noise.

### SP3 — Re-profile and choose remaining non-batch work

Purpose: optimize the bottleneck that remains after SP2 rather than continuing
the old profile blindly.

Produce a new Phase-4 accounting table whose major categories sum to within 5%
of measured Phase-4 wall time. Choose at most one next mechanism from:

1. **Chunk-aligned/windowed GPU row residency**
   - pursue only when the full store is exact and useful but does not fit the
     next prefix/resource envelope;
   - key residency on stable canonical chunks/windows, not exact row ranges;
   - preserve a deterministic exact file-backed fallback.
2. **Refresh source-pass fusion**
   - pursue only if direct accumulation/launch/scatter work becomes at least
     10% of Phase 4 after I/O removal;
   - treat changed accumulation grouping as requiring raw/signed validation.
3. **Normalization traffic reduction**
   - pursue only if normalization becomes a material share;
   - preserve the stable row-L1 denominator baseline.
4. **Active-CPU Phase-0 construction**
   - pursue only if active CPU passes SP1 and its extra Phase-0 cost is at least
     5% of end-to-end finalist time.
5. **Mapped-row Phase-0 gather/layout**
   - pursue only if cold/warm evidence shows the mapped gather itself remains a
     material, repeatable cost after cache attribution.

Explicitly do not spend runs on ranking, frontier planning, Phase 5, larger
contraction tiles, b128/b256, or the old prepared-range cache unless the new
profile demonstrates that their bottleneck has returned.

**Gate SP3:** one profile-driven next mechanism either passes its own exact
10%-of-target-phase continuation gate or is rejected before a full 12B run.
SP3 may close with no new implementation if SP2 leaves no sufficiently large
and safe opportunity.

### SP4 — Close formal telemetry and lifecycle evidence

Purpose: finish the two formal gates left unresolved by the July 27 campaign.

#### SP4.1 Telemetry overhead

Run paired 4B no-policy-change controls with detailed telemetry on and off:

- same immutable snapshot;
- same cache protocol and resource envelope;
- reversed order pair;
- identical compact/signed result;
- event counts and sink status recorded; and
- completion and phase deltas reported.

**Gate:** telemetry overhead is less than 2% in the paired mean. If not, reduce
diagnostic event volume or move expensive hashes to diagnostic-only mode.

#### SP4.2 Checkpoint lifecycle

Create a matched, attributable cold protocol using job-private staged files or
another explicitly exclusive and recorded cache identity. Compare:

- the reference lifecycle/fallback;
- mapped selective rows with lifecycle release/advice; and
- at least the first three Phase-4 batches with CUDA-event and OS/cgroup
  evidence.

Never use node-global `drop_caches`. The old two-batch artifact remains
historical context but cannot satisfy this gate.

**Gate:** the current mechanism shows the expected owned-page/reload behavior,
completes three matched transition batches, preserves exactness, and does not
harm warm steady-state timing.

### SP5 — Compose and formally promote the short-prefix finalist

Candidate composition order:

1. mapped selective decoder rows;
2. accepted checkpoint lifecycle;
3. lazy encoder or SP1-passing active CPU;
4. SP2 GPU row tier; and
5. at most one SP3 mechanism.

Do not validate all combinations. Each mechanism first passes an isolated
gate, then one composed finalist is selected.

Formal matrix:

| Model/provider | Prompts | Minimum repeats | Role |
|---|---|---:|---|
| 1B PLT | `828_base`, `361_base`, `94_base` | 3 for `361_base`, 1 for other prompts | exact regression and prompt breadth |
| 1B CLT | `361_base`; add `828_base` if touched path is provider-general | 1-2 | cross-provider regression |
| 4B PLT | `361_base` plus one holdout prompt when available | 2 canonical, 1 holdout | model transfer |
| 12B PLT | `361_base`; one holdout only after admission | 2 canonical | stress and promotion evidence |

Before launching this matrix, add either the minimum manifest support described
in LS0 or explicit fixed PLT prompt-breadth cases to the performance harness.
This login-safe harness work may start during SP and does not begin the scaling
experiment campaign.

Every formal result includes:

- immutable dual-repo snapshot provenance;
- exact semantic and workload fingerprints;
- raw/order, execution, and signed graph evidence;
- cache-state classification;
- paired/repeated phase and completion timing;
- resource peaks and row-store byte accounting;
- explicit fallback/admission outcomes; and
- mechanism-by-mechanism disposition.

Promotion products:

- `reports/YYYY-MM-DD_short_prefix_performance_results.md`;
- machine-readable performance and mechanism evidence;
- append-only `experiments/logs/YYYY-MM.jsonl` records;
- compact `EXPERIMENTS.md` update if the current interpretation changes; and
- a separately reviewable recommendation for any launch-default change.

**Gate SP5:** the exact short-prefix finalist is reproducible at 1B/4B/12B,
the ten-minute 12B achievement and all accepted mechanisms have durable formal
evidence, and bounded candidates are visibly separated from exact mechanisms.

## 6. Campaign LS — length, prompt, and model scaling

`LS` means length scaling. Start LS only after SP5 freezes the exact short-
prefix finalist, except that login-safe harness work and fixture planning may
begin earlier.

### LS0 — Add a campaign manifest and prepare workloads

The current `exact-trace-perf` CLI has fixed suites and does not express an
arbitrary prompt/prefix matrix. Extend the project harness with a typed
performance-campaign manifest. It should identify:

- campaign and workload IDs;
- fixture catalog and immutable fixture fingerprints;
- model/provider case;
- prompt family and development/holdout role;
- trajectory ID;
- exact prefix-token count and prefix-token hash;
- target position/token identity;
- expected operational class;
- candidate/control profiles;
- probe/full-run disposition;
- resource and stop envelopes; and
- comparison/evidence policy.

Do not replace the existing fixed suites immediately. Preserve them as short-
prefix regression aliases while the manifest path is tested.

#### LS0.1 Length ladder

Use one frozen prompt and deterministic trajectory to construct prefixes near:

- 124 tokens;
- 256 tokens;
- 512 tokens; and
- 1,024 tokens.

Record actual token counts rather than assuming the requested label. Each
prefix is a distinct workload with its own target token and prefix hash. The
same prompt/trajectory isolates length better than unrelated prompts, even
though the target position changes.

The committed fixture catalog currently reaches only about 248 tokens. Generate
longer deterministic trajectories in a SLURM job; do not load the model on a
login node. Do not manufacture long contexts by undeclared arbitrary text
concatenation.

#### LS0.2 Prompt-generalization matrix

At selected length bands, use:

- one development prompt used for optimization;
- at least two held-out prompt families; and
- current canonical prompts plus newly prepared GSM8K prompts where suitable.

Prefer three prompts per representative length band if resource cost permits.
Match prompts approximately by actual token length, then report active-feature
and row-store differences rather than claiming token matching makes them equal.

**Gate LS0:** all workloads have immutable prompt/trajectory/prefix/target
fingerprints, development versus holdout roles are frozen before tuning, and
the campaign can be listed/dry-run without loading a model.

### LS1 — Establish the current-source 1B reference curve

Use the SP5 exact finalist as the candidate and mapped lazy exact as the
mechanism control where those differ.

Order:

1. 124-token full reference;
2. 256-token full reference;
3. 512-token early probe, then full run if admitted;
4. 1,024-token early probe, then full run if admitted; and
5. repeat any surprising point before interpreting the curve.

The early probe should:

- complete Phase 0 and resource accounting;
- run a declared first-N Phase-4 batches and refreshes;
- report active features, unique/occurrence decoder rows, planned `K x N`
  bytes, refresh bytes/time, Phase-1 peak HBM, host charge, and CUDA peaks;
- project completion time with an explicit uncertainty range;
- produce `probe_completed`, not an ordinary successful trace result; and
- release all runtime resources and close telemetry.

Stop before a full run when:

- projected HBM exceeds the declared safety envelope;
- host charge approaches the Slurm hard limit;
- projected walltime exceeds the application/Slurm limit;
- required row-store capacity is refused without an exact fallback; or
- early telemetry shows a correctness/fallback violation.

Historical long-prefix Cardinal results are priors only:

- old 1B, 424-token target: about 857 seconds total and 745 seconds Phase 4;
- old 4B, 424-token target: about 5,558 seconds total and 5,221 seconds Phase 4.

They do not replace current-source Granite controls.

**Gate LS1:** the 1B reference curve contains exact completed artifacts for
every admitted length, explicit probe outcomes for refused lengths, and enough
telemetry to explain time in terms of tokens, active features, and row-store
bytes.

### LS2 — Optimize the 1B length curve

Use only the development prompt for iteration.

Storage/residency ladder:

1. full GPU-resident row tier when the entire canonical store fits;
2. chunk-aligned/windowed GPU residency when full residency does not fit;
3. file-backed exact fallback.

At every length, compare against the same current-source workload control.
Do not extrapolate a 124-token speedup to 1,024 tokens without running the
longer case.

Profile-driven candidate selection:

- if row read/H2D remains dominant, improve residency/layout;
- if accumulation becomes dominant, consider one source-fusion design;
- if Phase 1 becomes the capacity limit, isolate physical session capacity
  without changing later semantic batching;
- if Phase 0 dominates, revisit active encoder construction or mapped-row
  layout; and
- if no component offers at least 10% target-phase opportunity, stop optimizing
  that length and record the envelope.

Use small-model iteration for implementation speed, but keep the mechanism
capability/shape-based and test large synthetic shapes on the login-safe path.

**Gate LS2:** the 1B candidate is exact at all admitted development lengths,
has repeated wins outside timing noise, and falls back predictably when full
GPU residency stops fitting.

### LS3 — Validate prompt generalization at 1B

Freeze the LS2 candidate before running holdouts.

For each selected length band:

- run the development prompt once as a contemporaneous control;
- run at least two held-out prompts;
- record actual tokens, active features, decoder-row counts, logical row-store
  bytes, refresh counts/bytes, phase times, and resource peaks; and
- compare per-prompt results, not only the aggregate mean.

Acceptance:

- exactness on every admitted holdout;
- no prompt-specific fallback or memory failure hidden by the average;
- directionally consistent speedup;
- residual runtime variation explainable by measured workload variables; and
- no tuning after observing holdout outcomes except to reject the mechanism or
  define an explicitly narrower applicability scope.

**Gate LS3:** the mechanism generalizes across held-out prompts or is assigned a
precise, evidence-backed applicability boundary.

### LS4 — Transfer the scaling finalist to 4B

Run:

1. canonical short prefix;
2. one mid-length development prefix;
3. the longest resource-admitted development prefix; and
4. at least one held-out prompt/length combination.

Begin each new length with the early probe. Reopen optimization only if the 4B
profile shows a different component consuming at least 10% of the relevant
phase or end-to-end time. Do not repeat a broad parameter sweep merely because
the model size changed.

**Gate LS4:** exact 4B transfer, completed artifacts for admitted cases,
credible resource projections, and either preserved speedups or a documented
model-size boundary.

### LS5 — Transfer the scaling finalist to 12B

Admission prerequisites:

- SP5 exact 12B short-prefix finalist;
- LS2 exact 1B length result;
- LS4 exact 4B transfer;
- canonical selected-config, required-mode, resource-policy, and execution-
  resolution controls with focused propagation and failure tests;
- projected row-store, HBM, host, and walltime fit;
- immutable dual-repo snapshot; and
- successful abbreviated 12B probe.

Run the minimum matrix needed to establish the envelope:

1. canonical short control;
2. one mid-length development prefix;
3. longest admitted development prefix; and
4. one held-out prompt only when the first three results justify the cost.

Use two repeats for any 12B result proposed as a stable performance claim.

**Gate LS5:** exact completed 12B artifacts for admitted workloads, explicit
probe/refusal records elsewhere, and a measured rather than extrapolated
large-model scaling boundary.

#### Ordered 4B-to-12B feasibility and scoped freeze gate

The 4B finalist is now `strict_exact` through the 512-token development rung.
The next length rung is therefore 4B/1,024, not a direct jump to 12B/1,024.
Obtaining a reusable 12B/1,024 point remains the eventual operational goal, but
it does not override the prefix/model scaling ladder or justify long queue and
runtime requests that hide an impractical configuration.

First land and test the config-selection and resource-control slice:

1. one canonical selected-config/launch record consumed by local and Slurm
   rendering;
2. runtime-level `preferred | required` mechanism selection that rejects
   `auto + required` and fails at admission mismatch, copy/append failure, or
   fallback use;
3. separate planning envelope, actual scheduler request, and runtime
   `off | measure_only | enforce` policy; and
4. machine-readable requested-to-effective field deltas with resolution stage,
   reason, and semantic/fidelity/physical classification.

Use the controls first at the frozen 4B/1,024 workload. Run a deterministic,
preheated, required-`cuda_windowed`, `measure_only` transition probe in a
one-hour full-H200/400-GiB job. Probe enough physical batches to observe more
than initial admission and one refresh cycle. Audit mechanism resolution,
fallback/copy counters, active rows, HBM, anonymous/file host charge, and a
conservative projection. Admit the full candidate and matched `cpu_exact`
control only when each is projected to finish within a two-hour job. Classify
the pair before advancing the model-size axis. Run the frozen 828/512 holdout
before claiming broad LS4 transfer; it may overlap preparation of the first
12B rung but cannot be omitted from LS4 closure.

Job `1741690` completed that launch attempt far enough to expose the next
architectural boundary. Prepared-bundle validation, full-H200 admission,
preheat, model load, and Phase 0 succeeded, then Phase 1 filled one H200 to
142,553 of 143,771 MiB before failing. The current NNsight runtime constructs
the forward graph by expanding the identical prefix across the full session
lane capacity; later Phase-3/4 physical batches may be smaller than that
capacity but cannot grow beyond it. Reducing the shared session capacity is
therefore a useful survival probe and fallback, not the general closure for
this rung. The primary exception was additionally masked by unsafe NNsight
exception formatting and must be preserved correctly before the next GPU test.

The single-forward multi-VJP Phase-1 mechanism is now implemented as the opt-in
`single_forward_batched_vjp` engine. It retains one physical forward lane,
groups logical cotangents by source layer, uses batched autograd VJPs, restores
canonical row/tape order, and leaves `duplicated_lanes` as the unchanged
default. Login-safe tensor parity, realized-NNSight lifecycle compatibility,
selection propagation, failure cleanup, and architecture guards pass. This is
not yet model/GPU acceptance. Qualify it through the frozen 4B/256 then 4B/512
candidate-probe and matched-pair ladder, requiring strict semantic/graph parity
plus measured Phase-1 HBM reduction, before a 4B/1,024 rerun. A saved-tensor
CPU-offload prototype remains secondary only if the batched engine leaves a
measured residual activation wall and demonstrates bounded transfer traffic.

After 4B/1,024 is classified, transfer the unchanged mechanism through ordered
12B development rungs: canonical short, 256, 512, then 1,024 tokens. Existing
12B short-prefix measurements may inform projections but do not admit the
current required-`cuda_windowed` configuration. For every new 12B rung:

1. freeze the deterministic prefix and target plus one immutable dual-repo
   workspace;
2. begin with a preheated required-mode transition probe in a job no longer
   than one hour;
3. compare a conservative full-run projection against a two-hour operational
   ceiling;
4. run and classify the candidate and matched control before admitting the next
   length; and
5. stop at the first unsafe, fallback, or over-two-hour point and optimize only
   its measured binding component before resuming the ladder.

The cancelled 12B/1,024 32-batch `600G/8h` job `1740811` consumed no runtime.
Keep its immutable bundle as superseded preparation evidence; it is not an
admitted ladder result and must not be relaunched unchanged.

The downstream freeze card must include immutable two-repo provenance,
workload hashes, canonical fidelity, requested/resolved mode, fallback/copy
counters, allocation and measured resources, cache protocol, and exact scope.
Passing the ordered gate may freeze a narrow 12B development point for reuse by
the Llama 8B top-k transcoder tracing project. It does not establish arbitrary
prompts, complete LS5, or mutate defaults.

### LS6 — Publish the performance envelope

Produce plots/tables for:

- total, Phase 0, Phase 1, Phase 3, Phase 4, and refresh time versus prefix
  tokens;
- the same times versus active-feature count;
- Phase-4 and refresh time versus logical row-store bytes;
- cumulative refresh bytes and avoided H2D bytes;
- full-GPU, windowed, and file-backed residency regions;
- peak CUDA allocated/reserved, external framebuffer, host RSS/anonymous/file
  charge, and cgroup peak;
- prompt-to-prompt residuals within length bands; and
- 1B/4B/12B transfer ratios.

Fit simple descriptive curves only to summarize the measured campaign. These
curves are performance-report artifacts, not governor calibration bundles.

The final report must state:

- fastest exact profile by measured operating region;
- exact fallback outside each region;
- development and held-out prompts;
- confidence/repeat count and cache protocol;
- unresolved anomalies;
- rejected mechanisms; and
- which future run would most reduce remaining uncertainty.

**Gate LS6:** another researcher can select a measured exact profile or fallback
from the report without interpreting internal stage letters or consulting
governor calibration artifacts.

## 7. Ownership and implementation seams

| Work | Owner | Primary seams |
|---|---|---|
| Mechanism ledger and campaign manifests | project | `src/nlp_research_project/exact_trace_bench/perf_cli.py`, report schemas |
| Active-CPU diagnostic evidence | sibling observability/runtime | encoder materialization context, Phase-4 row commit, refresh telemetry |
| GPU feature-row tier | sibling Phase-4 storage | `row_store.py`, `phase4_storage.py`, influence solver read interface |
| Exact solver validation | sibling | `graph.py`, focused row-store/solver tests |
| Profiles, probes, gates, immutable launches | project | exact-trace bench CLI/runner and SLURM packaging |
| Long fixture/trajectory preparation | project | fixture-prep/full-answer harness |
| Scaling aggregation and plots | project | performance report/extraction modules |
| Experiment provenance | project | `experiments/logs/YYYY-MM.jsonl`, compact `EXPERIMENTS.md` update |

Keep mechanism, telemetry, harness, and promotion decisions in separate commits
where practical. Review both repository diffs before every immutable GPU gate.

### 7.1 Immediate source map

Start SP0 and the project-side reporting work in:

- `src/nlp_research_project/exact_trace_bench/perf_cli.py`;
- `tests/test_exact_trace_perf_cli.py`;
- `docs/performance_optimization_loop.md`; and
- the eventual short-prefix report under `reports/`.

Start SP1 in the sibling at:

- `circuit_tracer/attribution/context_nnsight.py`;
- `circuit_tracer/attribution/nnsight/phases/phase4_batches.py`;
- `circuit_tracer/attribution/nnsight/phases/phase4_storage.py`;
- `circuit_tracer/attribution/nnsight/phases/phase4_diagnostics.py`; and
- the observability event/recorder modules.

Start SP2 in the sibling at:

- `circuit_tracer/attribution/nnsight/row_store.py`;
- `circuit_tracer/attribution/nnsight/phases/phase4_storage.py`;
- `circuit_tracer/attribution/nnsight/phases/phase4_influence.py`;
- `circuit_tracer/graph.py`;
- `tests/test_nnsight_phase4.py`;
- `tests/test_partial_influences.py`; and
- `tests/test_phase4_resource_architecture.py`.

The project path above is relative to this repository. Sibling paths are
relative to `../circuit-tracer_chunked`.

## 8. Evidence artifact layout

Use an external scratch root organized as:

```text
exact_trace_bench/granite/
├── smoke/performance_optimization/<run-id>/
├── baseline/performance_optimization/<run-id>/
├── sweep/performance_optimization/<run-id>/
├── long_trace/performance_optimization/<run-id>/
└── analysis/performance_optimization/<run-id>/
```

Each campaign root should contain:

```text
campaign_manifest.json
snapshot_manifest.json
mechanism_claims.json
configs/
candidates/
comparisons/
probes/
resource_summaries/
performance_report.json
performance_report.md
```

The report schema should make these independent:

- runner success;
- probe/full outcome;
- raw mechanism exactness;
- execution-order exactness;
- graph exactness;
- performance gate;
- resource/admission gate; and
- overall mechanism disposition.

A missing field fails the corresponding evidence gate rather than becoming
implicit success.

## 9. Stop/go rules

- Stop at the first failed smaller-model or diagnostic gate.
- Do not spend a full 12B run on a mechanism that failed 1B exactness.
- Do not spend a full long-prefix run when the early probe refuses capacity or
  walltime.
- Do not combine two unproven mechanisms in the same first experiment.
- Repeat surprising results before redesigning around them.
- Do not treat compact bounded parity as exact semantics.
- Do not treat an uncontrolled cache state as a cold comparison.
- Do not let a performance result change a baseline, governor bundle, fidelity
  authorization, or launch default implicitly.
- Preserve rejected mechanisms and their evidence so they are not repeated
  without a newly measured bottleneck.
- Keep multi-GPU deferred until a single-H200 workload is compute-saturated or
  its exact admitted working set genuinely cannot fit.

## 10. Initial execution checklist

Start here:

1. [x] Add this plan to the active roadmap and documentation index.
2. [x] Create the SP0 mechanism claim-ledger schema and populate it from the
       July 27 report.
3. [x] Add diagnostic-only encoder and Phase-3/early-Phase-4 evidence for SP1.
4. [x] Reproduce the active-CPU divergence on 12B PLT `361_base`, as directed
       for the current fast workload.
5. [x] Decide active CPU's exact/bounded disposition from the earliest differing
       boundary.
6. [x] Define the GPU feature-row-tier interface and exact capacity estimator.
7. [x] Implement focused append/read/fallback/cleanup tests.
8. [x] Run SP2 control/candidate variants from immutable snapshots (12B at the
       user's direction rather than the planned 1B entry case).
9. [x] Re-profile before selecting any SP3 work.
10. [x] Close SP4 telemetry-overhead and cold-transition evidence.
11. [x] Compose and run the SP5 formal short-prefix matrix (promotion gate
        failed; bounded and canonical paths separated).
12. [x] Extend the harness with the LS0 campaign manifest and dry-run listing.
13. [x] Prepare deterministic 256/512/1,024-token trajectories in SLURM.
14. [x] Run the LS1 1B reference/probe ladder.
15. [x] Optimize only the measured 1B scaling bottleneck, then freeze it.
16. [x] Validate held-out prompts before 4B/12B transfer and record the
        narrower arbitrary-prompt applicability boundary.
17. [x] Publish the short-prefix report; scaling report remains an LS deliverable.
18. [x] Transfer the frozen finalist through the admitted 4B/129, 256, and 512
        development rungs with canonical comparisons.
19. [x] Land one canonical selected-config/launch record and its provenance.
20. [x] Land first-class `preferred | required` mechanism selection with
        fail-closed admission, append/copy, and fallback behavior.
21. [x] Separate planning envelope, scheduler allocation, and runtime
        `off | measure_only | enforce` resource policy with durable telemetry.
22. [x] Persist machine-readable requested-to-effective resolution deltas and
        complete focused propagation and failure-injection tests for items
        19-21.
23. [x] Prepare and run the preheated, required-mode 4B/1,024 candidate
        transition probe in a one-hour full-H200/400-GiB job; retain its Phase-1
        capacity refusal as diagnostic evidence, not a completed rung.
24. [x] Qualify the implemented opt-in single-forward batched-VJP engine at
        frozen 4B/256, then 4B/512, with preheated required-mode probes, matched
        pairs, strict graph comparison, Phase-1 HBM, runtime, group/autograd
        telemetry, and failure-stage review. The login-safe implementation and
        architecture portion is complete; this item closes only on GPU evidence.
25. [x] Rerun and classify frozen 4B/1,024 with the qualified mechanism. Do not
        close this item with only a workload-specific legacy session cap.
26. [x] Validate the frozen 828/512 holdout before broad LS4 closure.
27. [x] Freeze the 12B short, 256, 512, and 1,024 development ladder with no
        per-job operational ceiling above two hours.
28. [x] Probe, complete, and classify 12B short before admitting 256.
29. [x] Probe, complete, and classify 12B/256 before admitting 512.
30. [x] Probe, complete, and classify 12B/512 before admitting 1,024.
31. [x] Probe, complete, and classify 12B/1,024; repeat and freeze it only when
        exact, mechanism-correct, resource-safe, and practical below two hours.

Items 24-31 are complete. Jobs `1832086` and `1834396` repeat-qualified the
selected 12B/1,024 compact circuit and resource envelope in the frozen dynamic-
row, single-forward batched-VJP regime; auxiliary typed error buckets remain
bounded rather than exact. The immediate work now runs in parallel: CIE validates
the tagged library point directly against original upstream on Llama 8B through
prefix 500, while this worktree hardens graph/timing/device evidence and tests
`cuda_full`, normalization, and one measured feature-dataflow seam in separate
arms. Gemma 27B remains an exploratory ladder only after the next stable 12B
optimization point.

Items 19-22 landed on 2026-08-07 across the project harness and sibling
runtime. The focused gates cover launch/config fingerprints, per-trace config
selection, scheduler request versus allocation, measure-only behavior,
required-mode propagation, admission/allocation/copy/fallback failures, and
post-admission/post-Phase-4 effective identity. Project tests passed 94 focused
cases and the full suite passed after relocating the preheat helpers into the
Slurm subsystem; sibling focused tests passed 20 with seven CUDA-only skips.
The direct 12B/1,024 job `1740811` was cancelled pending with zero runtime after
review showed that it bypassed the scaling ladder and encoded an undesirable
eight-hour operational ceiling. Preserve its immutable bundle as superseded
preparation evidence. Versioned remaining-gate manifests now own the corrected
one-hour-probe/two-hour-full-run policy.

### 2026-07-29 execution update

At the user's direction, the available H200 allocation was used to evaluate
SP2 directly on the 12B workload instead of running the planned 1B/4B ladder.
The implementation, exact/bounded variants, traffic-accounting correction, and
SP3 prepared-row follow-up are recorded in
`reports/2026-07-29_exact_trace_sp2_row_tier_results.md`.

- [x] Define the GPU feature-row-tier interface and exact capacity estimator.
- [x] Implement focused append/read/fallback/cleanup and telemetry tests.
- [x] Complete a 12B mapped-lazy control and full SP2 variants from immutable
      dual-repository snapshots.
- [x] Re-profile and evaluate the prepared-row SP3 objective.
- [x] Reject promotion because the fast variants are not exact and the exact
      variant is 1.4% slower in Phase 4.

The 1B/4B promotion ladder, SP4, and SP5 remain unexecuted. Subsequent work is
authorized under SP2.2's explicit bounded lane: restore the fast CUDA consumer,
evaluate the prepared CPU mode independently, and measure a bounded-HBM
windowed CUDA provider before selecting a new SP3 bottleneck.

SP2.2 is now implemented and characterized in
`reports/2026-07-29_exact_trace_sp2_2_selectable_modes_results.md`.

- [x] Add explicit `cpu_exact`, `cpu_prepared`, `cuda_full`,
      `cuda_windowed`, and `auto` modes with typed profile wiring.
- [x] Keep `cpu_exact` as the exact default and make all accelerated modes
      explicit/default-off.
- [x] Validate full CUDA, prepared CPU, and artificially constrained 512 MiB
      window admission on the 12B H200 workload.
- [x] Complete a full selectable `cuda_full` trace: 190.59-second Phase 4,
      42.71 GiB peak CUDA reservation, bounded gate pass.
- [x] Complete synchronous and pinned double-buffered `cuda_windowed` traces.
      Double buffering reduced Phase 4 from 373.20 to 300.91 seconds while
      preserving the synchronous window graph exactly and holding CUDA
      reservation at 38.82 GiB.
- [x] Record the actual window stream: 111.87 GB read from host and sent H2D,
      536.86 MB total window HBM, 536.86 MB pinned staging, and a 4.18 GB host
      mirror.
- [x] Re-profile SP3 after full residency. Feature-batch execution is 145.28
      of 190.59 seconds; normalization is 17.45 seconds (9.2%), direct
      accumulation 0.98 seconds, and row reads 0.37 seconds. No allowed
      non-batch target meets the 10% continuation threshold, so SP3 closes
      without another implementation.

The bounded result does not unblock exact SP5 promotion. The next meaningful
choice is SP4 formal evidence or LS longer-prefix characterization using
`cuda_full` until its explicit budget/safety gate refuses, then
`cuda_windowed`, with `cpu_exact` as the atomic exact fallback.

SP0 is now complete and the login-safe portion of LS0 is implemented:

- [x] Validate a machine-readable mechanism ledger with stable evidence IDs,
      all required evidence categories, and explicit promotion separation.
- [x] Register the repeated mapped-lazy 12B result as the campaign control
      without modifying the frozen scientific baseline registry.
- [x] Extract current 1B/4B/12B controls and unresolved SP1/SP4 gates into
      `reports/2026-07-29_exact_trace_sp0_claim_ledger.md`.
- [x] Add a typed 124/256/512/1,024-token campaign manifest and login-safe
      listing command while retaining all fixed suite aliases.
- [ ] Generate the long deterministic trajectory and populate immutable
      fixture, trajectory, prefix, and target fingerprints.
- [ ] Add and freeze the two held-out prompt families before LS tuning.

SP4.1 now has two exact formal pairs:

- [x] Add explicit telemetry-on/off profiles and a true disabled observer.
- [x] Run off→on v3: 163.614817 s versus 179.769358 s completion
      (9.873% overhead), with exact signed compact parity.
- [x] Retain all batch timing events while reducing Phase-4 resource samples
      from 65 batches to batches 1-3, every 32nd batch, and the final batch.
- [x] Run reversed on→off v4: 178.782410 s versus 157.858818 s completion
      (13.254% overhead), again with exact signed compact parity.
- [x] Reduce incremental event-sink overhead below the preregistered 2% gate.
      Unified first-three/every-32/final Phase-4 resource sampling and a
      64-event bounded JSONL flush interval passed the v5 reversed-pair gate:
      telemetry-on mean 184.783162 seconds versus off mean 185.862352 seconds,
      or -0.581% measured overhead. Both on runs recorded 9,835 events, zero
      sink errors, and a maximum 63-event crash-loss window.
- [x] Execute SP4.2 from a genuinely job-private staged 4B transcoder cache.
      The private lifecycle issued all 34 owned decoder-range releases (85
      GiB requested) and reduced the observed cgroup file charge by 4.426 GiB;
      the matched shared-scope reference refused all 34. Both full runs were
      exact with zero Phase-4 decoder reloads. Three-batch probes recorded CUDA
      events plus OS/cgroup-v1 evidence, and warm batches showed no material
      regression. See
      `reports/2026-07-31_exact_trace_sp4_lifecycle_results.md`.

The SP4.1 and SP4.2 run IDs, snapshots, evidence, and decisions are documented
in `reports/2026-07-29_exact_trace_sp4_intermediate_results.md` and
`reports/2026-07-31_exact_trace_sp4_lifecycle_results.md`. Continue in the
user-selected order: open-ended SP1, then SP5/LS0 including deterministic
trajectory generation.

SP1 is closed. Direct duplicate-aware host
materialization removed the old approximately 0.98 GiB GPU occurrence table
and bulk per-layer gather removed the pathological tiny host reads. A matched
12B full pair measured:

- direct active CPU: 368.70-second completion, 44.66-second Phase 0,
  286.69-second Phase 4, and 129 semantic batches;
- mapped lazy: 457.30-second completion, 34.43-second Phase 0,
  389.50-second Phase 4, and 130 semantic batches; and
- active-CPU savings: 19.4% completion and 26.4% Phase 4.

The matched graphs shared 8,190/8,192 features, with feature Jaccard
0.999511838, all-edge Jaccard 0.999500125, signed normalized L1 0.000471032,
exact Top-64 through Top-1024 support, exact sign agreement, and exact target
token. This meets canonical `exact` but not `strict_exact`. Persisted diagnostic
sidecars localized the first numerical difference
after byte-exact Phase-3 feature rows, in the placement-specific seed-influence
reduction (maximum absolute delta `9.0245e-7`). `active_cpu` and mapped lazy are
both retained exact candidates; the former has the stronger measured
runtime/memory result and the latter has broader repeat/prompt evidence. No
default is selected until LS1 compares their scope. See
`reports/2026-07-31_exact_trace_sp1_results.md`.

### 2026-07-31 SP5 and LS0 execution update

LS0 is complete. One deterministic 1,329-token trajectory supplies frozen
development prefixes at 129/256/512/1,024 tokens, and two held-out prompt
families supply frozen 256- and 512-token workloads. All eight workloads have immutable
prompt, trajectory, prefix, and target fingerprints, and the strict login-safe
campaign listing passes.

The SP5 formal matrix is executed and admits exact candidates without selecting
a default:

- [x] Run three 1B and two 4B/12B canonical-prompt repetitions of the composed
      selective profile; all were exact on `361_base`, and 12B completed in
      443.99s and 443.27s (471.38s and 470.14s harness wall).
- [x] Add 1B prompt breadth, a 4B holdout, and CLT regression evidence.
- [x] Classify held-out 1B `94_base` as exact but not strict_exact: feature,
      edge, and weighted-edge Jaccard were 0.999756, 0.997004, and 0.996279,
      with signed L1 0.003728.
- [x] Retain canonical full-page Phase 0 as the strict audit/reference path and
      selective Phase 0 as the faster exact candidate; preserve historical
      profile aliases.
- [x] Measure the strict cost: 12B completion 1905.57s, Phase 0 1370.02s,
      Phase 4 476.53s, peak CUDA reservation 38.82 GiB, exact compact graph.
- [x] Record the remaining 4B source-regime conflict: current canonical
      `361_base` versus the older mechanism control has feature/edge/weighted
      Jaccard 0.999024/0.998401/0.998881 and signed L1 0.001119.
- [x] Publish
      `reports/2026-07-31_short_prefix_performance_results.md` and update the
      mechanism ledger without changing launch defaults or the frozen
      scientific baseline.

SP5 is closed as a completed characterization. Selective Phase 0 and active CPU
are allowed into comparative promotion selection because both meet `exact` on
their measured scope. LS1 is the next step: broaden that scope, measure resource
scaling, then select or decline a default candidate. Source-invariant Phase-0
localization remains a `strict_exact` improvement objective, not an LS1 blocker.

### 2026-08-03 LS1 execution update

LS1 now uses a thin preparation helper over the existing full-answer shard
runner. The helper revalidates the frozen campaign file, prompt, trajectory,
prefix-token, and target fingerprints; materializes one ordinary trace spec and
one ordinary shard; applies the selected performance profile; and prints the
existing runner command. It does not introduce another trace execution path.

The first same-source 1B pair completed for `ls0-361-dev-129` on H200 from
project commit `3899d8e` and sibling commit `f7b657b` under a warm sequential
control-then-candidate cache protocol:

- canonical full-Phase-0 control: 70.546-second trace wall time,
  15.91-second Phase 0, and 40.20-second Phase 4;
- selective-Phase-0 candidate: 61.185-second trace wall time,
  6.97-second Phase 0, and 39.76-second Phase 4;
- candidate improvement: 13.3% trace wall time and 56.2% Phase 0;
- both: 70,084 active features, 8,192 retained features, 33 Phase-4 semantic
  batches, and a 2,296,792,848-byte logical feature-row store; and
- compact graph: `strict_exact` with feature/edge/weighted-edge Jaccard 1.0,
  signed and magnitude normalized L1 0.0, exact signs, exact Top-64 through
  Top-1024 support, and exact target token.

Incremental full-answer telemetry is now persisted beside each token artifact.
The formal pair each closed 3,188 telemetry events without sink errors. The
first cache-misconfigured attempt failed before model load and is not scientific
evidence; earlier completed controls from pre-telemetry-fix source revisions are
timing evidence only, not members of the formal pair.

The reversed-order 256-token pair also completed strict-exact. Candidate then
control trace wall times were 108.443 and 116.092 seconds, a 6.6% candidate
improvement. Selective Phase 0 took 7.56 seconds versus 15.42 seconds for the
canonical control, while Phase 4 was flat at 85.83 versus 85.71 seconds. Active
features increased to 136,997 and the logical feature-row store to 4.490 GB.
Peak CUDA reservation increased materially from 25.95 GiB at 129 tokens to
50.73 GiB at 256 tokens. Therefore 512 remains a probe-then-full rung for HBM
admission even though adjacent runtime scaling projects it inside the normal
direct-run time regime. The admitted 512 probe must still be followed by the
full control/candidate pair to capture late memory and file-cache behavior.

The 512-token four-batch candidate probe then completed as `probe_completed`
in 18.604 seconds and admitted the full pair. It observed 276,667 active
features, a 9,066,930,924-byte logical row store, 97.67 GiB peak CUDA
allocation, and 107.06 GiB peak CUDA reservation (74.5% of the H200), while
closing all 521 incremental telemetry events without errors. The complete
control-then-candidate pair subsequently finished with runner trace times of
190.142 and 182.257 seconds, respectively: a 4.15% candidate improvement.
Selective Phase 0 took 7.95 seconds versus 16.37 seconds for canonical Phase 0;
Phase 4 was flat at 156.76 versus 156.19 seconds. Both runs sustained all 33
Phase-4 batches after the probe boundary. Late file charge reached about 70.45
GiB and cgroup memory peak reached 78.75 GiB without a late-run slowdown or
resource refusal. The compact graphs were `strict_exact`: feature, retained-
edge, all-edge, and weighted-edge Jaccard 1.0; normalized and signed normalized
L1 deviation 0.0; exact signs and Top-64 through Top-1024 support; and exact
target token. At 1024 tokens, naive adjacent HBM scaling exceeds device
capacity, so that rung remains probe-only until its governor/fallback behavior
is observed.

The declared 1024-token candidate probe refused capacity in Phase 1 and no full
run was launched. Phase 0 completed in 18.46 seconds with 563,277 active
features, then the model LM head attempted a 128.00 GiB CUDA allocation while
95.12 GiB was already in use (91.42 GiB PyTorch allocated); the H200 had only
44.67 GiB free. The failure artifact preserves the wrapped NNsight exception
and original PyTorch `OutOfMemoryError`, and its telemetry sink closed all seven
events without errors. This makes the LS1 envelope explicit: 129, 256, and 512
tokens have completed strict-exact pairs, while 1024 is refused by Phase-1
full-position logit materialization. LS1 is complete. LS2 starts by retaining
only the already-declared final-token logits physically during Phase 1, without
changing semantic source batching or graph construction, then repeating this
same 1024-token transition probe.

LS2 Phase-1 final-token logit materialization is implemented in sibling commit
`c685713`. When Phase 0 declares `last_token`, supported Hugging Face causal-LM
forwards receive their exact one-token logit-slice argument; full-logit requests
and unsupported models preserve the old path with explicit fallback metadata.
Focused tests and lint passed, and the broader replay/Phase-4/governor set had
84 passes plus one unrelated pre-existing 187-line architecture-guard failure
in untouched Phase-4 operation functions.

The frozen 1024-token transition probe then completed as `probe_completed` in
334.015 seconds. Phase 1 used `logits_to_keep=1`, completed in 1.86 seconds, and
measured 106.48 GiB peak CUDA allocation plus 111.49 GiB reservation instead
of attempting the former 128 GiB output allocation. The full probe reached
124.77 GiB maximum observed CUDA reservation, 89.25% of H200 capacity and close
to the 90% envelope. Phase 3 took 100.61 seconds and the first four Phase-4
batches took 4.89, 37.32, 65.60, and 104.25 seconds (212.63 seconds total).
Telemetry localized that next boundary: the 1,297,790,208-byte exact active
decoder-row set exceeded the profile's 1 GiB admission cap, refused fused row
residency, and fell back to repeated 64-chunk decoder scans. The logical row
store preallocation was 18,459,713,844 bytes. Do not launch the full 1024 trace
from this profile. LS2 should next expose a dedicated long-prefix profile with
a narrowly raised active-row cap, repeat the transition probe, and admit a full
trace only if fused rows both remove the scan growth and remain inside HBM.

The active-row follow-up and HBM-safe split are now implemented as explicit,
1B-only registered profiles. The preparation helper accepts a registered
`--profile-name` override while preserving the frozen workload and declared
campaign role. The 1536 MiB active-row probe admitted all 563,277 exact rows
(1,297,790,208 bytes) from the Phase-0 fused seed and reduced the same first
four Phase-4 batches from 212.63 to 7.33 seconds, but a 256-row execution caused
CUDA reservation to reach 124.90 GiB (89.35% of H200). The follow-up profile
preserved 256-row semantic batches while splitting physical execution at 128
rows. Its probe completed in 18.795 seconds, reduced four physical Phase-4
executions to 2.77 seconds, and held peak reservation to the Phase-1 level of
111.49 GiB (79.75%).

Before using the automatic final-token logit path at 1024, a complete 512-token
same-profile proof compared sibling commit `c685713` with the earlier full-logit
artifact. The graphs were `strict_exact`: all 8,192 features and 20,000 edges,
weights, signs, normalized L1 metrics, Top-64 through Top-1024, and target token
matched exactly. Runner time changed from 182.257 to 179.265 seconds (1.64%,
within noise), while Phase-1 peak CUDA allocation fell from 97.67 to 40.95 GiB
and reservation from 107.06 to 43.06 GiB. The exact path is therefore
automatically promoted on supported models based on the large memory win; full
logits and explicit unsupported-model fallbacks remain in code.

The admitted reversed-order 1024 full pair completed from project commit
`385cac4` and sibling commit `c685713`. The selective candidate took 316.037
seconds runner wall and 301.75 seconds attribution core; the matched canonical-
Phase-0 control took 319.062 and 304.89 seconds. Selective Phase 0 was 8.73
versus 15.61 seconds (44.1% faster), but Phase 4 was flat/reversed within noise
at 282.75 versus 279.24 seconds, leaving only a 0.95% runner-level candidate
win. Both completed 65 physical executions for the same 33 semantic batches,
used an 18,459,713,844-byte logical row store, held CUDA reservation at 111.49
GiB, closed 5,231 telemetry events without sink errors, and remained stable as
late file charge reached 80.25 GiB and cgroup peak reached 92.94 GiB. The full
graphs were `strict_exact` at every canonical metric. The next LS2 target is
not decoder replay: Phase-4 refresh consumed 231.7--234.7 seconds, about 74% of
end-to-end time, while feature-batch work took about 45 seconds. Prove the
128-row physical split against an earlier 512 artifact, then evaluate the
existing exact windowed/GPU feature-row influence modes within the measured
1024 HBM headroom.

The 512-token physical-split proof completed before that bounded experiment.
The 128-row execution profile produced a 189.137-second runner trace and a
163.73-second Phase 4, compared with 179.265 and 153.44 seconds for the matched
256-row physical execution. The split doubled physical executions from 33 to
65 while preserving all 33 semantic batches. It was `strict_exact`: feature,
edge, weighted-edge, and all-edge Jaccard were 1.0; normalized and signed
normalized L1 deviation were 0.0; signs, Top-64 through Top-1024 support, and
target token matched exactly. This is a 5.5% runner regression, so b128 is not
an exact performance promotion. Retain it only as an explicit HBM-safety
envelope for combinations such as bounded windowed CUDA that need more device
headroom. The next experiment must use the existing full-answer launch path:
first admit `cuda_windowed` with a short transition probe, then run the complete
1024 trace because the exact reference itself finishes in about five minutes.

That experiment exposed and repaired a general full-answer integration gap.
The prepared trace spec contained the row-influence mode and budgets, but the
canonical full-answer `TraceRequest` bridge did not copy them into
`RowStoragePlan`; the first probe/full diagnostic therefore resolved to
`cpu_exact` despite the requested graph knobs. Project commit `dd8ac10` wires
all four controls and adds focused regression coverage. The corrected probe
resolved `cuda_windowed`, admitted a 119-row double-buffer window with
536,239,704 bytes each of HBM and pinned staging, and owned an 18,459,713,844-
byte signed host mirror without fallback.

The mandatory corrected 1024-token full trace is a large, exact win. Runner
wall fell from 316.037 to 164.218 seconds, attribution core from 301.75 to
149.33 seconds, and Phase 4 from 282.75 to 129.62 seconds. Refresh fell from
234.704 to 79.404 seconds (66.2%), while feature-batch work was effectively
flat at 45.40 versus 47.56 seconds. The provider streamed 362,207,388,972 bytes
through 1,643 window reads/prefetches; host staging consumed 9.57 seconds and
stream synchronization 0.008 seconds. CUDA reservation remained 111.49 GiB.
The host mirror increased late cgroup charge to 102.46 GiB, including 21.11
GiB anonymous and 81.07 GiB file memory, safely inside the 200 GiB allocation.
The compact graph was `strict_exact` against the matched `cpu_exact` artifact
at every canonical field and Top-64 through Top-1024. The 1B/1024 profile is
therefore non-dominated and retained. Its measured scope is this development
workload until shorter-length crossover and prompt holdouts are completed; the
historically bounded CUDA family must not be relabeled globally from one exact
point.

The same corrected profile completed the remaining admitted development
lengths and closes LS2. Against the fastest previously measured exact candidate
at each length, runner wall changed as follows: 61.185 to 56.485 seconds at 129
tokens (7.7%), 108.443 to 73.584 at 256 (32.1%), 179.265 to 104.163 at 512
(41.9%), and 316.037 to 164.218 at 1024 (48.0%). Phase-4 wall was 33.600,
47.957, 75.005, and 129.723 seconds, respectively. The 129, 512, and 1024
graphs were `strict_exact`. The 256 graph was `exact`: feature and edge support,
shared signs, target, and Top-64 through Top-1024 were exact; weighted Jaccard
was 0.999999999251 and normalized magnitude/signed L1 were
0.000000000749. All telemetry sinks closed without errors.

Windowed traffic scaled from 40.82 GB at 129 tokens through 85.54 and 168.31
GB to 362.21 GB at 1024. The fixed owned window remained about 536 MB; the
signed host mirror scaled with the logical row store from 2.30 GB to 18.46 GB.
Maximum observed cgroup charge remained below 104 GiB and CUDA reservation
remained 111.49 GiB at the largest rung. The explicit profile
`ls2-1b-long-prefix-cuda-windowed-512mib-b128-v1` is now the frozen LS2 1B
development finalist. This is an automatic promotion within its demonstrated
1B length scope: it is exact at every admitted development length and
non-dominated on runtime and admitted resources. It is not a global default or
a global fidelity reclassification. Do not tune it after seeing LS3 holdouts;
holdout evidence may accept it, reject it, or narrow its applicability.

LS3 ran the frozen profile without post-holdout tuning on prompts 828 and 94 at
256 and 512 tokens, with matched `cpu_exact` controls and reversed pair order.
All eight traces completed and all telemetry sinks closed cleanly. Windowed
CUDA reduced runner wall by 36.7%, 38.9%, 46.0%, and 47.8%, respectively, and
reduced Phase-4 refresh by 69.7--71.1%. The mode resolved as requested without
fallback; logical row stores ranged from 4.88 to 10.00 GB, fixed windows stayed
near 535--537 MB, maximum observed CUDA reservation was 47.53 GiB, and maximum
observed cgroup current was 87.36 GiB.

Fidelity was prompt-dependent at 256 tokens. Prompt 828/256 was `exact`
(feature/edge/weighted Jaccard 0.999512/0.999900/0.999946, normalized and signed
L1 0.0000544, exact signs, target, and Top-64 through Top-1024). Prompt 94/256
was only `close` (0.993672/0.992528/0.991711 Jaccard, 0.008323 normalized and
signed L1, exact signs and target, but 63/64 through 1019/1024 Top-K overlap).
Both 512-token holdouts were `strict_exact`. Because the same prompt is close
at 256 and strict-exact at 512, prompt and length alone do not provide a safe
pre-run exact selector. LS3 therefore rejects broad automatic exact promotion
for unseen prompts while retaining the profile as a non-dominated selectable
close-or-better 1B mode with atomic `cpu_exact` fallback. The earlier LS2 exact
development scope and individually measured exact holdout points remain valid;
an arbitrary exact request continues to use `cpu_exact`. LS3 is complete by an
evidence-backed applicability boundary, and LS4 may now test 4B transfer without
reopening or tuning the frozen 1B holdouts.

LS4 preparation is implemented without a new launch path. The frozen
`ls4-gemma3-4b-transfer-v1` manifest reuses the LS0 prompt, prefix, and target
token sequences while changing only the traced model/provider to Gemma 3 4B
PLT; this isolates model-size transfer from response-generation differences.
It declares development rungs at 129, 256, 512, and 1024 tokens plus the
828/512 holdout, all behind early probes. The matched registered profiles share
selective Phase 0, exact final-token logits where supported, b128 physical
execution, a 4 GiB active-decoder-row cap, and all remaining controls. Their
only row-influence difference is `cpu_exact` versus a 512 MiB double-buffered
`cuda_windowed` provider with a 16 GiB HBM safety margin. The existing
preparation helper resolves the candidate to `gemma3_4b_plt` and the ordinary
full-answer shard runner. Begin with the 129-token candidate transition probe;
do not admit longer rungs until the preceding complete pair is classified.

The cache-corrected 129-token LS4 candidate transition probe passed admission.
It completed in 220.740 seconds, dominated by 203.817 seconds of Phase-0 4B
checkpoint traversal. The requested `cuda_windowed` provider resolved without
fallback, admitted a 536,411,344-byte / 722-row window over a 3,043,502,868-byte
logical host mirror, and held all 92,869 active decoder rows (475,489,280 bytes)
under the 4 GiB cap. Four physical Phase-4 executions covering 511 selected
features took 4.307 seconds. Maximum observed CUDA reservation was 35.91 GiB;
cgroup current reached 174.46 GiB, including about 161.49 GiB of warm checkpoint
file cache, inside the declared 200 GiB envelope. Admit the complete matched
129-token pair under the warm sequential control-then-candidate protocol. The
first attempt that pointed `HF_HUB_CACHE` at the nested 1B-only `/hub` directory
failed before model load and wrote no token result; it is environment diagnostics,
not scientific evidence.

The warm sequential 129-token LS4 full pair is `strict_exact` and advances the
4B ladder. Matched `cpu_exact` and `cuda_windowed` runner wall was 125.835 and
98.511 seconds (21.7% reduction); Phase 4 was 83.426 and 58.267 seconds (30.2%),
and refresh was 40.951 and 15.391 seconds (62.4%). Feature-batch work was flat
at 38.910 versus 39.359 seconds. All 8,192 feature IDs, 20,000 retained edges
and weights, signs, target, and Top-64 through Top-1024 matched field for field.
The candidate streamed 67,130,542,388 bytes through 497 window reads, with 1.73
seconds of host staging and 0.0024 seconds of synchronization. Maximum sampled
cgroup current was 183.21 GiB and CUDA reservation remained 35.91 GiB. This is
a non-dominated exact point, but it is only the 4B/129 development scope. Admit
the 256-token candidate transition probe; do not skip its host-capacity gate.

The warm 256-token LS4 transition probe passed that gate in 44.750 seconds.
Phase 0 took 29.277 seconds and four physical Phase-4 executions took 3.150
seconds. All 188,771 active decoder rows (966,507,520 bytes) were resident
without fallback. `cuda_windowed` admitted a roughly 536 MB window over the
already allocated 6,186,403,212-byte logical host mirror. Maximum sampled
cgroup current was 182.14 GiB and CUDA reservation was 49.72 GiB. Admit the
complete 256-token matched pair under the same warm control-then-candidate
protocol; classify it before probing 512.

The complete 256-token LS4 pair is `exact` and exposes a host-capacity boundary.
Runner wall fell from 204.179 to 129.403 seconds (36.6%), Phase 4 from 162.783
to 84.898 seconds (47.8%), and refresh from 114.343 to 34.264 seconds (70.0%).
Feature-batch work was 44.990 versus 47.273 seconds. The graphs shared
8,191/8,192 features (Jaccard 0.999756); all 20,000 edge identities and weights,
normalized and signed L1, shared signs, target, and Top-64 through Top-1024 were
exact, so the point meets canonical `exact`. The candidate streamed
147,711,797,332 bytes through 1,312 reads from a 6,186,403,212-byte host mirror.
CUDA reservation remained 49.72 GiB, but maximum sampled cgroup current reached
196.52 GiB, only 3.48 GiB below the declared host envelope. Do not launch the
current mirrored profile at 512: its mirror alone would add roughly another 6
GB before other prefix growth. Reopen only the newly exposed host-residency
boundary. Evaluate whether the existing exact file-backed row store can feed
the same bounded CUDA windows without a second full signed RAM mirror; retain
the current mode regardless because it is non-dominated on admitted 4B/129--256.

The LS4 file-backed CUDA-window experiment closes that boundary as a rejected
implementation family. Version 1 streamed the canonical memmap through bounded
pinned-host and CUDA windows and completed the 4B/256 trace with `strict_exact`
fidelity: all 8,192 features, 20,000 retained edges and weights, signs, target,
and Top-64 through Top-1024 matched. It removed ownership of the
6,186,403,212-byte signed host mirror, but repeated mapped reads accumulated in
the cgroup file cache. Trace wall was 189.640 seconds and Phase 4 was 158.212
seconds, versus 129.403 and 84.898 seconds for mirrored `cuda_windowed`; wrapper
wall was 225.399 seconds and peak cgroup usage was 198.542 GiB. The candidate
was therefore slower without a meaningful measured total-memory win.

Version 2 read directly from the canonical file into the pinned buffer and
discarded each bounded range afterward. This reduced mapped-file process RSS,
but made refresh storage-bound: Phase 4 reached 371.35 seconds versus 162.783
seconds for `cpu_exact`. The resource wrapper reached 201.053 GiB and the 200
GiB guard stopped the run before packaging, leaving zero trace-result rows.
Both variants are dominated and have been removed from live runtime/profile
selection. Their sibling implementation series and project registration are
preserved in `experiments/patches/exact_trace_performance_20260803/` with an
indexed restoration guide. The independently useful prepared-workload resource
guard helper remains active.

Do not force the next 4B/512 or any 12B trace into this interactive allocation.
Use the existing per-trace launch machinery with a RAM request sized for the
workload. The interactive node remains appropriate for guarded probes and full
runs that fit comfortably; a guard refusal is a capacity result, not a reason
to squeeze a larger trace into the node.

The 4B/512 candidate transition probe has therefore moved to the ordinary
immutable-snapshot full-answer job path. Job `1703320_0` requests one H200,
400 GiB host memory, 32 CPUs, and two hours on `rai-gpu-grn`; it runs only four
Phase-4 executions from the frozen `ls4-4b-361-dev-512` workload. The submitted
snapshot pins project commit `15e537c` and sibling commit `aec77af`, after the
dominated file-window family was removed. Admit no 512 full trace until this
job completes and its provider resolution, active-row residency, HBM usage,
host usage, and probe runtime are inspected. A scheduler-validation attempt on
the 250 GiB short QOS was refused before submission; switching to the normal
QOS was the intended high-memory route. Slurm did not honor the environment-form
test-only flag on the accepted request and created `1703320`; treat it as the
single real probe and do not submit a duplicate.

The first submitted 4B/512 pair (`1712546_0`, `1712550_0`) is diagnostic only:
both jobs received a 70 GiB H200 slice, and the candidate resolved to
`cpu_exact`. Full-H200 job `1714134` then preheated the complete 49-file,
178.11 GiB asset set and produced the valid `cpu_exact` control before a
post-trace wrapper validator rejected the native mode. Candidate-only job
`1721614` reused the frozen workload and runtime revisions, repeated the
preheat, required `cuda_windowed`, and completed successfully on a full H200.

The accepted 4B/512 comparison is `strict_exact`. Candidate trace wall is
168.067 seconds versus 346.773 seconds for the control (51.5% reduction,
2.06x); Phase 4 is 121.596 versus 276.521 seconds; and refresh is 53.521 versus
201.158 seconds. All 8,192 features, all 20,000 edges and weights, signs,
target token, and Top-64 through Top-1,024 views match. The candidate resolved
to `cuda_windowed`, issued no fallback reads or copy failures, and recorded no
telemetry drops or sink errors. Retain it as a non-dominated selectable 4B
development profile; do not promote it to an arbitrary-prompt or global
default.

The full H200 and 400 GiB allocation is a demonstrated safe execution contract
for this frozen 4B/512 workload. Peak CUDA reservation was 63.08 GiB on a
139.80 GiB device, leaving substantial HBM headroom. The 400 GiB request was
also operationally ample, but it is not a measured minimum: the control cgroup
owned about 191.5 GiB of shared file cache, while the later candidate cgroup
owned only about 13.1 GiB of those pages. Candidate anonymous memory reached
21.13 GiB, consistent with its 11.92 GiB host mirror. Do not infer a total host-
memory reduction or downsize the formal 400 GiB request from the candidate's
34.37 GiB MaxRSS. Keep the stale 200 GiB estimate stop disabled for this active
optimization family while retaining Slurm limits and measure-only telemetry.

### 2026-08-07 LS4 result and next-work gate

The completed review is
`reports/2026-08-04_exact_trace_performance_review_provisional.md`; the filename
is retained for provenance, but its status and content are final. The ranked
next work is:

1. define one canonical selected-config/launch record, including the workload,
   mode requirement, preheat, resource policy, scheduler request, snapshot, and
   monitoring selection;
2. separate planning envelope, Slurm allocation, and runtime policy with typed
   `off | measure_only | enforce` behavior;
3. implement first-class `preferred | required` mechanism selection, rejecting
   `auto + required` and failing at admission, append/copy failure, and fallback
   use sites rather than only after the trace;
4. persist a machine-readable requested-to-effective execution delta with
   resolution stage and reason, then exercise the controls with focused
   propagation and failure-injection tests;
5. use those controls for the unchanged `ls4-4b-361-dev-1024` transition probe:
   preheated, required `cuda_windowed`, `measure_only`, full H200, 400 GiB,
   and a one-hour job;
6. admit and classify the full 4B/1,024 candidate/control pair only when each is
   projected to complete within two hours;
7. run the frozen 828/512 holdout before broad LS4 closure;
8. transfer to 12B in order at short, 256, 512, and 1,024 tokens, beginning each
   rung with a one-hour probe and admitting only full runs projected below two
   hours;
9. stop at the first unsafe or over-ceiling rung, profile that exact point, and
   promote only its binding optimization before resuming the ladder;
10. retain cancelled zero-runtime job `1740811` and its immutable 12B/1,024
    bundle as superseded preparation evidence, not a ladder result;
11. before changing normalization, replace wall-only CUDA timing with event-
   aware attribution; if normalization remains material, test one exact batched
   or fused denominator application without changing reduction order;
12. prove any normalization candidate first against an existing smaller frozen
   artifact, then at 4B/512, before mixing it into the 1,024-token evidence; and
13. do not request jobs longer than two hours for this ladder; an over-ceiling
    projection is an optimization or checkpointing gate, not a reason to widen
    the request.

The single-owner canonical host-row design remains conditional. Unequal page-
cache ownership prevents a reliable duplicate-store charge measurement, so do
not open that refactor until job-private ownership or explicit file-residency
telemetry demonstrates that duplication is actually binding. The rejected
file-only window family remains closed.
