# Bounded candidate reassessment

Date: 2026-07-26

Status: derived engineering analysis; no default or fidelity-scope promotion

## Question

Several optimization rows were previously set aside because they did not
preserve the canonical exact comparison. This report re-evaluates the saved
compact artifacts under the current worst-step `bounded` policy:

- feature Jaccard >= 0.98;
- all-edge Jaccard >= 0.98;
- all-edge Top-256 Jaccard >= 0.98;
- all-edge weighted Jaccard >= 0.98;
- normalized edge-magnitude L1 deviation <= 0.02;
- exact target-token identity.

The analysis reran the current `graph_compare.compare_artifact_dirs` against
the persisted artifact pairs. Compact artifacts contain edge magnitudes, not
internal edge signs. A bounded pass is therefore a scoped engineering result,
not canonical exact semantics or signed-edge parity.

## Current active-row era

| Candidate | Result | Bounded status | Current disposition |
|---|---:|---|---|
| PLT active rows, c4096 | 84.82s completion; Phase 0 15.49s; Phase 4 61.33s | exact pass | canonical exact incumbent |
| PLT active rows, c65536 | 79.37s completion; Phase 0 11.25s; Phase 4 60.18s | pass: feature 0.994643, edge 0.996008, Top-256 0.992218, weighted 0.995133, L1 0.004879 | retain as bounded chunk regime; 6.4% matched completion gain |
| PLT active rows + tape window 2, c4096 | 101.81s; Phase 4 66.98s | exact pass | reject for speed; active rows removed the tape's decoder-reuse benefit |
| Phase-0 ranges, c4096 | 84.39s completion; Phase 0 15.89s | exact graph pass; mechanism refused | retain safe fallback only; no selective read occurred |
| CLT capacity 128/256/512 | 213.98s / 139.50s / 105.01s | exact pass | explicit low-memory modes; none passed the joint memory/runtime promotion gate |

The c65536 row was rejected only for canonical exact promotion. It is a valid
bounded candidate. The other current-source negative results were rejected for
speed, mechanism effectiveness, or the declared memory/runtime Pareto gate,
not because the bounded graph budget was too strict.

## Historical 1B PLT Wave A

The warm c4096 reference repeat was 2,512.24s.

| Candidate | Time | Speedup | Worst bounded evidence | Bounded |
|---|---:|---:|---|---|
| decoder chunk c8192 | 1,441.52s | 1.74x | feature 0.997805; edge 0.997304; Top-256 1.0; weighted 0.997552; L1 0.002451 | pass |
| decoder chunk c16384 | 952.01s | 2.64x | feature 0.997805; edge 0.999600; Top-256 1.0; weighted 0.999748; L1 0.000252 | pass |
| decoder chunk c32768 | 626.54s | 4.01x | feature 0.997318; edge 0.997204; Top-256 1.0; weighted 0.997464; L1 0.002539 | pass |
| logical/semantic feature batch b256 | 1,496.97s / 1,459.63s | 1.68x / 1.72x | feature 0.993187; edge 0.994615; Top-256 0.984496; weighted 0.994756; L1 0.005258 | pass |
| semantic refresh g8 | 2,070.62s | 1.21x | feature 0.995372; edge 0.988566; Top-256 0.992218; weighted 0.988704; L1 0.011360 | pass |
| coupled b512/c8192 | 494.49s | 5.08x | feature 0.976595; Top-256 0.961686 | **fail** |
| logical b512 | 849.78s | 2.96x | feature 0.969704; edge 0.978435; Top-256 0.954198; L1 0.021790 | **fail** |
| logical b768 | 580.38s | 4.33x | feature 0.951638; edge 0.904490; Top-256 0.875458; L1 0.100220 | **fail** |
| logical/coupled b1024 | 507.09s / 208.55s | 4.95x / 12.05x | broad overlap 0.67-0.87; L1 0.24-0.26 | **fail** |
| logical/coupled b1536 | 425.42s / 151.63s | 5.91x / 16.57x | broad overlap 0.48-0.77; L1 0.38-0.46 | **fail** |

The current 1B active-row profile already retains the useful physical
execution-256 shape. Larger chunks remain useful only as a bounded Phase-0
speed lever. The high-batch corners remain outside the bounded budget and
should not be revived.

## Historical 1B CLT Wave A

All successful CLT rows pass the current bounded policy; most also satisfy the
exact thresholds. The useful speed row was decoder chunk c10080 at 73.31s
versus the 105.02s warm reference (1.43x), with feature/edge/Top-256 1.0,
weighted 0.999999983, and L1 1.72e-8. The b1500 rows were approximately
91-100s but require the previously recorded high-HBM envelope. These are
historical calibration observations; the newer CLT capacity profiles expose
the memory/runtime tradeoff more directly.

## Historical 4B PLT Wave B

The reference was 4,854.59s. Every independently informative candidate passes
the current bounded policy.

| Candidate | Time | Speedup | Feature / edge / Top-256 / weighted / L1 |
|---|---:|---:|---|
| execution b256 | 3,204.37s | 1.52x | 0.999024 / 0.998401 / 1.0 / 0.998881 / 0.001119 |
| execution b512 | 3,040.99s | 1.60x | 1.0 / 1.0 / 1.0 / 0.999999990 / 9.82e-9 |
| decoder chunk c16384 | 1,947.77s | 2.49x | 0.997075 / 0.996108 / 1.0 / 0.996665 / 0.003340 |
| decoder chunk c32768 | 1,291.93s | 3.76x | 0.993430 / 0.983242 / 0.992218 / 0.985080 / 0.015032 |
| execution b256 + c32768 | 1,056.56s | 4.60x | 0.995372 / 0.997403 / 1.0 / 0.997154 / 0.002850 |
| semantic feature/execution b256 | 3,442.40s | 1.41x | 0.994158 / 0.995510 / 0.992218 / 0.994840 / 0.005173 |

Artifact-local recomputation shows that the execution-b512 row is the
near-exact compact result and execution-b256 is the 0.998-level row. The
append-only 2026-07 experiment summary reversed those two metric assignments;
the timings and the conclusion that both pass bounded are unchanged.

## Historical 12B PLT Wave B

The reference was 22,572.62s. Every independently informative candidate passes
the current bounded policy.

| Candidate | Time | Speedup | Feature / edge / Top-256 / weighted / L1 |
|---|---:|---:|---|
| execution b128 | 13,310.86s | 1.70x | 0.999756 / 0.989456 / 1.0 / 0.990086 / 0.009963 |
| execution b256 | 10,551.32s | 2.14x | 0.999024 / 0.999500 / 1.0 / 0.999529 / 0.000471 |
| decoder chunk c16384 | 7,685.26s | 2.94x | 0.997562 / 0.988961 / 1.0 / 0.989772 / 0.010281 |
| decoder chunk c32768 | 5,494.78s | 4.11x | 0.995615 / 0.988763 / 1.0 / 0.989573 / 0.010482 |
| execution b128 + c32768 | 5,337.68s | 4.23x | 0.995615 / 0.985013 / 1.0 / 0.986757 / 0.013331 |
| semantic feature/execution b128 | 27,618.01s | 0.82x | 0.996345 / 0.984029 / 1.0 / 0.985648 / 0.014456 |

The execution-b256 row is both faster and closer to the canonical compact
artifact than b128. It is the strongest bounded execution-envelope candidate
to combine with active-row residency. The semantic-feature row passes bounded
but is slower than the reference and should remain rejected for performance.

## Disposition

Retain for new active-row transfer tests:

1. model-specific canonical execution at c4096 for exact mechanism transfer;
2. 4B execution b512 and 12B execution b256 as bounded Phase-4 candidates;
3. larger decoder chunks as independent bounded Phase-0 candidates;
4. combined execution-envelope plus high-chunk profiles only after the
   corresponding isolated mechanisms pass.

Do not revive:

- decoder cache, tape, or prefetch after active-row residency;
- 1B physical/logical batches of 512 or larger;
- semantic feature grouping that did not improve runtime;
- Phase-0 contiguous range loading for the observed sparse active-row layout.

For any high-chunk campaign, compare the optimized mechanism with a
same-chunk conservative reference to test mechanism parity, and compare the
high-chunk regime with c4096 only under the bounded policy.
