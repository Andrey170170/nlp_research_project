# VJP fidelity localization plan

Status: implemented and frozen; first submissions `1774854`, `1774856`, and
`1774857` failed before trace startup due wrapper import ordering; corrected
replacements `1789268`, `1789266`, and `1789265` are pending
Date: 2026-08-11

## Decision boundary

The 4B/256 `single_forward_batched_vjp` qualification achieved the memory goal
but failed bounded fidelity. Sampled framebuffer peak fell from 37,003 to 14,597
MiB (60.6%), while feature Jaccard was 0.97827, edge and weighted-edge Jaccard
were about 0.9542, and normalized L1 deviation was 0.04683. The observed runtime
improvement is provisional because the arms overlapped and did not have
identical cache ownership. No 512 or scaling-ladder advancement is admitted.

The first Phase-4 frontier already differs, so the earliest useful comparison is
Phase 3. The prior composite mode changed both forward width and backward
primitive; the qualification cannot attribute the drift to either one alone.

## Decomposed selector

The project and sibling runtime now share three canonical presets:

| Preset | Forward graph | VJP kernel |
|---|---|---|
| `duplicated_lanes` | `logical_capacity` | `nnsight_injected` |
| `single_forward_serial_vjp` | `single_lane` | `autograd_serial` |
| `single_forward_batched_vjp` | `single_lane` | `autograd_batched` |

Advanced diagnostic callers may select the two axes explicitly, but must provide
both and must omit `backward_engine_mode`. Unsupported combinations fail closed.
The absent selection still resolves to `duplicated_lanes`.

## Frozen diagnostic

[`ls4_4b_vjp_decomposition_diagnostic_v1.json`](../experiments/performance_campaigns/ls4_4b_vjp_decomposition_diagnostic_v1.json)
freezes one 4B/256 workload with three preheated H200 arms. All arms require
`cuda_windowed`, use `measure_only`, capture Phase-0 donor and Phase-3
gradient/row/seed bundles, and stop after the first Phase-4 execution batch.

Compare artifacts in this order:

1. Phase-0 support and activations.
2. Phase-1 target state.
3. Per-layer Phase-3 raw gradients.
4. Phase-3 normalized row and denominator.
5. Phase-3 seed influences, ranks, and initial frontier.
6. First Phase-4 batch identity.

Interpretation is deliberately nested and keeps the composite contrast visible:

- serial and batched agree: this rules out batched-vmap execution as the main
  cause conditional on the shared single-lane/direct-autograd stack; it does
  **not** distinguish forward width from NNSight injection;
- serial and batched differ: `is_grads_batched`/vmap is an active cause within
  the single-lane direct-autograd stack; compare the first divergent artifact;
- control and serial differ: this is a composite contrast between
  logical-capacity/NNSight-injected and single-lane/direct-autograd execution,
  not a clean forward-width effect;
- all three agree through Phase 3 but diverge in Phase 4: localize later
  grouping/reassembly behavior before any full graph rerun.

If serial and batched agree while both differ from control, a subsequent
physical-lane or injected-backward microbatch diagnostic is required to split
forward width from backward injection semantics.

This transition probe is localization evidence only. A repaired engine must
return to the frozen 4B/256 full comparison and pass bounded, preferably exact,
before 4B/512 is reconsidered.

## Launch attempts

The first control, serial, and batched submissions failed after 17, 15, and 17
seconds respectively, before cache validation, preheat, GPU work, output
creation, or tracing. The wrapper invoked the snapshot-owned prepared-expectation
module before exporting the snapshot project/library `PYTHONPATH`, so the stale
external environment could not import it. These attempts carry no scientific or
resource evidence.

The wrapper now establishes immutable snapshot imports before any snapshot Python
module is called. A regression assertion fixes that ordering, and an integration
check with `PYTHONPATH` initially unset resolved all five control expectations.
Fresh read-only v2 bundles were prepared from a new immutable project+sibling
snapshot. Replacement jobs are control `1789268`, serial `1789266`, and batched
`1789265`; all were pending for priority at the initial check.
