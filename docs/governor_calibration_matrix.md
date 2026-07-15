# Governor Calibration Matrix

Status: Executable campaign contract
Last updated: 2026-07-15

This campaign calibrates physical cost models for the staged tracing governor.
It does not broaden semantic-relaxation evidence. Every row uses strict fp32
tracing, one Granite H200, `361_base`, one completion/trace, incremental
telemetry, and immutable project+sibling snapshots.

The executable source is
`src/nlp_research_project/exact_trace_bench/scenarios/governor_calibration.py`.
Generate the four resource-homogeneous scenario files with:

```bash
UV_CACHE_DIR=.uv-cache uv run python experiments/build_governor_calibration_configs.py
```

## Matrix

The generator creates 38 runs across the providers whose weights are available
on CHPC:

| Provider | Cases | Purpose |
|---|---:|---|
| Gemma 3 1B CLT | 15 | reference/repeat plus isolated scheduling, cache, storage, residency, and placement effects |
| Gemma 3 1B PLT | 15 | same causal contrasts for same-layer topology |
| Gemma 3 4B PLT | 4 | reference, matched session pair, tiled-policy transfer check |
| Gemma 3 12B PLT | 4 | reference, matched session pair, tiled-policy transfer check |

The 1B rows change one causal factor relative to their provider reference:

1. reference physical configuration;
2. high-session member of the session pair with Phase-1/3/4 physical batches
   fixed below both session capacities;
3. lower-session member with those same Phase-1/3/4 widths;
4. lower Phase-1 source batch only;
5. lower Phase-3 microbatch only;
6. lower Phase-4 microbatch only;
7. decoder-cache contrast (CLT: 8 GiB to zero; PLT: zero to 4 GiB);
8. replay-window contrast (four layers to two);
9. prefetch-depth contrast (two to zero);
10. recompute with zero replay-tile cache;
11. recompute with the reference replay cache (CLT 1 GiB, PLT 4 GiB);
12. tiled row storage against full file-backed storage;
13. eager against lazy encoder residency;
14. scratch against node-local spill placement;
15. exact reference repeat for run-noise estimation.

The 4B/12B lower-session rows estimate how session residency scales with model
dimensions. Their tiled rows test whether the 1B PLT policy-relative model
transfers; until those rows pass, non-full policy predictions for those profiles
remain extrapolated.

## Required Measurements

Each scenario must preserve the canonical incremental event stream and expose:

- phase elapsed time and operation/batch counts;
- phase-local CUDA peak allocated and reserved bytes plus end allocation;
- host RSS and available/cgroup budget observations;
- decoder-cache hits and misses;
- replay-tile cache hits, misses, and evictions;
- row-store bytes read and written;
- every selected configuration, calibration-support classification, phase
  prediction, observation, and prediction error;
- compact graph parity against the original corrected-hook artifact baseline.

Allocated and reserved CUDA measurements are parallel observations, not terms
to add. Cache-state order is recorded because later array tasks may inherit a
warm file cache even though each process-local decoder/replay cache starts
empty.

## Promotion

Promote a profile revision only after fitting on causal rows and checking it on
held-out rows. At minimum:

- use the two 1B reference repeats to report run noise;
- fit session, Phase-1, Phase-3, Phase-4, decoder-cache, replay-window,
  prefetch, replay-cache, storage-policy, residency, and placement terms from
  their isolated pairs;
- verify the fitted model ranks each held-out reference and larger-model row
  correctly before it becomes an ordinary execution profile;
- retain explicit support ranges and evidence IDs; values inside safety limits
  but outside observed support remain legal extrapolations with lower
  confidence;
- never fit a running trace and then use that fit to govern the same trace.

The immediate governor-v0.3 gate is intentionally smaller: one unconstrained
1B CLT run and one unconstrained 1B PLT run. Those validate execution and parity;
they are not enough to promote the calibration model.
