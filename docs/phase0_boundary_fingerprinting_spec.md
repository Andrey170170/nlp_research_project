# Phase-0 Boundary Fingerprinting Spec

## Purpose

Localize the first cross-cluster divergence for matched single-step `94_base`
runs by adding compact fingerprints around the Phase-0 sparse-encode boundary.

The earlier compare-mode experiment (baseline vs `fp32` vs `fp64` threshold
compare) was negative, so this spec shifts the investigation upstream.

## Decision summary

Instrument the Phase-0 boundary in three segments:

1. **Pre-CLT input** — the tensor entering CLT (`mlp_in_cache`)
2. **CLT pre-mask path** — transcoder constants, preactivation, and margin
3. **Post-mask path** — JumpReLU membership and post-mask activation

Use compact fingerprints only:

- hashes,
- compact sampled stats,
- per-layer counts,
- global aggregate hashes.

Do not emit dense tensors into JSON artifacts.

## Required fields

### Pre-CLT input

For each layer:

- shape
- element count
- sampled hash
- sampled stats

Also emit a global hash across per-layer hashes.

### CLT encode constants

For each layer:

- sampled encoder-weight fingerprint
- encoder-bias hash/stats
- JumpReLU threshold hash/stats
- per-layer constant fingerprint hash

Also emit a global hash across layer constants.

### Preactivation / margin / mask / post-mask

For each layer:

- preactivation sampled hash/stats
- compare-margin sampled hash/stats
- mask membership hash
- post-mask activation sampled hash/stats
- near-threshold counts for expanded epsilons

Also emit global hashes across per-layer:

- preactivation
- margin
- mask membership
- post-mask activation

## Chosen tradeoffs

### Chosen

- **Sampled dense fingerprints** instead of full dense dumps or full dense hashes
  for every tensor.
- Keep full near-threshold counts / membership summaries where those are already
  part of the diagnostic decision path.
- Preserve existing checkpoint structure and add fields instead of replacing the
  current schema.

### Rejected for now

- Full dense tensor serialization: too large and unnecessary.
- Immediate deterministic shadow replay: higher cost before we know whether the
  mismatch starts before or after CLT encode.
- More compare-mode variants: low information after the negative 6-job matrix.

## Expected interpretation logic

- If `mlp_in_cache` already differs, the root cause is upstream of CLT encode.
- If pre-CLT input matches but preactivation / margin differs, focus on CLT
  encode precision / determinism.
- If preactivation matches but mask/post-mask differs, focus on boundary logic.

## Validation

- Extend existing cross-cluster debug artifact tests to check new fields are
  preserved through summary / stream emission.
- Extend CLT diagnostic tests to assert new Phase-0 fingerprint fields exist and
  reflect post-zero-position masking.
- Only run safe local validation (`uv run ...`), no GPU workloads outside SLURM.
