# Direction 2: Cross-Run Prefix Caching — Implementation Plan

## What we're building

A validation system that proves consecutive tracing steps share identical prefix features, and measures how much redundant work exists. No changes to Andrei's fork. All new files.

## Background (why this works)

When the tracing loop generates token-by-token, each step's input is the previous step's input plus one new token appended at the end. Inside `attribute()`, the model runs forward and records which features are active at each position. Because transformers process left-to-right (each position only sees tokens before it), features at position P only depend on tokens 0..P. Adding a new token at the end cannot change features at earlier positions. This means most of the work `attribute()` does on step N+1 was already done on step N.

## What we're NOT doing

- Not editing `trace_pipeline_chunked.py` or any existing file
- Not modifying the circuit-tracer fork
- Not trying to actually skip computation yet (that requires fork changes from Andrei)

## Files we'll create

| File | Purpose |
|------|---------|
| `prefix_cache.py` | The cache class that stores and compares per-position features |
| `trace_pipeline_cached.py` | A new tracing entrypoint that uses the cache for validation (calls `attribute()` normally, then checks cache correctness) |
| `scripts/trace_prefix_cache_bench.ascend.sbatch` | SLURM script to run the cached pipeline on OSC |
| `tests/test_prefix_cache_correctness.py` | Offline comparison of cached vs computed outputs |

## Step-by-step

### Step 1: Build `prefix_cache.py`

A class called `PrefixCache` with three methods:
- `store(input_token_ids, active_features, activation_values)` — after each step, save the features per position
- `lookup(input_token_ids)` — given the next step's tokens, find how many prefix positions are already cached and return their stored features
- `compare(cached_features, fresh_features)` — check whether the cached values match what `attribute()` just recomputed, return match stats (how many matched, how many didn't, percentage of work that was redundant)

The cache holds one entry at a time (the most recent step). Each call to `store` overwrites the previous entry.

### Step 2: Build `trace_pipeline_cached.py`

A tracing loop similar to `trace_completion_compact_chunked` but:
- Creates a `PrefixCache` before the loop starts
- Each step: calls `attribute()` normally (full computation, no shortcuts)
- After getting results: calls `cache.compare()` to check cached prefix features against freshly computed ones
- Then calls `cache.store()` to save this step's features for next time
- Logs per-step: number of cached positions, number that matched, percentage redundant, wall-clock time of attribution
- Saves a `cache_validation.json` at the end with all the per-step stats

This gives us two things:
1. Proof that caching is correct (features match)
2. Measurement of how much work is redundant (the number we report)

### Step 3: Write the SLURM script

`scripts/trace_prefix_cache_bench.ascend.sbatch` that:
- Runs `trace_pipeline_cached.py` on one GSM8K prompt (index 94, shortest of the three benchmarked prompts)
- Temperature 0 (deterministic, so results are reproducible)
- Max 20 steps (enough to show the pattern without burning hours of GPU time)
- Outputs to scratch under a `prefix_cache_bench/` directory

### Step 4: Push to OSC and run

- `git push`
- SSH into OSC, `sbatch` the job
- Wait for it to finish, pull `cache_validation.json` back

### Step 5: Analyze results

Look at `cache_validation.json`:
- Do cached features match at every step? (correctness gate)
- What fraction of features are reusable at step 5, 10, 15, 20? (redundancy measurement)
- How does attribution wall time grow with step number? (baseline for later speedup comparison)

### Step 6: Write spec for Andrei

Once we have numbers proving the cache is correct, write a short spec doc:
- "Here's the cache object, here's what's in it"
- "Here's proof that prefix features don't change (our validation results)"
- "Here's the interface we want: `attribute()` accepts `prefix_cache=`, skips Phase 0 for cached positions"

This gives Andrei everything he needs to wire it up on his side.

## What we'd report in the final paper

- The empirical proof that prefix features are invariant across steps
- The measured redundancy percentage (e.g. "at step 50, 97% of Phase 0 work is redundant")
- The projected speedup if Phase 0 were skipped for cached positions
- If Andrei wires up his side in time: actual speedup measurements

## Risks

- If `attribute()` doesn't return per-position feature data in a way we can slice (it might return a flat tensor) — we'll need to figure out the position mapping from `active_features` tensor format, which is `[layer, position, feature_idx]` per row, so this should be fine
- If features DON'T match across steps — that would mean the transformer isn't position-local, which would be surprising and itself a finding worth reporting
