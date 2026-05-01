# Direction 2: Cross-Run Prefix Caching — Implementation Plan

## What we're building

A validation system that proves consecutive tracing steps share identical prefix features, and measures how much redundant work exists. No changes to Andrei's fork. All new files.

## Background (why this works)

When the tracing loop generates token-by-token, each step's input is the previous step's input plus one new token appended at the end. Inside `attribute()`, the model runs forward and records which features are active at each position. Because transformers process left-to-right (each position only sees tokens before it), features at position P only depend on tokens 0..P. Adding a new token at the end cannot change features at earlier positions. This means most of the work `attribute()` does on step N+1 was already done on step N.

## What we're NOT doing

- Not editing `trace_pipeline_chunked.py` or any existing file
- Not modifying the circuit-tracer fork
- Not trying to actually skip computation yet (that requires fork changes from Andrei)

## Files

| File | Purpose | Status |
|------|---------|--------|
| `prefix_caching/__init__.py` | Makes the folder importable as a Python package | DONE |
| `prefix_caching/cache.py` | `PrefixCache` class — stores features per position, compares across steps | DONE |
| `prefix_caching/trace_pipeline_cached.py` | Tracing loop with cache validation (calls `attribute()` normally, then checks cache) | DONE |
| `scripts/trace_prefix_cache_bench.ascend.sbatch` | SLURM script to run the cached pipeline on OSC | DONE |
| `tests/test_prefix_cache_correctness.py` | Offline check that all prefix features matched | DONE |

## How it works

### `PrefixCache` (cache.py)

Three methods:
- `store(input_token_ids, active_features)` — after each step, groups features by position and saves them
- `compare(fresh_active_features)` — checks whether cached features at each prefix position match what `attribute()` just recomputed. Returns a `CompareResult` with match counts and rates
- `clear()` — resets between completions

The `active_features` tensor has shape `(F, 3)` where each row is `[layer, position, feature_idx]`. The cache groups these into sets keyed by position, so comparison is a set equality check per position.

### Tracing loop (trace_pipeline_cached.py)

Same flow as `trace_completion_compact_chunked` but wraps each step:
1. Call `attribute()` normally (full computation, no shortcuts)
2. Compare result against cache from previous step
3. Store current result in cache for next step
4. Log match stats + wall-clock timing

Outputs:
- Normal trace artifacts (`step_NNN.npz`, `completion.json`) — same as existing pipeline
- `cache_validation.json` — per-step comparison stats (the new thing)

### SLURM job

Runs on Ascend (A100), GSM8K prompt index 94, temperature 0, 20 steps, batch size 128, chunk size 2048. These settings are known to work from Wave-1 benchmarks.

## Step-by-step to run

### Step 1: Push code to OSC

```bash
git add prefix_caching/ scripts/trace_prefix_cache_bench.ascend.sbatch tests/test_prefix_cache_correctness.py
git commit -m "Direction 2: prefix cache validation scaffold"
git push
```

### Step 2: Submit the job

```bash
ssh <your-user>@ascend.osc.edu
cd /path/to/nlp_research_project
git pull
mkdir -p logs
sbatch scripts/trace_prefix_cache_bench.ascend.sbatch
```

No new environment setup needed — the job uses the same `uv run` and `.venv` as all existing scripts. The only new dependency is our `prefix_caching/` package which is pure Python (no pip install required, it's imported directly from the repo).

### Step 3: Check job status

```bash
squeue -u $USER
```

### Step 4: When it finishes, check results

```bash
uv run python tests/test_prefix_cache_correctness.py \
    --results-dir /fs/scratch/PAS3272/kopanev.1/prefix_cache_bench/prompt_000/completion_000
```

### Step 5: Look at raw data

```bash
cat /fs/scratch/PAS3272/kopanev.1/prefix_cache_bench/prompt_000/completion_000/cache_validation.json
```

## What the results tell us

From `cache_validation.json`, each step after the first has a `comparison` block:
- `position_match_rate: 1.0` means every prefix position's features were identical → caching is safe
- `feature_match_rate` tells us what fraction of all features were already known → this is the redundancy %
- `attribution_seconds` shows how long each step took → baseline for future speedup measurement

## Next steps (after validation passes)

### Step 6: Write spec for Andrei

Once we have numbers proving the cache is correct:
- "Here's the cache object, here's what's in it"
- "Here's proof that prefix features don't change (our validation results)"
- "Here's the interface we want: `attribute()` accepts `prefix_cache=`, skips Phase 0 for cached positions"

### Step 7: If Andrei wires up the fork

We update `trace_pipeline_cached.py` to actually pass the cache into `attribute()` and measure real speedups.

## What we'd report in the final paper

- The empirical proof that prefix features are invariant across steps
- The measured redundancy percentage (e.g. "at step 20, X% of Phase 0 work is redundant")
- The projected speedup if Phase 0 were skipped for cached positions
- If Andrei wires up his side in time: actual speedup measurements

## Risks

- If `attribute()` doesn't return per-position feature data in a way we can slice — we handle this because `active_features` has `[layer, position, feature_idx]` rows, so position grouping works directly
- If features DON'T match across steps — that would mean the transformer isn't position-local, which would be surprising and itself a finding worth reporting
