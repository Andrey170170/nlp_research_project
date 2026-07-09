# CHPC Granite Setup

Status: Current CHPC setup notes
Last updated: 2026-07-09

## Workspace

Required layout:

```text
/uufs/chpc.utah.edu/common/home/$USER/projects/
├── nlp_research_project/
└── circuit-tracer_chunked/
```

The project environment installs `../circuit-tracer_chunked` as an editable
dependency. Keep both checkouts cleanly recorded before serious runs:

```bash
git status --short --branch
git -C ../circuit-tracer_chunked status --short --branch
```

## Python

Use the CHPC system Python 3.12 with `uv`. Keep the `uv` cache local to the
project or scratch so setup does not depend on home-cache behavior:

```bash
UV_CACHE_DIR=.uv-cache uv sync
.venv/bin/python --version
.venv/bin/exact-trace-bench verify-imports
```

On the current CHPC login environment, `uv run` can print command output and
then fail to exit promptly. Use `uv sync` for installation, then prefer the
`.venv/bin/...` entrypoints for runtime checks and Granite SLURM jobs.

For direct sibling-library work:

```bash
cd ../circuit-tracer_chunked
UV_CACHE_DIR=../nlp_research_project/.uv-cache uv sync --extra dev
```

## Scratch and Caches

Current default scratch root:

```text
/scratch/general/vast/$USER/nlp_research_project/exact_trace_bench
```

Override when needed:

```bash
export EXACT_TRACE_BENCH_SCRATCH_ROOT=/some/other/exact_trace_bench
```

Granite SLURM templates source `slurm/exact_trace_bench/chpc_env.sh`, which
defaults large runtime caches to:

```text
/scratch/general/vast/$USER/nlp_research_project/uv-cache
/scratch/general/vast/$USER/nlp_research_project/huggingface
/scratch/general/vast/$USER/nlp_research_project/matplotlib
/scratch/general/vast/$USER/nlp_research_project/tmp
```

## SLURM Profiles

Use the harness cluster key `granite` for new CHPC work:

```bash
UV_CACHE_DIR=.uv-cache uv run exact-trace-bench build-scenarios \
  --cluster granite \
  --all-tiers

UV_CACHE_DIR=.uv-cache uv run exact-trace-bench launch-plan \
  --cluster granite \
  --scenarios-file experiments/generated/exact_trace_bench/exact_trace_bench_fast_granite_scenarios.json \
  --immutable-workspace
```

Granite templates default to:

| Workload | Account | Partition | QOS |
|---|---|---|---|
| GPU traces/fixtures/full-answer | `marasovic` | `granite-gpu-guest` | `granite-gpu-guest` |
| CPU transcoder prefetch | `marasovic` | `granite-guest` | `granite-guest` |

If you want the RAI H200 partition, override at submit time:

```bash
sbatch --account=rai --partition=rai-gpu-grn --qos=rai-gpu-grn ...
```

See `docs/chpc_resource_pools.md` for the current candidate GPU pools and
job-class routing.

## Weight Prefetch

Do not download model or transcoder weights on login nodes. Use SLURM:

```bash
sbatch slurm/exact_trace_bench/download_model_and_transcoders.granite.sbatch
```

By default this prefetches:

- `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `google/gemma-3-12b-it`
- `gemmascope2-clt-1b-medium-affine`
- `gemmascope2-plt-1b-small`, `gemmascope2-plt-4b-small`,
  `gemmascope2-plt-12b-small`

For transcoder-only prefetches:

```bash
sbatch slurm/exact_trace_bench/download_transcoders.granite.sbatch
```

The `.env` file should contain `HF_TOKEN=...` if gated Hugging Face access is
needed. The job loads it without printing secrets.

Model weights for `google/gemma-3-1b-it` are normally pulled by the first
model-loading SLURM job into `HF_HOME`. To pre-seed from OSC instead, copy the
old Hugging Face cache into:

```text
/scratch/general/vast/$USER/nlp_research_project/huggingface
```

## Likely Local-Only Artifacts To Copy From OSC

These are intentionally ignored by git but may be needed for local continuity:

- `.env` with `HF_TOKEN` and any local API keys.
- Hugging Face cache for Gemma/GemmaScope weights.
- Previous scratch outputs if they are needed for baselines, comparisons, or
  extraction.
- `experiments/extracted/`, `experiments/figures/`, and `logs/` if you need old
  local analysis products.
- Any local `experiments/generated/**` files that were never committed.

Generated trace artifacts, raw `.pt` graphs, and large scratch trees should stay
out of git.
