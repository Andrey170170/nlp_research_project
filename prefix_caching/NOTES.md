# Working notes — Direction 2 prefix caching

Running scratchpad of concepts, decisions, and gotchas from building the
forward-pass prefix cache.  Updated as we go.  Not the spec — that's
[BRIEF.md](BRIEF.md).

## Vocabulary

### `kwarg` — keyword argument

A function parameter passed **by name** rather than by position.

```python
# positional:
attribute(prompt, model)

# kwarg:
attribute(prompt, model, prefix_cache=my_cache)
#                        ^^^^^^^^^^^^ kwarg
```

When the design says *"thread the `prefix_cache=` kwarg through
`attribute()`"* it means:

1. Add `prefix_cache=None` as a new parameter to the outer function
2. Also add it to each nested function that's called from it
3. Each level passes it down the chain

Default value `None` = "no cache, behave exactly as the library did
before."  Callers that don't know about the cache keep working unchanged.

### Sparsification

Post-forward-pass filter that decides which features count as "active."
If it picks the top-K globally across all positions, the boundary shifts
each time the prompt grows, so the **same underlying activations** can
look like slightly different feature sets from step to step.  This is
why our baseline run shows 99.9% feature match but only 30–70% position
match: individual features are identical, but the per-position "set of
active features" fluctuates at the edge.

**Decision:** cache the raw *pre*-sparsification MLP activations, not
the post-sparsification feature list.  Sparsification re-runs fresh each
step.  This keeps the cached trace byte-identical to an uncached trace,
which is what the brief demands.

### Phase 0 / Phase 3 / Phase 4

Stages inside `circuit_tracer`'s attribution pipeline:
- **Phase 0** — the forward pass: runs the model, collects MLP
  input/output at each position, encodes through transcoders to get the
  activation matrix.  **This is where our cache plugs in.**
- **Phase 3** — logit attribution.
- **Phase 4** — feature attribution via chunked decoder.
- (There are 1 and 2 internally but they're fast.)

Phases 3 and 4 are where the backward-pass reuse (stretch goal) would
plug in.  Not our target for Milestone 2.

## Architecture reminders

### Two repos, two workflows

- **Library** (`circuit-tracer_chunked`) — we have a fork under
  `jjivandas`.  All caching code goes here.
- **Experiment harness** (`nlp_research_project`) — we push directly
  to Andrei's repo on the `jay/prefix-caching` branch, no fork.

### Why the cache goes in the library

`nlp_research_project` is just the SLURM scripts and evaluation code.
The actual model/attribution logic (where the cache plugs in) lives in
`circuit-tracer_chunked`.  Keeping the cache in the library means it
becomes reusable by other teams using circuit-tracer, not tangled up
with our eval harness.

### No editing code mid-run

`nnsight` (the tracing engine that circuit-tracer wraps) reads the
Python source files at runtime.  If we change them while a SLURM job is
running, the job crashes.  Rules:
- **Safe to edit any time:** new files that nothing imports yet
  (e.g. `prefix_cache.py`, `test_prefix_cache.py`)
- **Only edit when no job is in `R` state:** existing library files
  (`attribute.py`, `attribute_nnsight.py`, `replacement_model_nnsight.py`)
- **SLURM `.sbatch` files:** safe any time (SLURM snapshots at submit)

## SLURM know-how

### Commands

```bash
squeue -M ascend -u $USER                                     # active jobs
sacct -M ascend -j <id> --format=JobID,State,ExitCode,Elapsed # history
tail -f logs/slurm-<jobname>-<id>.out                         # live stdout
tail -f logs/slurm-<jobname>-<id>.err                         # live stderr
```

Login nodes are **CPU-only**.  GPU happens on compute nodes, reached via
`sbatch`.  You submit on login; SLURM schedules to a compute node.

### Known-good config for prompt 94 on Ascend

From the working `scripts/trace_prefix_cache_bench.jay.ascend.sbatch`:

```
partition=quad          (A100-80GB + 1 TiB RAM)
cpus-per-task=10
mem=700G
gpus-per-task=1
time=03:00:00           (produces ~5 steps; 20 steps would need ~12h)
```

Attribution flags:

```
--attribution-batch-size 128
--feature-batch-size 128
--logit-batch-size 128
--decoder-chunk-size 2048
--cross-batch-decoder-cache-bytes 12884901888   # 12 GiB
--max-feature-nodes 8192
--max-edges 20000
--max-n-logits 3
--desired-logit-prob 0.8
```

The standard Ascend partition OOM-kills prompt 94 during Phase 0 setup
(335 GiB host RAM isn't enough for the ~3.37M active features × the
necessary scaffolding).  Always use quad for long prompts.

### Output layout on scratch

```
/fs/scratch/PAS2136/jjivandas/temporal_prefix_cache/
    baseline_exact/       # uncached gold standard
    forward_cache/        # cached runs for comparison
    backward_reuse/       # (stretch, later)
    analysis/             # fidelity reports
```

## Current status

| Milestone | State | Notes |
|---|---|---|
| M1 — Baseline understanding | done | library mapped, baseline run produced 5 step files on prompt 94 |
| M2 — Forward cache path | in progress | `prefix_cache.py` written; kwarg-threading still to do |
| M3 — Diagnostics | bundled into M2 | `CacheDiagnostics` dataclass in `prefix_cache.py` |
| M4 — Backward reuse | stretch | deferred until M2 shows fidelity |
| M5 — Evaluation | 20% | baseline exists; need cached run for comparison |
| M6 — Recommendation | not started | last step |

## What the baseline run told us

Job 5011889 ran to its 3h time limit, produced 5 trace steps on prompt
94.  Key numbers from the stdout log:

```
Step 000: features=3371343 attrib=649.5s   (~11 min, no prior cache)
Step 001: pos_rate=0.7105 feat_rate=0.9992 attrib=2408.0s  (~40 min)
Step 002: pos_rate=0.7013 feat_rate=0.9992 attrib=584.8s   (~10 min)
Step 003: pos_rate=0.2949 feat_rate=0.9990 attrib=2474.3s  (~41 min)
Step 004: pos_rate=0.2911 feat_rate=0.9991 attrib=2536.8s  (~42 min)
```

Takeaways:
1. Feature match rate is **~99.9%** every step — underlying activations
   are the same.  Confirms the caching premise.
2. Position match rate drops to ~30% at step 3 — sparsification is
   reshuffling which features count as "active" at each position.
   Confirms our decision to cache pre-sparsification tensors and
   re-sparsify every step.
3. Per-step time varies 4×.  Average ~35 min/step.  Full 20 steps need
   ~12h.  For fidelity measurement, 5 steps is plenty.
