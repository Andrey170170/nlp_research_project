# CHPC Resource Pools

Status: Current submit-target notes
Last updated: 2026-07-11

This page records practical GPU pools for the large-scale circuit-tracing
harness. Values come from Slurm `sinfo`, `scontrol`, and account associations on
2026-07-09. Re-check before long campaigns.

## Operational Job Classes

Use these classes in run names, descriptions, logs, and planning docs. The older
`fast` / `anomaly` / `long_eval` labels remain in scenario generators, but they
should not be the main project-level split for new CHPC work.

| Class | Use |
|---|---|
| `setup_prefetch` | Download Gemma/GemmaScope weights and build reusable caches |
| `smoke` | Minimal GPU/model/transcoder load and one-scenario tracing checks |
| `baseline` | Rebuild canonical validated traces for a new machine or code state |
| `sweep` | Harness/resource/knob sweeps across many scenarios |
| `long_trace` | Large-context, large-graph, or high-memory single traces |
| `full_answer` | Multi-token/full-answer tracing campaigns |
| `analysis` | Extraction, comparison, calibration, plotting, and reports |

## Preferred Large-Trace Pools

These are the best candidates when a single GPU needs at least 80GB VRAM and the
job benefits from about 1T or more host RAM.

| Priority | Cluster | Account | Partition / QOS | Candidate hardware | Node RAM | Walltime | Notes |
|---|---|---|---|---|---:|---:|---|
| 1 | `granite` | `rai` | `rai-gpu-grn` | `grn027`-`grn032`: `8x h200` | ~2.044T | 14d | Best default for large traces if available. |
| 2 | `granite` | `marasovic` or guest access | `granite-gpu-guest` | same H200 nodes exposed as guest | ~2.044T | 3d | Flexible fallback, shorter walltime. |
| 3 | `granite` | `marasovic` or guest access | `granite-gpu-guest` | `grn067`: `4x h200nvl` | ~1.532T | 3d | Good H200NVL high-memory fallback. |
| 4 | `granite` | `marasovic` or guest access | `granite-gpu-guest` | `grn080`: `2x h200nvl` | ~2.300T | 3d | Very high host RAM; availability may vary. |
| 5 | `notchpeak` | `owner-gpu-guest` | `notchpeak-gpu-guest` | `notch366`: `8x a100_80gb_pcie` | ~1.532T | 3d | 80GB A100 pool, good if H200 is busy. |
| 6 | `notchpeak` | `owner-gpu-guest` | `notchpeak-gpu-guest` | `notch347`/`notch348`: A100 nodes | ~1.020T | 3d | A100 nodes with 1T host RAM; VRAM size not encoded in GRES. |

## Lab-Owned A100 Pool

| Cluster | Account | Partition / QOS | Nodes | Hardware | Node RAM | Walltime | Use |
|---|---|---|---|---|---:|---:|---|
| `notchpeak` | `marasovic-gpu-np` | `marasovic-gpu-np` | `notch369`, `notch370` | `2x a100`, 64 CPU | ~252G | 12h | Lab smokes, moderate baselines, debugging; not the 1T+ host-RAM path. |

The lab A100 partition is real and directly associated with this user. Slurm
showed both nodes as `MIXED` on 2026-07-09, each partly allocated.

## Other Useful Pools

| Cluster | Account | Partition / QOS | Hardware | Node RAM | Walltime | Use |
|---|---|---|---|---:|---:|---|
| `granite` | `marasovic` | `granite-gpu` | public `h100nvl` / `l40s` | ~380G-764G | 3d | H100/L40S work that does not need 1T host RAM. |
| `granite` | `rai` | `rai-gpu-grn-short` | H200 pool, limited QOS | max `1 GPU`, `12 CPU`, `250G` | 8h | Quick H200 smoke/load tests. |
| `notchpeak` | `notchpeak-gpu` | `notchpeak-gpu` | V100/2080Ti/3090/A100 mix | ~188G-508G | 3d | Lower-priority fallback. |
| `kingspeak` | `kingspeak-gpu` | `kingspeak-gpu` | TitanX | ~60G | 3d | Not suitable for current large GemmaScope traces. |
| `lonepeak` | `lonepeak-gpu` | `lonepeak-gpu` | 1080Ti | ~60G | 3d | Not suitable for current large GemmaScope traces. |

## Validated 1B Gate Sizing

For H200 `361_base` validation with 16 CPUs:

| Variant | Minimum request | Evidence |
|---|---:|---|
| Gemma 3 1B CLT, batch 1000 | `64G` | Job `1613108` reached essentially all of a `32G` request and entered a materially slower reclaim regime. |
| Gemma 3 1B PLT, batch 128 | `100G` | Job `1613109` used about 67.2 GiB MaxRSS and completed within the baseline timing range. |

These are validation floors, not governor estimates. Preserve additional
headroom for profiling, larger fixtures, or concurrent artifact capture.

## Submit Examples

RAI H200 large trace:

```bash
sbatch \
  --account=rai \
  --partition=rai-gpu-grn \
  --qos=rai-gpu-grn \
  --gres=gpu:h200:1 \
  --mem=1000G \
  slurm/exact_trace_bench/trace_weekend_exact_chunked.granite.sbatch
```

Granite guest H200/H200NVL fallback:

```bash
sbatch \
  --account=marasovic \
  --partition=granite-gpu-guest \
  --qos=granite-gpu-guest \
  --gpus-per-task=1 \
  --mem=1000G \
  slurm/exact_trace_bench/trace_weekend_exact_chunked.granite.sbatch
```

Lab A100 smoke:

```bash
sbatch \
  -M notchpeak \
  --account=marasovic-gpu-np \
  --partition=marasovic-gpu-np \
  --qos=marasovic-gpu-np \
  --gres=gpu:a100:1 \
  --mem=220G \
  slurm/exact_trace_bench/trace_weekend_exact_chunked.granite.sbatch
```

If running from Notchpeak, keep code and cache paths on shared filesystems that
the compute nodes can see. The current cache root under
`/scratch/general/vast/$USER/nlp_research_project` is intended for Granite;
verify visibility/performance from Notchpeak before relying on it for a long
campaign.
