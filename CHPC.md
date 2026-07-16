# CHPC Project Operations

Status: Current operational guidance
Last live verification: 2026-07-16

This is the practical Utah CHPC guide for this repository. It covers routine
GPU selection, request sizing, walltime, and launch hygiene. Hardware inventory
and queue state change continuously: `mychpc batch --gpus`, `freegpus`, and
`sbatch --test-only` override any stale availability statement in this file.

## Non-Negotiable Rules

- Never load Gemma/GemmaScope models or run tracing on a login node.
- Run reproducible experiments through the project harness and ordinary Slurm,
  not through the `chpc` interactive-development helper.
- Every Slurm run that executes project code uses one verified read-only
  project+sibling snapshot. Pass absolute source and snapshot paths.
- Keep model caches, environments, secrets, outputs, and temporary row stores
  outside the snapshot.
- Record the project and sibling commits, snapshot manifest, scenario file,
  output root, GPU model, and Slurm job IDs.
- Do not mix test-bed artifacts into the H200 baseline. Hardware changes are an
  experimental factor and require an explicit bridge comparison.

## Evidence Classes

| Class | Purpose | Hardware rule |
|---|---|---|
| CPU development | Unit tests, schema checks, planning, lint, typing | Interactive CPU allocation or small login-safe command |
| Smoke | Imports, model/provider loading, one short trace, failure/telemetry paths | Any suitable GPU; label hardware and do not promote graph metrics |
| Characterization | Timing, memory, and knob behavior | Keep one GPU model fixed within the campaign |
| Baseline | Canonical graph and runtime reference | Full RAI H200 unless the baseline definition explicitly changes |
| Hardware bridge | Quantify cross-GPU drift | Fixed tracing knobs and paired artifacts on both GPU models |

## Default GPU Routing

These are starting envelopes, not substitutes for telemetry or governor
admission. Request the smallest shape that still leaves reasonable failure
headroom.

| Work | Preferred GPU | CPU / host RAM | Initial walltime | Notes |
|---|---|---:|---:|---|
| Login-safe tests | CPU | 1-4 CPU / 8-16G | 10-30m | No model or transcoder loading |
| 1B PLT load/telemetry smoke | H200 MIG 18GB or lab A100 | 12 CPU / 64-100G | 20-30m | Use a single scenario; batch 64-128 |
| 1B PLT functional trace | Lab A100 | 12 CPU / 100-160G | 30-60m | Stable owner route; not an H200 parity result |
| 1B CLT functional trace | RTX PRO 6000 Blackwell or lab A100 with reduced Phase-1 batch | 12 CPU / 150-200G | 30-60m | Canonical `b1000` has an approximately 89GiB CUDA-reserved peak |
| Canonical 1B CLT baseline | Full H200 | 12 CPU / 200G | 1h | Preserve `b1000` and H200 hardware |
| Canonical 1B PLT baseline | Full H200 | 12 CPU / 200G | 1-1.5h | Preserve `b128`; include export time |
| 1B upward sweep row | Full H200 | 12 CPU / 200G | CLT 1h; PLT 2h | Array concurrency `%1` on short QOS |
| 4B PLT baseline/sweep | Full H200 | 12 CPU / 300-400G | 2h | Requires regular RAI QOS; short QOS caps memory at 250G |
| 12B PLT baseline/sweep | Full H200 | 12 CPU / 400G | 4-6h | Start at 5h until calibrated |
| Long trace/full answer | Full H200 | 12 CPU / 400G+ | Measured p95 + margin | Use regular QOS and checkpointable units where possible |

For a CLT smoke on a GPU below 96GB, reduce the Phase-1 source batch and use
the bounded/tiled storage mechanisms. That is a functional test configuration,
not the canonical baseline.

## Pool Guidance

### RAI H200

Use `rai-gpu-grn` for canonical baselines, calibration, 4B/12B work, and any
result intended for scientific comparison. Full H200 nodes have about 2TB host
RAM. The `rai-gpu-grn-short` QOS is useful for jobs up to 8h, but is limited to
one GPU, 12 CPUs, and 250G RAM per job. Use regular `rai-gpu-grn` when the host
memory or walltime exceeds those limits.

The 18GB H200 MIG slice (`h200_1g.18gb`) is a good same-architecture PLT smoke
target. It has fractional compute and cannot run the canonical 1B CLT shape.
Avoid launching MIG smokes into the same short-QOS user limit as an active H200
array campaign unless the queue interaction is intentional.

### RTX PRO 6000 Blackwell

Granite has 96GB RTX PRO 6000 Blackwell Max-Q GPUs (`rtxpr6000bl`). The SOC
class nodes `grn073`-`grn075` each expose eight GPUs and about 1.5TB host RAM.
The current environment uses PyTorch 2.10 + CUDA 12.8 and includes `sm_120`
support.

The associated route is:

```text
account=marasovic
partition=soc-gpu-class-grn
qos=soc-gpu-students-grn
gres=gpu:rtxpr6000bl:1
```

Use this owner-priority route only when the SOC class allocation policy permits
the research workload. The same GPUs are exposed through
`granite-gpu-guest`, but that route is preemptable. Blackwell is an excellent
1B functional test bed and can fit the observed CLT peak, but its artifacts are
not interchangeable with the H200 baseline.

### Lab A100

The lab-owned `marasovic-gpu-np` partition contains `notch369` and `notch370`,
each with two A100 GPUs, 64 CPUs, and about 252G host RAM. This is the preferred
stable route for routine 1B PLT development and reduced-memory CLT smokes. Run a
short `nvidia-smi` probe before defining hardware-specific presets because the
generic `a100` GRES does not encode VRAM size.

### Guest And Public GPUs

H100NVL, H200NVL, A100, and A6000 guest pools are useful only for opportunistic
smokes or explicit hardware bridges. Guest jobs can be preempted without a
usable artifact; pending or preempted tasks are infrastructure outcomes, not
scientific evidence. Public 3090/V100/T4 routes are fallback load-test targets,
not exact-trace performance references.

GH200 is excluded from the normal routing policy because its Grace CPU is ARM;
the current x86 environment cannot be reused directly.

## Walltime Policy

Use prior successful telemetry from the same GPU, model/provider, prompt class,
and mechanism family. A practical initial request is:

```text
walltime = round_up_15m(max(15m, 1.3 * observed_p95 + cold_cache_allowance))
```

Use 10-20 minutes of cold-cache allowance when model/provider weights may not be
resident. After two representative successes:

- if elapsed time is below 25% of the request, reduce the next request;
- if elapsed time is above 80%, increase the next request by 25-50%;
- if the run times out, inspect the last completed phase before changing both
  walltime and mechanisms;
- never use a multi-hour request for a one-minute smoke merely because the
  template permits it.

Array rows should be independently restartable. Keep `%1` where required by
QOS policy or when one-at-a-time execution protects cache, host RAM, or
comparison consistency.

## Launch Checklist

1. Classify the run as `smoke`, `baseline`, `sweep`, `long_trace`,
   `full_answer`, or `analysis`.
2. Select the GPU from the routing table and keep the evidence class explicit
   in `run_name`, `run_description`, and `run_goal`.
3. Refresh access and idle inventory with `mychpc batch --gpus` and `freegpus`.
4. Render the harness launch plan and inspect GPU, CPU, RAM, walltime, output
   root, and array throttle.
5. Run `sbatch --test-only` with the exact final scheduler shape.
6. Create or reuse one verified immutable project+sibling snapshot.
7. Submit, record the job IDs, and confirm `scontrol show job` matches the plan.
8. Inspect incremental logs and telemetry before treating scheduler completion
   as a passed validation gate.

Detailed node inventory and submit examples remain in
`docs/chpc_resource_pools.md`; environment and cache setup remains in
`docs/chpc_setup.md`.
