# Scripts archive

Current exact-bench execution should use the packaged CLI:

```bash
uv run exact-trace-bench --help
```

Canonical SLURM templates live under `slurm/exact_trace_bench/` and are normally
rendered/submitted by the CLI so jobs run from immutable workspace snapshots.

Files under `scripts/archive/` are historical executable provenance or temporary
compatibility helpers. Do not use them as current launch templates without first
checking current docs and defaults.
