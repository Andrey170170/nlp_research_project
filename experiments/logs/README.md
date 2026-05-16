# Experiment logs

Status: Current append-only structured experiment record area

Use monthly JSONL files for structured records that would make root
`EXPERIMENTS.md` too long.

Conventions:

- one JSON object per line,
- include `date`, `type`, `summary`, `project`, and `sibling_library` when known,
- include scratch roots, snapshot paths, SLURM job IDs, and validation metrics when
  they matter,
- keep narrative interpretation in `EXPERIMENTS.md` only when it changes the
  current baseline or research interpretation.

The pre-split long-form narrative is preserved at:

- `docs/history/experiment_log_legacy_2026-05-16.md`
