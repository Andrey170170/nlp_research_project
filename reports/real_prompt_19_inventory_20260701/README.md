# Real-prompt 19-pair trace inventory

Status: neutral inventory for analysis-design discussion.  This directory contains no role labels, role-matched pairs, graph metrics, scorecards, or conclusions.

Created: 2026-07-01T20:19:45.476102-04:00

## Counts

- Prompt pairs: `19`
- Trajectories: `38`
- Token graph rows: `7714`
- Token rows by wave: `{"v1_prior9":3723,"v2_combined10":3991}`
- All trajectories aggregate-ok: `True`

## Files

- `prompt_pairs.md` — human-readable 19-row prompt-pair table.
- `prompt_pairs.csv` — prompt-pair table with source paths.
- `trajectories.csv` — one row per correct/wrong trajectory, including run roots and audit paths.
- `tokens.csv` — one row per generated token/graph with token text and `graph.npz` path.
- `trajectory_texts.md` — generated completions for role-design inspection.
- `trajectory_texts.jsonl` — machine-readable generated completions.
- `source_manifest.json` — source paths, counts, and warnings.

## Source waves

- `v1_prior9`: first real-prompt wave, 9 completed prompt pairs, from `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/real-prompt-screen-v1/trace_launch_prep_fullseq_wreuse_w1600`.
- `v2_combined10`: second/carryover wave, 10 completed prompt pairs, from `/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/real-prompt-screen-v2/trace_launch_prep_fullseq_wreuse_w1600_combined10`.

Prior unified role-matched pct20 metric output, intentionally not reused as canonical inventory:
`/fs/scratch/PAS2836/kopanev.1/exact_trace_bench/cardinal/fast/real-prompt-role-matched-pct20-19prompts-20260626`

## Important boundary

Use this as the stable substrate for discussion.  Any future role labels should live in a separate versioned table keyed by `(wave, prompt_id, label, generated_index)` or by `trajectory_name + generated_index`.  Any future pair manifests or graph metrics should be derived from this inventory plus an explicit role/metric version.
