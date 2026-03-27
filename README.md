# Weekend exact chunked benchmark commands

Generate the fixed wave-1 scenario files locally first:

```bash
uv run python experiments/build_weekend_exact_chunked_benchmark_configs.py
```

Prepare the late-prefix fixtures on Ascend:

```bash
sbatch --time=01:00:00 scripts/prepare_weekend_prefix_fixtures.ascend.sbatch
```

Run wave 1 on Ascend:

```bash
sbatch --time=02:30:00 --array=0-$(($(uv run python experiments/print_scenario_count.py --scenarios-file experiments/generated/weekend_exact_chunked_wave1_ascend_scenarios.json)-1)) --export=ALL,SCENARIOS_FILE=experiments/generated/weekend_exact_chunked_wave1_ascend_scenarios.json,OUTPUT_ROOT=/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked/ascend/wave1 scripts/trace_weekend_exact_chunked.ascend.sbatch
```

Run wave 1 on Cardinal:

```bash
sbatch --time=02:00:00 --array=0-$(($(uv run python experiments/print_scenario_count.py --scenarios-file experiments/generated/weekend_exact_chunked_wave1_cardinal_scenarios.json)-1)) --export=ALL,SCENARIOS_FILE=experiments/generated/weekend_exact_chunked_wave1_cardinal_scenarios.json,OUTPUT_ROOT=/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked/cardinal/wave1 scripts/trace_weekend_exact_chunked.cardinal.sbatch
```

After wave 1, copy and edit the follow-up selection file, then regenerate follow-up configs:

```bash
cp experiments/weekend_exact_chunked_followup_selection.template.json experiments/weekend_exact_chunked_followup_selection.json
uv run python experiments/build_weekend_exact_chunked_benchmark_configs.py --selection-file experiments/weekend_exact_chunked_followup_selection.json
```

Run wave 2 cache sweeps on Ascend:

```bash
sbatch --time=01:30:00 --array=0-$(($(uv run python experiments/print_scenario_count.py --scenarios-file experiments/generated/weekend_exact_chunked_wave2_ascend_scenarios.json)-1)) --export=ALL,SCENARIOS_FILE=experiments/generated/weekend_exact_chunked_wave2_ascend_scenarios.json,OUTPUT_ROOT=/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked/ascend/wave2 scripts/trace_weekend_exact_chunked.ascend.sbatch
```

Run wave 2 cache sweeps on Cardinal:

```bash
sbatch --time=01:00:00 --array=0-$(($(uv run python experiments/print_scenario_count.py --scenarios-file experiments/generated/weekend_exact_chunked_wave2_cardinal_scenarios.json)-1)) --export=ALL,SCENARIOS_FILE=experiments/generated/weekend_exact_chunked_wave2_cardinal_scenarios.json,OUTPUT_ROOT=/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked/cardinal/wave2 scripts/trace_weekend_exact_chunked.cardinal.sbatch
```

Run late-prefix validation on Ascend:

```bash
sbatch --time=02:00:00 --array=0-$(($(uv run python experiments/print_scenario_count.py --scenarios-file experiments/generated/weekend_exact_chunked_validation_ascend_scenarios.json)-1)) --export=ALL,SCENARIOS_FILE=experiments/generated/weekend_exact_chunked_validation_ascend_scenarios.json,OUTPUT_ROOT=/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked/ascend/validation scripts/trace_weekend_exact_chunked.ascend.sbatch
```

Run late-prefix validation on Cardinal:

```bash
sbatch --time=01:30:00 --array=0-$(($(uv run python experiments/print_scenario_count.py --scenarios-file experiments/generated/weekend_exact_chunked_validation_cardinal_scenarios.json)-1)) --export=ALL,SCENARIOS_FILE=experiments/generated/weekend_exact_chunked_validation_cardinal_scenarios.json,OUTPUT_ROOT=/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked/cardinal/validation scripts/trace_weekend_exact_chunked.cardinal.sbatch
```
