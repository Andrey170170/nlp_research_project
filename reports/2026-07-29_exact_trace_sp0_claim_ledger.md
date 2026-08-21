# Exact-trace SP0 incumbent and mechanism claim ledger

Status: SP0 complete; LS0 login-safe manifest scaffold implemented

Date: 2026-07-29

Plan: `plans/2026-07-29_exact_trace_performance_optimization_loop.md`

## Outcome

The short-prefix incumbent and every retained optimization mechanism now have a
stable machine-readable identity and disposition in
`experiments/performance_campaigns/sp0_mechanism_claims.json`. The ledger is
validated by the `exact-trace-perf validate-claims` command and focused tests.
It does not modify the frozen scientific baseline registry or promote a launch
default, governor decision, or fidelity scope.

The exact incumbent remains mapped selective decoder rows with lazy encoder
placement, b64/c4096, and the canonical CPU reduction path. CUDA full
residency, CUDA windowing, prepared CPU rows, and active-CPU encoder placement
remain explicit bounded research mechanisms. The exact signed-host row mirror
and inferior short-prefix execution envelopes remain rejected.

## Same-regime short-prefix controls

| Case | Subprocess | Completion | Phase 0 | Phase 4 | Peak CUDA reserved | Evidence |
|---|---:|---:|---:|---:|---:|---|
| 1B PLT `361_base` | 108.62s | unavailable | unavailable | unavailable | unavailable | registered same-regime mechanism control |
| 4B PLT `361_base` repeat 1 | 345.39s | 310.83s | 128.51s | 158.37s | 26.24 GiB | compact-strict pass |
| 4B PLT `361_base` repeat 2 | 234.46s | 206.61s | 39.89s | 149.12s | 26.24 GiB | compact-strict pass |
| 12B PLT `361_base` repeat 1 | 586.30s | 555.62s | 63.43s | 458.42s | 38.82 GiB | exact repeat reference |
| 12B PLT `361_base` repeat 2 | 563.27s | 539.42s | 62.59s | 443.74s | 38.82 GiB | exact repeat reference |

Cache state was warm or uncontrolled. These values are durable controls, not
claims of cold-cache timing.

## Disposition summary

| Mechanism | Disposition | Remaining formal evidence |
|---|---|---|
| mapped selective decoder rows | `exact_promotion_candidate` | SP4.2 and SP5 |
| mapped lazy c4096 incumbent | `exact_opt_in` | SP5 repetitions |
| checkpoint lifecycle release/advice | `exact_promotion_candidate` | SP4.2 |
| phase-scoped telemetry | `exact_promotion_candidate` | SP4.1 |
| active-CPU encoder residency | `bounded_research` | SP1 localization |
| full CUDA row consumer | `bounded_research` | longer-prefix envelope |
| windowed CUDA row consumer | `bounded_research` | longer-prefix envelope |
| prepared CPU row consumer | `bounded_research` | optional scaling characterization |
| exact signed-host row mirror | `rejected` | none; exact but slower |
| active pinned, b128/b256, larger tiles | `rejected` | none without a new bottleneck |

## LS0 scaffold

`experiments/performance_campaigns/ls0_prefix_scaling.json` expresses planned
124/256/512/1,024-token development workloads with typed model/provider,
prompt role, trajectory, target, resource, profile, and evidence fields. It can
be listed without importing or loading a model. Its strict `--require-frozen`
gate intentionally fails until the generated trajectory and all immutable
token fingerprints are populated.

Validation at this milestone:

- 122 project tests passed;
- affected-file Ruff passed; and
- both committed JSON documents passed `jq`.

## Promotion separation

SP0 registers evidence only:

- mechanism selection: recorded per claim;
- same-regime mechanism control: recorded separately;
- scientific baseline: unchanged;
- launch default: unchanged;
- governor or calibration state: unchanged; and
- bounded fidelity authorization: unchanged.
