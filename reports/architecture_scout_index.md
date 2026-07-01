# Architecture scouting report index

Date: 2026-06-30

Scope: read-only scouting/research for architecture cleanup of the sibling
`../circuit-tracer_chunked` library and this repo's exact-trace harness. The
sibling library is the priority.

Method: followed the requested `improve-codebase-architecture` vocabulary:
**module**, **interface**, **depth**, **seam**, **adapter**, **leverage**,
**locality**, **deletion test**, and **shallow module**. No code was changed and
no GPU/model-loading work was run.

Reports:

- `reports/circuit_tracer_architecture_scout.md` — primary findings for the
  sibling `circuit_tracer` library.
- `reports/harness_architecture_scout.md` — secondary findings for this repo's
  exact-trace harness.
- `reports/architecture_review.html` — visual summary of the top deepening
  candidates.

## Top recommendation

Start with `../circuit-tracer_chunked/circuit_tracer/attribution/attribute_nnsight.py`.
It is the highest-leverage target because it currently concentrates exact-trace
orchestration, phase policy, row-store storage, replay validation, telemetry,
debug bundle capture, prefix-view handling, and session reuse in one mega-module.
The current top-level NNSight attribution interface has dozens of knobs, many of
which belong to separate policy groups. Deepening this area should produce the
largest improvement in readability and changeability without needing to redesign
the whole library first.

## Candidate priority ladder

1. **Strong: NNSight attribution deepening**
   - Files: `attribute_nnsight.py`, `context_nnsight.py`, adjacent tests.
   - Why first: largest file, widest interface, most exact-trace fork logic.

2. **Strong: exact-trace policy/config seam**
   - Files: `attribute.py`, `attribute_nnsight.py`, harness runner/knob taxonomy.
   - Why second: the same policy surface leaks into the harness and causes 50+
     argument forwarding.

3. **Strong: row-store / replay / prefix-view locality**
   - Files: `attribute_nnsight.py`, `context_nnsight.py`, tests around row-store,
     replay validation, prefix metadata, full-sequence sessions.
   - Why: these are research-critical invariants with good test footholds.

4. **Worth exploring: Cross-layer transcoder runtime helpers**
   - Files: `transcoder/cross_layer_transcoder.py`.
   - Why: core math, decoder cache, diagnostics, and loading are mixed.

5. **Worth exploring: Harness full-answer runner**
   - Files: `src/nlp_research_project/exact_trace_bench/full_answer/runner.py`.
   - Why: biggest harness seam with sibling internals; should improve after the
     library exposes deeper modules.

## Documentation note

There is no obvious ADR process in either repo. For this project, the useful
documentation source is `docs/`, root `README.md`/`AGENTS.md`/`EXPERIMENTS.md`,
and `experiments/logs/*.jsonl`. Those docs are valuable but sometimes stale, so
architecture work should treat them as context to reconcile, not as binding
architecture decisions unless they match current code and experiment logs.

The sibling library has much less architecture documentation: mainly `README.md`,
`RESEARCH_USAGE.md`, and `circuit_tracer/utils/MAPPING_INFO.md`.

## Non-goals for this pass

- No implementation.
- No detailed new interfaces yet.
- No GPU/model-loading validation.
- No broad scratch/log filesystem analysis.
