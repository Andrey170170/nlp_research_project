# Phase 2 Campaign Orchestration Spec

## 1. Problem statement

After the exact-trace eval harness and frozen workspace execution model exist, the next step is **campaign orchestration over isolated experimental branches**, not an always-on autonomous research daemon.

We need a control plane that can launch bounded parallel experiments, compare outcomes with the standardized eval harness, and keep long-running work safe under SLURM. Workspace freezing is necessary, but not sufficient; the main deliverable in this phase is control-plane discipline around isolated workspaces, job tracking, and evaluation gating.

## 2. Scope and non-goals

### Scope

- Controller-driven campaigns over frozen project workspaces.
- One agent = one hypothesis, one workspace, one bounded task, one evaluation plan.
- Parallel execution of isolated branches/workspaces with eval-based comparison.
- SLURM-owned runtime lifecycle, with a poller/state tracker for progress and recovery.
- Integration with existing foundations: standardized eval harness, extraction/aggregation, compact graph comparison, frozen project workspace snapshots, frozen sibling library snapshots, and import/environment resolution from frozen workspaces.

### Non-goals

- No free-form autonomous agent loop.
- No recursive agent spawning.
- No auto-merge.
- No always-on LLM session managing the run lifecycle.
- No aggressive queue heuristics.
- No broad open-ended research search.

## 3. Proposed approach

### Controller responsibilities

- Define campaign lanes, budgets, and stop conditions.
- Materialize frozen workspaces and attach each agent to exactly one.
- Submit SLURM jobs, track state transitions, and resume from artifacts.
- Trigger eval runs and compare outcomes across branches.
- Enforce gating rules before allowing a branch to advance.

### Agent contract

- Input: a single hypothesis, workspace snapshot, bounded task, and evaluation plan.
- Behavior: execute until a terminal agent state is reached; do not stop mid-task to ask for permission.
- Output: artifacts, logs, summaries, and a terminal result.
- Terminal agent states: `SUBMITTED_JOBS`, `NO_IMPROVEMENT`, `BLOCKED_HARD`, `FAILED_VALIDATION`.

### Workspace / execution model

- Each agent operates in an immutable workspace snapshot.
- Library dependencies are also frozen from sibling snapshots.
- Code/import resolution must come from the frozen workspace view, not the live repo.
- Long-running lifecycle is owned by SLURM plus a lightweight poller/state tracker, not by a live model session.

### Manifests / state model

Each campaign and run should emit manifests that capture:

- campaign id, agent id, hypothesis id
- workspace snapshot path and dependency snapshot path
- task spec, eval plan, and budget limits
- submitted job ids
- artifact locations and comparison outputs

Minimal job/run states:

- `pending`
- `running`
- `completed`
- `failed`
- `oom`
- `timed_out`
- `cancelled`

These are job states only; they must be kept distinct from agent terminal states above.

### Evaluation and gating

- Every lane ends with standardized eval plus graph comparison.
- Promotion decisions are based on both runtime and semantic preservation.
- `94_base` remains a blocker/gating check for replay/scheduling changes; do not treat it as a normal optimization target.
- Any change that improves runtime but shifts graph similarity needs explicit promotion criteria, not silent acceptance.

### Initial experiment lanes

Start with bounded parallel lanes:

1. Hidden-knob exposure/sweeps.
2. Memory bottleneck instrumentation/profiling.
3. Prompt-94 anomaly investigation.
4. Frontier-preserving replay-locality/scheduling variants, but only after graph-compare gating is working.

## 4. Acceptance criteria

- A campaign can launch multiple isolated branches/workspaces and track them end-to-end.
- Each agent runs one bounded task to a terminal state without live-session intervention.
- Job and agent states are separately represented and recoverable from manifests.
- Eval output is sufficient to compare branches and gate promotion.
- `94_base` is enforced as a blocker for replay/scheduling changes.
- No recursive spawning, auto-merge, or always-on loop is required for the first usable version.

## 5. Risks and open questions

- Promotion criteria when runtime improves but graph similarity shifts.
- Retry/resubmission policy after `failed`, `oom`, or `timed_out`.
- Controller crash recovery and idempotency.
- Concurrency and budget limits per campaign lane.
- Partial artifact handling and how much must be retained for comparison.
- When, if ever, agents may propose new branches of search instead of staying within an assigned lane.
