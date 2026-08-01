# Canonical exact-trace fidelity taxonomy

Date: 2026-07-31

Status: user-selected canonical taxonomy; this decision supersedes the earlier
binary interpretation in which any compact-graph difference forced an
otherwise practically exact mechanism into the bounded category.

## Scope of the claim

Fidelity names describe agreement of the persisted compact graph contract:
feature identities, retained edge identities and weights, edge signs, ranked
edge support, and target token. They do not claim that every uncaptured raw
intermediate tensor is identical.

`strict_exact` compares semantic artifact fields directly. It does not require
the surrounding `.npz` container bytes or serialization metadata to be
identical when the decoded arrays are identical.

## Fidelity levels

| Level | Required compact-graph agreement | Interpretation |
|---|---|---|
| `strict_exact` | feature IDs, edge indices, weights, signs, ranks, and target are field-for-field identical; support metrics `1.0`; normalized L1 `0.0` | Audit/reference equality under the compact artifact contract |
| `exact` | feature, edge, and weighted-edge agreement at least `0.995`; normalized magnitude and signed L1 at most `0.005`; target token, shared signs, and Top-256 support exact | Practically identical; differences are limited floating-point boundary effects |
| `close` | feature, edge, and weighted-edge agreement at least `0.99`; normalized magnitude and signed L1 at most `0.01`; target token exact | Small but visible numerical variation |
| `bounded` | feature, edge, and weighted-edge agreement at least `0.98`; normalized magnitude and signed L1 at most `0.02`; target token exact | Meaningfully approximate but explicitly controlled |
| `best_effort` | soft fidelity scoring without a guaranteed hard floor | Optimization is allowed to trade additional fidelity for resources or runtime |
| `research` | unrestricted, fully labeled observations | Exploratory evidence only |

Thresholds apply to the worst required metric and worst evaluated step, not an
average that can hide an outlier. Provider- or campaign-specific contracts may
add stronger Top-K or resource constraints but must not silently weaken these
floors.

## Promotion and retention policy

Every mechanism that meets `bounded` or a stronger level is retained as a
supported selectable solution. Only `strict_exact` and `exact` results are
eligible for automatic promotion consideration. Exactness is an admission
condition for promotion, not the promotion decision itself: an eligible
candidate must still be selected using runtime, memory, provider compatibility,
prompt-length admission constraints, and the strength and breadth of its
evidence. Automatic promotion does not require one mechanism to be the single
global default for every workload.

`close` and `bounded` implementations remain explicit selectable modes but are
not candidates for automatic default promotion. They may be chosen when the
caller requests that fidelity or when a separately reviewed policy explicitly
permits it.

Mechanisms below the `bounded` floor remain in experiment evidence but are not
automatically registered as defaults. Default selection must expose the
effective fidelity level and preserve atomic fallback to an admitted stronger
mode when a resource or compatibility check refuses a faster mechanism.

An `exact` label therefore permits, but never compels, promotion. Scientific
reports must record the measured metrics, evaluated workload scope, scaling
limits, resource results, and the comparative reason for selecting or not
selecting an eligible exact candidate.

## Orthogonal execution modes

Fidelity is separate from implementation and placement. Existing selectable
execution choices remain independent axes, including:

- feature-row influence: `cpu_exact`, `cpu_prepared`, `cuda_full`,
  `cuda_windowed`, and `auto`;
- encoder placement: `lazy`, `active_cpu`, and `active_pinned_cpu`;
- Phase 0: canonical full traversal or selective mapped decoder rows; and
- provider/model/prompt-length admission and resource budgets.

Names such as `canonical`, `selective`, `cpu_exact`, or `cuda_windowed`
describe how a result is computed. The fidelity label describes how closely
that result matches its declared reference.

## Current-result interpretation

- Canonical full traversal is `strict_exact` when its compact fields match.
- Selective Phase 0 is `exact` on the measured prompt-breadth result: feature
  Jaccard `0.999756`, edge Jaccard `0.997004`, weighted-edge Jaccard
  `0.996279`, signed normalized L1 `0.003728`, exact Top-256 support, exact
  shared signs, and exact target token.
- The active-CPU result is `exact`: feature/edge/weighted Jaccard
  `0.999512/0.999500/0.999529` and signed normalized L1 `0.000471`.
- The measured CUDA-full/prepared family near `0.989` broad graph agreement is
  `bounded` and therefore remains supported as an explicit selectable mode
  under its resource/admission contract, but is not automatically promotable.

Longer-prefix LS evaluation must continue reporting drift. A mechanism keeps
its promoted fidelity label only over the workload scope for which the floors
have been demonstrated; new scaling evidence may broaden that scope or trigger
fallback without erasing the retained implementation.
