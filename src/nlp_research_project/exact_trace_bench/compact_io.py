"""
Shared utilities for temporal circuit stability analysis.

Provides graph sparsification, compact serialisation, and temporal
metric computation used by both the tracing pipeline and analysis scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch


# ── configuration ────────────────────────────────────────────────────
STABLE_CORE_WINDOW = 10  # W for stable-core detection
STABLE_CORE_PERSISTENCE = 0.8  # p threshold
MAX_EDGES = 10_000  # default cap on retained edges per step
FEATURE_ID_BASE = 1_000_000
CANONICAL_TYPED_BUCKET_NAMES = (
    "feature<-feature",
    "feature<-error",
    "feature<-token",
    "logit<-feature",
    "logit<-error",
    "logit<-token",
)


# ── per-step compact data ────────────────────────────────────────────


@dataclass
class StepData:
    """Compact representation of a single generation step's circuit."""

    step_idx: int
    # Sparse edge representation (COO format)
    row_idx: np.ndarray  # int32
    col_idx: np.ndarray  # int32
    weights: np.ndarray  # float32, normalised to sum=1
    # Feature identity
    feature_ids: np.ndarray  # (F, 3) int64: [layer, position, feature_idx]
    # Generation metadata
    token_text: str
    logprob: float | None
    n_features: int


@dataclass
class BucketedCompact:
    step: StepData
    bucket_row_idx: np.ndarray
    bucket_col_idx: np.ndarray
    bucket_weights: np.ndarray
    bucket_ids: np.ndarray
    bucket_names: np.ndarray
    bucket_metadata_json: str
    error_node_shape: np.ndarray
    token_ids: np.ndarray
    logit_token_ids: np.ndarray


@dataclass(frozen=True)
class CompactGraph:
    """A compact graph loaded and validated at the serialization seam."""

    path: Path
    step: StepData
    compact_save_format: str | None
    bucket_row_idx: np.ndarray | None
    bucket_col_idx: np.ndarray | None
    bucket_weights: np.ndarray | None
    bucket_ids: np.ndarray | None
    bucket_names: tuple[str, ...]
    bucket_metadata: dict[str, dict[str, Any]]
    error_node_shape: tuple[int, ...] | None
    token_ids: np.ndarray | None
    logit_token_ids: np.ndarray | None


@dataclass(frozen=True)
class FeaturePositionSummary:
    feature_count: int
    typed_feature_endpoint_count: int
    max_position: int | None
    future_position_count: int

    @property
    def has_future_positions(self) -> bool:
        return self.future_position_count > 0


# ── graph helpers ────────────────────────────────────────────────────


def extract_circuit_subgraph(
    adj: torch.Tensor, n_features: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract feature→feature and feature→logit edges from full adjacency.

    Error and token (embed) nodes carry extreme gradient magnitudes (up to
    1e38, plus NaN/Inf) that swamp the real circuit.  Restricting to
    feature-only edges gives a clean, interpretable subgraph.

    Returns (feat_block, logit_block, logit_start_row).
    """
    logit_start = adj.shape[0] - 2  # circuit-tracer puts logit nodes last
    feat_block = adj[:n_features, :n_features]
    logit_block = adj[logit_start:, :n_features]
    return feat_block, logit_block, logit_start


# NEW ── decoder-aware importance scoring ──────────────────────────────────
def decoder_aware_feature_scores(
    adj: torch.Tensor,
    activation_values: torch.Tensor,
    n_features: int,
) -> torch.Tensor:
    """Feature importance: s_i = a_i^2 * ||D_i||_F^2.

    D_i is the downstream write vector for feature i — column i of the
    feature-subgraph adjacency (feat→feat and feat→logit blocks).
    ||D_i||_F^2 captures how broadly feature i writes downstream;
    a_i^2 weights by firing strength.  Product ranks features by their
    actual causal impact on the traced output.

    Returns a (n_features,) float32 score tensor.
    """
    feat_block, logit_block, _ = extract_circuit_subgraph(adj, n_features)
    fb = feat_block.float()
    lb = logit_block.float()
    d_norms_sq = fb.pow(2).sum(dim=0) + lb.pow(2).sum(dim=0)  # (F,)
    a_sq = activation_values[:n_features].float().pow(2)  # (F,)
    return a_sq * d_norms_sq


# NEW
def _sparsify_decoder_aware(
    feat_block: torch.Tensor,
    logit_block: torch.Tensor,
    logit_start: int,
    activation_values: torch.Tensor,
    n_features: int,
    max_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select edges by ranking features with s_i = a_i^2 * ||D_i||_F^2.

    Features are sorted by score descending.  Their outgoing edges
    (column non-zeros in the feature and logit blocks) are accumulated in
    that order until the edge budget is exhausted.  If the top-scored
    feature alone exceeds the budget, its edges are trimmed by absolute
    weight.
    """
    fb = feat_block.float()
    lb = logit_block.float()

    d_norms_sq = fb.pow(2).sum(dim=0) + lb.pow(2).sum(dim=0)  # (F,)
    a_sq = activation_values[:n_features].float().pow(2)  # (F,)
    scores = a_sq * d_norms_sq  # (F,)

    ranked = torch.argsort(scores, descending=True)

    # Greedy budget: count non-zero outgoing edges per feature
    nz_per_feat = (fb != 0).sum(dim=0) + (lb != 0).sum(dim=0)  # (F,)
    cumsum = torch.cumsum(nz_per_feat[ranked].float(), dim=0)
    n_keep = int((cumsum <= max_edges).sum().item())
    n_keep = max(n_keep, 1)  # always include at least the top-scored feature

    kept_feats = ranked[:n_keep]  # (n_keep,) original feature indices
    keep_mask = torch.zeros(n_features, dtype=torch.bool)
    keep_mask[kept_feats] = True
    kept_col_order = torch.where(keep_mask)[0]  # stable original-index order

    # Feature → feature edges
    ff_r, ff_c = (fb[:, keep_mask] != 0).nonzero(as_tuple=True)
    ff_col_orig = kept_col_order[ff_c]
    ff_vals = fb[ff_r, ff_col_orig].abs()

    # Feature → logit edges
    fl_r, fl_c = (lb[:, keep_mask] != 0).nonzero(as_tuple=True)
    fl_col_orig = kept_col_order[fl_c]
    fl_vals = lb[fl_r, fl_col_orig].abs()

    row_idx = torch.cat([ff_r, fl_r + logit_start]).to(torch.int32)
    col_idx = torch.cat([ff_col_orig, fl_col_orig]).to(torch.int32)
    vals = torch.cat([ff_vals, fl_vals])

    # Trim to budget (handles single-feature overshoot)
    if vals.numel() > max_edges:
        _, top_idx = torch.topk(vals, k=max_edges, sorted=False)
        row_idx = row_idx[top_idx]
        col_idx = col_idx[top_idx]
        vals = vals[top_idx]

    if vals.numel() == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    total = float(vals.double().sum().item())
    if total == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    norm_weights = (vals.double() / total).float().numpy()
    return row_idx.numpy(), col_idx.numpy(), norm_weights


# END NEW


def sparsify_edges(
    adj: torch.Tensor,
    n_features: int,
    max_edges: int = MAX_EDGES,
    activation_values: torch.Tensor | None = None,  # NEW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select feature-circuit edges, normalised to sum=1.

    When *activation_values* are provided uses the decoder-aware score
        s_i = a_i^2 * ||D_i||_F^2
    to rank features (D_i = downstream write vector = column i of the
    feature-subgraph adjacency).  Edges are included in feature-score
    order until the budget is reached.

    Without activation_values falls back to top-K by absolute edge weight.

    Returns (row_idx, col_idx, weights) in the *original* adjacency
    matrix coordinate system.
    """
    feat_block, logit_block, logit_start = extract_circuit_subgraph(adj, n_features)

    # NEW: dispatch to decoder-aware path when activations available
    if activation_values is not None:
        return _sparsify_decoder_aware(
            feat_block,
            logit_block,
            logit_start,
            activation_values,
            n_features,
            max_edges,
        )
    # END NEW

    # Legacy path: top-K edges by absolute weight
    ff_flat = feat_block.abs().float().view(-1)
    fl_flat = logit_block.abs().float().view(-1)
    combined = torch.cat([ff_flat, fl_flat])

    k = min(max_edges, int((combined != 0).sum().item()))
    if k == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    topk_vals, topk_idx = torch.topk(combined, k, sorted=False)

    topk64 = topk_vals.double()
    kept_mass = float(topk64.sum().item())
    if kept_mass == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    ff_size = ff_flat.numel()
    n_feat_cols = feat_block.shape[1]
    n_logit_cols = logit_block.shape[1]

    rows: list[int] = []
    cols: list[int] = []
    in_ff = topk_idx < ff_size

    # Feature→feature indices
    ff_idx = topk_idx[in_ff]
    rows.extend((ff_idx // n_feat_cols).tolist())
    cols.extend((ff_idx % n_feat_cols).tolist())

    # Feature→logit indices (offset into logit block)
    fl_idx = topk_idx[~in_ff] - ff_size
    rows.extend((fl_idx // n_logit_cols + logit_start).tolist())
    cols.extend((fl_idx % n_logit_cols).tolist())

    norm_weights = (topk64 / kept_mass).float().numpy()
    return (
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        norm_weights,
    )


# ── compact serialisation ────────────────────────────────────────────


_BASE_COMPACT_FIELDS = frozenset(
    {
        "row_idx",
        "col_idx",
        "weights",
        "feature_ids",
        "token_text",
        "logprob",
        "n_features",
        "step_idx",
    }
)
_TYPED_COMPACT_FIELDS = frozenset(
    {
        "bucket_row_idx",
        "bucket_col_idx",
        "bucket_weights",
        "bucket_ids",
        "bucket_names",
        "bucket_metadata_json",
        "error_node_shape",
        "token_ids",
        "logit_token_ids",
    }
)
_STEP_PATH_RE = re.compile(r"(?:token_|step_)(\d+)")


def _require_scalar(data: Any, field: str) -> np.ndarray:
    value = np.asarray(data[field])
    if value.ndim != 0:
        raise ValueError(f"{field} must be a scalar")
    return value


def _require_integer_scalar(data: Any, field: str) -> np.ndarray:
    value = _require_scalar(data, field)
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{field} must be an integer scalar")
    return value


def _require_integer_vector(data: Any, field: str) -> np.ndarray:
    value = np.asarray(data[field])
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{field} must be a one-dimensional integer array")
    return value


def _step_index_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = _STEP_PATH_RE.fullmatch(Path(part).stem)
        if match:
            return int(match.group(1))
    return None


def _load_bucket_metadata(
    raw: np.ndarray, *, names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    try:
        rows = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("bucket_metadata_json must contain valid JSON") from exc
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("bucket_metadata_json must contain a list of objects")
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = row.get("bucket")
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("each bucket metadata row must identify a bucket")
        if bucket in metadata:
            raise ValueError(f"duplicate bucket metadata for {bucket!r}")
        metadata[bucket] = row
    if set(metadata) != set(names):
        raise ValueError("bucket metadata names must match bucket_names")
    return metadata


def _metadata_number(row: dict[str, Any], field: str, *, bucket: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"bucket metadata {field} must be numeric for {bucket!r}")
    converted = float(value)
    if not np.isfinite(converted) or converted < 0:
        raise ValueError(
            f"bucket metadata {field} must be finite and nonnegative for {bucket!r}"
        )
    return converted


def _validate_bucket_metadata(
    metadata: dict[str, dict[str, Any]],
    *,
    names: tuple[str, ...],
    bucket_ids: np.ndarray,
    bucket_weights: np.ndarray,
) -> None:
    required = {
        "bucket",
        "raw_total_abs_mass",
        "retained_abs_mass",
        "retained_fraction",
        "raw_nnz",
        "retained_nnz",
        "policy",
        "weights_signed",
    }
    for bucket_id, name in enumerate(names):
        row = metadata[name]
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"bucket metadata is missing required fields for {name!r}: {missing}"
            )
        retained_nnz = row["retained_nnz"]
        raw_nnz = row["raw_nnz"]
        if (
            isinstance(retained_nnz, bool)
            or not isinstance(retained_nnz, int)
            or retained_nnz < 0
        ):
            raise ValueError(f"bucket metadata retained_nnz is invalid for {name!r}")
        actual_nnz = int(np.count_nonzero(bucket_ids == bucket_id))
        if retained_nnz != actual_nnz:
            raise ValueError(
                f"bucket metadata retained_nnz does not match edges for {name!r}"
            )
        if (
            isinstance(raw_nnz, bool)
            or not isinstance(raw_nnz, int)
            or raw_nnz < retained_nnz
        ):
            raise ValueError(
                f"bucket metadata raw_nnz must be an integer >= retained_nnz for {name!r}"
            )

        raw_mass = _metadata_number(row, "raw_total_abs_mass", bucket=name)
        retained_mass = _metadata_number(row, "retained_abs_mass", bucket=name)
        tolerance = max(1e-9, 1e-9 * raw_mass)
        if retained_mass > raw_mass + tolerance:
            raise ValueError(
                f"bucket metadata retained mass exceeds raw mass for {name!r}"
            )
        actual_retained_mass = float(
            np.abs(bucket_weights[bucket_ids == bucket_id]).astype(np.float64).sum()
        )
        if not np.isclose(retained_mass, actual_retained_mass, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"bucket metadata retained mass does not match weights for {name!r}"
            )

        fraction = row["retained_fraction"]
        if raw_mass == 0.0:
            if retained_mass != 0.0 or fraction is not None:
                raise ValueError(
                    f"bucket metadata retained_fraction must be None for zero raw mass in {name!r}"
                )
        else:
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                raise ValueError(
                    f"bucket metadata retained_fraction must be numeric for {name!r}"
                )
            fraction_value = float(fraction)
            expected_fraction = retained_mass / raw_mass
            if (
                not np.isfinite(fraction_value)
                or not 0.0 <= fraction_value <= 1.0
                or not np.isclose(
                    fraction_value, expected_fraction, rtol=1e-9, atol=1e-12
                )
            ):
                raise ValueError(
                    f"bucket metadata retained_fraction is inconsistent for {name!r}"
                )

        policy = row["policy"]
        if not isinstance(policy, dict) or set(policy) != {"top_p", "cap"}:
            raise ValueError(
                f"bucket metadata policy must contain top_p and cap for {name!r}"
            )
        top_p = policy["top_p"]
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not np.isfinite(float(top_p))
            or not 0.0 < float(top_p) <= 1.0
        ):
            raise ValueError(f"bucket metadata policy top_p is invalid for {name!r}")
        cap = policy["cap"]
        if cap is not None and (
            isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
        ):
            raise ValueError(f"bucket metadata policy cap is invalid for {name!r}")
        if not isinstance(row["weights_signed"], bool):
            raise ValueError(
                f"bucket metadata weights_signed must be bool for {name!r}"
            )


def _feature_endpoint_positions(values: np.ndarray, *, n_pos: int) -> np.ndarray:
    if n_pos <= 0:
        raise ValueError("typed feature endpoints require a positive position domain")
    block = n_pos * FEATURE_ID_BASE
    return (np.asarray(values, dtype=np.int64) % block) // FEATURE_ID_BASE


def _validate_typed_endpoint_domains(
    *,
    names: tuple[str, ...],
    rows: np.ndarray,
    cols: np.ndarray,
    bucket_ids: np.ndarray,
    error_node_shape: np.ndarray,
    token_ids: np.ndarray,
    logit_token_ids: np.ndarray,
    feature_ids: np.ndarray,
) -> None:
    if error_node_shape.shape != (2,):
        raise ValueError("error_node_shape must contain exactly two dimensions")
    n_pos = len(token_ids)
    if int(error_node_shape[-1]) != n_pos:
        raise ValueError("error_node_shape position dimension must match token_ids")
    error_count = int(np.prod(error_node_shape, dtype=np.int64))
    if feature_ids.size and (
        int(feature_ids[:, 1].max()) >= n_pos
        or int(feature_ids[:, 2].max()) >= FEATURE_ID_BASE
    ):
        raise ValueError("feature_ids is outside the typed feature identity domain")
    encoded_selected_features = set(
        (
            feature_ids[:, 0] * n_pos * FEATURE_ID_BASE
            + feature_ids[:, 1] * FEATURE_ID_BASE
            + feature_ids[:, 2]
        ).tolist()
    )
    for bucket_id, name in enumerate(names):
        mask = bucket_ids == bucket_id
        bucket_rows = rows[mask]
        bucket_cols = cols[mask]
        if name.startswith("feature<-"):
            _feature_endpoint_positions(bucket_rows, n_pos=n_pos)
            if any(
                int(value) not in encoded_selected_features for value in bucket_rows
            ):
                raise ValueError(
                    f"{name} target row is not a persisted selected feature"
                )
        elif name.startswith("logit<-") and bucket_rows.size:
            if int(bucket_rows.max()) >= len(logit_token_ids):
                raise ValueError(f"{name} row is outside logit_token_ids")

        if name.endswith("<-feature"):
            _feature_endpoint_positions(bucket_cols, n_pos=n_pos)
            if any(
                int(value) not in encoded_selected_features for value in bucket_cols
            ):
                raise ValueError(
                    f"{name} source column is not a persisted selected feature"
                )
        elif name.endswith("<-error") and bucket_cols.size:
            if int(bucket_cols.max()) >= error_count:
                raise ValueError(f"{name} column is outside error_node_shape")
        elif name.endswith("<-token") and bucket_cols.size:
            if int(bucket_cols.max()) >= len(token_ids):
                raise ValueError(f"{name} column is outside token_ids")


def load_historical_compact_graph(
    path: Path,
    *,
    expected_step_idx: int | None = None,
    validate_step_path: bool = True,
) -> CompactGraph:
    """Load and validate one historical legacy or mixed compact artifact.

    New-run artifacts must use ``typed_compact_graph.load_typed_compact_graph``.
    This explicitly named adapter exists only for provenance-bound artifacts
    written before the canonical typed-only v2 schema.
    """
    graph_path = Path(path)
    with np.load(str(graph_path), allow_pickle=False) as data:
        fields = set(data.files)
        missing = sorted(_BASE_COMPACT_FIELDS - fields)
        if missing:
            raise ValueError(f"compact graph is missing required fields: {missing}")

        row_idx = _require_integer_vector(data, "row_idx").copy()
        col_idx = _require_integer_vector(data, "col_idx").copy()
        weights = np.asarray(data["weights"])
        if weights.ndim != 1 or not np.issubdtype(weights.dtype, np.floating):
            raise ValueError("weights must be a one-dimensional floating array")
        weights = weights.copy()
        if len({len(row_idx), len(col_idx), len(weights)}) != 1:
            raise ValueError("compact COO arrays have mismatched lengths")
        if row_idx.size and int(row_idx.min()) < 0:
            raise ValueError("row_idx contains a negative node index")
        if col_idx.size and int(col_idx.min()) < 0:
            raise ValueError("col_idx contains a negative node index")
        if not np.isfinite(weights).all():
            raise ValueError("weights contains a non-finite value")

        n_features = int(_require_integer_scalar(data, "n_features"))
        if n_features < 0:
            raise ValueError("n_features must be non-negative")
        feature_ids = np.asarray(data["feature_ids"])
        if (
            feature_ids.ndim != 2
            or feature_ids.shape != (n_features, 3)
            or not np.issubdtype(feature_ids.dtype, np.integer)
        ):
            raise ValueError("feature_ids must be an (n_features, 3) integer array")
        feature_ids = feature_ids.copy()
        if feature_ids.size and int(feature_ids.min()) < 0:
            raise ValueError("feature_ids contains a negative identity value")
        if len({tuple(int(value) for value in row) for row in feature_ids}) != len(
            feature_ids
        ):
            raise ValueError("feature_ids must contain unique identities")
        if col_idx.size and int(col_idx.max()) >= n_features:
            raise ValueError("col_idx is outside the feature-node domain")

        step_idx = int(_require_integer_scalar(data, "step_idx"))
        if step_idx < 0:
            raise ValueError("step_idx must be non-negative")
        path_step_idx = (
            _step_index_from_path(graph_path) if validate_step_path else None
        )
        required_step_idx = (
            int(expected_step_idx) if expected_step_idx is not None else path_step_idx
        )
        if required_step_idx is not None and step_idx != required_step_idx:
            expected_label = (
                f"expected step {required_step_idx}"
                if expected_step_idx is not None
                else f"path index {required_step_idx}"
            )
            raise ValueError(
                f"graph step_idx {step_idx} does not match {expected_label}: "
                f"{graph_path}"
            )

        token_text_raw = _require_scalar(data, "token_text")
        if token_text_raw.dtype.kind not in {"U", "S"}:
            raise ValueError("token_text must be a string scalar")
        logprob = float(_require_scalar(data, "logprob"))
        if np.isinf(logprob):
            raise ValueError("logprob must be finite or NaN when unavailable")
        step = StepData(
            step_idx=step_idx,
            row_idx=row_idx,
            col_idx=col_idx,
            weights=weights,
            feature_ids=feature_ids,
            token_text=str(token_text_raw),
            logprob=logprob if not np.isnan(logprob) else None,
            n_features=n_features,
        )

        typed_fields_present = _TYPED_COMPACT_FIELDS & fields
        save_format = (
            str(_require_scalar(data, "compact_save_format"))
            if "compact_save_format" in fields
            else None
        )
        is_typed = bool(typed_fields_present) or save_format is not None
        if not is_typed:
            return CompactGraph(
                path=graph_path,
                step=step,
                compact_save_format=None,
                bucket_row_idx=None,
                bucket_col_idx=None,
                bucket_weights=None,
                bucket_ids=None,
                bucket_names=(),
                bucket_metadata={},
                error_node_shape=None,
                token_ids=None,
                logit_token_ids=None,
            )
        missing_typed = sorted(_TYPED_COMPACT_FIELDS - fields)
        if missing_typed:
            raise ValueError(
                f"typed compact graph is missing required fields: {missing_typed}"
            )
        if save_format not in {None, "typed_bucketed"}:
            raise ValueError(f"unsupported compact_save_format: {save_format!r}")

        bucket_row_idx = _require_integer_vector(data, "bucket_row_idx").copy()
        bucket_col_idx = _require_integer_vector(data, "bucket_col_idx").copy()
        bucket_ids = _require_integer_vector(data, "bucket_ids").copy()
        bucket_weights = np.asarray(data["bucket_weights"])
        if bucket_weights.ndim != 1 or not np.issubdtype(
            bucket_weights.dtype, np.floating
        ):
            raise ValueError("bucket_weights must be a one-dimensional floating array")
        bucket_weights = bucket_weights.copy()
        if (
            len(
                {
                    len(bucket_row_idx),
                    len(bucket_col_idx),
                    len(bucket_weights),
                    len(bucket_ids),
                }
            )
            != 1
        ):
            raise ValueError("typed bucket COO arrays have mismatched lengths")
        if bucket_row_idx.size and int(bucket_row_idx.min()) < 0:
            raise ValueError("bucket_row_idx contains a negative node identity")
        if bucket_col_idx.size and int(bucket_col_idx.min()) < 0:
            raise ValueError("bucket_col_idx contains a negative node identity")
        if not np.isfinite(bucket_weights).all():
            raise ValueError("bucket_weights contains a non-finite value")

        names_raw = np.asarray(data["bucket_names"])
        if names_raw.ndim != 1 or names_raw.dtype.kind not in {"U", "S"}:
            raise ValueError("bucket_names must be a one-dimensional string array")
        names = tuple(str(value) for value in names_raw.tolist())
        if len(names) != len(set(names)):
            raise ValueError("bucket_names must be unique")
        if set(names) != set(CANONICAL_TYPED_BUCKET_NAMES):
            raise ValueError(
                "bucket_names must contain the canonical six typed buckets"
            )
        if bucket_ids.size and (
            int(bucket_ids.min()) < 0 or int(bucket_ids.max()) >= len(names)
        ):
            raise ValueError("bucket_ids contains an id outside bucket_names")

        metadata_raw = _require_scalar(data, "bucket_metadata_json")
        metadata = _load_bucket_metadata(metadata_raw, names=names)
        _validate_bucket_metadata(
            metadata,
            names=names,
            bucket_ids=bucket_ids,
            bucket_weights=bucket_weights,
        )

        error_node_shape = _require_integer_vector(data, "error_node_shape").copy()
        if error_node_shape.size and int(error_node_shape.min()) < 0:
            raise ValueError("error_node_shape contains a negative dimension")
        token_ids = _require_integer_vector(data, "token_ids").copy()
        if token_ids.size and int(token_ids.min()) < 0:
            raise ValueError("token_ids contains a negative token id")
        logit_token_ids = _require_integer_vector(data, "logit_token_ids").copy()
        if logit_token_ids.size and int(logit_token_ids.min()) < -1:
            raise ValueError("logit_token_ids contains an invalid token id")
        _validate_typed_endpoint_domains(
            names=names,
            rows=bucket_row_idx,
            cols=bucket_col_idx,
            bucket_ids=bucket_ids,
            error_node_shape=error_node_shape,
            token_ids=token_ids,
            logit_token_ids=logit_token_ids,
            feature_ids=feature_ids,
        )

        return CompactGraph(
            path=graph_path,
            step=step,
            compact_save_format=save_format or "typed_bucketed",
            bucket_row_idx=bucket_row_idx,
            bucket_col_idx=bucket_col_idx,
            bucket_weights=bucket_weights,
            bucket_ids=bucket_ids,
            bucket_names=names,
            bucket_metadata=metadata,
            error_node_shape=tuple(int(value) for value in error_node_shape),
            token_ids=token_ids,
            logit_token_ids=logit_token_ids,
        )


def summarize_feature_positions(
    graph: CompactGraph, *, max_position_exclusive: int
) -> FeaturePositionSummary:
    """Validate the independent-prefix position constraint for one graph."""
    if max_position_exclusive < 0:
        raise ValueError("max_position_exclusive must be non-negative")
    positions = graph.step.feature_ids[:, 1]
    typed_positions: list[np.ndarray] = []
    if graph.bucket_ids is not None:
        assert graph.bucket_row_idx is not None
        assert graph.bucket_col_idx is not None
        n_pos = max(
            len(graph.token_ids) if graph.token_ids is not None else 0,
            graph.error_node_shape[-1] if graph.error_node_shape else 0,
        )
        for bucket_id, name in enumerate(graph.bucket_names):
            mask = graph.bucket_ids == bucket_id
            if name.startswith("feature<-"):
                typed_positions.append(
                    _feature_endpoint_positions(graph.bucket_row_idx[mask], n_pos=n_pos)
                )
            if name.endswith("<-feature"):
                typed_positions.append(
                    _feature_endpoint_positions(graph.bucket_col_idx[mask], n_pos=n_pos)
                )
    all_positions = (
        np.concatenate([positions, *typed_positions]) if typed_positions else positions
    )
    return FeaturePositionSummary(
        feature_count=int(positions.size),
        typed_feature_endpoint_count=sum(len(values) for values in typed_positions),
        max_position=int(all_positions.max()) if all_positions.size else None,
        future_position_count=int(
            np.count_nonzero(all_positions >= max_position_exclusive)
        ),
    )


def load_historical_compact_step(path: Path) -> StepData:
    """Load historical StepData; canonical v2 graphs have no legacy projection."""
    return load_historical_compact_graph(path).step


def load_historical_step_from_pt(
    pt_path: Path, step_idx: int, *, max_edges: int = MAX_EDGES
) -> StepData:
    """Load a raw .pt graph and convert to compact StepData."""
    graph = torch.load(pt_path, map_location="cpu", weights_only=False)
    adj = graph["adjacency_matrix"]
    af = graph["active_features"]  # (F, 3): layer, pos, feat_idx
    n_features = af.shape[0]

    activation_values = graph.get(
        "activation_values"
    )  # NEW: (F,) bfloat16, may be absent
    row_idx, col_idx, weights = sparsify_edges(
        adj,
        n_features,
        max_edges=max_edges,
        activation_values=activation_values,  # NEW
    )

    feature_ids = af.numpy().astype(np.int64)
    del graph, adj

    return StepData(
        step_idx=step_idx,
        row_idx=row_idx,
        col_idx=col_idx,
        weights=weights,
        feature_ids=feature_ids,
        token_text="",
        logprob=None,
        n_features=n_features,
    )


# ── feature helpers ──────────────────────────────────────────────────


def feature_set_from_ids(feature_ids: np.ndarray) -> set[tuple[int, int]]:
    """Extract position-agnostic (layer, feature_idx) set from feature_ids array."""
    return {(int(row[0]), int(row[2])) for row in feature_ids}


def edge_dict_from_coo(
    row_idx: np.ndarray, col_idx: np.ndarray, weights: np.ndarray
) -> dict[tuple[int, int], float]:
    """Convert COO arrays to {(row, col): weight} dict for Jaccard computation."""
    return {(int(r), int(c)): float(w) for r, c, w in zip(row_idx, col_idx, weights)}


# ── temporal metrics ─────────────────────────────────────────────────


def unweighted_jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 1.0


def weighted_jaccard(
    e1: dict[tuple[int, int], float],
    e2: dict[tuple[int, int], float],
) -> float:
    """wJacc = sum min(w1, w2) / sum max(w1, w2) over union of edges."""
    all_edges = set(e1) | set(e2)
    if not all_edges:
        return 1.0
    num = sum(min(e1.get(e, 0.0), e2.get(e, 0.0)) for e in all_edges)
    den = sum(max(e1.get(e, 0.0), e2.get(e, 0.0)) for e in all_edges)
    return num / den if den else 1.0


def compute_temporal_metrics(
    steps: list[StepData],
    *,
    stable_core_window: int = STABLE_CORE_WINDOW,
    stable_core_persistence: float = STABLE_CORE_PERSISTENCE,
) -> dict[str, list]:
    """Compute all temporal metrics over an ordered list of StepData.

    Returns a dict of parallel lists keyed by metric name.
    """
    edge_wjaccard: list[float] = []
    edge_jaccard: list[float] = []
    feature_jaccard: list[float] = []
    core_sizes: list[float] = []
    core_masses: list[float] = []
    edge_history: list[set[tuple[int, int]]] = []

    prev_edges: dict[tuple[int, int], float] | None = None
    prev_features: set[tuple[int, int]] | None = None

    for sd in steps:
        edges = edge_dict_from_coo(sd.row_idx, sd.col_idx, sd.weights)
        edge_set = set(edges.keys())
        features = feature_set_from_ids(sd.feature_ids)

        edge_history.append(edge_set)

        # Consecutive overlap
        if prev_edges is not None and prev_features is not None:
            edge_wjaccard.append(weighted_jaccard(prev_edges, edges))
            edge_jaccard.append(unweighted_jaccard(set(prev_edges.keys()), edge_set))
            feature_jaccard.append(unweighted_jaccard(prev_features, features))
        else:
            edge_wjaccard.append(float("nan"))
            edge_jaccard.append(float("nan"))
            feature_jaccard.append(float("nan"))

        prev_edges = edges
        prev_features = features

        # Stable core
        W = stable_core_window
        p = stable_core_persistence
        if len(edge_history) >= W:
            window = edge_history[-W:]
            all_edges_in_window = set().union(*window)
            core = {
                e
                for e in all_edges_in_window
                if sum(1 for s in window if e in s) / W >= p
            }
            core_sizes.append(len(core))
            core_masses.append(sum(edges.get(e, 0.0) for e in core))
        else:
            core_sizes.append(float("nan"))
            core_masses.append(float("nan"))

    return {
        "edge_wjaccard": edge_wjaccard,
        "edge_jaccard": edge_jaccard,
        "feature_jaccard": feature_jaccard,
        "core_sizes": core_sizes,
        "core_masses": core_masses,
    }
