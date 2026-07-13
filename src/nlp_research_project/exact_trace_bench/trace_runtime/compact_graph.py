"""Conversion of compact attribution results into persisted graph domain objects."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

import circuit_utils
from circuit_utils import StepData


DEFAULT_TYPED_BUCKET_POLICIES = {
    "feature<-feature": {"top_p": 0.95, "cap": 1_000_000},
    "feature<-error": {"top_p": 0.99, "cap": 16_000},
    "feature<-token": {"top_p": 0.95, "cap": 250_000},
    "logit<-feature": {"top_p": 1.0, "cap": None},
    "logit<-error": {"top_p": 1.0, "cap": None},
    "logit<-token": {"top_p": 1.0, "cap": None},
}


def compact_result_to_step_data(
    compact_result: dict[str, Any],
    step_idx: int,
    *,
    token_text: str = "",
    logprob: float | None = None,
    max_edges: int = 10_000,
) -> StepData:
    feature_feature_edges = compact_result["feature_feature_edges"]
    logit_feature_edges = compact_result["logit_feature_edges"]
    feature_row_node_indices = (
        compact_result["feature_row_node_indices"]
        .to(dtype=torch.int64)
        .detach()
        .cpu()
    )
    selected_features = (
        compact_result["selected_features"].to(dtype=torch.int64).detach().cpu()
    )
    active_features = (
        compact_result["active_features"].to(dtype=torch.int64).detach().cpu()
    )
    feature_ids = active_features[selected_features].numpy().astype(np.int64)
    n_features = feature_ids.shape[0]
    selected_feature_local_index = {
        int(feature_idx): local_idx
        for local_idx, feature_idx in enumerate(selected_features.tolist())
    }

    ff_flat = feature_feature_edges.abs().float().reshape(-1)
    lf_flat = logit_feature_edges.abs().float().reshape(-1)
    combined = torch.cat([ff_flat, lf_flat])
    k = min(max_edges, int((combined != 0).sum().item()))
    if k == 0:
        return _empty_step(step_idx, feature_ids, token_text, logprob, n_features)

    topk_vals, topk_idx = torch.topk(combined, k, sorted=False)
    topk64 = topk_vals.double()
    kept_mass = float(topk64.sum().item())
    if kept_mass == 0:
        return _empty_step(step_idx, feature_ids, token_text, logprob, n_features)

    ff_size = ff_flat.numel()
    n_selected = int(selected_features.numel())
    rows: list[int] = []
    cols: list[int] = []
    for flat_idx in topk_idx.tolist():
        if flat_idx < ff_size:
            local_row = flat_idx // n_selected
            local_col = flat_idx % n_selected
            active_row = int(feature_row_node_indices[local_row].item())
            try:
                row = selected_feature_local_index[active_row]
            except KeyError as exc:
                raise ValueError(
                    "compact feature row is not present in selected_features"
                ) from exc
            rows.append(row)
            cols.append(int(local_col))
            continue
        logit_flat_idx = flat_idx - ff_size
        local_logit_row = logit_flat_idx // n_selected
        local_col = logit_flat_idx % n_selected
        rows.append(n_features + int(local_logit_row))
        cols.append(int(local_col))

    return StepData(
        step_idx=step_idx,
        row_idx=np.array(rows, dtype=np.int32),
        col_idx=np.array(cols, dtype=np.int32),
        weights=(topk64 / kept_mass).float().numpy(),
        feature_ids=feature_ids,
        token_text=token_text,
        logprob=logprob,
        n_features=n_features,
    )


def _empty_step(
    step_idx: int,
    feature_ids: np.ndarray,
    token_text: str,
    logprob: float | None,
    n_features: int,
) -> StepData:
    return StepData(
        step_idx=step_idx,
        row_idx=np.empty(0, dtype=np.int32),
        col_idx=np.empty(0, dtype=np.int32),
        weights=np.empty(0, dtype=np.float32),
        feature_ids=feature_ids,
        token_text=token_text,
        logprob=logprob,
        n_features=n_features,
    )


def _bucket_top_indices(values: torch.Tensor, policy: dict[str, Any]) -> torch.Tensor:
    flat_abs = values.abs().float().reshape(-1)
    nz = torch.where(flat_abs != 0)[0]
    if nz.numel() == 0:
        return nz
    nz_vals = flat_abs[nz]
    order = torch.argsort(nz_vals, descending=True)
    keep_n = int(nz.numel())
    top_p = float(policy.get("top_p", 1.0))
    if top_p < 1.0:
        sorted_vals = nz_vals[order].double()
        total = sorted_vals.sum()
        if float(total.item()) > 0:
            cdf = torch.cumsum(sorted_vals, dim=0) / total
            keep_n = (
                int(
                    torch.searchsorted(cdf, torch.tensor(top_p, dtype=cdf.dtype)).item()
                )
                + 1
            )
    cap = policy.get("cap")
    if cap is not None:
        keep_n = min(keep_n, int(cap))
    return nz[order[:keep_n]]


def compact_result_to_bucketed_compact(
    compact_result: dict[str, Any],
    step_idx: int,
    *,
    token_text: str = "",
    logprob: float | None = None,
    max_edges: int = 20_000,
    policies: dict[str, dict[str, Any]] | None = None,
) -> circuit_utils.BucketedCompact:
    step = compact_result_to_step_data(
        compact_result,
        step_idx,
        token_text=token_text,
        logprob=logprob,
        max_edges=max_edges,
    )
    bucket_policies = {**DEFAULT_TYPED_BUCKET_POLICIES, **(policies or {})}
    active_features = compact_result["active_features"].to(torch.int64).detach().cpu()
    selected_features = (
        compact_result["selected_features"].to(torch.int64).detach().cpu()
    )
    feature_rows = (
        compact_result["feature_row_node_indices"].to(torch.int64).detach().cpu()
    )
    logit_rows = (
        compact_result["logit_row_node_indices"].to(torch.int64).detach().cpu()
    )
    n_error = int(compact_result.get("n_error_nodes", 0))
    n_token = int(compact_result.get("n_token_nodes", 0))
    n_pos = max(n_token, 1)

    def encode_feature_ids(feature_ids: torch.Tensor) -> torch.Tensor:
        return (
            feature_ids[:, 0].to(torch.int64) * n_pos * 1_000_000
            + feature_ids[:, 1].to(torch.int64) * 1_000_000
            + feature_ids[:, 2].to(torch.int64)
        )

    selected_feature_ids = active_features[selected_features]
    bucket_specs = [
        ("feature<-feature", compact_result["feature_feature_edges"], feature_rows, selected_feature_ids),
        ("feature<-error", compact_result["feature_error_edges"], feature_rows, None),
        ("feature<-token", compact_result["feature_token_edges"], feature_rows, None),
        ("logit<-feature", compact_result["logit_feature_edges"], logit_rows, selected_feature_ids),
        ("logit<-error", compact_result["logit_error_edges"], logit_rows, None),
        ("logit<-token", compact_result["logit_token_edges"], logit_rows, None),
    ]
    all_r: list[Any] = []
    all_c: list[Any] = []
    all_w: list[Any] = []
    all_b: list[Any] = []
    metadata: list[dict[str, Any]] = []
    for bucket_id, (name, matrix, row_nodes, feature_cols) in enumerate(bucket_specs):
        mat = matrix.detach().cpu()
        keep = _bucket_top_indices(mat, bucket_policies[name])
        raw_abs = mat.abs().double().reshape(-1)
        raw_mass = float(raw_abs.sum().item())
        retained_mass = float(raw_abs[keep].sum().item()) if keep.numel() else 0.0
        n_cols = int(mat.shape[1]) if mat.ndim == 2 else 0
        rr = keep // n_cols if n_cols else torch.empty(0, dtype=torch.int64)
        cc = keep % n_cols if n_cols else torch.empty(0, dtype=torch.int64)
        rr = rr.to(torch.int64)
        cc = cc.to(torch.int64)
        if name.startswith("feature<-"):
            row_global = (
                encode_feature_ids(active_features[row_nodes[rr]])
                if rr.numel()
                else torch.empty(0, dtype=torch.int64)
            )
        else:
            row_global = rr
        if name.endswith("<-feature"):
            assert feature_cols is not None
            col_global = encode_feature_ids(feature_cols[cc])
        else:
            col_global = cc
        vals = mat.reshape(-1)[keep].float() if keep.numel() else torch.empty(0)
        all_r.append(row_global.numpy().astype(np.int64))
        all_c.append(col_global.numpy().astype(np.int64))
        all_w.append(vals.numpy().astype(np.float32))
        all_b.append(np.full(int(keep.numel()), bucket_id, dtype=np.int16))
        metadata.append(
            {
                "bucket": name,
                "raw_total_abs_mass": raw_mass,
                "retained_abs_mass": retained_mass,
                "retained_fraction": retained_mass / raw_mass if raw_mass else None,
                "raw_nnz": int((raw_abs != 0).sum().item()),
                "retained_nnz": int(keep.numel()),
                "policy": bucket_policies[name],
                "weights_signed": True,
            }
        )
    return circuit_utils.BucketedCompact(
        step=step,
        bucket_row_idx=np.concatenate(all_r),
        bucket_col_idx=np.concatenate(all_c),
        bucket_weights=np.concatenate(all_w),
        bucket_ids=np.concatenate(all_b),
        bucket_names=np.asarray([spec[0] for spec in bucket_specs]),
        bucket_metadata_json=json.dumps(metadata, sort_keys=True),
        error_node_shape=np.asarray(
            [n_error // n_pos if n_pos else 0, n_pos], dtype=np.int32
        ),
        token_ids=compact_result["input_tokens"].detach().cpu().numpy().astype(np.int64),
        logit_token_ids=np.asarray(
            [getattr(t, "vocab_idx", -1) for t in compact_result["logit_targets"]],
            dtype=np.int64,
        ),
    )
