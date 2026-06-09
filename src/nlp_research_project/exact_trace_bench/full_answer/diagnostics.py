from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..io_utils import read_json, write_json

from .stability import _jaccard_from_keys, _load_npz, _row_keys, _topk_overlap

TOP_KS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12000, 16000)

FeatureKey = tuple[int, int, int]


def _key(row: np.ndarray | tuple[int, int, int]) -> FeatureKey:
    return (int(row[0]), int(row[1]), int(row[2]))


def _feature_set(rows: np.ndarray) -> set[FeatureKey]:
    return {_key(row) for row in np.asarray(rows, dtype=np.int64)}


def _load_graph(path: Path) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _edge_and_mass(
    graph: dict[str, Any],
) -> tuple[dict[tuple[Any, Any], float], dict[FeatureKey, float]]:
    features = np.asarray(graph["feature_ids"], dtype=np.int64)
    n_features = int(graph.get("n_features", np.asarray(features.shape[0])))
    edges: dict[tuple[Any, Any], float] = {}
    mass: dict[FeatureKey, float] = defaultdict(float)
    for row, col, weight in zip(graph["row_idx"], graph["col_idx"], graph["weights"]):
        col_i = int(col)
        row_i = int(row)
        if not 0 <= col_i < n_features:
            continue
        source = _key(features[col_i])
        target: Any = ("logit", row_i - n_features)
        if row_i < n_features:
            target = _key(features[row_i])
            mass[target] += abs(float(weight))
        mass[source] += abs(float(weight))
        edges[(target, source)] = abs(float(weight))
    return edges, dict(mass)


def _jaccard(a: set[Any], b: set[Any]) -> float | None:
    union = a | b
    return None if not union else len(a & b) / len(union)


def _weighted_jaccard(
    left: dict[FeatureKey, float], right: dict[FeatureKey, float]
) -> float | None:
    ltot = sum(left.values())
    rtot = sum(right.values())
    if ltot <= 0.0 and rtot <= 0.0:
        return None
    keys = set(left) | set(right)
    num = 0.0
    den = 0.0
    for key in keys:
        lw = left.get(key, 0.0) / ltot if ltot > 0 else 0.0
        rw = right.get(key, 0.0) / rtot if rtot > 0 else 0.0
        num += min(lw, rw)
        den += max(lw, rw)
    return num / den if den > 0 else None


def _mass_fraction_on(
    keys: set[FeatureKey], mass: dict[FeatureKey, float]
) -> float | None:
    total = sum(mass.values())
    return None if total <= 0 else sum(mass.get(key, 0.0) for key in keys) / total


def _project_feature_key(key: FeatureKey, grouping: str) -> tuple[int, ...]:
    if grouping == "layer":
        return (key[0],)
    if grouping == "layer_position":
        return (key[0], key[1])
    if grouping == "layer_position_bucket_10":
        return (key[0], (key[1] // 10) * 10)
    raise ValueError(f"unknown feature grouping: {grouping}")


def _group_mass(
    feature_mass: dict[FeatureKey, float], *, grouping: str
) -> dict[tuple[int, ...], float]:
    grouped: dict[tuple[int, ...], float] = defaultdict(float)
    for key, value in feature_mass.items():
        grouped[_project_feature_key(key, grouping)] += float(value)
    return dict(grouped)


def _weighted_jaccard_generic(
    left: dict[tuple[int, ...], float], right: dict[tuple[int, ...], float]
) -> float | None:
    ltot = sum(left.values())
    rtot = sum(right.values())
    if ltot <= 0.0 and rtot <= 0.0:
        return None
    num = 0.0
    den = 0.0
    for key in set(left) | set(right):
        lw = left.get(key, 0.0) / ltot if ltot > 0.0 else 0.0
        rw = right.get(key, 0.0) / rtot if rtot > 0.0 else 0.0
        num += min(lw, rw)
        den += max(lw, rw)
    return num / den if den > 0.0 else None


def _projected_group_summary(
    selected_left: set[FeatureKey],
    selected_right: set[FeatureKey],
    left_mass: dict[FeatureKey, float],
    right_mass: dict[FeatureKey, float],
    *,
    grouping: str,
) -> dict[str, Any]:
    left_groups = {_project_feature_key(key, grouping) for key in selected_left}
    right_groups = {_project_feature_key(key, grouping) for key in selected_right}
    shared_groups = left_groups & right_groups
    left_group_mass = _group_mass(left_mass, grouping=grouping)
    right_group_mass = _group_mass(right_mass, grouping=grouping)
    left_total = sum(left_mass.values())
    right_total = sum(right_mass.values())
    left_unique = selected_left - selected_right
    right_unique = selected_right - selected_left
    left_unique_with_other_group = {
        key
        for key in left_unique
        if _project_feature_key(key, grouping) in right_groups
    }
    right_unique_with_other_group = {
        key
        for key in right_unique
        if _project_feature_key(key, grouping) in left_groups
    }
    return {
        "left_group_count": len(left_groups),
        "right_group_count": len(right_groups),
        "shared_group_count": len(shared_groups),
        "group_count_jaccard": _jaccard(left_groups, right_groups),
        "group_mass_weighted_jaccard": _weighted_jaccard_generic(
            left_group_mass, right_group_mass
        ),
        "left_mass_fraction_on_shared_groups": None
        if left_total <= 0.0
        else sum(left_group_mass.get(group, 0.0) for group in shared_groups)
        / left_total,
        "right_mass_fraction_on_shared_groups": None
        if right_total <= 0.0
        else sum(right_group_mass.get(group, 0.0) for group in shared_groups)
        / right_total,
        "left_unique_count_with_other_side_group": len(left_unique_with_other_group),
        "right_unique_count_with_other_side_group": len(right_unique_with_other_group),
        "left_unique_mass_fraction_with_other_side_group": None
        if left_total <= 0.0
        else sum(left_mass.get(key, 0.0) for key in left_unique_with_other_group)
        / left_total,
        "right_unique_mass_fraction_with_other_side_group": None
        if right_total <= 0.0
        else sum(right_mass.get(key, 0.0) for key in right_unique_with_other_group)
        / right_total,
    }


def _normalized_weighted_jaccard_from_rows(
    left_rows: np.ndarray,
    left_scores: np.ndarray,
    right_rows: np.ndarray,
    right_scores: np.ndarray,
) -> float | None:
    left_keys = _row_keys(left_rows)
    right_keys = _row_keys(right_rows)
    _common, left_idx, right_idx = np.intersect1d(
        left_keys,
        right_keys,
        assume_unique=False,
        return_indices=True,
    )
    left_scores64 = np.abs(np.asarray(left_scores, dtype=np.float64))
    right_scores64 = np.abs(np.asarray(right_scores, dtype=np.float64))
    ltot = float(left_scores64.sum())
    rtot = float(right_scores64.sum())
    if ltot <= 0.0 and rtot <= 0.0:
        return None
    left_norm = left_scores64 / ltot if ltot > 0.0 else np.zeros_like(left_scores64)
    right_norm = right_scores64 / rtot if rtot > 0.0 else np.zeros_like(right_scores64)
    numerator = float(np.minimum(left_norm[left_idx], right_norm[right_idx]).sum())
    denominator = float(
        (1.0 if ltot > 0.0 else 0.0) + (1.0 if rtot > 0.0 else 0.0) - numerator
    )
    return numerator / denominator if denominator > 0.0 else None


def _edge_weighted_jaccard(
    left: dict[tuple[Any, Any], float], right: dict[tuple[Any, Any], float]
) -> float | None:
    keys = set(left) | set(right)
    if not keys:
        return None
    den = sum(max(left.get(k, 0.0), right.get(k, 0.0)) for k in keys)
    return (
        None
        if den <= 0
        else sum(min(left.get(k, 0.0), right.get(k, 0.0)) for k in keys) / den
    )


def _selected_phase3_lookup(
    active_rows: np.ndarray,
    scores: np.ndarray,
    selected: set[FeatureKey],
) -> tuple[dict[FeatureKey, float], dict[FeatureKey, int], set[FeatureKey]]:
    if not selected:
        return {}, {}, set()

    active_rows = np.asarray(active_rows, dtype=np.int64)
    scores64 = np.abs(np.asarray(scores, dtype=np.float64))
    if active_rows.size == 0:
        return {}, {}, set()

    active_keys = _row_keys(active_rows)
    active_order = np.argsort(active_keys)
    sorted_active_keys = active_keys[active_order]
    selected_rows = np.asarray(tuple(selected), dtype=np.int64)
    selected_keys = _row_keys(selected_rows)
    positions = np.searchsorted(sorted_active_keys, selected_keys)
    within = positions < sorted_active_keys.size
    matches = np.zeros(selected_keys.shape[0], dtype=bool)
    matches[within] = sorted_active_keys[positions[within]] == selected_keys[within]
    if not np.any(matches):
        return {}, {}, set()

    matched_selected_rows = selected_rows[matches]
    matched_active_idx = active_order[positions[matches]]

    score_order = np.argsort(scores64, kind="stable")[::-1]
    ranks = np.empty(scores64.size, dtype=np.int32)
    ranks[score_order] = np.arange(1, scores64.size + 1, dtype=np.int32)

    mass = {
        _key(row): float(scores64[idx])
        for row, idx in zip(matched_selected_rows, matched_active_idx)
    }
    rank_map = {
        _key(row): int(ranks[idx])
        for row, idx in zip(matched_selected_rows, matched_active_idx)
    }
    return mass, rank_map, set(mass)


def _selected_key_presence(
    active_rows: np.ndarray, selected: set[FeatureKey]
) -> set[FeatureKey]:
    if not selected:
        return set()
    active_rows = np.asarray(active_rows, dtype=np.int64)
    if active_rows.size == 0:
        return set()
    active_keys = _row_keys(active_rows)
    active_order = np.argsort(active_keys)
    sorted_active_keys = active_keys[active_order]
    selected_rows = np.asarray(tuple(selected), dtype=np.int64)
    selected_keys = _row_keys(selected_rows)
    positions = np.searchsorted(sorted_active_keys, selected_keys)
    within = positions < sorted_active_keys.size
    matches = np.zeros(selected_keys.shape[0], dtype=bool)
    matches[within] = sorted_active_keys[positions[within]] == selected_keys[within]
    return {_key(row) for row in selected_rows[matches]}


def _selected_union_weighted_jaccard(
    left: dict[FeatureKey, float], right: dict[FeatureKey, float]
) -> float | None:
    keys = set(left) | set(right)
    if not keys:
        return None
    ltot = sum(left.get(key, 0.0) for key in keys)
    rtot = sum(right.get(key, 0.0) for key in keys)
    if ltot <= 0.0 and rtot <= 0.0:
        return None
    num = 0.0
    den = 0.0
    for key in keys:
        lw = left.get(key, 0.0) / ltot if ltot > 0.0 else 0.0
        rw = right.get(key, 0.0) / rtot if rtot > 0.0 else 0.0
        num += min(lw, rw)
        den += max(lw, rw)
    return num / den if den > 0.0 else None


def _trace_alignment(left: Path, right: Path) -> dict[str, Any] | None:
    if not left.exists() or not right.exists():
        return None
    lobj = read_json(left)
    robj = read_json(right)
    fields = [
        "target_position",
        "generated_index",
        "target_token_id",
        "target_token_text",
    ]
    checks = {f"{field}_equal": lobj.get(field) == robj.get(field) for field in fields}
    lmeta = (
        lobj.get("prefix_view_metadata")
        if isinstance(lobj.get("prefix_view_metadata"), dict)
        else {}
    )
    rmeta = (
        robj.get("prefix_view_metadata")
        if isinstance(robj.get("prefix_view_metadata"), dict)
        else {}
    )
    lhash = (
        lmeta.get("prefix_token_ids_sha256")
        or lobj.get("prefix_hash")
        or lobj.get("prefix_text_hash")
        or lobj.get("prompt_hash")
    )
    rhash = (
        rmeta.get("prefix_token_ids_sha256")
        or robj.get("prefix_hash")
        or robj.get("prefix_text_hash")
        or robj.get("prompt_hash")
    )
    linput_hash = lmeta.get("input_token_ids_sha256") or lobj.get(
        "input_token_ids_sha256"
    )
    rinput_hash = rmeta.get("input_token_ids_sha256") or robj.get(
        "input_token_ids_sha256"
    )
    checks["prefix_hash_equal"] = lhash == rhash
    checks["left_prefix_hash"] = lhash
    checks["right_prefix_hash"] = rhash
    checks["left_prefix_token_ids_sha256"] = lhash
    checks["right_prefix_token_ids_sha256"] = rhash
    checks["left_input_token_ids_sha256"] = linput_hash
    checks["right_input_token_ids_sha256"] = rinput_hash
    return checks


def _grouped(
    selected_left: set[FeatureKey],
    selected_right: set[FeatureKey],
    lm: dict[FeatureKey, float],
    rm: dict[FeatureKey, float],
    *,
    by: str,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for key in selected_left | selected_right:
        group = (
            str(key[0])
            if by == "layer"
            else f"{(key[1] // 10) * 10}-{(key[1] // 10) * 10 + 9}"
        )
        groups.setdefault(
            group,
            {
                "left_count": 0,
                "right_count": 0,
                "shared_count": 0,
                "left_graph_node_mass_fraction": None,
                "right_graph_node_mass_fraction": None,
            },
        )
    ltot = sum(lm.values())
    rtot = sum(rm.values())
    for group in groups:
        keys = {
            k
            for k in selected_left | selected_right
            if (
                str(k[0])
                if by == "layer"
                else f"{(k[1] // 10) * 10}-{(k[1] // 10) * 10 + 9}"
            )
            == group
        }
        groups[group]["left_count"] = len(keys & selected_left)
        groups[group]["right_count"] = len(keys & selected_right)
        groups[group]["shared_count"] = len(keys & selected_left & selected_right)
        if ltot > 0:
            groups[group]["left_graph_node_mass_fraction"] = (
                sum(lm.get(k, 0.0) for k in keys) / ltot
            )
        if rtot > 0:
            groups[group]["right_graph_node_mass_fraction"] = (
                sum(rm.get(k, 0.0) for k in keys) / rtot
            )
    return dict(sorted(groups.items()))


def _frontier_summary(
    bundle: dict[str, np.ndarray],
    selected: set[FeatureKey],
    mass: dict[FeatureKey, float],
    total_mass: float,
) -> dict[str, Any]:
    active = np.asarray(bundle["active_features"], dtype=np.int64)
    out = {}
    for name in ("frontier_pre_locality", "frontier_post_locality"):
        if name in bundle:
            keys = {
                _key(active[int(i)])
                for i in np.asarray(bundle[name], dtype=np.int64)
                if int(i) < active.shape[0]
            }
            hit = keys & selected
            out[name] = {
                "selected_count": len(hit),
                "selected_mass_fraction": None
                if total_mass <= 0
                else sum(mass.get(key, 0.0) for key in hit) / total_mass,
            }
    return out


def diagnose_full_answer_stability(
    left_token_dir: Path, right_token_dir: Path, *, output_json: Path | None = None
) -> dict[str, Any]:
    left_token_dir = Path(left_token_dir)
    right_token_dir = Path(right_token_dir)
    caps = {
        "trace_json": (left_token_dir / "trace.json").exists()
        and (right_token_dir / "trace.json").exists(),
        "phase0_donor_bundle": (left_token_dir / "phase0_donor_bundle.npz").exists()
        and (right_token_dir / "phase0_donor_bundle.npz").exists(),
        "phase3_seed_bundle": (left_token_dir / "phase3_seed_bundle.npz").exists()
        and (right_token_dir / "phase3_seed_bundle.npz").exists(),
        "compact_graph": (left_token_dir / "graph.npz").exists()
        and (right_token_dir / "graph.npz").exists(),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis_kind": "full_answer_independent_prefix_vs_full_sequence_diagnostics",
        "left_token_dir": str(left_token_dir),
        "right_token_dir": str(right_token_dir),
        "artifact_capabilities": caps,
        "alignment_checks": {},
        "summary": {},
        "rank_overlap_curves": {},
        "top_disagreements": {},
    }
    if caps["trace_json"]:
        result["alignment_checks"] = (
            _trace_alignment(
                left_token_dir / "trace.json", right_token_dir / "trace.json"
            )
            or {}
        )

    selected_left: set[FeatureKey] = set()
    selected_right: set[FeatureKey] = set()
    left_unique: set[FeatureKey] = set()
    right_unique: set[FeatureKey] = set()
    left_unique_present_on_right: set[FeatureKey] = set()
    right_unique_present_on_left: set[FeatureKey] = set()
    left_graph_mass: dict[FeatureKey, float] = {}
    right_graph_mass: dict[FeatureKey, float] = {}
    if caps["compact_graph"]:
        lg = _load_graph(left_token_dir / "graph.npz")
        rg = _load_graph(right_token_dir / "graph.npz")
        selected_left = _feature_set(lg["feature_ids"])
        selected_right = _feature_set(rg["feature_ids"])
        left_unique = selected_left - selected_right
        right_unique = selected_right - selected_left
        le, left_graph_mass = _edge_and_mass(lg)
        re, right_graph_mass = _edge_and_mass(rg)
        shared = selected_left & selected_right
        result["summary"].update(
            {
                "compact_feature_count_jaccard": _jaccard(
                    selected_left, selected_right
                ),
                "edge_jaccard": _jaccard(set(le), set(re)),
                "weighted_edge_jaccard": _edge_weighted_jaccard(le, re),
                "graph_node_mass_weighted_jaccard": _weighted_jaccard(
                    left_graph_mass, right_graph_mass
                ),
                "left_graph_node_mass_fraction_on_exact_shared_features": _mass_fraction_on(
                    shared, left_graph_mass
                ),
                "right_graph_node_mass_fraction_on_exact_shared_features": _mass_fraction_on(
                    shared, right_graph_mass
                ),
            }
        )
        result["rank_overlap_curves"]["graph_node_mass"] = _topk_overlap(
            np.asarray(list(left_graph_mass), dtype=np.int64),
            np.asarray(list(left_graph_mass.values())),
            np.asarray(list(right_graph_mass), dtype=np.int64),
            np.asarray(list(right_graph_mass.values())),
            ks=TOP_KS,
        )
        result["grouped_decompositions"] = {
            "by_layer": _grouped(
                selected_left,
                selected_right,
                left_graph_mass,
                right_graph_mass,
                by="layer",
            ),
            "by_position_bucket_10": _grouped(
                selected_left,
                selected_right,
                left_graph_mass,
                right_graph_mass,
                by="position",
            ),
        }
        result["feature_grouping"] = {
            grouping: _projected_group_summary(
                selected_left,
                selected_right,
                left_graph_mass,
                right_graph_mass,
                grouping=grouping,
            )
            for grouping in (
                "layer",
                "layer_position",
                "layer_position_bucket_10",
            )
        }

    if caps["phase0_donor_bundle"]:
        lp0 = _load_npz(left_token_dir / "phase0_donor_bundle.npz")
        rp0 = _load_npz(right_token_dir / "phase0_donor_bundle.npz")
        result["summary"]["phase0_active_jaccard"] = _jaccard_from_keys(
            _row_keys(lp0["active_features"]), _row_keys(rp0["active_features"])
        )

    phase3_left_selected_mass: dict[FeatureKey, float] = {}
    phase3_left_selected_ranks: dict[FeatureKey, int] = {}
    phase3_right_selected_mass: dict[FeatureKey, float] = {}
    phase3_right_selected_ranks: dict[FeatureKey, int] = {}
    phase3_left_union_mass: dict[FeatureKey, float] = {}
    phase3_left_union_ranks: dict[FeatureKey, int] = {}
    phase3_right_union_mass: dict[FeatureKey, float] = {}
    phase3_right_union_ranks: dict[FeatureKey, int] = {}
    phase3_left_active_rows = np.empty((0, 3), dtype=np.int64)
    phase3_right_active_rows = np.empty((0, 3), dtype=np.int64)
    phase3_left_scores = np.empty(0, dtype=np.float64)
    phase3_right_scores = np.empty(0, dtype=np.float64)
    if caps["phase3_seed_bundle"]:
        lp3 = _load_npz(left_token_dir / "phase3_seed_bundle.npz")
        rp3 = _load_npz(right_token_dir / "phase3_seed_bundle.npz")
        phase3_left_active_rows = np.asarray(lp3["active_features"], dtype=np.int64)
        phase3_right_active_rows = np.asarray(rp3["active_features"], dtype=np.int64)
        phase3_left_scores = np.abs(
            np.asarray(lp3["seed_feature_influences"], dtype=np.float64)
        )
        phase3_right_scores = np.abs(
            np.asarray(rp3["seed_feature_influences"], dtype=np.float64)
        )
        selected_union = selected_left | selected_right
        (
            phase3_left_selected_mass,
            phase3_left_selected_ranks,
            _phase3_left_selected_present,
        ) = _selected_phase3_lookup(
            phase3_left_active_rows, phase3_left_scores, selected_left
        )
        (
            phase3_right_selected_mass,
            phase3_right_selected_ranks,
            _phase3_right_selected_present,
        ) = _selected_phase3_lookup(
            phase3_right_active_rows, phase3_right_scores, selected_right
        )
        (
            phase3_left_union_mass,
            phase3_left_union_ranks,
            _phase3_left_union_present,
        ) = _selected_phase3_lookup(
            phase3_left_active_rows, phase3_left_scores, selected_union
        )
        (
            phase3_right_union_mass,
            phase3_right_union_ranks,
            _phase3_right_union_present,
        ) = _selected_phase3_lookup(
            phase3_right_active_rows, phase3_right_scores, selected_union
        )
        left_unique_present_on_right = _selected_key_presence(
            phase3_right_active_rows, left_unique
        )
        right_unique_present_on_left = _selected_key_presence(
            phase3_left_active_rows, right_unique
        )
        left_total_mass = float(phase3_left_scores.sum())
        right_total_mass = float(phase3_right_scores.sum())
        result["summary"].update(
            {
                "phase3_active_weighted_jaccard": _normalized_weighted_jaccard_from_rows(
                    phase3_left_active_rows,
                    phase3_left_scores,
                    phase3_right_active_rows,
                    phase3_right_scores,
                ),
                "phase3_selected_union_weighted_jaccard": _selected_union_weighted_jaccard(
                    phase3_left_union_mass,
                    phase3_right_union_mass,
                )
                if selected_left or selected_right
                else None,
                "left_phase3_selected_mass_fraction_on_exact_shared_compact_features_selected_denominator": _mass_fraction_on(
                    selected_left & selected_right, phase3_left_selected_mass
                )
                if selected_left or selected_right
                else None,
                "right_phase3_selected_mass_fraction_on_exact_shared_compact_features_selected_denominator": _mass_fraction_on(
                    selected_left & selected_right, phase3_right_selected_mass
                )
                if selected_left or selected_right
                else None,
                "left_phase3_selected_mass_fraction_on_exact_shared_compact_features": (
                    sum(
                        phase3_left_selected_mass.get(k, 0.0)
                        for k in selected_left & selected_right
                    )
                    / left_total_mass
                    if (selected_left or selected_right) and left_total_mass > 0.0
                    else None
                ),
                "right_phase3_selected_mass_fraction_on_exact_shared_compact_features": (
                    sum(
                        phase3_right_selected_mass.get(k, 0.0)
                        for k in selected_left & selected_right
                    )
                    / right_total_mass
                    if (selected_left or selected_right) and right_total_mass > 0.0
                    else None
                ),
            }
        )
        result["rank_overlap_curves"]["phase3_abs_seed"] = _topk_overlap(
            phase3_left_active_rows,
            phase3_left_scores,
            phase3_right_active_rows,
            phase3_right_scores,
            ks=TOP_KS,
        )
        result["summary"]["phase3_topk"] = result["rank_overlap_curves"][
            "phase3_abs_seed"
        ]
        result["frontier_tags"] = {
            "left": _frontier_summary(
                lp3, selected_left, phase3_left_selected_mass, left_total_mass
            ),
            "right": _frontier_summary(
                rp3, selected_right, phase3_right_selected_mass, right_total_mass
            ),
        }

    lranks = phase3_left_union_ranks
    rranks = phase3_right_union_ranks

    def one_side(
        unique: set[FeatureKey],
        mass: dict[FeatureKey, float],
        seed: dict[FeatureKey, float],
        seed_ranks: dict[FeatureKey, int],
        other_seed_present: set[FeatureKey],
        other_ranks: dict[FeatureKey, int],
    ) -> list[dict[str, Any]]:
        rows = []
        for k in sorted(unique, key=lambda x: (-mass.get(x, 0.0), x))[:20]:
            rank = other_ranks.get(k)
            rows.append(
                {
                    "key": list(k),
                    "graph_node_mass": float(mass.get(k, 0.0)),
                    "phase3_abs_seed": seed.get(k),
                    "phase3_rank": seed_ranks.get(k),
                    "other_side_present": k in other_seed_present,
                    "other_side_rank": rank,
                    "other_side_topk_8192": rank is not None and rank <= 8192,
                    "core_frontier_tag": "exact_shared"
                    if k in selected_left & selected_right
                    else ("latent_other_side" if k in other_seed_present else "orphan"),
                }
            )
        return rows

    if selected_left or selected_right:
        result["core_frontier_counts"] = dict(
            Counter(
                ["exact_shared" for _ in selected_left & selected_right]
                + [
                    "left_latent_other_side"
                    if k in left_unique_present_on_right
                    else "left_orphan"
                    for k in left_unique
                ]
                + [
                    "right_latent_other_side"
                    if k in right_unique_present_on_left
                    else "right_orphan"
                    for k in right_unique
                ]
            )
        )
        result["top_disagreements"] = {
            "left_unique": one_side(
                left_unique,
                left_graph_mass,
                phase3_left_selected_mass,
                phase3_left_selected_ranks,
                left_unique_present_on_right,
                rranks,
            ),
            "right_unique": one_side(
                right_unique,
                right_graph_mass,
                phase3_right_selected_mass,
                phase3_right_selected_ranks,
                right_unique_present_on_left,
                lranks,
            ),
        }

    if output_json is not None:
        write_json(output_json, result)
    return result
