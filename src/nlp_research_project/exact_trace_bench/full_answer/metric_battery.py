from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Any, Iterable, Sequence, cast

import numpy as np

from .decoder_signature_cache import DecoderSignatureStore
from .temporal import (
    GraphSnapshot,
    _decode_feature_feature_id,
    _feature_feature_n_pos,
    _layer_flow,
)

DEFAULT_DERIVED_KS = (128, 512, 1024, 2048, 4096, 8192)
DEFAULT_TOP_P_THRESHOLDS = (0.50, 0.80, 0.90, 0.95)
DEFAULT_RBO_PERSISTENCE = (0.90, 0.98)
DEFAULT_SOFT_THRESHOLDS = (0.70, 0.80, 0.90)

Feature = tuple[int, int, int]
PositionlessFeature = tuple[int, int]
EdgeKey = tuple[object, object]


def _row(
    *,
    bucket: str,
    metric: str,
    value: float | int | None,
    params: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    params = {} if params is None else params
    return {
        "bucket": bucket,
        "metric": metric,
        "params": params,
        "params_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
        "value": value,
        **extra,
    }


def _mass(weights: dict[Any, float]) -> float:
    return float(sum(float(v) for v in weights.values()))


def _ranked_keys(weights: dict[Any, float]) -> list[Any]:
    return sorted(weights, key=lambda key: (-float(weights[key]), repr(key)))


def jaccard(a: set[Any], b: set[Any]) -> float | None:
    union = a | b
    return None if not union else float(len(a & b) / len(union))


def weighted_jaccard(a: dict[Any, float], b: dict[Any, float]) -> float | None:
    keys = set(a) | set(b)
    if not keys:
        return None
    den = sum(max(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    if den <= 0.0:
        return None
    num = sum(min(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    return float(num / den)


def _top_k(weights: dict[Any, float], k: int) -> set[Any]:
    ranked = _ranked_keys(weights)
    return set(ranked[: min(int(k), len(ranked))])


def _core(weights: dict[Any, float], threshold: float) -> set[Any]:
    total = _mass(weights)
    if total <= 0.0:
        return set()
    running = 0.0
    out: set[Any] = set()
    for key in _ranked_keys(weights):
        if running / total >= threshold:
            break
        out.add(key)
        running += float(weights[key])
    return out


def derived_k_rows(
    a: dict[Any, float],
    b: dict[Any, float],
    *,
    bucket: str,
    ks: Sequence[int] = DEFAULT_DERIVED_KS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_a = _mass(a)
    total_b = _mass(b)
    for k in ks:
        top_a = _top_k(a, k)
        top_b = _top_k(b, k)
        union = top_a | top_b
        shared = top_a & top_b
        params = {"K": int(k)}
        rows.append(
            _row(
                bucket=bucket,
                metric="derived_k_jaccard",
                params=params,
                value=jaccard(top_a, top_b),
            )
        )
        rows.append(
            _row(
                bucket=bucket,
                metric="derived_k_weighted_jaccard",
                params=params,
                value=weighted_jaccard(
                    {key: a[key] for key in union if key in a},
                    {key: b[key] for key in union if key in b},
                ),
            )
        )
        denom = min(int(k), len(a), len(b))
        rows.append(
            _row(
                bucket=bucket,
                metric="derived_k_overlap_fraction",
                params=params,
                value=(len(shared) / denom if denom else None),
            )
        )
        rows.append(
            _row(
                bucket=bucket,
                metric="derived_k_mass_fraction_a",
                params=params,
                value=(
                    sum(float(a[key]) for key in top_a) / total_a if total_a else None
                ),
            )
        )
        rows.append(
            _row(
                bucket=bucket,
                metric="derived_k_mass_fraction_b",
                params=params,
                value=(
                    sum(float(b[key]) for key in top_b) / total_b if total_b else None
                ),
            )
        )
    return rows


def top_p_core_rows(
    a: dict[Any, float],
    b: dict[Any, float],
    *,
    bucket: str,
    thresholds: Sequence[float] = DEFAULT_TOP_P_THRESHOLDS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_a = _mass(a)
    total_b = _mass(b)
    for threshold in thresholds:
        core_a = _core(a, float(threshold))
        core_b = _core(b, float(threshold))
        shared = core_a & core_b
        params = {"p": float(threshold)}
        rows.extend(
            [
                _row(
                    bucket=bucket,
                    metric="top_p_core_jaccard",
                    params=params,
                    value=jaccard(core_a, core_b),
                ),
                _row(
                    bucket=bucket,
                    metric="top_p_core_weighted_jaccard",
                    params=params,
                    value=weighted_jaccard(
                        {key: a[key] for key in core_a | core_b if key in a},
                        {key: b[key] for key in core_a | core_b if key in b},
                    ),
                ),
                _row(
                    bucket=bucket,
                    metric="top_p_core_size_a",
                    params=params,
                    value=len(core_a),
                ),
                _row(
                    bucket=bucket,
                    metric="top_p_core_size_b",
                    params=params,
                    value=len(core_b),
                ),
                _row(
                    bucket=bucket,
                    metric="top_p_core_shared_mass_fraction_a",
                    params=params,
                    value=(
                        sum(float(a[key]) for key in shared) / total_a
                        if total_a
                        else None
                    ),
                ),
                _row(
                    bucket=bucket,
                    metric="top_p_core_shared_mass_fraction_b",
                    params=params,
                    value=(
                        sum(float(b[key]) for key in shared) / total_b
                        if total_b
                        else None
                    ),
                ),
            ]
        )
    return rows


def rbo_ext(left: Sequence[Any], right: Sequence[Any], *, p: float) -> float | None:
    """Finite-list extrapolated rank-biased overlap.

    Uses Webber et al.'s extrapolated form:
    ``(1-p) * sum(A_d * p**(d-1)) + A_k * p**k`` where ``A_d`` is
    agreement at depth ``d`` and ``k=max(len(left), len(right))``.
    """

    if not 0.0 < p < 1.0:
        raise ValueError("RBO persistence p must be in (0, 1)")
    depth = max(len(left), len(right))
    if depth == 0:
        return None
    left_seen: set[Any] = set()
    right_seen: set[Any] = set()
    weighted = 0.0
    agreement = 0.0
    overlap = 0
    for d in range(1, depth + 1):
        if d <= len(left):
            item = left[d - 1]
            if item not in left_seen:
                left_seen.add(item)
                if item in right_seen:
                    overlap += 1
        if d <= len(right):
            item = right[d - 1]
            if item not in right_seen:
                right_seen.add(item)
                if item in left_seen:
                    overlap += 1
        agreement = overlap / d
        weighted += agreement * (p ** (d - 1))
    return float((1.0 - p) * weighted + agreement * (p**depth))


def rbo_rows(
    a: dict[Any, float],
    b: dict[Any, float],
    *,
    bucket: str,
    persistence: Sequence[float] = DEFAULT_RBO_PERSISTENCE,
) -> list[dict[str, Any]]:
    left = _ranked_keys(a)
    right = _ranked_keys(b)
    return [
        _row(
            bucket=bucket,
            metric="rbo_ext",
            params={"persistence": float(p)},
            value=rbo_ext(left, right, p=float(p)),
        )
        for p in persistence
    ]


def _cosine_raw_union(a: dict[Any, float], b: dict[Any, float]) -> float | None:
    keys = set(a) | set(b)
    if not keys:
        return None
    dot = sum(float(a.get(key, 0.0)) * float(b.get(key, 0.0)) for key in keys)
    norm_a = math.sqrt(sum(float(a.get(key, 0.0)) ** 2 for key in keys))
    norm_b = math.sqrt(sum(float(b.get(key, 0.0)) ** 2 for key in keys))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    return float(dot / (norm_a * norm_b))


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for pos in range(i, j):
            ranks[order[pos]] = avg_rank
        i = j
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    lx = np.asarray(left, dtype=np.float64)
    rx = np.asarray(right, dtype=np.float64)
    lx = lx - float(lx.mean())
    rx = rx - float(rx.mean())
    den = float(np.sqrt(np.sum(lx * lx) * np.sum(rx * rx)))
    if den <= 0.0:
        return None
    return float(np.sum(lx * rx) / den)


def _spearman_intersection(a: dict[Any, float], b: dict[Any, float]) -> float | None:
    shared = sorted(set(a) & set(b), key=repr)
    if len(shared) < 2:
        return None
    ranks_a = _rankdata([float(a[key]) for key in shared])
    ranks_b = _rankdata([float(b[key]) for key in shared])
    return _pearson(ranks_a, ranks_b)


def union_support_similarity_rows(
    a: dict[Any, float], b: dict[Any, float], *, bucket: str
) -> list[dict[str, Any]]:
    shared = set(a) & set(b)
    total_a = _mass(a)
    total_b = _mass(b)
    return [
        _row(
            bucket=bucket,
            metric="union_raw_weight_cosine",
            value=_cosine_raw_union(a, b),
        ),
        _row(
            bucket=bucket,
            metric="intersection_spearman",
            value=_spearman_intersection(a, b),
        ),
        _row(bucket=bucket, metric="intersection_count", value=len(shared)),
        _row(
            bucket=bucket,
            metric="intersection_mass_fraction_a",
            value=(sum(float(a[key]) for key in shared) / total_a if total_a else None),
        ),
        _row(
            bucket=bucket,
            metric="intersection_mass_fraction_b",
            value=(sum(float(b[key]) for key in shared) / total_b if total_b else None),
        ),
    ]


def _feature_label_to_tuple(label: object) -> Feature | None:
    if isinstance(label, tuple):
        if len(label) == 4 and label[0] == "feature":
            parts = cast(tuple[Any, Any, Any, Any], label)
            return int(parts[1]), int(parts[2]), int(parts[3])
        if len(label) == 3 and all(isinstance(x, int) for x in label):
            parts3 = cast(tuple[Any, Any, Any], label)
            return int(parts3[0]), int(parts3[1]), int(parts3[2])
    return None


def feature_incident_mass(edge_map: dict[EdgeKey, float]) -> dict[Feature, float]:
    masses: dict[Feature, float] = defaultdict(float)
    for (target, source), weight in edge_map.items():
        mass = float(weight)
        source_feature = _feature_label_to_tuple(source)
        target_feature = _feature_label_to_tuple(target)
        if source_feature is not None:
            masses[source_feature] += mass
        if target_feature is not None:
            masses[target_feature] += mass
    return dict(masses)


def positionless_feature_mass(
    feature_mass: dict[Feature, float],
) -> dict[PositionlessFeature, float]:
    out: dict[PositionlessFeature, float] = defaultdict(float)
    for layer, _position, feature_id in feature_mass:
        out[(int(layer), int(feature_id))] += float(
            feature_mass[(layer, _position, feature_id)]
        )
    return dict(out)


def _normalize_distribution(values: dict[Any, float]) -> dict[Any, float]:
    total = _mass(values)
    if total <= 0.0:
        return {}
    return {key: float(value) / total for key, value in values.items()}


def _l1(a: dict[Any, float], b: dict[Any, float]) -> float | None:
    keys = set(a) | set(b)
    if not keys:
        return None
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def collapsed_flow_rows(a: GraphSnapshot, b: GraphSnapshot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    flow_pairs: list[tuple[str, dict[Any, float], dict[Any, float]]] = [
        ("all_edge_layer_flow", _layer_flow(a.all_edges), _layer_flow(b.all_edges)),
    ]
    ff_layer_a, ff_positionless_a = a.feature_feature_collapsed_flows
    ff_layer_b, ff_positionless_b = b.feature_feature_collapsed_flows
    flow_pairs.extend(
        [
            ("feature_feature_layer_flow", ff_layer_a, ff_layer_b),
            ("feature_feature_positionless_flow", ff_positionless_a, ff_positionless_b),
        ]
    )
    for bucket, left, right in flow_pairs:
        left_norm = _normalize_distribution(left)
        right_norm = _normalize_distribution(right)
        l1 = _l1(left_norm, right_norm)
        rows.extend(
            [
                _row(
                    bucket=bucket,
                    metric="weighted_jaccard",
                    value=weighted_jaccard(left, right),
                ),
                _row(bucket=bucket, metric="normalized_l1_distance", value=l1),
                _row(
                    bucket=bucket,
                    metric="total_variation_distance",
                    value=(l1 / 2.0 if l1 is not None else None),
                ),
            ]
        )
    return rows


def _feature_feature_bucket_positionless_mass(
    snap: GraphSnapshot,
) -> dict[PositionlessFeature, float]:
    n_pos = _feature_feature_n_pos(snap)
    edges = snap.bucket_edges.get("feature<-feature")
    masses: dict[PositionlessFeature, float] = defaultdict(float)
    if not n_pos or not edges:
        return {}
    for (target, source), weight in edges.items():
        for encoded in (target, source):
            layer, _position, feature_id = _decode_feature_feature_id(encoded, n_pos)
            masses[(layer, feature_id)] += float(weight)
    return dict(masses)


def _greedy_soft_matches(
    candidates: list[tuple[float, PositionlessFeature, PositionlessFeature]],
    threshold: float,
) -> list[tuple[PositionlessFeature, PositionlessFeature, float]]:
    used_left: set[PositionlessFeature] = set()
    used_right: set[PositionlessFeature] = set()
    matches: list[tuple[PositionlessFeature, PositionlessFeature, float]] = []
    for similarity, left_key, right_key in candidates:
        if similarity < threshold:
            break
        if left_key in used_left or right_key in used_right:
            continue
        used_left.add(left_key)
        used_right.add(right_key)
        matches.append((left_key, right_key, similarity))
    return matches


def _soft_match_candidates(
    left: dict[PositionlessFeature, float],
    right: dict[PositionlessFeature, float],
    *,
    decoder_store: DecoderSignatureStore,
    min_threshold: float,
) -> tuple[list[tuple[float, PositionlessFeature, PositionlessFeature]], int]:
    candidates: list[tuple[float, PositionlessFeature, PositionlessFeature]] = []
    compared = 0
    layers = sorted({layer for layer, _fid in left} & {layer for layer, _fid in right})
    for layer in layers:
        left_ids = sorted(fid for layer_id, fid in left if layer_id == layer)
        right_ids = sorted(fid for layer_id, fid in right if layer_id == layer)
        if not left_ids or not right_ids:
            continue
        cosine = decoder_store.cosine_matrix(layer, left_ids, right_ids)
        compared += int(cosine.size)
        left_arr, right_arr = np.nonzero(cosine >= min_threshold)
        for i, j in zip(left_arr.tolist(), right_arr.tolist()):
            candidates.append(
                (float(cosine[i, j]), (layer, left_ids[i]), (layer, right_ids[j]))
            )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates, compared


def soft_feature_matching_rows(
    left: dict[PositionlessFeature, float],
    right: dict[PositionlessFeature, float],
    *,
    decoder_store: DecoderSignatureStore,
    bucket: str = "positionless_feature_nodes",
    thresholds: Sequence[float] = DEFAULT_SOFT_THRESHOLDS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_left = _mass(left)
    total_right = _mass(right)
    sorted_thresholds = sorted({float(threshold) for threshold in thresholds})
    candidates, compared = _soft_match_candidates(
        left,
        right,
        decoder_store=decoder_store,
        min_threshold=min(sorted_thresholds) if sorted_thresholds else 1.0,
    )
    for threshold in thresholds:
        matches = _greedy_soft_matches(candidates, threshold=float(threshold))
        weighted_overlap = sum(
            min(float(left[lkey]), float(right[rkey])) * similarity
            for lkey, rkey, similarity in matches
        )
        matched_mass_left = sum(float(left[lkey]) for lkey, _rkey, _sim in matches)
        matched_mass_right = sum(float(right[rkey]) for _lkey, rkey, _sim in matches)
        denominator = total_left + total_right - weighted_overlap
        params = {"tau": float(threshold)}
        rows.extend(
            [
                _row(
                    bucket=bucket,
                    metric="decoder_soft_weighted_jaccard",
                    params=params,
                    value=(
                        weighted_overlap / denominator if denominator > 0.0 else None
                    ),
                ),
                _row(
                    bucket=bucket,
                    metric="decoder_soft_match_count",
                    params=params,
                    value=len(matches),
                ),
                _row(
                    bucket=bucket,
                    metric="decoder_soft_candidate_count",
                    params=params,
                    value=sum(
                        1
                        for similarity, _lkey, _rkey in candidates
                        if similarity >= float(threshold)
                    ),
                ),
                _row(
                    bucket=bucket,
                    metric="decoder_soft_compared_pair_count",
                    params=params,
                    value=compared,
                ),
                _row(
                    bucket=bucket,
                    metric="decoder_soft_matched_mass_fraction_a",
                    params=params,
                    value=(matched_mass_left / total_left if total_left else None),
                ),
                _row(
                    bucket=bucket,
                    metric="decoder_soft_matched_mass_fraction_b",
                    params=params,
                    value=(matched_mass_right / total_right if total_right else None),
                ),
            ]
        )
    return rows


def _edge_buckets(
    a: GraphSnapshot, b: GraphSnapshot
) -> Iterable[tuple[str, dict[Any, float], dict[Any, float]]]:
    yield "all_edges", a.all_edges, b.all_edges
    for name in sorted(set(a.bucket_edges) | set(b.bucket_edges)):
        yield name, a.bucket_edges.get(name, {}), b.bucket_edges.get(name, {})


def metric_battery_rows(
    a: GraphSnapshot,
    b: GraphSnapshot,
    *,
    decoder_store: DecoderSignatureStore | None = None,
    derived_ks: Sequence[int] = DEFAULT_DERIVED_KS,
    top_p_thresholds: Sequence[float] = DEFAULT_TOP_P_THRESHOLDS,
    rbo_persistence: Sequence[float] = DEFAULT_RBO_PERSISTENCE,
    soft_thresholds: Sequence[float] = DEFAULT_SOFT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Compute the Phase-1 long-form metric battery for two graph snapshots."""

    rows: list[dict[str, Any]] = []
    for bucket, left, right in _edge_buckets(a, b):
        rows.extend(derived_k_rows(left, right, bucket=bucket, ks=derived_ks))
        rows.extend(
            top_p_core_rows(left, right, bucket=bucket, thresholds=top_p_thresholds)
        )
        rows.extend(rbo_rows(left, right, bucket=bucket, persistence=rbo_persistence))
        rows.extend(union_support_similarity_rows(left, right, bucket=bucket))

    feature_mass_a = feature_incident_mass(a.all_edges)
    feature_mass_b = feature_incident_mass(b.all_edges)
    positionless_a = positionless_feature_mass(feature_mass_a)
    positionless_b = positionless_feature_mass(feature_mass_b)
    node_buckets: list[tuple[str, dict[Any, float], dict[Any, float]]] = [
        ("feature_nodes", feature_mass_a, feature_mass_b),
        ("positionless_feature_nodes", positionless_a, positionless_b),
    ]
    ff_bucket_a = _feature_feature_bucket_positionless_mass(a)
    ff_bucket_b = _feature_feature_bucket_positionless_mass(b)
    if ff_bucket_a or ff_bucket_b:
        node_buckets.append(
            (
                "feature_feature_bucket_positionless_feature_nodes",
                ff_bucket_a,
                ff_bucket_b,
            )
        )
    for bucket, left, right in node_buckets:
        rows.extend(derived_k_rows(left, right, bucket=bucket, ks=derived_ks))
        rows.extend(
            top_p_core_rows(left, right, bucket=bucket, thresholds=top_p_thresholds)
        )
        rows.extend(rbo_rows(left, right, bucket=bucket, persistence=rbo_persistence))
        rows.extend(union_support_similarity_rows(left, right, bucket=bucket))

    rows.extend(collapsed_flow_rows(a, b))

    if decoder_store is not None:
        rows.extend(
            soft_feature_matching_rows(
                positionless_a,
                positionless_b,
                decoder_store=decoder_store,
                bucket="positionless_feature_nodes",
                thresholds=soft_thresholds,
            )
        )
        if ff_bucket_a or ff_bucket_b:
            rows.extend(
                soft_feature_matching_rows(
                    ff_bucket_a,
                    ff_bucket_b,
                    decoder_store=decoder_store,
                    bucket="feature_feature_bucket_positionless_feature_nodes",
                    thresholds=soft_thresholds,
                )
            )
    return rows
