from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .signed_graph import decode_feature_endpoint, infer_n_pos, load_signed_graph

try:
    from nlp_research_project.exact_trace_bench.full_answer.decoder_signature_cache import (
        DecoderSignatureStore,
    )
except Exception:  # pragma: no cover - optional dependency path
    DecoderSignatureStore: Any | None = None

METRIC_VERSION = "core_v1"
DECODE_ERRORS = (ValueError, TypeError, IndexError)


def weighted_jaccard(a: dict[Any, float], b: dict[Any, float]) -> float:
    keys = set(a) | set(b)
    den = sum(max(abs(a.get(k, 0.0)), abs(b.get(k, 0.0))) for k in keys)
    return (
        float("nan")
        if den == 0
        else sum(min(abs(a.get(k, 0.0)), abs(b.get(k, 0.0))) for k in keys) / den
    )


def cosine(a: dict[Any, float], b: dict[Any, float]) -> float:
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return float("nan")
    return sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) & set(b)) / (na * nb)


def _add(m: dict[Any, float], k: Any, w: float) -> None:
    m[k] = m.get(k, 0.0) + abs(float(w))


def _mapped_token_id(graph: Any, src: int) -> int:
    if graph.token_ids is not None and 0 <= src < len(graph.token_ids):
        return int(graph.token_ids[src])
    return int(src)


def _mapped_logit_id(graph: Any, row_id: int) -> int:
    if graph.logit_token_ids is not None and 0 <= row_id < len(graph.logit_token_ids):
        return int(graph.logit_token_ids[row_id])
    return int(row_id)


def _iter_edge_arrays(rows: Any, cols: Any, weights: Any) -> zip:
    if not (len(rows) == len(cols) == len(weights)):
        raise ValueError(
            "bucket edge arrays have mismatched lengths: "
            f"rows={len(rows)}, cols={len(cols)}, weights={len(weights)}"
        )
    return zip(rows, cols, weights)


def _maps(path: str) -> tuple[dict[str, dict[Any, float]], int]:
    g = load_signed_graph(path)
    n_pos = infer_n_pos(g)
    out = {
        "feature_node": {},
        "feature_feature": {},
        "feature_token": {},
        "logit_feature": {},
    }
    errors = 0
    for bucket in ("feature<-token", "logit<-feature", "feature<-feature"):
        rows, cols, ws = g.bucket_edge_arrays(bucket)
        for r, c, w in _iter_edge_arrays(rows, cols, ws):
            try:
                if bucket == "feature<-token":
                    ep = decode_feature_endpoint(int(r), n_pos)
                    _add(out["feature_node"], (ep.layer, ep.feature_id), w)
                    _add(
                        out["feature_token"],
                        (ep.layer, ep.feature_id, _mapped_token_id(g, int(c))),
                        w,
                    )
                elif bucket == "logit<-feature":
                    ep = decode_feature_endpoint(int(c), n_pos)
                    _add(out["feature_node"], (ep.layer, ep.feature_id), w)
                    _add(
                        out["logit_feature"],
                        (_mapped_logit_id(g, int(r)), ep.layer, ep.feature_id),
                        w,
                    )
                else:
                    er = decode_feature_endpoint(int(r), n_pos)
                    ec = decode_feature_endpoint(int(c), n_pos)
                    _add(out["feature_node"], (er.layer, er.feature_id), w)
                    _add(out["feature_node"], (ec.layer, ec.feature_id), w)
                    _add(
                        out["feature_feature"],
                        (er.layer, er.feature_id, ec.layer, ec.feature_id),
                        w,
                    )
            except DECODE_ERRORS:
                errors += 1
    return out, errors


def _assignments(cluster_manifest: Path | None) -> dict[tuple[int, int], str]:
    if cluster_manifest is None:
        return {}
    p = cluster_manifest.parent / "feature_cluster_assignments.csv"
    if not p.exists():
        return {}
    with p.open(newline="") as h:
        return {
            (int(r["layer"]), int(r["feature_id"])): r["cluster_id"]
            for r in csv.DictReader(h)
        }


def _collapse(
    maps: dict[str, dict[Any, float]],
    assn: dict[tuple[int, int], str],
    *,
    assn_available: bool,
) -> dict[str, dict[Any, float]]:
    if not assn_available:
        return {
            "cluster_node": {},
            "cluster_feature_feature": {},
            "cluster_logit_feature": {},
        }
    out = {
        "cluster_node": {},
        "cluster_feature_feature": {},
        "cluster_logit_feature": {},
    }
    for (layer, fid), w in maps["feature_node"].items():
        cid = assn.get((layer, fid), f"unclustered_L{layer}_F{fid}")
        _add(out["cluster_node"], cid, w)
    for (tl, tf, sl, sf), w in maps["feature_feature"].items():
        tcid = assn.get((tl, tf), f"unclustered_L{tl}_F{tf}")
        scid = assn.get((sl, sf), f"unclustered_L{sl}_F{sf}")
        _add(out["cluster_feature_feature"], (tcid, scid), w)
    for (logit, layer, fid), w in maps["logit_feature"].items():
        cid = assn.get((layer, fid), f"unclustered_L{layer}_F{fid}")
        _add(out["cluster_logit_feature"], (logit, cid), w)
    return out


def _decoder_soft_metrics(
    ma: dict[str, dict[Any, float]],
    mb: dict[str, dict[Any, float]],
    store: Any,
    thresholds: list[float],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    layer_masses_a: dict[int, dict[int, float]] = {}
    layer_masses_b: dict[int, dict[int, float]] = {}
    for (layer, fid), mass in ma["feature_node"].items():
        layer_masses_a.setdefault(int(layer), {})[int(fid)] = float(mass)
    for (layer, fid), mass in mb["feature_node"].items():
        layer_masses_b.setdefault(int(layer), {})[int(fid)] = float(mass)

    layer_candidates: dict[
        int,
        tuple[
            dict[int, float],
            dict[int, float],
            list[int],
            list[int],
            list[tuple[float, int, int]],
        ],
    ] = {}
    for layer in sorted(set(layer_masses_a) | set(layer_masses_b)):
        fa = layer_masses_a.get(layer, {})
        fb = layer_masses_b.get(layer, {})
        if not fa or not fb:
            continue
        fids_a = list(fa)
        fids_b = list(fb)
        cos = store.cosine_matrix(layer, fids_a, fids_b)
        candidates: list[tuple[float, int, int]] = []
        for ia, _fid_a in enumerate(fids_a):
            for ib, _fid_b in enumerate(fids_b):
                val = float(cos[ia, ib])
                if math.isnan(val):
                    continue
                candidates.append((val, ia, ib))
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        layer_candidates[layer] = (fa, fb, fids_a, fids_b, candidates)

    sum_mass_a = sum(sum(v.values()) for v in layer_masses_a.values())
    sum_mass_b = sum(sum(v.values()) for v in layer_masses_b.values())
    for threshold in thresholds:
        matched_mass = 0.0
        for fa, fb, fids_a, fids_b, candidates in layer_candidates.values():
            used_a: set[int] = set()
            used_b: set[int] = set()
            for cos_val, ia, ib in candidates:
                if cos_val < threshold:
                    break
                if ia in used_a or ib in used_b:
                    continue
                used_a.add(ia)
                used_b.add(ib)
                mass_a = fa[fids_a[ia]]
                mass_b = fb[fids_b[ib]]
                matched_mass += min(mass_a, mass_b) * cos_val
        denom = sum_mass_a + sum_mass_b - matched_mass
        out[f"decoder_soft_matched_mass_{threshold:.2f}"] = matched_mass
        out[f"decoder_soft_denominator_{threshold:.2f}"] = denom
        out[f"decoder_soft_jaccard_{threshold:.2f}"] = (
            matched_mass / denom if denom else float("nan")
        )
        out[f"decoder_soft_matched_frac_a_{threshold:.2f}"] = (
            matched_mass / sum_mass_a if sum_mass_a else float("nan")
        )
        out[f"decoder_soft_matched_frac_b_{threshold:.2f}"] = (
            matched_mass / sum_mass_b if sum_mass_b else float("nan")
        )
    return out


def run_metrics(
    pair_manifest: Path,
    output_csv: Path,
    *,
    cluster_manifest: Path | None = None,
    decoder_cache_dir: Path | None = None,
    thresholds: list[float] | None = None,
    max_pairs: int | None = None,
) -> None:
    thresholds = thresholds or [0.70, 0.80, 0.90]
    manifest = json.loads(pair_manifest.read_text())
    pairs = manifest["pairs"][:max_pairs]
    assn = _assignments(cluster_manifest)
    assn_available = cluster_manifest is not None
    cluster_version = manifest.get("metadata", {}).get("cluster_version", "unknown")
    if cluster_manifest and cluster_manifest.exists():
        cluster_version = json.loads(cluster_manifest.read_text()).get(
            "cluster_version", cluster_version
        )
    decoder_store = None
    if decoder_cache_dir is not None:
        if DecoderSignatureStore is None:
            raise RuntimeError("DecoderSignatureStore is unavailable")
        decoder_store = DecoderSignatureStore(decoder_cache_dir)
    cache: dict[str, tuple[dict[str, dict[Any, float]], int]] = {}
    rows: list[dict[str, Any]] = []
    for p in pairs:
        for side in ("a", "b"):
            gp = p[f"graph_path_{side}"]
            if gp not in cache:
                cache[gp] = _maps(gp)
        ma, ea = cache[p["graph_path_a"]]
        mb, eb = cache[p["graph_path_b"]]
        row = {
            **p,
            "metric_version": METRIC_VERSION,
            "cluster_version": cluster_version,
            "decode_errors_a": ea,
            "decode_errors_b": eb,
            "decoder_soft_available": str(decoder_store is not None).lower(),
        }
        for name in (
            "feature_node",
            "feature_feature",
            "feature_token",
            "logit_feature",
        ):
            row[f"{name}_weighted_jaccard"] = weighted_jaccard(ma[name], mb[name])
            row[f"{name}_cosine"] = cosine(ma[name], mb[name])
        ca, cb = (
            _collapse(ma, assn, assn_available=assn_available),
            _collapse(mb, assn, assn_available=assn_available),
        )
        for name in (
            "cluster_node",
            "cluster_feature_feature",
            "cluster_logit_feature",
        ):
            row[f"{name}_weighted_jaccard"] = (
                weighted_jaccard(ca[name], cb[name]) if assn_available else ""
            )
            row[f"{name}_cosine"] = cosine(ca[name], cb[name]) if assn_available else ""
        if decoder_store is None:
            for t in thresholds:
                row[f"decoder_soft_matched_mass_{t:.2f}"] = ""
                row[f"decoder_soft_denominator_{t:.2f}"] = ""
                row[f"decoder_soft_jaccard_{t:.2f}"] = ""
                row[f"decoder_soft_matched_frac_a_{t:.2f}"] = ""
                row[f"decoder_soft_matched_frac_b_{t:.2f}"] = ""
        else:
            row.update(_decoder_soft_metrics(ma, mb, decoder_store, thresholds))
        rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else ["empty"]
    with output_csv.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
