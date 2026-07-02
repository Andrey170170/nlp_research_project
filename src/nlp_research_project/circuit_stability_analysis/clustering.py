from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.cluster.vq import kmeans2
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

from .profiles import VIEWS, load_feature_profile_bundle

try:
    from nlp_research_project.exact_trace_bench.full_answer.decoder_signature_cache import (
        DecoderSignatureStore,
    )
except Exception:  # pragma: no cover - optional dependency path
    DecoderSignatureStore = None  # type: ignore[assignment]


def _harmonic(vals: list[float]) -> float:
    vals = [max(0.0, float(v)) for v in vals]
    if not vals or min(vals) <= 0:
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def _layer_affinity(
    index: list[dict[str, Any]],
    mats: dict[str, sparse.csr_matrix],
    idxs: list[int],
    neighbors: int,
    min_affinity: float,
    *,
    layer: int,
    decoder_store: Any | None = None,
    decoder_prior_weight: float = 0.0,
) -> tuple[sparse.csr_matrix, dict[str, float]]:
    sims = {v: (mats[v][idxs] @ mats[v][idxs].T).toarray() for v in VIEWS}
    decoder_sim: np.ndarray | None = None
    if decoder_store is not None and decoder_prior_weight > 0:
        try:
            feature_ids = [int(index[i]["feature_id"]) for i in idxs]
            decoder_sim = np.asarray(
                decoder_store.cosine_matrix(layer, feature_ids, feature_ids),
                dtype=np.float64,
            )
        except Exception as exc:  # pragma: no cover - depends on external cache
            raise RuntimeError(
                f"decoder cosine failed for layer {layer}: {exc}"
            ) from exc
            decoder_sim = None
    n = len(idxs)
    aff = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a = _harmonic([sims[v][i, j] for v in VIEWS])
            if decoder_sim is not None:
                a *= 1.0 + decoder_prior_weight * max(0.0, float(decoder_sim[i, j]))
            if a >= min_affinity:
                aff[i, j] = aff[j, i] = a
    knn = np.zeros_like(aff)
    k = min(neighbors, max(0, n - 1))
    if k:
        for i in range(n):
            for j in np.argsort(-aff[i])[:k]:
                if aff[i, j] >= min_affinity:
                    knn[i, j] = knn[j, i] = aff[i, j]
    means = {
        v: float(np.mean(np.maximum(0, sims[v][np.triu_indices(n, 1)])))
        if n > 1
        else 0.0
        for v in VIEWS
    }
    means["functional"] = float(np.mean(knn[knn > 0])) if np.any(knn > 0) else 0.0
    if decoder_sim is not None and n > 1:
        means["decoder"] = float(
            np.mean(np.maximum(0, decoder_sim[np.triu_indices(n, 1)]))
        )
    return sparse.csr_matrix(knn), means


def cluster_features(
    profiles_dir: Path,
    output_dir: Path,
    *,
    min_observation_count: int = 1,
    min_total_abs_mass: float = 0.0,
    top_n_per_layer: int | None = 512,
    neighbors: int = 10,
    clusters_per_layer: int = 2,
    min_affinity: float = 0.0,
    decoder_cache_dir: Path | None = None,
    decoder_prior_weight: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    bundle = load_feature_profile_bundle(profiles_dir)
    profile_audit_path = bundle.profiles_dir / "graph_profile_audit.json"
    profile_audit = (
        json.loads(profile_audit_path.read_text())
        if profile_audit_path.exists()
        else {}
    )
    out = (
        output_dir / "feature_clusters" / "clusters_v1"
        if output_dir.name != "clusters_v1"
        else output_dir
    )
    out.mkdir(parents=True, exist_ok=True)
    decoder_store = None
    decoder_enabled = False
    if decoder_cache_dir is not None and decoder_prior_weight > 0:
        if DecoderSignatureStore is None:
            raise RuntimeError("DecoderSignatureStore is unavailable")
        decoder_store = DecoderSignatureStore(decoder_cache_dir)
        decoder_enabled = True
    rows = []
    layers: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(bundle.index):
        layer = int(r["layer"])
        obs = int(r.get("obs_count", 0))
        mass = float(r.get("total_abs_mass", 0) or 0)
        if obs >= min_observation_count and mass >= min_total_abs_mass:
            layers[layer].append(i)
    if top_n_per_layer is not None:
        for layer, idxs in list(layers.items()):
            layers[layer] = sorted(
                idxs,
                key=lambda i: -float(bundle.index[i].get("total_abs_mass", 0) or 0),
            )[:top_n_per_layer]
    included = {i for idxs in layers.values() for i in idxs}
    assignments: dict[int, tuple[str, str]] = {}
    summaries = []
    ablations = []
    rng = np.random.default_rng(seed)
    for layer, idxs in sorted(layers.items()):
        if len(idxs) < 2:
            for i in idxs:
                assignments[i] = (f"L{layer}_singleton_{i}", "too_few_layer_features")
            continue
        aff, means = _layer_affinity(
            bundle.index,
            bundle.matrices,
            idxs,
            neighbors,
            min_affinity,
            layer=layer,
            decoder_store=decoder_store,
            decoder_prior_weight=decoder_prior_weight,
        )
        for k, v in means.items():
            ablations.append({"layer": layer, "view": k, "mean_affinity": v})
        n_comp, labels_cc = csgraph.connected_components(aff, directed=False)
        if aff.nnz == 0 or len(idxs) <= clusters_per_layer:
            labels = np.arange(len(idxs)) if aff.nnz == 0 else labels_cc
            reason = "no_edges" if aff.nnz == 0 else "small_layer"
        else:
            try:
                k = min(clusters_per_layer, len(idxs) - 1)
                lap = csgraph.laplacian(aff, normed=True)
                vals, vecs = eigsh(lap, k=k, which="SM")
                del vals
                _cent, labels = kmeans2(vecs, k, minit="points", seed=rng)
                reason = "clustered"
            except Exception:
                labels = labels_cc
                reason = "fallback_connected_components"
        for local, i in enumerate(idxs):
            assignments[i] = (f"L{layer}_C{int(labels[local])}", reason)
        for lab in sorted(set(int(x) for x in labels)):
            members = [idxs[j] for j, x in enumerate(labels) if int(x) == lab]
            summaries.append(
                {
                    "layer": layer,
                    "cluster_id": f"L{layer}_C{lab}",
                    "size": len(members),
                    "total_abs_mass": sum(
                        float(bundle.index[m].get("total_abs_mass", 0) or 0)
                        for m in members
                    ),
                }
            )
    for i, r in enumerate(bundle.index):
        cid, reason = assignments.get(
            i, (f"unclustered_L{r['layer']}_{r['feature_id']}", "excluded_threshold")
        )
        rows.append(
            {
                **r,
                "cluster_version": "clusters_v1",
                "profile_version": profile_audit.get("profile_version", "unknown"),
                "cluster_id": cid,
                "assignment_status": "clustered" if i in included else "unclustered",
                "reason": reason,
            }
        )
    _write_csv(out / "feature_cluster_assignments.csv", rows)
    _write_csv(out / "cluster_summary.csv", summaries)
    _write_csv(out / "view_ablation_quality.csv", ablations)
    with (out / "cluster_exemplars.jsonl").open("w") as h:
        for s in summaries:
            members = [r for r in rows if r["cluster_id"] == s["cluster_id"]]
            h.write(
                json.dumps(
                    {
                        **s,
                        "exemplars": sorted(
                            members,
                            key=lambda r: -float(r.get("total_abs_mass", 0) or 0),
                        )[:5],
                    }
                )
                + "\n"
            )
    quality = {
        "cluster_count": len(summaries),
        "assigned_feature_count": len(included),
        "decoder_prior_enabled": decoder_enabled,
        "top_n_per_layer": top_n_per_layer,
        "bucket_sign_mode": profile_audit.get("bucket_sign_mode"),
        "cluster_size_min": min((int(s["size"]) for s in summaries), default=0),
        "cluster_size_max": max((int(s["size"]) for s in summaries), default=0),
        "cluster_size_mean": float(np.mean([int(s["size"]) for s in summaries]))
        if summaries
        else 0.0,
    }
    (out / "cluster_quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True)
    )
    manifest = {
        "cluster_version": "clusters_v1",
        "within_layer": True,
        "decoder_prior_enabled": decoder_enabled,
        "decoder_prior_weight": decoder_prior_weight,
        "top_n_per_layer": top_n_per_layer,
        "min_observation_count": min_observation_count,
        "min_total_abs_mass": min_total_abs_mass,
        "neighbors": neighbors,
        "clusters_per_layer": clusters_per_layer,
        "min_affinity": min_affinity,
        "seed": seed,
        "profiles_dir": str(bundle.profiles_dir),
        "profile_version": profile_audit.get("profile_version"),
        "bucket_sign_mode": profile_audit.get("bucket_sign_mode"),
        "tokens_path": profile_audit.get("tokens_path"),
        "roles_path": profile_audit.get("roles_path"),
    }
    (out / "cluster_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for r in rows for k in r}) if rows else ["empty"]
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
