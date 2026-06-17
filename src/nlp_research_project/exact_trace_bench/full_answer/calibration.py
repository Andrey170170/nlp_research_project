from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from ..io_utils import (
    ensure_dir,
    iter_jsonl,
    read_json,
    write_csv,
    write_json,
    write_jsonl,
)
from .decoder_signature_cache import DecoderSignatureStore
from .metric_battery import metric_battery_rows
from .temporal import GraphSnapshot, _load_snapshot, discover_graph_paths

DEFAULT_CALIBRATION_LAGS = (1, 2, 5, 10, 20)
SCORECARD_CATEGORIES = ("null", "temporal", "noise")
PAIR_CHECKPOINT_DIR = "pair_rows"
PAIR_CONTEXT_DIR = "pair_context"


@dataclass(frozen=True)
class CalibrationPair:
    pair_id: str
    left_graph: Path
    right_graph: Path
    pair_category: str
    sub_category: str | None = None
    fixture: str | None = None
    trajectory: str | None = None
    position_band: str | None = None
    generated_index: int | None = None
    generated_index_a: int | None = None
    generated_index_b: int | None = None
    lag: int | None = None
    metadata: dict[str, Any] | None = None

    def context(self, left: GraphSnapshot, right: GraphSnapshot) -> dict[str, Any]:
        generated_index_a = self.generated_index_a
        if generated_index_a is None:
            generated_index_a = left.generated_index
        generated_index_b = self.generated_index_b
        if generated_index_b is None:
            generated_index_b = right.generated_index
        lag = self.lag
        if (
            lag is None
            and generated_index_a is not None
            and generated_index_b is not None
        ):
            lag = int(generated_index_b) - int(generated_index_a)
        return {
            "pair_id": self.pair_id,
            "pair_category": self.pair_category,
            "sub_category": self.sub_category,
            "fixture": self.fixture,
            "trajectory": self.trajectory,
            "position_band": self.position_band,
            "generated_index": self.generated_index,
            "generated_index_a": generated_index_a,
            "generated_index_b": generated_index_b,
            "lag": lag,
            "left_graph": str(self.left_graph),
            "right_graph": str(self.right_graph),
            "token_text_a": left.token_text,
            "token_text_b": right.token_text,
            **(self.metadata or {}),
        }


def position_band_for_index(index: int, generated_count: int) -> str:
    if generated_count <= 1:
        return "early"
    r = index / max(generated_count - 1, 1)
    if r < 0.1:
        return "early"
    if r <= 0.8:
        return "mid"
    return "late"


def _sample_indices(indices: list[int], *, max_count: int | None) -> list[int]:
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=max_count).round().astype(int)
    return [indices[int(pos)] for pos in np.unique(positions)]


def _trajectory_pairs(entry: dict[str, Any]) -> list[CalibrationPair]:
    run_root = Path(entry["run_root"])
    graph_paths = discover_graph_paths(run_root, max_tokens=entry.get("max_tokens"))
    by_index = {
        int(path.parent.name.removeprefix("token_")): path for path in graph_paths
    }
    indices = sorted(by_index)
    generated_count = len(indices)
    lags = [int(x) for x in entry.get("lags", DEFAULT_CALIBRATION_LAGS)]
    stride = max(1, int(entry.get("stride", 1)))
    max_pairs_per_lag = entry.get("max_pairs_per_lag")
    pairs: list[CalibrationPair] = []
    trajectory = entry.get("trajectory") or entry.get("name") or run_root.name
    pair_category = entry.get("pair_category", "temporal")
    for lag in lags:
        starts = [idx for idx in indices[::stride] if idx + lag in by_index]
        starts = _sample_indices(
            starts,
            max_count=int(max_pairs_per_lag) if max_pairs_per_lag is not None else None,
        )
        for idx in starts:
            sub_category = entry.get("sub_category") or (
                "adjacent" if lag == 1 else "lagged"
            )
            pairs.append(
                CalibrationPair(
                    pair_id=f"{trajectory}:lag{lag}:{idx}->{idx + lag}",
                    left_graph=by_index[idx],
                    right_graph=by_index[idx + lag],
                    pair_category=pair_category,
                    sub_category=sub_category,
                    fixture=entry.get("fixture"),
                    trajectory=trajectory,
                    position_band=position_band_for_index(idx, generated_count),
                    generated_index=idx,
                    generated_index_a=idx,
                    generated_index_b=idx + lag,
                    lag=lag,
                    metadata=entry.get("metadata"),
                )
            )
    return pairs


def _explicit_pairs(entries: Iterable[dict[str, Any]]) -> list[CalibrationPair]:
    pairs: list[CalibrationPair] = []
    for i, entry in enumerate(entries):
        pairs.append(
            CalibrationPair(
                pair_id=entry.get("pair_id", f"explicit:{i:06d}"),
                left_graph=Path(entry["left_graph"]),
                right_graph=Path(entry["right_graph"]),
                pair_category=entry["pair_category"],
                sub_category=entry.get("sub_category"),
                fixture=entry.get("fixture"),
                trajectory=entry.get("trajectory"),
                position_band=entry.get("position_band"),
                generated_index=entry.get("generated_index"),
                generated_index_a=entry.get("generated_index_a"),
                generated_index_b=entry.get("generated_index_b"),
                lag=entry.get("lag"),
                metadata=entry.get("metadata"),
            )
        )
    return pairs


def _matched_relative_pairs(entry: dict[str, Any]) -> list[CalibrationPair]:
    left_root = Path(entry["left_run_root"])
    right_root = Path(entry["right_run_root"])
    left_paths = discover_graph_paths(
        left_root, max_tokens=entry.get("left_max_tokens")
    )
    right_paths = discover_graph_paths(
        right_root, max_tokens=entry.get("right_max_tokens")
    )
    left_by_pos = {i: path for i, path in enumerate(left_paths)}
    right_by_pos = {i: path for i, path in enumerate(right_paths)}
    sample_count = int(entry.get("sample_count", 20))
    count = min(len(left_paths), len(right_paths), sample_count)
    if count <= 0:
        return []
    left_positions = np.linspace(0, len(left_paths) - 1, num=count).round().astype(int)
    right_positions = (
        np.linspace(0, len(right_paths) - 1, num=count).round().astype(int)
    )
    label = entry.get("name") or f"{left_root.name}__vs__{right_root.name}"
    pairs: list[CalibrationPair] = []
    for rank, (left_pos, right_pos) in enumerate(zip(left_positions, right_positions)):
        left_pos_i = int(left_pos)
        right_pos_i = int(right_pos)
        pairs.append(
            CalibrationPair(
                pair_id=f"{label}:relative:{rank:03d}",
                left_graph=left_by_pos[left_pos_i],
                right_graph=right_by_pos[right_pos_i],
                pair_category=entry.get("pair_category", "null"),
                sub_category=entry.get("sub_category", "matched_relative"),
                fixture=entry.get("fixture"),
                trajectory=label,
                position_band=position_band_for_index(left_pos_i, len(left_paths)),
                generated_index=left_pos_i,
                generated_index_a=left_pos_i,
                generated_index_b=right_pos_i,
                lag=None,
                metadata=entry.get("metadata"),
            )
        )
    return pairs


def load_calibration_pairs(manifest_path: Path) -> list[CalibrationPair]:
    manifest = read_json(manifest_path)
    pairs: list[CalibrationPair] = []
    pairs.extend(_explicit_pairs(manifest.get("pairs", [])))
    for entry in manifest.get("trajectories", []):
        pairs.extend(_trajectory_pairs(entry))
    for entry in manifest.get("matched_relative_pairs", []):
        pairs.extend(_matched_relative_pairs(entry))
    if not pairs:
        raise ValueError(
            f"calibration manifest did not resolve any pairs: {manifest_path}"
        )
    return pairs


def _snapshot_loader() -> Any:
    cache: dict[Path, GraphSnapshot] = {}

    def load(path: Path) -> GraphSnapshot:
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = _load_snapshot(resolved)
        return cache[resolved]

    return load


def _safe_pair_stem(pair_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", pair_id).strip("._")
    return stem[:180] if stem else "pair"


def _pair_rows_path(output_dir: Path, pair: CalibrationPair) -> Path:
    return output_dir / PAIR_CHECKPOINT_DIR / f"{_safe_pair_stem(pair.pair_id)}.jsonl"


def _pair_context_path(output_dir: Path, pair: CalibrationPair) -> Path:
    return output_dir / PAIR_CONTEXT_DIR / f"{_safe_pair_stem(pair.pair_id)}.json"


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _compute_pair_rows(
    pair: CalibrationPair,
    *,
    decoder_cache_dir: Path | None,
    decoder_cosine_device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left = _load_snapshot(pair.left_graph)
    right = _load_snapshot(pair.right_graph)
    context = pair.context(left, right)
    decoder_store = (
        DecoderSignatureStore(decoder_cache_dir, cosine_device=decoder_cosine_device)
        if decoder_cache_dir is not None
        else None
    )
    rows = [
        {**context, **metric_row}
        for metric_row in metric_battery_rows(left, right, decoder_store=decoder_store)
    ]
    return context, rows


def _run_pair_checkpoint(
    pair: CalibrationPair,
    *,
    output_dir: Path,
    decoder_cache_dir: Path | None,
    decoder_cosine_device: str,
    resume: bool,
) -> dict[str, Any]:
    rows_path = _pair_rows_path(output_dir, pair)
    context_path = _pair_context_path(output_dir, pair)
    if resume and rows_path.exists() and context_path.exists():
        return {
            "pair_id": pair.pair_id,
            "status": "skipped_existing",
            "rows_path": str(rows_path),
            "context_path": str(context_path),
            "metric_row_count": sum(1 for _ in iter_jsonl(rows_path)),
        }

    context, rows = _compute_pair_rows(
        pair,
        decoder_cache_dir=decoder_cache_dir,
        decoder_cosine_device=decoder_cosine_device,
    )
    _atomic_write_jsonl(rows_path, rows)
    _atomic_write_json(context_path, context)
    return {
        "pair_id": pair.pair_id,
        "status": "completed",
        "rows_path": str(rows_path),
        "context_path": str(context_path),
        "metric_row_count": len(rows),
    }


def _read_pair_outputs(
    *, output_dir: Path, pairs: list[CalibrationPair]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    missing: list[str] = []
    for pair in pairs:
        rows_path = _pair_rows_path(output_dir, pair)
        context_path = _pair_context_path(output_dir, pair)
        if not rows_path.exists() or not context_path.exists():
            missing.append(pair.pair_id)
            continue
        contexts.append(read_json(context_path))
        rows.extend(iter_jsonl(rows_path))
    if missing:
        raise RuntimeError(
            f"missing calibration pair checkpoints for {len(missing)} pairs: "
            f"{missing[:10]}"
        )
    return rows, contexts


def _tabular_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        tabular = dict(row)
        tabular.pop("params", None)
        out.append(tabular)
    return out


def _numeric_values(rows: Iterable[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get("value")
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return values


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def _is_distance_metric(metric: str) -> bool:
    metric_lower = metric.lower()
    return "distance" in metric_lower or metric_lower.endswith("_l1")


def build_scorecard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("bucket")),
            str(row.get("metric")),
            str(row.get("params_json", "{}")),
            str(row.get("position_band") or "unknown"),
            str(row.get("role_cluster") or "all"),
        )
        groups.setdefault(key, []).append(row)

    scorecard: list[dict[str, Any]] = []
    temporal_scorecard_sub_categories = {"adjacent", "role_matched_within_trajectory"}
    for (
        bucket,
        metric,
        params_json,
        position_band,
        role_cluster,
    ), group_rows in sorted(groups.items()):
        category_values: dict[str, list[float]] = {}
        for category in SCORECARD_CATEGORIES:
            if category == "temporal":
                selected = [
                    row
                    for row in group_rows
                    if row.get("pair_category") == "temporal"
                    and (
                        row.get("sub_category") in temporal_scorecard_sub_categories
                        or row.get("lag") == 1
                    )
                ]
            else:
                selected = [
                    row for row in group_rows if row.get("pair_category") == category
                ]
            category_values[category] = _numeric_values(selected)

        null_mean, null_sd = _mean_sd(category_values["null"])
        temporal_mean, _temporal_sd = _mean_sd(category_values["temporal"])
        noise_mean, noise_sd = _mean_sd(category_values["noise"])
        distance_metric = _is_distance_metric(metric)
        separation_score = None
        passes = False
        if (
            null_mean is not None
            and temporal_mean is not None
            and noise_mean is not None
        ):
            if distance_metric:
                denom = null_mean - noise_mean
                separation_score = (
                    (temporal_mean - noise_mean) / denom if abs(denom) > 1e-12 else None
                )
                noise_margin = temporal_mean >= noise_mean + 2.0 * (noise_sd or 0.0)
                null_margin = temporal_mean <= null_mean - 2.0 * (null_sd or 0.0)
            else:
                denom = noise_mean - null_mean
                separation_score = (
                    (noise_mean - temporal_mean) / denom if abs(denom) > 1e-12 else None
                )
                noise_margin = temporal_mean <= noise_mean - 2.0 * (noise_sd or 0.0)
                null_margin = temporal_mean >= null_mean + 2.0 * (null_sd or 0.0)
            passes = bool(noise_margin and null_margin)

        noise_values = category_values["noise"]
        noise_worst = None
        if noise_values:
            noise_worst = max(noise_values) if distance_metric else min(noise_values)
        scorecard.append(
            {
                "bucket": bucket,
                "metric": metric,
                "params_json": params_json,
                "position_band": position_band,
                "role_cluster": role_cluster,
                "metric_direction": "lower_is_more_similar"
                if distance_metric
                else "higher_is_more_similar",
                "null_count": len(category_values["null"]),
                "null_mean": null_mean,
                "null_sd": null_sd,
                "temporal_adjacent_count": len(category_values["temporal"]),
                "temporal_adjacent_mean": temporal_mean,
                "noise_count": len(noise_values),
                "noise_mean": noise_mean,
                "noise_sd": noise_sd,
                "noise_worst_case": noise_worst,
                "separation_score": separation_score,
                "passes_calibration": passes,
            }
        )
    return scorecard


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return True


def run_metric_calibration(
    *,
    manifest_path: Path,
    output_dir: Path,
    decoder_cache_dir: Path | None = None,
    write_parquet: bool = True,
    workers: int = 1,
    resume: bool = True,
    decoder_cosine_device: str = "cpu",
) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(output_dir / PAIR_CHECKPOINT_DIR)
    ensure_dir(output_dir / PAIR_CONTEXT_DIR)
    manifest = read_json(manifest_path)
    pairs = load_calibration_pairs(manifest_path)
    workers = max(1, int(workers))
    checkpoint_results: list[dict[str, Any]] = []
    if workers == 1:
        for idx, pair in enumerate(pairs, start=1):
            result = _run_pair_checkpoint(
                pair,
                output_dir=output_dir,
                decoder_cache_dir=decoder_cache_dir,
                decoder_cosine_device=decoder_cosine_device,
                resume=resume,
            )
            checkpoint_results.append(result)
            print(
                f"[{idx}/{len(pairs)}] {result['status']} {pair.pair_id} "
                f"rows={result['metric_row_count']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_pair = {
                executor.submit(
                    _run_pair_checkpoint,
                    pair,
                    output_dir=output_dir,
                    decoder_cache_dir=decoder_cache_dir,
                    decoder_cosine_device=decoder_cosine_device,
                    resume=resume,
                ): pair
                for pair in pairs
            }
            completed = 0
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                completed += 1
                result = future.result()
                checkpoint_results.append(result)
                print(
                    f"[{completed}/{len(pairs)}] {result['status']} {pair.pair_id} "
                    f"rows={result['metric_row_count']}",
                    flush=True,
                )

    rows, resolved_pairs = _read_pair_outputs(output_dir=output_dir, pairs=pairs)

    scorecard = build_scorecard(rows)
    tabular = _tabular_rows(rows)
    write_jsonl(output_dir / "metric_rows.jsonl", rows)
    write_csv(
        output_dir / "metric_rows.csv",
        tabular,
        preferred_headers=[
            "pair_id",
            "pair_category",
            "sub_category",
            "fixture",
            "trajectory",
            "position_band",
            "generated_index",
            "generated_index_a",
            "generated_index_b",
            "lag",
            "bucket",
            "metric",
            "params_json",
            "value",
        ],
    )
    parquet_written = False
    if write_parquet:
        parquet_written = _write_parquet(output_dir / "metric_rows.parquet", tabular)
    write_csv(output_dir / "scorecard.csv", scorecard)
    write_json(output_dir / "scorecard.json", scorecard)
    write_json(output_dir / "pair_manifest_resolved.json", resolved_pairs)
    summary = {
        "analysis_kind": "full_answer_metric_calibration",
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "decoder_cache_dir": str(decoder_cache_dir) if decoder_cache_dir else None,
        "decoder_soft_matching_enabled": decoder_cache_dir is not None,
        "decoder_cosine_device": decoder_cosine_device,
        "workers": workers,
        "resume": resume,
        "pair_count": len(pairs),
        "metric_row_count": len(rows),
        "scorecard_row_count": len(scorecard),
        "checkpoint_results": checkpoint_results,
        "parquet_written": parquet_written,
        "manifest": manifest,
    }
    write_json(output_dir / "calibration_summary.json", summary)
    return summary
