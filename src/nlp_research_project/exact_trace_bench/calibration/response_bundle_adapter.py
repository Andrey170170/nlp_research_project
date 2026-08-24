"""Typed adaptation from project observations to sibling response models."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..typed_compact_graph import CANONICAL_BUCKET_NAMES


_SCHEMA_VERSION = 2
_NUMERIC_ALIASES = {
    "attribution_batch_size": "source_batch_size",
    "feature_batch_size": "feature_batch_size",
    "logit_batch_size": "logit_batch_size",
    "attribution_update_interval": "frontier_refresh_stride",
    "decoder_chunk_size": "decoder_fetch_chunk_size",
    "cross_batch_decoder_cache_bytes": "decoder_cache_bytes",
    "nnsight_session_capacity": "session_capacity",
    "phase1_trace_batch_size_max": "phase1_source_batch_size",
    "phase3_compute_microbatch_max_rows": "logit_microbatch_size",
    "phase4_execution_batch_max_rows": "feature_microbatch_size",
    "phase4_compute_microbatch_max_rows": "feature_microbatch_size",
    "chunked_feature_replay_window": "replay_window",
    "error_vector_prefetch_lookahead": "prefetch_depth",
    "replay_tile_cache_bytes": "replay_tile_cache_bytes",
}
_WORKLOAD_ALIASES = {
    "prompt_token_count": "prompt_token_count",
    "max_active_features": "estimated_active_features",
    "n_steps_traced": "target_count",
}
_CATEGORICAL_FEATURES = (
    "provider_profile",
    "provider_type",
    "architecture",
    "row_store_policy",
    "encoder_residency",
)
_FIDELITY_BUCKET_METRICS = (
    "support_jaccard",
    "weighted_jaccard",
    "top256_jaccard",
)
_FIDELITY_TARGETS = (
    "overall_mean_feature_jaccard",
    *(
        f"overall_mean_bucket_{bucket.replace('<-', '_').replace('-', '_')}_{metric}"
        for bucket in CANONICAL_BUCKET_NAMES
        for metric in _FIDELITY_BUCKET_METRICS
    ),
)
_PLANNING_NUMERIC_FEATURES = frozenset(
    {
        "decoder_fetch_chunk_size",
        "decoder_cache_bytes",
        "session_capacity",
        "phase1_source_batch_size",
        "source_microbatch_size",
        "feature_microbatch_size",
        "logit_microbatch_size",
        "replay_window",
        "prefetch_depth",
        "replay_tile_cache_bytes",
    }
)
_ROW_STORE_ALIASES = {
    "full_file": "file_backed_full",
    "file_backed_full": "file_backed_full",
    "tiled": "tiled",
    "recompute": "recompute",
}
_ENCODER_RESIDENCY_ALIASES = {
    "lazy": "lazy_per_request",
    "lazy_per_request": "lazy_per_request",
    "eager": "eager",
}


class ResponseModelApi(Protocol):
    def publish(self, *, observations: tuple[Path, ...], output: Path) -> Any: ...
    def validate(self, *, bundle: Path) -> Any: ...


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sorted_pairs(values: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(values.items()))


def _planning_baselines(observation: Mapping[str, Any]) -> dict[str, float]:
    runtime = observation.get("runtime")
    plans = runtime.get("planning") if isinstance(runtime, Mapping) else None
    if not isinstance(plans, list):
        return {}
    baselines: dict[str, float] = {}
    for plan in plans:
        if not isinstance(plan, Mapping):
            continue
        objective = plan.get("selected_objective")
        if not isinstance(objective, list):
            continue
        for item in objective:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            name, raw = item
            value = _finite(raw)
            if value is None:
                continue
            if name == "predicted_walltime_high_seconds":
                baselines["predicted_walltime_high"] = value
    return baselines


def _selected_planning_coordinates(observation: Mapping[str, Any]) -> dict[str, float]:
    runtime = observation.get("runtime")
    plans = runtime.get("planning") if isinstance(runtime, Mapping) else None
    if not isinstance(plans, list):
        return {}
    selected: dict[str, float] = {}
    for plan in plans:
        if not isinstance(plan, Mapping):
            continue
        vector = plan.get("selected_vector")
        if not isinstance(vector, Mapping):
            continue
        for name in _PLANNING_NUMERIC_FEATURES:
            if (value := _finite(vector.get(name))) is not None:
                selected[name] = value
    return selected


class PublicSiblingResponseModelApi:
    """Use the sibling registry while keeping project schema knowledge here."""

    def __init__(self) -> None:
        try:
            self._module = importlib.import_module(
                "circuit_tracer.governor.response_models"
            )
        except ImportError as error:
            raise RuntimeError(
                "the sibling public governor response-model API is unavailable"
            ) from error

    def _sample(self, path: Path) -> Any:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"unsupported calibration observation schema in {path}")
        campaign = raw.get("campaign")
        if not isinstance(campaign, Mapping):
            raise ValueError(f"calibration observation lacks campaign metadata: {path}")
        split = self._module.CalibrationSplit(str(campaign.get("split")))

        scope_raw = raw.get("scope") if isinstance(raw.get("scope"), Mapping) else {}
        configuration = (
            raw.get("configuration")
            if isinstance(raw.get("configuration"), Mapping)
            else {}
        )
        workload = (
            raw.get("workload") if isinstance(raw.get("workload"), Mapping) else {}
        )
        numeric: dict[str, float] = {}
        for source, target in _NUMERIC_ALIASES.items():
            if (value := _finite(configuration.get(source))) is not None:
                numeric[target] = value
        for source, target in _WORKLOAD_ALIASES.items():
            if (value := _finite(workload.get(source))) is not None:
                numeric[target] = value
        numeric.update(_selected_planning_coordinates(raw))

        provider_family = str(scope_raw.get("transcoder_provider_family") or "")
        architecture = str(scope_raw.get("transcoder_architecture") or "")
        categorical = {
            "provider_profile": str(
                scope_raw.get("governor_profile_name") or "unknown"
            ),
            "provider_type": "gemmascope2"
            if provider_family.startswith("gemmascope2")
            else provider_family or "unknown",
            "architecture": architecture or "unknown",
            "row_store_policy": _ROW_STORE_ALIASES.get(
                str(configuration.get("feature_row_retention") or ""), "unknown"
            ),
            "encoder_residency": _ENCODER_RESIDENCY_ALIASES.get(
                str(configuration.get("exact_encoder_residency") or ""), "unknown"
            ),
        }
        scope = {
            "provider_profile": categorical["provider_profile"],
            "provider_type": categorical["provider_type"],
            "architecture": categorical["architecture"],
        }

        outcome_raw = (
            raw.get("outcome") if isinstance(raw.get("outcome"), Mapping) else {}
        )
        status = str(outcome_raw.get("status") or "unknown")
        outcome_aliases = {
            "success": "completed",
            "infrastructure_failure": "failed",
            "execution_error": "failed",
        }
        outcome = self._module.CalibrationOutcome(outcome_aliases.get(status, status))
        uncertainty = (
            raw.get("uncertainty")
            if isinstance(raw.get("uncertainty"), Mapping)
            else {}
        )
        censoring_raw = str(uncertainty.get("censoring") or "infrastructure")
        censoring = self._module.CensoringKind(
            "infrastructure"
            if censoring_raw == "infrastructure_or_unknown"
            else censoring_raw
        )

        targets: dict[str, float] = {}
        runtime = raw.get("runtime") if isinstance(raw.get("runtime"), Mapping) else {}
        if (
            walltime := _finite(runtime.get("walltime_seconds"))
        ) is not None and walltime > 0:
            targets["predicted_walltime_high"] = walltime
        fidelity = (
            raw.get("fidelity") if isinstance(raw.get("fidelity"), Mapping) else {}
        )
        comparison_raw = fidelity.get("comparison")
        comparison: Mapping[str, Any] = (
            comparison_raw if isinstance(comparison_raw, Mapping) else {}
        )
        for name in _FIDELITY_TARGETS:
            if (value := _finite(comparison.get(name))) is not None:
                targets[name] = value

        provenance = (
            raw.get("provenance") if isinstance(raw.get("provenance"), Mapping) else {}
        )
        fingerprints = {
            "observation": str(raw.get("observation_fingerprint") or ""),
        }
        execution = provenance.get("execution_fingerprints")
        semantic = provenance.get("semantic_fingerprints")
        if isinstance(execution, list) and execution:
            fingerprints["execution"] = ",".join(
                sorted(str(item) for item in execution)
            )
        if isinstance(semantic, list) and semantic:
            fingerprints["semantic"] = ",".join(sorted(str(item) for item in semantic))

        return self._module.CalibrationSample(
            sample_id=str(raw.get("observation_id") or path.resolve()),
            split=split,
            scope=_sorted_pairs(scope),
            numeric_coordinates=_sorted_pairs(numeric),
            categorical_coordinates=_sorted_pairs(categorical),
            outcome=outcome,
            censoring=censoring,
            targets=_sorted_pairs(targets),
            analytic_baselines=_sorted_pairs(_planning_baselines(raw)),
            provenance_fingerprints=_sorted_pairs(fingerprints),
        )

    def _fit_config(self, samples: tuple[Any, ...]) -> Any:
        numeric_features = tuple(
            sorted(
                {name for sample in samples for name, _ in sample.numeric_coordinates}
            )
        )
        models = [
            self._module.FitSpec(
                model_kind="feasibility_envelope",
                target="feasible",
                numeric_features=numeric_features,
                categorical_features=_CATEGORICAL_FEATURES,
            ),
            self._module.FitSpec(
                model_kind="positive_log_ratio_ridge",
                target="predicted_walltime_high",
                numeric_features=numeric_features,
                categorical_features=_CATEGORICAL_FEATURES,
            ),
        ]
        models.extend(
            self._module.FitSpec(
                model_kind="local_conservative_fidelity",
                target=target,
                numeric_features=numeric_features,
                categorical_features=_CATEGORICAL_FEATURES,
            )
            for target in _FIDELITY_TARGETS
        )
        return self._module.ResponseFitConfig(models=tuple(models))

    def publish(self, *, observations: tuple[Path, ...], output: Path) -> Any:
        samples = tuple(
            sorted(
                (self._sample(path) for path in observations),
                key=lambda row: row.sample_id,
            )
        )
        dataset = self._module.CalibrationDataset(samples=samples)
        bundle = self._module.fit_response_bundle(
            dataset, self._fit_config(dataset.fit_samples)
        )
        bundle.write(output)
        return {
            "content_fingerprint": bundle.content_fingerprint,
            "fit_sample_count": bundle.diagnostics["fit_sample_count"],
            "heldout_sample_count": bundle.diagnostics["heldout_sample_count"],
            "model_count": len(bundle.models),
        }

    def validate(self, *, bundle: Path) -> Any:
        loaded = self._module.load_response_bundle(bundle)
        return {
            "valid": True,
            "content_fingerprint": loaded.content_fingerprint,
            "model_count": len(loaded.models),
            "diagnostics": dict(loaded.diagnostics),
        }
