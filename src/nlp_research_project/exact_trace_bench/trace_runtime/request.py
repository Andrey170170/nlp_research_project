"""Project policy for constructing canonical sibling trace requests."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from circuit_tracer import (
    AdmissionMode,
    AttributionProblem,
    DecoderCachePolicy,
    ExecutionConstraints,
    FidelityMode,
    FrontierExpansionPlan,
    FrontierSemantics,
    GovernorFidelityPolicy,
    ObservabilityPolicy,
    ReplayPlan,
    RowStoragePlan,
    SessionPlan,
    TraceEvidence,
    TraceRequest,
    TraceSemantics,
)
from circuit_tracer.governor import (
    RECORDED_PROVIDER_PROFILES,
    CachePolicy,
    EncoderResidency,
    PhysicalExecutionRequirements,
    ProviderProfile,
    ResourceEnvelope,
    RowStorePolicy,
    StorageTier,
)
from circuit_tracer.governor.calibration import FidelityBudget as SiblingFidelityBudget
from circuit_tracer.governor.response_models import (
    ResponseBundle,
    load_response_bundle,
)

from nlp_research_project.exact_trace_bench.calibration_observations import (
    NormalizedFidelityPolicy,
    parse_fidelity_policy,
)


@dataclass(frozen=True)
class TracePolicy:
    """Reusable per-step scientific and physical tracing policy.

    The model and growing prompt belong to each generated step. Everything else
    is resolved once from the campaign scenario and reused as one typed template.
    """

    semantics: TraceSemantics
    execution: ExecutionConstraints
    max_n_logits: int
    desired_logit_prob: float
    evidence: TraceEvidence
    resources: ResourceEnvelope | None = None
    provider_profile: ProviderProfile | None = None
    physical_requirements: PhysicalExecutionRequirements | None = None
    governor_fidelity: GovernorFidelityPolicy = field(
        default_factory=GovernorFidelityPolicy
    )
    governor_admission_mode: AdmissionMode = AdmissionMode.ENFORCE
    response_bundle: ResponseBundle | None = field(default=None, repr=False)

    def request(
        self,
        *,
        model: Any,
        prompt: str | list[int] | Any,
        telemetry_jsonl_path: str | Path | None = None,
        telemetry_context: Mapping[str, object] | None = None,
    ) -> TraceRequest:
        execution = self.execution
        if telemetry_jsonl_path is not None or telemetry_context is not None:
            observability = replace(
                execution.observability,
                telemetry_jsonl_path=telemetry_jsonl_path,
                telemetry_context={}
                if telemetry_context is None
                else telemetry_context,
            )
            execution = replace(execution, observability=observability)
        return TraceRequest(
            problem=AttributionProblem(
                model=model,
                prompt=prompt,
                max_n_logits=self.max_n_logits,
                desired_logit_prob=self.desired_logit_prob,
            ),
            semantics=self.semantics,
            execution=execution,
            evidence=self.evidence,
            physical_requirements=self.physical_requirements,
            governor_fidelity=self.governor_fidelity,
            governor_admission_mode=self.governor_admission_mode,
            response_bundle=self.response_bundle,
        )


def trace_policy_from_scenario(
    scenario: Mapping[str, Any],
    *,
    sparsification: Any | None = None,
) -> TracePolicy:
    """Resolve one scenario into the canonical sibling-owned policy types."""

    phase0_decoder_row_ranges = _bool_knob(
        scenario, "phase0_decoder_row_ranges", False
    )
    resources, provider_profile = _governor_policy_from_scenario(scenario)
    governor_fidelity = _governor_fidelity_from_scenario(scenario)
    governor_admission_mode = AdmissionMode(
        str(scenario.get("governor_admission_mode", "enforce"))
    )
    response_bundle = _response_bundle_from_scenario(scenario)
    physical_requirements = (
        _physical_requirements_from_scenario(scenario)
        if provider_profile is not None
        else None
    )

    phase4_planning = bool(
        scenario.get("plan_feature_batch_size", False)
        or scenario.get("auto_scale_feature_batch_size", False)
    )
    semantics = TraceSemantics(
        source_batch_size=int(scenario.get("attribution_batch_size", 256)),
        feature_batch_size=_optional_int(scenario.get("feature_batch_size")),
        logit_batch_size=_optional_int(scenario.get("logit_batch_size")),
        max_feature_nodes=_optional_int(scenario.get("max_feature_nodes")),
        diagnostic_feature_cap=_optional_int(scenario.get("diagnostic_feature_cap")),
        update_interval=int(scenario.get("attribution_update_interval", 4)),
        exact_trace_internal_dtype=_choice(
            scenario, "exact_trace_internal_dtype", "fp32"
        ),
        phase0_activation_threshold_compare_mode=_choice(
            scenario, "phase0_activation_threshold_compare_mode", "baseline"
        ),
        sparsification=sparsification,
        frontier=FrontierSemantics(
            scheduler=_choice(scenario, "phase4_scheduler_mode", "locality"),
            refresh_policy=_choice(scenario, "phase4_refresh_policy", "standard"),
            refresh_interval_multiplier=int(
                scenario.get("phase4_refresh_interval_multiplier", 1)
            ),
            ranker=_choice(scenario, "phase4_ranker", "argsort"),
            phase3_buffer_relative_epsilon=scenario.get(
                "phase3_frontier_buffer_relative_epsilon"
            ),
            phase3_buffer_max_extra=int(
                scenario.get("phase3_frontier_buffer_max_extra", 0)
            ),
            phase4_buffer_relative_epsilon=scenario.get(
                "phase4_frontier_buffer_relative_epsilon"
            ),
            phase4_buffer_max_extra_per_refresh=int(
                scenario.get("phase4_frontier_buffer_max_extra_per_refresh", 0)
            ),
            phase4_buffer_max_extra_total=int(
                scenario.get("phase4_frontier_buffer_max_extra_total", 0)
            ),
        ),
    )
    execution = ExecutionConstraints(
        session=SessionPlan(
            capacity=_optional_int(scenario.get("nnsight_session_capacity")),
            phase3_microbatch_max_rows=_optional_int(
                scenario.get("phase3_compute_microbatch_max_rows")
            ),
            phase4_execution_batch_max_rows=_optional_int(
                _phase4_execution_batch_max_rows(scenario)
            ),
            phase1_trace_batch_policy=_choice(
                scenario, "phase1_trace_batch_policy", "legacy"
            ),
            phase1_trace_batch_size_max=_optional_int(
                scenario.get("phase1_trace_batch_size_max")
            ),
            decoder_cache=_decoder_cache_policy(
                scenario.get("cross_batch_decoder_cache_bytes")
            ),
        ),
        storage=RowStoragePlan(
            retention=_choice(scenario, "feature_row_retention", "full_file"),
            full_retention_backend=_choice(
                scenario, "full_retention_backend", "full_file"
            ),
            feature_column_tile_size=int(
                scenario.get("feature_row_column_tile_size", 2048)
            ),
            influence_row_tile_size=int(scenario.get("influence_row_tile_size", 4096)),
            influence_column_tile_size=int(
                scenario.get("influence_column_tile_size", 2048)
            ),
            cache_control=_choice(scenario, "row_store_cache_control", "off"),
            temp_root_policy=_choice(scenario, "row_store_temp_root_policy", "default"),
            temp_root=scenario.get("row_store_temp_root"),
            preallocate=bool(scenario.get("row_store_preallocate", True)),
            replay_tile_cache_bytes=_optional_int(
                scenario.get("replay_tile_cache_bytes")
            ),
            exact_encoder_residency=_choice(
                scenario, "exact_encoder_residency", "lazy"
            ),
        ),
        replay=ReplayPlan(
            feature_window=int(scenario.get("chunked_feature_replay_window") or 4),
            error_vector_prefetch_lookahead=_int_default(
                scenario.get("error_vector_prefetch_lookahead"), 2
            ),
            stage_encoder_vecs_on_cpu=scenario.get("stage_encoder_vecs_on_cpu"),
            stage_error_vectors_on_cpu=scenario.get("stage_error_vectors_on_cpu"),
            decoder_contraction_tile=_optional_int(scenario.get("row_subchunk_size")),
            phase0_donor_bundle=scenario.get("phase0_donor_bundle"),
            phase0_mode=_choice(scenario, "phase0_replay_mode", "disabled"),
            phase0_donor_context_policy=_choice(
                scenario, "phase0_donor_context_policy", "strict"
            ),
            phase3_gradient_donor_bundle=scenario.get("phase3_gradient_donor_bundle"),
            phase3_gradient_mode=_choice(
                scenario, "phase3_gradient_replay_mode", "disabled"
            ),
            phase3_row_donor_bundle=scenario.get("phase3_row_donor_bundle"),
            phase3_row_mode=_choice(scenario, "phase3_row_replay_mode", "disabled"),
            phase3_validation_policy=_choice(
                scenario, "phase3_replay_validation_policy", "strict"
            ),
        ),
        frontier=FrontierExpansionPlan(
            scheduler_debug=bool(scenario.get("phase4_scheduler_debug", False)),
            scheduler_telemetry_detail=_choice(
                scenario, "phase4_scheduler_telemetry_detail", "normal"
            ),
            refresh_optimization=_choice(scenario, "phase4_refresh_optimization", "v1"),
            refresh_prepared_chunk_cache_bytes=int(
                scenario.get("phase4_refresh_prepared_chunk_cache_bytes", 0)
            ),
            refresh_active_row_accumulation=_choice(
                scenario, "phase4_refresh_active_row_accumulation", "direct_v1"
            ),
            row_executor=_choice(scenario, "phase4_row_executor", "batched"),
            row_reduction=_choice(scenario, "phase4_row_reduction", "gpu_v1"),
            feature_batch_planning=phase4_planning,
            feature_batch_size_max=_optional_int(
                scenario.get("feature_batch_size_max")
            ),
            feature_batch_target_reserved_fraction=float(
                scenario.get("feature_batch_target_reserved_fraction", 0.9)
            ),
            feature_batch_min_free_fraction=float(
                scenario.get("feature_batch_min_free_fraction", 0.05)
            ),
            feature_batch_probe_batches=int(
                scenario.get("feature_batch_probe_batches", 1)
            ),
            feature_vjp_tape_batch_window=int(
                scenario.get("feature_vjp_tape_batch_window", 1)
            ),
            feature_vjp_tape_max_bytes=int(
                scenario.get("feature_vjp_tape_max_bytes", 0)
            ),
            decoder_page_prefetch_depth=int(
                scenario.get("decoder_page_prefetch_depth", 0)
            ),
            decoder_active_row_residency=bool(
                scenario.get("decoder_active_row_residency", False)
            ),
            decoder_active_row_max_bytes=int(
                scenario.get("decoder_active_row_max_bytes", 0)
            ),
            phase0_decoder_row_ranges=phase0_decoder_row_ranges,
        ),
        observability=ObservabilityPolicy(
            verbose=bool(scenario.get("verbose_attribution", False)),
            profile=bool(scenario.get("profile_attribution", False)),
            profile_log_interval=int(scenario.get("profile_log_interval", 1)),
            telemetry_max_events=_optional_int(scenario.get("telemetry_max_events")),
            phase4_anomaly_debug=bool(scenario.get("phase4_anomaly_debug", False)),
            cross_cluster_debug=bool(scenario.get("cross_cluster_debug", False)),
            capture_phase0_donor_bundle=bool(
                scenario.get("capture_phase0_donor_bundle", False)
            ),
            capture_phase3_seed_bundle=bool(
                scenario.get("capture_phase3_seed_bundle", False)
            ),
            capture_phase3_gradient_bundle=bool(
                scenario.get("capture_phase3_gradient_bundle", False)
            ),
            capture_phase3_row_bundle=bool(
                scenario.get("capture_phase3_row_bundle", False)
            ),
            capture_feature_semantic_descriptors=bool(
                scenario.get("capture_feature_semantic_descriptors", False)
            ),
            semantic_descriptor_top_k=int(
                scenario.get("semantic_descriptor_top_k", 2048)
            ),
            semantic_descriptor_dim=int(scenario.get("semantic_descriptor_dim", 64)),
        ),
        offload=None if scenario.get("no_offload", False) else "cpu",
        compact_output=not bool(scenario.get("save_raw", False)),
    )
    return TracePolicy(
        semantics=semantics,
        execution=execution,
        max_n_logits=int(scenario.get("max_n_logits", 5)),
        desired_logit_prob=float(scenario.get("desired_logit_prob", 0.9)),
        evidence=TraceEvidence(
            name="exact-trace-bench-scenario",
            version="c2",
            metadata={
                "scenario_name": scenario.get("name"),
                "method": scenario.get("method", "exact"),
                "governor_profile_name": (
                    None if provider_profile is None else provider_profile.profile_name
                ),
                "governor_admission_mode": governor_admission_mode.value,
            },
        ),
        resources=resources,
        provider_profile=provider_profile,
        physical_requirements=physical_requirements,
        governor_fidelity=governor_fidelity,
        governor_admission_mode=governor_admission_mode,
        response_bundle=response_bundle,
    )


def _response_bundle_from_scenario(
    scenario: Mapping[str, Any],
) -> ResponseBundle | None:
    raw = scenario.get("governor_response_bundle_path")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError("governor_response_bundle_path must be a non-empty path")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("governor_response_bundle_path must be absolute")
    return load_response_bundle(path)


def _governor_policy_from_scenario(
    scenario: Mapping[str, Any],
) -> tuple[ResourceEnvelope | None, ProviderProfile | None]:
    profile_name = scenario.get("governor_profile_name")
    envelope_value = scenario.get("governor_resource_envelope")
    if (profile_name is None) != (envelope_value is None):
        raise ValueError(
            "governor_profile_name and governor_resource_envelope must be supplied together"
        )
    if profile_name is None:
        return None, None
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError("governor_profile_name must be a non-empty string")
    try:
        provider_profile = RECORDED_PROVIDER_PROFILES[profile_name]
    except KeyError as error:
        known = ", ".join(sorted(RECORDED_PROVIDER_PROFILES))
        raise ValueError(
            f"unknown governor_profile_name {profile_name!r}; expected one of: {known}"
        ) from error
    if not isinstance(envelope_value, Mapping):
        raise ValueError("governor_resource_envelope must be an object")

    envelope = _cap_walltime_to_slurm(dict(envelope_value))
    if "cache_policy" in envelope:
        envelope["cache_policy"] = CachePolicy(envelope["cache_policy"])
    if "spill_roots" in envelope:
        envelope["spill_roots"] = tuple(envelope["spill_roots"])
    try:
        resources = ResourceEnvelope(**envelope)
    except TypeError as error:
        raise ValueError(f"invalid governor_resource_envelope: {error}") from error
    return resources, provider_profile


def _cap_walltime_to_slurm(
    envelope: dict[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    now_seconds: float | None = None,
    reserve_seconds: float = 120.0,
) -> dict[str, Any]:
    """Cap a requested governor walltime to the live allocation's remainder."""

    values = os.environ if environ is None else environ
    end_time = values.get("SLURM_JOB_END_TIME")
    if end_time is None:
        return envelope
    try:
        allocation_end = float(end_time)
    except ValueError as error:
        raise ValueError("SLURM_JOB_END_TIME must be epoch seconds") from error
    now = time.time() if now_seconds is None else now_seconds
    remaining = max(1.0, allocation_end - now - reserve_seconds)
    requested = float(envelope["walltime_seconds"])
    envelope["walltime_seconds"] = min(requested, remaining)
    return envelope


def _governor_fidelity_from_scenario(
    scenario: Mapping[str, Any],
) -> GovernorFidelityPolicy:
    normalized = parse_fidelity_policy(scenario)
    return _sibling_fidelity_policy(normalized)


def _sibling_fidelity_policy(
    policy: NormalizedFidelityPolicy,
) -> GovernorFidelityPolicy:
    """Bridge the project contract while the sibling four-mode API lands."""

    mode = FidelityMode(policy.mode)
    kwargs: dict[str, Any] = {"mode": mode, "override_fields": policy.override_fields}
    fields = getattr(GovernorFidelityPolicy, "__dataclass_fields__", {})
    if "evidence_name" in fields:
        kwargs["evidence_name"] = policy.evidence_name
    if "evidence_version" in fields:
        kwargs["evidence_version"] = policy.evidence_version
    if "budget" in fields and (
        policy.budget.metrics or policy.budget.allowed_sensitive_axes
    ):
        kwargs["budget"] = SiblingFidelityBudget(
            metric_floors=tuple(
                (name, metric.minimum)
                for name, metric in policy.budget.metrics.items()
            ),
            allowed_sensitive_axes=policy.budget.allowed_sensitive_axes,
            confidence=max(
                (metric.confidence for metric in policy.budget.metrics.values()),
                default=0.95,
            ),
            penalty_weight=(
                policy.budget.penalty
                if policy.budget.penalty is not None
                else 1.0
            ),
        )
    if "penalty" in fields:
        kwargs["penalty"] = policy.budget.penalty
    if "fidelity_budget" in fields:
        kwargs["fidelity_budget"] = {
            "metrics": {
                name: {
                    "minimum": metric.minimum,
                    "confidence": metric.confidence,
                }
                for name, metric in policy.budget.metrics.items()
            },
            "penalty": policy.budget.penalty,
        }
    return GovernorFidelityPolicy(
        **kwargs,
    )


def _physical_requirements_from_scenario(
    scenario: Mapping[str, Any],
) -> PhysicalExecutionRequirements:
    row_store_policy = None
    if scenario.get("governor_required_row_store_policy") is not None:
        row_store_policy = RowStorePolicy(str(scenario["governor_required_row_store_policy"]))
    encoder_residency = None
    if scenario.get("governor_required_encoder_residency") is not None:
        encoder_residency = (
            EncoderResidency.LAZY_PER_REQUEST
            if scenario["governor_required_encoder_residency"] == "lazy"
            else EncoderResidency.EAGER
        )
    spill_target = None
    if scenario.get("governor_required_spill_target") is not None:
        spill_target = StorageTier(str(scenario["governor_required_spill_target"]))
    return PhysicalExecutionRequirements(
        decoder_fetch_chunk_size=_optional_int(scenario.get("decoder_chunk_size")),
        decoder_cache_bytes=_optional_int(scenario.get("cross_batch_decoder_cache_bytes")),
        session_capacity=_optional_int(scenario.get("nnsight_session_capacity")),
        phase1_source_batch_size=(
            _optional_int(scenario.get("phase1_trace_batch_size_max"))
            if scenario.get("phase1_trace_batch_policy") == "cap_effective_batches"
            else None
        ),
        feature_microbatch_size=_optional_int(_phase4_execution_batch_max_rows(scenario)),
        logit_microbatch_size=_optional_int(
            scenario.get("phase3_compute_microbatch_max_rows")
        ),
        replay_window=_optional_int(scenario.get("chunked_feature_replay_window")),
        prefetch_depth=_optional_int(scenario.get("error_vector_prefetch_lookahead")),
        replay_tile_cache_bytes=_optional_int(
            scenario.get("governor_required_replay_tile_cache_bytes")
        ),
        encoder_residency=encoder_residency,
        row_store_policy=row_store_policy,
        spill_target=spill_target,
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _bool_knob(scenario: Mapping[str, Any], key: str, default: bool) -> bool:
    value = scenario.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a bool")
    return value


def _phase4_execution_batch_max_rows(scenario: Mapping[str, Any]) -> Any:
    value = scenario.get("phase4_execution_batch_max_rows")
    if value is not None:
        return value
    return scenario.get("phase4_compute_microbatch_max_rows")


def _decoder_cache_policy(value: Any) -> DecoderCachePolicy:
    max_bytes = _optional_int(value)
    if max_bytes is None or max_bytes == 0:
        return DecoderCachePolicy()
    return DecoderCachePolicy(enabled=True, max_bytes=max_bytes)


def _choice(scenario: Mapping[str, Any], key: str, default: str) -> Any:
    return scenario.get(key, default)


def _int_default(value: Any, default: int) -> int:
    return default if value is None else int(value)
