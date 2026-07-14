"""Project policy for constructing canonical sibling trace requests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from circuit_tracer import (
    AttributionProblem,
    DecoderCachePolicy,
    ExecutionConstraints,
    FrontierExpansionPlan,
    FrontierSemantics,
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
    ProviderProfile,
    ResourceEnvelope,
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
        )


def trace_policy_from_scenario(
    scenario: Mapping[str, Any],
    *,
    sparsification: Any | None = None,
) -> TracePolicy:
    """Resolve one scenario into the canonical sibling-owned policy types."""

    resources, provider_profile = _governor_policy_from_scenario(scenario)

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
            phase4_microbatch_max_rows=_optional_int(
                scenario.get("phase4_compute_microbatch_max_rows")
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
            },
        ),
        resources=resources,
        provider_profile=provider_profile,
    )


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

    envelope = dict(envelope_value)
    if "cache_policy" in envelope:
        envelope["cache_policy"] = CachePolicy(envelope["cache_policy"])
    if "spill_roots" in envelope:
        envelope["spill_roots"] = tuple(envelope["spill_roots"])
    try:
        resources = ResourceEnvelope(**envelope)
    except TypeError as error:
        raise ValueError(f"invalid governor_resource_envelope: {error}") from error
    return resources, provider_profile


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _decoder_cache_policy(value: Any) -> DecoderCachePolicy:
    max_bytes = _optional_int(value)
    if max_bytes is None or max_bytes == 0:
        return DecoderCachePolicy()
    return DecoderCachePolicy(enabled=True, max_bytes=max_bytes)


def _choice(scenario: Mapping[str, Any], key: str, default: str) -> Any:
    return scenario.get(key, default)


def _int_default(value: Any, default: int) -> int:
    return default if value is None else int(value)
