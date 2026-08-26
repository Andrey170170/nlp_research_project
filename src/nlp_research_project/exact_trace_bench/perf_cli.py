from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .config import DEFAULT_SCRATCH_ROOT, REPO_ROOT, base_trace_defaults
from .full_answer.launch_spec import (
    build_full_answer_launch_spec,
    render_local_full_answer_command,
)
from .full_answer.schemas import (
    build_trace_specs,
    write_shards,
    write_trace_selection,
    write_trace_specs,
)
from .full_answer.prepared_sequence import (
    PreparedSequenceEntry,
    build_prepared_sequence,
    load_prepared_sequence,
    run_prepared_sequence,
    write_prepared_sequence,
)
from .full_answer.prepared_workload import validate_prepared_workload
from .full_answer.selection import select_tokens
from .full_answer.sharding import build_lpt_shards
from .io_utils import read_json, write_json
from .performance_campaign import (
    freeze_performance_campaign,
    load_mechanism_claims,
    load_performance_campaign,
    render_campaign_workloads,
    resolve_frozen_campaign_workload,
)
from .scenarios.chpc_baseline import build_chpc_baseline_config
from .transcoder_config import PUBLIC_TRANSCODER_KNOB_KEYS
from .typed_compact_graph import CANONICAL_BUCKET_NAMES, DEFAULT_RETENTION_POLICY_ID
from .workspace import validate_launch_snapshot

DEFAULT_OUTPUT_ROOT = (
    DEFAULT_SCRATCH_ROOT / "granite" / "sweep" / "performance_optimization"
)
DEFAULT_BASELINE_REGISTRY = (
    REPO_ROOT
    / "experiments"
    / "baselines"
    / "exact_trace_performance_granite_20260726.json"
)
DEFAULT_MECHANISM_BASELINE_REGISTRY = (
    REPO_ROOT
    / "experiments"
    / "baselines"
    / "exact_trace_mechanism_granite_20260727.json"
)
RUNNER = REPO_ROOT / "experiments" / "run_sparsification_experiment.py"
SIBLING_ROOT = REPO_ROOT.parent / "circuit-tracer_chunked"

METRIC_KEYS: dict[str, str] = {
    "feature_jaccard": "worst_step_feature_jaccard",
    "target_token_match": "worst_step_target_token_match",
    **{
        f"bucket_{name.replace('<-', '_').replace('-', '_')}_exact": (
            f"worst_step_bucket_{name.replace('<-', '_').replace('-', '_')}_exact"
        )
        for name in CANONICAL_BUCKET_NAMES
    },
}
FIDELITY_LEVELS = (
    "strict_exact",
    "exact",
    "close",
    "bounded",
    "best_effort",
    "research",
)
CODE_RETENTION_ALLOWED_FIDELITIES = frozenset(
    {"strict_exact", "exact", "close", "bounded"}
)
EXACT_BASELINE_FIDELITIES = frozenset({"strict_exact", "exact"})
AUTOMATIC_PROMOTION_ALLOWED_FIDELITIES: frozenset[str] = frozenset()


def _typed_exact_thresholds(feature_jaccard: float) -> dict[str, float]:
    """Conservative Step-1a gate pending calibrated Step-1b bucket thresholds."""

    return {
        "worst_step_feature_jaccard_min": feature_jaccard,
        "worst_step_target_token_match_min": 1.0,
        **{
            f"worst_step_bucket_{name.replace('<-', '_').replace('-', '_')}_exact_min": 1.0
            for name in CANONICAL_BUCKET_NAMES
        },
    }


FIDELITY_THRESHOLDS = {
    "strict_exact": _typed_exact_thresholds(1.0),
    "exact": _typed_exact_thresholds(0.995),
    "close": _typed_exact_thresholds(0.99),
    "bounded": _typed_exact_thresholds(0.98),
    "best_effort": {},
    "research": {},
}
COMPARISON_SEMANTICS = {
    "strict_exact": "typed_bucket_v2_strict_exact",
    "exact": "typed_bucket_v2_exact_only_pending_step1b",
    "close": "typed_bucket_v2_exact_only_pending_step1b",
    "bounded": "typed_bucket_v2_exact_only_pending_step1b",
    "best_effort": "typed_bucket_v2_metrics_only",
    "research": "typed_bucket_v2_metrics_only",
}
COMPARISON_CLAIM_LIMITATION = (
    "Typed compact graphs preserve six signed retained edge buckets, but omit raw "
    "intermediates and non-retained state. Step 1a permits only exact per-bucket "
    "edge gating; calibrated non-exact and behavioral claims wait for Step 1b."
)
DEFAULT_RUN_GOAL = (
    "Improve exact-trace runtime without exceeding the selected parity budget."
)
PERFORMANCE_TARGET_SECONDS = {
    "performance/gemma3_1b_plt/361_base": 600.0,
    "performance/gemma3_4b_plt/361_base": 600.0,
}
MEASUREMENT_ONLY_PERFORMANCE_CASES = frozenset({"performance/gemma3_12b_plt/361_base"})
PERFORMANCE_STRETCH_TARGET_SECONDS = {
    "performance/gemma3_1b_plt/361_base": 300.0,
    "performance/gemma3_4b_plt/361_base": 600.0,
    "performance/gemma3_12b_plt/361_base": 3600.0,
}
ACTIVE_ROW_INITIAL_PREDICTED_DURATION_SECONDS = (65.0, 90.0)
ACTIVE_ROW_FUSED_TARGET_SECONDS = 245.0
ACTIVE_ROW_EXPECTED_BYTES = 66_158 * 1152 * 2
ACTIVE_ROW_EXPECTED_MIB_ALLOWANCE = (ACTIVE_ROW_EXPECTED_BYTES + 1024**2 - 1) // 1024**2
ACTIVE_ROW_FRAMEBUFFER_REFERENCE = {
    "run_id": "perf-plt-streaming-baseline-c65536-20260724-03",
    "resource_summary_path": (
        "/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/"
        "granite/sweep/performance_optimization/"
        "perf-plt-streaming-baseline-c65536-20260724-03/candidates/"
        "gemma3_1b_plt_361_base/resource_summary.json"
    ),
    "gpu_framebuffer_peak_mib": 26113.0,
}
LARGE_MODEL_HBM_PEAK_FRACTION_LIMIT = 0.90
_PLT_BOUNDED_TAPE_V1 = {
    "decoder_chunk_size": 65536,
    "nnsight_session_capacity": 256,
    "phase1_trace_batch_policy": "cap_effective_batches",
    "phase1_trace_batch_size_max": 128,
    "phase3_compute_microbatch_max_rows": 128,
    "phase4_execution_batch_max_rows": 256,
    "feature_vjp_tape_batch_window": 2,
    "feature_vjp_tape_max_bytes": 12 * 1024**3,
}
_LEGACY_CANDIDATE_OVERRIDES: dict[str, dict[str, Any]] = {
    "canonical": {},
    "clt-phase1-cap128-v1": {
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "nnsight_session_capacity": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 128,
    },
    "clt-phase1-cap256-v1": {
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 256,
        "nnsight_session_capacity": 256,
        "phase3_compute_microbatch_max_rows": 256,
        "phase4_execution_batch_max_rows": 256,
    },
    "clt-phase1-cap512-v1": {
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 512,
        "nnsight_session_capacity": 512,
        "phase3_compute_microbatch_max_rows": 512,
        "phase4_execution_batch_max_rows": 512,
    },
    "plt-bounded-fast-v1": {
        "decoder_chunk_size": 32768,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
    },
    "plt-bounded-fast-v2": {
        "decoder_chunk_size": 65536,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
    },
    "plt-bounded-fast-v3": {
        "decoder_chunk_size": 65536,
        "cross_batch_decoder_cache_bytes": 16 * 1024**3,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
    },
    "plt-bounded-tape-v1": {
        **_PLT_BOUNDED_TAPE_V1,
    },
    "plt-bounded-tape-prefetch-v1": {
        **_PLT_BOUNDED_TAPE_V1,
        "decoder_page_prefetch_depth": 1,
    },
    "plt-bounded-frontier-v1": {
        "decoder_chunk_size": 65536,
        "nnsight_session_capacity": 512,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 512,
    },
    "plt-active-rows-v1": {
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 1024**3,
    },
    "plt-active-rows-c65536-v1": {
        "decoder_chunk_size": 65536,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 1024**3,
    },
    "plt-phase0-coalesced-rows-v1": {
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 1024**3,
        "phase0_decoder_row_ranges": True,
    },
    "plt-active-rows-tape-v1": {
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
        "feature_vjp_tape_batch_window": 2,
        "feature_vjp_tape_max_bytes": 12 * 1024**3,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 1024**3,
    },
    "plt-active-rows-4b-c4096-v1": {
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 128,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 128,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 4 * 1024**3,
    },
    "plt-active-rows-4b-c65536-v1": {
        "decoder_chunk_size": 65536,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 128,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 128,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 4 * 1024**3,
    },
    "plt-active-rows-4b-b512-c4096-v1": {
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 512,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 512,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 4 * 1024**3,
    },
    "plt-active-rows-4b-b512-c65536-v1": {
        "decoder_chunk_size": 65536,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 512,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 512,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 4 * 1024**3,
    },
    "plt-active-rows-12b-c4096-v1": {
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 64,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 64,
        "phase3_compute_microbatch_max_rows": 64,
        "phase4_execution_batch_max_rows": 64,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 8 * 1024**3,
    },
    "plt-active-rows-12b-c32768-v1": {
        "decoder_chunk_size": 32768,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 64,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 64,
        "phase3_compute_microbatch_max_rows": 64,
        "phase4_execution_batch_max_rows": 64,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 8 * 1024**3,
    },
    "plt-active-rows-12b-c65536-v1": {
        "decoder_chunk_size": 65536,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 64,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 64,
        "phase3_compute_microbatch_max_rows": 64,
        "phase4_execution_batch_max_rows": 64,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 8 * 1024**3,
    },
    "plt-active-rows-12b-b256-c4096-v1": {
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 64,
        "phase3_compute_microbatch_max_rows": 64,
        "phase4_execution_batch_max_rows": 256,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 8 * 1024**3,
    },
    "plt-active-rows-12b-b256-c65536-v1": {
        "decoder_chunk_size": 65536,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 64,
        "phase3_compute_microbatch_max_rows": 64,
        "phase4_execution_batch_max_rows": 256,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 8 * 1024**3,
    },
}


class BaselineScope(str, Enum):
    SCIENTIFIC = "frozen_scientific"
    MECHANISM = "same_regime_mechanism"


@dataclass(frozen=True)
class BaselineRegistryEntry:
    key: str
    scope: BaselineScope
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class BaselineRegistryContract:
    registry_id: str
    scope: BaselineScope
    scope_declared: bool
    entries: Mapping[str, BaselineRegistryEntry]
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ProviderCapabilities:
    architecture: str
    decoder_output_topology: str
    layer_count: int | None
    supports_exact_chunked_provider: bool
    supports_exact_encoder_residency: bool
    supports_active_decoder_row_residency: bool
    supports_phase0_decoder_row_ranges: bool
    supports_decoder_row_source: bool


@dataclass(frozen=True)
class CapabilityRequirements:
    architecture: str | None = None
    decoder_output_topology: str | None = None
    supports_exact_chunked_provider: bool | None = None
    supports_exact_encoder_residency: bool | None = None
    supports_active_decoder_row_residency: bool | None = None
    supports_phase0_decoder_row_ranges: bool | None = None
    supports_decoder_row_source: bool | None = None
    minimum_layer_count: int | None = None
    maximum_layer_count: int | None = None

    def mismatch_reasons(self, capabilities: ProviderCapabilities) -> tuple[str, ...]:
        reasons: list[str] = []
        for field in (
            "architecture",
            "decoder_output_topology",
            "supports_exact_chunked_provider",
            "supports_exact_encoder_residency",
            "supports_active_decoder_row_residency",
            "supports_phase0_decoder_row_ranges",
            "supports_decoder_row_source",
        ):
            expected = getattr(self, field)
            observed = getattr(capabilities, field)
            if expected is not None and observed != expected:
                reasons.append(f"{field} requires {expected!r}, observed {observed!r}")
        if self.minimum_layer_count is not None and (
            capabilities.layer_count is None
            or capabilities.layer_count < self.minimum_layer_count
        ):
            reasons.append(
                f"layer_count requires >= {self.minimum_layer_count}, "
                f"observed {capabilities.layer_count!r}"
            )
        if self.maximum_layer_count is not None and (
            capabilities.layer_count is None
            or capabilities.layer_count > self.maximum_layer_count
        ):
            reasons.append(
                f"layer_count requires <= {self.maximum_layer_count}, "
                f"observed {capabilities.layer_count!r}"
            )
        return tuple(reasons)


@dataclass(frozen=True)
class SemanticBaselinePins:
    exact_trace_internal_dtype: str = "fp32"
    edge_retention_policy_id: str = DEFAULT_RETENTION_POLICY_ID


@dataclass(frozen=True)
class CompatibilityBaselinePins:
    decoder_chunk_size: int = 4096


@dataclass(frozen=True)
class BaselinePins:
    semantic: SemanticBaselinePins = SemanticBaselinePins()
    compatibility_mixed: CompatibilityBaselinePins = CompatibilityBaselinePins()

    def as_overrides(self) -> dict[str, Any]:
        return {
            "exact_trace_internal_dtype": self.semantic.exact_trace_internal_dtype,
            "edge_retention_policy_id": self.semantic.edge_retention_policy_id,
            "decoder_chunk_size": self.compatibility_mixed.decoder_chunk_size,
        }


@dataclass(frozen=True)
class PhysicalControls:
    values: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> PhysicalControls:
        if any(
            key in values
            for key in (
                "decoder_chunk_size",
                "exact_trace_internal_dtype",
                "edge_retention_policy_id",
            )
        ):
            raise ValueError(
                "semantic and compatibility-mixed values belong in baseline_pins"
            )
        return cls(tuple(values.items()))

    def as_overrides(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class EvidenceScope:
    exact_baseline: BaselineScope
    bounded_baseline: BaselineScope = BaselineScope.SCIENTIFIC

    def for_fidelity(self, fidelity: str) -> BaselineScope:
        return (
            self.exact_baseline
            if fidelity in EXACT_BASELINE_FIDELITIES
            else self.bounded_baseline
        )


@dataclass(frozen=True)
class CandidateVariant:
    requires: CapabilityRequirements
    baseline_pins: BaselinePins
    compatibility_controls: tuple[tuple[str, Any], ...]
    physical: PhysicalControls

    def overrides(self) -> dict[str, Any]:
        return {**dict(self.compatibility_controls), **self.physical.as_overrides()}


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    variants: tuple[CandidateVariant, ...]
    evidence_scope: EvidenceScope

    def variant_for(
        self, capabilities: ProviderCapabilities
    ) -> CandidateVariant | None:
        matches = [
            variant
            for variant in self.variants
            if not variant.requires.mismatch_reasons(capabilities)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"candidate profile {self.name!r} has overlapping capability variants"
            )
        return matches[0] if matches else None


_ANY_PROVIDER = CapabilityRequirements()
_CLT_CROSS_LAYER = CapabilityRequirements(
    architecture="clt",
    decoder_output_topology="cross_layer",
    supports_exact_chunked_provider=True,
)
_PLT_SAME_LAYER = CapabilityRequirements(
    architecture="plt",
    decoder_output_topology="same_layer",
    supports_exact_chunked_provider=True,
    supports_active_decoder_row_residency=True,
)
_PLT_MAPPED_DECODER_ROWS = CapabilityRequirements(
    **{
        **_PLT_SAME_LAYER.__dict__,
        "supports_phase0_decoder_row_ranges": True,
        "supports_decoder_row_source": True,
    }
)
_PLT_MAPPED_DECODER_ROWS_WITH_ENCODER_RESIDENCY = CapabilityRequirements(
    **{
        **_PLT_MAPPED_DECODER_ROWS.__dict__,
        "supports_exact_encoder_residency": True,
    }
)
_PLT_1B = CapabilityRequirements(
    **{
        **_PLT_SAME_LAYER.__dict__,
        "maximum_layer_count": 26,
    }
)
_PLT_4B = CapabilityRequirements(
    **{
        **_PLT_SAME_LAYER.__dict__,
        "minimum_layer_count": 27,
        "maximum_layer_count": 34,
    }
)
_PLT_12B = CapabilityRequirements(
    **{
        **_PLT_SAME_LAYER.__dict__,
        "minimum_layer_count": 35,
    }
)


def _candidate_variant(
    overrides: Mapping[str, Any],
    requires: CapabilityRequirements,
) -> CandidateVariant:
    values = dict(overrides)
    compatibility_controls: tuple[tuple[str, Any], ...] = ()
    if "decoder_chunk_size" in values:
        compatibility_controls = (
            ("decoder_chunk_size", int(values["decoder_chunk_size"])),
        )
    pins = BaselinePins(
        compatibility_mixed=CompatibilityBaselinePins(
            decoder_chunk_size=int(values.pop("decoder_chunk_size", 4096))
        )
    )
    return CandidateVariant(
        requires=requires,
        baseline_pins=pins,
        compatibility_controls=compatibility_controls,
        physical=PhysicalControls.from_mapping(values),
    )


def _profile_contracts() -> dict[str, CandidateProfile]:
    contracts: dict[str, CandidateProfile] = {
        "canonical": CandidateProfile(
            name="canonical",
            variants=(_candidate_variant({}, _ANY_PROVIDER),),
            evidence_scope=EvidenceScope(BaselineScope.SCIENTIFIC),
        )
    }
    clt_names = {
        "clt-phase1-cap128-v1",
        "clt-phase1-cap256-v1",
        "clt-phase1-cap512-v1",
    }
    plt_4b_names = {
        "plt-active-rows-4b-c4096-v1",
        "plt-active-rows-4b-c65536-v1",
        "plt-active-rows-4b-b512-c4096-v1",
        "plt-active-rows-4b-b512-c65536-v1",
    }
    plt_12b_names = {
        "plt-active-rows-12b-c4096-v1",
        "plt-active-rows-12b-c32768-v1",
        "plt-active-rows-12b-c65536-v1",
        "plt-active-rows-12b-b256-c4096-v1",
        "plt-active-rows-12b-b256-c65536-v1",
    }
    for name, overrides in _LEGACY_CANDIDATE_OVERRIDES.items():
        if name == "canonical":
            continue
        if name in clt_names:
            requirement = _CLT_CROSS_LAYER
            exact_scope = BaselineScope.SCIENTIFIC
        elif name in plt_4b_names:
            requirement = _PLT_4B
            exact_scope = BaselineScope.MECHANISM
        elif name in plt_12b_names:
            requirement = _PLT_12B
            exact_scope = BaselineScope.MECHANISM
        else:
            requirement = _PLT_1B
            exact_scope = (
                BaselineScope.MECHANISM
                if overrides.get("decoder_active_row_residency") is True
                else BaselineScope.SCIENTIFIC
            )
        if overrides.get("phase0_decoder_row_ranges") is True:
            requirement = CapabilityRequirements(
                **{
                    **requirement.__dict__,
                    "supports_phase0_decoder_row_ranges": True,
                }
            )
        contracts[name] = CandidateProfile(
            name=name,
            variants=(_candidate_variant(overrides, requirement),),
            evidence_scope=EvidenceScope(exact_scope),
        )

    contracts["plt-active-rows-large-c4096-v1"] = CandidateProfile(
        name="plt-active-rows-large-c4096-v1",
        variants=(
            _candidate_variant(
                _LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-4b-c4096-v1"],
                _PLT_4B,
            ),
            _candidate_variant(
                _LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-12b-c4096-v1"],
                _PLT_12B,
            ),
        ),
        evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
    )
    contracts["plt-selective-mapped-rows-large-v1"] = CandidateProfile(
        name="plt-selective-mapped-rows-large-v1",
        variants=(
            _candidate_variant(
                {
                    **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-4b-c4096-v1"],
                    "phase0_decoder_row_ranges": True,
                    "checkpoint_asset_scope": "job_private",
                },
                CapabilityRequirements(
                    **{
                        **_PLT_MAPPED_DECODER_ROWS.__dict__,
                        "minimum_layer_count": 27,
                        "maximum_layer_count": 34,
                    }
                ),
            ),
            _candidate_variant(
                {
                    **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-12b-c4096-v1"],
                    "phase0_decoder_row_ranges": True,
                    "checkpoint_asset_scope": "job_private",
                },
                CapabilityRequirements(
                    **{
                        **_PLT_MAPPED_DECODER_ROWS.__dict__,
                        "minimum_layer_count": 35,
                    }
                ),
            ),
        ),
        evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
    )
    contracts["sp5-exact-finalist-plt-v1"] = CandidateProfile(
        name="sp5-exact-finalist-plt-v1",
        variants=(
            _candidate_variant(
                {
                    **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-v1"],
                    "phase0_decoder_row_ranges": False,
                    "checkpoint_asset_scope": "shared",
                    "exact_encoder_residency": "lazy",
                    "feature_row_influence_mode": "cpu_exact",
                },
                CapabilityRequirements(
                    **{
                        **_PLT_SAME_LAYER.__dict__,
                        "maximum_layer_count": 26,
                    }
                ),
            ),
            _candidate_variant(
                {
                    **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-4b-c4096-v1"],
                    "phase0_decoder_row_ranges": False,
                    "checkpoint_asset_scope": "shared",
                    "exact_encoder_residency": "lazy",
                    "feature_row_influence_mode": "cpu_exact",
                },
                CapabilityRequirements(
                    **{
                        **_PLT_SAME_LAYER.__dict__,
                        "minimum_layer_count": 27,
                        "maximum_layer_count": 34,
                    }
                ),
            ),
            _candidate_variant(
                {
                    **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-12b-c4096-v1"],
                    "phase0_decoder_row_ranges": False,
                    "checkpoint_asset_scope": "shared",
                    "exact_encoder_residency": "lazy",
                    "feature_row_influence_mode": "cpu_exact",
                },
                CapabilityRequirements(
                    **{
                        **_PLT_SAME_LAYER.__dict__,
                        "minimum_layer_count": 35,
                    }
                ),
            ),
        ),
        evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
    )
    contracts["sp5-bounded-phase0-finalist-plt-v1"] = CandidateProfile(
        name="sp5-bounded-phase0-finalist-plt-v1",
        variants=tuple(
            _candidate_variant(
                {
                    **dict(variant.compatibility_controls),
                    **variant.physical.as_overrides(),
                    "phase0_decoder_row_ranges": True,
                },
                CapabilityRequirements(
                    **{
                        **_PLT_MAPPED_DECODER_ROWS.__dict__,
                        "minimum_layer_count": variant.requires.minimum_layer_count,
                        "maximum_layer_count": variant.requires.maximum_layer_count,
                    }
                ),
            )
            for variant in contracts["sp5-exact-finalist-plt-v1"].variants
        ),
        evidence_scope=EvidenceScope(
            BaselineScope.MECHANISM,
            bounded_baseline=BaselineScope.MECHANISM,
        ),
    )
    for alias, source in (
        ("sp5-canonical-phase0-reference-plt-v1", "sp5-exact-finalist-plt-v1"),
        (
            "sp5-selective-phase0-finalist-plt-v1",
            "sp5-bounded-phase0-finalist-plt-v1",
        ),
    ):
        source_profile = contracts[source]
        contracts[alias] = CandidateProfile(
            name=alias,
            variants=source_profile.variants,
            evidence_scope=source_profile.evidence_scope,
        )
    source_profile = contracts["sp5-selective-phase0-finalist-plt-v1"]
    contracts["ls2-1b-long-prefix-active-rows-1536mib-v1"] = CandidateProfile(
        name="ls2-1b-long-prefix-active-rows-1536mib-v1",
        variants=tuple(
            _candidate_variant(
                {
                    **dict(variant.compatibility_controls),
                    **variant.physical.as_overrides(),
                    "decoder_active_row_max_bytes": 1536 * 1024**2,
                },
                variant.requires,
            )
            for variant in source_profile.variants
            if variant.requires.maximum_layer_count == 26
        ),
        evidence_scope=source_profile.evidence_scope,
    )
    source_profile = contracts["ls2-1b-long-prefix-active-rows-1536mib-v1"]
    contracts["ls2-1b-long-prefix-active-rows-1536mib-b128-v1"] = CandidateProfile(
        name="ls2-1b-long-prefix-active-rows-1536mib-b128-v1",
        variants=tuple(
            _candidate_variant(
                {
                    **dict(variant.compatibility_controls),
                    **variant.physical.as_overrides(),
                    "phase4_execution_batch_max_rows": 128,
                },
                variant.requires,
            )
            for variant in source_profile.variants
        ),
        evidence_scope=source_profile.evidence_scope,
    )
    source_profile = contracts["ls2-1b-long-prefix-active-rows-1536mib-b128-v1"]
    contracts["ls2-1b-long-prefix-cuda-windowed-512mib-b128-v1"] = CandidateProfile(
        name="ls2-1b-long-prefix-cuda-windowed-512mib-b128-v1",
        variants=tuple(
            _candidate_variant(
                {
                    **dict(variant.compatibility_controls),
                    **variant.physical.as_overrides(),
                    "feature_row_influence_mode": "cuda_windowed",
                    "feature_row_gpu_window_max_bytes": 512 * 1024**2,
                    "feature_row_gpu_resident_safety_margin_bytes": 16 * 1024**3,
                },
                variant.requires,
            )
            for variant in source_profile.variants
        ),
        evidence_scope=EvidenceScope(
            BaselineScope.MECHANISM,
            bounded_baseline=BaselineScope.MECHANISM,
        ),
    )
    source_profile = contracts["sp5-canonical-phase0-reference-plt-v1"]
    contracts["ls2-1b-long-prefix-canonical-active-rows-1536mib-b128-v1"] = (
        CandidateProfile(
            name="ls2-1b-long-prefix-canonical-active-rows-1536mib-b128-v1",
            variants=tuple(
                _candidate_variant(
                    {
                        **dict(variant.compatibility_controls),
                        **variant.physical.as_overrides(),
                        "decoder_active_row_max_bytes": 1536 * 1024**2,
                        "phase4_execution_batch_max_rows": 128,
                    },
                    variant.requires,
                )
                for variant in source_profile.variants
                if variant.requires.maximum_layer_count == 26
            ),
            evidence_scope=source_profile.evidence_scope,
        )
    )
    source_profile = contracts["sp5-selective-phase0-finalist-plt-v1"]
    ls4_4b_variants = tuple(
        variant
        for variant in source_profile.variants
        if variant.requires.minimum_layer_count == 27
        and variant.requires.maximum_layer_count == 34
    )
    contracts["ls4-4b-transfer-cpu-exact-b128-v1"] = CandidateProfile(
        name="ls4-4b-transfer-cpu-exact-b128-v1",
        variants=ls4_4b_variants,
        evidence_scope=source_profile.evidence_scope,
    )
    contracts["ls4-4b-transfer-cuda-windowed-512mib-b128-v1"] = CandidateProfile(
        name="ls4-4b-transfer-cuda-windowed-512mib-b128-v1",
        variants=tuple(
            _candidate_variant(
                {
                    **dict(variant.compatibility_controls),
                    **variant.physical.as_overrides(),
                    "feature_row_influence_mode": "cuda_windowed",
                    "feature_row_gpu_window_max_bytes": 512 * 1024**2,
                    "feature_row_gpu_resident_safety_margin_bytes": 16 * 1024**3,
                },
                variant.requires,
            )
            for variant in ls4_4b_variants
        ),
        evidence_scope=EvidenceScope(
            BaselineScope.MECHANISM,
            bounded_baseline=BaselineScope.MECHANISM,
        ),
    )
    contracts["ls4-4b-transfer-cuda-windowed-512mib-b128-batched-vjp-v1"] = (
        CandidateProfile(
            name="ls4-4b-transfer-cuda-windowed-512mib-b128-batched-vjp-v1",
            variants=tuple(
                _candidate_variant(
                    {
                        **dict(variant.compatibility_controls),
                        **variant.physical.as_overrides(),
                        "feature_row_influence_mode": "cuda_windowed",
                        "feature_row_gpu_window_max_bytes": 512 * 1024**2,
                        "feature_row_gpu_resident_safety_margin_bytes": 16 * 1024**3,
                        "backward_engine_mode": "single_forward_batched_vjp",
                    },
                    variant.requires,
                )
                for variant in ls4_4b_variants
            ),
            evidence_scope=EvidenceScope(
                BaselineScope.MECHANISM,
                bounded_baseline=BaselineScope.MECHANISM,
            ),
        )
    )
    source_profile = contracts[
        "ls4-4b-transfer-cuda-windowed-512mib-b128-batched-vjp-v1"
    ]
    contracts["ls4-4b-self-consistency-cuda-windowed-width1-batched-vjp-v1"] = (
        CandidateProfile(
            name=("ls4-4b-self-consistency-cuda-windowed-width1-batched-vjp-v1"),
            variants=tuple(
                _candidate_variant(
                    {
                        **dict(variant.compatibility_controls),
                        **variant.physical.as_overrides(),
                        "feature_row_influence_requirement": "required",
                    },
                    variant.requires,
                )
                for variant in source_profile.variants
            ),
            evidence_scope=source_profile.evidence_scope,
        )
    )
    diagnostic_overrides = {
        "feature_row_influence_mode": "cuda_windowed",
        "feature_row_influence_requirement": "required",
        "feature_row_gpu_window_max_bytes": 512 * 1024**2,
        "feature_row_gpu_resident_safety_margin_bytes": 16 * 1024**3,
        "cross_cluster_debug": True,
        "capture_phase0_donor_bundle": True,
        "capture_phase3_seed_bundle": True,
        "capture_phase3_gradient_bundle": True,
        "capture_phase3_row_bundle": True,
    }
    for profile_suffix, backward_engine_mode in (
        ("duplicated-lanes", "duplicated_lanes"),
        ("single-forward-serial-vjp", "single_forward_serial_vjp"),
        ("single-forward-batched-vjp", "single_forward_batched_vjp"),
    ):
        profile_name = f"ls4-4b-vjp-diagnostic-{profile_suffix}-v1"
        contracts[profile_name] = CandidateProfile(
            name=profile_name,
            variants=tuple(
                _candidate_variant(
                    {
                        **dict(variant.compatibility_controls),
                        **variant.physical.as_overrides(),
                        **diagnostic_overrides,
                        "backward_engine_mode": backward_engine_mode,
                    },
                    variant.requires,
                )
                for variant in ls4_4b_variants
            ),
            evidence_scope=EvidenceScope(
                BaselineScope.MECHANISM,
                bounded_baseline=BaselineScope.MECHANISM,
            ),
        )

    def register_vjp_diagnostic_profile(
        *, profile_suffix: str, version: int, physical_overrides: Mapping[str, Any]
    ) -> None:
        profile_name = f"ls4-4b-vjp-diagnostic-{profile_suffix}-v{version}"
        contracts[profile_name] = CandidateProfile(
            name=profile_name,
            variants=tuple(
                _candidate_variant(
                    {
                        **dict(variant.compatibility_controls),
                        **variant.physical.as_overrides(),
                        **diagnostic_overrides,
                        **physical_overrides,
                    },
                    variant.requires,
                )
                for variant in ls4_4b_variants
            ),
            evidence_scope=EvidenceScope(
                BaselineScope.MECHANISM,
                bounded_baseline=BaselineScope.MECHANISM,
            ),
        )

    diagnostic_widths = {
        "wide-injected": {
            "backward_engine_mode": "duplicated_lanes",
            "nnsight_session_capacity": 128,
            "phase1_trace_batch_policy": "cap_effective_batches",
            "phase1_trace_batch_size_max": 128,
            "phase3_compute_microbatch_max_rows": 128,
            "phase4_execution_batch_max_rows": 128,
        },
        "narrow-injected": {
            "backward_engine_mode": "duplicated_lanes",
            "nnsight_session_capacity": 1,
            "phase1_trace_batch_policy": "cap_effective_batches",
            "phase1_trace_batch_size_max": 1,
            "phase3_compute_microbatch_max_rows": 1,
            "phase4_execution_batch_max_rows": 1,
        },
        "narrow-serial": {
            "backward_engine_mode": "single_forward_serial_vjp",
            "nnsight_session_capacity": 1,
            "phase1_trace_batch_policy": "cap_effective_batches",
            "phase1_trace_batch_size_max": 1,
            "phase3_compute_microbatch_max_rows": 1,
            "phase4_execution_batch_max_rows": 1,
        },
        "narrow-batched": {
            "backward_engine_mode": "single_forward_batched_vjp",
            "nnsight_session_capacity": 1,
            "phase1_trace_batch_policy": "cap_effective_batches",
            "phase1_trace_batch_size_max": 1,
            "phase3_compute_microbatch_max_rows": 1,
            "phase4_execution_batch_max_rows": 1,
        },
    }
    for profile_suffix, width_overrides in diagnostic_widths.items():
        register_vjp_diagnostic_profile(
            profile_suffix=profile_suffix,
            version=2,
            physical_overrides=width_overrides,
        )

    # Diagnostic-only structural-width response profiles. Session capacity,
    # duplicated forward-graph lanes, and injected backward capacity are one
    # coupled factor; they cannot be varied independently by this runtime.
    for session_width in (128, 64, 32, 16, 8, 4, 2, 1):
        register_vjp_diagnostic_profile(
            profile_suffix=f"injected-width-{session_width}-caps-1",
            version=1,
            physical_overrides={
                "backward_engine_mode": "duplicated_lanes",
                "nnsight_session_capacity": session_width,
                "phase1_trace_batch_policy": "cap_effective_batches",
                "phase1_trace_batch_size_max": 1,
                "phase3_compute_microbatch_max_rows": 1,
                "phase4_execution_batch_max_rows": 1,
            },
        )

    mapped_4b_requirements = CapabilityRequirements(
        **{
            **_PLT_MAPPED_DECODER_ROWS.__dict__,
            "minimum_layer_count": 27,
            "maximum_layer_count": 34,
        }
    )
    mapped_4b_controls = {
        **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-4b-c4096-v1"],
        "phase0_decoder_row_ranges": True,
    }
    for telemetry_enabled in (False, True):
        suffix = "on" if telemetry_enabled else "off"
        profile_name = f"plt-selective-mapped-rows-telemetry-{suffix}-4b-v1"
        contracts[profile_name] = CandidateProfile(
            name=profile_name,
            variants=(
                _candidate_variant(
                    {
                        **mapped_4b_controls,
                        "checkpoint_asset_scope": "shared",
                        "telemetry_enabled": telemetry_enabled,
                        "incremental_telemetry_jsonl": telemetry_enabled,
                        "profile_attribution": True,
                        "verbose_attribution": False,
                    },
                    mapped_4b_requirements,
                ),
            ),
            evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
        )
    contracts["plt-selective-mapped-rows-lifecycle-reference-4b-v1"] = CandidateProfile(
        name="plt-selective-mapped-rows-lifecycle-reference-4b-v1",
        variants=(
            _candidate_variant(
                {
                    **mapped_4b_controls,
                    "checkpoint_asset_scope": "shared",
                },
                mapped_4b_requirements,
            ),
        ),
        evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
    )
    contracts["plt-selective-mapped-rows-lifecycle-private-4b-v1"] = CandidateProfile(
        name="plt-selective-mapped-rows-lifecycle-private-4b-v1",
        variants=(
            _candidate_variant(
                {
                    **mapped_4b_controls,
                    "checkpoint_asset_scope": "job_private",
                },
                mapped_4b_requirements,
            ),
        ),
        evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
    )
    contracts["plt-selective-mapped-rows-gpu-store-12b-v1"] = CandidateProfile(
        name="plt-selective-mapped-rows-gpu-store-12b-v1",
        variants=(
            _candidate_variant(
                {
                    **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-12b-c4096-v1"],
                    "phase0_decoder_row_ranges": True,
                    "checkpoint_asset_scope": "job_private",
                    "feature_row_influence_mode": "cuda_full",
                    "feature_row_gpu_resident_max_bytes": 8 * 1024**3,
                    "feature_row_gpu_resident_safety_margin_bytes": 16 * 1024**3,
                },
                CapabilityRequirements(
                    **{
                        **_PLT_MAPPED_DECODER_ROWS.__dict__,
                        "minimum_layer_count": 35,
                    }
                ),
            ),
        ),
        evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
    )
    for mode, extra_overrides in (
        (
            "cpu-exact",
            {
                "feature_row_influence_mode": "cpu_exact",
            },
        ),
        (
            "cpu-prepared",
            {
                "feature_row_influence_mode": "cpu_prepared",
            },
        ),
        (
            "cuda-windowed",
            {
                "feature_row_influence_mode": "cuda_windowed",
                "feature_row_gpu_window_max_bytes": 512 * 1024**2,
                "feature_row_gpu_resident_safety_margin_bytes": 16 * 1024**3,
            },
        ),
        (
            "cuda-auto",
            {
                "feature_row_influence_mode": "auto",
                "feature_row_gpu_resident_max_bytes": 8 * 1024**3,
                "feature_row_gpu_window_max_bytes": 512 * 1024**2,
                "feature_row_gpu_resident_safety_margin_bytes": 16 * 1024**3,
            },
        ),
    ):
        profile_name = f"plt-selective-mapped-rows-{mode}-12b-v1"
        contracts[profile_name] = CandidateProfile(
            name=profile_name,
            variants=(
                _candidate_variant(
                    {
                        **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-12b-c4096-v1"],
                        "phase0_decoder_row_ranges": True,
                        "checkpoint_asset_scope": "job_private",
                        **extra_overrides,
                    },
                    CapabilityRequirements(
                        **{
                            **_PLT_MAPPED_DECODER_ROWS.__dict__,
                            "minimum_layer_count": 35,
                        }
                    ),
                ),
            ),
            evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
        )
    source_profile = contracts["plt-selective-mapped-rows-cuda-windowed-12b-v1"]
    contracts["ls5-12b-cuda-windowed-width1-batched-vjp-v1"] = CandidateProfile(
        name="ls5-12b-cuda-windowed-width1-batched-vjp-v1",
        variants=tuple(
            _candidate_variant(
                {
                    **dict(variant.compatibility_controls),
                    **variant.physical.as_overrides(),
                    "checkpoint_asset_scope": "shared",
                    "backward_engine_mode": "single_forward_batched_vjp",
                    "feature_row_influence_requirement": "required",
                },
                variant.requires,
            )
            for variant in source_profile.variants
        ),
        evidence_scope=source_profile.evidence_scope,
    )
    source_profile = contracts["ls5-12b-cuda-windowed-width1-batched-vjp-v1"]
    contracts["ls5-12b-cuda-windowed-width1-batched-vjp-dynamic-rows-v2"] = (
        CandidateProfile(
            name="ls5-12b-cuda-windowed-width1-batched-vjp-dynamic-rows-v2",
            variants=tuple(
                _candidate_variant(
                    {
                        **dict(variant.compatibility_controls),
                        **variant.physical.as_overrides(),
                        "decoder_active_row_max_bytes": 0,
                        "decoder_active_row_safety_margin_bytes": 16 * 1024**3,
                        "decoder_active_row_residency_requirement": "required",
                    },
                    variant.requires,
                )
                for variant in source_profile.variants
            ),
            evidence_scope=source_profile.evidence_scope,
        )
    )
    for encoder_mode in ("active_cpu",):
        profile_name = (
            f"plt-selective-mapped-rows-{encoder_mode.replace('_', '-')}-large-v1"
        )
        contracts[profile_name] = CandidateProfile(
            name=profile_name,
            variants=(
                _candidate_variant(
                    {
                        **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-4b-c4096-v1"],
                        "phase0_decoder_row_ranges": True,
                        "checkpoint_asset_scope": "job_private",
                        "exact_encoder_residency": encoder_mode,
                    },
                    CapabilityRequirements(
                        **{
                            **_PLT_MAPPED_DECODER_ROWS_WITH_ENCODER_RESIDENCY.__dict__,
                            "minimum_layer_count": 27,
                            "maximum_layer_count": 34,
                        }
                    ),
                ),
                _candidate_variant(
                    {
                        **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-12b-c4096-v1"],
                        "phase0_decoder_row_ranges": True,
                        "checkpoint_asset_scope": "job_private",
                        "exact_encoder_residency": encoder_mode,
                    },
                    CapabilityRequirements(
                        **{
                            **_PLT_MAPPED_DECODER_ROWS_WITH_ENCODER_RESIDENCY.__dict__,
                            "minimum_layer_count": 35,
                        }
                    ),
                ),
            ),
            evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
        )
    sp1_diagnostic_requirements = CapabilityRequirements(
        **{
            **_PLT_MAPPED_DECODER_ROWS_WITH_ENCODER_RESIDENCY.__dict__,
            "minimum_layer_count": 35,
        }
    )
    for suffix, encoder_mode in (("lazy", "lazy"), ("active-cpu", "active_cpu")):
        profile_name = f"plt-sp1-{suffix}-diagnostic-12b-v1"
        contracts[profile_name] = CandidateProfile(
            name=profile_name,
            variants=(
                _candidate_variant(
                    {
                        **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-12b-c4096-v1"],
                        "phase0_decoder_row_ranges": True,
                        "checkpoint_asset_scope": "shared",
                        "exact_encoder_residency": encoder_mode,
                        "capture_phase3_row_bundle": True,
                        "capture_phase3_seed_bundle": True,
                        "profile_attribution": True,
                    },
                    sp1_diagnostic_requirements,
                ),
            ),
            evidence_scope=EvidenceScope(BaselineScope.SCIENTIFIC),
        )
    for execution_rows in (128, 256):
        profile_name = f"plt-selective-mapped-rows-12b-b{execution_rows}-v1"
        contracts[profile_name] = CandidateProfile(
            name=profile_name,
            variants=(
                _candidate_variant(
                    {
                        **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-12b-c4096-v1"],
                        "nnsight_session_capacity": execution_rows,
                        "phase4_execution_batch_max_rows": execution_rows,
                        "phase0_decoder_row_ranges": True,
                        "checkpoint_asset_scope": "job_private",
                    },
                    CapabilityRequirements(
                        **{
                            **_PLT_MAPPED_DECODER_ROWS.__dict__,
                            "minimum_layer_count": 35,
                        }
                    ),
                ),
            ),
            evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
        )
    for row_subchunk_size in (16384, 65536):
        profile_name = (
            f"plt-selective-mapped-rows-active-cpu-12b-tile{row_subchunk_size}-v1"
        )
        contracts[profile_name] = CandidateProfile(
            name=profile_name,
            variants=(
                _candidate_variant(
                    {
                        **_LEGACY_CANDIDATE_OVERRIDES["plt-active-rows-12b-c4096-v1"],
                        "phase0_decoder_row_ranges": True,
                        "checkpoint_asset_scope": "job_private",
                        "exact_encoder_residency": "active_cpu",
                        "row_subchunk_size": row_subchunk_size,
                    },
                    CapabilityRequirements(
                        **{
                            **_PLT_MAPPED_DECODER_ROWS_WITH_ENCODER_RESIDENCY.__dict__,
                            "minimum_layer_count": 35,
                        }
                    ),
                ),
            ),
            evidence_scope=EvidenceScope(BaselineScope.MECHANISM),
        )
    return contracts


CANDIDATE_PROFILES = _profile_contracts()
GPU_SAMPLE_COMMAND = (
    "nvidia-smi",
    (
        "--query-gpu=timestamp,index,uuid,name,utilization.gpu,"
        "utilization.memory,power.draw,memory.used,memory.total"
    ),
    "--format=csv,noheader,nounits",
    "--loop=1",
)


def _supports_mapped_decoder_row_source(scenario: Mapping[str, Any]) -> bool:
    """Return whether the generated scenario has the verified mapped-row contract."""
    provider_family = scenario.get("transcoder_provider_family")
    decoder_chunk_size = scenario.get("decoder_chunk_size")
    return (
        scenario.get("transcoder_architecture") == "plt"
        and isinstance(provider_family, str)
        and provider_family.startswith("gemmascope2-plt-")
        and scenario.get("lazy_decoder") is True
        and isinstance(decoder_chunk_size, int)
        and not isinstance(decoder_chunk_size, bool)
        and decoder_chunk_size > 0
    )


@dataclass(frozen=True)
class Case:
    variant: str
    fixture: str

    @property
    def key(self) -> str:
        return f"performance/{self.variant}/{self.fixture}"

    def provider_capabilities(self) -> ProviderCapabilities:
        return _provider_capabilities_for_variant(self.variant)


def _provider_capabilities_for_variant(variant: str) -> ProviderCapabilities:
    """Derive provider capabilities independently of a particular fixture."""
    payload = build_chpc_baseline_config(variant=variant, cluster="granite")
    scenario = payload["scenarios"][0]
    architecture = str(scenario["transcoder_architecture"])
    layer_count_raw = scenario.get("layer_count")
    layer_count = int(layer_count_raw) if isinstance(layer_count_raw, int) else None
    same_layer = architecture == "plt"
    supports_mapped_rows = _supports_mapped_decoder_row_source(scenario)
    return ProviderCapabilities(
        architecture=architecture,
        decoder_output_topology=("same_layer" if same_layer else "cross_layer"),
        layer_count=layer_count,
        supports_exact_chunked_provider=True,
        supports_exact_encoder_residency=True,
        supports_active_decoder_row_residency=same_layer,
        supports_phase0_decoder_row_ranges=same_layer,
        supports_decoder_row_source=supports_mapped_rows,
    )


SUITES: dict[str, tuple[Case, ...]] = {
    "clt-smoke": (Case("gemma3_1b_clt", "361_base"),),
    "clt-pair": (
        Case("gemma3_1b_clt", "828_base"),
        Case("gemma3_1b_clt", "361_base"),
    ),
    "plt-hard": (Case("gemma3_1b_plt", "361_base"),),
    "plt-breadth": (
        Case("gemma3_1b_plt", "828_base"),
        Case("gemma3_1b_plt", "94_base"),
    ),
    "plt-4b": (Case("gemma3_4b_plt", "361_base"),),
    "plt-4b-holdout": (Case("gemma3_4b_plt", "828_base"),),
    "plt-12b": (Case("gemma3_12b_plt", "361_base"),),
    "plt-large": (
        Case("gemma3_4b_plt", "361_base"),
        Case("gemma3_12b_plt", "361_base"),
    ),
    "all": (
        Case("gemma3_1b_clt", "828_base"),
        Case("gemma3_1b_clt", "361_base"),
        Case("gemma3_1b_plt", "361_base"),
    ),
}


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def workspace_state(root: Path) -> dict[str, Any]:
    return {
        "path": str(root),
        "branch": _git_output(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git_output(root, "rev-parse", "HEAD"),
        "dirty_files": _git_output(root, "status", "--short").splitlines(),
        "source_content_sha256": source_content_sha256(root),
    }


def source_content_sha256(root: Path) -> str:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = sorted(
        Path(raw.decode("utf-8", errors="surrogateescape"))
        for raw in completed.stdout.split(b"\0")
        if raw
    )
    digest = hashlib.sha256()
    for relative_path in paths:
        digest.update(os.fsencode(str(relative_path)))
        digest.update(b"\0")
        path = root / relative_path
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(b"file\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"missing-or-nonfile\0")
        digest.update(b"\0")
    return digest.hexdigest()


def capture_source_state() -> dict[str, Any]:
    try:
        return {
            "workspace_mode": "live",
            "project": workspace_state(REPO_ROOT),
            "sibling": workspace_state(SIBLING_ROOT),
        }
    except (OSError, subprocess.CalledProcessError):
        manifest_path = REPO_ROOT.parent / ".exact_trace_bench_snapshot.json"
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "exact-trace-perf provenance requires git repositories or a valid "
                "immutable snapshot manifest"
            ) from exc
        if not isinstance(manifest, dict) or manifest.get("read_only") is not True:
            raise RuntimeError(
                "immutable snapshot manifest is missing or not read-only"
            )
        if (
            Path(str(manifest.get("snapshot_root", ""))).resolve()
            != REPO_ROOT.resolve()
        ):
            raise RuntimeError("snapshot manifest does not match the project workspace")
        snapshots = manifest.get("uv_source_snapshots")
        if not isinstance(snapshots, list):
            raise RuntimeError("snapshot manifest lacks editable source provenance")
        sibling_snapshot = next(
            (
                item
                for item in snapshots
                if isinstance(item, dict)
                and Path(str(item.get("snapshot_path", ""))).resolve()
                == SIBLING_ROOT.resolve()
            ),
            None,
        )
        project_state = manifest.get("repo_state")
        sibling_state = (
            sibling_snapshot.get("repo_state")
            if isinstance(sibling_snapshot, dict)
            else None
        )
        if not isinstance(project_state, dict) or not isinstance(sibling_state, dict):
            raise RuntimeError(
                "snapshot manifest lacks recorded project/sibling states"
            )
        return {
            "workspace_mode": "immutable",
            "snapshot_manifest_path": str(manifest_path.resolve()),
            "snapshot_manifest": manifest,
            "project": project_state,
            "sibling": sibling_state,
        }


def gpu_provenance(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    gpu_rows = [
        {
            "name": fields[0],
            "uuid": fields[1],
            "driver_version": fields[2],
            "memory_total_mib": int(fields[3]),
        }
        for line in completed.stdout.splitlines()
        if line.strip()
        for fields in ([field.strip() for field in line.split(",", maxsplit=3)],)
        if len(fields) == 4
    ]
    return {
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "slurm_job_name": env.get("SLURM_JOB_NAME"),
        "slurm_cluster_name": env.get("SLURM_CLUSTER_NAME"),
        "slurm_job_partition": env.get("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_returncode": completed.returncode,
        "gpus": gpu_rows,
    }


def assert_h200_allocation(
    provenance: dict[str, Any],
    *,
    allow_test_environment: bool,
) -> None:
    if allow_test_environment:
        return
    if not provenance.get("slurm_job_id"):
        raise RuntimeError(
            "exact-trace-perf run requires an active SLURM allocation; "
            "use --dry-run outside one"
        )
    gpus = provenance.get("gpus") or []
    if len(gpus) != 1:
        raise RuntimeError(
            f"exact-trace-perf requires exactly one visible GPU, found {len(gpus)}"
        )
    if "H200" not in str(gpus[0].get("name", "")).upper():
        names = ", ".join(str(gpu.get("name")) for gpu in gpus)
        raise RuntimeError(f"exact-trace-perf requires H200 GPU(s), found: {names}")
    if int(gpus[0].get("memory_total_mib") or 0) < 130_000:
        raise RuntimeError("exact-trace-perf requires one full, non-MIG H200")
    if provenance.get("slurm_cluster_name") != "granite":
        raise RuntimeError("exact-trace-perf timing requires SLURM cluster granite")
    if provenance.get("slurm_job_partition") != "rai-gpu-grn":
        raise RuntimeError(
            "exact-trace-perf timing requires SLURM partition rai-gpu-grn"
        )


def execution_evidence(
    provenance: dict[str, Any],
    *,
    allow_test_environment: bool,
) -> dict[str, Any]:
    return {
        "test_environment_override_used": allow_test_environment,
        "timing_evidence_eligible": not allow_test_environment,
        "timing_evidence_status": (
            "test_override_non_evidence"
            if allow_test_environment
            else "h200_allocation"
        ),
        "observed_gpu_names": [
            str(gpu.get("name")) for gpu in provenance.get("gpus") or []
        ],
    }


def _candidate_overrides(case: Case, candidate_profile: str) -> dict[str, Any]:
    profile = CANDIDATE_PROFILES[candidate_profile]
    variant = profile.variant_for(case.provider_capabilities())
    if variant is None:
        return {}
    return variant.overrides()


def _profile_variant(case: Case, candidate_profile: str) -> CandidateVariant:
    profile = CANDIDATE_PROFILES[candidate_profile]
    capabilities = case.provider_capabilities()
    variant = profile.variant_for(capabilities)
    if variant is None:
        alternatives = [
            "; ".join(candidate.requires.mismatch_reasons(capabilities))
            for candidate in profile.variants
        ]
        raise ValueError(
            f"candidate profile {candidate_profile!r} is not applicable to "
            f"{case.key}: " + " | ".join(alternatives)
        )
    return variant


def _baseline_scope(
    candidate_profile: str,
    fidelity: str,
) -> BaselineScope:
    return CANDIDATE_PROFILES[candidate_profile].evidence_scope.for_fidelity(fidelity)


def _baseline_registry_path(scope: BaselineScope) -> Path:
    if scope is BaselineScope.MECHANISM:
        return DEFAULT_MECHANISM_BASELINE_REGISTRY
    return DEFAULT_BASELINE_REGISTRY


def _load_baseline_registry_contract(
    registry_path: Path,
    *,
    expected_scope: BaselineScope,
) -> BaselineRegistryContract:
    payload = read_json(registry_path)
    declared_scope_raw = payload.get("scope")
    scope_declared = declared_scope_raw is not None
    if declared_scope_raw is None:
        if expected_scope is not BaselineScope.SCIENTIFIC:
            raise ValueError(
                f"baseline registry {registry_path} must declare scope "
                f"{expected_scope.value!r}"
            )
        scope = BaselineScope.SCIENTIFIC
    else:
        try:
            scope = BaselineScope(str(declared_scope_raw))
        except ValueError as exc:
            raise ValueError(
                f"baseline registry {registry_path} has unsupported scope "
                f"{declared_scope_raw!r}"
            ) from exc
    if scope is not expected_scope:
        raise ValueError(
            f"baseline registry {registry_path} scope {scope.value!r} does not "
            f"match required scope {expected_scope.value!r}"
        )
    entries_raw = payload.get("entries")
    if not isinstance(entries_raw, Mapping):
        raise ValueError(f"baseline registry {registry_path} must contain entries")
    entries: dict[str, BaselineRegistryEntry] = {}
    for key, entry_raw in entries_raw.items():
        if not isinstance(entry_raw, Mapping):
            raise ValueError(f"baseline registry entry {key!r} must be an object")
        entry_scope_raw = entry_raw.get("scope")
        if entry_scope_raw is None:
            if scope is BaselineScope.MECHANISM:
                raise ValueError(
                    f"mechanism baseline entry {key!r} must declare scope "
                    f"{scope.value!r}"
                )
            entry_scope = scope
        else:
            try:
                entry_scope = BaselineScope(str(entry_scope_raw))
            except ValueError as exc:
                raise ValueError(
                    f"baseline registry entry {key!r} has unsupported scope "
                    f"{entry_scope_raw!r}"
                ) from exc
            if entry_scope is not scope:
                raise ValueError(
                    f"baseline registry entry {key!r} scope "
                    f"{entry_scope.value!r} does not match registry scope "
                    f"{scope.value!r}"
                )
        entries[str(key)] = BaselineRegistryEntry(
            key=str(key),
            scope=entry_scope,
            payload=entry_raw,
        )
    return BaselineRegistryContract(
        registry_id=str(payload.get("registry_id") or ""),
        scope=scope,
        scope_declared=scope_declared,
        entries=entries,
        payload=payload,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_baseline_pin_facts(
    *,
    baseline_entry: Mapping[str, Any],
    scope: BaselineScope,
) -> dict[str, Any]:
    declared = baseline_entry.get("baseline_pins")
    if declared is not None and not isinstance(declared, Mapping):
        raise ValueError("baseline_pins must be an object")
    result_json_raw = baseline_entry.get("result_json")
    scenario_path_raw = baseline_entry.get("scenario_json")
    scenario_path = (
        Path(str(scenario_path_raw))
        if scenario_path_raw is not None
        else (
            Path(str(result_json_raw)).with_name("scenario.json")
            if result_json_raw is not None
            else None
        )
    )
    if scenario_path is not None and scenario_path.is_file():
        scenario = read_json(scenario_path)
        observed = {
            "exact_trace_internal_dtype": scenario.get("exact_trace_internal_dtype"),
            # Pre-Step-1a scenarios did not name the policy even though the
            # historical adapter verifies the same typed bucket metadata.
            "edge_retention_policy_id": scenario.get(
                "edge_retention_policy_id", DEFAULT_RETENTION_POLICY_ID
            ),
            "decoder_chunk_size": scenario.get("decoder_chunk_size"),
        }
        expected_sha = baseline_entry.get("scenario_sha256")
        observed_sha = _sha256_file(scenario_path)
        if expected_sha is not None and str(expected_sha) != observed_sha:
            raise ValueError(f"baseline scenario checksum mismatch for {scenario_path}")
        if declared is not None:
            mismatches = [
                f"{key}: declared={value!r}, observed={observed.get(key)!r}"
                for key, value in declared.items()
                if observed.get(key) != value
            ]
            if mismatches:
                raise ValueError(
                    "baseline pin declarations do not match referenced scenario: "
                    + "; ".join(mismatches)
                )
        if any(value is None for value in observed.values()):
            raise ValueError(
                f"referenced baseline scenario lacks typed pin facts: {scenario_path}"
            )
        return {
            "status": "verified_referenced_scenario",
            "values": observed,
            "scenario_json": str(scenario_path),
            "scenario_sha256": observed_sha,
        }
    if scope is BaselineScope.MECHANISM:
        raise ValueError(
            "same-regime mechanism exact baseline pins require a readable "
            "referenced scenario_json"
        )
    fallback = (
        dict(declared)
        if isinstance(declared, Mapping)
        else {
            "exact_trace_internal_dtype": "fp32",
            "edge_retention_policy_id": DEFAULT_RETENTION_POLICY_ID,
            "decoder_chunk_size": 4096,
        }
    )
    return {
        "status": "legacy_frozen_scientific_contract",
        "values": fallback,
        "scenario_json": None if scenario_path is None else str(scenario_path),
        "scenario_sha256": None,
    }


def _validate_exact_baseline_pins(
    *,
    case: Case,
    candidate_profile: str,
    baseline_entry: Mapping[str, Any],
    scope: BaselineScope,
) -> dict[str, Any]:
    candidate = _profile_variant(case, candidate_profile).baseline_pins.as_overrides()
    verification = _verify_baseline_pin_facts(
        baseline_entry=baseline_entry,
        scope=scope,
    )
    baseline = verification["values"]
    mismatches = [
        f"{key}: candidate={value!r}, baseline={baseline.get(key)!r}"
        for key, value in candidate.items()
        if baseline.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"candidate profile {candidate_profile!r} cannot run with "
            "--fidelity strict_exact/exact because compatibility-mixed or semantic "
            "baseline "
            "pins differ: " + "; ".join(mismatches)
        )
    return {
        **verification,
        "candidate_values": candidate,
    }


def _assert_generated_scenario_pins(
    scenario: Mapping[str, Any],
    pins: BaselinePins,
) -> None:
    expected = pins.as_overrides()
    mismatches = [
        f"{key}: expected={value!r}, generated={scenario.get(key)!r}"
        for key, value in expected.items()
        if scenario.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "generated candidate scenario violates typed baseline pins: "
            + "; ".join(mismatches)
        )


def _case_scenario(
    case: Case,
    fidelity: str,
    candidate_profile: str = "canonical",
    *,
    baseline_scope: BaselineScope | None = None,
    diagnostic_stop_mode: str = "none",
    diagnostic_stop_phase4_batches: int | None = None,
) -> dict[str, Any]:
    if diagnostic_stop_mode not in {
        "none",
        "phase0_probe",
        "phase3_probe",
        "transition_probe",
    }:
        raise ValueError("invalid diagnostic stop mode")
    if (
        diagnostic_stop_phase4_batches is not None
        and diagnostic_stop_phase4_batches <= 0
    ):
        raise ValueError("diagnostic stop phase4 batches must be positive")
    if (
        diagnostic_stop_mode != "transition_probe"
        and diagnostic_stop_phase4_batches is not None
    ):
        raise ValueError("diagnostic stop phase4 batches require transition_probe mode")
    payload = build_chpc_baseline_config(variant=case.variant, cluster="granite")
    scenario = next(
        row for row in payload["scenarios"] if row["fixture_name"] == case.fixture
    )
    scenario = dict(scenario)
    scenario.setdefault("edge_retention_policy_id", DEFAULT_RETENTION_POLICY_ID)
    scenario["name"] = f"perf_{case.variant}_{case.fixture}"
    scenario["stage"] = "exact_trace_performance_optimization"
    scenario["tier"] = "sweep"
    scenario["resource_profile"] = "performance_optimization_h200"
    scenario["fidelity_level"] = fidelity
    scenario["diagnostic_stop_mode"] = diagnostic_stop_mode
    scenario["diagnostic_stop_phase4_batches"] = diagnostic_stop_phase4_batches
    scenario.update(_candidate_overrides(case, candidate_profile))
    selected_variant = CANDIDATE_PROFILES[candidate_profile].variant_for(
        case.provider_capabilities()
    )
    if selected_variant is not None:
        _assert_generated_scenario_pins(scenario, selected_variant.baseline_pins)
    diagnostic = diagnostic_stop_mode != "none"
    scenario["baseline_check"] = {
        "enabled": not diagnostic,
        "mode": (
            "diagnostic"
            if diagnostic
            else ("gate" if FIDELITY_THRESHOLDS[fidelity] else "metrics")
        ),
        "registry_key": case.key,
        "baseline_required": not diagnostic,
        "scope": (baseline_scope or _baseline_scope(candidate_profile, fidelity)).value,
        "comparison_semantics": COMPARISON_SEMANTICS[fidelity],
        "claim_limitation": COMPARISON_CLAIM_LIMITATION,
        "thresholds": FIDELITY_THRESHOLDS[fidelity],
    }
    return {"defaults": payload["defaults"], "scenarios": [scenario]}


def _runner_command(
    *,
    scenario_file: Path,
    output_root: Path,
    run_id: str,
    run_goal: str = DEFAULT_RUN_GOAL,
    baseline_registry: Path = DEFAULT_BASELINE_REGISTRY,
) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--scenarios-file",
        str(scenario_file),
        "--output-root",
        str(output_root),
        "--baseline-registry",
        str(baseline_registry),
        "--fail-on-baseline-missing",
        "--fail-on-validation-fail",
        "--run-id",
        run_id,
        "--run-name",
        "Exact-trace performance optimization",
        "--run-goal",
        run_goal,
    ]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return float(ordered[index])


@dataclass(frozen=True)
class _GpuResourceSample:
    index: str | None
    uuid: str | None
    name: str | None
    sm_utilization: float
    memory_utilization: float
    power_watts: float
    framebuffer_used_mib: float
    framebuffer_total_mib: float

    @property
    def device_key(self) -> tuple[str, str]:
        if self.uuid:
            return ("uuid", self.uuid)
        if self.index:
            return ("index", self.index)
        return ("legacy", "unknown")


def _parse_gpu_resource_sample(line: str) -> _GpuResourceSample | None:
    fields = [field.strip() for field in next(csv.reader([line]))]
    if len(fields) == 9:
        index, uuid, name = fields[1:4]
        numeric_fields = fields[4:]
    elif len(fields) == 6:
        # Explicit compatibility for sample files emitted before device identity
        # was added. All such rows belong to one unknown device.
        index = uuid = name = None
        numeric_fields = fields[1:]
    else:
        return None
    try:
        sm, memory, power, framebuffer, total = (
            float(value) for value in numeric_fields
        )
    except ValueError:
        return None
    return _GpuResourceSample(
        index=index or None,
        uuid=uuid or None,
        name=name or None,
        sm_utilization=sm,
        memory_utilization=memory,
        power_watts=power,
        framebuffer_used_mib=framebuffer,
        framebuffer_total_mib=total,
    )


def _summarize_gpu_device(samples: Sequence[_GpuResourceSample]) -> dict[str, Any]:
    first = samples[0]
    sm = [sample.sm_utilization for sample in samples]
    memory = [sample.memory_utilization for sample in samples]
    power = [sample.power_watts for sample in samples]
    framebuffer = [sample.framebuffer_used_mib for sample in samples]
    total = [sample.framebuffer_total_mib for sample in samples]
    peak_total = max(total)
    return {
        "gpu_index": int(first.index)
        if first.index is not None and first.index.isdigit()
        else first.index,
        "gpu_uuid": first.uuid,
        "gpu_name": first.name,
        "sample_count": len(samples),
        "gpu_sm_utilization_mean_percent": sum(sm) / len(sm),
        "gpu_sm_utilization_p95_percent": _percentile(sm, 0.95),
        "gpu_sm_utilization_max_percent": max(sm),
        "gpu_memory_utilization_mean_percent": sum(memory) / len(memory),
        "gpu_memory_utilization_p95_percent": _percentile(memory, 0.95),
        "gpu_memory_utilization_max_percent": max(memory),
        "gpu_power_mean_watts": sum(power) / len(power),
        "gpu_power_max_watts": max(power),
        "gpu_framebuffer_peak_mib": max(framebuffer),
        "gpu_framebuffer_total_mib": peak_total,
        "gpu_framebuffer_peak_fraction": max(framebuffer) / peak_total
        if peak_total > 0
        else None,
    }


def _gpu_resource_summary(samples_path: Path) -> dict[str, Any]:
    if not samples_path.is_file():
        return {"gpu_sample_count": 0}
    samples: list[_GpuResourceSample] = []
    for line in samples_path.read_text(encoding="utf-8", errors="replace").splitlines():
        sample = _parse_gpu_resource_sample(line)
        if sample is not None:
            samples.append(sample)
    if not samples:
        return {"gpu_sample_count": 0}

    by_device: dict[tuple[str, str], list[_GpuResourceSample]] = {}
    for sample in samples:
        by_device.setdefault(sample.device_key, []).append(sample)
    devices = [
        _summarize_gpu_device(device_samples)
        for _key, device_samples in sorted(by_device.items())
    ]
    primary = max(
        devices,
        key=lambda device: (
            float(device["gpu_framebuffer_peak_mib"]),
            float(device["gpu_sm_utilization_mean_percent"]),
        ),
    )
    compatibility_metrics = {
        key: value
        for key, value in primary.items()
        if key.startswith("gpu_") and key not in {"gpu_index", "gpu_uuid", "gpu_name"}
    }
    return {
        "gpu_sample_count": len(samples),
        "gpu_device_count": len(devices),
        "gpu_devices": devices,
        "gpu_summary_scope": "single_device" if len(devices) == 1 else "busiest_device",
        "gpu_summary_device_index": primary["gpu_index"],
        "gpu_summary_device_uuid": primary["gpu_uuid"],
        "gpu_summary_device_name": primary["gpu_name"],
        **compatibility_metrics,
    }


class HostMemoryGuardTriggered(RuntimeError):
    pass


def _slurm_job_memory_cgroup_v1(
    *,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup/memory"),
) -> Path:
    for line in proc_cgroup_path.read_text(encoding="utf-8").splitlines():
        fields = line.split(":", maxsplit=2)
        if len(fields) != 3 or "memory" not in fields[1].split(","):
            continue
        relative_parts = Path(fields[2]).parts
        job_index = next(
            (
                index
                for index, part in enumerate(relative_parts)
                if part.startswith("job_")
            ),
            None,
        )
        if job_index is None:
            break
        job_relative = Path(*relative_parts[: job_index + 1])
        path = cgroup_root / job_relative.relative_to("/")
        required = (
            path / "memory.stat",
            path / "memory.usage_in_bytes",
            path / "memory.limit_in_bytes",
            path / "memory.failcnt",
        )
        if all(item.is_file() for item in required):
            return path
        break
    raise RuntimeError(
        "Slurm cgroup-v1 job memory counters are unavailable; "
        "refusing to run with --host-memory-stop-gib"
    )


def _read_host_memory_sample(cgroup_path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    for line in (cgroup_path / "memory.stat").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in {
            "total_rss",
            "total_cache",
            "total_unevictable",
        }:
            stats[fields[0]] = int(fields[1])
    missing = {
        "total_rss",
        "total_cache",
        "total_unevictable",
    } - stats.keys()
    if missing:
        raise RuntimeError(
            "Slurm cgroup-v1 memory.stat lacks required counters: "
            + ", ".join(sorted(missing))
        )
    return {
        "monotonic_seconds": time.monotonic(),
        "usage_bytes": int(
            (cgroup_path / "memory.usage_in_bytes").read_text(encoding="utf-8")
        ),
        "limit_bytes": int(
            (cgroup_path / "memory.limit_in_bytes").read_text(encoding="utf-8")
        ),
        "failcnt": int((cgroup_path / "memory.failcnt").read_text(encoding="utf-8")),
        **stats,
        "breakdown_total_bytes": (
            stats["total_rss"] + stats["total_cache"] + stats["total_unevictable"]
        ),
    }


def _host_memory_summary(
    samples: Sequence[dict[str, int]],
    *,
    cgroup_path: Path | None,
    stop_bytes: int | None,
    rss_stop_bytes: int | None,
    triggered: bool,
    trigger: str | None,
) -> dict[str, Any]:
    return {
        "host_memory_guard_enabled": (
            stop_bytes is not None or rss_stop_bytes is not None
        ),
        "host_memory_guard_triggered": triggered,
        "host_memory_guard_stop_bytes": stop_bytes,
        "host_memory_rss_guard_stop_bytes": rss_stop_bytes,
        "host_memory_guard_trigger": trigger,
        "host_memory_cgroup_path": str(cgroup_path)
        if cgroup_path is not None
        else None,
        "host_memory_sample_count": len(samples),
        "host_memory_peak_usage_bytes": max(
            (sample["usage_bytes"] for sample in samples), default=None
        ),
        "host_memory_peak_rss_bytes": max(
            (sample["total_rss"] for sample in samples), default=None
        ),
        "host_memory_peak_cache_bytes": max(
            (sample["total_cache"] for sample in samples), default=None
        ),
        "host_memory_peak_unevictable_bytes": max(
            (sample["total_unevictable"] for sample in samples), default=None
        ),
        "host_memory_peak_breakdown_total_bytes": max(
            (sample["breakdown_total_bytes"] for sample in samples), default=None
        ),
        "host_memory_limit_bytes": (samples[-1]["limit_bytes"] if samples else None),
        "host_memory_failcnt_initial": samples[0]["failcnt"] if samples else None,
        "host_memory_failcnt_final": samples[-1]["failcnt"] if samples else None,
    }


def _stop_process(
    process: subprocess.Popen[Any] | None,
    *,
    process_group: bool = False,
) -> None:
    if process is None:
        return
    if process_group:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    else:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if process_group:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()
    else:
        if process_group:
            # The wrapper can exit before its descendants. Ensure nothing remains
            # in the isolated runner group after the graceful group signal.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _stream_runner(
    command: Sequence[str],
    *,
    output_root: Path,
    host_memory_stop_gib: float | None = None,
    host_rss_stop_gib: float | None = None,
) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    output_root.mkdir(parents=True, exist_ok=True)
    samples_path = output_root / "gpu_samples.csv"
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    samples_handle = samples_path.open("w", encoding="utf-8")
    sampler: subprocess.Popen[Any] | None = None
    process: subprocess.Popen[Any] | None = None
    runner_completed = False
    sampler_was_running = False
    offsets: dict[Path, int] = {}
    host_memory_stop_bytes = (
        int(host_memory_stop_gib * 1024**3)
        if host_memory_stop_gib is not None
        else None
    )
    host_rss_stop_bytes = (
        int(host_rss_stop_gib * 1024**3) if host_rss_stop_gib is not None else None
    )
    host_memory_cgroup = (
        _slurm_job_memory_cgroup_v1()
        if host_memory_stop_bytes is not None or host_rss_stop_bytes is not None
        else None
    )
    host_memory_samples: list[dict[str, int]] = []
    host_memory_guard_triggered = False
    host_memory_guard_trigger: str | None = None

    def guard_trigger(sample: dict[str, int]) -> str | None:
        if (
            host_memory_stop_bytes is not None
            and sample["usage_bytes"] >= host_memory_stop_bytes
        ):
            return "cgroup_usage"
        if (
            host_rss_stop_bytes is not None
            and sample["total_rss"] >= host_rss_stop_bytes
        ):
            return "cgroup_total_rss"
        return None

    if host_memory_cgroup is not None:
        initial_sample = _read_host_memory_sample(host_memory_cgroup)
        host_memory_samples.append(initial_sample)
        if (
            host_memory_stop_bytes is not None
            and host_memory_stop_bytes >= initial_sample["limit_bytes"]
        ):
            raise ValueError(
                "--host-memory-stop-gib must be below the Slurm cgroup memory "
                f"limit ({initial_sample['limit_bytes'] / 1024**3:.2f} GiB)"
            )
        host_memory_guard_trigger = guard_trigger(initial_sample)
        if host_memory_guard_trigger is not None:
            host_memory_guard_triggered = True
            raise HostMemoryGuardTriggered(
                "host memory guard refused to start the runner: current "
                f"{host_memory_guard_trigger} is already at its stop threshold"
            )
    try:
        sampler = subprocess.Popen(
            GPU_SAMPLE_COMMAND,
            cwd=REPO_ROOT,
            env=env,
            stdout=samples_handle,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            start_new_session=True,
        )
        while process.poll() is None:
            if host_memory_cgroup is not None:
                sample = _read_host_memory_sample(host_memory_cgroup)
                host_memory_samples.append(sample)
                host_memory_guard_trigger = guard_trigger(sample)
                if host_memory_guard_trigger is not None:
                    host_memory_guard_triggered = True
                    _stop_process(process, process_group=True)
                    break
            _tail_logs(output_root, offsets)
            time.sleep(1)
        _tail_logs(output_root, offsets)
        runner_completed = True
    finally:
        if not runner_completed:
            _stop_process(process, process_group=True)
        sampler_was_running = sampler is not None and sampler.poll() is None
        _stop_process(sampler)
        samples_handle.close()
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall_seconds = time.perf_counter() - start
        cpu_seconds = (
            usage_after.ru_utime
            + usage_after.ru_stime
            - usage_before.ru_utime
            - usage_before.ru_stime
        )
        allocated_cpus = int(
            env.get("SLURM_CPUS_PER_TASK") or env.get("SLURM_CPUS_ON_NODE") or 1
        )
        gpu_summary = _gpu_resource_summary(samples_path)
        gpu_sample_count = int(gpu_summary["gpu_sample_count"])
        if sampler is None:
            sampling_status = "failed_to_start"
            sampling_failure_reason = "GPU sampler did not start"
        elif not sampler_was_running:
            sampling_status = "exited_early"
            sampling_failure_reason = (
                "GPU sampler exited before harness cleanup; "
                f"returncode={sampler.returncode}"
            )
        elif gpu_sample_count == 0:
            sampling_status = "no_samples"
            sampling_failure_reason = "GPU sampler produced no usable samples"
        else:
            sampling_status = "ok"
            sampling_failure_reason = None
        resource_summary = {
            **gpu_summary,
            **_host_memory_summary(
                host_memory_samples,
                cgroup_path=host_memory_cgroup,
                stop_bytes=host_memory_stop_bytes,
                rss_stop_bytes=host_rss_stop_bytes,
                triggered=host_memory_guard_triggered,
                trigger=host_memory_guard_trigger,
            ),
            "sample_file": str(samples_path),
            "gpu_sampling_status": sampling_status,
            "gpu_sampling_failure_reason": sampling_failure_reason,
            "gpu_sampler_returncode": (
                sampler.returncode if sampler is not None else None
            ),
            "resource_validation_passed": sampling_status == "ok",
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "allocated_cpus": allocated_cpus,
            "cpu_one_core_utilization_percent": (
                100.0 * cpu_seconds / wall_seconds if wall_seconds > 0 else None
            ),
            "cpu_allocated_utilization_percent": (
                100.0 * cpu_seconds / wall_seconds / allocated_cpus
                if wall_seconds > 0 and allocated_cpus > 0
                else None
            ),
        }
        write_json(output_root / "host_memory_samples.json", host_memory_samples)
        write_json(output_root / "resource_summary.json", resource_summary)
    if process is None:
        raise RuntimeError("Trace runner did not start")
    if host_memory_guard_triggered:
        trigger_value = (
            host_memory_samples[-1]["usage_bytes"]
            if host_memory_guard_trigger == "cgroup_usage"
            else host_memory_samples[-1]["total_rss"]
        )
        stop_gib = (
            host_memory_stop_gib
            if host_memory_guard_trigger == "cgroup_usage"
            else host_rss_stop_gib
        )
        raise HostMemoryGuardTriggered(
            "host memory guard terminated the runner process group at "
            f"{trigger_value / 1024**3:.2f} GiB "
            f"(stop threshold {stop_gib:.2f} GiB; "
            f"trigger={host_memory_guard_trigger})"
        )
    return int(process.returncode or 0)


def _tail_logs(output_root: Path, offsets: dict[Path, int]) -> None:
    for path in sorted(output_root.glob("*/run.log")):
        offset = offsets.get(path, 0)
        with path.open(encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            for line in handle:
                print(f"[{path.parent.name}] {line}", end="", flush=True)
            offsets[path] = handle.tell()


def _baseline_entries(
    registry_path: Path = DEFAULT_BASELINE_REGISTRY,
) -> dict[str, dict[str, Any]]:
    expected_scope = (
        BaselineScope.MECHANISM
        if registry_path == DEFAULT_MECHANISM_BASELINE_REGISTRY
        else BaselineScope.SCIENTIFIC
    )
    contract = _load_baseline_registry_contract(
        registry_path,
        expected_scope=expected_scope,
    )
    return {key: dict(entry.payload) for key, entry in contract.entries.items()}


def _active_row_mechanism_gate(
    requested: bool,
    diagnostics: Any,
    max_bytes: int,
    *,
    phase0_ranges_requested: bool = False,
    expected_bytes: int | None = ACTIVE_ROW_EXPECTED_BYTES,
) -> tuple[bool | None, list[str]]:
    def is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    if not requested:
        if phase0_ranges_requested:
            return (
                False,
                [
                    "phase0_decoder_row_ranges requires "
                    "decoder_active_row_residency=true"
                ],
            )
        return None, []
    if not is_int(max_bytes) or max_bytes < 0:
        return (
            False,
            ["decoder_active_row_max_bytes must be a non-negative integer"],
        )
    if not isinstance(diagnostics, dict):
        return False, ["active-row diagnostics missing from result artifact summary"]

    reasons: list[str] = []
    resident = diagnostics.get("resident")
    build = diagnostics.get("build")
    phase4 = diagnostics.get("phase4")
    if diagnostics.get("requested") is not True:
        reasons.append("active-row runtime did not record requested=true")
    if diagnostics.get("effective") is not True:
        reasons.append("active-row residency was not effective")
    fallback = diagnostics.get("fallback_reason")
    if fallback is not None:
        reasons.append(f"active-row fallback was recorded: {fallback}")
    if not all(isinstance(value, dict) for value in (resident, build, phase4)):
        reasons.append("active-row resident/build/phase4 diagnostics are incomplete")
        return False, reasons

    assert isinstance(resident, dict)
    assert isinstance(build, dict)
    assert isinstance(phase4, dict)
    seed = diagnostics.get("seed")
    if not isinstance(seed, dict):
        reasons.append("active-row fused-seed diagnostics are incomplete")
        return False, reasons
    range_diagnostics = (
        diagnostics.get("phase0_decoder_row_ranges")
        if phase0_ranges_requested
        else None
    )
    resident_bytes = resident.get("bytes")
    estimated_bytes = resident.get("estimated_bytes")
    if not is_int(resident_bytes) or resident_bytes <= 0:
        reasons.append("active-row resident bytes must be positive")
    if not is_int(estimated_bytes) or estimated_bytes <= 0:
        reasons.append("active-row estimated bytes must be positive")
    if (
        is_int(resident_bytes)
        and is_int(estimated_bytes)
        and resident_bytes != estimated_bytes
    ):
        reasons.append("active-row resident bytes differ from admitted estimate")
    recorded_max_bytes = diagnostics.get("max_bytes_requested")
    if recorded_max_bytes != max_bytes:
        reasons.append("active-row recorded byte ceiling differs from configuration")
    if is_int(resident_bytes):
        if max_bytes > 0 and resident_bytes > max_bytes:
            reasons.append("active-row resident bytes exceed the configured byte cap")
        if max_bytes == 0:
            effective_budget = diagnostics.get("effective_budget_bytes")
            safety_margin = diagnostics.get("safety_margin_bytes")
            hbm = diagnostics.get("hbm")
            if diagnostics.get("admission_policy") != "live_hbm_headroom":
                reasons.append("dynamic active-row admission policy was not recorded")
            if not is_int(safety_margin) or safety_margin <= 0:
                reasons.append("dynamic active-row safety margin must be positive")
            if not is_int(effective_budget) or effective_budget <= 0:
                reasons.append("dynamic active-row effective budget must be positive")
            elif resident_bytes > effective_budget:
                reasons.append("active-row resident bytes exceed the dynamic budget")
            if not isinstance(hbm, dict):
                reasons.append("dynamic active-row HBM observation is missing")
            elif not all(
                is_int(hbm.get(key)) and hbm.get(key) > 0
                for key in ("free_bytes", "total_bytes")
            ):
                reasons.append("dynamic active-row HBM headroom is invalid")
    if (
        expected_bytes is not None
        and is_int(resident_bytes)
        and not (expected_bytes / 10 <= resident_bytes <= expected_bytes * 10)
    ):
        reasons.append(
            "active-row resident bytes are outside the predicted order of magnitude"
        )
    if not is_int(resident.get("row_count")) or resident["row_count"] <= 0:
        reasons.append("active-row resident row count must be positive")
    if not is_int(resident.get("owner_count")) or resident.get("owner_count") != 1:
        reasons.append("active-row owner count must equal 1")
    if not is_int(build.get("count")) or build.get("count") != 1:
        reasons.append("active-row build count must equal 1")
    if build.get("source") != "phase0_fused_seed":
        reasons.append("active-row build source must equal phase0_fused_seed")
    traversal = build.get("traversal_bytes")
    loaded = build.get("decoder_load_bytes")
    if not is_int(traversal) or traversal != 0 or not is_int(loaded) or loaded != 0:
        reasons.append(
            "fused active-row materialization must add zero traversal/load bytes"
        )
    if (
        not is_int(build.get("decoder_page_load_count"))
        or build["decoder_page_load_count"] != 0
    ):
        reasons.append(
            "fused active-row materialization must add zero decoder page loads"
        )
    shared_traversal = seed.get("shared_traversal_bytes")
    shared_loaded = seed.get("shared_decoder_load_bytes")
    effective_phase0_ranges = (
        isinstance(range_diagnostics, dict)
        and range_diagnostics.get("effective") is True
    )
    if effective_phase0_ranges:
        logical_materialized_bytes = range_diagnostics.get("logical_materialized_bytes")
        if (
            not is_int(shared_traversal)
            or shared_traversal <= 0
            or shared_traversal != logical_materialized_bytes
        ):
            reasons.append(
                "fused active-row range seed traversal bytes must equal "
                "Phase0 logical materialized bytes"
            )
        if (
            not is_int(shared_loaded)
            or shared_loaded != 0
            or not is_int(seed.get("shared_decoder_page_load_count"))
            or seed.get("shared_decoder_page_load_count") != 0
        ):
            reasons.append(
                "effective Phase0 ranges must record zero legacy decoder "
                "page loads and load bytes"
            )
    else:
        if (
            not is_int(shared_traversal)
            or shared_traversal <= 0
            or not is_int(shared_loaded)
            or shared_loaded <= 0
            or shared_traversal != shared_loaded
        ):
            reasons.append(
                "fused active-row seed shared traversal/load bytes are "
                "missing or inconsistent"
            )
        if (
            not is_int(seed.get("shared_decoder_page_load_count"))
            or seed["shared_decoder_page_load_count"] <= 0
        ):
            reasons.append(
                "fused active-row seed shared decoder page-load count must be positive"
            )
    seed_bytes = seed.get("bytes")
    if (
        not is_int(seed_bytes)
        or seed_bytes <= 0
        or not is_int(resident_bytes)
        or seed_bytes > resident_bytes
    ):
        reasons.append(
            "active-row seed bytes must be positive and no larger than resident bytes"
        )
    unique_row_count = seed.get("unique_row_count")
    resident_row_count = resident.get("row_count")
    if (
        not is_int(unique_row_count)
        or unique_row_count <= 0
        or not is_int(resident_row_count)
        or unique_row_count > resident_row_count
    ):
        reasons.append(
            "active-row seed unique-row count must be positive and no larger than resident rows"
        )
    if (
        not is_int(seed.get("materialization_h2d_bytes"))
        or seed.get("materialization_h2d_bytes") != resident_bytes
    ):
        reasons.append("active-row seed H2D bytes must equal resident bytes")
    if not is_int(seed.get("missing_keys")) or seed.get("missing_keys") != 0:
        reasons.append("active-row seed must cover every final decoder-row key")
    if not isinstance(resident.get("device"), str) or not resident["device"].startswith(
        "cuda"
    ):
        reasons.append("active-row residency device must be CUDA")
    if (
        not is_int(phase4.get("decoder_page_load_count_delta"))
        or phase4.get("decoder_page_load_count_delta") != 0
        or not is_int(phase4.get("decoder_load_bytes_delta"))
        or phase4.get("decoder_load_bytes_delta") != 0
    ):
        reasons.append(
            "Phase4 decoder page-load and load-byte deltas must both equal zero"
        )
    if phase0_ranges_requested:
        if not isinstance(range_diagnostics, dict):
            reasons.append(
                "Phase0 decoder-row-range diagnostics missing from active-row evidence"
            )
            return False, reasons
        if range_diagnostics.get("requested") is not True:
            reasons.append(
                "Phase0 decoder-row-range runtime did not record requested=true"
            )
        range_effective = range_diagnostics.get("effective") is True
        fallback_reason = range_diagnostics.get("fallback_reason")
        if not range_effective:
            reasons.append(
                "Phase0 decoder-row ranges fell back to exact full-page reads"
                + (
                    f": {fallback_reason}"
                    if isinstance(fallback_reason, str) and fallback_reason
                    else " without a structured fallback reason"
                )
            )
            return False, reasons
        if fallback_reason is not None:
            reasons.append(
                f"effective Phase0 decoder-row ranges recorded fallback: {fallback_reason}"
            )
        range_unique_rows = range_diagnostics.get("unique_row_count")
        backend = range_diagnostics.get("backend")
        baseline_page_bytes = range_diagnostics.get("baseline_full_page_bytes")
        if backend == "mapped_safetensors":
            backend_requests = range_diagnostics.get("backend_request_count")
            mapping_count = range_diagnostics.get("mapping_count")
            block_count = range_diagnostics.get("block_count")
            explicit_read_count = range_diagnostics.get("read_count")
            backend_materialized_bytes = range_diagnostics.get(
                "backend_materialized_bytes"
            )
            if not is_int(backend_requests) or backend_requests <= 0:
                reasons.append(
                    "mapped safetensors backend request count must be positive"
                )
            if not is_int(mapping_count) or mapping_count <= 0:
                reasons.append("mapped safetensors mapping count must be positive")
            if not is_int(block_count) or block_count <= 0:
                reasons.append("mapped safetensors block count must be positive")
            if not is_int(explicit_read_count) or explicit_read_count != 0:
                reasons.append(
                    "mapped safetensors must use mapped gathers without explicit reads"
                )
            if (
                not is_int(backend_materialized_bytes)
                or backend_materialized_bytes <= 0
                or not is_int(baseline_page_bytes)
                or baseline_page_bytes <= 0
                or backend_materialized_bytes >= baseline_page_bytes
            ):
                reasons.append(
                    "mapped safetensors backend materialized bytes must be positive "
                    "and below baseline full-page bytes"
                )
        elif backend == "coalesced_ranges":
            range_count = range_diagnostics.get("range_request_count")
            if (
                not is_int(range_count)
                or range_count <= 0
                or not is_int(range_unique_rows)
                or range_unique_rows <= 0
                or range_count * 2 > range_unique_rows
            ):
                reasons.append(
                    "Phase0 decoder range count must be positive and at most half "
                    "the unique-row count"
                )
            logical_bytes = range_diagnostics.get("logical_materialized_bytes")
            if (
                not is_int(logical_bytes)
                or logical_bytes <= 0
                or not is_int(baseline_page_bytes)
                or baseline_page_bytes <= 0
                or logical_bytes >= baseline_page_bytes
            ):
                reasons.append(
                    "Phase0 logical materialized bytes must be positive and below "
                    "baseline full-page bytes"
                )
        else:
            reasons.append("Phase0 decoder-row ranges must record a recognized backend")
    return not reasons, reasons


def _active_row_framebuffer_gate(
    case: Case,
    requested: bool,
    resource_summary: dict[str, Any],
) -> tuple[bool | None, dict[str, Any], list[str]]:
    if not requested:
        return None, {"status": "not_applicable"}, []
    capabilities = case.provider_capabilities()
    large_model = (
        capabilities.decoder_output_topology == "same_layer"
        and capabilities.layer_count is not None
        and capabilities.layer_count > 26
    )
    if large_model:
        candidate = resource_summary.get("gpu_framebuffer_peak_mib")
        total = resource_summary.get("gpu_framebuffer_total_mib")
        comparison = {
            "status": "unavailable",
            "candidate_peak_mib": candidate,
            "sampled_total_mib": total,
            "peak_fraction_limit": LARGE_MODEL_HBM_PEAK_FRACTION_LIMIT,
        }
        if (
            not isinstance(candidate, (int, float))
            or not isinstance(total, (int, float))
            or float(total) <= 0
        ):
            return (
                False,
                comparison,
                [
                    "large-model active-row HBM comparison unavailable: "
                    "sampled peak or total is missing"
                ],
            )
        peak_fraction = float(candidate) / float(total)
        comparison["candidate_peak_fraction"] = peak_fraction
        passed = peak_fraction <= LARGE_MODEL_HBM_PEAK_FRACTION_LIMIT
        comparison["status"] = "passed" if passed else "exceeded_fraction_limit"
        return (
            passed,
            comparison,
            []
            if passed
            else [
                f"candidate peak framebuffer fraction {peak_fraction:.4f} exceeds "
                f"{LARGE_MODEL_HBM_PEAK_FRACTION_LIMIT:.2f} of sampled H200 total"
            ],
        )

    reference = dict(ACTIVE_ROW_FRAMEBUFFER_REFERENCE)
    candidate = resource_summary.get("gpu_framebuffer_peak_mib")
    reference_peak_mib = reference["gpu_framebuffer_peak_mib"]
    limit = reference_peak_mib + ACTIVE_ROW_EXPECTED_MIB_ALLOWANCE
    comparison = {
        "status": "unavailable",
        "candidate_peak_mib": candidate,
        "reference_peak_mib": reference_peak_mib,
        "expected_resident_bytes": ACTIVE_ROW_EXPECTED_BYTES,
        "allowance_mib": ACTIVE_ROW_EXPECTED_MIB_ALLOWANCE,
        "allowance_rounding": "ceil_bytes_to_mib",
        "limit_mib": limit,
        "reference": reference,
    }
    if not isinstance(candidate, (int, float)):
        return (
            False,
            comparison,
            ["active-row framebuffer comparison unavailable: candidate peak missing"],
        )
    passed = float(candidate) <= float(limit)
    comparison["status"] = "passed" if passed else "exceeded_reference"
    reasons = (
        []
        if passed
        else [
            f"candidate peak framebuffer {float(candidate):.1f} MiB exceeds "
            f"audited limit {float(limit):.1f} MiB "
            f"(reference {float(reference_peak_mib):.1f} MiB + "
            f"{ACTIVE_ROW_EXPECTED_MIB_ALLOWANCE} MiB resident-row allowance)"
        ]
    )
    return passed, comparison, reasons


def _result_report(
    case: Case,
    *,
    candidate_root: Path,
    baseline_entry: dict[str, Any],
    runner_returncode: int = 0,
) -> dict[str, Any]:
    scenario_root = candidate_root / f"perf_{case.variant}_{case.fixture}"
    result = read_json(scenario_root / "result.json")
    scenario_path = scenario_root / "scenario.json"
    scenario = read_json(scenario_path) if scenario_path.is_file() else {}
    comparison_semantics = (scenario.get("baseline_check") or {}).get(
        "comparison_semantics"
    )
    fidelity_level = str(
        scenario.get("fidelity_level")
        or (
            "strict_exact"
            if comparison_semantics in {None, COMPARISON_SEMANTICS["strict_exact"]}
            else "research"
        )
    )
    code_retention_allowed = fidelity_level in CODE_RETENTION_ALLOWED_FIDELITIES
    automatic_promotion_allowed = (
        fidelity_level in AUTOMATIC_PROMOTION_ALLOWED_FIDELITIES
    )
    active_rows_requested = scenario.get("decoder_active_row_residency") is True
    phase0_ranges_requested = scenario.get("phase0_decoder_row_ranges") is True
    active_rows_max_bytes_raw = scenario.get("decoder_active_row_max_bytes", 0)
    active_rows_max_bytes = (
        int(active_rows_max_bytes_raw)
        if isinstance(active_rows_max_bytes_raw, int)
        else 0
    )
    comparison_path = scenario_root / "baseline_compare.json"
    comparison = read_json(comparison_path) if comparison_path.is_file() else {}
    candidate_duration_raw = result.get("duration_seconds")
    candidate_duration = (
        float(candidate_duration_raw)
        if isinstance(candidate_duration_raw, (int, float))
        else None
    )
    diagnostic_stop_mode = str(scenario.get("diagnostic_stop_mode", "none"))
    if diagnostic_stop_mode != "none":
        return {
            "case": case.key,
            "status": result.get("status"),
            "runner_returncode": runner_returncode,
            "diagnostic_stop_mode": diagnostic_stop_mode,
            "diagnostic_stop_phase4_batches": scenario.get(
                "diagnostic_stop_phase4_batches"
            ),
            "diagnostic_report": result.get("artifact_summary") or {},
            "profiling_summary": result.get("profiling_summary") or {},
            "candidate_duration_seconds": candidate_duration,
            "baseline_duration_seconds": None,
            "speedup": None,
            "parity_passed": None,
            "performance_passed": None,
            "resource_gate_passed": None,
            "mechanism_validation_passed": None,
            "promotion_eligible": False,
            "fidelity_level": fidelity_level,
            "code_retention_allowed": code_retention_allowed,
            "code_retention_eligible": False,
            "code_retention_selected": False,
            "code_retention_decision": "not_eligible",
            "automatic_promotion_allowed": automatic_promotion_allowed,
            "automatic_promotion_eligible": False,
            "automatic_promotion_selected": False,
            "reconciliation_required": False,
            "scientific_acceptance_attempted": False,
            "acceptance_status": "diagnostic_not_scientific",
            "passed": False,
        }
    baseline_duration = float(baseline_entry["duration_seconds"])
    metrics = {
        label: comparison.get(metric_key) for label, metric_key in METRIC_KEYS.items()
    }
    parity_passed = result.get("baseline_check", {}).get("passed") is True
    parity_failure_reasons = result.get("baseline_check", {}).get("failure_reasons", [])
    performance_failure_reasons: list[str] = []
    reconciliation_required = False
    capabilities = case.provider_capabilities()
    large_model = (
        capabilities.decoder_output_topology == "same_layer"
        and capabilities.layer_count is not None
        and capabilities.layer_count > 26
    )
    if active_rows_requested and not large_model:
        performance_target = ACTIVE_ROW_FUSED_TARGET_SECONDS
        stretch_target = None
        performance_requirement = "active_row_1b_fused_target"
        performance_passed = bool(
            candidate_duration is not None and candidate_duration <= performance_target
        )
        stretch_passed = None
        reconciliation_required = not performance_passed
        if candidate_duration is None:
            performance_failure_reasons.append(
                "active-row duration is missing; the fused engineering target cannot be evaluated"
            )
        elif not performance_passed:
            performance_failure_reasons.append(
                f"active-row duration {candidate_duration:.2f}s exceeds the reconciled "
                f"fused engineering target {performance_target:.2f}s; "
                "reconciliation is required"
            )
    else:
        performance_target = PERFORMANCE_TARGET_SECONDS.get(case.key)
        stretch_target = PERFORMANCE_STRETCH_TARGET_SECONDS.get(case.key)
        performance_requirement = (
            "fixed_target" if performance_target is not None else "none"
        )
        performance_passed = (
            candidate_duration <= performance_target
            if performance_target is not None and candidate_duration is not None
            else None
        )
        if performance_target is not None and candidate_duration is None:
            performance_passed = False
        stretch_passed = (
            candidate_duration <= stretch_target
            if stretch_target is not None and candidate_duration is not None
            else None
        )
        if stretch_target is not None and candidate_duration is None:
            stretch_passed = False
        if performance_passed is False:
            if candidate_duration is None:
                performance_failure_reasons.append(
                    f"performance target {performance_target:.2f}s could not be "
                    "evaluated because candidate duration is missing"
                )
            else:
                performance_failure_reasons.append(
                    f"candidate duration {candidate_duration:.2f}s exceeds "
                    f"performance target {performance_target:.2f}s"
                )
    resource_summary = (
        read_json(candidate_root / "resource_summary.json")
        if (candidate_root / "resource_summary.json").is_file()
        else {}
    )
    resource_validation_passed = (
        resource_summary.get("resource_validation_passed") is True
    )
    resource_failure_reasons = (
        []
        if resource_validation_passed
        else [
            str(
                resource_summary.get("gpu_sampling_failure_reason")
                or "GPU resource sampling was not validated"
            )
        ]
    )
    diagnostics = (result.get("artifact_summary") or {}).get(
        "decoder_active_row_residency"
    )
    mechanism_passed, mechanism_failure_reasons = _active_row_mechanism_gate(
        active_rows_requested,
        diagnostics,
        active_rows_max_bytes,
        phase0_ranges_requested=phase0_ranges_requested,
        expected_bytes=None if large_model else ACTIVE_ROW_EXPECTED_BYTES,
    )
    framebuffer_passed, framebuffer_comparison, framebuffer_failure_reasons = (
        _active_row_framebuffer_gate(case, active_rows_requested, resource_summary)
    )
    resource_failure_reasons.extend(framebuffer_failure_reasons)
    resource_gate_passed = resource_validation_passed and (
        framebuffer_passed is not False
    )
    promotion_eligible = case.key not in MEASUREMENT_ONLY_PERFORMANCE_CASES
    acceptance_failure_reasons = (
        []
        if promotion_eligible
        else [
            "measurement-only case has no fitted same-envelope control or "
            "reviewed hard runtime target; optimization acceptance is ineligible"
        ]
    )
    passed = (
        parity_passed
        and performance_passed is not False
        and promotion_eligible
        and resource_gate_passed
        and mechanism_passed is not False
        and not reconciliation_required
        and runner_returncode == 0
    )
    code_retention_eligible = bool(code_retention_allowed and parity_passed)
    automatic_promotion_eligible = bool(automatic_promotion_allowed and passed)
    return {
        "case": case.key,
        "status": result.get("status"),
        "runner_returncode": runner_returncode,
        "candidate_duration_seconds": candidate_duration,
        "baseline_duration_seconds": baseline_duration,
        "speedup": (
            baseline_duration / candidate_duration
            if candidate_duration is not None and candidate_duration > 0
            else None
        ),
        **metrics,
        "worst_step_evidence": comparison.get("worst_step_evidence", {}),
        "profiling_summary": result.get("profiling_summary") or {},
        "resource_summary": resource_summary,
        "comparison_semantics": comparison_semantics
        or COMPARISON_SEMANTICS["strict_exact"],
        "claim_limitation": COMPARISON_CLAIM_LIMITATION,
        "exact_semantics_claim_allowed": False,
        "fidelity_level": fidelity_level,
        "code_retention_allowed": code_retention_allowed,
        "code_retention_eligible": code_retention_eligible,
        "code_retention_selected": False,
        "code_retention_decision": (
            "eligible_pending_comparative_selection"
            if code_retention_eligible
            else "not_eligible"
        ),
        "automatic_promotion_allowed": automatic_promotion_allowed,
        "automatic_promotion_eligible": automatic_promotion_eligible,
        "automatic_promotion_selected": False,
        "automatic_promotion_decision": (
            "eligible_pending_comparative_selection"
            if automatic_promotion_eligible
            else "not_eligible"
        ),
        "performance_target_seconds": performance_target,
        "performance_requirement": performance_requirement,
        "performance_stretch_target_seconds": stretch_target,
        "active_row_initial_predicted_duration_seconds": (
            list(ACTIVE_ROW_INITIAL_PREDICTED_DURATION_SECONDS)
            if active_rows_requested and not large_model
            else None
        ),
        "active_row_fused_target_seconds": (
            ACTIVE_ROW_FUSED_TARGET_SECONDS
            if active_rows_requested and not large_model
            else None
        ),
        "reconciliation_required": reconciliation_required,
        "decoder_active_row_residency": diagnostics,
        "phase0_decoder_row_ranges_requested": phase0_ranges_requested,
        "phase0_decoder_row_ranges": (
            diagnostics.get("phase0_decoder_row_ranges")
            if isinstance(diagnostics, dict)
            else None
        ),
        "mechanism_validation_passed": mechanism_passed,
        "mechanism_validation_status": "not_applicable"
        if mechanism_passed is None
        else ("passed" if mechanism_passed else "failed"),
        "framebuffer_comparison": framebuffer_comparison,
        "framebuffer_passed": framebuffer_passed,
        "parity_passed": parity_passed,
        "performance_passed": performance_passed,
        "performance_stretch_passed": stretch_passed,
        "promotion_eligible": promotion_eligible,
        "acceptance_status": (
            "eligible" if promotion_eligible else "measurement_only_not_accepted"
        ),
        "resource_validation_passed": resource_validation_passed,
        "resource_gate_passed": resource_gate_passed,
        "passed": passed,
        "parity_failure_reasons": parity_failure_reasons,
        "performance_failure_reasons": performance_failure_reasons,
        "acceptance_failure_reasons": acceptance_failure_reasons,
        "mechanism_failure_reasons": mechanism_failure_reasons,
        "resource_failure_reasons": resource_failure_reasons,
        "failure_reasons": [
            *parity_failure_reasons,
            *performance_failure_reasons,
            *acceptance_failure_reasons,
            *mechanism_failure_reasons,
            *resource_failure_reasons,
        ],
        "scenario_root": str(scenario_root),
    }


def _print_report(report: dict[str, Any]) -> None:
    def render(value: Any, digits: int) -> str:
        return f"{float(value):.{digits}f}" if value is not None else "n/a"

    profiling = report.get("profiling_summary") or {}
    phase_timings = (
        f"completion={render(profiling.get('completion_end_to_end_seconds'), 2)}s "
        f"attribution={render(profiling.get('attribution_duration_seconds'), 2)}s "
        f"phase0={render(profiling.get('phase0_duration_seconds'), 2)}s "
        f"phase3={render(profiling.get('phase3_duration_seconds'), 2)}s "
        f"phase4={render(profiling.get('phase4_duration_seconds'), 2)}s "
        f"phase4_batch={render(profiling.get('phase4_avg_batch_seconds'), 2)}s "
        f"batches={profiling.get('phase4_batches_observed', 'n/a')}/"
        f"{profiling.get('phase4_total_batches', 'n/a')}"
    )
    target = report.get("performance_target_seconds")
    stretch_target = report.get("performance_stretch_target_seconds")
    performance_gate = report.get("performance_passed")
    performance_status = (
        "n/a" if performance_gate is None else ("PASS" if performance_gate else "FAIL")
    )
    resource_gate = report.get("resource_gate_passed")
    resource_status = (
        "n/a" if resource_gate is None else ("PASS" if resource_gate else "FAIL")
    )
    mechanism_gate = report.get("mechanism_validation_passed")
    mechanism_status = (
        "n/a" if mechanism_gate is None else ("PASS" if mechanism_gate else "FAIL")
    )
    reconciliation_status = (
        "REQUIRED" if report.get("reconciliation_required") else "OK"
    )
    acceptance_status = (
        "ELIGIBLE" if report.get("promotion_eligible", True) else "MEASUREMENT_ONLY"
    )
    typed_bucket_metrics = "".join(
        f"{name.replace('<-', '_')}="
        f"{render(report.get(f'bucket_{name.replace("<-", "_").replace("-", "_")}_exact'), 0)} "
        for name in CANONICAL_BUCKET_NAMES
    )
    print(
        f"{report['case']}: "
        f"candidate={render(report.get('candidate_duration_seconds'), 2)}s "
        f"baseline={render(report.get('baseline_duration_seconds'), 2)}s "
        f"speedup={render(report.get('speedup'), 3)}x "
        f"target={render(target, 2)}s "
        f"stretch={render(stretch_target, 2)}s "
        f"{phase_timings} "
        f"feature={render(report.get('feature_jaccard'), 6)} "
        f"{typed_bucket_metrics}"
        f"token={render(report.get('target_token_match'), 0)} "
        f"parity={'n/a' if report.get('parity_passed') is None else ('PASS' if report.get('parity_passed') else 'FAIL')} "
        f"performance={performance_status} "
        f"stretch={'n/a' if report.get('scientific_acceptance_attempted') is False else ('PASS' if report.get('performance_stretch_passed') else 'MISS')} "
        f"resource={resource_status} "
        f"mechanism={mechanism_status} "
        f"acceptance={acceptance_status} "
        f"reconciliation={reconciliation_status} "
        f"gate={'PASS' if report['passed'] else 'FAIL'}"
    )


def _gate_summary(reports: Sequence[dict[str, Any]]) -> dict[str, bool | None]:
    if reports and all(
        report.get("scientific_acceptance_attempted") is False for report in reports
    ):
        return {
            "parity_passed": None,
            "performance_passed": None,
            "resource_gate_passed": None,
            "mechanism_validation_passed": None,
            "promotion_eligible": False,
            "reconciliation_required": False,
            "scientific_acceptance_attempted": False,
            "diagnostic_execution_passed": all(
                report["runner_returncode"] == 0 for report in reports
            ),
            "passed": False,
        }
    parity_passed = all(report["parity_passed"] for report in reports)
    performance_results = [
        report["performance_passed"]
        for report in reports
        if report["performance_passed"] is not None
    ]
    performance_passed = all(performance_results) if performance_results else None
    resource_gate_passed = bool(reports) and all(
        report.get("resource_gate_passed") is True for report in reports
    )
    mechanism_results = [
        report.get("mechanism_validation_passed")
        for report in reports
        if report.get("mechanism_validation_passed") is not None
    ]
    mechanism_validation_passed = all(mechanism_results) if mechanism_results else None
    reconciliation_required = any(
        report.get("reconciliation_required") is True for report in reports
    )
    promotion_eligible = bool(reports) and all(
        report.get("promotion_eligible", True) is True for report in reports
    )
    code_retention_eligible = bool(reports) and all(
        report.get("code_retention_eligible", report.get("parity_passed")) is True
        for report in reports
    )
    automatic_promotion_allowed = False
    passed = (
        bool(reports)
        and parity_passed
        and performance_passed is not False
        and promotion_eligible
        and resource_gate_passed
        and mechanism_validation_passed is not False
        and not reconciliation_required
        and all(report["runner_returncode"] == 0 for report in reports)
    )
    automatic_promotion_eligible = bool(automatic_promotion_allowed and passed)
    return {
        "parity_passed": parity_passed,
        "performance_passed": performance_passed,
        "resource_gate_passed": resource_gate_passed,
        "mechanism_validation_passed": mechanism_validation_passed,
        "promotion_eligible": promotion_eligible,
        "code_retention_eligible": code_retention_eligible,
        "code_retention_selected": False,
        "automatic_promotion_allowed": automatic_promotion_allowed,
        "automatic_promotion_eligible": automatic_promotion_eligible,
        "automatic_promotion_selected": False,
        "reconciliation_required": reconciliation_required,
        "passed": passed,
    }


def _run(args: argparse.Namespace) -> int:
    cases = SUITES[args.suite]
    diagnostic = args.diagnostic_stop_mode != "none"
    profile_variants = {
        case.key: _profile_variant(case, args.candidate_profile) for case in cases
    }
    scope = _baseline_scope(args.candidate_profile, args.fidelity)
    baseline_registry_path = _baseline_registry_path(scope)
    baseline_registry_contract = _load_baseline_registry_contract(
        baseline_registry_path,
        expected_scope=scope,
    )
    baseline_entries = (
        _baseline_entries()
        if baseline_registry_path == DEFAULT_BASELINE_REGISTRY
        else _baseline_entries(baseline_registry_path)
    )
    if not diagnostic:
        missing_baselines = [
            case.key for case in cases if case.key not in baseline_entries
        ]
        if missing_baselines:
            raise ValueError(
                f"{scope.value} baseline is not registered for: "
                + ", ".join(missing_baselines)
            )
    verified_baseline_pins: dict[str, dict[str, Any]] = {}
    if args.fidelity in EXACT_BASELINE_FIDELITIES and not diagnostic:
        for case in cases:
            verified_baseline_pins[case.key] = _validate_exact_baseline_pins(
                case=case,
                candidate_profile=args.candidate_profile,
                baseline_entry=baseline_entries[case.key],
                scope=scope,
            )
    run_id = args.run_id or time.strftime("perf-%Y%m%d-%H%M%S")
    run_root = args.output_root / run_id
    if args.dry_run:
        for case in cases:
            scenario_file = run_root / "configs" / f"{case.variant}_{case.fixture}.json"
            candidate_root = run_root / "candidates" / f"{case.variant}_{case.fixture}"
            print(
                f"{case.key}: "
                f"{shlex.join(_runner_command(scenario_file=scenario_file, output_root=candidate_root, run_id=run_id, run_goal=args.run_goal, baseline_registry=baseline_registry_path))}"
            )
        return 0

    provenance = gpu_provenance()
    assert_h200_allocation(
        provenance,
        allow_test_environment=args.allow_non_h200_test_only,
    )
    source_state = capture_source_state()
    evidence = execution_evidence(
        provenance,
        allow_test_environment=args.allow_non_h200_test_only,
    )
    baseline_registry = baseline_registry_contract.payload
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "configs").mkdir()
    (run_root / "candidates").mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "run_goal": args.run_goal,
        "candidate_profile": args.candidate_profile,
        "candidate_profile_contract": {
            case.key: {
                "requires": profile_variants[case.key].requires.__dict__,
                "baseline_pins": profile_variants[
                    case.key
                ].baseline_pins.as_overrides(),
                "physical": profile_variants[case.key].physical.as_overrides(),
            }
            for case in cases
        },
        "candidate_overrides_by_case": {
            case.key: _candidate_overrides(case, args.candidate_profile)
            for case in cases
        },
        "suite": args.suite,
        "fidelity": args.fidelity,
        "code_retention_allowed": (args.fidelity in CODE_RETENTION_ALLOWED_FIDELITIES),
        "code_retention_policy": (
            "Bounded-or-stronger fidelity is necessary but not sufficient for "
            "runtime inclusion. Retain only a non-dominated implementation or "
            "one with a distinct operational role; archive dominated candidates "
            "as indexed patches instead of selectable dead code."
        ),
        "automatic_promotion_allowed": (
            args.fidelity in AUTOMATIC_PROMOTION_ALLOWED_FIDELITIES
        ),
        "automatic_promotion_policy": (
            "Automatic promotion is disabled until Step 1b adds the calibrated "
            "correctness contract. This harness does not mutate defaults."
        ),
        "comparison_semantics": COMPARISON_SEMANTICS[args.fidelity],
        "claim_limitation": COMPARISON_CLAIM_LIMITATION,
        "exact_semantics_claim_allowed": False,
        "thresholds": FIDELITY_THRESHOLDS[args.fidelity],
        "baseline_scope": scope.value,
        "baseline_scope_declared": baseline_registry_contract.scope_declared,
        "baseline_registry": str(baseline_registry_path),
        "baseline_registry_id": baseline_registry.get("registry_id"),
        "baseline_registry_provenance": baseline_registry.get("source_provenance"),
        "baseline_references": {
            case.key: baseline_entries.get(case.key) for case in cases
        },
        "verified_baseline_pins": verified_baseline_pins,
        "workspace_mode": source_state.get("workspace_mode", "live"),
        "live_workspace_rationale": (
            "The isolated optimization worktree is the candidate under test; "
            "the CLI freezes and verifies both repository states for each case."
        )
        if source_state.get("workspace_mode", "live") == "live"
        else None,
        "no_edits_during_run_enforced": True,
        "source_state_before": source_state,
        "execution_provenance": provenance,
        "execution_evidence": evidence,
        "governor_state_policy": "read_only_no_promotion_or_default_mutation",
        "host_memory_stop_gib": args.host_memory_stop_gib,
        "host_rss_stop_gib": args.host_rss_stop_gib,
        "diagnostic_stop_mode": args.diagnostic_stop_mode,
        "diagnostic_stop_phase4_batches": args.diagnostic_stop_phase4_batches,
        "scientific_acceptance_attempted": not diagnostic,
        "cases": [case.key for case in cases],
    }
    write_json(run_root / "run_manifest.json", manifest)

    reports: list[dict[str, Any]] = []
    write_json(
        run_root / "performance_report.json",
        {
            **manifest,
            "status": "running",
            "passed": False,
            "reports": reports,
        },
    )
    current_case: str | None = None
    try:
        for case in cases:
            current_case = case.key
            before_case = capture_source_state()
            if before_case != source_state:
                raise RuntimeError(
                    f"Source state changed before {case.key}; refusing mixed-source run"
                )
            scenario_file = run_root / "configs" / f"{case.variant}_{case.fixture}.json"
            write_json(
                scenario_file,
                _case_scenario(
                    case,
                    args.fidelity,
                    args.candidate_profile,
                    baseline_scope=scope,
                    diagnostic_stop_mode=args.diagnostic_stop_mode,
                    diagnostic_stop_phase4_batches=args.diagnostic_stop_phase4_batches,
                ),
            )
            candidate_root = run_root / "candidates" / f"{case.variant}_{case.fixture}"
            command = _runner_command(
                scenario_file=scenario_file,
                output_root=candidate_root,
                run_id=run_id,
                run_goal=args.run_goal,
                baseline_registry=baseline_registry_path,
            )
            print(f"Running {case.key}: {shlex.join(command)}", flush=True)
            returncode = _stream_runner(
                command,
                output_root=candidate_root,
                host_memory_stop_gib=args.host_memory_stop_gib,
                host_rss_stop_gib=args.host_rss_stop_gib,
            )
            after_case = capture_source_state()
            if after_case != source_state:
                raise RuntimeError(
                    f"Source state changed during {case.key}; result is not comparable"
                )
            scenario_root = candidate_root / f"perf_{case.variant}_{case.fixture}"
            has_result = (scenario_root / "result.json").is_file()
            if returncode != 0 and not has_result:
                raise RuntimeError(
                    f"Existing sparsification runner failed for {case.key} "
                    f"with exit code {returncode}"
                )
            report = _result_report(
                case,
                candidate_root=candidate_root,
                baseline_entry=baseline_entries.get(case.key, {}),
                runner_returncode=returncode,
            )
            reports.append(report)
            _print_report(report)
            write_json(
                run_root / "performance_report.json",
                {
                    **manifest,
                    "status": "running",
                    "passed": False,
                    "reports": reports,
                },
            )
            if returncode != 0 and report["parity_passed"] is True:
                raise RuntimeError(
                    f"Existing sparsification runner exited {returncode} after a "
                    f"passing comparison for {case.key}"
                )
    except Exception as error:
        write_json(
            run_root / "performance_report.json",
            {
                **manifest,
                "status": "failed",
                "passed": False,
                "current_case": current_case,
                "error": f"{type(error).__name__}: {error}",
                "source_state_after": capture_source_state(),
                "reports": reports,
            },
        )
        raise

    summary = {
        **manifest,
        "status": "completed",
        "source_state_after": capture_source_state(),
        **_gate_summary(reports),
        "reports": reports,
    }
    write_json(run_root / "performance_report.json", summary)
    print(f"Report: {run_root / 'performance_report.json'}")
    if diagnostic:
        return 0 if summary.get("diagnostic_execution_passed") is True else 1
    return 0 if summary["passed"] else 1


def _list_suites() -> int:
    for name, cases in SUITES.items():
        rendered = ", ".join(case.key for case in cases)
        print(f"{name}: {rendered}")
    return 0


def _validate_claims(path: Path) -> int:
    payload = load_mechanism_claims(path)
    print(
        f"{payload['campaign_id']}: {len(payload['claims'])} validated mechanism claims"
    )
    return 0


def _list_campaign(path: Path, *, require_frozen: bool) -> int:
    payload, workloads = load_performance_campaign(
        path,
        require_frozen=require_frozen,
    )
    print(f"{payload['campaign_id']}: {len(workloads)} workloads")
    for line in render_campaign_workloads(workloads):
        print(line)
    return 0


def _freeze_campaign(path: Path, *, output: Path | None) -> int:
    payload = freeze_performance_campaign(path, output_path=output)
    destination = output or path
    print(f"{payload['campaign_id']}: froze {len(payload['workloads'])} workloads")
    print(f"Manifest: {destination}")
    return 0


def _campaign_launcher_variant(workload: Mapping[str, Any]) -> str:
    model = workload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("campaign workload model must be an object")
    variant = model.get("variant")
    provider = model.get("provider")
    if not isinstance(variant, str) or not isinstance(provider, str):
        raise ValueError("campaign workload model variant/provider must be strings")
    launcher_variant = f"{variant}_{provider}"
    # This validation also keeps profile capability selection aligned with the
    # existing performance launcher rather than inventing a campaign-only map.
    build_chpc_baseline_config(variant=launcher_variant, cluster="granite")
    return launcher_variant


def _full_answer_shard_command(
    *,
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    output_root: Path,
    run_id: str,
    workload_id: str,
    profile_role: str,
    execution_mode: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "exact-trace-bench",
        "run-full-answer-shard",
        "--trajectory",
        str(trajectory_path),
        "--trace-specs",
        str(trace_specs_path),
        "--shards",
        str(shards_path),
        "--shard-id",
        "0",
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--run-name",
        "Exact-trace long-prefix scaling",
        "--run-description",
        f"{workload_id} {profile_role} {execution_mode}",
        "--run-goal",
        "Establish and optimize the exact-trace long-prefix scaling envelope.",
    ]


def _prepare_campaign_workload(args: argparse.Namespace) -> int:
    if (
        args.workspace_project_root is not None
        and args.workspace_library_root is not None
    ):
        validate_launch_snapshot(
            workspace_root=args.workspace_project_root,
            library_root=args.workspace_library_root,
            import_roots=(
                args.workspace_project_root / "src",
                args.workspace_project_root,
                args.workspace_library_root,
            ),
        )
    campaign_repo_root = (
        REPO_ROOT
        if args.workspace_project_root is None
        else args.workspace_project_root.resolve()
    )
    resolved = resolve_frozen_campaign_workload(
        args.manifest,
        args.workload_id,
        repo_root=campaign_repo_root,
    )
    workload = resolved.workload
    comparison_policy = workload.get("comparison_policy")
    if not isinstance(comparison_policy, Mapping):
        raise ValueError("campaign workload comparison_policy must be an object")
    required_execution_mode = comparison_policy.get("required_execution_mode")
    if (
        required_execution_mode is not None
        and args.execution_mode != required_execution_mode
    ):
        raise ValueError(
            f"campaign workload requires execution mode {required_execution_mode!r}, "
            f"not {args.execution_mode!r}"
        )
    required_resource_policy = comparison_policy.get("runtime_resource_policy")
    if (
        required_resource_policy is not None
        and args.runtime_resource_policy != required_resource_policy
    ):
        raise ValueError(
            "campaign workload requires runtime resource policy "
            f"{required_resource_policy!r}, not {args.runtime_resource_policy!r}"
        )
    profiles = workload.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("campaign workload profiles must be an object")
    profile_name = args.profile_name or profiles.get(args.profile_role)
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError(
            f"campaign workload lacks a {args.profile_role!r} profile name"
        )
    launcher_variant = _campaign_launcher_variant(workload)
    fixture = workload.get("fixture")
    fixture_name = (
        str(fixture.get("fixture_name"))
        if isinstance(fixture, Mapping)
        else args.workload_id
    )
    case = Case(launcher_variant, fixture_name)
    profile_variant = _profile_variant(case, profile_name)

    template = build_chpc_baseline_config(
        variant=launcher_variant,
        cluster="granite",
    )["scenarios"][0]
    allowed_template_keys = set(base_trace_defaults()) | set(
        PUBLIC_TRANSCODER_KNOB_KEYS
    )
    graph_overrides = {
        key: value for key, value in template.items() if key in allowed_template_keys
    }
    graph_overrides.update(profile_variant.overrides())
    if args.execution_mode == "full":
        diagnostic_stop_mode = "none"
        diagnostic_stop_phase4_batches = None
    elif args.execution_mode == "phase0-probe":
        diagnostic_stop_mode = "phase0_probe"
        diagnostic_stop_phase4_batches = None
    elif args.execution_mode == "phase3-probe":
        diagnostic_stop_mode = "phase3_probe"
        diagnostic_stop_phase4_batches = None
    else:
        diagnostic_stop_mode = "transition_probe"
        diagnostic_stop_phase4_batches = args.probe_batches
    graph_overrides.update(
        {
            "diagnostic_stop_mode": diagnostic_stop_mode,
            "diagnostic_stop_phase4_batches": diagnostic_stop_phase4_batches,
            "incremental_telemetry_jsonl": True,
            "profile_attribution": True,
            "profile_log_interval": 1,
            "runtime_resource_policy": args.runtime_resource_policy,
            "resource_planning_envelope": dict(workload["resource_envelope"]),
            "correctness_probe_mode": args.correctness_probe_mode,
            "correctness_policy_id": args.correctness_policy_id,
        }
    )
    if args.feature_row_influence_requirement is not None:
        graph_overrides["feature_row_influence_requirement"] = (
            args.feature_row_influence_requirement
        )
    selection = select_tokens(
        dict(resolved.trajectory),
        explicit_indices=[resolved.generated_index],
    )
    specs = build_trace_specs(
        dict(resolved.trajectory),
        selection,
        graph_knob_overrides=graph_overrides,
    )
    if len(specs) != 1:
        raise RuntimeError("campaign workload preparation must produce one trace spec")
    spec = specs[0]
    prefix = workload["prefix"]
    target = workload["target"]
    if (
        spec["prefix_token_count"] != prefix["actual_tokens"]
        or spec["target_position"] != target["absolute_position"]
        or spec["target_token_id"] != target["token_id"]
    ):
        raise RuntimeError("prepared trace spec diverges from frozen campaign target")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = output_dir / "trace_selection.json"
    specs_path = output_dir / "trace_specs.jsonl"
    shards_path = output_dir / "shards.json"
    write_trace_selection(selection_path, selection)
    write_trace_specs(specs_path, specs)
    write_shards(
        shards_path,
        build_lpt_shards(specs, shard_count=1, trace_specs_file=specs_path),
    )
    run_id = args.run_id or (
        f"{resolved.campaign_id}-{args.workload_id}-{args.profile_role}-"
        f"{args.execution_mode}"
    )
    launch_output_root = (
        args.launch_output_root.resolve()
        if args.launch_output_root is not None
        else output_dir / "output"
    )
    scheduler_request = {
        "cluster": args.allocation_cluster,
        "resource_profile": args.allocation_profile,
        "account": args.allocation_account,
        "partition": args.allocation_partition,
        "qos": args.allocation_qos,
        "gpus_per_task": args.allocation_gpus_per_task,
        "cpus_per_task": args.allocation_cpus_per_task,
        "memory": args.allocation_mem,
        "walltime": args.allocation_walltime,
    }
    if args.preheat_policy == "file_cache" and not args.preheat_path:
        raise ValueError("file_cache preheat policy requires --preheat-path")
    launch_spec = build_full_answer_launch_spec(
        trajectory_path=resolved.trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        output_root=launch_output_root,
        shard_selection="0",
        run={
            "run_id": run_id,
            "run_name": "Exact-trace long-prefix scaling",
            "run_description": (
                f"{args.workload_id} {args.profile_role} {args.execution_mode}"
            ),
            "run_goal": (
                "Establish and optimize the exact-trace long-prefix scaling envelope."
            ),
        },
        specs=specs,
        planning_envelope=workload["resource_envelope"],
        scheduler_request=scheduler_request,
        runtime_resource_policy=args.runtime_resource_policy,
        runtime_resource_override_rationale=(args.runtime_resource_override_rationale),
        preheat={
            "policy": args.preheat_policy,
            "paths": [str(path.resolve()) for path in args.preheat_path],
        },
        workspace={
            "policy": "immutable_snapshot_required",
            "project_root": (
                None
                if args.workspace_project_root is None
                else str(args.workspace_project_root.resolve())
            ),
            "library_root": (
                None
                if args.workspace_library_root is None
                else str(args.workspace_library_root.resolve())
            ),
        },
        monitoring={
            "gpu_sampler": True,
            "runtime_resource_samples": args.runtime_resource_policy != "off",
            "incremental_trace_telemetry": True,
            "postrun_mechanism_validation": True,
        },
    )
    declared_arms = comparison_policy.get("arms")
    if isinstance(declared_arms, Mapping) and args.profile_role in declared_arms:
        declared_arm = declared_arms[args.profile_role]
        if not isinstance(declared_arm, Mapping):
            raise ValueError(
                f"comparison_policy.arms.{args.profile_role} must be an object"
            )
        mechanism = launch_spec.mechanism_selections[0]
        mismatches = {
            key: (expected, mechanism.get(key))
            for key, expected in declared_arm.items()
            if key not in {"config_pins", "physical_caps", "derived_execution"}
            if mechanism.get(key) != expected
        }
        declared_physical_caps = declared_arm.get("physical_caps")
        if declared_physical_caps is not None:
            if not isinstance(declared_physical_caps, Mapping):
                raise ValueError(
                    f"comparison_policy.arms.{args.profile_role}.physical_caps "
                    "must be an object"
                )
            mismatches.update(
                {
                    f"physical_caps.{key}": (expected, spec["graph_knobs"].get(key))
                    for key, expected in declared_physical_caps.items()
                    if spec["graph_knobs"].get(key) != expected
                }
            )
        declared_config_pins = declared_arm.get("config_pins")
        if declared_config_pins is not None:
            if not isinstance(declared_config_pins, Mapping):
                raise ValueError(
                    f"comparison_policy.arms.{args.profile_role}.config_pins "
                    "must be an object"
                )
            mismatches.update(
                {
                    f"config_pins.{key}": (expected, spec["graph_knobs"].get(key))
                    for key, expected in declared_config_pins.items()
                    if spec["graph_knobs"].get(key) != expected
                }
            )
        declared_derived_execution = declared_arm.get("derived_execution")
        if declared_derived_execution is not None:
            if not isinstance(declared_derived_execution, Mapping):
                raise ValueError(
                    f"comparison_policy.arms.{args.profile_role}.derived_execution "
                    "must be an object"
                )
            derived_execution = {
                "actual_phase1_forward_trace_width": mechanism.get(
                    "forward_lane_count"
                ),
                "backward_batch_capacity": spec["graph_knobs"].get(
                    "nnsight_session_capacity"
                ),
            }
            mismatches.update(
                {
                    f"derived_execution.{key}": (
                        expected,
                        derived_execution.get(key),
                    )
                    for key, expected in declared_derived_execution.items()
                    if derived_execution.get(key) != expected
                }
            )
        if mismatches:
            raise ValueError(
                f"prepared profile {profile_name!r} disagrees with declared "
                f"comparison arm {args.profile_role!r}: {mismatches!r}"
            )
    launch_spec_path = output_dir / "launch_spec.json"
    write_json(launch_spec_path, launch_spec.to_record())
    command = render_local_full_answer_command(launch_spec)
    write_json(
        output_dir / "prepared_workload.json",
        {
            "schema_version": 1,
            "campaign_id": resolved.campaign_id,
            "workload_id": args.workload_id,
            "manifest_path": str(resolved.manifest_path.resolve()),
            "manifest_sha256": resolved.manifest_sha256,
            "trajectory_path": str(resolved.trajectory_path.resolve()),
            "fixture_catalog_path": str(resolved.fixture_catalog_path.resolve()),
            "prompt_path": str(resolved.prompt_path.resolve()),
            "prefix_token_count": len(resolved.prefix_token_ids),
            "prefix_token_ids_sha256": prefix["token_ids_sha256"],
            "generated_index": resolved.generated_index,
            "target_position": target["absolute_position"],
            "target_token_id": target["token_id"],
            "target_token_text": target["token_text"],
            "launcher": "existing_full_answer_shard_runner",
            "launcher_variant": launcher_variant,
            "profile_role": args.profile_role,
            "profile_name": profile_name,
            "profile_selection_source": (
                "explicit_override"
                if args.profile_name is not None
                else "campaign_role"
            ),
            "profile_contract": {
                "requires": profile_variant.requires.__dict__,
                "baseline_pins": profile_variant.baseline_pins.as_overrides(),
                "physical": profile_variant.physical.as_overrides(),
            },
            "execution_mode": args.execution_mode,
            "diagnostic_stop_mode": diagnostic_stop_mode,
            "diagnostic_stop_phase4_batches": diagnostic_stop_phase4_batches,
            "resource_envelope": workload["resource_envelope"],
            "comparison_policy": workload["comparison_policy"],
            "trace_specs_path": str(specs_path),
            "shards_path": str(shards_path),
            "launch_output_root": str(launch_output_root),
            "launch_spec_path": str(launch_spec_path),
            "launch_selection_fingerprint": launch_spec.to_record()[
                "selection_fingerprint"
            ],
            "launch_command": command,
        },
    )
    print(f"Prepared {args.workload_id} {args.profile_role} {args.execution_mode}")
    print(f"Bundle: {output_dir}")
    print(f"Launch: {shlex.join(command)}")
    return 0


def _run_prepared_campaign_workload(args: argparse.Namespace) -> int:
    """Validate or run one immutable prepared command through the resource guard."""

    workload = validate_prepared_workload(args.prepared_workload)
    launch_spec = workload.launch_spec
    if args.host_memory_stop_gib is not None or args.host_rss_stop_gib is not None:
        raise ValueError(
            "resource overrides must be represented in launch_spec.json; "
            "renderer-only guard overrides are forbidden"
        )
    runtime_policy = launch_spec["runtime_resource_policy"]
    planning_envelope = launch_spec["planning_envelope"]
    assert isinstance(planning_envelope, Mapping)
    host_memory_stop_gib = None
    host_rss_stop_gib = None
    if runtime_policy == "enforce":
        raw_host_memory = planning_envelope.get("host_memory_stop_gib")
        raw_host_rss = planning_envelope.get("host_rss_stop_gib")
        host_memory_stop_gib = (
            None if raw_host_memory is None else float(raw_host_memory)
        )
        host_rss_stop_gib = None if raw_host_rss is None else float(raw_host_rss)
    if host_memory_stop_gib is not None and host_memory_stop_gib <= 0:
        raise ValueError(
            "prepared workload host-memory guard thresholds must be positive"
        )
    if host_rss_stop_gib is not None and host_rss_stop_gib <= 0:
        raise ValueError(
            "prepared workload host-memory guard thresholds must be positive"
        )
    output_root = workload.launch_output_root
    print(f"Running prepared workload: {workload.path}")
    print(f"Launch output: {output_root}")
    print(
        "Host guards: "
        f"cgroup={host_memory_stop_gib if host_memory_stop_gib is not None else 'off'} "
        f"GiB, rss={host_rss_stop_gib if host_rss_stop_gib is not None else 'off'} GiB"
    )
    if args.validate_only:
        print("Prepared workload validation: passed")
        return 0
    return _stream_runner(
        list(workload.launch_command),
        output_root=output_root,
        host_memory_stop_gib=host_memory_stop_gib,
        host_rss_stop_gib=host_rss_stop_gib,
    )


def _parse_sequence_entry(value: str) -> tuple[str, Path]:
    entry_id, separator, path = value.partition("=")
    if not separator or not entry_id or not path:
        raise argparse.ArgumentTypeError("sequence entries must use ENTRY_ID=PATH")
    return entry_id, Path(path)


def _prepare_campaign_sequence(args: argparse.Namespace) -> int:
    record = build_prepared_sequence(args.entry)
    output = write_prepared_sequence(args.output, record)
    print(f"Prepared sequence: {output}")
    print(f"Sequence fingerprint: {record['sequence_fingerprint']}")
    return 0


def _prepared_workload_args(
    entry: PreparedSequenceEntry, *, validate_only: bool
) -> argparse.Namespace:
    return argparse.Namespace(
        prepared_workload=entry.prepared_workload_path,
        host_memory_stop_gib=None,
        host_rss_stop_gib=None,
        validate_only=validate_only,
    )


def _validate_prepared_sequence_entry(entry: PreparedSequenceEntry) -> dict[str, Any]:
    expectations = entry.mechanism_expectations
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "slurm/exact_trace_bench/validate_full_answer_feature_row_mode.py"
        ),
        "--label",
        entry.entry_id,
        "--output-root",
        str(entry.launch_output_root),
    ]
    for expectation, flag in (
        ("feature_row_influence_mode", "--expected-mode"),
        ("backward_engine_mode", "--expected-backward-engine-mode"),
        ("forward_graph_mode", "--expected-forward-graph-mode"),
        ("vjp_kernel_mode", "--expected-vjp-kernel-mode"),
        ("forward_lane_count", "--expected-forward-lane-count"),
        ("session_capacity", "--expected-session-capacity"),
        ("backward_batch_capacity", "--expected-backward-batch-capacity"),
        (
            "phase1_trace_batch_size_max",
            "--expected-phase1-trace-batch-size-max",
        ),
        (
            "actual_phase1_forward_trace_width",
            "--expected-phase1-effective-trace-batch-size",
        ),
        ("phase3_batch_size", "--expected-phase3-batch-size"),
        ("phase4_batch_size", "--expected-phase4-batch-size"),
        (
            "decoder_active_row_residency",
            "--expected-decoder-active-row-residency",
        ),
        (
            "decoder_active_row_residency_requirement",
            "--expected-decoder-active-row-residency-requirement",
        ),
        (
            "decoder_active_row_max_bytes",
            "--expected-decoder-active-row-max-bytes",
        ),
        (
            "decoder_active_row_safety_margin_bytes",
            "--expected-decoder-active-row-safety-margin-bytes",
        ),
    ):
        value = expectations[expectation]
        command.extend([flag, str(int(value) if isinstance(value, bool) else value)])
    require_full_completion = expectations["require_full_completion"]
    if not isinstance(require_full_completion, bool):
        raise RuntimeError(
            f"prepared mechanism expectations for {entry.entry_id} have an "
            "invalid require_full_completion value"
        )
    if require_full_completion:
        # This gate also makes the validator strictly reopen every compact graph.
        command.append("--require-full-completion")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"mechanism validation failed for {entry.entry_id}: {detail}"
        )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"mechanism validation returned no object for {entry.entry_id}"
        )
    return payload


def _run_prepared_campaign_sequence(args: argparse.Namespace) -> int:
    sequence = load_prepared_sequence(
        args.prepared_sequence, require_outputs_absent=True
    )
    if args.validate_only:
        for entry in sequence.entries:
            _run_prepared_campaign_workload(
                _prepared_workload_args(entry, validate_only=True)
            )
        print("Prepared sequence validation: passed")
        return 0
    return run_prepared_sequence(
        sequence,
        state_root=args.state_root,
        preflight_entry=lambda entry: _run_prepared_campaign_workload(
            _prepared_workload_args(entry, validate_only=True)
        ),
        run_entry=lambda entry: _run_prepared_campaign_workload(
            _prepared_workload_args(entry, validate_only=False)
        ),
        validate_entry=_validate_prepared_sequence_entry,
    )


def _inspect_prepared_campaign_sequence(args: argparse.Namespace) -> int:
    sequence = load_prepared_sequence(
        args.prepared_sequence, require_outputs_absent=True
    )
    if args.format == "prepared-paths":
        for entry in sequence.entries:
            print(entry.prepared_workload_path)
    elif args.format == "output-paths":
        for entry in sequence.entries:
            print(entry.launch_output_root)
    else:
        print(json.dumps(sequence.record, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="H200 exact-trace performance and graph-parity loop"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List fixed performance suites")
    claims = subparsers.add_parser(
        "validate-claims",
        help="Validate a machine-readable mechanism claim ledger",
    )
    claims.add_argument("path", type=Path)
    campaign = subparsers.add_parser(
        "list-campaign",
        help="Validate and list a typed performance campaign without model loading",
    )
    campaign.add_argument("manifest", type=Path)
    campaign.add_argument(
        "--require-frozen",
        action="store_true",
        help="Fail when any workload or immutable fingerprint remains planned",
    )
    freeze_campaign = subparsers.add_parser(
        "freeze-campaign",
        help="Freeze campaign prefix and target fingerprints from trajectories",
    )
    freeze_campaign.add_argument("manifest", type=Path)
    freeze_campaign.add_argument("--output", type=Path)
    prepare_campaign = subparsers.add_parser(
        "prepare-campaign-workload",
        help=(
            "Verify one frozen workload and prepare it for the existing "
            "full-answer shard runner"
        ),
    )
    prepare_campaign.add_argument("manifest", type=Path)
    prepare_campaign.add_argument("workload_id")
    prepare_campaign.add_argument(
        "--profile-role",
        required=True,
        help="Named arm from the frozen workload profiles object",
    )
    prepare_campaign.add_argument(
        "--profile-name",
        choices=tuple(CANDIDATE_PROFILES),
        help=(
            "Use an explicit registered profile while preserving the campaign "
            "role and frozen workload"
        ),
    )
    prepare_campaign.add_argument(
        "--execution-mode",
        choices=("full", "phase0-probe", "phase3-probe", "transition-probe"),
        default="full",
    )
    prepare_campaign.add_argument(
        "--probe-batches",
        type=int,
        default=4,
        help="Phase-4 batches for transition-probe mode (default: 4)",
    )
    prepare_campaign.add_argument("--output-dir", type=Path, required=True)
    prepare_campaign.add_argument("--launch-output-root", type=Path)
    prepare_campaign.add_argument("--run-id")
    prepare_campaign.add_argument(
        "--feature-row-influence-requirement",
        choices=("preferred", "required"),
    )
    prepare_campaign.add_argument(
        "--runtime-resource-policy",
        choices=("off", "measure_only", "enforce"),
        default="measure_only",
    )
    prepare_campaign.add_argument(
        "--correctness-probe-mode",
        choices=("off", "smoke", "required"),
        default="off",
        help=(
            "Run the versioned post-trace correctness probe. Smoke records evidence "
            "without failing an otherwise successful trace; required fails closed."
        ),
    )
    prepare_campaign.add_argument(
        "--correctness-policy-id",
        choices=("behavioral_closure_v1",),
        default="behavioral_closure_v1",
        help="Versioned correctness policy identity (default: behavioral_closure_v1)",
    )
    prepare_campaign.add_argument(
        "--runtime-resource-override-rationale",
        default=(
            "optimization measurement uses scheduler limits while estimate-based "
            "runtime guards remain disabled"
        ),
    )
    prepare_campaign.add_argument(
        "--preheat-policy", choices=("none", "file_cache"), default="none"
    )
    prepare_campaign.add_argument(
        "--preheat-path", type=Path, action="append", default=[]
    )
    prepare_campaign.add_argument("--workspace-project-root", type=Path)
    prepare_campaign.add_argument("--workspace-library-root", type=Path)
    prepare_campaign.add_argument("--allocation-cluster", default="granite")
    prepare_campaign.add_argument("--allocation-profile")
    prepare_campaign.add_argument("--allocation-account")
    prepare_campaign.add_argument("--allocation-partition")
    prepare_campaign.add_argument("--allocation-qos")
    prepare_campaign.add_argument("--allocation-gpus-per-task", type=int)
    prepare_campaign.add_argument("--allocation-cpus-per-task", type=int)
    prepare_campaign.add_argument("--allocation-mem")
    prepare_campaign.add_argument("--allocation-walltime")
    run_prepared = subparsers.add_parser(
        "run-prepared-campaign-workload",
        help=(
            "Execute a prepared full-answer command unchanged through the existing "
            "GPU sampler and host-memory guard"
        ),
    )
    run_prepared.add_argument("prepared_workload", type=Path)
    run_prepared.add_argument("--host-memory-stop-gib", type=float)
    run_prepared.add_argument("--host-rss-stop-gib", type=float)
    run_prepared.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the prepared workload and launch specification without running",
    )
    prepare_sequence = subparsers.add_parser(
        "prepare-campaign-sequence",
        help="Freeze an ordered, same-allocation sequence of prepared workloads",
    )
    prepare_sequence.add_argument(
        "--entry", type=_parse_sequence_entry, action="append", required=True
    )
    prepare_sequence.add_argument("--output", type=Path, required=True)
    run_sequence = subparsers.add_parser(
        "run-prepared-campaign-sequence",
        help="Execute a frozen prepared sequence after one wrapper-owned preheat",
    )
    run_sequence.add_argument("prepared_sequence", type=Path)
    run_sequence.add_argument("--state-root", type=Path, required=True)
    run_sequence.add_argument("--validate-only", action="store_true")
    inspect_sequence = subparsers.add_parser(
        "inspect-prepared-campaign-sequence",
        help="Validate and inspect a frozen prepared sequence",
    )
    inspect_sequence.add_argument("prepared_sequence", type=Path)
    inspect_sequence.add_argument(
        "--format", choices=("json", "prepared-paths", "output-paths"), default="json"
    )
    run = subparsers.add_parser("run", help="Run one suite and enforce parity")
    run.add_argument("suite", choices=tuple(SUITES))
    run.add_argument(
        "--fidelity",
        choices=FIDELITY_LEVELS,
        default="bounded",
    )
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--run-id")
    run.add_argument("--run-goal", default=DEFAULT_RUN_GOAL)
    run.add_argument(
        "--candidate-profile",
        choices=tuple(CANDIDATE_PROFILES),
        default="canonical",
    )
    run.add_argument(
        "--host-memory-stop-gib",
        type=float,
        help=(
            "Terminate the runner process group when Slurm cgroup-v1 job memory "
            "usage reaches this threshold; disabled by default."
        ),
    )
    run.add_argument(
        "--host-rss-stop-gib",
        type=float,
        help=(
            "Terminate the runner process group when Slurm cgroup-v1 total_rss "
            "reaches this threshold; file cache alone does not trigger it. "
            "Disabled by default."
        ),
    )
    run.add_argument(
        "--diagnostic-stop-mode",
        choices=("none", "phase0_probe", "phase3_probe", "transition_probe"),
        default="none",
    )
    run.add_argument("--diagnostic-stop-phase4-batches", type=int)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--allow-non-h200-test-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return _list_suites()
    if args.command == "validate-claims":
        return _validate_claims(args.path)
    if args.command == "list-campaign":
        return _list_campaign(args.manifest, require_frozen=args.require_frozen)
    if args.command == "freeze-campaign":
        return _freeze_campaign(args.manifest, output=args.output)
    if args.command == "prepare-campaign-workload":
        if args.probe_batches <= 0:
            raise ValueError("--probe-batches must be positive")
        return _prepare_campaign_workload(args)
    if args.command == "run-prepared-campaign-workload":
        return _run_prepared_campaign_workload(args)
    if args.command == "prepare-campaign-sequence":
        return _prepare_campaign_sequence(args)
    if args.command == "run-prepared-campaign-sequence":
        return _run_prepared_campaign_sequence(args)
    if args.command == "inspect-prepared-campaign-sequence":
        return _inspect_prepared_campaign_sequence(args)
    if args.host_memory_stop_gib is not None and args.host_memory_stop_gib <= 0:
        raise ValueError("--host-memory-stop-gib must be positive")
    if args.host_rss_stop_gib is not None and args.host_rss_stop_gib <= 0:
        raise ValueError("--host-rss-stop-gib must be positive")
    if (
        args.diagnostic_stop_phase4_batches is not None
        and args.diagnostic_stop_phase4_batches <= 0
    ):
        raise ValueError("--diagnostic-stop-phase4-batches must be positive")
    if (
        args.diagnostic_stop_mode != "transition_probe"
        and args.diagnostic_stop_phase4_batches is not None
    ):
        raise ValueError(
            "--diagnostic-stop-phase4-batches requires --diagnostic-stop-mode transition_probe"
        )
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
