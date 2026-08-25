"""Bounded, typed evidence for correctness-sensitive feature frontiers.

The compact graph is an accepted rendering artifact, not enough evidence to
explain selection-boundary churn.  This module retains every selected feature
and a bounded band immediately below the cutoff, together with only the local
quantities needed by deterministic alias matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class FeatureKey:
    """One position-local feature identity."""

    layer: int
    position: int
    feature_id: int

    def __post_init__(self) -> None:
        if self.layer < 0 or self.position < 0 or self.feature_id < 0:
            raise ValueError("feature key components must be non-negative")


@dataclass(frozen=True)
class FeatureFeatureEdge:
    """A signed ``feature <- feature`` neighborhood contribution."""

    source: FeatureKey
    weight: float

    def __post_init__(self) -> None:
        _require_finite("feature<-feature weight", self.weight)


@dataclass(frozen=True)
class LogitFeatureEdge:
    """A signed ``logit <- feature`` neighborhood contribution."""

    logit_token_id: int
    weight: float

    def __post_init__(self) -> None:
        if self.logit_token_id < 0:
            raise ValueError("logit token id must be non-negative")
        _require_finite("logit<-feature weight", self.weight)


@dataclass(frozen=True)
class FrontierRecord:
    """Evidence for one selected or immediately-near-cutoff feature."""

    key: FeatureKey
    selected: bool
    rank: int
    selection_score: float
    cutoff_distance: float
    activation: float
    signed_target_effect: float
    influence: float
    seed_rank: int | None = None
    final_selected_rank: int | None = None
    feature_from_feature: tuple[FeatureFeatureEdge, ...] = ()
    logit_from_feature: tuple[LogitFeatureEdge, ...] = ()

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("frontier rank must be non-negative")
        for name in (
            "selection_score",
            "cutoff_distance",
            "activation",
            "signed_target_effect",
            "influence",
        ):
            _require_finite(name, getattr(self, name))
        for name in ("seed_rank", "final_selected_rank"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when present")
        _require_unique(
            tuple(edge.source for edge in self.feature_from_feature),
            "feature<-feature sources",
        )
        _require_unique(
            tuple(edge.logit_token_id for edge in self.logit_from_feature),
            "logit<-feature token ids",
        )

    @property
    def edge_mass(self) -> float:
        return math.fsum(
            abs(edge.weight)
            for edge in (*self.feature_from_feature, *self.logit_from_feature)
        )


@dataclass(frozen=True)
class FrontierBounds:
    """Fail-closed evidence bounds for one typed graph."""

    max_selected: int
    max_near_cutoff: int

    def __post_init__(self) -> None:
        if self.max_selected <= 0:
            raise ValueError("max_selected must be positive")
        if self.max_near_cutoff < 0:
            raise ValueError("max_near_cutoff must be non-negative")


@dataclass(frozen=True)
class FrontierEvidence:
    """Complete selected frontier plus one explicitly bounded comparison band."""

    graph_fingerprint: str
    provider_fingerprint: str
    decoder_fingerprint: str
    cutoff_score: float
    relative_cutoff_gap: float
    cutoff_tie_count: int
    near_cutoff_count: int
    selected: tuple[FrontierRecord, ...]
    near_cutoff: tuple[FrontierRecord, ...]
    bounds: FrontierBounds

    def __post_init__(self) -> None:
        for name in (
            "graph_fingerprint",
            "provider_fingerprint",
            "decoder_fingerprint",
        ):
            _require_fingerprint(name, getattr(self, name))
        _require_finite("cutoff_score", self.cutoff_score)
        _require_finite("relative_cutoff_gap", self.relative_cutoff_gap)
        if self.relative_cutoff_gap < 0:
            raise ValueError("relative_cutoff_gap must be non-negative")
        if self.cutoff_tie_count < 0:
            raise ValueError("cutoff_tie_count must be non-negative")
        if self.near_cutoff_count < 0:
            raise ValueError("near_cutoff_count must be non-negative")
        if self.near_cutoff_count != len(self.near_cutoff):
            raise ValueError(
                "near_cutoff_count must equal the retained near-cutoff records"
            )
        if len(self.selected) > self.bounds.max_selected:
            raise ValueError("selected frontier exceeds its evidence bound")
        if len(self.near_cutoff) > self.bounds.max_near_cutoff:
            raise ValueError("near-cutoff frontier exceeds its evidence bound")
        if any(not record.selected for record in self.selected):
            raise ValueError("selected frontier contains an unselected record")
        if any(record.selected for record in self.near_cutoff):
            raise ValueError("near-cutoff frontier contains a selected record")
        _require_unique(
            tuple(record.key for record in self.records),
            "frontier feature keys",
        )
        _require_unique(
            tuple(record.rank for record in self.records),
            "frontier ranks",
        )

    @property
    def records(self) -> tuple[FrontierRecord, ...]:
        return self.selected + self.near_cutoff


def build_frontier_evidence(
    *,
    graph_fingerprint: str,
    provider_fingerprint: str,
    decoder_fingerprint: str,
    cutoff_score: float,
    relative_cutoff_gap: float,
    cutoff_tie_count: int,
    near_cutoff_count: int,
    selected: tuple[FrontierRecord, ...],
    near_cutoff: tuple[FrontierRecord, ...],
    bounds: FrontierBounds,
) -> FrontierEvidence:
    """Canonicalize bounded evidence without silently dropping any record."""

    return FrontierEvidence(
        graph_fingerprint=graph_fingerprint,
        provider_fingerprint=provider_fingerprint,
        decoder_fingerprint=decoder_fingerprint,
        cutoff_score=cutoff_score,
        relative_cutoff_gap=relative_cutoff_gap,
        cutoff_tie_count=cutoff_tie_count,
        near_cutoff_count=near_cutoff_count,
        selected=tuple(sorted(selected, key=_record_order)),
        near_cutoff=tuple(sorted(near_cutoff, key=_record_order)),
        bounds=bounds,
    )


def _record_order(record: FrontierRecord) -> tuple[int, FeatureKey]:
    return record.rank, record.key


def _require_unique[T](values: tuple[T, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _require_finite(label: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _require_fingerprint(label: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a sha256 fingerprint")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a sha256 fingerprint") from exc
