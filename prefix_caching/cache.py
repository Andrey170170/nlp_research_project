"""PrefixCache: stores and compares per-position active features across
consecutive tracing steps.

The ``active_features`` tensor returned by ``attribute()`` has shape
``(F, 3)`` where each row is ``[layer, position, feature_idx]``.  Features
at a given *position* depend only on tokens at that position and earlier,
so they cannot change when a new token is appended.  This class stores the
feature set from step N and compares it against step N+1 to empirically
verify that invariant and measure how much work is redundant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CompareResult:
    """Result of comparing cached prefix features against freshly computed ones."""

    # How many token positions were in the cache from the previous step.
    cached_positions: int
    # How many of those positions had an identical feature set in the fresh run.
    matched_positions: int
    # Total features across all cached positions (sum of per-position counts).
    cached_feature_count: int
    # How many of those features appeared identically in the fresh run.
    matched_feature_count: int

    @property
    def position_match_rate(self) -> float:
        if self.cached_positions == 0:
            return 0.0
        return self.matched_positions / self.cached_positions

    @property
    def feature_match_rate(self) -> float:
        if self.cached_feature_count == 0:
            return 0.0
        return self.matched_feature_count / self.cached_feature_count

    def to_dict(self) -> dict:
        return {
            "cached_positions": self.cached_positions,
            "matched_positions": self.matched_positions,
            "cached_feature_count": self.cached_feature_count,
            "matched_feature_count": self.matched_feature_count,
            "position_match_rate": round(self.position_match_rate, 6),
            "feature_match_rate": round(self.feature_match_rate, 6),
        }


@dataclass
class PrefixCache:
    """Stores active features from one tracing step so the next step can
    compare against them.

    Usage::

        cache = PrefixCache()

        for step_idx in range(max_steps):
            result = attribute(...)  # full computation, no shortcuts
            active_features = result["active_features"]  # (F, 3) tensor

            if cache.has_data:
                stats = cache.compare(active_features)
                # stats tells you how much matched

            cache.store(input_token_ids, active_features)
    """

    # Maps position -> set of (layer, feature_idx) tuples.
    _features_by_position: dict[int, set[tuple[int, int]]] = field(
        default_factory=dict, repr=False
    )
    # The token IDs that were used when this cache was populated.
    _token_ids: tuple[int, ...] = ()

    @property
    def has_data(self) -> bool:
        return len(self._features_by_position) > 0

    @property
    def cached_position_count(self) -> int:
        return len(self._features_by_position)

    def store(
        self,
        input_token_ids: torch.Tensor,
        active_features: torch.Tensor,
    ) -> None:
        """Save the per-position feature sets from the current step.

        Parameters
        ----------
        input_token_ids:
            1-D tensor of token IDs for the current step's input.
        active_features:
            ``(F, 3)`` int tensor where each row is
            ``[layer, position, feature_idx]``.
        """
        self._token_ids = tuple(input_token_ids.tolist())

        # Group features by position.
        features_np = active_features.cpu().numpy()
        by_pos: dict[int, set[tuple[int, int]]] = {}
        for row in features_np:
            layer, position, feat_idx = int(row[0]), int(row[1]), int(row[2])
            if position not in by_pos:
                by_pos[position] = set()
            by_pos[position].add((layer, feat_idx))

        self._features_by_position = by_pos

    def compare(self, fresh_active_features: torch.Tensor) -> CompareResult:
        """Compare cached prefix features against freshly computed features.

        Only compares positions that exist in the cache (i.e. positions from
        the previous step).  The new position (the appended token) is ignored
        since it has no cached data to compare against.

        Parameters
        ----------
        fresh_active_features:
            ``(F, 3)`` int tensor from the current step's ``attribute()`` call.

        Returns
        -------
        CompareResult with match counts and rates.
        """
        if not self.has_data:
            return CompareResult(
                cached_positions=0,
                matched_positions=0,
                cached_feature_count=0,
                matched_feature_count=0,
            )

        # Group fresh features by position.
        fresh_np = fresh_active_features.cpu().numpy()
        fresh_by_pos: dict[int, set[tuple[int, int]]] = {}
        for row in fresh_np:
            layer, position, feat_idx = int(row[0]), int(row[1]), int(row[2])
            if position not in fresh_by_pos:
                fresh_by_pos[position] = set()
            fresh_by_pos[position].add((layer, feat_idx))

        cached_positions = 0
        matched_positions = 0
        cached_feature_count = 0
        matched_feature_count = 0

        for pos, cached_set in self._features_by_position.items():
            cached_positions += 1
            cached_feature_count += len(cached_set)

            fresh_set = fresh_by_pos.get(pos, set())
            matched = cached_set & fresh_set
            matched_feature_count += len(matched)

            if cached_set == fresh_set:
                matched_positions += 1

        return CompareResult(
            cached_positions=cached_positions,
            matched_positions=matched_positions,
            cached_feature_count=cached_feature_count,
            matched_feature_count=matched_feature_count,
        )

    def clear(self) -> None:
        """Reset the cache (e.g. between completions)."""
        self._features_by_position.clear()
        self._token_ids = ()
