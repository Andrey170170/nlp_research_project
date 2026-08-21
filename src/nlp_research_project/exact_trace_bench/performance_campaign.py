from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .full_answer.schemas import load_trajectory
from .io_utils import read_json, write_json


class MechanismDisposition(str, Enum):
    EXACT_PROMOTION_CANDIDATE = "exact_promotion_candidate"
    EXACT_OPT_IN = "exact_opt_in"
    BOUNDED_RESEARCH = "bounded_research"
    REJECTED = "rejected"


class WorkloadPreparationStatus(str, Enum):
    PLANNED = "planned"
    FROZEN = "frozen"


@dataclass(frozen=True)
class CampaignWorkload:
    workload_id: str
    preparation_status: WorkloadPreparationStatus
    model_variant: str
    provider: str
    prompt_family: str
    prompt_role: str
    requested_prefix_tokens: int
    actual_prefix_tokens: int | None
    profiles: tuple[tuple[str, str], ...]
    run_disposition: str

    def profile_for(self, role: str) -> str | None:
        return dict(self.profiles).get(role)

    @property
    def candidate_profile(self) -> str | None:
        """Compatibility view for ordinary paired campaigns."""

        return self.profile_for("candidate")

    @property
    def control_profile(self) -> str | None:
        """Compatibility view for ordinary paired campaigns."""

        return self.profile_for("control")


@dataclass(frozen=True)
class ResolvedCampaignWorkload:
    """Frozen campaign workload verified against its source files."""

    campaign_id: str
    workload: Mapping[str, Any]
    manifest_path: Path
    manifest_sha256: str
    fixture_catalog_path: Path
    prompt_path: Path
    trajectory_path: Path
    trajectory: Mapping[str, Any]
    generated_index: int
    prefix_token_ids: tuple[int, ...]


_MECHANISM_REQUIRED_SECTIONS = (
    "identity",
    "applicability",
    "semantics",
    "correctness",
    "performance",
    "resources",
    "disposition",
    "promotion_separation",
)
_WORKLOAD_REQUIRED_FIELDS = (
    "workload_id",
    "preparation_status",
    "fixture",
    "model",
    "prompt",
    "trajectory",
    "prefix",
    "target",
    "operational_class",
    "profiles",
    "run_disposition",
    "resource_envelope",
    "comparison_policy",
)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    digest = _require_nonempty_string(value, label=label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def validate_mechanism_claims(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("mechanism claims schema_version must be 1")
    _require_nonempty_string(payload.get("campaign_id"), label="campaign_id")
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("mechanism claims must contain a non-empty claims list")
    seen: set[str] = set()
    for index, raw_claim in enumerate(claims):
        claim = _require_mapping(raw_claim, label=f"claims[{index}]")
        missing = [key for key in _MECHANISM_REQUIRED_SECTIONS if key not in claim]
        if missing:
            raise ValueError(
                f"claims[{index}] is missing required sections: {', '.join(missing)}"
            )
        identity = _require_mapping(
            claim["identity"], label=f"claims[{index}].identity"
        )
        mechanism_id = _require_nonempty_string(
            identity.get("mechanism_id"),
            label=f"claims[{index}].identity.mechanism_id",
        )
        if mechanism_id in seen:
            raise ValueError(f"duplicate mechanism_id: {mechanism_id}")
        seen.add(mechanism_id)
        _require_nonempty_string(
            identity.get("implementation_commit"),
            label=f"claims[{index}].identity.implementation_commit",
        )
        disposition = _require_mapping(
            claim["disposition"], label=f"claims[{index}].disposition"
        )
        try:
            MechanismDisposition(disposition.get("status"))
        except ValueError as error:
            raise ValueError(
                f"claims[{index}].disposition.status is invalid"
            ) from error
        promotion = _require_mapping(
            claim["promotion_separation"],
            label=f"claims[{index}].promotion_separation",
        )
        for axis in ("mechanism", "baseline", "default", "governor"):
            _require_nonempty_string(
                promotion.get(axis),
                label=f"claims[{index}].promotion_separation.{axis}",
            )


def load_mechanism_claims(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"mechanism claims must be an object: {path}")
    validate_mechanism_claims(payload)
    return payload


def _validate_frozen_workload(workload: Mapping[str, Any], *, label: str) -> None:
    fixture = _require_mapping(workload["fixture"], label=f"{label}.fixture")
    trajectory = _require_mapping(workload["trajectory"], label=f"{label}.trajectory")
    prefix = _require_mapping(workload["prefix"], label=f"{label}.prefix")
    target = _require_mapping(workload["target"], label=f"{label}.target")
    _require_sha256(
        fixture.get("catalog_sha256"), label=f"{label}.fixture.catalog_sha256"
    )
    _require_sha256(
        fixture.get("prompt_sha256"), label=f"{label}.fixture.prompt_sha256"
    )
    _require_sha256(trajectory.get("sha256"), label=f"{label}.trajectory.sha256")
    actual_tokens = prefix.get("actual_tokens")
    if not isinstance(actual_tokens, int) or actual_tokens <= 0:
        raise ValueError(f"{label}.prefix.actual_tokens must be a positive integer")
    _require_sha256(
        prefix.get("token_ids_sha256"), label=f"{label}.prefix.token_ids_sha256"
    )
    target_position = target.get("absolute_position")
    target_token_id = target.get("token_id")
    if not isinstance(target_position, int) or target_position != actual_tokens:
        raise ValueError(
            f"{label}.target.absolute_position must equal prefix.actual_tokens"
        )
    if not isinstance(target_token_id, int) or target_token_id < 0:
        raise ValueError(f"{label}.target.token_id must be a non-negative integer")


def validate_performance_campaign(
    payload: Mapping[str, Any],
    *,
    require_frozen: bool = False,
) -> tuple[CampaignWorkload, ...]:
    if payload.get("schema_version") != 1:
        raise ValueError("performance campaign schema_version must be 1")
    _require_nonempty_string(payload.get("campaign_id"), label="campaign_id")
    execution_policy = payload.get("execution_policy")
    ordered_profile_roles: tuple[str, ...] = ()
    if execution_policy is not None:
        policy = _require_mapping(execution_policy, label="execution_policy")
        raw_roles = policy.get("ordered_profile_roles")
        if raw_roles is not None:
            if not isinstance(raw_roles, list) or not raw_roles:
                raise ValueError(
                    "execution_policy.ordered_profile_roles must be a non-empty list"
                )
            ordered_profile_roles = tuple(
                _require_nonempty_string(
                    role,
                    label=f"execution_policy.ordered_profile_roles[{index}]",
                )
                for index, role in enumerate(raw_roles)
            )
            if len(set(ordered_profile_roles)) != len(ordered_profile_roles):
                raise ValueError(
                    "execution_policy.ordered_profile_roles must not contain duplicates"
                )
    workloads = payload.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("performance campaign must contain workloads")
    seen: set[str] = set()
    validated: list[CampaignWorkload] = []
    for index, raw_workload in enumerate(workloads):
        label = f"workloads[{index}]"
        workload = _require_mapping(raw_workload, label=label)
        missing = [key for key in _WORKLOAD_REQUIRED_FIELDS if key not in workload]
        if missing:
            raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
        workload_id = _require_nonempty_string(
            workload.get("workload_id"), label=f"{label}.workload_id"
        )
        if workload_id in seen:
            raise ValueError(f"duplicate workload_id: {workload_id}")
        seen.add(workload_id)
        try:
            status = WorkloadPreparationStatus(workload.get("preparation_status"))
        except ValueError as error:
            raise ValueError(f"{label}.preparation_status is invalid") from error
        model = _require_mapping(workload["model"], label=f"{label}.model")
        prompt = _require_mapping(workload["prompt"], label=f"{label}.prompt")
        prefix = _require_mapping(workload["prefix"], label=f"{label}.prefix")
        profiles = _require_mapping(workload["profiles"], label=f"{label}.profiles")
        for profile_role, profile_name in profiles.items():
            _require_nonempty_string(profile_role, label=f"{label}.profiles role")
            _require_nonempty_string(
                profile_name, label=f"{label}.profiles.{profile_role}"
            )
        missing_ordered_roles = [
            role for role in ordered_profile_roles if role not in profiles
        ]
        if missing_ordered_roles:
            raise ValueError(
                f"{label}.profiles is missing ordered roles: "
                + ", ".join(missing_ordered_roles)
            )
        requested_tokens = prefix.get("requested_tokens")
        if not isinstance(requested_tokens, int) or requested_tokens <= 0:
            raise ValueError(f"{label}.prefix.requested_tokens must be positive")
        actual_tokens = prefix.get("actual_tokens")
        if actual_tokens is not None and (
            not isinstance(actual_tokens, int) or actual_tokens <= 0
        ):
            raise ValueError(f"{label}.prefix.actual_tokens must be positive or null")
        if require_frozen and status is not WorkloadPreparationStatus.FROZEN:
            raise ValueError(f"{workload_id} is not frozen")
        if status is WorkloadPreparationStatus.FROZEN:
            _validate_frozen_workload(workload, label=label)
        validated.append(
            CampaignWorkload(
                workload_id=workload_id,
                preparation_status=status,
                model_variant=_require_nonempty_string(
                    model.get("variant"), label=f"{label}.model.variant"
                ),
                provider=_require_nonempty_string(
                    model.get("provider"), label=f"{label}.model.provider"
                ),
                prompt_family=_require_nonempty_string(
                    prompt.get("family"), label=f"{label}.prompt.family"
                ),
                prompt_role=_require_nonempty_string(
                    prompt.get("role"), label=f"{label}.prompt.role"
                ),
                requested_prefix_tokens=requested_tokens,
                actual_prefix_tokens=actual_tokens,
                profiles=tuple(
                    (str(profile_role), str(profile_name))
                    for profile_role, profile_name in profiles.items()
                ),
                run_disposition=_require_nonempty_string(
                    workload.get("run_disposition"),
                    label=f"{label}.run_disposition",
                ),
            )
        )
    return tuple(validated)


def load_performance_campaign(
    path: Path,
    *,
    require_frozen: bool = False,
) -> tuple[dict[str, Any], tuple[CampaignWorkload, ...]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"performance campaign must be an object: {path}")
    workloads = validate_performance_campaign(payload, require_frozen=require_frozen)
    return payload, workloads


def resolve_frozen_campaign_workload(
    manifest_path: Path,
    workload_id: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> ResolvedCampaignWorkload:
    """Resolve one workload and recheck every frozen input fingerprint.

    This is deliberately preparation-only. Execution remains owned by the
    ordinary full-answer shard runner.
    """
    payload, _ = load_performance_campaign(manifest_path, require_frozen=True)
    workload = next(
        (
            row
            for row in payload["workloads"]
            if isinstance(row, Mapping) and row.get("workload_id") == workload_id
        ),
        None,
    )
    if workload is None:
        raise ValueError(f"campaign workload not found: {workload_id}")
    fixture = _require_mapping(workload["fixture"], label=f"{workload_id}.fixture")
    trajectory_ref = _require_mapping(
        workload["trajectory"], label=f"{workload_id}.trajectory"
    )
    prefix = _require_mapping(workload["prefix"], label=f"{workload_id}.prefix")
    target = _require_mapping(workload["target"], label=f"{workload_id}.target")
    catalog_path = _resolve_campaign_path(
        fixture.get("catalog"),
        repo_root=repo_root,
        label=f"{workload_id}.fixture.catalog",
    )
    prompt_path = _resolve_campaign_path(
        fixture.get("prompt_file"),
        repo_root=repo_root,
        label=f"{workload_id}.fixture.prompt_file",
    )
    trajectory_path = _resolve_campaign_path(
        trajectory_ref.get("path"),
        repo_root=repo_root,
        label=f"{workload_id}.trajectory.path",
    )
    expected_files = (
        (catalog_path, fixture["catalog_sha256"], "fixture catalog"),
        (prompt_path, fixture["prompt_sha256"], "prompt"),
        (trajectory_path, trajectory_ref["sha256"], "trajectory"),
    )
    for path, expected_sha256, label in expected_files:
        observed_sha256 = _file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"{workload_id} {label} fingerprint mismatch: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )

    trajectory = dict(load_trajectory(trajectory_path))
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if trajectory.get("prompt_text") != prompt_text:
        raise ValueError(
            f"{workload_id} trajectory prompt_text does not match {prompt_path}"
        )
    prompt_token_ids = [int(token_id) for token_id in trajectory["prompt_token_ids"]]
    generated_tokens = list(trajectory["generated_tokens"])
    full_token_ids = [
        *prompt_token_ids,
        *(int(row["token_id"]) for row in generated_tokens),
    ]
    actual_tokens = int(prefix["actual_tokens"])
    if actual_tokens >= len(full_token_ids):
        raise ValueError(
            f"{workload_id} has no target token after its {actual_tokens}-token prefix"
        )
    prefix_token_ids = tuple(full_token_ids[:actual_tokens])
    observed_prefix_sha256 = _token_ids_sha256(prefix_token_ids)
    if observed_prefix_sha256 != prefix["token_ids_sha256"]:
        raise ValueError(
            f"{workload_id} prefix token fingerprint mismatch: expected "
            f"{prefix['token_ids_sha256']}, observed {observed_prefix_sha256}"
        )
    generated_index = actual_tokens - len(prompt_token_ids)
    if generated_index < 0 or generated_index >= len(generated_tokens):
        raise ValueError(f"{workload_id} target generated index is out of bounds")
    target_row = generated_tokens[generated_index]
    expected_target = {
        "absolute_position": actual_tokens,
        "token_id": int(target_row["token_id"]),
        "token_text": str(target_row["token_text"]),
    }
    observed_target = {
        "absolute_position": target.get("absolute_position"),
        "token_id": target.get("token_id"),
        "token_text": target.get("token_text"),
    }
    if observed_target != expected_target:
        raise ValueError(
            f"{workload_id} frozen target mismatch: expected {expected_target!r}, "
            f"observed {observed_target!r}"
        )
    return ResolvedCampaignWorkload(
        campaign_id=str(payload["campaign_id"]),
        workload=workload,
        manifest_path=manifest_path,
        manifest_sha256=_file_sha256(manifest_path),
        fixture_catalog_path=catalog_path,
        prompt_path=prompt_path,
        trajectory_path=trajectory_path,
        trajectory=trajectory,
        generated_index=generated_index,
        prefix_token_ids=prefix_token_ids,
    )


def render_campaign_workloads(workloads: Sequence[CampaignWorkload]) -> list[str]:
    rendered: list[str] = []
    for workload in workloads:
        if (
            workload.control_profile is not None
            and workload.candidate_profile is not None
        ):
            profiles = f"{workload.control_profile}->{workload.candidate_profile}"
        else:
            profiles = ",".join(
                f"{role}:{profile}" for role, profile in workload.profiles
            )
        rendered.append(
            f"{workload.workload_id}: status={workload.preparation_status.value} "
            f"model={workload.model_variant}/{workload.provider} "
            f"prompt={workload.prompt_family}/{workload.prompt_role} "
            f"prefix={workload.actual_prefix_tokens or '?'}"
            f"/{workload.requested_prefix_tokens} "
            f"profiles={profiles} "
            f"run={workload.run_disposition}"
        )
    return rendered


def _resolve_campaign_path(value: Any, *, repo_root: Path, label: str) -> Path:
    path = Path(_require_nonempty_string(value, label=label))
    return path if path.is_absolute() else repo_root / path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_performance_campaign(
    manifest_path: Path,
    *,
    output_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Populate immutable workload fingerprints from generated trajectories."""
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"performance campaign must be an object: {manifest_path}")
    validate_performance_campaign(payload)
    trajectory_cache: dict[Path, tuple[dict[str, Any], str]] = {}
    for index, raw_workload in enumerate(payload["workloads"]):
        workload = _require_mapping(raw_workload, label=f"workloads[{index}]")
        fixture = _require_mapping(
            workload["fixture"], label=f"workloads[{index}].fixture"
        )
        trajectory_ref = _require_mapping(
            workload["trajectory"], label=f"workloads[{index}].trajectory"
        )
        prefix = _require_mapping(
            workload["prefix"], label=f"workloads[{index}].prefix"
        )
        target = _require_mapping(
            workload["target"], label=f"workloads[{index}].target"
        )
        catalog_path = _resolve_campaign_path(
            fixture.get("catalog"),
            repo_root=repo_root,
            label=f"workloads[{index}].fixture.catalog",
        )
        prompt_path = _resolve_campaign_path(
            fixture.get("prompt_file"),
            repo_root=repo_root,
            label=f"workloads[{index}].fixture.prompt_file",
        )
        trajectory_path = _resolve_campaign_path(
            trajectory_ref.get("path"),
            repo_root=repo_root,
            label=f"workloads[{index}].trajectory.path",
        )
        if trajectory_path not in trajectory_cache:
            trajectory_cache[trajectory_path] = (
                dict(load_trajectory(trajectory_path)),
                _file_sha256(trajectory_path),
            )
        trajectory, trajectory_sha256 = trajectory_cache[trajectory_path]
        prompt_text = prompt_path.read_text(encoding="utf-8")
        if trajectory.get("prompt_text") != prompt_text:
            raise ValueError(
                f"{workload['workload_id']} trajectory prompt_text does not match "
                f"{prompt_path}"
            )
        prompt_token_ids = list(trajectory["prompt_token_ids"])
        generated_tokens = list(trajectory["generated_tokens"])
        generated_token_ids = [int(row["token_id"]) for row in generated_tokens]
        full_token_ids = [*prompt_token_ids, *generated_token_ids]
        requested_tokens = int(prefix["requested_tokens"])
        if requested_tokens < len(prompt_token_ids):
            raise ValueError(
                f"{workload['workload_id']} requested prefix precedes generated text"
            )
        if requested_tokens >= len(full_token_ids):
            raise ValueError(
                f"{workload['workload_id']} needs a target token at position "
                f"{requested_tokens}, but trajectory has only {len(full_token_ids)} tokens"
            )
        generated_index = requested_tokens - len(prompt_token_ids)
        target_row = generated_tokens[generated_index]
        fixture["catalog_sha256"] = _file_sha256(catalog_path)
        fixture["prompt_sha256"] = _file_sha256(prompt_path)
        trajectory_ref["sha256"] = trajectory_sha256
        prefix["actual_tokens"] = requested_tokens
        prefix["token_ids_sha256"] = _token_ids_sha256(
            full_token_ids[:requested_tokens]
        )
        target["absolute_position"] = requested_tokens
        target["token_id"] = int(target_row["token_id"])
        target["token_text"] = str(target_row["token_text"])
        workload["preparation_status"] = WorkloadPreparationStatus.FROZEN.value
    validate_performance_campaign(payload, require_frozen=True)
    destination = output_path or manifest_path
    write_json(destination, payload)
    return payload
