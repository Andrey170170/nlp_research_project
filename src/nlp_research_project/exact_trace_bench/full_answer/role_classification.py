from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Mapping

from ..io_utils import ensure_dir, read_json, write_json, write_jsonl
from .calibration import position_band_for_index
from .schemas import load_trajectory
from .temporal import discover_graph_paths


ROLE_SCHEMA_VERSION = 1
DEFAULT_ROLE_CLUSTERS = (
    "math_number",
    "math_operator",
    "answer_marker",
    "answer_number",
    "punctuation_format",
    "reasoning_text",
)

ROLE_FOR_CLUSTER = {
    "math_number": "math_number",
    "math_operator": "math_operator",
    "answer_marker": "answer_marker",
    "answer_number": "final_answer_number",
    "punctuation_format": "punctuation_format",
    "reasoning_text": "reasoning_text",
}

_NUMBER_RE = re.compile(r"^[\s,$%+-]*\d[\d,]*(?:\.\d+)?[%\s]*$")
_OPERATOR_RE = re.compile(r"^[\s]*(?:[+\-*/=×÷]|equals?|total:?|sum:?)[\s]*$", re.I)
_PUNCT_RE = re.compile(r"^[\s.,;:!?()\[\]{}\-–—'\"`]+$")
_ANSWER_MARKER_RE = re.compile(r"(answer|therefore|thus|so|finally|####|final)", re.I)
_REASONING_WORD_RE = re.compile(r"[A-Za-z]")
_ARITH_CONTEXT_RE = re.compile(
    r"\d\s*(?:[+\-*/=×÷]|equals?|total|sum)\s*\d|(?:[+\-*/=×÷]|=)\s*\d|\d\s*(?:[+\-*/=×÷]|=)",
    re.I,
)


@dataclass(frozen=True)
class TokenRole:
    generated_index: int
    absolute_token_position: int
    token_id: int
    token_text: str
    role: str
    role_cluster: str
    confidence: float
    tags: tuple[str, ...]
    rationale: str
    char_start: int
    char_end: int
    context: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_index": self.generated_index,
            "absolute_token_position": self.absolute_token_position,
            "token_id": self.token_id,
            "token_text": self.token_text,
            "role": self.role,
            "role_cluster": self.role_cluster,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "rationale": self.rationale,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "context": self.context,
        }


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def generated_text(trajectory: Mapping[str, Any]) -> str:
    return "".join(
        str(token.get("token_text", "")) for token in trajectory["generated_tokens"]
    )


def _token_spans(trajectory: Mapping[str, Any]) -> list[tuple[int, int]]:
    offset = 0
    spans: list[tuple[int, int]] = []
    for token in trajectory["generated_tokens"]:
        text = str(token.get("token_text", ""))
        start = offset
        offset += len(text)
        spans.append((start, offset))
    return spans


def _answer_start_char(text: str) -> int | None:
    markers = [
        r"####",
        r"final\s+answer\s*: ?",
        r"answer\s+is\s*",
        r"therefore,?\s+the\s+answer\s+is\s*",
    ]
    starts = []
    for pattern in markers:
        match = re.search(pattern, text, flags=re.I)
        if match is not None:
            starts.append(match.start())
    return min(starts) if starts else None


def classify_generated_token_roles(
    trajectory: Mapping[str, Any],
    *,
    context_chars: int = 36,
) -> list[TokenRole]:
    """Classify generated tokens into coarse functional roles.

    This is intentionally heuristic and auditable.  It is meant to seed a human /
    agent review pass and role-matched calibration manifests, not to be a final
    semantic classifier.
    """

    text = generated_text(trajectory)
    spans = _token_spans(trajectory)
    answer_start = _answer_start_char(text)
    roles: list[TokenRole] = []
    tokens = trajectory["generated_tokens"]
    for token, (start, end) in zip(tokens, spans):
        token_text = str(token.get("token_text", ""))
        stripped = token_text.strip()
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        context = text[context_start:context_end]
        tags: list[str] = []
        in_answer_region = answer_start is not None and start >= answer_start
        if bool(token.get("is_stop")) or "<end_of_turn>" in token_text:
            role = "stop_special"
            cluster = "punctuation_format"
            confidence = 0.98
            rationale = "stop/special token"
        elif in_answer_region and _NUMBER_RE.match(stripped):
            role = "final_answer_number"
            cluster = "answer_number"
            confidence = 0.92
            tags.append("answer_region")
            rationale = "numeric token after answer marker"
        elif _ANSWER_MARKER_RE.search(token_text):
            role = "answer_marker"
            cluster = "answer_marker"
            confidence = 0.84
            rationale = "answer/discourse marker token"
        elif _NUMBER_RE.match(stripped):
            role = "math_number"
            cluster = "math_number"
            confidence = 0.88
            if _ARITH_CONTEXT_RE.search(context):
                tags.append("arithmetic_context")
                confidence = 0.94
            if in_answer_region:
                tags.append("answer_region")
            rationale = "numeric token"
        elif _OPERATOR_RE.match(stripped):
            role = "math_operator"
            cluster = "math_operator"
            confidence = 0.86
            rationale = "operator/equality/total token"
        elif _PUNCT_RE.match(token_text):
            role = "punctuation_format"
            cluster = "punctuation_format"
            confidence = 0.9
            rationale = "punctuation/formatting token"
        elif _REASONING_WORD_RE.search(token_text):
            role = "reasoning_text"
            cluster = "reasoning_text"
            confidence = 0.74
            if in_answer_region:
                tags.append("answer_region")
            rationale = "ordinary reasoning text token"
        else:
            role = "other"
            cluster = "reasoning_text"
            confidence = 0.5
            rationale = "fallback token role"
        roles.append(
            TokenRole(
                generated_index=int(token["generated_index"]),
                absolute_token_position=int(token["absolute_token_position"]),
                token_id=int(token["token_id"]),
                token_text=token_text,
                role=role,
                role_cluster=cluster,
                confidence=confidence,
                tags=tuple(tags),
                rationale=rationale,
                char_start=start,
                char_end=end,
                context=context,
            )
        )
    return roles


def write_role_classification(
    *,
    trajectory_path: Path,
    output_path: Path,
    name: str | None = None,
    prompt_id: str | None = None,
    label: str | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    trajectory = load_trajectory(trajectory_path)
    roles = classify_generated_token_roles(trajectory)
    counts = Counter(role.role_cluster for role in roles)
    payload = {
        "schema_version": ROLE_SCHEMA_VERSION,
        "classification_kind": "heuristic_token_role_v1",
        "created_at": _now(),
        "trajectory_path": str(trajectory_path),
        "trajectory_id": trajectory["trajectory_id"],
        "name": name,
        "prompt_id": prompt_id,
        "label": label,
        "run_root": str(run_root) if run_root is not None else None,
        "token_count": len(roles),
        "role_cluster_counts": dict(sorted(counts.items())),
        "role_counts": dict(sorted(Counter(role.role for role in roles).items())),
        "tokens": [role.as_dict() for role in roles],
    }
    ensure_dir(output_path.parent)
    write_json(output_path, payload)
    return payload


def _load_launch_records(launch_prep_manifest: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(launch_prep_manifest)
    return {str(record["name"]): dict(record) for record in payload.get("records", [])}


def classify_analysis_pair_manifest(
    *,
    pair_manifest_path: Path,
    launch_prep_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    pair_manifest = read_json(pair_manifest_path)
    records_by_name = _load_launch_records(launch_prep_manifest)
    ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for pair in pair_manifest.get("pairs", []):
        prompt_id = str(pair["prompt_id"])
        for label in ("correct", "wrong"):
            side = pair[label]
            name = str(side["name"])
            launch_record = records_by_name[name]
            out_path = output_dir / f"{name}.roles.json"
            payload = write_role_classification(
                trajectory_path=Path(launch_record["trajectory"]),
                output_path=out_path,
                name=name,
                prompt_id=prompt_id,
                label=label,
                run_root=Path(side["run_root"]),
            )
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "label": label,
                    "name": name,
                    "trajectory_path": launch_record["trajectory"],
                    "run_root": side["run_root"],
                    "classification_path": str(out_path),
                    "token_count": payload["token_count"],
                    "role_cluster_counts": payload["role_cluster_counts"],
                }
            )
    catalog = {
        "schema_version": ROLE_SCHEMA_VERSION,
        "classification_kind": "heuristic_token_role_catalog_v1",
        "created_at": _now(),
        "source_pair_manifest": str(pair_manifest_path),
        "launch_prep_manifest": str(launch_prep_manifest),
        "output_dir": str(output_dir),
        "record_count": len(rows),
        "records": rows,
        "aggregate_role_cluster_counts": dict(
            sorted(
                sum(
                    (Counter(row["role_cluster_counts"]) for row in rows), Counter()
                ).items()
            )
        ),
    }
    write_json(output_dir / "role_classification_catalog.json", catalog)
    write_jsonl(output_dir / "role_classification_catalog.jsonl", rows)
    return catalog


def _classification_by_prompt_label(
    catalog_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    catalog = read_json(catalog_path)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for record in catalog["records"]:
        payload = read_json(Path(record["classification_path"]))
        out[(str(record["prompt_id"]), str(record["label"]))] = {
            "record": record,
            "payload": payload,
        }
    return out


def _graphs_by_index(run_root: Path) -> dict[int, Path]:
    return {
        int(path.parent.name.removeprefix("token_")): path
        for path in discover_graph_paths(run_root)
    }


def _tokens_by_role(
    payload: Mapping[str, Any], roles: set[str]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for token in payload["tokens"]:
        role = str(token["role_cluster"])
        if role in roles:
            grouped[role].append(token)
    return grouped


def _sample_items(items: list[Any], max_count: int | None) -> list[Any]:
    if max_count is None or max_count <= 0 or len(items) <= max_count:
        return items
    positions = np_unique_linspace_indices(len(items), max_count)
    return [items[index] for index in positions]


def np_unique_linspace_indices(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    if count <= 1:
        return [0]
    return sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})


def build_role_matched_calibration_manifest(
    *,
    classification_catalog: Path,
    output_path: Path,
    prompt_ids: list[str] | None = None,
    role_clusters: list[str] | None = None,
    max_temporal_pairs_per_role: int = 40,
    max_null_pairs_per_role: int = 30,
    max_noise_pairs_per_role: int = 30,
) -> dict[str, Any]:
    roles = set(role_clusters or DEFAULT_ROLE_CLUSTERS)
    by_prompt_label = _classification_by_prompt_label(classification_catalog)
    all_prompt_ids = sorted({prompt_id for prompt_id, _label in by_prompt_label})
    selected_prompt_ids = [str(x) for x in (prompt_ids or all_prompt_ids)]
    graph_cache: dict[str, dict[int, Path]] = {}

    def graphs_for(run_root: str) -> dict[int, Path]:
        if run_root not in graph_cache:
            graph_cache[run_root] = _graphs_by_index(Path(run_root))
        return graph_cache[run_root]

    pairs: list[dict[str, Any]] = []

    for prompt_id in selected_prompt_ids:
        for label in ("correct", "wrong"):
            entry = by_prompt_label[(prompt_id, label)]
            payload = entry["payload"]
            record = entry["record"]
            grouped = _tokens_by_role(payload, roles)
            graphs = graphs_for(str(record["run_root"]))
            for role, tokens in sorted(grouped.items()):
                noise_candidates = [
                    token for token in tokens if int(token["generated_index"]) in graphs
                ]
                for token in _sample_items(noise_candidates, max_noise_pairs_per_role):
                    token_i = int(token["generated_index"])
                    pairs.append(
                        {
                            "pair_id": f"role_noise_identity:{prompt_id}:{label}:{role}:{token_i}",
                            "left_graph": str(graphs[token_i]),
                            "right_graph": str(graphs[token_i]),
                            "pair_category": "noise",
                            "sub_category": "identity_self_same_role",
                            "fixture": prompt_id,
                            "trajectory": str(record["name"]),
                            "position_band": position_band_for_index(
                                token_i, int(record["token_count"])
                            ),
                            "generated_index": token_i,
                            "generated_index_a": token_i,
                            "generated_index_b": token_i,
                            "lag": 0,
                            "metadata": {
                                "role_cluster": role,
                                "left_role": token["role"],
                                "right_role": token["role"],
                                "prompt_id": prompt_id,
                                "answer_label": label,
                                "pair_design": "role_matched_identity_noise_ceiling",
                                "noise_anchor_kind": "identity_self_pair",
                            },
                        }
                    )
                candidates = [
                    (left, right)
                    for left, right in zip(tokens, tokens[1:])
                    if int(left["generated_index"]) in graphs
                    and int(right["generated_index"]) in graphs
                ]
                for left, right in _sample_items(
                    candidates, max_temporal_pairs_per_role
                ):
                    left_i = int(left["generated_index"])
                    right_i = int(right["generated_index"])
                    pairs.append(
                        {
                            "pair_id": f"role_temporal:{prompt_id}:{label}:{role}:{left_i}->{right_i}",
                            "left_graph": str(graphs[left_i]),
                            "right_graph": str(graphs[right_i]),
                            "pair_category": "temporal",
                            "sub_category": "role_matched_within_trajectory",
                            "fixture": prompt_id,
                            "trajectory": str(record["name"]),
                            "position_band": position_band_for_index(
                                left_i, int(record["token_count"])
                            ),
                            "generated_index": left_i,
                            "generated_index_a": left_i,
                            "generated_index_b": right_i,
                            "lag": right_i - left_i,
                            "metadata": {
                                "role_cluster": role,
                                "left_role": left["role"],
                                "right_role": right["role"],
                                "prompt_id": prompt_id,
                                "answer_label": label,
                                "pair_design": "role_matched_temporal",
                            },
                        }
                    )

    # Same-prompt correct-vs-wrong role-matched nulls.
    for prompt_id in selected_prompt_ids:
        if (prompt_id, "correct") not in by_prompt_label or (
            prompt_id,
            "wrong",
        ) not in by_prompt_label:
            continue
        left_entry = by_prompt_label[(prompt_id, "correct")]
        right_entry = by_prompt_label[(prompt_id, "wrong")]
        left_grouped = _tokens_by_role(left_entry["payload"], roles)
        right_grouped = _tokens_by_role(right_entry["payload"], roles)
        left_graphs = graphs_for(str(left_entry["record"]["run_root"]))
        right_graphs = graphs_for(str(right_entry["record"]["run_root"]))
        for role in sorted(set(left_grouped) & set(right_grouped)):
            count = min(len(left_grouped[role]), len(right_grouped[role]))
            candidates = []
            for rank in range(count):
                left = left_grouped[role][rank]
                right = right_grouped[role][rank]
                if (
                    int(left["generated_index"]) in left_graphs
                    and int(right["generated_index"]) in right_graphs
                ):
                    candidates.append((rank, left, right))
            for rank, left, right in _sample_items(candidates, max_null_pairs_per_role):
                left_i = int(left["generated_index"])
                right_i = int(right["generated_index"])
                pairs.append(
                    {
                        "pair_id": f"role_null_same_prompt:{prompt_id}:{role}:{rank:03d}",
                        "left_graph": str(left_graphs[left_i]),
                        "right_graph": str(right_graphs[right_i]),
                        "pair_category": "null",
                        "sub_category": "same_prompt_correct_wrong_same_role",
                        "fixture": prompt_id,
                        "trajectory": f"{prompt_id}:correct_vs_wrong:{role}",
                        "position_band": position_band_for_index(
                            left_i, int(left_entry["record"]["token_count"])
                        ),
                        "generated_index": left_i,
                        "generated_index_a": left_i,
                        "generated_index_b": right_i,
                        "lag": None,
                        "metadata": {
                            "role_cluster": role,
                            "left_role": left["role"],
                            "right_role": right["role"],
                            "prompt_id": prompt_id,
                            "pair_design": "role_matched_same_prompt_null",
                        },
                    }
                )

    manifest = {
        "schema_version": ROLE_SCHEMA_VERSION,
        "status": "role_matched_calibration_manifest",
        "created_at": _now(),
        "description": "Explicit calibration pairs matched by heuristic generated-token role cluster. Temporal pairs are consecutive tokens within the same role cluster, not necessarily adjacent generated positions. Null pairs are same-prompt correct-vs-wrong tokens matched by role-rank.",
        "classification_catalog": str(classification_catalog),
        "prompt_ids": selected_prompt_ids,
        "role_clusters": sorted(roles),
        "noise_anchor_kind": "identity_self_pair",
        "max_pairs_per_role": {
            "temporal": max_temporal_pairs_per_role,
            "null": max_null_pairs_per_role,
            "noise": max_noise_pairs_per_role,
        },
        "pair_count": len(pairs),
        "pair_counts_by_category": dict(
            Counter(pair["pair_category"] for pair in pairs)
        ),
        "pair_counts_by_role": dict(
            sorted(Counter(pair["metadata"]["role_cluster"] for pair in pairs).items())
        ),
        "pairs": pairs,
    }
    ensure_dir(output_path.parent)
    write_json(output_path, manifest)
    return manifest


def apply_role_classification_reviews(
    *,
    classification_catalog: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Apply `*.roles.review.json` sidecars to copied classification files.

    The original heuristic files are left untouched. Corrections are expected to
    specify `generated_index` and `suggested_role_cluster`; when a correction is
    applied, the token's `role_cluster` is updated and `role` is mapped to a
    canonical role for that cluster.
    """

    catalog = read_json(classification_catalog)
    ensure_dir(output_dir)
    out_records: list[dict[str, Any]] = []
    total_corrections = 0
    missing_reviews: list[str] = []
    for record in catalog["records"]:
        source_path = Path(record["classification_path"])
        source = read_json(source_path)
        review_path = source_path.with_name(
            source_path.name.removesuffix(".roles.json") + ".roles.review.json"
        )
        corrections_by_index: dict[int, dict[str, Any]] = {}
        if review_path.exists():
            review = read_json(review_path)
            for correction in review.get("corrections", []):
                try:
                    index = int(correction["generated_index"])
                except (KeyError, TypeError, ValueError):
                    continue
                corrections_by_index[index] = correction
        else:
            missing_reviews.append(str(review_path))

        applied = 0
        for token in source.get("tokens", []):
            correction = corrections_by_index.get(int(token["generated_index"]))
            if correction is None:
                continue
            suggested = str(correction.get("suggested_role_cluster", ""))
            if suggested not in ROLE_FOR_CLUSTER:
                continue
            token["review_original_role"] = token.get("role")
            token["review_original_role_cluster"] = token.get("role_cluster")
            token["role_cluster"] = suggested
            token["role"] = ROLE_FOR_CLUSTER[suggested]
            tags = list(token.get("tags", []))
            if "subagent_review_corrected" not in tags:
                tags.append("subagent_review_corrected")
            token["tags"] = tags
            token["review_reason"] = correction.get("reason")
            applied += 1

        source["classification_kind"] = "subagent_reviewed_token_role_v1"
        source["source_classification_path"] = str(source_path)
        source["review_path"] = str(review_path) if review_path.exists() else None
        source["review_applied_correction_count"] = applied
        source["reviewed_at"] = _now()
        source["role_cluster_counts"] = dict(
            sorted(Counter(token["role_cluster"] for token in source["tokens"]).items())
        )
        source["role_counts"] = dict(
            sorted(Counter(token["role"] for token in source["tokens"]).items())
        )
        out_path = output_dir / source_path.name.replace(
            ".roles.json", ".reviewed.roles.json"
        )
        write_json(out_path, source)
        new_record = dict(record)
        new_record["classification_path"] = str(out_path)
        new_record["source_classification_path"] = str(source_path)
        new_record["review_path"] = str(review_path) if review_path.exists() else None
        new_record["review_applied_correction_count"] = applied
        new_record["role_cluster_counts"] = source["role_cluster_counts"]
        out_records.append(new_record)
        total_corrections += applied

    out_catalog = {
        "schema_version": ROLE_SCHEMA_VERSION,
        "classification_kind": "subagent_reviewed_token_role_catalog_v1",
        "created_at": _now(),
        "source_classification_catalog": str(classification_catalog),
        "output_dir": str(output_dir),
        "record_count": len(out_records),
        "total_applied_corrections": total_corrections,
        "missing_review_count": len(missing_reviews),
        "missing_reviews": missing_reviews,
        "records": out_records,
        "aggregate_role_cluster_counts": dict(
            sorted(
                sum(
                    (Counter(row["role_cluster_counts"]) for row in out_records),
                    Counter(),
                ).items()
            )
        ),
    }
    write_json(output_dir / "role_classification_catalog.json", out_catalog)
    write_jsonl(output_dir / "role_classification_catalog.jsonl", out_records)
    return out_catalog
