from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROLE_FIELDS = (
    "region",
    "span_type",
    "token_group",
    "fine_role",
    "coarse_role",
    "tags",
    "confidence",
    "role_confidence",
    "role_version",
    "analysis_group",
)
KEY_FIELDS = ("wave", "prompt_id", "label", "generated_index")


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="") as h:
        return [dict(r) for r in csv.DictReader(h)]


def _frac(row: dict[str, Any], name: str, default: float) -> float:
    try:
        return float(row.get(name, "") or default)
    except ValueError:
        return default


def _text(row: dict[str, Any]) -> str:
    return str(row.get("target_token_text") or row.get("token_text") or "")


def _gfrac(row: dict[str, Any]) -> float | None:
    for name in ("generated_fraction_start", "global_frac"):
        val = row.get(name, "")
        if val != "":
            try:
                return float(val)
            except ValueError:
                continue
    return None


def _key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row.get("wave", "")),
        str(row.get("prompt_id", "")),
        str(row.get("label", "")),
        str(row.get("generated_index", "")),
    )


def _pair(
    pid: int,
    cat: str,
    sub: str,
    a: dict[str, Any],
    b: dict[str, Any],
    role_version: str,
    cluster_version: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pair_id": f"pair_{pid:08d}",
        "pair_category": cat,
        "pair_subcategory": sub,
        "role_version": role_version,
        "cluster_version": cluster_version,
        "wave": a.get("wave", ""),
        "prompt_id": a.get("prompt_id", ""),
        "label_a": a.get("label", ""),
        "label_b": b.get("label", ""),
        "trajectory_name_a": a.get("trajectory_name", ""),
        "trajectory_name_b": b.get("trajectory_name", ""),
        "generated_index_a": a.get("generated_index", ""),
        "generated_index_b": b.get("generated_index", ""),
        "token_text_a": a.get("token_text", ""),
        "token_text_b": b.get("token_text", ""),
        "graph_path_a": a.get("graph_path", ""),
        "graph_path_b": b.get("graph_path", ""),
        "global_frac_a": a.get("global_frac", ""),
        "global_frac_b": b.get("global_frac", ""),
        "region_frac_a": a.get("region_frac", ""),
        "region_frac_b": b.get("region_frac", ""),
    }
    for f in ROLE_FIELDS:
        row[f"{f}_a"] = a.get(f, "")
        row[f"{f}_b"] = b.get(f, "")
    return row


def build_pair_manifest(
    tokens: Path,
    output: Path,
    *,
    roles: Path | None = None,
    role_version: str = "unknown",
    cluster_version: str = "unknown",
    lags: list[int] | None = None,
    final_window: int = 10,
    null_cap_per_trajectory: int = 20,
) -> dict[str, Any]:
    lags = sorted(set(lags or [1, 2, 5, 10, 20]))
    if roles is not None and not roles.exists():
        raise FileNotFoundError(f"missing roles csv: {roles}")
    role_rows = _read_csv(roles)
    role_by_key = {_key(r): r for r in role_rows}
    rows = []
    for r in _read_csv(tokens):
        merged = {**r, **role_by_key.get(_key(r), {})}
        merged["token_text"] = _text(merged)
        merged["_idx"] = int(merged.get("generated_index", 0) or 0)
        rows.append(merged)
    rows.sort(
        key=lambda r: (
            r.get("wave", ""),
            r.get("prompt_id", ""),
            r.get("label", ""),
            r.get("trajectory_name", ""),
            r["_idx"],
        )
    )
    by_traj: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_traj[
            (
                str(r.get("wave", "")),
                str(r.get("prompt_id", "")),
                str(r.get("label", "")),
                str(r.get("trajectory_name", "")),
            )
        ].append(r)
    for seq in by_traj.values():
        seq.sort(key=lambda r: r["_idx"])
        total = len(seq)
        for i, r in enumerate(seq):
            gfrac = _gfrac(r)
            r["global_frac"] = (
                gfrac if gfrac is not None else (i / (total - 1) if total > 1 else 0.0)
            )
            if role_by_key and r.get("region"):
                seg_start = i
                seg_end = i
                while seg_start > 0 and seq[seg_start - 1].get("region", "") == r.get(
                    "region", ""
                ):
                    seg_start -= 1
                while seg_end + 1 < total and seq[seg_end + 1].get(
                    "region", ""
                ) == r.get("region", ""):
                    seg_end += 1
                seg_len = seg_end - seg_start + 1
                r["region_frac"] = (
                    (i - seg_start) / (seg_len - 1) if seg_len > 1 else 0.0
                )
            elif role_by_key:
                r["region_frac"] = ""
            else:
                r["region_frac"] = ""
    for r in rows:
        if "global_frac" not in r:
            r["global_frac"] = ""
        if "region_frac" not in r:
            r["region_frac"] = ""
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    def add(cat: str, sub: str, a: dict[str, Any], b: dict[str, Any]) -> None:
        sig = (cat, sub, str(_key(a)), str(_key(b)), a.get("trajectory_name", ""))
        if sig in seen:
            return
        seen.add(sig)
        pairs.append(_pair(len(pairs), cat, sub, a, b, role_version, cluster_version))

    for seq in by_traj.values():
        seq.sort(key=lambda r: r["_idx"])
        pos = {r["_idx"]: r for r in seq}
        for a in seq:
            for lag in lags:
                b = pos.get(a["_idx"] + lag)
                if not b:
                    continue
                add("temporal", "adjacent" if lag == 1 else f"lag_{lag}", a, b)
                if role_by_key:
                    for f in ("region", "analysis_group", "token_group", "fine_role"):
                        if a.get(f) and a.get(f) == b.get(f):
                            add(
                                "role_temporal",
                                f"same_{f}_{'adjacent' if lag == 1 else f'lag_{lag}'}",
                                a,
                                b,
                            )

    by_wp: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_wp[(str(r.get("wave", "")), str(r.get("prompt_id", "")))].append(r)
    for group in by_wp.values():
        corr = [r for r in group if r.get("label") == "correct"]
        wrong = [r for r in group if r.get("label") == "wrong"]
        for c in corr:
            if wrong:
                w = min(
                    wrong,
                    key=lambda x: (
                        abs(
                            float(x.get("global_frac", 0.0))
                            - float(c.get("global_frac", 0.0))
                        ),
                        x["_idx"],
                    ),
                )
                add("correct_wrong", "relative_position", c, w)
        if final_window > 0:
            corr_tail = sorted(corr, key=lambda r: r["_idx"])[-final_window:]
            wrong_tail = sorted(wrong, key=lambda r: r["_idx"])[-final_window:]
            for c, w in zip(corr_tail, wrong_tail):
                add("correct_wrong", f"final_window_{final_window}", c, w)

    if null_cap_per_trajectory > 0:
        by_wl: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_wl[(str(r.get("wave", "")), str(r.get("label", "")))].append(r)
        counts: Counter[str] = Counter()
        for a in rows:
            key = (str(a.get("wave", "")), str(a.get("label", "")))
            candidates = [
                r for r in by_wl[key] if r.get("prompt_id") != a.get("prompt_id")
            ]
            same_role_field = "region" if a.get("region") else "analysis_group"
            same_role = [
                r
                for r in candidates
                if a.get(same_role_field)
                and r.get(same_role_field) == a.get(same_role_field)
            ]
            traj = str(a.get("trajectory_name", ""))
            if counts[traj] >= null_cap_per_trajectory:
                continue
            chosen = same_role or candidates
            if chosen:
                b = min(
                    chosen,
                    key=lambda x: (
                        abs(
                            float(x.get("global_frac", 0.0))
                            - float(a.get("global_frac", 0.0))
                        ),
                        x.get("prompt_id", ""),
                        x["_idx"],
                    ),
                )
                add(
                    "null_control",
                    "cross_prompt_same_role_nearest_fraction"
                    if same_role
                    else "cross_prompt_nearest_fraction",
                    a,
                    b,
                )
                counts[traj] += 1

    pair_counts = Counter((p["pair_category"], p["pair_subcategory"]) for p in pairs)
    manifest = {
        "metadata": {
            "pair_manifest_version": "pairs_v1",
            "role_version": role_version,
            "cluster_version": cluster_version,
            "token_count": len(rows),
            "pair_count": len(pairs),
        },
        "counts_by_category_subcategory": {
            f"{k[0]}/{k[1]}": v for k, v in sorted(pair_counts.items())
        },
        "role_field_counts": {f: sum(1 for r in rows if r.get(f)) for f in ROLE_FIELDS},
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
