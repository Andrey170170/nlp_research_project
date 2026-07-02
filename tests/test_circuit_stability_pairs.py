from __future__ import annotations

import csv
import json
from pathlib import Path

from nlp_research_project.circuit_stability_analysis.cli import main
from nlp_research_project.circuit_stability_analysis.pairs import build_pair_manifest


def _csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as h:
        fieldnames = sorted({k for row in rows for k in row})
        w = csv.DictWriter(h, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_pair_manifest_categories_and_cli(tmp_path: Path) -> None:
    rows = []
    for prompt in ("p1", "p2"):
        for label in ("correct", "wrong"):
            for i in range(4):
                canonical = prompt == "p1"
                rows.append(
                    {
                        "wave": "w",
                        "prompt_id": prompt,
                        "label": label,
                        "trajectory_name": f"{prompt}_{label}",
                        "generated_index": str(i),
                        "target_token_text": f"{prompt}_{label}_tok{i}"
                        if canonical
                        else "",
                        "token_text": "" if canonical else f"{prompt}_{label}_tok{i}",
                        "target_token_id": str(100 + i) if canonical else "",
                        "graph_path": f"g/{prompt}/{label}/{i}.npz",
                        "generated_fraction_start": ""
                        if i in (1, 2) and canonical
                        else f"{i / 3}"
                        if canonical
                        else "",
                        "global_frac": f"{i / 3}" if not canonical else "",
                    }
                )
    tokens = tmp_path / "tokens.csv"
    _csv(tokens, rows)
    roles = tmp_path / "roles.csv"
    _csv(
        roles,
        [
            {
                **r,
                "region": "r" if int(r["generated_index"]) < 2 else "s",
                "analysis_group": "ag",
                "coarse_role": "cr",
                "role_confidence": "0.9",
                "role_version": "roles_v1",
                "token_group": "tg",
                "fine_role": "fr",
                "span_type": "body",
                "tags": "",
                "confidence": "1",
            }
            for r in rows
        ],
    )
    out = tmp_path / "pairs.json"
    m = build_pair_manifest(
        tokens,
        out,
        roles=roles,
        role_version="rv",
        cluster_version="cv",
        lags=[1, 2],
        final_window=2,
        null_cap_per_trajectory=1,
    )
    cats = {(p["pair_category"], p["pair_subcategory"]) for p in m["pairs"]}
    assert ("temporal", "adjacent") in cats
    assert ("temporal", "lag_2") in cats
    assert any(c == "role_temporal" and "same_region" in s for c, s in cats)
    assert ("correct_wrong", "relative_position") in cats
    assert ("correct_wrong", "final_window_2") in cats
    assert ("null_control", "cross_prompt_same_role_nearest_fraction") in cats
    assert m["metadata"]["role_version"] == "rv"
    final_window_pairs = [
        p for p in m["pairs"] if p["pair_subcategory"] == "final_window_2"
    ]
    assert len(final_window_pairs) == 4
    sample = next(
        p
        for p in m["pairs"]
        if p["trajectory_name_a"] == "p1_correct" and p["generated_index_a"] == "1"
    )
    assert sample["token_text_a"] == "p1_correct_tok1"
    assert abs(float(sample["global_frac_a"]) - (1 / 3)) < 1e-9
    assert abs(float(sample["region_frac_a"]) - 1.0) < 1e-9
    assert abs(float(sample["region_frac_b"])) < 1e-9
    try:
        build_pair_manifest(
            tokens,
            tmp_path / "missing_roles.json",
            roles=tmp_path / "does_not_exist.csv",
        )
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass
    main(
        [
            "build-pair-manifest",
            "--tokens",
            str(tokens),
            "--roles",
            str(roles),
            "--output",
            str(tmp_path / "cli.json"),
            "--lags",
            "1,2",
            "--null-cap-per-trajectory",
            "0",
        ]
    )
    assert json.loads((tmp_path / "cli.json").read_text())["pairs"]
