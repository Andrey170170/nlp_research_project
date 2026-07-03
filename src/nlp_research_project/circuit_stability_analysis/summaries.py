from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as h:
        return [dict(r) for r in csv.DictReader(h)]


def _num(v: str) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for r in rows for k in r}) if rows else ["empty"]
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _metric_column(name: str) -> bool:
    return (
        name.endswith("weighted_jaccard")
        or name.endswith("cosine")
        or (name.startswith("decoder_soft_") and name != "decoder_soft_available")
        or name.startswith("decode_errors")
    )


def _mean(vals: list[float]) -> float | None:
    return None if not vals else sum(vals) / len(vals)


def summarize_metrics(metric_rows: Path, output_dir: Path) -> None:
    rows = _rows(metric_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = [k for k in (rows[0].keys() if rows else []) if _metric_column(k)]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = (
            r.get("prompt_id", ""),
            r.get("pair_category", ""),
            r.get("pair_subcategory", ""),
            f"{r.get('label_a', '')}|{r.get('label_b', '')}",
            f"{r.get('region_a', '')}|{r.get('region_b', '')}",
            f"{r.get('analysis_group_a', '')}|{r.get('analysis_group_b', '')}",
        )
        groups[key].append(r)
    out = []
    for key, vals in sorted(groups.items()):
        rec: dict[str, Any] = {
            "prompt_id": key[0],
            "pair_category": key[1],
            "pair_subcategory": key[2],
            "label_pair": key[3],
            "region_pair": key[4],
            "analysis_group_pair": key[5],
            "pair_count": len(vals),
        }
        for n in numeric:
            xs = [_num(v.get(n, "")) for v in vals]
            xs2 = [x for x in xs if x is not None]
            if xs2:
                rec[f"mean_{n}"] = sum(xs2) / len(xs2)
        out.append(rec)
    _write(output_dir / "prompt_metric_summary.csv", out)

    agg_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for r in out:
        key = (
            r.get("pair_category", ""),
            r.get("pair_subcategory", ""),
            r.get("label_pair", ""),
            r.get("region_pair", ""),
            r.get("analysis_group_pair", ""),
        )
        agg_groups[key].append(r)
    agg_out: list[dict[str, Any]] = []
    for key, vals in sorted(agg_groups.items()):
        rec: dict[str, Any] = {
            "pair_category": key[0],
            "pair_subcategory": key[1],
            "label_pair": key[2],
            "region_pair": key[3],
            "analysis_group_pair": key[4],
            "prompt_count": len(vals),
            "total_pair_count": sum(int(v.get("pair_count", 0) or 0) for v in vals),
        }
        for n in numeric:
            xs = [_num(v.get(f"mean_{n}", "")) for v in vals]
            xs2 = [x for x in xs if x is not None]
            if xs2:
                rec[f"mean_{n}"] = sum(xs2) / len(xs2)
        agg_out.append(rec)
    _write(output_dir / "aggregate_metric_summary.csv", agg_out)

    tg: dict[tuple[str, str, str], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        tg[
            (
                r.get("prompt_id", ""),
                r.get("pair_category", ""),
                r.get("pair_subcategory", ""),
            )
        ][r.get("label_a", "")].append(r)
    deltas = []
    for key, by_label in sorted(tg.items()):
        if "correct" not in by_label or "wrong" not in by_label:
            continue
        rec: dict[str, Any] = {
            "prompt_id": key[0],
            "pair_category": key[1],
            "pair_subcategory": key[2],
        }
        for n in numeric:
            cs = [
                x
                for x in (_num(r.get(n, "")) for r in by_label["correct"])
                if x is not None
            ]
            ws = [
                x
                for x in (_num(r.get(n, "")) for r in by_label["wrong"])
                if x is not None
            ]
            if cs and ws:
                rec[f"delta_{n}"] = sum(cs) / len(cs) - sum(ws) / len(ws)
        deltas.append(rec)
    _write(output_dir / "correct_wrong_delta.csv", deltas)

    agg_delta_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in deltas:
        agg_delta_groups[
            (r.get("pair_category", ""), r.get("pair_subcategory", ""))
        ].append(r)
    agg_deltas: list[dict[str, Any]] = []
    for key, vals in sorted(agg_delta_groups.items()):
        rec: dict[str, Any] = {
            "pair_category": key[0],
            "pair_subcategory": key[1],
            "prompt_count": len(vals),
        }
        for n in numeric:
            xs = [_num(v.get(f"delta_{n}", "")) for v in vals]
            xs2 = [x for x in xs if x is not None]
            if xs2:
                rec[f"delta_{n}"] = sum(xs2) / len(xs2)
        agg_deltas.append(rec)
    _write(output_dir / "aggregate_correct_wrong_delta.csv", agg_deltas)
    versions = {
        "metric_versions": sorted({r.get("metric_version", "") for r in rows}),
        "role_versions": sorted({r.get("role_version", "") for r in rows}),
        "cluster_versions": sorted({r.get("cluster_version", "") for r in rows}),
    }
    (output_dir / "summary_audit.json").write_text(
        json.dumps(
            {"row_count": len(rows), "summary_group_count": len(out), **versions},
            indent=2,
            sort_keys=True,
        )
    )
