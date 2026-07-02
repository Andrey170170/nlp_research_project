from __future__ import annotations

import csv
import json
from pathlib import Path

from nlp_research_project.circuit_stability_analysis.cli import main
from nlp_research_project.circuit_stability_analysis.inventory import (
    reconstruct_token_offsets,
)
from nlp_research_project.circuit_stability_analysis.roles import classify_trajectory


def _rows(tokens: list[str], name: str = "traj") -> list[dict[str, str]]:
    return [
        {
            "wave": "w",
            "prompt_id": "1",
            "fixture": "1_base",
            "label": "correct",
            "trajectory_name": name,
            "trajectory_id": name,
            "generated_index": str(i),
            "target_token_text": token,
        }
        for i, token in enumerate(tokens)
    ]


def _classify(tokens: list[str]) -> dict[int, dict[str, str]]:
    grouped = reconstruct_token_offsets(_rows(tokens))
    roles = classify_trajectory(next(iter(grouped.values())))
    return {int(r["generated_index"]): r for r in roles}


def test_final_answer_marker_prefers_later_final_answer_over_answer_paragraph() -> None:
    roles = _classify(
        [
            "**",
            "Answer",
            ":",
            "**",
            " 5",
            ".",
            "\n\n",
            "Final",
            " answer",
            ":",
            " 6",
            "<end_of_turn>",
        ]
    )

    assert roles[1]["region"] != "answer_statement"
    assert roles[7]["region"] == "answer_statement"
    assert roles[10]["token_group"] == "answer_quantity"
    assert roles[11]["region"] == "stop"


def test_whitespace_and_newline_are_distinct_controls() -> None:
    roles = _classify([" ", "\n\n", "Text"])

    assert roles[0]["token_group"] == "format_marker"
    assert roles[0]["analysis_group"] == "excluded_whitespace"
    assert "pure_horizontal_whitespace" in roles[0]["tags"]
    assert roles[1]["analysis_group"] == "formatting_control"
    assert "structural_newline" in roles[1]["tags"]


def test_math_punctuation_vs_sentence_punctuation() -> None:
    roles = _classify(["Compute", " 1", " +", " 2", " =", " 3", ".", "5", " Done", "."])

    assert roles[6]["token_group"] == "equation_punctuation"
    assert roles[9]["token_group"] == "sentence_punctuation"


def test_bold_numbered_header_and_latex_dollar_delimiters() -> None:
    roles = _classify(
        [
            "**",
            "1",
            ".",
            " Calculate",
            ":**",
            "\n",
            "Total",
            " is",
            " $",
            "2",
            " +",
            " ",
            "3",
            " =",
            " ",
            "5",
            "$",
            ".",
            "\nFinal",
            " answer",
            ":",
            " 5",
            "<end_of_turn>",
        ]
    )

    assert roles[1]["token_group"] == "format_marker"
    assert roles[2]["token_group"] == "format_marker"
    assert roles[8]["token_group"] == "equation_punctuation"
    assert roles[16]["token_group"] == "equation_punctuation"


def test_cli_writes_roles_summary_and_review_packet(tmp_path: Path) -> None:
    tokens = [
        "Let",
        "'",
        "s",
        " compute",
        " 2",
        " +",
        " 3",
        " =",
        " 5",
        ".",
        "\n",
        "Final",
        " answer",
        ":",
        " 5",
        "<end_of_turn>",
    ]
    token_csv = tmp_path / "tokens.csv"
    with token_csv.open("w", newline="") as handle:
        fields = list(_rows(tokens)[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_rows(tokens, name="486_correct_seed3003_attempt000004"))

    texts = tmp_path / "trajectory_texts.jsonl"
    texts.write_text(
        json.dumps(
            {
                "wave": "w",
                "prompt_id": "1",
                "fixture": "1_base",
                "label": "correct",
                "name": "486_correct_seed3003_attempt000004",
                "generated_text": "".join(tokens),
            }
        )
        + "\n"
    )

    main(
        [
            "generate-roles",
            "--tokens",
            str(token_csv),
            "--trajectory-texts",
            str(texts),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    roles_path = tmp_path / "out" / "roles" / "roles_regex.csv"
    summary_path = tmp_path / "out" / "roles" / "roles_regex_summary.json"
    packet_path = tmp_path / "out" / "roles" / "roles_regex_review_packet.md"
    assert roles_path.exists()
    assert packet_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["row_count_equals_token_count"] is True
    assert summary["exactly_one_role_per_token"] is True
    assert summary["stop_count"] == 1
    assert "analysis_group_counts" in summary
    assert "review_priority_counts" in summary

    main(["freeze-roles-v1", "--output-dir", str(tmp_path / "out")])
    assert (tmp_path / "out" / "roles" / "roles_v1.csv").exists()

    main(
        [
            "freeze-roles-v1",
            "--output-dir",
            str(tmp_path / "out"),
            "--make-review-iter01",
        ]
    )
    reviewed_path = tmp_path / "out" / "roles" / "roles_review_iter01.csv"
    main(
        [
            "freeze-roles-v1",
            "--output-dir",
            str(tmp_path / "out"),
            "--reviewed",
            str(reviewed_path),
        ]
    )
    reviewed_summary = json.loads(
        (tmp_path / "out" / "roles" / "roles_v1_summary.json").read_text()
    )
    assert reviewed_summary["row_count"] == len(tokens)
