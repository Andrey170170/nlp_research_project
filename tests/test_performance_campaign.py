from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench import perf_cli
from nlp_research_project.exact_trace_bench.performance_campaign import (
    freeze_performance_campaign,
    validate_mechanism_claims,
    validate_performance_campaign,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIMS = REPO_ROOT / "experiments/performance_campaigns/sp0_mechanism_claims.json"
CAMPAIGN = REPO_ROOT / "experiments/performance_campaigns/ls0_prefix_scaling.json"


def test_committed_sp0_claims_validate() -> None:
    payload = json.loads(CLAIMS.read_text())
    validate_mechanism_claims(payload)
    statuses = {claim["disposition"]["status"] for claim in payload["claims"]}
    assert statuses == {
        "exact_promotion_candidate",
        "exact_opt_in",
        "bounded_research",
        "rejected",
    }


def test_committed_ls0_scaffold_lists_without_model_loading() -> None:
    payload = json.loads(CAMPAIGN.read_text())
    workloads = validate_performance_campaign(payload)
    assert [workload.requested_prefix_tokens for workload in workloads] == [
        129,
        256,
        512,
        1024,
        256,
        256,
        512,
        512,
    ]
    assert [workload.prompt_role for workload in workloads] == [
        "development",
        "development",
        "development",
        "development",
        "holdout",
        "holdout",
        "holdout",
        "holdout",
    ]


def test_ls0_development_and_holdout_workloads_are_frozen() -> None:
    payload = json.loads(CAMPAIGN.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)
    assert {workload.prompt_role for workload in workloads} == {
        "development",
        "holdout",
    }


def test_frozen_workload_requires_immutable_hashes() -> None:
    payload = json.loads(CAMPAIGN.read_text())
    payload["workloads"][0]["fixture"]["catalog_sha256"] = None
    with pytest.raises(ValueError, match="catalog_sha256"):
        validate_performance_campaign(payload)


def test_perf_cli_campaign_and_claim_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert perf_cli.main(["validate-claims", str(CLAIMS)]) == 0
    assert "validated mechanism claims" in capsys.readouterr().out
    assert perf_cli.main(["list-campaign", str(CAMPAIGN)]) == 0
    output = capsys.readouterr().out
    assert "ls0-361-length-scaling-v1: 8 workloads" in output
    assert "ls0-361-dev-1024" in output
    assert "ls0-828-holdout-512" in output
    assert "ls0-94-holdout-512" in output


@pytest.mark.parametrize("fixture", ["828", "94"])
def test_long_holdout_trajectory_extends_frozen_256_prefix(fixture: str) -> None:
    root = REPO_ROOT / "experiments/generated/performance_campaigns/ls0"
    short = json.loads((root / f"{fixture}_trajectory.json").read_text())
    long = json.loads((root / f"{fixture}_long_trajectory.json").read_text())
    assert long["prompt_token_ids"] == short["prompt_token_ids"]
    assert long["generated_tokens"][: len(short["generated_tokens"])] == short[
        "generated_tokens"
    ]
    assert len(long["prompt_token_ids"]) + len(long["generated_tokens"]) > 512


def test_freeze_campaign_derives_prefix_and_target_fingerprints(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    catalog_path = tmp_path / "catalog.json"
    trajectory_path = tmp_path / "trajectory.json"
    manifest_path = tmp_path / "campaign.json"
    prompt_path.write_text("prompt", encoding="utf-8")
    catalog_path.write_text('{"schema_version": 1}', encoding="utf-8")
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trajectory_id": "trajectory-v1",
                "prompt_token_count": 2,
                "prompt_token_ids": [10, 11],
                "prompt_text": "prompt",
                "generated_tokens": [
                    {
                        "generated_index": index,
                        "absolute_token_position": index + 2,
                        "token_id": 20 + index,
                        "token_text": f"t{index}",
                        "is_stop": False,
                    }
                    for index in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )
    workload = json.loads(CAMPAIGN.read_text())["workloads"][0]
    workload["workload_id"] = "freeze-test"
    workload["preparation_status"] = "planned"
    workload["fixture"] = {
        "catalog": "catalog.json",
        "catalog_sha256": None,
        "fixture_name": "test",
        "prompt_file": "prompt.txt",
        "prompt_sha256": None,
    }
    workload["trajectory"] = {
        "trajectory_id": "trajectory-v1",
        "path": "trajectory.json",
        "sha256": None,
    }
    workload["prefix"] = {
        "requested_tokens": 4,
        "actual_tokens": None,
        "token_ids_sha256": None,
    }
    workload["target"] = {
        "absolute_position": None,
        "token_id": None,
        "token_text": None,
    }
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "freeze-test-v1",
                "workloads": [workload],
            }
        ),
        encoding="utf-8",
    )

    frozen = freeze_performance_campaign(manifest_path, repo_root=tmp_path)

    frozen_workload = frozen["workloads"][0]
    assert frozen_workload["preparation_status"] == "frozen"
    assert frozen_workload["prefix"]["actual_tokens"] == 4
    assert len(frozen_workload["prefix"]["token_ids_sha256"]) == 64
    assert frozen_workload["target"] == {
        "absolute_position": 4,
        "token_id": 22,
        "token_text": "t2",
    }
    assert validate_performance_campaign(frozen, require_frozen=True)
