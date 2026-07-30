from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench import perf_cli
from nlp_research_project.exact_trace_bench.performance_campaign import (
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
        124,
        256,
        512,
        1024,
    ]
    assert all(workload.prompt_role == "development" for workload in workloads)


def test_ls0_frozen_gate_fails_closed_while_trajectory_is_planned() -> None:
    payload = json.loads(CAMPAIGN.read_text())
    with pytest.raises(ValueError, match="is not frozen"):
        validate_performance_campaign(payload, require_frozen=True)


def test_frozen_workload_requires_immutable_hashes() -> None:
    payload = json.loads(CAMPAIGN.read_text())
    payload["workloads"][0]["preparation_status"] = "frozen"
    with pytest.raises(ValueError, match="catalog_sha256"):
        validate_performance_campaign(payload)


def test_perf_cli_campaign_and_claim_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert perf_cli.main(["validate-claims", str(CLAIMS)]) == 0
    assert "validated mechanism claims" in capsys.readouterr().out
    assert perf_cli.main(["list-campaign", str(CAMPAIGN)]) == 0
    output = capsys.readouterr().out
    assert "ls0-361-length-scaling-v1: 4 workloads" in output
    assert "ls0-361-dev-1024" in output
