from __future__ import annotations

import json
from pathlib import Path

from nlp_research_project.exact_trace_bench.calibration.response_bundles import (
    publish_response_bundle,
    validate_response_bundle,
)
from nlp_research_project.exact_trace_bench.calibration.response_bundle_adapter import (
    PublicSiblingResponseModelApi,
)
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    CANONICAL_BUCKET_NAMES,
)
from circuit_tracer.governor.response_models import load_response_bundle


class FakeApi:
    def publish(
        self, *, observations: tuple[Path, ...], output: Path
    ) -> dict[str, int]:
        output.write_text("bundle")
        return {"published": len(observations)}

    def validate(self, *, bundle: Path) -> dict[str, bool]:
        return {"valid": bundle.read_text() == "bundle"}


def test_response_bundle_publish_then_validate_uses_adapter(tmp_path: Path) -> None:
    observation = tmp_path / "observation.json"
    observation.write_text("{}")
    bundle = tmp_path / "bundle.json"

    result = publish_response_bundle([observation], output=bundle, api=FakeApi())

    assert result["publish"] == {"published": 1}
    assert result["validation"] == {"valid": True}
    assert validate_response_bundle(bundle, api=FakeApi()) == {"valid": True}


def _observation(sample_id: str, split: str, walltime: float) -> dict[str, object]:
    return {
        "schema_version": 2,
        "observation_id": sample_id,
        "observation_fingerprint": f"fingerprint-{sample_id}",
        "campaign": {"split": split},
        "scope": {
            "governor_profile_name": "granite_h200_4b_plt_b128_c4096_cache0",
            "transcoder_provider_family": "gemmascope2-plt-4b-small",
            "transcoder_architecture": "plt",
        },
        "configuration": {
            "attribution_batch_size": 128,
            "feature_batch_size": 128,
            "logit_batch_size": 128,
            "decoder_chunk_size": 4096,
            "nnsight_session_capacity": 128,
            "phase4_execution_batch_max_rows": 128,
            "feature_row_retention": "full_file",
            "exact_encoder_residency": "lazy",
        },
        "workload": {
            "prompt_token_count": 124,
            "max_active_features": 8192,
            "n_steps_traced": 1,
        },
        "outcome": {"status": "success"},
        "uncertainty": {"censoring": "none"},
        "runtime": {
            "walltime_seconds": walltime,
            "planning": [
                {
                    "selected_objective": [
                        ["predicted_walltime_high_seconds", walltime * 1.1]
                    ],
                    "selected_vector": {"session_capacity": 256},
                }
            ],
        },
        "fidelity": {
            "comparison": {
                "overall_mean_feature_jaccard": 0.999,
                **{
                    f"overall_mean_bucket_{bucket.replace('<-', '_').replace('-', '_')}_support_jaccard": 0.998
                    for bucket in CANONICAL_BUCKET_NAMES
                },
                **{
                    f"overall_mean_bucket_{bucket.replace('<-', '_').replace('-', '_')}_weighted_jaccard": 0.997
                    for bucket in CANONICAL_BUCKET_NAMES
                },
                **{
                    f"overall_mean_bucket_{bucket.replace('<-', '_').replace('-', '_')}_top256_jaccard": 1.0
                    for bucket in CANONICAL_BUCKET_NAMES
                },
            }
        },
        "provenance": {
            "semantic_fingerprints": ["semantic"],
            "execution_fingerprints": [f"execution-{sample_id}"],
        },
    }


def test_public_adapter_fits_deterministic_bundle_and_excludes_heldout(
    tmp_path: Path,
) -> None:
    paths = []
    for index, split in enumerate(("fit", "fit", "fit", "heldout")):
        path = tmp_path / f"observation-{index}.json"
        observation = _observation(f"sample-{index}", split, 10 + index)
        if split == "heldout":
            observation["configuration"]["cross_batch_decoder_cache_bytes"] = 4096
        path.write_text(json.dumps(observation))
        paths.append(path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_result = publish_response_bundle(paths, output=first)
    second_result = publish_response_bundle(reversed(paths), output=second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result["validation"]["diagnostics"]["fit_sample_count"] == 3
    assert first_result["validation"]["diagnostics"]["heldout_ids"] == ["sample-3"]
    assert (
        second_result["validation"]["content_fingerprint"]
        == (first_result["validation"]["content_fingerprint"])
    )
    loaded = load_response_bundle(first)
    assert all(
        "decoder_cache_bytes" not in artifact.numeric_features
        for artifact in loaded.models
    )
    model_targets = {artifact.target for artifact in loaded.models}
    assert {
        "overall_mean_edge_jaccard",
        "overall_mean_weighted_edge_jaccard",
        "overall_mean_top256_edge_jaccard",
    }.isdisjoint(model_targets)
    assert {
        f"overall_mean_bucket_{bucket.replace('<-', '_').replace('-', '_')}_{metric}"
        for bucket in CANONICAL_BUCKET_NAMES
        for metric in ("support_jaccard", "weighted_jaccard", "top256_jaccard")
    }.issubset(model_targets)


def test_adapter_uses_selected_vector_and_runtime_categories(tmp_path: Path) -> None:
    observation = _observation("fit", "fit", 10)
    observation["configuration"]["nnsight_session_capacity"] = 128
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(observation))

    sample = PublicSiblingResponseModelApi()._sample(path)

    assert dict(sample.numeric_coordinates)["session_capacity"] == 256
    assert dict(sample.categorical_coordinates)["row_store_policy"] == (
        "file_backed_full"
    )
    assert dict(sample.categorical_coordinates)["encoder_residency"] == (
        "lazy_per_request"
    )
