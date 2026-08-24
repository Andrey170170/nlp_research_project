from __future__ import annotations

import ast
import inspect
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circuit_tracer import resolve_trace_request  # noqa: E402

from experiments.run_sparsification_experiment import build_command  # noqa: E402
from nlp_research_project.exact_trace_bench.trace_runtime.generation import (  # noqa: E402
    CompletionPlan,
    trace_completion_compact_chunked,
)
from nlp_research_project.exact_trace_bench.trace_runtime.artifacts import (  # noqa: E402
    CaptureArtifactContractError,
)
from nlp_research_project.exact_trace_bench.trace_runtime.completion_workspace import (  # noqa: E402
    CompletionWorkspace,
)
from nlp_research_project.exact_trace_bench.trace_runtime.request import (  # noqa: E402
    trace_policy_from_scenario,
)
from nlp_research_project.exact_trace_bench.trace_runtime import (  # noqa: E402
    tracing as tracing_module,
)
from nlp_research_project.exact_trace_bench.trace_runtime.tracing import (  # noqa: E402
    DiagnosticTraceCompletion,
    extract_compact_chunked_attribution,
)
from nlp_research_project.exact_trace_bench.trace_runtime.step_artifacts import (  # noqa: E402
    StepArtifactWriter,
)


class FakeModel:
    backend = "nnsight"
    provider_id = "c2-project-test"


def _trace_result(output: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        output=output,
        status="succeeded",
        semantic_fingerprint="semantic-test",
        execution_fingerprint="execution-test",
        telemetry_summary={},
        telemetry_events=(),
        admission_report=None,
    )


def _typed_compact_result(**extras: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "active_features": torch.tensor([[0, 0, 7]], dtype=torch.int64),
        "selected_features": torch.tensor([0], dtype=torch.int64),
        "feature_row_node_indices": torch.tensor([0], dtype=torch.int64),
        "logit_row_node_indices": torch.tensor([0], dtype=torch.int64),
        "feature_feature_edges": torch.tensor([[1.0]]),
        "feature_error_edges": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        "feature_token_edges": torch.tensor([[0.0, 0.0]]),
        "logit_feature_edges": torch.tensor([[0.5]]),
        "logit_error_edges": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        "logit_token_edges": torch.tensor([[0.0, 0.0]]),
        "n_error_nodes": 4,
        "n_token_nodes": 2,
        "input_tokens": torch.tensor([1, 2], dtype=torch.int64),
        "logit_targets": [SimpleNamespace(vocab_idx=9)],
        "semantic_fingerprint": "semantic-test",
        "execution_fingerprint": "execution-test",
    }
    payload.update(extras)
    return payload


def test_multistep_seam_injects_trace_result_fingerprints(monkeypatch) -> None:
    output = _typed_compact_result()
    del output["semantic_fingerprint"]
    del output["execution_fingerprint"]
    monkeypatch.setattr(
        tracing_module, "trace_one", lambda *_args, **_kwargs: _trace_result(output)
    )

    compact = extract_compact_chunked_attribution(
        FakeModel(),
        [1, 2],
        policy=trace_policy_from_scenario({"method": "exact"}),
    )

    assert isinstance(compact, dict)
    assert compact["semantic_fingerprint"] == "semantic-test"
    assert compact["execution_fingerprint"] == "execution-test"
    assert "semantic_fingerprint" not in output
    assert "execution_fingerprint" not in output


def test_multistep_seam_rejects_output_fingerprint_conflict(monkeypatch) -> None:
    output = _typed_compact_result(semantic_fingerprint="untrusted-output-value")
    monkeypatch.setattr(
        tracing_module, "trace_one", lambda *_args, **_kwargs: _trace_result(output)
    )

    with pytest.raises(ValueError, match="conflicts with authoritative TraceResult"):
        extract_compact_chunked_attribution(
            FakeModel(),
            [1, 2],
            policy=trace_policy_from_scenario({"method": "exact"}),
        )


def _fingerprints(overrides: dict[str, object]) -> tuple[str, str]:
    policy = trace_policy_from_scenario(
        {
            "name": "fingerprint",
            "method": "exact",
            "attribution_batch_size": 8,
            **overrides,
        }
    )
    plan = resolve_trace_request(policy.request(model=FakeModel(), prompt=[1, 2]))
    return plan.semantic_fingerprint, plan.execution_fingerprint


def test_scenario_maps_explicit_telemetry_enablement() -> None:
    disabled = trace_policy_from_scenario(
        {"name": "telemetry-off", "method": "exact", "telemetry_enabled": False}
    )
    enabled = trace_policy_from_scenario(
        {"name": "telemetry-on", "method": "exact", "telemetry_enabled": True}
    )
    assert disabled.execution.observability.telemetry_enabled is False
    assert enabled.execution.observability.telemetry_enabled is True


def test_semantic_frontier_knobs_change_only_semantic_fingerprint() -> None:
    baseline = _fingerprints({})
    for key, value in (
        ("phase4_refresh_policy", "deferred_v1"),
        ("phase4_refresh_interval_multiplier", 2),
        ("phase4_ranker", "topk_v1"),
        ("phase4_scheduler_mode", "planner_v1"),
        ("phase3_frontier_buffer_relative_epsilon", 0.01),
        ("phase4_frontier_buffer_max_extra_total", 4),
    ):
        changed = _fingerprints({key: value})
        assert changed[0] != baseline[0], key
        assert changed[1] == baseline[1], key


def test_physical_frontier_knobs_change_only_execution_fingerprint() -> None:
    baseline = _fingerprints({})
    for key, value in (
        ("phase4_refresh_optimization", "off"),
        ("phase4_refresh_prepared_chunk_cache_bytes", 1024),
        ("phase4_refresh_active_row_accumulation", "zero_fill"),
        ("phase4_row_executor", "streaming_v1"),
        ("phase4_row_reduction", "off"),
        (
            "feature_vjp_tape_max_bytes",
            1024,
        ),
        ("decoder_page_prefetch_depth", 1),
        ("decoder_active_row_max_bytes", 1024),
        ("decoder_active_row_safety_margin_bytes", 1024),
    ):
        changed = _fingerprints({key: value})
        assert changed[0] == baseline[0], key
        assert changed[1] != baseline[1], key
    changed = _fingerprints(
        {
            "decoder_active_row_residency": True,
            "decoder_active_row_residency_requirement": "required",
            "decoder_active_row_safety_margin_bytes": 1024,
        }
    )
    assert changed[0] == baseline[0]
    assert changed[1] != baseline[1]
    changed = _fingerprints(
        {
            "feature_vjp_tape_batch_window": 2,
            "feature_vjp_tape_max_bytes": 1024,
        }
    )
    assert changed[0] == baseline[0]
    assert changed[1] != baseline[1]
    changed = _fingerprints(
        {
            "transcoder_architecture": "plt",
            "decoder_active_row_residency": True,
            "decoder_active_row_max_bytes": 1024,
            "phase0_decoder_row_ranges": True,
        }
    )
    assert changed[0] == baseline[0]
    assert changed[1] != baseline[1]


def test_diagnostic_stop_scenario_maps_to_typed_execution_policy() -> None:
    phase0 = trace_policy_from_scenario(
        {"method": "exact", "diagnostic_stop_mode": "phase0_probe"}
    )
    assert phase0.execution.diagnostic_stop.mode == "phase0_probe"
    assert phase0.execution.diagnostic_stop.phase4_batches is None

    transition = trace_policy_from_scenario(
        {
            "method": "exact",
            "diagnostic_stop_mode": "transition_probe",
            "diagnostic_stop_phase4_batches": 3,
        }
    )
    assert transition.execution.diagnostic_stop.mode == "transition_probe"
    assert transition.execution.diagnostic_stop.phase4_batches == 3


def test_phase0_decoder_row_ranges_requires_a_bool() -> None:
    with pytest.raises(ValueError, match="phase0_decoder_row_ranges must be a bool"):
        trace_policy_from_scenario(
            {
                "name": "invalid-phase0-ranges",
                "method": "exact",
                "phase0_decoder_row_ranges": 1,
            }
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "phase0_decoder_row_ranges": True,
                "decoder_active_row_residency": True,
                "decoder_active_row_max_bytes": 1024,
            },
            "PLT-compatible provider",
        ),
        (
            {
                "transcoder_architecture": "plt",
                "phase0_decoder_row_ranges": True,
                "decoder_active_row_max_bytes": 1024,
            },
            "decoder_active_row_residency=true",
        ),
        (
            {
                "transcoder_architecture": "plt",
                "phase0_decoder_row_ranges": True,
                "decoder_active_row_residency": True,
            },
            "positive decoder_active_row_safety_margin_bytes",
        ),
        (
            {
                "transcoder_architecture": "plt",
                "phase0_decoder_row_ranges": True,
                "decoder_active_row_residency": True,
                "decoder_active_row_max_bytes": 1024,
                "reuse_phase0_window_state": True,
            },
            "incompatible with reuse_phase0_window_state",
        ),
    ],
)
def test_phase0_decoder_row_ranges_rejects_invalid_scenario_dependencies(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        trace_policy_from_scenario(
            {"name": "invalid-phase0-ranges", "method": "exact", **overrides}
        )


def test_exact_child_boundary_is_scenario_file_not_flat_flags(tmp_path: Path) -> None:
    output = tmp_path / "scenario" / "artifacts"
    command = build_command(output, {"method": "exact", "name": "typed"})
    assert command[1:3] == [
        "-m",
        "nlp_research_project.exact_trace_bench.trace_runtime",
    ]
    assert command[3:] == [
        "--scenario-file",
        str(output.parent / "scenario.json"),
        "--output-dir",
        str(output),
    ]
    assert not any(argument.startswith("--phase") for argument in command)


def test_completion_orchestrator_remains_readable() -> None:
    source = inspect.getsource(trace_completion_compact_chunked)
    assert len(source.splitlines()) < 200
    signature = inspect.signature(trace_completion_compact_chunked)
    assert set(signature.parameters) == {
        "model",
        "prompt",
        "output_dir",
        "prompt_idx",
        "completion_idx",
        "completion",
        "trace_policy",
    }


def test_no_stale_project_runtime_references() -> None:
    assert not (ROOT / "trace_pipeline_chunked.py").exists()
    assert not (ROOT / "trace_pipeline.py").exists()
    assert not (ROOT / "explore_pipeline.py").exists()
    assert not (ROOT / "prefix_caching" / "trace_pipeline_cached.py").exists()
    production_roots = [ROOT / "src", ROOT / "experiments", ROOT / "slurm"]
    stale_text = []
    stale_imports = []
    for root in production_roots:
        for path in root.rglob("*.py"):
            text = path.read_text()
            if "attribute_nnsight" in text or "trace_pipeline_chunked" in text:
                stale_text.append(path.relative_to(ROOT))
            for node in ast.walk(ast.parse(text, filename=str(path))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                    if node.module == "circuit_tracer" and any(
                        alias.name == "attribute" for alias in node.names
                    ):
                        stale_imports.append((path.relative_to(ROOT), node.lineno))
                else:
                    continue
                if any(
                    name == "trace_pipeline"
                    or name.startswith("trace_pipeline.")
                    or name == "circuit_tracer.attribute"
                    or name.startswith("circuit_tracer.attribute.")
                    for name in names
                ):
                    stale_imports.append((path.relative_to(ROOT), node.lineno))
    assert stale_text == []
    assert stale_imports == []


def test_active_sparsification_runtime_has_no_old_patch_mode() -> None:
    production = [
        ROOT / "experiments" / "run_sparsification_experiment.py",
        ROOT / "experiments" / "build_sparsification_experiment_configs.py",
        ROOT
        / "experiments"
        / "generated"
        / "sparsification_calibration_scenarios.json",
        ROOT / "src" / "nlp_research_project" / "exact_trace_bench" / "trace_runtime",
    ]
    stale = []
    for path in production:
        files = path.rglob("*.py") if path.is_dir() else [path]
        stale.extend(
            file.relative_to(ROOT) for file in files if "old_patch" in file.read_text()
        )
    assert stale == []


def test_completion_preserves_compact_artifact_layout(
    monkeypatch, tmp_path: Path
) -> None:
    from nlp_research_project.exact_trace_bench.trace_runtime import generation

    compact = _typed_compact_result(
        semantic_fingerprint="semantic",
        execution_fingerprint="execution",
        telemetry_events=[{"event": "phase_complete"}],
        phase4_feature_batch_size=8,
    )
    monkeypatch.setattr(
        generation,
        "extract_compact_chunked_attribution",
        lambda *_args, **_kwargs: compact,
    )

    class Tokenizer:
        eos_token_id = 9
        pad_token_id = 0
        unk_token_id = -1

        def convert_tokens_to_ids(self, _token: str) -> int:
            return -1

        def decode(self, values, **_kwargs) -> str:
            return "done" if list(values) else ""

    class Model(FakeModel):
        tokenizer = Tokenizer()
        transcoders = None

        def ensure_tokenized(self, _prompt: str) -> torch.Tensor:
            return torch.tensor([1, 2], dtype=torch.int64)

        def generate(self, input_ids: torch.Tensor, **_kwargs):
            sequence = torch.cat(
                [input_ids, torch.tensor([[9]], dtype=input_ids.dtype)], dim=1
            )
            return SimpleNamespace(
                sequences=sequence,
                scores=[torch.zeros((1, 16), dtype=torch.float32)],
            )

    policy = trace_policy_from_scenario(
        {"name": "artifact", "method": "exact", "attribution_batch_size": 8}
    )
    manifest = trace_completion_compact_chunked(
        Model(),
        "prompt",
        output_dir=tmp_path,
        prompt_idx=0,
        completion_idx=0,
        completion=CompletionPlan(max_steps=2),
        trace_policy=policy,
    )
    root = tmp_path / "prompt_000" / "completion_000"
    assert (root / "step_000.npz").exists()
    assert (root / "telemetry.jsonl").exists()
    persisted = json.loads((root / "completion.json").read_text())
    assert manifest["n_steps_traced"] == 1
    assert persisted["semantic_fingerprint"] == "semantic"
    assert persisted["execution_fingerprint"] == "execution"
    assert persisted["steps"][0]["phase4_feature_batch_size"] == 8


def test_step_artifact_writer_preserves_phase4_timing_runtime_metadata(
    tmp_path: Path,
) -> None:
    workspace = CompletionWorkspace.create(
        tmp_path,
        prompt_index=0,
        completion_index=0,
    )
    writer = StepArtifactWriter(
        workspace=workspace,
        trace_policy=trace_policy_from_scenario({"method": "exact"}),
        model=FakeModel(),
    )
    timing_by_substage = {
        "executor_compute_batch": {
            "population_count": 4,
            "cuda_sample_count": 2,
            "cuda_estimated_total_elapsed_ms": 12.5,
        },
        "refresh_influence_matmul": {
            "population_count": 3,
            "cuda_sample_count": 1,
            "cuda_estimated_total_elapsed_ms": 7.25,
        },
    }
    step = writer.write(
        step_index=0,
        prefix_token_count=2,
        compact_result=_typed_compact_result(
            phase4_timing_backend="cuda_events_systematic_sample_deferred_v1",
            phase4_timing_cuda_event_elapsed_ms=19.75,
            phase4_timing_by_substage=timing_by_substage,
        ),
        token_result={"token_id": 9, "token_text": "done", "token_logprob": -0.1},
        attribution_seconds=1.0,
        token_generation_seconds=0.1,
        step_started=time.perf_counter(),
        stop=True,
    )

    assert step["phase4_timing_backend"] == (
        "cuda_events_systematic_sample_deferred_v1"
    )
    assert step["phase4_timing_cuda_event_elapsed_ms"] == 19.75
    assert step["phase4_timing_by_substage"] == timing_by_substage

    workspace.write_json("completion.json", {"steps": [step]})
    persisted = json.loads((workspace.root / "completion.json").read_text())
    assert persisted["steps"][0]["phase4_timing_by_substage"] == timing_by_substage


def test_completion_preserves_decoder_prefetch_diagnostics(
    monkeypatch, tmp_path: Path
) -> None:
    from nlp_research_project.exact_trace_bench.trace_runtime import generation

    monkeypatch.setattr(
        generation,
        "extract_compact_chunked_attribution",
        lambda *_args, **_kwargs: _typed_compact_result(),
    )

    class Tokenizer:
        eos_token_id = 9
        pad_token_id = 0
        unk_token_id = -1

        def convert_tokens_to_ids(self, _token: str) -> int:
            return -1

        def decode(self, values, **_kwargs) -> str:
            return "done" if list(values) else ""

    class Transcoders:
        def get_diagnostic_snapshot(self) -> dict[str, int | float]:
            return {
                "decoder_prefetch_request_count": 3,
                "decoder_prefetch_load_count": 2,
                "decoder_prefetch_load_bytes": 128,
                "decoder_prefetch_cache_hit_count": 1,
                "decoder_prefetch_consume_hit_count": 2,
                "decoder_prefetch_host_wait_count": 1,
                "decoder_prefetch_host_wait_seconds": 0.25,
                "decoder_prefetch_in_flight_count": 0,
                "decoder_prefetch_in_flight_high_watermark": 2,
                "decoder_prefetch_in_flight_bytes": 0,
                "decoder_prefetch_in_flight_bytes_high_watermark": 128,
                "decoder_prefetch_consumer_active_count": 0,
                "decoder_prefetch_consumer_active_bytes": 0,
                "decoder_prefetch_consumer_retained_count": 0,
                "decoder_prefetch_consumer_retained_bytes": 0,
                "decoder_prefetch_consumer_retained_bytes_high_watermark": 64,
                "decoder_prefetch_consumer_retirement_count": 2,
                "decoder_prefetch_consumer_backpressure_count": 1,
                "decoder_prefetch_consumer_backpressure_seconds": 0.5,
                "decoder_prefetch_pipeline_owned_final_page_count": 0,
                "decoder_prefetch_pipeline_owned_final_page_high_watermark": 1,
                "decoder_prefetch_pipeline_owned_final_page_bytes": 0,
                "decoder_prefetch_pipeline_owned_final_page_bytes_high_watermark": 64,
                "decoder_prefetch_owner_count": 0,
                "decoder_prefetch_owner_high_watermark": 1,
                "decoder_prefetch_owner_open_count": 2,
                "decoder_prefetch_owner_close_count": 2,
            }

    class Model(FakeModel):
        tokenizer = Tokenizer()
        transcoders = Transcoders()

        def ensure_tokenized(self, _prompt: str) -> torch.Tensor:
            return torch.tensor([1, 2], dtype=torch.int64)

        def generate(self, input_ids: torch.Tensor, **_kwargs):
            return SimpleNamespace(
                sequences=torch.cat(
                    [input_ids, torch.tensor([[9]], dtype=input_ids.dtype)], dim=1
                ),
                scores=[torch.zeros((1, 16), dtype=torch.float32)],
            )

    manifest = trace_completion_compact_chunked(
        Model(),
        "prompt",
        output_dir=tmp_path,
        prompt_idx=0,
        completion_idx=0,
        completion=CompletionPlan(max_steps=1),
        trace_policy=trace_policy_from_scenario(
            {"name": "prefetch", "method": "exact", "attribution_batch_size": 8}
        ),
    )

    diagnostics = manifest["steps"][0]["transcoder_diagnostics"]
    assert diagnostics["decoder_prefetch_request_count"] == 3
    assert diagnostics["decoder_prefetch_host_wait_seconds"] == 0.25
    assert diagnostics["decoder_prefetch_in_flight_high_watermark"] == 2
    assert diagnostics["decoder_prefetch_in_flight_bytes_high_watermark"] == 128
    assert diagnostics["decoder_prefetch_consumer_retirement_count"] == 2
    assert diagnostics["decoder_prefetch_owner_high_watermark"] == 1
    persisted = json.loads(
        (tmp_path / "prompt_000" / "completion_000" / "completion.json").read_text()
    )
    assert persisted["steps"][0]["transcoder_diagnostics"] == diagnostics


def test_diagnostic_completion_persists_telemetry_without_graph_packaging(
    monkeypatch, tmp_path: Path
) -> None:
    from nlp_research_project.exact_trace_bench.trace_runtime import generation

    monkeypatch.setattr(
        generation,
        "extract_compact_chunked_attribution",
        lambda *_args, **_kwargs: DiagnosticTraceCompletion(
            semantic_fingerprint="semantic-probe",
            execution_fingerprint="execution-probe",
            telemetry_summary={
                "diagnostic_stop_mode": "transition_probe",
                "phase4_batches_completed": 2,
            },
            telemetry_events=(
                {
                    "scope": "run",
                    "name": "attribute.probe_completed",
                    "attrs": {"status": "probe_completed"},
                },
            ),
            diagnostic_artifacts={
                "phase3_seed_bundle": {
                    "status": "captured",
                    "active_features": torch.tensor([[0, 0, 1], [0, 1, 2]]),
                    "activation_values": torch.tensor([0.25, 0.5]),
                    "seed_feature_influences": torch.tensor([0.125, 0.0625]),
                    "frontier_pre_locality": torch.tensor([0, 1]),
                    "frontier_post_locality": torch.tensor([1, 0]),
                    "queue_size": 2,
                    "actual_max_feature_nodes": 2,
                    "total_active_features": 2,
                    "planner_compute_dtype": "float32",
                    "influence_compute_dtype": "float32",
                }
            },
        ),
    )

    class Tokenizer:
        eos_token_id = 9
        pad_token_id = 0
        unk_token_id = -1

        def convert_tokens_to_ids(self, _token: str) -> int:
            return -1

        def decode(self, _values, **_kwargs) -> str:
            return ""

    class Model(FakeModel):
        tokenizer = Tokenizer()
        transcoders = None

        def ensure_tokenized(self, _prompt: str) -> torch.Tensor:
            return torch.tensor([1, 2], dtype=torch.int64)

        def generate(self, *_args, **_kwargs):
            raise AssertionError("diagnostic probes must not generate a token")

    manifest = trace_completion_compact_chunked(
        Model(),
        "prompt",
        output_dir=tmp_path,
        prompt_idx=0,
        completion_idx=0,
        completion=CompletionPlan(max_steps=2),
        trace_policy=trace_policy_from_scenario(
            {
                "method": "exact",
                "diagnostic_stop_mode": "transition_probe",
                "diagnostic_stop_phase4_batches": 2,
                "capture_phase3_seed_bundle": True,
            }
        ),
    )

    root = tmp_path / "prompt_000" / "completion_000"
    assert manifest["status"] == "probe_completed"
    assert manifest["diagnostic_stop_mode"] == "transition_probe"
    assert manifest["phase4_batches_completed"] == 2
    assert manifest["graph_packaging_mode"] == "diagnostic_no_graph"
    assert manifest["n_steps_traced"] == 0
    assert not (root / "step_000.npz").exists()
    assert manifest["sidecar_status"]["phase3_seed_bundle"] == "captured"
    assert manifest["capture_artifact_status"]["requested"] == ["phase3_seed_bundle"]
    assert manifest["capture_artifact_status"]["written"] == ["phase3_seed_bundle"]
    assert manifest["capture_artifact_status"]["complete"] is True
    assert (root / "step_000_phase3_seed_bundle.npz").is_file()
    assert (root / "step_000_capture_artifacts.json").is_file()
    assert (root / "telemetry.jsonl").is_file()


def test_step_artifact_writer_fails_closed_and_persists_missing_capture_status(
    tmp_path: Path,
) -> None:
    workspace = CompletionWorkspace.create(
        tmp_path,
        prompt_index=0,
        completion_index=0,
    )
    writer = StepArtifactWriter(
        workspace=workspace,
        trace_policy=trace_policy_from_scenario(
            {
                "method": "exact",
                "capture_phase3_gradient_bundle": True,
            }
        ),
        model=FakeModel(),
    )

    with pytest.raises(CaptureArtifactContractError, match="missing=phase3_gradient"):
        writer.write_diagnostic_sidecars(0, {})

    status = json.loads(
        (workspace.root / "step_000_capture_artifacts.json").read_text()
    )
    assert status["requested"] == ["phase3_gradient_bundle"]
    assert status["written"] == []
    assert status["missing"] == ["phase3_gradient_bundle"]
    assert status["failed"] == []
    assert status["complete"] is False


def test_step_artifact_writer_rejects_truncated_capture_payload(
    tmp_path: Path,
) -> None:
    workspace = CompletionWorkspace.create(
        tmp_path,
        prompt_index=0,
        completion_index=0,
    )
    writer = StepArtifactWriter(
        workspace=workspace,
        trace_policy=trace_policy_from_scenario(
            {
                "method": "exact",
                "capture_phase3_gradient_bundle": True,
            }
        ),
        model=FakeModel(),
    )

    with pytest.raises(CaptureArtifactContractError, match="failed=phase3_gradient"):
        writer.write_diagnostic_sidecars(
            0,
            {"phase3_gradient_bundle": {"status": "captured"}},
        )

    status = json.loads(
        (workspace.root / "step_000_capture_artifacts.json").read_text()
    )
    assert status["missing"] == []
    assert status["failed"][0]["name"] == "phase3_gradient_bundle"
    assert status["failed"][0]["error_type"] == "ValueError"
    assert "missing required fields" in status["failed"][0]["message"]
    assert status["complete"] is False
