from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
from nlp_research_project.exact_trace_bench.trace_runtime.request import (  # noqa: E402
    trace_policy_from_scenario,
)


class FakeModel:
    backend = "nnsight"
    provider_id = "c2-project-test"


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
    ):
        changed = _fingerprints({key: value})
        assert changed[0] == baseline[0], key
        assert changed[1] != baseline[1], key
    changed = _fingerprints(
        {
            "feature_vjp_tape_batch_window": 2,
            "feature_vjp_tape_max_bytes": 1024,
        }
    )
    assert changed[0] == baseline[0]
    assert changed[1] != baseline[1]


def test_exact_child_boundary_is_scenario_file_not_flat_flags(tmp_path: Path) -> None:
    output = tmp_path / "scenario" / "artifacts"
    command = build_command(output, {"method": "exact", "name": "typed"})
    assert command[1:3] == ["-m", "nlp_research_project.exact_trace_bench.trace_runtime"]
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
        ROOT / "experiments" / "generated" / "sparsification_calibration_scenarios.json",
        ROOT / "src" / "nlp_research_project" / "exact_trace_bench" / "trace_runtime",
    ]
    stale = []
    for path in production:
        files = path.rglob("*.py") if path.is_dir() else [path]
        stale.extend(
            file.relative_to(ROOT)
            for file in files
            if "old_patch" in file.read_text()
        )
    assert stale == []


def test_completion_preserves_compact_artifact_layout(monkeypatch, tmp_path: Path) -> None:
    from nlp_research_project.exact_trace_bench.trace_runtime import generation

    compact = {
        "active_features": torch.tensor([[0, 0, 7]], dtype=torch.int64),
        "selected_features": torch.tensor([0], dtype=torch.int64),
        "feature_row_node_indices": torch.tensor([0], dtype=torch.int64),
        "feature_feature_edges": torch.tensor([[1.0]]),
        "logit_feature_edges": torch.tensor([[0.5]]),
        "semantic_fingerprint": "semantic",
        "execution_fingerprint": "execution",
        "telemetry_events": [{"event": "phase_complete"}],
        "phase4_feature_batch_size": 8,
    }
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
