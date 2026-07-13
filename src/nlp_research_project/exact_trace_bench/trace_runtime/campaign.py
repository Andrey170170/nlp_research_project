"""Readable project-owned orchestration for one persisted trace scenario."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from circuit_tracer import SparsificationConfig

from .generation import CompletionPlan, trace_completion_compact_chunked
from .prompts import load_prompts, write_prompt_record
from .provider import ProviderLoadPolicy, get_model_transcoder_metadata
from .request import trace_policy_from_scenario


def run_scenario_file(scenario_file: Path, output_dir: Path) -> None:
    scenario = json.loads(scenario_file.read_text())
    if not isinstance(scenario, dict):
        raise TypeError("scenario file must contain one JSON object")
    run_campaign(scenario, output_dir=output_dir)


def run_campaign(scenario: Mapping[str, Any], *, output_dir: Path) -> None:
    """Load once, then execute the prompt/completion campaign."""

    if scenario.get("method", "exact") == "old_patch":
        raise ValueError("old_patch scenarios use the historical project launcher")
    if scenario.get("save_raw", False):
        raise ValueError("C2 canonical campaign supports compact trace results only")

    output_dir.mkdir(parents=True, exist_ok=True)
    provider = ProviderLoadPolicy.from_scenario(scenario)
    model = provider.load()
    prompts = load_prompts(model, scenario)
    sparsification = _sparsification_from_scenario(scenario)
    trace_policy = trace_policy_from_scenario(
        scenario,
        sparsification=sparsification,
    )
    _write_run_config(output_dir, scenario, provider, model)

    completions = int(scenario.get("completions", 1))
    total = len(prompts) * completions
    completed = 0
    for prompt_index, prepared in enumerate(prompts):
        write_prompt_record(output_dir, prompt_index, prepared)
        for completion_index in range(completions):
            completed += 1
            print(f"\n{'=' * 60}")
            print(
                f"Completion {completed}/{total}: prompt {prompt_index}, "
                f"completion {completion_index}"
            )
            print(f"{'=' * 60}")
            _run_completion(
                model=model,
                prompt=prepared.text,
                prompt_metadata=prepared.metadata,
                prompt_index=prompt_index,
                completion_index=completion_index,
                output_dir=output_dir,
                scenario=scenario,
                sparsification=sparsification,
                trace_policy=trace_policy,
            )
    print(f"\nPipeline complete! {completed} completions traced to {output_dir}")


def _run_completion(
    *,
    model: Any,
    prompt: str,
    prompt_metadata: Mapping[str, Any],
    prompt_index: int,
    completion_index: int,
    output_dir: Path,
    scenario: Mapping[str, Any],
    sparsification: SparsificationConfig | None,
    trace_policy: Any,
) -> None:
    """Run generation with one typed trace template and artifact policy."""

    del sparsification  # already owned by trace_policy.semantics
    trace_completion_compact_chunked(
        model,
        prompt,
        output_dir=output_dir,
        prompt_idx=prompt_index,
        completion_idx=completion_index,
        completion=CompletionPlan(
            temperature=float(scenario.get("temperature", 0.7)),
            max_steps=int(scenario.get("max_steps", 256)),
            max_edges=int(scenario.get("max_edges", 10_000)),
            incremental_telemetry_jsonl=bool(
                scenario.get("incremental_telemetry_jsonl", False)
            ),
            prompt_token_count=int(prompt_metadata["prompt_token_count"]),
            prompt_source=str(prompt_metadata["prompt_source"]),
            fixture_name=prompt_metadata.get("fixture_name"),
            fixture_kind=prompt_metadata.get("fixture_kind"),
        ),
        trace_policy=trace_policy,
    )


def _sparsification_from_scenario(
    scenario: Mapping[str, Any],
) -> SparsificationConfig | None:
    per_bucket = scenario.get("sparsify_per_layer_position_topk")
    global_cap = scenario.get("sparsify_global_cap")
    if per_bucket is None and global_cap is None:
        return None
    return SparsificationConfig(
        per_layer_position_topk=per_bucket,
        global_cap=global_cap,
    )


def _write_run_config(
    output_dir: Path,
    scenario: Mapping[str, Any],
    provider: ProviderLoadPolicy,
    model: Any,
) -> None:
    transcoder = get_model_transcoder_metadata(model)
    payload = {
        **scenario,
        "output_dir": str(output_dir),
        "patch_type": "canonical_typed_trace_runtime",
        "uses_monkeypatch": False,
        "provider_load_policy": provider.__dict__,
        "transcoder": transcoder,
        "graph_packaging_mode": "compact_chunked_no_full_graph",
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2, default=str))
