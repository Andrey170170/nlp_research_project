from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from nlp_research_project.exact_trace_bench.full_answer.schemas import (
    validate_trajectory,
)
from nlp_research_project.exact_trace_bench.full_answer.trajectory import (
    build_trajectory,
    extracted_final_answer,
    generated_answer_text,
    sample_trajectories_until_success,
    trajectory_matches_expected_answer,
)
from nlp_research_project.exact_trace_bench.trace_runtime import provider, support


def test_build_trajectory_schema_from_token_ids() -> None:
    trajectory = build_trajectory(
        trajectory_id="fixture_tok",
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4, 5],
        token_texts=[" A", "<eos>"],
        token_logprobs=[-0.2, -0.1],
        stop_token_ids={5},
    )
    validate_trajectory(trajectory)
    assert trajectory["prompt_token_count"] == 3
    assert trajectory["generated_tokens"][0]["absolute_token_position"] == 3
    assert trajectory["generated_tokens"][0]["logprob"] == -0.2
    assert trajectory["generated_tokens"][1]["is_stop"] is True


def test_trajectory_success_predicate_uses_final_text_and_length() -> None:
    trajectory = build_trajectory(
        trajectory_id="predicate",
        prompt_token_ids=[1],
        generated_token_ids=[2, 3],
        token_texts=[" 42", "\n"],
    )
    assert generated_answer_text(trajectory) == " 42\n"
    assert extracted_final_answer(trajectory) == "42"
    assert trajectory_matches_expected_answer(
        trajectory, expected_answer="42", max_generated_tokens=2
    )
    assert not trajectory_matches_expected_answer(
        trajectory, expected_answer="42", max_generated_tokens=1
    )


def test_trajectory_success_predicate_extracts_final_answer_label() -> None:
    trajectory = build_trajectory(
        trajectory_id="predicate_final",
        prompt_token_ids=[1],
        generated_token_ids=[2, 3, 4],
        token_texts=["work 140 - 80", "\nFinal answer: 60", "<end_of_turn>"],
    )
    assert extracted_final_answer(trajectory) == "60"
    assert trajectory_matches_expected_answer(
        trajectory, expected_answer="60", max_generated_tokens=3
    )


class _FakeIds:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def unsqueeze(self, _dim: int) -> "_FakeBatch":
        return _FakeBatch(self.values)


class _FakeRow:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def tolist(self) -> list[int]:
        return self._values


class _FakeBatch:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __getitem__(self, index: int) -> _FakeRow:
        assert index == 0
        return _FakeRow(self.values)


class _FakeModel:
    tokenizer = SimpleNamespace(eos_token_id=999, pad_token_id=None, unk_token_id=-1)

    def ensure_tokenized(self, _prompt: str) -> _FakeIds:
        return _FakeIds([10, 11])


def test_sampling_loop_saves_every_attempt_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Question?", encoding="utf-8")
    state = {"calls": 0}

    def generate_next_token(_model, input_ids, *, temperature: float):
        state["calls"] += 1
        token_text = "wrong" if state["calls"] == 1 else "42"
        return {
            "token_id": 100 + state["calls"],
            "token_text": token_text,
            "token_logprob": None,
            "next_input_ids": input_ids,
        }

    monkeypatch.setattr(provider, "load_model", lambda exact_chunked_decoder: _FakeModel())
    monkeypatch.setattr(support, "generate_next_token", generate_next_token)

    manifest = sample_trajectories_until_success(
        prompt_path=prompt_path,
        output_dir=tmp_path / "samples",
        expected_answer="42",
        max_success_tokens=1,
        max_attempts=3,
        max_new_tokens=1,
        base_seed=7,
    )

    assert manifest["status"] == "success"
    assert [row["seed"] for row in manifest["attempts"]] == [7, 8]
    assert (tmp_path / "samples" / "attempt_000001.json").exists()
    assert (tmp_path / "samples" / "attempt_000002.json").exists()
    saved_manifest = json.loads(
        (tmp_path / "samples" / "manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["success_attempt"]["attempt"] == 2


def test_sampling_collect_all_continues_after_success(
    tmp_path: Path, monkeypatch
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Question?", encoding="utf-8")
    outputs = iter(["42", "wrong", "42"])

    def generate_next_token(_model, input_ids, *, temperature: float):
        token_text = next(outputs)
        return {
            "token_id": 100,
            "token_text": token_text,
            "token_logprob": None,
            "next_input_ids": input_ids,
        }

    monkeypatch.setattr(provider, "load_model", lambda exact_chunked_decoder: _FakeModel())
    monkeypatch.setattr(support, "generate_next_token", generate_next_token)

    manifest = sample_trajectories_until_success(
        prompt_path=prompt_path,
        output_dir=tmp_path / "samples",
        expected_answer="42",
        max_success_tokens=1,
        max_attempts=3,
        max_new_tokens=1,
        base_seed=7,
        collect_all=True,
    )

    assert manifest["status"] == "completed"
    assert [row["success"] for row in manifest["attempts"]] == [True, False, True]
    assert manifest["success_attempt"]["attempt"] == 1
    assert [row["attempt"] for row in manifest["success_attempts"]] == [1, 3]
    assert (tmp_path / "samples" / "attempt_000003.json").exists()
