from __future__ import annotations

from nlp_research_project.exact_trace_bench.full_answer.schemas import (
    validate_trajectory,
)
from nlp_research_project.exact_trace_bench.full_answer.trajectory import (
    build_trajectory,
)


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
