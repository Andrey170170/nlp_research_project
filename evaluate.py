"""
Correctness evaluation for traced completions.

Two-stage approach:
  1. Regex extraction of final numeric answer (GSM8K convention)
  2. GPT-5.4 judge for semantic correctness

Reads completion.json + prompt_meta.json, writes evaluation.json
alongside each completion.

Usage:
    python evaluate.py --traces-dir /fs/scratch/PAS3272/kopanev.1/traces
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import openai  # noqa: E402


def is_complete_completion(completion_dir: Path) -> bool:
    """Return True if a traced completion finished and has a manifest."""
    return (completion_dir / "completion.json").exists()


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output.

    Looks for patterns like:
      - "Final answer: 42"
      - "#### 42"
      - "the answer is 42"
    """
    # Try GSM8K convention first: #### <number>
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    # Try "Final answer: <number>"
    m = re.search(r"[Ff]inal\s+[Aa]nswer\s*[:=]\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    # Try "the answer is <number>"
    m = re.search(r"[Tt]he\s+answer\s+is\s+(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    return None


def extract_ground_truth(answer_text: str) -> str:
    """Extract numeric answer from GSM8K ground truth (#### format)."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number in the text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", answer_text)
    return numbers[-1].replace(",", "") if numbers else ""


def judge_with_gpt(
    question: str,
    model_answer: str,
    ground_truth: str,
    *,
    model: str = "gpt-5.4-mini",
    max_retries: int = 3,
) -> dict:
    """Ask GPT to judge if the model's answer is correct."""
    client = openai.OpenAI()

    prompt = f"""You are evaluating a math problem solution. Determine if the model's final numeric answer matches the ground truth.

Question: {question}

Model's full response:
{model_answer}

Ground truth answer: {ground_truth}

Does the model's final numeric answer match the ground truth? Consider the answer correct if it arrives at the same number, even if the reasoning path differs.

Respond in JSON format:
{{"correct": true/false, "model_final_answer": "<the number the model gave>", "explanation": "<brief explanation>"}}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            return result
        except (openai.RateLimitError, openai.APIConnectionError) as e:
            wait = 2 ** (attempt + 1)
            print(f"    API error: {e}, retrying in {wait}s...")
            time.sleep(wait)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    Parse error: {e}, retrying...")
            time.sleep(1)

    return {
        "correct": False,
        "explanation": "Failed after retries",
        "model_final_answer": "",
    }


def evaluate_completion(
    completion_dir: Path,
    prompt_meta: dict,
    *,
    judge_model: str = "gpt-5.4-mini",
) -> dict:
    """Evaluate a single completion directory."""
    manifest_path = completion_dir / "completion.json"
    if not manifest_path.exists():
        return {"error": f"No completion.json in {completion_dir}"}

    manifest = json.loads(manifest_path.read_text())
    model_answer = manifest["completion_text"]
    ground_truth_raw = prompt_meta["ground_truth_answer"]
    ground_truth = extract_ground_truth(ground_truth_raw)
    question = prompt_meta["question"]

    # Stage 1: regex extraction
    regex_answer = extract_numeric_answer(model_answer)
    regex_match = regex_answer is not None and regex_answer == ground_truth

    # Stage 2: GPT judge
    judge_result = judge_with_gpt(
        question, model_answer, ground_truth, model=judge_model
    )

    evaluation = {
        "model_answer": model_answer[:1000],
        "ground_truth": ground_truth,
        "regex_extracted_answer": regex_answer,
        "regex_match": regex_match,
        "judge_correct": judge_result.get("correct", False),
        "judge_model_final_answer": judge_result.get("model_final_answer", ""),
        "judge_explanation": judge_result.get("explanation", ""),
        "judge_model": judge_model,
    }

    eval_path = completion_dir / "evaluation.json"
    eval_path.write_text(json.dumps(evaluation, indent=2))
    return evaluation


def run_evaluation(args: argparse.Namespace) -> None:
    traces_dir = Path(args.traces_dir)
    if not traces_dir.exists():
        print(f"Traces directory not found: {traces_dir}")
        return

    prompt_dirs = sorted(traces_dir.glob("prompt_*"))
    if not prompt_dirs:
        print(f"No prompt directories found in {traces_dir}")
        return

    total_correct = 0
    total_evaluated = 0

    for prompt_dir in prompt_dirs:
        meta_path = prompt_dir / "prompt_meta.json"
        if not meta_path.exists():
            print(f"Skipping {prompt_dir.name}: no prompt_meta.json")
            continue

        prompt_meta = json.loads(meta_path.read_text())
        print(f"\n{prompt_dir.name}: {prompt_meta['question'][:80]}...")

        completion_dirs = sorted(prompt_dir.glob("completion_*"))
        for comp_dir in completion_dirs:
            if not is_complete_completion(comp_dir):
                print(
                    f"  Warning: skipping {comp_dir.name} "
                    "(missing completion.json; run likely interrupted)"
                )
                continue

            # Skip if already evaluated
            if (comp_dir / "evaluation.json").exists() and not args.force:
                existing = json.loads((comp_dir / "evaluation.json").read_text())
                correct = existing.get("judge_correct", False)
                total_correct += int(correct)
                total_evaluated += 1
                print(f"  {comp_dir.name}: already evaluated (correct={correct})")
                continue

            print(f"  Evaluating {comp_dir.name}...")
            result = evaluate_completion(
                comp_dir, prompt_meta, judge_model=args.judge_model
            )

            if "error" not in result:
                correct = result["judge_correct"]
                total_correct += int(correct)
                total_evaluated += 1
                print(
                    f"    regex={result['regex_extracted_answer']!r} "
                    f"judge={correct} "
                    f"({result['judge_explanation'][:60]})"
                )
            else:
                print(f"    Error: {result['error']}")

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete: {total_correct}/{total_evaluated} correct")
    if total_evaluated:
        print(f"Accuracy: {total_correct / total_evaluated:.1%}")

    # Save summary
    summary = {
        "total_evaluated": total_evaluated,
        "total_correct": total_correct,
        "accuracy": total_correct / total_evaluated if total_evaluated else 0,
    }
    (traces_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate traced completions")
    parser.add_argument(
        "--traces-dir",
        required=True,
        help="Directory containing prompt_*/completion_* traces",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-5.4-mini",
        help="OpenAI model for judging (default: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if evaluation.json exists",
    )
    args = parser.parse_args()
    run_evaluation(args)
