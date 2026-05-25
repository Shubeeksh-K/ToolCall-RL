"""Run the baseline tool-call evaluation through LiteLLM."""

from __future__ import annotations

from dataclasses import asdict

from litellm import completion

from toolcall_rl.evaluation.cases import SEED_EVAL_CASES, EvalCase
from toolcall_rl.evaluation.schemas import SYSTEM_PROMPT
from toolcall_rl.evaluation.scoring import score_response


def run_baseline(
    model: str = "ollama_chat/smollm:1.7b",
    cases: list[EvalCase] | None = None,
) -> list[dict]:
    """Run the seed baseline eval with the same LiteLLM method as the notebook."""

    eval_cases = cases or SEED_EVAL_CASES
    rows = []

    for case in eval_cases:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": case.prompt},
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content or ""
        score = score_response(response_text, case)
        rows.append(
            {
                "case": asdict(case),
                "response": response_text,
                "score": asdict(score),
            }
        )

    return rows


def print_summary(rows: list[dict]) -> None:
    """Print a compact baseline summary."""

    total = len(rows)
    passed = sum(row["score"]["total_reward"] == 4 for row in rows)
    total_reward = sum(row["score"]["total_reward"] for row in rows)

    print(f"cases: {total}")
    print(f"passed: {passed}/{total}")
    print(f"total_reward: {total_reward}/{total * 4}")
    print()

    for index, row in enumerate(rows, start=1):
        score = row["score"]
        case = row["case"]
        print(f"{index}. {case['expected_tool']} reward={score['total_reward']}/4")
        print(f"prompt: {case['prompt']}")
        print(f"response: {row['response']}")
        print()


def main() -> None:
    rows = run_baseline()
    print_summary(rows)


if __name__ == "__main__":
    main()
