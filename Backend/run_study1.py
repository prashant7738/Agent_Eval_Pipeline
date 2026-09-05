"""
Study 1 pipeline (roadmap.txt Days 2-3): generate agent responses on the GAIA
task subset, score them with ROUGE-L, then run SelfCheckGPT + a per-task
Bradley-Terry ranking on a smaller resampled subset.

Run gaia_loader.py first to produce data/gaia_tasks.json.
"""

import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv

from agent import run_agent
from rouge_l import rouge_l
from selfcheckgpt import consistency_score
from llm_judge import pairwise_compare
from bradley_terry import fit_bradley_terry

DATA_DIR = Path(__file__).parent / "data"
TASKS_PATH = DATA_DIR / "gaia_tasks.json"
RESPONSES_PATH = DATA_DIR / "responses.json"
ROUGE_PATH = DATA_DIR / "rouge_scores.json"
SELFCHECK_PATH = DATA_DIR / "selfcheck_results.json"

SELFCHECK_SUBSET_SIZE = 8
SELFCHECK_K = 3
SLEEP_BETWEEN_CALLS = 1.0  # be polite to the free-tier rate limit


def load_tasks():
    with open(TASKS_PATH) as f:
        return json.load(f)


def generate_responses(tasks):
    """Stage 2: run the single-call agent once per task at temperature 0."""
    responses = []
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] generating response for {task['task_id']}")
        result = run_agent(task["question"], temperature=0.0)
        responses.append({
            "task_id": task["task_id"],
            "level": task["level"],
            "question": task["question"],
            "reference_answer": task["reference_answer"],
            "response": result["answer"],
            "raw_response": result["raw_response"],
        })
        time.sleep(SLEEP_BETWEEN_CALLS)

    DATA_DIR.mkdir(exist_ok=True)
    with open(RESPONSES_PATH, "w") as f:
        json.dump(responses, f, indent=2)
    return responses


def score_rouge(responses):
    """Stage 1: ROUGE-L of each response against its GAIA reference answer."""
    scores = []
    for r in responses:
        score = rouge_l(r["response"], r["reference_answer"])
        scores.append({"task_id": r["task_id"], "level": r["level"], **score})

    with open(ROUGE_PATH, "w") as f:
        json.dump(scores, f, indent=2)

    f1s = [s["f1"] for s in scores]
    print(f"ROUGE-L F1: mean={sum(f1s) / len(f1s):.3f}, n={len(f1s)}")
    return scores


def run_selfcheck_and_ranking(tasks, subset_size=SELFCHECK_SUBSET_SIZE, k=SELFCHECK_K):
    """
    Stage 3: draw k resampled responses per task, score SelfCheckGPT consistency,
    and pairwise-judge the k samples against each other to fit a per-task
    Bradley-Terry ranking (items = the k sampled responses for that one task --
    this is the scoped-down substitute for ranking two agent configurations).
    """
    rng = random.Random(42)
    subset = rng.sample(tasks, min(subset_size, len(tasks)))

    results = []
    for i, task in enumerate(subset, start=1):
        print(f"[{i}/{len(subset)}] sampling k={k} responses for {task['task_id']}")
        samples = []
        for _ in range(k):
            result = run_agent(task["question"], temperature=1.0)
            samples.append(result["answer"])
            time.sleep(SLEEP_BETWEEN_CALLS)

        consistency = consistency_score(samples) if len(set(samples)) > 1 or len(samples) > 1 else 0.0

        items = [f"sample_{j}" for j in range(k)]
        comparisons = []
        for a in range(k):
            for b in range(a + 1, k):
                verdict_ab = pairwise_compare(task["question"], samples[a], samples[b])
                verdict_ba = pairwise_compare(task["question"], samples[b], samples[a])
                time.sleep(SLEEP_BETWEEN_CALLS)

                # Only keep comparisons that agree when the A/B order is swapped --
                # this is the position-bias control noted in llm_judge.py.
                if verdict_ab == "A" and verdict_ba == "B":
                    comparisons.append((items[a], items[b]))
                elif verdict_ab == "B" and verdict_ba == "A":
                    comparisons.append((items[b], items[a]))

        bt_strength = fit_bradley_terry(comparisons, items=items) if comparisons else {}

        results.append({
            "task_id": task["task_id"],
            "level": task["level"],
            "question": task["question"],
            "samples": samples,
            "consistency_score": consistency,
            "bradley_terry_strength": bt_strength,
            "n_comparisons_used": len(comparisons),
        })

    with open(SELFCHECK_PATH, "w") as f:
        json.dump(results, f, indent=2)

    scores = [r["consistency_score"] for r in results]
    print(f"SelfCheckGPT consistency: mean={sum(scores) / len(scores):.3f}, n={len(scores)}")
    return results


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    tasks = load_tasks()
    responses = generate_responses(tasks)
    score_rouge(responses)
    run_selfcheck_and_ranking(tasks)
    print("Study 1 pipeline complete. See files/data/*.json")
