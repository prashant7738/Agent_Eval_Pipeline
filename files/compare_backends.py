"""
Groq-vs-Gemini backend comparison (recreated per the 97-task rerun): Gemini's free-tier
quota (~20 requests/day, see pipeline_status memory) rules out running it on all 97 tasks,
so this script runs Gemini on a fixed stratified subset and compares against Groq's scores
for the SAME task_ids (pulled from the canonical data/rouge_scores.json, produced by
run_study1.py on the full 97-task set) so the comparison is on identical questions.

Run run_study1.py first (for the Groq/full-97 responses+ROUGE scores), then this script.
"""

import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google.genai.errors import ClientError

from agent import run_agent
from rouge_l import rouge_l

DATA_DIR = Path(__file__).parent / "data"
TASKS_PATH = DATA_DIR / "gaia_tasks.json"
GROQ_ROUGE_PATH = DATA_DIR / "rouge_scores.json"
GEMINI_RESPONSES_PATH = DATA_DIR / "responses_gemini.json"
GEMINI_ROUGE_PATH = DATA_DIR / "rouge_scores_gemini.json"

SUBSET_SIZE = 50  # expanded from 18 to improve backend comparison; fits ~2.5 days at 20/day quota
LEVEL_SHARES = {1: 0.35, 2: 0.45, 3: 0.20}  # matches gaia_loader.py's original split
SEED = 42
# Free tier is 5 requests/min per model (confirmed via 429 quotaValue), not just 20/day --
# 13s keeps us under that with margin.
SLEEP_BETWEEN_CALLS = 13.0
MAX_RETRIES = 3


def select_subset(tasks, n_total=SUBSET_SIZE, seed=SEED):
    """Build a canonical deterministic ordering (stratified by level, then shuffled)
    so that increasing n_total later keeps the original subset as a prefix -- no re-sampling
    or task duplication when the size expands."""
    rng = random.Random(seed)
    
    by_level = {level: [t for t in tasks if t["level"] == level] for level in [1, 2, 3]}
    for level in [1, 2, 3]:
        rng.shuffle(by_level[level])
    
    # Interleave tasks from each level (round-robin) to distribute across the full ordering
    canonical_order = []
    for i in range(max(len(by_level[level]) for level in [1, 2, 3])):
        for level in [1, 2, 3]:
            if i < len(by_level[level]):
                canonical_order.append(by_level[level][i])
    
    return canonical_order[:n_total]


def generate_gemini_responses(subset):
    """Resumable: skips task_ids already present in GEMINI_RESPONSES_PATH from a prior
    (possibly crashed) run, and writes to disk after every task instead of only at the end."""
    responses = []
    if GEMINI_RESPONSES_PATH.exists():
        with open(GEMINI_RESPONSES_PATH) as f:
            responses = json.load(f)
    done_ids = {r["task_id"] for r in responses}

    for i, task in enumerate(subset, start=1):
        if task["task_id"] in done_ids:
            print(f"[{i}/{len(subset)}] skipping {task['task_id']} (already done)")
            continue

        print(f"[{i}/{len(subset)}] generating Gemini response for {task['task_id']}")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = run_agent(task["question"], backend="gemini", temperature=0.0)
                break
            except ClientError as e:
                if attempt == MAX_RETRIES:
                    raise
                print(f"  attempt {attempt} failed ({e}), sleeping 65s before retry")
                time.sleep(65)

        responses.append({
            "task_id": task["task_id"],
            "level": task["level"],
            "question": task["question"],
            "reference_answer": task["reference_answer"],
            "response": result["answer"],
            "raw_response": result["raw_response"],
        })
        with open(GEMINI_RESPONSES_PATH, "w") as f:
            json.dump(responses, f, indent=2)
        time.sleep(SLEEP_BETWEEN_CALLS)

    return responses


def score_gemini_rouge(responses):
    scores = []
    for r in responses:
        score = rouge_l(r["response"], r["reference_answer"])
        scores.append({"task_id": r["task_id"], "level": r["level"], **score})

    with open(GEMINI_ROUGE_PATH, "w") as f:
        json.dump(scores, f, indent=2)
    return scores


def print_comparison(gemini_scores):
    gemini_f1s = [s["f1"] for s in gemini_scores]
    gemini_mean = sum(gemini_f1s) / len(gemini_f1s)
    print(f"\nGemini ROUGE-L F1: mean={gemini_mean:.3f}, n={len(gemini_f1s)}")

    if not GROQ_ROUGE_PATH.exists():
        print("(run_study1.py hasn't been run yet -- no Groq scores to compare against)")
        return

    with open(GROQ_ROUGE_PATH) as f:
        groq_scores_by_id = {s["task_id"]: s["f1"] for s in json.load(f)}

    matched_groq_f1s = [groq_scores_by_id[s["task_id"]] for s in gemini_scores if s["task_id"] in groq_scores_by_id]
    if matched_groq_f1s:
        groq_mean = sum(matched_groq_f1s) / len(matched_groq_f1s)
        print(f"Groq ROUGE-L F1 (same {len(matched_groq_f1s)} task_ids): mean={groq_mean:.3f}")
    else:
        print("(no matching task_ids found in Groq's rouge_scores.json)")


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    subset = select_subset(tasks)
    print(f"Selected {len(subset)} tasks for Gemini subset (levels: "
          f"{sorted(t['level'] for t in subset)})")

    responses = generate_gemini_responses(subset)
    scores = score_gemini_rouge(responses)
    print_comparison(scores)
