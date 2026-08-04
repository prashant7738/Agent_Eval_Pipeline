"""
One-off comparison: run the Gemini backend (falls back to DuckDuckGo snippets,
see agent.py) on the same 18 GAIA tasks used for the canonical Groq-based
Study 1 results, and compare mean ROUGE-L F1. Does NOT overwrite responses.json /
rouge_scores.json -- those stay the canonical Groq run already used to build the
human-study labeling sheets.
"""

import json
import time
from pathlib import Path

from dotenv import load_dotenv

from agent import run_agent
from rouge_l import rouge_l

DATA_DIR = Path(__file__).parent / "data"
TASKS_PATH = DATA_DIR / "gaia_tasks.json"
GROQ_ROUGE_PATH = DATA_DIR / "rouge_scores.json"
GEMINI_RESPONSES_PATH = DATA_DIR / "responses_gemini.json"
GEMINI_ROUGE_PATH = DATA_DIR / "rouge_scores_gemini.json"


def main():
    load_dotenv(Path(__file__).parent / ".env")
    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    responses = []
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] gemini response for {task['task_id']}")
        result = run_agent(task["question"], temperature=0.0, backend="gemini")
        responses.append({
            "task_id": task["task_id"],
            "level": task["level"],
            "question": task["question"],
            "reference_answer": task["reference_answer"],
            "response": result["answer"],
            "raw_response": result["raw_response"],
            "grounded": result.get("grounded", False),
        })
        time.sleep(1.0)

    with open(GEMINI_RESPONSES_PATH, "w") as f:
        json.dump(responses, f, indent=2)

    scores = []
    for r in responses:
        score = rouge_l(r["response"], r["reference_answer"])
        scores.append({"task_id": r["task_id"], "level": r["level"], **score})
    with open(GEMINI_ROUGE_PATH, "w") as f:
        json.dump(scores, f, indent=2)

    gemini_f1s = [s["f1"] for s in scores]
    gemini_mean = sum(gemini_f1s) / len(gemini_f1s)
    print(f"\nGemini ROUGE-L F1: mean={gemini_mean:.3f}, n={len(gemini_f1s)}")

    if GROQ_ROUGE_PATH.exists():
        with open(GROQ_ROUGE_PATH) as f:
            groq_scores = json.load(f)
        groq_f1s = [s["f1"] for s in groq_scores]
        groq_mean = sum(groq_f1s) / len(groq_f1s)
        print(f"Groq ROUGE-L F1:   mean={groq_mean:.3f}, n={len(groq_f1s)}")


if __name__ == "__main__":
    main()
