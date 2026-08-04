"""
Study 2 setup (roadmap.txt Day 4): build the human-labeling sheets and get the
LLM judge's labels on the same pool, ahead of the human annotation round.

The labeling pool combines Study 1's main responses (one per task) with the
SelfCheckGPT resampled responses, since the single-agent-config task set alone
(15-20 tasks) is smaller than the 30 responses the roadmap asks the humans to
label. Run run_study1.py first to produce responses.json and selfcheck_results.json.
"""

import csv
import json
import random
from pathlib import Path

from dotenv import load_dotenv

from llm_judge import rubric_label

DATA_DIR = Path(__file__).parent / "data"
RESPONSES_PATH = DATA_DIR / "responses.json"
SELFCHECK_PATH = DATA_DIR / "selfcheck_results.json"
GAIA_TASKS_PATH = DATA_DIR / "gaia_tasks.json"

POOL_PATH = DATA_DIR / "human_study_pool.json"
CALIBRATION_CSV = DATA_DIR / "calibration_sheet.csv"
ANNOTATOR1_CSV = DATA_DIR / "annotator1_sheet.csv"
ANNOTATOR2_CSV = DATA_DIR / "annotator2_sheet.csv"
JUDGE_LABELS_PATH = DATA_DIR / "judge_labels.json"

MAIN_SAMPLE_SIZE = 30
CALIBRATION_SIZE = 6
CSV_FIELDS = ["item_id", "task_id", "level", "question", "reference_answer", "response", "label"]


def build_pool():
    """Combine main responses and SelfCheckGPT samples into one labeling pool."""
    pool = []

    # Load task reference answers for lookup
    task_refs = {}
    if GAIA_TASKS_PATH.exists():
        with open(GAIA_TASKS_PATH) as f:
            for task in json.load(f):
                task_refs[task["task_id"]] = task["reference_answer"]

    with open(RESPONSES_PATH) as f:
        for r in json.load(f):
            pool.append({
                "item_id": f"main_{r['task_id']}",
                "task_id": r["task_id"],
                "level": r["level"],
                "question": r["question"],
                "reference_answer": r["reference_answer"],
                "response": r["response"],
            })

    if SELFCHECK_PATH.exists():
        with open(SELFCHECK_PATH) as f:
            for entry in json.load(f):
                for j, sample in enumerate(entry["samples"]):
                    # Look up reference answer from gaia_tasks.json if available
                    ref_answer = task_refs.get(entry["task_id"], "")
                    pool.append({
                        "item_id": f"selfcheck_{entry['task_id']}_{j}",
                        "task_id": entry["task_id"],
                        "level": entry["level"],
                        "question": entry["question"],
                        "reference_answer": ref_answer,
                        "response": sample,
                    })

    return pool


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "label": ""})


def prepare_sheets(seed: int = 42):
    pool = build_pool()
    rng = random.Random(seed)
    rng.shuffle(pool)

    calibration = pool[:CALIBRATION_SIZE]
    main_sample = pool[CALIBRATION_SIZE:CALIBRATION_SIZE + MAIN_SAMPLE_SIZE]

    DATA_DIR.mkdir(exist_ok=True)
    with open(POOL_PATH, "w") as f:
        json.dump({"calibration": calibration, "main_sample": main_sample}, f, indent=2)

    _write_csv(CALIBRATION_CSV, calibration)

    # Each annotator gets the same 30 items in an independently randomized order,
    # and labels them blind to the other annotator / judge (per the paper's protocol).
    order_a = main_sample[:]
    order_b = main_sample[:]
    random.Random(seed + 1).shuffle(order_a)
    random.Random(seed + 2).shuffle(order_b)
    _write_csv(ANNOTATOR1_CSV, order_a)
    _write_csv(ANNOTATOR2_CSV, order_b)

    print(f"Calibration sheet ({len(calibration)} items): {CALIBRATION_CSV}")
    print(f"Annotator sheets ({len(main_sample)} items each): {ANNOTATOR1_CSV}, {ANNOTATOR2_CSV}")
    print("Fill in the 'label' column with one of: correct, partial, incorrect, hallucinated")
    return calibration, main_sample


def label_with_judge(main_sample):
    """Run the LLM judge on the same 30-item pool, independently of the humans."""
    judge_labels = {}
    for i, item in enumerate(main_sample, start=1):
        print(f"[{i}/{len(main_sample)}] judging {item['item_id']}")
        judge_labels[item["item_id"]] = rubric_label(
            item["question"], item["reference_answer"], item["response"]
        )

    with open(JUDGE_LABELS_PATH, "w") as f:
        json.dump(judge_labels, f, indent=2)
    print(f"Judge labels saved to {JUDGE_LABELS_PATH}")
    return judge_labels


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    _, main_sample = prepare_sheets()
    label_with_judge(main_sample)
