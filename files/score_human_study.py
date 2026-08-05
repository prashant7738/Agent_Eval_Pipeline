"""
Study 2 scoring (roadmap.txt Day 5): compute human-human kappa, flag
disagreements for adjudication, then compute judge-human kappa against the
adjudicated labels. Run this after both annotator CSVs have their 'label'
column filled in.
"""

import csv
import json
from pathlib import Path

from kappa import cohens_kappa, interpret_kappa

DATA_DIR = Path(__file__).parent / "data"
ANNOTATOR1_CSV = DATA_DIR / "annotator1_sheet.csv"
ANNOTATOR2_CSV = DATA_DIR / "annotator2_sheet.csv"
JUDGE_LABELS_PATH = DATA_DIR / "judge_labels.json"
ADJUDICATION_CSV = DATA_DIR / "adjudication_needed.csv"
FINAL_LABELS_PATH = DATA_DIR / "final_human_labels.json"

VALID_LABELS = {"correct", "partial", "incorrect", "hallucinated"}


def _read_labels(path):
    labels = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            label = row["label"].strip().lower()
            if label not in VALID_LABELS:
                raise ValueError(f"{path}: item {row['item_id']} has invalid/empty label {label!r}")
            labels[row["item_id"]] = label
    return labels


def human_human_kappa():
    labels_a = _read_labels(ANNOTATOR1_CSV)
    labels_b = _read_labels(ANNOTATOR2_CSV)
    item_ids = sorted(labels_a.keys())
    if item_ids != sorted(labels_b.keys()):
        raise ValueError("Annotator sheets must label the same set of item_ids.")

    result = cohens_kappa([labels_a[i] for i in item_ids], [labels_b[i] for i in item_ids])
    print(f"Human-human kappa = {result['kappa']} ({interpret_kappa(result['kappa'])}), "
          f"observed={result['observed_agreement']}, n={result['n']}")

    disagreements = [i for i in item_ids if labels_a[i] != labels_b[i]]

    # preserve any adjudicated_label already filled in from a previous run
    existing_adjudications = {}
    if ADJUDICATION_CSV.exists():
        with open(ADJUDICATION_CSV) as f:
            for row in csv.DictReader(f):
                existing_adjudications[row["item_id"]] = row["adjudicated_label"].strip()

    with open(ADJUDICATION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "annotator1_label", "annotator2_label", "adjudicated_label"])
        writer.writeheader()
        for item_id in disagreements:
            writer.writerow({
                "item_id": item_id,
                "annotator1_label": labels_a[item_id],
                "annotator2_label": labels_b[item_id],
                "adjudicated_label": existing_adjudications.get(item_id, ""),
            })
    print(f"{len(disagreements)} disagreements written to {ADJUDICATION_CSV} for adjudication")

    return labels_a, labels_b, result


def build_adjudicated_labels(labels_a, labels_b):
    """Agreed items keep their shared label; disagreements come from the filled adjudication CSV."""
    adjudicated = {i: label for i, label in labels_a.items() if labels_b[i] == label}

    if ADJUDICATION_CSV.exists():
        with open(ADJUDICATION_CSV) as f:
            for row in csv.DictReader(f):
                label = row["adjudicated_label"].strip().lower()
                if not label:
                    raise ValueError(
                        f"{ADJUDICATION_CSV}: item {row['item_id']} still needs an adjudicated_label"
                    )
                if label not in VALID_LABELS:
                    raise ValueError(f"{ADJUDICATION_CSV}: invalid adjudicated_label {label!r}")
                adjudicated[row["item_id"]] = label

    with open(FINAL_LABELS_PATH, "w") as f:
        json.dump(adjudicated, f, indent=2)
    return adjudicated


def judge_human_kappa(adjudicated):
    with open(JUDGE_LABELS_PATH) as f:
        judge_labels = json.load(f)

    item_ids = sorted(set(adjudicated.keys()) & set(judge_labels.keys()))
    result = cohens_kappa([adjudicated[i] for i in item_ids], [judge_labels[i] for i in item_ids])
    print(f"Judge-human kappa = {result['kappa']} ({interpret_kappa(result['kappa'])}), "
          f"observed={result['observed_agreement']}, n={result['n']}")
    print("Confusion matrix (rows=human, cols=judge):")
    for label in result["labels"]:
        print(f"  {label:14s} {result['confusion_matrix'][label]}")
    return result


if __name__ == "__main__":
    labels_a, labels_b, hh_result = human_human_kappa()

    if hh_result["kappa"] < 0.6:
        print("\nWARNING: human-human kappa is below 0.6 (substantial agreement threshold).")
        print("Per the protocol, revise the rubric and re-pilot before trusting judge-human kappa.\n")

    if ADJUDICATION_CSV.exists():
        with open(ADJUDICATION_CSV) as f:
            pending = [row for row in csv.DictReader(f) if not row["adjudicated_label"].strip()]
        if pending:
            print(f"\n{len(pending)} disagreements still need 'adjudicated_label' filled in "
                  f"{ADJUDICATION_CSV} before judge-human kappa can be computed. Stopping here.")
            raise SystemExit(0)

    adjudicated = build_adjudicated_labels(labels_a, labels_b)
    judge_human_kappa(adjudicated)
