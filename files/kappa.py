"""
Cohen's Kappa: chance-corrected agreement between two raters (e.g. LLM judge vs.
human-adjudicated label) on categorical labels.

kappa = (p_o - p_e) / (1 - p_e)
p_o = observed agreement, p_e = agreement expected by chance given each rater's
label distribution marginals.
"""

from collections import Counter
from typing import List, Tuple, Dict


def cohens_kappa(rater_a: List[str], rater_b: List[str]) -> Dict:
    """
    rater_a, rater_b: parallel lists of labels (same order, same items).
    Returns kappa, observed agreement, expected agreement, and the confusion matrix.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("Rating lists must be the same length (same items, same order).")

    n = len(rater_a)
    labels = sorted(set(rater_a) | set(rater_b))

    # Confusion matrix: confusion[a_label][b_label] = count
    confusion = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(rater_a, rater_b):
        confusion[a][b] += 1

    # Observed agreement: fraction on the diagonal.
    agree = sum(confusion[label][label] for label in labels)
    p_o = agree / n

    # Expected agreement from each rater's marginal label distribution.
    count_a = Counter(rater_a)
    count_b = Counter(rater_b)
    p_e = sum((count_a[label] / n) * (count_b[label] / n) for label in labels)

    if p_e == 1.0:
        kappa = 1.0  # degenerate case: both raters used only one label, in perfect sync
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    return {
        "kappa": round(kappa, 4),
        "observed_agreement": round(p_o, 4),
        "expected_agreement": round(p_e, 4),
        "n": n,
        "confusion_matrix": confusion,
        "labels": labels,
    }


def per_class_prf(confusion: Dict[str, Dict[str, int]], labels: List[str], pred_is_rows: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Per-class precision/recall/F1 from a cohens_kappa() confusion matrix.
    confusion[a][b]: a = rows, b = columns. pred_is_rows=True treats rows (rater_a) as the
    predicted label and columns (rater_b) as gold -- matches how this repo calls
    cohens_kappa(judge_labels, human_labels).
    """
    metrics = {}
    for label in labels:
        if pred_is_rows:
            tp = confusion[label][label]
            fp = sum(confusion[label][b] for b in labels) - tp
            fn = sum(confusion[a][label] for a in labels) - tp
        else:
            tp = confusion[label][label]
            fn = sum(confusion[label][b] for b in labels) - tp
            fp = sum(confusion[a][label] for a in labels) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    return metrics


def pabak(p_o: float) -> float:
    """Prevalence-and-bias-adjusted kappa: 2*p_o - 1. Intended for a 2-category collapse."""
    return round(2 * p_o - 1, 4)


def collapse_binary(labels: List[str], positive: Tuple[str, ...] = ("correct", "partial")) -> List[str]:
    """Collapse a 4-category rubric to a binary right/wrong label for binary-vs-categorical agreement."""
    return ["right" if label in positive else "wrong" for label in labels]


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) conventional interpretation scale -- cite this if you use it."""
    if kappa < 0:
        return "poor (worse than chance)"
    elif kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    else:
        return "almost perfect"


if __name__ == "__main__":
    # Sanity check: strong but imperfect agreement.
    human = ["correct", "correct", "partial", "incorrect", "hallucinated", "correct", "partial"]
    judge = ["correct", "partial", "partial", "incorrect", "hallucinated", "correct", "correct"]

    result = cohens_kappa(human, judge)
    print(f"kappa = {result['kappa']}  ({interpret_kappa(result['kappa'])})")
    print(f"observed agreement = {result['observed_agreement']}, expected = {result['expected_agreement']}")
    print(f"n = {result['n']}")
