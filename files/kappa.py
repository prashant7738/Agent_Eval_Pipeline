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
