"""
Bradley-Terry model (MLE via Minorization-Maximization / Zermelo's algorithm)
and a simple Elo update rule, for converting pairwise judge comparisons into rankings.

Bradley-Terry: P(i beats j) = exp(b_i) / (exp(b_i) + exp(b_j))
Fitting via MM algorithm (Hunter, 2004) converges close to linearly in the number
of comparisons -- this is the complexity claim to cite in your paper.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

Comparison = Tuple[str, str]  # (winner, loser)


def fit_bradley_terry(
    comparisons: List[Comparison],
    items: List[str] = None,
    iterations: int = 200,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Fit Bradley-Terry strengths using the MM algorithm.

    comparisons: list of (winner, loser) tuples, one per pairwise judgment.
    Returns: dict of item -> strength score (not yet normalized to probabilities).
    """
    if items is None:
        items = sorted({x for pair in comparisons for x in pair})

    # wins[i] = number of times i won
    wins = defaultdict(int)
    # games[(i, j)] = number of times i and j were compared (either order)
    games = defaultdict(int)

    for winner, loser in comparisons:
        wins[winner] += 1
        games[frozenset((winner, loser))] += 1

    # Initialize all strengths equal.
    strength = {item: 1.0 for item in items}

    for _ in range(iterations):
        new_strength = {}
        max_delta = 0.0

        for i in items:
            denom = 0.0
            for j in items:
                if i == j:
                    continue
                n_ij = games.get(frozenset((i, j)), 0)
                if n_ij == 0:
                    continue
                denom += n_ij / (strength[i] + strength[j])

            if denom == 0:
                new_strength[i] = strength[i]  # no comparisons involving i; leave unchanged
            else:
                new_strength[i] = wins[i] / denom

            max_delta = max(max_delta, abs(new_strength[i] - strength[i]))

        # Normalize to prevent unbounded growth (scale invariant model).
        total = sum(new_strength.values())
        n = len(items)
        strength = {k: v * n / total for k, v in new_strength.items()}

        if max_delta < tol:
            break

    return strength


def win_probability(strength: Dict[str, float], i: str, j: str) -> float:
    """P(i beats j) under the fitted Bradley-Terry model."""
    return strength[i] / (strength[i] + strength[j])


def elo_update(rating_a: float, rating_b: float, a_won: bool, k: float = 32.0) -> Tuple[float, float]:
    """
    Standard sequential Elo update. O(1) per comparison, O(1) space per pair.
    Returns updated (rating_a, rating_b).
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a

    score_a = 1.0 if a_won else 0.0
    score_b = 1.0 - score_a

    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)
    return new_a, new_b


def run_elo(comparisons: List[Comparison], items: List[str] = None, k: float = 32.0, start: float = 1000.0) -> Dict[str, float]:
    """Run sequential Elo updates over a list of (winner, loser) comparisons, in order."""
    if items is None:
        items = sorted({x for pair in comparisons for x in pair})
    ratings = {item: start for item in items}

    for winner, loser in comparisons:
        new_w, new_l = elo_update(ratings[winner], ratings[loser], a_won=True, k=k)
        ratings[winner], ratings[loser] = new_w, new_l

    return ratings


if __name__ == "__main__":
    # Sanity check: agent_A should clearly outrank agent_B if it wins most comparisons.
    comparisons = [
        ("agent_A", "agent_B"), ("agent_A", "agent_B"), ("agent_A", "agent_B"),
        ("agent_A", "agent_B"), ("agent_B", "agent_A"),  # A wins 4/5
    ]

    bt_strength = fit_bradley_terry(comparisons)
    print("Bradley-Terry strengths:", {k: round(v, 3) for k, v in bt_strength.items()})
    print("P(A beats B):", round(win_probability(bt_strength, "agent_A", "agent_B"), 3))

    elo_ratings = run_elo(comparisons)
    print("Elo ratings:", {k: round(v, 1) for k, v in elo_ratings.items()})
