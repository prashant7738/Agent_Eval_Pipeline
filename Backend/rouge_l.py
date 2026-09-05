"""
ROUGE-L: reference-based similarity via Longest Common Subsequence (LCS).

Complexity: O(m*n) time, O(min(m,n)) space (row-reuse DP).
This is the exact algorithm to cite/analyze in your paper's Study 1 section.
"""

from typing import List


def _tokenize(text: str) -> List[str]:
    """Whitespace + lowercase tokenization. Swap for a smarter tokenizer if needed."""
    return text.lower().split()


def lcs_length(a: List[str], b: List[str]) -> int:
    """
    Standard LCS dynamic program, O(m*n) time, O(min(m,n)) space.

    dp[j] after processing row i = length of LCS of a[:i] and b[:j].
    We keep only the previous row (rolling array) instead of a full m*n table.
    """
    # Make 'b' the shorter sequence for less memory use.
    if len(a) < len(b):
        a, b = b, a
    m, n = len(a), len(b)

    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(candidate: str, reference: str, beta: float = 1.2) -> dict:
    """
    Returns precision, recall, and F-measure (as in Lin, 2004).

    beta > 1 weights recall higher than precision, matching the original ROUGE paper's
    default emphasis on reference coverage.
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "lcs_length": 0}

    lcs = lcs_length(cand_tokens, ref_tokens)

    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        beta_sq = beta ** 2
        f1 = ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "lcs_length": lcs,
    }


if __name__ == "__main__":
    # Quick correctness check against known behavior.
    ref = "the cat sat on the mat"
    cand_exact = "the cat sat on the mat"
    cand_partial = "a cat was sitting on the mat"
    cand_none = "completely unrelated text here"

    for name, cand in [("exact match", cand_exact), ("partial overlap", cand_partial), ("no overlap", cand_none)]:
        result = rouge_l(cand, ref)
        print(f"{name:20s} -> {result}")
