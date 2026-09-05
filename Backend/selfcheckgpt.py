"""
SelfCheckGPT-style consistency check (Manakul et al., 2023), simplified.

Original method uses NLI (entailment/contradiction) between sentences across samples.
This version uses token-overlap (Jaccard) similarity between samples as a cheaper proxy --
note this simplification explicitly in your paper's methodology/limitations section,
it is a legitimate scoping decision but should not be presented as identical to the
original NLI-based scorer.

Idea: sample k responses to the same prompt at temperature > 0. If the model "knows"
the fact, samples should agree. Low average pairwise similarity = high hallucination risk.

Cost: k model calls per prompt before any comparison happens -- this is the complexity
tradeoff to cite (SelfCheckGPT is the expensive stage in the pipeline, hence why it's
only run on a subset in Study 1).
"""

import os
from typing import List

SAMPLING_MODEL = "llama-3.1-8b-instant"  # cheaper/faster model is fine for repeated sampling

_client = None


def _get_client():
    """Lazy-init so this module can be imported (and consistency_score tested) without an API key."""
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def sample_responses(prompt: str, k: int = 5, model: str = SAMPLING_MODEL, temperature: float = 1.0) -> List[str]:
    """Draw k stochastic samples for the same prompt. O(k) model calls."""
    client = _get_client()
    samples = []
    for _ in range(k):
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200,
        )
        samples.append(completion.choices[0].message.content.strip())
    return samples


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two texts."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def consistency_score(samples: List[str]) -> float:
    """
    Average pairwise Jaccard similarity across all sample pairs, in [0, 1].
    Higher = more consistent = lower hallucination risk under this proxy.
    O(k^2) comparisons for k samples -- negligible next to the O(k) generation cost.
    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to measure consistency.")

    pairs = 0
    total = 0.0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            total += _jaccard(samples[i], samples[j])
            pairs += 1

    return round(total / pairs, 4)


def selfcheck(prompt: str, k: int = 5, model: str = SAMPLING_MODEL) -> dict:
    """Full pipeline: sample k responses, return them plus the consistency score."""
    samples = sample_responses(prompt, k=k, model=model)
    score = consistency_score(samples)
    return {"prompt": prompt, "k": k, "samples": samples, "consistency_score": score}


if __name__ == "__main__":
    # Requires GROQ_API_KEY -- expected to fail without it.
    result = selfcheck("What is the capital of Australia? Answer in one sentence.", k=3)
    print(f"Consistency score: {result['consistency_score']}")
    for i, s in enumerate(result["samples"]):
        print(f"Sample {i+1}: {s}")
