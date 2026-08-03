"""
LLM judge using Groq's free API (OpenAI-compatible endpoint).

Setup:
  1. Get a free API key: https://console.groq.com/keys
  2. pip install groq
  3. export GROQ_API_KEY="your-key-here"

Free-tier models worth using here (check console.groq.com for current list/limits):
  - "llama-3.3-70b-versatile"   (strong general judge, good default)
  - "llama-3.1-8b-instant"      (much faster/cheaper, use for SelfCheckGPT sampling)

If you'd rather run fully local with zero API limits, see ollama_judge.py notes
at the bottom of this file.
"""

import os

JUDGE_MODEL = "llama-3.3-70b-versatile"

_client = None


def _get_client():
    """Lazy-init so this module can be imported without an API key already set."""
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def pairwise_compare(task: str, response_a: str, response_b: str, model: str = JUDGE_MODEL) -> str:
    """
    Ask the judge which response better completes the task.
    Returns "A", "B", or "TIE".

    Note: swap A/B order across calls in your experiment loop to control for
    position bias -- this is a known LLM-as-judge failure mode, mention it in
    your threats-to-validity section.
    """
    prompt = f"""You are evaluating two AI agent responses to the same task.

Task: {task}

Response A: {response_a}

Response B: {response_b}

Which response better and more accurately completes the task? Consider correctness,
completeness, and whether claims are supported. Reply with EXACTLY one word:
"A", "B", or "TIE". Do not explain."""

    client = _get_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )
    verdict = completion.choices[0].message.content.strip().upper()

    if "A" in verdict and "B" not in verdict:
        return "A"
    elif "B" in verdict and "A" not in verdict:
        return "B"
    return "TIE"


RUBRIC_LABELS = ["correct", "partial", "incorrect", "hallucinated"]

RUBRIC_PROMPT = """You are labeling an AI agent's response using this exact rubric:

- correct: matches the reference or task intent with no unsupported claims
- partial: the main answer is right but includes a minor unsupported or extraneous claim
- incorrect: the main answer does not match the reference or fails the task
- hallucinated: contains a specific factual claim not supported by the reference or task context

Task: {task}
Reference answer: {reference}
Agent response: {response}

Reply with EXACTLY one word from: correct, partial, incorrect, hallucinated."""


def rubric_label(task: str, reference: str, response: str, model: str = JUDGE_MODEL) -> str:
    """Label a single response using the fixed 4-category rubric. Returns one of RUBRIC_LABELS."""
    prompt = RUBRIC_PROMPT.format(task=task, reference=reference, response=response)

    client = _get_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )
    raw = completion.choices[0].message.content.strip().lower()

    for label in RUBRIC_LABELS:
        if label in raw:
            return label
    return "incorrect"  # fallback if the model doesn't follow format; log these cases


if __name__ == "__main__":
    # Requires GROQ_API_KEY to be set -- this will fail without it, that's expected.
    task = "What is the capital of France?"
    verdict = pairwise_compare(task, "The capital of France is Paris.", "I think it might be Lyon.")
    print("Pairwise verdict:", verdict)

    label = rubric_label(task, reference="Paris", response="The capital of France is Paris.")
    print("Rubric label:", label)
