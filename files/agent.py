"""
The single agent config used for the whole study (per roadmap.txt Day 1: "use
one simple agent config -- a single LLM call with web search -- instead of two").

DuckDuckGo snippets (web_search.py) stuffed into one Groq chat completion.

Pipeline per task: search for context, feed the question + context to the LLM
in one call, ask for a short final answer in a fixed format so it can be
parsed for ROUGE-L scoring against the GAIA reference.
"""

import os
import re
from typing import Dict

from web_search import web_search, format_search_context

AGENT_MODEL = "llama-3.3-70b-versatile"

_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _groq_client


AGENT_PROMPT = """You are an AI agent answering a benchmark question. You may use the
web search results below, but rely on your own knowledge if they are not helpful.

Question: {question}

Web search results:
{context}

Give a short, direct final answer to the question. Do not explain your reasoning.
End your reply with exactly one line in this format:
FINAL ANSWER: <answer>"""


def _extract_final_answer(text: str) -> str:
    """Pull out the 'FINAL ANSWER: ...' line; fall back to the last non-empty line."""
    match = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def run_agent(
    question: str,
    model: str = None,
    temperature: float = 0.0,
    max_search_results: int = 5,
) -> Dict:
    """Run the single-call agent on one question: search -> one Groq call -> parsed answer."""
    results = web_search(question, max_results=max_search_results)
    context = format_search_context(results)
    prompt = AGENT_PROMPT.format(question=question, context=context)

    client = _get_groq_client()
    completion = client.chat.completions.create(
        model=model or AGENT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=300,
    )
    raw = completion.choices[0].message.content.strip()

    return {
        "question": question,
        "answer": _extract_final_answer(raw),
        "raw_response": raw,
        "search_results": results,
    }


if __name__ == "__main__":
    # Requires GROQ_API_KEY -- expected to fail without it.
    result = run_agent("What is the capital of Australia?")
    print("Answer:", result["answer"])
    print("Raw:", result["raw_response"])

