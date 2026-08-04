"""
The single agent config used for the whole study (per roadmap.txt Day 1: "use
one simple agent config -- a single LLM call with web search -- instead of two").

Two backends are supported:
  - "groq" (default): DuckDuckGo snippets (web_search.py) stuffed into one Groq
    chat completion. This is what Study 1's saved results were generated with.
  - "gemini": Gemini's native Google Search grounding tool, which runs real
    iterative search instead of a single snippet dump. Needs GEMINI_API_KEY and
    `pip install google-genai`. Not yet used for any saved results -- switching
    backend for future runs does not change data already in files/data/.

Pipeline per task: search for context (backend-dependent), feed the question +
context to the LLM in one call, ask for a short final answer in a fixed format
so it can be parsed for ROUGE-L scoring against the GAIA reference.
"""

import os
import re
from typing import Dict

from web_search import web_search, format_search_context

AGENT_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"

_groq_client = None
_gemini_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _groq_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _gemini_client


AGENT_PROMPT = """You are an AI agent answering a benchmark question. You may use the
web search results below, but rely on your own knowledge if they are not helpful.

Question: {question}

Web search results:
{context}

Give a short, direct final answer to the question. Do not explain your reasoning.
End your reply with exactly one line in this format:
FINAL ANSWER: <answer>"""

GEMINI_PROMPT = """You are an AI agent answering a benchmark question. Use Google Search
to look up whatever facts you need -- do multiple searches if the question requires
combining information from more than one source.

Question: {question}

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


def _run_agent_groq(question: str, model: str, temperature: float, max_search_results: int) -> Dict:
    results = web_search(question, max_results=max_search_results)
    context = format_search_context(results)
    prompt = AGENT_PROMPT.format(question=question, context=context)

    client = _get_groq_client()
    completion = client.chat.completions.create(
        model=model,
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


def _run_agent_gemini(question: str, model: str, temperature: float) -> Dict:
    from google.genai import types

    client = _get_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=GEMINI_PROMPT.format(question=question),
        config=types.GenerateContentConfig(
            temperature=temperature,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    raw = response.text.strip()

    return {
        "question": question,
        "answer": _extract_final_answer(raw),
        "raw_response": raw,
        "search_results": [],  # grounding metadata is on response.candidates[0], not surfaced here
    }


def run_agent(
    question: str,
    model: str = None,
    temperature: float = 0.0,
    max_search_results: int = 5,
    backend: str = "groq",
) -> Dict:
    """
    Run the single-call agent on one question: search -> one LLM call -> parsed answer.
    backend: "groq" (DuckDuckGo snippets, used for Study 1's saved results) or
    "gemini" (native Google Search grounding).
    """
    if backend == "groq":
        return _run_agent_groq(question, model or AGENT_MODEL, temperature, max_search_results)
    elif backend == "gemini":
        return _run_agent_gemini(question, model or GEMINI_MODEL, temperature)
    else:
        raise ValueError(f"Unknown backend {backend!r}, expected 'groq' or 'gemini'")


if __name__ == "__main__":
    # Requires GROQ_API_KEY -- expected to fail without it.
    result = run_agent("What is the capital of Australia?")
    print("Answer:", result["answer"])
    print("Raw:", result["raw_response"])

