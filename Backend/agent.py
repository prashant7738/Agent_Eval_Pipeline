"""
The agent config(s) used for the study. Default is Groq (per roadmap.txt Day 1:
"use one simple agent config -- a single LLM call with web search -- instead of
two"); an optional Gemini backend was added later for a backend comparison.

Groq path: DuckDuckGo snippets (web_search.py) stuffed into one Groq chat completion.
Gemini path: native Google Search grounding, falling back to the same DuckDuckGo
snippet-stuffing approach if grounding fails (free-tier quota, transient errors, etc).

Pipeline per task: search for context, feed the question + context to the LLM
in one call, ask for a short final answer in a fixed format so it can be
parsed for ROUGE-L scoring against the GAIA reference.
"""

import os
import re
from typing import Dict

from web_search import web_search, format_search_context

AGENT_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-3.5-flash-lite"  # gemini-3.5-flash-lite is more robust against internal function call errors than 3.6

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


def _extract_final_answer(text: str) -> str:
    """Pull out the 'FINAL ANSWER: ...' line; fall back to the last non-empty line."""
    match = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def run_agent(
    question: str,
    backend: str = "groq",
    model: str = None,
    temperature: float = 0.0,
    max_search_results: int = 5,
) -> Dict:
    """Run the single-call agent on one question: search -> one LLM call -> parsed answer."""
    if backend == "groq":
        return _run_agent_groq(question, model=model, temperature=temperature, max_search_results=max_search_results)
    elif backend == "gemini":
        return _run_agent_gemini(question, model=model, temperature=temperature, max_search_results=max_search_results)
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected 'groq' or 'gemini')")


def _run_agent_groq(
    question: str,
    model: str = None,
    temperature: float = 0.0,
    max_search_results: int = 5,
) -> Dict:
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


def _run_agent_gemini(
    question: str,
    model: str = None,
    temperature: float = 0.0,
    max_search_results: int = 5,
    use_grounding: bool = False,
) -> Dict:
    """DuckDuckGo snippet-stuffing by default (same approach as the Groq path), so the
    two backends are directly comparable. Native Google Search grounding is opt-in via
    use_grounding=True, but confirmed 429/RESOURCE_EXHAUSTED on this free-tier key as of
    2026-08-09 -- leave it off by default so it doesn't burn scarce quota on every call."""
    from google.genai import types

    client = _get_gemini_client()
    gemini_model = model or GEMINI_MODEL

    if use_grounding:
        try:
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            response = client.models.generate_content(
                model=gemini_model,
                contents=AGENT_PROMPT.format(question=question, context="(using live Google Search grounding)"),
                config=types.GenerateContentConfig(
                    tools=[grounding_tool],
                    temperature=temperature,
                    max_output_tokens=300,
                ),
            )
            if response.text is None:
                raise ValueError(f"empty grounded response, finish_reason={response.candidates[0].finish_reason if response.candidates else None!r}")
            raw = response.text.strip()
            return {
                "question": question,
                "answer": _extract_final_answer(raw),
                "raw_response": raw,
                "search_results": [],
            }
        except Exception as e:
            print(f"[agent] Gemini grounding failed ({e!r}), falling back to DuckDuckGo snippets")

    results = web_search(question, max_results=max_search_results)
    context = format_search_context(results)
    prompt = AGENT_PROMPT.format(question=question, context=context)

    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=300),
    )
    if response.text is None:
        finish_reason = response.candidates[0].finish_reason if response.candidates else None
        print(f"[agent] Gemini fallback also returned no text, finish_reason={finish_reason!r}")
        raw = ""
    else:
        raw = response.text.strip()

    return {
        "question": question,
        "answer": _extract_final_answer(raw) if raw else "",
        "raw_response": raw,
        "search_results": results,
    }


if __name__ == "__main__":
    # Requires GROQ_API_KEY -- expected to fail without it.
    result = run_agent("What is the capital of Australia?")
    print("Answer:", result["answer"])
    print("Raw:", result["raw_response"])

