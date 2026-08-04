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
GEMINI_MODEL = "gemini-flash-latest"  # only model with nonzero free-tier quota on this
# key/project (20 requests/day); gemini-2.0-flash and others return quota limit: 0.

_groq_client = None
_gemini_client = None
_gemini_grounding_warned = False


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

GEMINI_GROUNDED_PROMPT = """You are an AI agent answering a benchmark question. Use Google Search
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


def _gemini_text(response) -> str:
    """
    response.text can be None -- e.g. gemini-flash-latest enables "thinking" by
    default, and if the thinking tokens eat the whole output budget the model
    never emits a final answer. Log the reason and return "" instead of crashing.
    """
    if response.text is not None:
        return response.text.strip()

    reason = None
    if response.candidates:
        reason = response.candidates[0].finish_reason
    print(f"[gemini] empty response text (finish_reason={reason}); treating as no answer")
    return ""


def _run_agent_gemini(question: str, model: str, temperature: float, max_search_results: int) -> Dict:
    """
    Try Gemini's native Google Search grounding first. That tool is gated behind
    billing being enabled on the underlying Google Cloud project even on the
    free API tier, so on quota/permission errors this falls back to the same
    DuckDuckGo-snippet prompt used by the Groq backend -- still tests whether a
    stronger model helps given the same search input, without requiring billing.
    """
    from google.genai import types

    # Keep config minimal -- adding max_output_tokens/thinking_config here was
    # observed to sometimes trigger finish_reason=MALFORMED_FUNCTION_CALL with
    # empty text on this model, even with no tools passed. Bare temperature-only
    # config matches what was verified working in manual smoke tests.
    gen_config_kwargs = dict(temperature=temperature)

    client = _get_gemini_client()
    try:
        response = client.models.generate_content(
            model=model,
            contents=GEMINI_GROUNDED_PROMPT.format(question=question),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                **gen_config_kwargs,
            ),
        )
        raw = _gemini_text(response)
        if raw:
            return {
                "question": question,
                "answer": _extract_final_answer(raw),
                "raw_response": raw,
                "search_results": [],  # grounding metadata is on response.candidates[0], not surfaced here
                "grounded": True,
            }
    except Exception as e:
        global _gemini_grounding_warned
        if not _gemini_grounding_warned:
            print(f"[gemini] native grounding unavailable ({e}); falling back to DuckDuckGo snippets for all calls")
            _gemini_grounding_warned = True

    results = web_search(question, max_results=max_search_results)
    context = format_search_context(results)
    prompt = AGENT_PROMPT.format(question=question, context=context)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**gen_config_kwargs),
    )
    raw = _gemini_text(response)

    return {
        "question": question,
        "answer": _extract_final_answer(raw),
        "raw_response": raw,
        "search_results": results,
        "grounded": False,
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
        return _run_agent_gemini(question, model or GEMINI_MODEL, temperature, max_search_results)
    else:
        raise ValueError(f"Unknown backend {backend!r}, expected 'groq' or 'gemini'")


if __name__ == "__main__":
    # Requires GROQ_API_KEY -- expected to fail without it.
    result = run_agent("What is the capital of Australia?")
    print("Answer:", result["answer"])
    print("Raw:", result["raw_response"])

