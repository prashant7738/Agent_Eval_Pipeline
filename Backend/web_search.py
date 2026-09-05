"""
Thin wrapper around DuckDuckGo search (via the `ddgs` package) -- the "web search"
tool for the single-LLM-call agent described in roadmap.txt Day 1.

No API key required, which is why it was picked over a paid search API.
"""

from typing import List, Dict


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Return up to max_results {title, href, body} dicts for the query."""
    from ddgs import DDGS

    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        # Search backends occasionally rate-limit/hiccup -- degrade to no context
        # rather than crashing the whole agent run.
        print(f"[web_search] search failed for {query!r}: {e}")
        return []


def format_search_context(results: List[Dict[str, str]]) -> str:
    """Render search results as numbered snippets for the agent prompt."""
    if not results:
        return "(no search results available)"

    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"[{i}] {title}\n{body}\nSource: {href}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    results = web_search("capital of Australia", max_results=3)
    print(format_search_context(results))
