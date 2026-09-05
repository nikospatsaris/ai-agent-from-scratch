"""Tools the research agent can call.

Each function is exposed to the model with `@tool`. The docstring is what the
model reads when deciding whether to call it, so keep them precise.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import requests
from ddgs import DDGS
from langchain_core.tools import tool

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia returns 403 for requests without a descriptive User-Agent.
USER_AGENT = "PythonAIAgentFromScratch/1.0 (educational research agent)"
OUTPUT_FILE = Path(__file__).parent / "research_output.txt"


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for current information about a query.

    Use this for recent events, news, or anything that may have changed
    recently. Returns a numbered list of title, URL, and snippet.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:
        return f"Web search failed: {exc}"

    if not results:
        return f"No web results found for {query!r}."

    return "\n\n".join(
        f"{i}. {r.get('title', 'Untitled')}\n   {r.get('href', '')}\n   {r.get('body', '')}"
        for i, r in enumerate(results, start=1)
    )


@tool
def search_wikipedia(query: str, max_pages: int = 2) -> str:
    """Look up background information on Wikipedia.

    Use this for established facts, definitions, and history rather than
    breaking news. Returns the intro section of the best-matching articles.
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": max_pages,
        "prop": "extracts",
        "explaintext": 1,
        "exintro": 1,
        "exlimit": "max",
    }
    try:
        response = requests.get(
            WIKIPEDIA_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
    except Exception as exc:
        return f"Wikipedia lookup failed: {exc}"

    if not pages:
        return f"No Wikipedia article found for {query!r}."

    articles = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        title = page.get("title", "Untitled")
        extract = (page.get("extract") or "").strip() or "(no summary available)"
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        articles.append(f"## {title}\n{url}\n\n{extract}")

    return "\n\n---\n\n".join(articles)


@tool
def save_research(topic: str, content: str) -> str:
    """Append a research write-up to research_output.txt.

    Only call this when the user explicitly asks for the results to be saved.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"--- Research Output ---\n"
        f"Topic: {topic}\n"
        f"Timestamp: {timestamp}\n\n"
        f"{content}\n\n"
    )
    with OUTPUT_FILE.open("a", encoding="utf-8") as handle:
        handle.write(entry)

    return f"Saved {len(content)} characters to {OUTPUT_FILE.name}."
