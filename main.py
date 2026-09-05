"""An AI research agent: it searches, reads, and returns a structured summary.

Run it with `python main.py`, then ask a research question at the prompt.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from tools import save_research, search_web, search_wikipedia

load_dotenv()

# Windows consoles default to a legacy codepage (cp1253 here); research output
# is routinely non-ASCII, which would otherwise raise UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You are a research assistant.

Answer the user's query by gathering information with your tools:
- `search_web` for current events, news, and anything recent.
- `search_wikipedia` for background, definitions, and established facts.
- `save_research` only when the user explicitly asks you to save the results.

Search before you answer rather than relying on memory, and cite every source
you actually used. If the sources disagree or the evidence is thin, say so in
the summary instead of papering over it.
"""


class ResearchResponse(BaseModel):
    """The structured result of a research task."""

    topic: str = Field(description="The topic that was researched")
    summary: str = Field(description="A thorough summary of the findings")
    sources: list[str] = Field(description="URLs or article titles the findings came from")
    tools_used: list[str] = Field(description="Names of the tools used to gather information")


def build_agent():
    """Wire the model, the tools, and the output schema into an agent."""
    llm = ChatAnthropic(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
    )
    return create_agent(
        model=llm,
        tools=[search_web, search_wikipedia, save_research],
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(ResearchResponse),
    )


def trace(message) -> None:
    """Print a one-line trace of what the agent is doing (like verbose=True)."""
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            print(f"  -> {call['name']}({call['args']})")
    elif isinstance(message, ToolMessage):
        preview = " ".join(str(message.content).split())
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f"  <- {message.name}: {preview}")


def research(agent, query: str):
    """Stream the agent to completion, tracing tool calls as they happen."""
    final_state = None
    printed = 0
    for state in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        messages = state.get("messages", [])
        for message in messages[printed:]:
            trace(message)
        printed = len(messages)
        final_state = state
    return final_state


def show(result: ResearchResponse) -> None:
    print(f"\nTOPIC\n  {result.topic}")
    print(f"\nSUMMARY\n  {result.summary}")
    print("\nSOURCES")
    for source in result.sources:
        print(f"  - {source}")
    print("\nTOOLS USED")
    for name in result.tools_used:
        print(f"  - {name}")


def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    agent = build_agent()
    print(f"Research agent ready ({MODEL}). Ctrl-C to quit.\n")

    while True:
        try:
            query = input("What can I help you research? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return 0

        print()
        try:
            state = research(agent, query)
        except Exception as exc:
            print(f"\nThe agent failed: {type(exc).__name__}: {exc}\n")
            continue

        result = (state or {}).get("structured_response")
        if result is None:
            print("\nNo structured response was produced. Raw messages:")
            for message in (state or {}).get("messages", []):
                print(f"  {type(message).__name__}: {str(message.content)[:300]}")
        else:
            show(result)
        print()


if __name__ == "__main__":
    sys.exit(main())
