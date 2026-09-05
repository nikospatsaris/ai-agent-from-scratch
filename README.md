# Python AI Agent From Scratch

A research agent that takes a question, gathers information with tools, and
returns a validated, structured answer.

Rebuilt from [techwithtim/PythonAIAgentFromScratch](https://github.com/techwithtim/PythonAIAgentFromScratch)
on the current LangChain API — see [What changed](#what-changed-from-the-original)
for why the original code no longer runs.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your key from
[console.anthropic.com](https://console.anthropic.com/settings/keys):

```
ANTHROPIC_API_KEY="sk-ant-..."
```

## Run

```bash
python main.py
```

```
What can I help you research? the 2024 Nobel prize in physics

  -> search_web({'query': '2024 Nobel Prize physics winners'})
  <- search_web: 1. The Nobel Prize in Physics 2024 - NobelPrize.org ...

TOPIC
  2024 Nobel Prize in Physics
SUMMARY
  Awarded jointly to John J. Hopfield and Geoffrey E. Hinton ...
SOURCES
  - https://www.nobelprize.org/prizes/physics/2024/summary/
TOOLS USED
  - search_web
```

Type `exit` or press Ctrl-C to quit.

## How it works

| File | Role |
|---|---|
| `main.py` | Defines the output schema, builds the agent, runs the prompt loop |
| `tools.py` | The three tools the model can call |

The agent loop is `model -> tools -> model -> ...` until the model stops
requesting tools. `create_agent` runs that loop; you supply the model, the
tools, and the shape of the answer.

**Tools** (`tools.py`) are plain Python functions wrapped in `@tool`. The
docstring is the tool description the model reads to decide when to call it,
so the docstrings are part of the program, not just comments.

- `search_web` — DuckDuckGo, for current events
- `search_wikipedia` — Wikipedia REST API, for background facts
- `save_research` — appends a write-up to `research_output.txt`

**Structured output.** `ResearchResponse` is a Pydantic model, passed as
`response_format=ToolStrategy(ResearchResponse)`. The model returns the answer
as a tool call matching that schema, and LangChain validates it before handing
it back on `state["structured_response"]` — so you get a typed object, not a
string you have to parse and hope about.

**Adaptive thinking** is on (`thinking={"type": "adaptive"}`), letting the
model reason before deciding which tools to call.

## What changed from the original

The original was written in late 2024 and no longer runs as published:

| Original | Problem | This version |
|---|---|---|
| `from langchain.agents import create_tool_calling_agent, AgentExecutor` | Removed from `langchain` in 1.0 (now in `langchain-classic`) | `create_agent` |
| `PydanticOutputParser` + `parser.parse(raw["output"][0]["text"])` | Format instructions in the prompt; parsing indexes into a raw content block and throws on any deviation | `response_format` — schema-validated by the framework |
| `langchain-community` tools | Package is [officially sunset](https://github.com/langchain-ai/langchain-community/issues/674) | Tools written directly against each API |
| `wikipedia` package | Last released 2014; sends no User-Agent, so Wikimedia now returns **403** | Wikipedia REST API with a proper User-Agent |
| `duckduckgo-search` | Renamed to `ddgs` | `ddgs` |
| `claude-3-5-sonnet-20241022` | Superseded | `claude-opus-5` |
| `from langchain_openai import ChatOpenAI` | Imported but never used; forces an unneeded dependency | Removed |
| Unpinned `requirements.txt` | Resolves to versions the code can't run against | Pinned |

One addition of my own: `main.py` reconfigures stdout to UTF-8. Windows
consoles default to a legacy codepage, and research summaries are routinely
non-ASCII — without it, printing a result containing something like "née"
crashes with `UnicodeEncodeError`.

## Extending it

Add a tool by writing a function in `tools.py` with `@tool` and a clear
docstring, then adding it to the `tools=[...]` list in `build_agent()`. Add a
field to the answer by adding it to `ResearchResponse`.
