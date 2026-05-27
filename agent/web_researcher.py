"""Web researcher agent that uses Claude with tool use to search and browse the web."""

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any

import anthropic

SEARCH_TOOL = {
    "name": "search_web",
    "description": (
        "Search the web for information about a topic. Returns a list of results "
        "with titles, URLs, and snippets."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

FETCH_TOOL = {
    "name": "fetch_page",
    "description": (
        "Fetch the text content of a web page. Returns the visible text content. "
        "Use this to read articles and pages found via search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
        },
        "required": ["url"],
    },
}

TOOLS = [SEARCH_TOOL, FETCH_TOOL]

# ---------------------------------------------------------------------------
# Lightweight web helpers (no third-party deps beyond anthropic)
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """Call the DuckDuckGo instant-answer / HTML search endpoint."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return [{"title": "Search error", "url": "", "snippet": str(exc)}]

    results = []
    # Parse result titles + URLs from the HTML (no BeautifulSoup dep)
    blocks = re.findall(
        r'class="result__title".*?href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )
    for url_raw, title_raw, snippet_raw in blocks[:max_results]:
        clean = lambda s: re.sub(r"<[^>]+>", "", s).strip()
        url_parsed = url_raw
        # DuckDuckGo wraps URLs in a redirect
        if "uddg=" in url_raw:
            m = re.search(r"uddg=([^&]+)", url_raw)
            if m:
                url_parsed = urllib.parse.unquote(m.group(1))
        results.append(
            {
                "title": clean(title_raw),
                "url": url_parsed,
                "snippet": clean(snippet_raw),
            }
        )

    if not results:
        # Fallback: grab any visible links
        for href, text in re.findall(r'href="(https?://[^"]+)"[^>]*>([^<]{10,80})', html):
            results.append({"title": text.strip(), "url": href, "snippet": ""})
            if len(results) >= max_results:
                break

    return results


def _fetch_page(url: str, max_chars: int = 6000) -> str:
    """Fetch a URL and return stripped text content."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text" not in content_type:
                return f"[Non-text content: {content_type}]"
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"[Fetch error: {exc}]"

    # Strip scripts/styles
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.I)
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "search_web":
        results = _duckduckgo_search(
            tool_input["query"], tool_input.get("max_results", 5)
        )
        return json.dumps(results, indent=2)
    elif tool_name == "fetch_page":
        return _fetch_page(tool_input["url"])
    return f"[Unknown tool: {tool_name}]"


# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    return """You are an expert research analyst. Your job is to research a topic thoroughly using web search and produce a high-quality, well-structured report.

Your report must:
1. Have a clear executive summary
2. Cover the topic in depth with multiple sections
3. Include specific facts, figures, and examples with sources cited inline
4. Present multiple perspectives where relevant
5. Have a conclusions/takeaways section
6. Be written in clear, professional prose

Use the search_web tool to find relevant sources and fetch_page to read them in detail.
Aim to consult at least 4-6 distinct sources before writing your final report.
Your final output should be ONLY the formatted report — no meta-commentary."""


def build_user_prompt(
    topic: str,
    agent_id: int,
    iteration: int,
    previous_iterations: list[dict],
) -> str:
    lines = [
        f"Research topic: {topic}",
        f"(You are Agent {agent_id + 1} in iteration {iteration + 1})",
        "",
    ]

    if previous_iterations:
        lines.append(
            "## Context from previous iterations\n"
            "Below are all reports produced so far and their quality scores. "
            "Use this context to write a BETTER report — address weaknesses, "
            "fill gaps, improve depth, and correct any inaccuracies.\n"
        )
        for iter_data in previous_iterations:
            iter_num = iter_data["iteration"]
            lines.append(f"### Iteration {iter_num} results\n")
            for res in iter_data["results"]:
                lines.append(
                    f"**Agent {res['agent_id'] + 1}** — Score: {res['score']}/10\n"
                    f"**Evaluator feedback:** {res.get('feedback', 'No feedback')}\n\n"
                    f"**Report:**\n{res['report']}\n\n---\n"
                )

    lines.append(
        "\nNow research the topic thoroughly and produce the best report you can. "
        "Return ONLY the final report text."
    )
    return "\n".join(lines)


def run_research_agent(
    client: anthropic.Anthropic,
    topic: str,
    agent_id: int,
    iteration: int,
    previous_iterations: list[dict],
    max_tool_rounds: int = 8,
) -> str:
    """Run a single research agent and return the final report text."""
    messages: list[dict] = [
        {
            "role": "user",
            "content": build_user_prompt(topic, agent_id, iteration, previous_iterations),
        }
    ]

    for _ in range(max_tool_rounds):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=build_system_prompt(),
            tools=TOOLS,
            messages=messages,
        )

        # Collect assistant message
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract text from final response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "[No text in response]"

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(
                        f"    [Agent {agent_id + 1}] Tool: {block.name}("
                        + json.dumps(block.input)[:120]
                        + ")"
                    )
                    result = _execute_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            time.sleep(0.3)  # be polite to upstream servers
        else:
            break

    # Fallback: extract whatever text exists
    for block in (response.content if response else []):
        if hasattr(block, "text"):
            return block.text
    return "[Agent did not produce a report]"
