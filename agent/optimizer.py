"""
Optimizer agent — generates candidate count_unique implementations.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm_client import LLMClient

_SYSTEM = """\
You are an expert Python performance engineer.
Your job is to rewrite `count_unique(items)` to be as fast as possible
while remaining 100% correct.

Rules:
- The function MUST be named `count_unique` and accept a single positional argument `items`.
- It MUST return an integer (the number of unique values in `items`).
- No imports outside the Python standard library.
- No hardcoding results for specific inputs.
- The implementation must be general — it must work for any hashable element type
  (ints, strings, floats, None, booleans, tuples, etc.).
- Do NOT use unhashable types (lists, dicts) as set/dict keys.
- Do NOT use any external libraries.

Output format:
Return ONLY a Python code block — nothing else, no explanation, no markdown,
just the raw Python source starting with `def count_unique(items):`.
"""


def generate_candidate(
    client: LLMClient,
    iteration: int,
    previous_results: list[dict],
    critic_feedback: str = "",
) -> str:
    """
    Ask the optimizer LLM to generate an improved count_unique implementation.
    Returns raw Python source code.
    """
    history_lines = []
    for r in previous_results[-5:]:  # last 5 to keep context tight
        status = "ACCEPTED" if r.get("accepted") else "rejected"
        history_lines.append(
            f"  Iteration {r['iteration']}: {status}, "
            f"runtime={r.get('runtime_ms', '?'):.3f}ms, "
            f"improvement={r.get('improvement_pct', 0):.1f}%"
        )
    history_summary = "\n".join(history_lines) if history_lines else "  (no previous attempts)"

    critic_section = ""
    if critic_feedback:
        critic_section = f"\nCritic feedback on the last candidate:\n{critic_feedback}\n"

    user = f"""\
Iteration {iteration + 1}: Optimize `count_unique(items)`.

Baseline implementation (O(n²) — this is what you are replacing):
```python
def count_unique(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return len(result)
```

Progress so far:
{history_summary}
{critic_section}
Write a faster version. The simplest correct optimization is using a set.
Push further if you can — explore frozenset tricks, early-exit patterns,
or numpy if the input is numeric (but standard library only).

Return ONLY the Python function source, no markdown fences, no explanation.
"""

    raw = client.complete(system=_SYSTEM, user=user, max_tokens=800)
    return _extract_code(raw)


def _extract_code(text: str) -> str:
    """Strip markdown fences if the model wrapped the code."""
    lines = text.strip().splitlines()

    # Remove leading ```python or ``` fence
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # Remove trailing ``` fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()
