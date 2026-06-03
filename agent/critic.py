"""
Critic agent — reviews candidate code before benchmarking to catch
hardcoding, cheating, or obviously broken logic.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm_client import LLMClient

_SYSTEM = """\
You are a strict code reviewer. You review candidate Python implementations
of `count_unique(items)` before they are benchmarked.

You are looking for:
1. CORRECTNESS: Does the function correctly count unique elements for any
   hashable input (ints, strings, floats, None, booleans, tuples)?
2. CHEATING: Is the function hardcoded for specific inputs? Does it use
   lookup tables or pre-computed results?
3. INTERFACE: Is it named `count_unique` and accept a single argument `items`?
4. STANDARD LIBRARY ONLY: Does it avoid external dependencies (no numpy, pandas, etc.)?
5. GENERAL CORRECTNESS: Does it handle edge cases — empty list, single element,
   all duplicates, mixed types?

Output format — respond with ONLY valid JSON:
{
  "approved": true/false,
  "concerns": "one-sentence summary of any issues, or 'none' if approved",
  "suggestion": "one-sentence improvement hint if rejected, or empty string"
}
"""


def review_candidate(client: LLMClient, source: str) -> tuple[bool, str]:
    """
    Ask the critic to review candidate source code.
    Returns (approved: bool, feedback: str).
    """
    user = f"""\
Review this count_unique implementation:

```python
{source}
```

Respond with JSON only.
"""

    import json

    raw = client.complete(system=_SYSTEM, user=user, max_tokens=300)
    raw = raw.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(raw)
        approved = bool(data.get("approved", False))
        concerns = data.get("concerns", "no feedback")
        suggestion = data.get("suggestion", "")
        feedback = concerns
        if suggestion:
            feedback += f" Suggestion: {suggestion}"
        return approved, feedback
    except (json.JSONDecodeError, KeyError):
        # If the model returns garbled JSON, give benefit of the doubt
        # but flag it with a note
        if "true" in raw.lower() and "false" not in raw.lower():
            return True, "Critic returned non-JSON (assumed approved)"
        return False, f"Critic returned unparseable response: {raw[:200]}"
