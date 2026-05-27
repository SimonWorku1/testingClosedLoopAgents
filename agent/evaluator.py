"""LLM-based evaluator that scores research reports on a quality rubric."""

import json
import re
import time

import anthropic


def _api_call_with_backoff(fn, max_retries: int = 6):
    """Call fn(); on RateLimitError retry with exponential backoff (2s, 4s, 8s, …)."""
    delay = 2
    for attempt in range(max_retries):
        try:
            return fn()
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise
            print(f"    [evaluator rate limit] backing off {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)

RUBRIC = """
You are a strict academic editor evaluating a research report. Score the report on each dimension below (0–2 points each, total 0–10):

1. **Accuracy & Factual Correctness** (0-2): Are claims verifiable and accurate? Are sources cited?
2. **Depth & Comprehensiveness** (0-2): Does the report cover the topic thoroughly with specifics, data, and examples?
3. **Structure & Clarity** (0-2): Is the report well-organized with clear sections, logical flow, and readable prose?
4. **Multiple Perspectives** (0-2): Does the report present different viewpoints or dimensions of the topic?
5. **Actionable Conclusions** (0-2): Does the report draw meaningful insights and conclusions from the research?

Return your evaluation as JSON in exactly this format (no other text):
{
  "scores": {
    "accuracy": <integer 0, 1, or 2>,
    "depth": <integer 0, 1, or 2>,
    "structure": <integer 0, 1, or 2>,
    "perspectives": <integer 0, 1, or 2>,
    "conclusions": <integer 0, 1, or 2>
  },
  "total": <0-10>,
  "feedback": "<2-3 sentences on main strengths and specific weaknesses to improve>"
}
"""


def evaluate_report(
    client: anthropic.Anthropic,
    report: str,
    topic: str,
) -> tuple[int, str]:
    """
    Evaluate a report and return (score, feedback).
    Score is 0-10; feedback is a short string.
    """
    prompt = (
        f"Topic being researched: {topic}\n\n"
        f"=== REPORT TO EVALUATE ===\n{report}\n=== END REPORT ===\n\n"
        "Evaluate this report using the rubric. Return JSON only."
    )

    response = _api_call_with_backoff(
        lambda: client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=RUBRIC,
            messages=[{"role": "user", "content": prompt}],
        )
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
        total = int(data.get("total", 0))
        feedback = data.get("feedback", "")
        # Validate scores sum correctly
        scores = data.get("scores", {})
        computed = sum(scores.values())
        if abs(computed - total) > 1:
            total = computed
        return total, feedback
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract a number from the raw response
        match = re.search(r'"total"\s*:\s*(\d+)', raw)
        if match:
            return int(match.group(1)), raw[:200]
        return 0, f"Evaluation parse error: {raw[:200]}"
