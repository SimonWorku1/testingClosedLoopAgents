"""LLM-based evaluator that scores research reports on a quality rubric."""

import json
import re

import anthropic

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
    "accuracy": <0-2>,
    "depth": <0-2>,
    "structure": <0-2>,
    "perspectives": <0-2>,
    "conclusions": <0-2>
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

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=RUBRIC,
        messages=[{"role": "user", "content": prompt}],
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
