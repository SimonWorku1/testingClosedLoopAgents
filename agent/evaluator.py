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
You are a brutally strict academic peer reviewer for a top-tier journal. Most reports are mediocre. A score of 2 on any dimension must be EARNED — it requires exceptional, publication-quality work on that dimension. When in doubt, score lower.

Score the report on each dimension (0–2 points each, total 0–10):

1. **Accuracy & Factual Correctness** (0-2):
   - 0: Vague, unverifiable, or clearly wrong claims.
   - 1: Some specific facts with partial citations, but gaps or unverified claims remain.
   - 2: Every major claim is specific, verifiable, and cited with a named source (author, publication, date, or URL). No hand-wavy generalities.

2. **Depth & Comprehensiveness** (0-2):
   - 0: Surface-level overview, no data or examples.
   - 1: Covers the main points but missing key subtopics, quantitative data, or concrete examples.
   - 2: Exhaustive coverage with specific statistics, named studies, quantitative comparisons, and concrete real-world examples for every major claim.

3. **Structure & Clarity** (0-2):
   - 0: Disorganized, hard to follow, or wall-of-text prose.
   - 1: Adequate structure but transitions are rough, sections unbalanced, or prose unclear in places.
   - 2: Flawless organization with a clear executive summary, well-scoped sections, smooth transitions, and polished professional prose throughout.

4. **Multiple Perspectives** (0-2):
   - 0: Single viewpoint only.
   - 1: Mentions other views but does not develop them fairly or with evidence.
   - 2: Fairly presents at least 3 distinct, well-evidenced perspectives (e.g., scientific consensus, dissenting research, industry, regulatory, consumer) with named sources for each.

5. **Actionable Conclusions** (0-2):
   - 0: No conclusions, or only restates what was already said.
   - 1: Draws some insights but they are generic or obvious.
   - 2: Derives specific, non-obvious, evidence-backed recommendations or takeaways that a decision-maker could act on directly.

A score of 9–10 should be rare and reflect genuinely exceptional research. Score 8 should require solid work across all dimensions with no significant gaps. Be harsh — it forces better reports.

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
  "feedback": "<3-4 sentences: name the specific gaps, missing sources, or weak sections that prevented a higher score>"
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
