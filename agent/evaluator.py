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
You are the harshest peer reviewer at a top-tier academic journal. You reject 80% of submissions on first review. Your default is to find what's missing, not what's present. Most reports deserve 3–5/10 on first pass. A score of 8+ is reserved for work that could be published as-is.

When scoring, assume the reader is an expert who will notice every missing citation, every unsupported claim, every shallow section. If you cannot point to a SPECIFIC strength on a dimension, score it 0 or 1. Default to the LOWER score when uncertain.

Score each dimension (0–2 points, total 0–10):

1. **Accuracy & Citation Quality** (0-2):
   - 0: Any unsourced claim, vague attribution ("studies show"), or made-up statistic.
   - 1: Most major claims cited, but with weak sources (Wikipedia, blogs, undated articles) OR missing inline citations on key numbers.
   - 2: Every quantitative claim and every non-trivial assertion has an inline citation to a PRIMARY source (peer-reviewed paper, official agency report, named expert with credentials). Sources include publication date and ideally a URL. At least 6 distinct primary sources cited.

2. **Depth & Specificity** (0-2):
   - 0: General overview anyone could write without research.
   - 1: Some specifics, but mostly summary-level. Missing quantitative comparisons, missing key subtopics, or examples are illustrative rather than evidentiary.
   - 2: Every claim is backed by SPECIFIC numbers (percentages, dollar figures, sample sizes, effect sizes, dates) AND concrete examples (named studies, named products, named events, named people). Includes at least one quantitative comparison or trend analysis with data points.

3. **Structure, Clarity & Rigor** (0-2):
   - 0: Disorganized, repetitive, or unclear.
   - 1: Adequate structure but uneven section depth, weak transitions, occasional jargon without definition, or prose that reads like notes rather than finished writing.
   - 2: Executive summary that genuinely summarises (not just restates the topic). Logical section ordering. Each section is substantive (no filler). Defines technical terms. Prose is publication-ready — no hedging filler, no AI-isms like "it is important to note", "in conclusion", "delve into", etc.

4. **Multiple Perspectives & Counterpoints** (0-2):
   - 0: Single viewpoint, or "balanced" only in lip service.
   - 1: Mentions 2 perspectives but one is clearly favoured; counterpoints are strawmen or undeveloped.
   - 2: Presents at least 4 distinct, well-evidenced perspectives (e.g., scientific consensus, dissenting research, industry, regulatory, consumer/public health, historical) with named sources for EACH. Explicitly engages with the strongest counter-argument to the main thesis.

5. **Actionable, Non-Obvious Insights** (0-2):
   - 0: No conclusions, or conclusions that merely restate the body.
   - 1: Draws conclusions but they are generic ("more research is needed", "consumers should be informed") or could have been written without the research.
   - 2: Derives SPECIFIC, NON-OBVIOUS insights that emerge only from synthesising the evidence presented. Includes concrete recommendations for at least two distinct audiences (e.g., regulators, consumers, researchers) that are directly traceable to specific findings in the report. Identifies open questions the evidence cannot yet resolve.

CRITICAL CALIBRATION:
- A polished, well-structured report with citations but generic conclusions is at most 6/10.
- A report missing inline citations on quantitative claims cannot score above 1 on Accuracy, regardless of how well-written it is.
- A report that does not explicitly engage with counter-arguments cannot score 2 on Perspectives.
- A report whose conclusions could have been written without the research cannot score above 1 on Insights.
- If you find yourself wanting to give a high score because the report is "well-written" or "comprehensive", that is not enough. Check each dimension's bar above.

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
  "feedback": "<4-6 sentences naming the SPECIFIC missing citations, unsupported claims, weak sections, missing perspectives, or generic conclusions that prevented full marks. Quote specific phrases from the report. Be ruthless and concrete.>"
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
