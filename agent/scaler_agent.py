"""
The control agent: given the trace of (instances → CPU) readings so far,
decide how many instances to provision next to drive CPU toward the target.

The model never sees the hidden traffic load or the exponent in the CPU
formula — it has to infer the relationship from the readings, exactly like a
real operator tuning a cluster blind.
"""

import json
import re

from llm_client import LLMClient

TARGET_CPU = 15.0
BAND = (14.0, 16.0)

_SYSTEM_PROMPT = """You are an autonomous server auto-scaling control agent.

Goal: choose the number of server instances so the cluster's average CPU
utilization stabilizes at EXACTLY 15% (acceptable band: 14%-16%).

What you know about the (hidden) dynamics:
- More instances => lower CPU utilization. Fewer instances => higher CPU.
- The relationship is NON-LINEAR: roughly  CPU ≈ traffic / (instances ** k)
  with k a little above 1. You must estimate it from the readings.
- Background traffic drifts a little every iteration, so keep re-estimating
  from the MOST RECENT readings rather than trusting old ones.
- CPU is clamped to [1, 100]. A reading of exactly 100 means the cluster is
  SATURATED (true demand is higher than shown) — scale up aggressively.
  A reading of exactly 1 means heavily over-provisioned — scale down.

How to choose the next instance count:
- With a recent reading of CPU0 at N0 instances, a strong estimate for hitting
  15% is:  N_next = N0 * (CPU0 / 15) ** (1 / 1.2), rounded to an integer >= 1.
- When the last reading is already inside 14-16%, make only a tiny adjustment
  (or hold) — do not overshoot and leave the band.
- You have a hard limit on iterations, so converge fast but do not oscillate.

Respond with ONLY a JSON object, no explanation outside it:
{"instances": <integer >= 1>, "reasoning": "<one short sentence>"}"""


def _build_trace_text(history: list[dict]) -> str:
    if not history:
        return "(No readings yet — this is the first move.)"
    lines = ["Readings so far (instances -> resulting CPU):"]
    for r in history:
        flag = "  <-- in band" if BAND[0] <= r["cpu_utilization"] <= BAND[1] else ""
        lines.append(
            f"  iter {r['iteration']:2d}: {r['instances']:>4d} instances "
            f"-> CPU {r['cpu_utilization']:5.2f}%  (error {r['error']:+.2f}%){flag}"
        )
    return "\n".join(lines)


def decide_instances(
    client: LLMClient,
    history: list[dict],
    current_instances: int,
) -> tuple[int, str]:
    """Ask the model for the next instance count. Returns (instances, reasoning)."""
    trace = _build_trace_text(history)
    last = history[-1] if history else None

    user_message = (
        f"Target CPU: {TARGET_CPU:.0f}% (band {BAND[0]:.0f}-{BAND[1]:.0f}%).\n"
        f"Current instance count: {current_instances}.\n"
        + (f"Most recent CPU: {last['cpu_utilization']:.2f}% "
           f"(error {last['error']:+.2f}%).\n" if last else "")
        + f"\n{trace}\n\n"
        "Choose the next instance count to move CPU toward 15%. JSON only."
    )

    raw = client.complete(
        system=_SYSTEM_PROMPT,
        user=user_message,
        max_tokens=200,
    ).strip()

    # Strip markdown code fences if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Extract just the JSON object in case there's surrounding text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    # Replace JS-only non-finite values that are invalid JSON
    raw = re.sub(r":\s*NaN\b", ": 0", raw)
    raw = re.sub(r":\s*Infinity\b", ": 0", raw)
    raw = re.sub(r":\s*-Infinity\b", ": 0", raw)

    data = json.loads(raw)

    instances = int(round(float(data.get("instances", current_instances))))
    instances = max(1, instances)  # never provision 0 or negative
    reasoning = str(data.get("reasoning", "")).strip()

    return instances, reasoning
