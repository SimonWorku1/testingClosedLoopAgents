import json
import anthropic

CHANNELS = ["search", "social", "video", "display"]

_SYSTEM_PROMPT = """You are a digital advertising optimization agent allocating a daily budget across 4 channels to maximize total user sign-ups.

Channels: search, social, video, display

Rules:
- Your allocations must sum to EXACTLY the daily budget given. Not one cent more or less.
- Maintain at least $5 on every channel each day so you keep observing its performance.
- SATURATION WARNING: Concentrating too much spend on a single channel in one day sharply reduces its efficiency. Spreading spend avoids this penalty.
- Balance exploration (keep watching all channels) with exploitation (shift more budget toward the lowest cost-per-acquisition channels).
- The best-looking channel on day 1 may degrade quickly under heavy spend — stay adaptive.

Respond with ONLY a valid JSON object, no explanation:
{"search": <number>, "social": <number>, "video": <number>, "display": <number>}"""


HISTORY_WINDOW = 5  # days of recent detail sent to Claude per call


def _build_history_text(channel_history: list[dict], daily_budget: float) -> str:
    if not channel_history:
        return "(No history yet — this is Day 1.)"

    # Cumulative stats across ALL days (computed from full history on disk)
    totals = {ch: {"spend": 0.0, "sign_ups": 0} for ch in CHANNELS}
    for record in channel_history:
        for ch in CHANNELS:
            data = record["channel_results"].get(ch, {})
            totals[ch]["spend"] += data.get("spend", 0)
            totals[ch]["sign_ups"] += data.get("sign_ups", 0)

    lines = [f"Cumulative stats across all {len(channel_history)} days:"]
    for ch in CHANNELS:
        t = totals[ch]
        if t["sign_ups"] > 0:
            avg_cpa = t["spend"] / t["sign_ups"]
            lines.append(f"  {ch}: ${t['spend']:.0f} spent, {t['sign_ups']} sign-ups, avg CPA ${avg_cpa:.1f}")
        else:
            lines.append(f"  {ch}: ${t['spend']:.0f} spent, no sign-ups yet")

    # Recent detail: only the last HISTORY_WINDOW days
    recent = channel_history[-HISTORY_WINDOW:]
    lines.append(f"\nLast {len(recent)} day(s) in detail:")
    for record in recent:
        day = record["day"]
        day_su = record["day_sign_ups"]
        lines.append(f"\n  Day {day} — {day_su} sign-ups:")
        for ch in CHANNELS:
            data = record["channel_results"].get(ch, {})
            spend = data.get("spend", 0)
            su = data.get("sign_ups", 0)
            cpa = data.get("effective_cpa", 0)
            if spend > 0:
                lines.append(f"    {ch}: ${spend:.0f} → {su} sign-ups (CPA ${cpa:.1f})")

    return "\n".join(lines)


def decide_allocation(
    client: anthropic.Anthropic,
    remaining_budget: float,
    daily_budget: float,
    channel_history: list[dict],
) -> dict[str, float]:
    """Ask Claude for the next day's allocation. Returns a channel → spend dict."""
    history_text = _build_history_text(channel_history, daily_budget)

    user_message = (
        f"Remaining budget: ${remaining_budget:.2f}\n"
        f"Today's budget to allocate: ${daily_budget:.2f}\n\n"
        f"{history_text}\n\n"
        f"Allocate ${daily_budget:.2f} across the 4 channels. "
        f"Allocations must sum to exactly ${daily_budget:.2f}. JSON only."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Extract just the JSON object in case there's surrounding text
    import re
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    # Replace JS-only non-finite values that are invalid JSON
    raw = re.sub(r":\s*NaN\b", ": 0", raw)
    raw = re.sub(r":\s*Infinity\b", ": 0", raw)
    raw = re.sub(r":\s*-Infinity\b", ": 0", raw)

    allocation = json.loads(raw)

    # Ensure all channels present
    for ch in CHANNELS:
        allocation.setdefault(ch, 0.0)

    # Enforce minimum $5 per channel (only when budget allows)
    min_spend = 5.0
    if daily_budget >= min_spend * len(CHANNELS):
        for ch in CHANNELS:
            if allocation[ch] < min_spend:
                deficit = min_spend - allocation[ch]
                allocation[ch] = min_spend
                # Deduct from the highest-spend channel
                biggest = max((c for c in CHANNELS if c != ch), key=lambda c: allocation[c])
                allocation[biggest] = max(0, allocation[biggest] - deficit)

    # Normalize to exact daily budget (fix floating-point drift)
    total = sum(allocation.values())
    if total > 0 and abs(total - daily_budget) > 0.01:
        factor = daily_budget / total
        allocation = {ch: round(v * factor, 2) for ch, v in allocation.items()}
        # Fix rounding residual on the first channel
        diff = round(daily_budget - sum(allocation.values()), 2)
        allocation[CHANNELS[0]] = round(allocation[CHANNELS[0]] + diff, 2)

    return allocation
