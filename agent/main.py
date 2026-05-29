#!/usr/bin/env python3
"""
Ad Budget Optimizer — closed-loop agent that allocates a $1,000 budget
across 4 advertising channels to maximize sign-ups.

Each day: Claude decides the allocation → AdEnvironment returns results →
agent analyses per-channel CPA → Claude reallocates for the next day.
Stops when budget is exhausted.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from ad_environment import AdEnvironment
from budget_agent import CHANNELS, decide_allocation

DAILY_BUDGET = 100.0


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_header(total_budget: float) -> None:
    print(f"\n{'='*62}")
    print(f"  AD BUDGET OPTIMIZER")
    print(f"  Total budget: ${total_budget:.0f}  |  Daily budget: ${DAILY_BUDGET:.0f}")
    print(f"  Channels: {', '.join(CHANNELS)}")
    print(f"{'='*62}")


def _print_day(day: int, result: dict, top_channel: str, next_allocation: dict | None) -> None:
    cr = result["channel_results"]
    print(
        f"\nDay {day:2d} | "
        f"Spent ${result['spend']:.0f} → {result['sign_ups']} sign-ups | "
        f"Budget left: ${result['remaining_budget']:.0f}"
    )
    print(f"  {'Channel':<10} {'Spend':>8} {'Sign-ups':>9} {'CPA':>8}")
    print(f"  {'-'*38}")
    for ch in CHANNELS:
        data = cr.get(ch, {"spend": 0, "sign_ups": 0, "effective_cpa": 0})
        cpa_str = f"${data['effective_cpa']:.1f}" if data["sign_ups"] > 0 else "  n/a"
        marker = " ◄ best" if ch == top_channel else ""
        print(f"  {ch:<10} ${data['spend']:>6.0f}   {data['sign_ups']:>6}   {cpa_str:>6}{marker}")

    if next_allocation:
        shift_summary = "  → Next day: " + " | ".join(
            f"{ch} ${v:.0f}" for ch, v in next_allocation.items()
        )
        # Describe the biggest shift
        prev_spend = {ch: cr.get(ch, {}).get("spend", 0) for ch in CHANNELS}
        biggest_gain = max(CHANNELS, key=lambda c: next_allocation[c] - prev_spend.get(c, 0))
        biggest_loss = min(CHANNELS, key=lambda c: next_allocation[c] - prev_spend.get(c, 0))
        gain = next_allocation[biggest_gain] - prev_spend.get(biggest_gain, 0)
        loss = next_allocation[biggest_loss] - prev_spend.get(biggest_loss, 0)
        if abs(gain) > 1 and abs(loss) > 1:
            print(
                f"  → Shifting ${gain:.0f} toward {biggest_gain}, "
                f"${abs(loss):.0f} away from {biggest_loss}."
            )
        print(shift_summary)


def _print_summary(day: int, total_sign_ups: int, channel_history: list[dict]) -> None:
    print(f"\n{'='*62}")
    print(f"  SIMULATION COMPLETE")
    print(f"  Days run:        {day}")
    print(f"  Total sign-ups:  {total_sign_ups}")
    print(f"  Sign-ups/day:    {total_sign_ups / day:.1f}")

    totals = {ch: {"spend": 0.0, "sign_ups": 0} for ch in CHANNELS}
    for record in channel_history:
        for ch in CHANNELS:
            data = record["channel_results"].get(ch, {})
            totals[ch]["spend"] += data.get("spend", 0)
            totals[ch]["sign_ups"] += data.get("sign_ups", 0)

    print(f"\n  {'Channel':<10} {'Total Spend':>12} {'Sign-ups':>10} {'Avg CPA':>10}")
    print(f"  {'-'*46}")
    for ch in CHANNELS:
        t = totals[ch]
        avg_cpa = t["spend"] / t["sign_ups"] if t["sign_ups"] > 0 else float("inf")
        cpa_str = f"${avg_cpa:.1f}" if avg_cpa != float("inf") else "  n/a"
        print(f"  {ch:<10} ${t['spend']:>10.0f}   {t['sign_ups']:>7}   {cpa_str:>8}")
    print(f"{'='*62}")


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_simulation(output_dir: Path) -> int:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    env = AdEnvironment()
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_header(env.total_budget)

    channel_history: list[dict] = []
    day = 0
    total_sign_ups = 0

    # Day 1: equal split
    allocation = {ch: round(DAILY_BUDGET / len(CHANNELS), 2) for ch in CHANNELS}

    while env.total_budget >= 1.0:
        day += 1

        # Scale down on the final day when budget < DAILY_BUDGET
        actual_budget = min(DAILY_BUDGET, env.total_budget)
        if actual_budget < DAILY_BUDGET - 0.01:
            factor = actual_budget / DAILY_BUDGET
            allocation = {ch: round(v * factor, 2) for ch, v in allocation.items()}
            diff = round(actual_budget - sum(allocation.values()), 2)
            allocation[CHANNELS[0]] = round(allocation[CHANNELS[0]] + diff, 2)

        result = env.spend_daily_budget(allocation)
        total_sign_ups += result["sign_ups"]

        # Identify today's best channel
        cr = result["channel_results"]
        top_channel = max(
            (ch for ch in CHANNELS if cr.get(ch, {}).get("sign_ups", 0) > 0),
            key=lambda ch: cr[ch]["sign_ups"],
            default=CHANNELS[0],
        )

        record = {
            "day": day,
            "allocation": allocation.copy(),
            "result": result,
            "channel_results": cr,
            "day_sign_ups": result["sign_ups"],
            "total_sign_ups": total_sign_ups,
        }
        channel_history.append(record)

        # Decide next allocation before printing so we can show the shift
        next_allocation = None
        if env.total_budget >= 1.0:
            next_budget = min(DAILY_BUDGET, env.total_budget)
            next_allocation = decide_allocation(client, env.total_budget, next_budget, channel_history)

        _print_day(day, result, top_channel, next_allocation)

        if next_allocation:
            allocation = next_allocation

    _print_summary(day, total_sign_ups, channel_history)

    # Persist run data
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_path = output_dir / f"run_{timestamp}.json"
    run_path.write_text(json.dumps({
        "timestamp": timestamp,
        "total_sign_ups": total_sign_ups,
        "days_run": day,
        "history": channel_history,
    }, indent=2))

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"total_sign_ups={total_sign_ups}\n")
            f.write(f"days_run={day}\n")

    return total_sign_ups


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = Path("outputs")
    try:
        score = run_simulation(output_dir)
        print(f"\nFinal score: {score} sign-ups")
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
