#!/usr/bin/env python3
"""
Ad Budget Optimizer — self-invoking mode.

Each GitHub Actions run executes ONE day, saves state as an artifact,
then re-triggers itself for the next day. Stops when budget is exhausted.

Can also be run locally as a full simulation loop (omit --day flag).
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from ad_environment import AdEnvironment
from budget_agent import CHANNELS, decide_allocation

DAILY_BUDGET = 100.0
MAX_DAYS = 20  # hard cap against infinite self-invocation


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_header(remaining_budget: float, day: int) -> None:
    print(f"\n{'='*62}")
    print(f"  AD BUDGET OPTIMIZER  —  Day {day}")
    print(f"  Budget remaining: ${remaining_budget:.2f}  |  Daily budget: ${DAILY_BUDGET:.0f}")
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
        print("  → Next day: " + " | ".join(f"{ch} ${v:.0f}" for ch, v in next_allocation.items()))


def _print_final_summary(history: list[dict]) -> None:
    total_sign_ups = history[-1]["total_sign_ups"]
    days = len(history)
    print(f"\n{'='*62}")
    print(f"  BUDGET EXHAUSTED — FINAL SUMMARY")
    print(f"  Days run:        {days}")
    print(f"  Total sign-ups:  {total_sign_ups}")
    print(f"  Sign-ups/day:    {total_sign_ups / days:.1f}")

    totals = {ch: {"spend": 0.0, "sign_ups": 0} for ch in CHANNELS}
    for record in history:
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
# Single-day mode (GitHub Actions)
# ---------------------------------------------------------------------------

def day_mode(day: int, state_file: Path, output_dir: Path) -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load persisted state (missing on day 1)
    if state_file.exists():
        state = json.loads(state_file.read_text())
    else:
        state = {
            "remaining_budget": 1000.0,
            "next_allocation": {ch: round(DAILY_BUDGET / len(CHANNELS), 2) for ch in CHANNELS},
            "total_sign_ups": 0,
            "history": [],
        }

    remaining = state["remaining_budget"]
    allocation = state["next_allocation"]
    total_sign_ups = state["total_sign_ups"]
    history = state["history"]

    _print_header(remaining, day)

    # Scale down allocation on the final day
    actual_budget = min(DAILY_BUDGET, remaining)
    if actual_budget < DAILY_BUDGET - 0.01:
        factor = actual_budget / DAILY_BUDGET
        allocation = {ch: round(v * factor, 2) for ch, v in allocation.items()}
        diff = round(actual_budget - sum(allocation.values()), 2)
        allocation[CHANNELS[0]] = round(allocation[CHANNELS[0]] + diff, 2)

    # Run today
    env = AdEnvironment()
    env.total_budget = remaining
    result = env.spend_daily_budget(allocation)
    total_sign_ups += result["sign_ups"]

    cr = result["channel_results"]
    top_channel = max(
        (ch for ch in CHANNELS if cr.get(ch, {}).get("sign_ups", 0) > 0),
        key=lambda ch: cr[ch]["sign_ups"],
        default=CHANNELS[0],
    )

    record = {
        "day": day,
        "allocation": allocation,
        "channel_results": cr,
        "day_sign_ups": result["sign_ups"],
        "total_sign_ups": total_sign_ups,
        "remaining_budget_after": result["remaining_budget"],
    }
    history.append(record)

    done = result["remaining_budget"] < 1.0

    # Ask Claude for tomorrow's allocation (if there's budget left)
    next_allocation = None
    if not done:
        next_budget = min(DAILY_BUDGET, result["remaining_budget"])
        next_allocation = decide_allocation(
            client, result["remaining_budget"], next_budget, history
        )

    _print_day(day, result, top_channel, next_allocation)
    if done:
        _print_final_summary(history)

    # Save updated state atomically
    new_state = {
        "remaining_budget": result["remaining_budget"],
        "next_allocation": next_allocation or allocation,
        "total_sign_ups": total_sign_ups,
        "history": history,
    }
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(new_state, indent=2))
    os.replace(tmp, state_file)

    # Save per-day report
    day_report = output_dir / f"day_{day:02d}.json"
    day_report.write_text(json.dumps(record, indent=2))

    # Write GitHub outputs
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"done={'true' if done else 'false'}\n")
            f.write(f"day_sign_ups={result['sign_ups']}\n")
            f.write(f"total_sign_ups={total_sign_ups}\n")
            f.write(f"remaining_budget={result['remaining_budget']:.2f}\n")
            f.write(f"top_channel={top_channel}\n")


# ---------------------------------------------------------------------------
# Local full-loop mode
# ---------------------------------------------------------------------------

def local_mode(output_dir: Path) -> int:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    env = AdEnvironment()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  AD BUDGET OPTIMIZER  (local full-loop)")
    print(f"  Total budget: ${env.total_budget:.0f}  |  Daily budget: ${DAILY_BUDGET:.0f}")
    print(f"{'='*62}")

    history: list[dict] = []
    total_sign_ups = 0
    day = 0
    allocation = {ch: round(DAILY_BUDGET / len(CHANNELS), 2) for ch in CHANNELS}

    while env.total_budget >= 1.0:
        day += 1
        actual_budget = min(DAILY_BUDGET, env.total_budget)
        if actual_budget < DAILY_BUDGET - 0.01:
            factor = actual_budget / DAILY_BUDGET
            allocation = {ch: round(v * factor, 2) for ch, v in allocation.items()}
            diff = round(actual_budget - sum(allocation.values()), 2)
            allocation[CHANNELS[0]] = round(allocation[CHANNELS[0]] + diff, 2)

        result = env.spend_daily_budget(allocation)
        total_sign_ups += result["sign_ups"]
        cr = result["channel_results"]
        top_channel = max(
            (ch for ch in CHANNELS if cr.get(ch, {}).get("sign_ups", 0) > 0),
            key=lambda ch: cr[ch]["sign_ups"],
            default=CHANNELS[0],
        )

        record = {
            "day": day, "allocation": allocation.copy(),
            "channel_results": cr, "day_sign_ups": result["sign_ups"],
            "total_sign_ups": total_sign_ups,
            "remaining_budget_after": result["remaining_budget"],
        }
        history.append(record)

        next_allocation = None
        if env.total_budget >= 1.0:
            next_allocation = decide_allocation(
                client, env.total_budget, min(DAILY_BUDGET, env.total_budget), history
            )

        _print_day(day, result, top_channel, next_allocation)
        if next_allocation:
            allocation = next_allocation

    _print_final_summary(history)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    (output_dir / f"run_{timestamp}.json").write_text(
        json.dumps({"total_sign_ups": total_sign_ups, "days": day, "history": history}, indent=2)
    )
    return total_sign_ups


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="outputs")
    parser.add_argument("--day", type=int, default=None,
                        help="Single-day mode (1-indexed). Omit for full local loop.")
    parser.add_argument("--state-file", default="outputs/state.json")
    args = parser.parse_args()

    try:
        if args.day is not None:
            day_mode(args.day, Path(args.state_file), Path(args.output_dir))
        else:
            score = local_mode(Path(args.output_dir))
            print(f"\nFinal score: {score} sign-ups")
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
