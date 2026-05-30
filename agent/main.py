#!/usr/bin/env python3
"""
Ad Budget Optimizer — single job with durable checkpointing.

Runs the full multi-day simulation in one process. After each day it
writes state.json and (when --git-checkpoint is set) commits + pushes it
to a dedicated branch, so if the VM dies the next run resumes from the
last completed day instead of losing all progress.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ad_environment import AdEnvironment
from budget_agent import CHANNELS, decide_allocation
from llm_client import LLMClient, make_client

DAILY_BUDGET = 100.0


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
# Full-loop mode  (single job, resumable + durable checkpoint)
# ---------------------------------------------------------------------------

MAX_DAY_RETRIES = 3  # retries per day before giving up and saving progress


def _run_one_day(
    client: LLMClient,
    env_remaining: float,
    allocation: dict,
    history: list[dict],
    day: int,
    total_sign_ups: int,
) -> tuple[dict, dict | None]:
    """
    Run a single day and return (record, next_allocation).
    Raises on unrecoverable error so the caller can retry.
    """
    actual_budget = min(DAILY_BUDGET, env_remaining)
    if actual_budget < DAILY_BUDGET - 0.01:
        factor = actual_budget / DAILY_BUDGET
        allocation = {ch: round(v * factor, 2) for ch, v in allocation.items()}
        diff = round(actual_budget - sum(allocation.values()), 2)
        allocation[CHANNELS[0]] = round(allocation[CHANNELS[0]] + diff, 2)

    env = AdEnvironment()
    env.total_budget = env_remaining
    result = env.spend_daily_budget(allocation)
    total_sign_ups_after = total_sign_ups + result["sign_ups"]

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
        "total_sign_ups": total_sign_ups_after,
        "remaining_budget_after": result["remaining_budget"],
    }

    next_allocation = None
    if result["remaining_budget"] >= 1.0:
        next_budget = min(DAILY_BUDGET, result["remaining_budget"])
        next_allocation = decide_allocation(client, result["remaining_budget"], next_budget, history + [record])

    _print_day(day, result, top_channel, next_allocation)
    return record, next_allocation


def _save_state(state_file: Path, remaining: float, allocation: dict,
                total_sign_ups: int, history: list[dict]) -> None:
    state = {
        "remaining_budget": remaining,
        "next_allocation": allocation,
        "total_sign_ups": total_sign_ups,
        "history": history,
    }
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_file)


def _git_checkpoint(checkpoint_dir: Path, day: int) -> None:
    """
    Commit and push the checkpoint dir to its branch so it survives VM death.
    Failures are logged but never abort the run — the local state.json is
    still intact, and the next successful day will push again.
    """
    d = str(checkpoint_dir)
    try:
        subprocess.run(["git", "-C", d, "add", "-A"], check=True)
        # Skip the commit/push if nothing actually changed
        staged = subprocess.run(["git", "-C", d, "diff", "--cached", "--quiet"])
        if staged.returncode == 0:
            return
        subprocess.run(
            ["git", "-C", d, "commit", "-m", f"checkpoint: day {day}"],
            check=True, stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "-C", d, "push", "origin", "HEAD:budget-state"],
            check=True, stdout=subprocess.DEVNULL,
        )
        print(f"  [checkpoint: day {day} pushed to budget-state]")
    except subprocess.CalledProcessError as exc:
        print(f"  [checkpoint warning: git push failed ({exc}); progress kept locally]",
              file=sys.stderr)


def local_mode(output_dir: Path, state_file: Path, checkpoint_dir: Path | None = None) -> int:
    client = make_client()
    print(f"  [LLM provider: {client.provider_name}  |  model: {client.model}]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from saved state if it exists, otherwise start fresh
    if state_file.exists():
        state = json.loads(state_file.read_text())
        remaining = state["remaining_budget"]
        allocation = state["next_allocation"]
        total_sign_ups = state["total_sign_ups"]
        history = state["history"]
        start_day = len(history) + 1
        print(f"\n{'='*62}")
        print(f"  AD BUDGET OPTIMIZER  — resuming from Day {start_day}")
        print(f"  Budget remaining: ${remaining:.2f}  |  Days completed: {len(history)}")
        print(f"{'='*62}")
    else:
        remaining = 1000.0
        allocation = {ch: round(DAILY_BUDGET / len(CHANNELS), 2) for ch in CHANNELS}
        total_sign_ups = 0
        history = []
        start_day = 1
        print(f"\n{'='*62}")
        print(f"  AD BUDGET OPTIMIZER")
        print(f"  Total budget: ${remaining:.0f}  |  Daily budget: ${DAILY_BUDGET:.0f}")
        print(f"  Channels: {', '.join(CHANNELS)}")
        print(f"{'='*62}")

    day = start_day - 1

    while remaining >= 1.0:
        day += 1

        # Retry loop: handles transient API/JSON errors without losing progress
        last_exc = None
        for attempt in range(1, MAX_DAY_RETRIES + 1):
            try:
                record, next_allocation = _run_one_day(
                    client, remaining, allocation, history, day, total_sign_ups
                )
                break
            except Exception as exc:
                last_exc = exc
                print(f"  [Day {day} attempt {attempt}/{MAX_DAY_RETRIES} failed: {exc}]", file=sys.stderr)
                if attempt < MAX_DAY_RETRIES:
                    import time
                    time.sleep(2 ** attempt)
        else:
            # All retries exhausted — save what we have and abort
            print(f"\nDay {day} failed after {MAX_DAY_RETRIES} attempts. Progress saved to {state_file}.", file=sys.stderr)
            _save_state(state_file, remaining, allocation, total_sign_ups, history)
            raise last_exc

        # Day succeeded — commit progress immediately
        total_sign_ups = record["total_sign_ups"]
        remaining = record["remaining_budget_after"]
        history.append(record)
        if next_allocation:
            allocation = next_allocation

        # Append to the running daily log
        with open(output_dir / "daily_log.txt", "a") as f:
            top = max(record["channel_results"], key=lambda ch: record["channel_results"][ch]["sign_ups"])
            f.write(
                f"Day {day:2d} | Spent ${DAILY_BUDGET:.0f} | "
                f"Sign-ups: {record['day_sign_ups']:3d} | "
                f"Total: {total_sign_ups:4d} | "
                f"Budget left: ${remaining:.0f} | "
                f"Top: {top}\n"
            )

        with open(output_dir / "daily_log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        # Persist state after every successful day so a crash can resume
        _save_state(state_file, remaining, allocation, total_sign_ups, history)

        # Durably checkpoint off-VM so a hard crash doesn't lose progress
        if checkpoint_dir is not None:
            _git_checkpoint(checkpoint_dir, day)

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
    parser.add_argument("--state-file", default="outputs/state.json")
    parser.add_argument("--git-checkpoint", default=None,
                        help="Directory (a git worktree on the state branch) to "
                             "commit+push after each day for durable resume.")
    args = parser.parse_args()

    state_file = Path(args.state_file)
    checkpoint_dir = Path(args.git_checkpoint) if args.git_checkpoint else None

    try:
        score = local_mode(Path(args.output_dir), state_file, checkpoint_dir)
        print(f"\nFinal score: {score} sign-ups")
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
