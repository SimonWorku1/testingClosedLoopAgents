#!/usr/bin/env python3
"""
WNBA Prop Picks Optimizer — closed-loop agent.

Iteratively makes over/under predictions on WNBA player props (Points, Rebounds,
Assists, Pts+Reb+Ast) using a 10-game rolling average as the line. Succeeds
when any 20-pick window has ≤1 mistake (≥19/20 correct). Fails if it cannot
reach that threshold within MAX_TOTAL_PICKS total picks.

Data: real WNBA game logs from stats.nba.com (cached on first run).
Lines: 10-game rolling average ± 0.5 (deterministic, same formula as PrizePicks).
Single-job design with durable checkpointing — crash and resume from last batch.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from llm_client import LLMClient, make_client
from picks_agent import make_picks
from wnba_data import build_prop_questions, fetch_gamelogs

PICKS_PER_BATCH = 5       # props sent to the agent per iteration
WINDOW_SIZE = 20          # sliding window for success check
MAX_MISTAKES = 1          # mistakes allowed in the window
MAX_TOTAL_PICKS = 400     # fail if threshold not reached by here
MAX_BATCH_RETRIES = 3


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_header(total_questions: int, resuming: bool, start_pick: int) -> None:
    print(f"\n{'='*70}")
    print("  WNBA PROP PICKS OPTIMIZER")
    print(f"  Target: {WINDOW_SIZE - MAX_MISTAKES}/{WINDOW_SIZE} correct in any "
          f"{WINDOW_SIZE}-pick window")
    print(f"  Max picks: {MAX_TOTAL_PICKS}  |  Dataset: {total_questions} questions")
    if resuming:
        print(f"  >> Resuming from pick #{start_pick}")
    print(f"{'='*70}")


def _pick_result_char(correct: bool) -> str:
    return "✓" if correct else "✗"


def _print_batch(batch_num: int, picks: list[dict], questions: list[dict],
                 pick_history: list[dict], window: deque) -> None:
    print(f"\n--- Batch {batch_num} (picks {picks[0]['pick_num']}-"
          f"{picks[-1]['pick_num']}) ---")

    for p in picks:
        q = next(q for q in questions if q["question_id"] == p["question_id"])
        marker = _pick_result_char(p["correct"])
        conf_bar = "●" * p["confidence"] + "○" * (5 - p["confidence"])
        print(
            f"  {marker} Pick {p['pick_num']:3d} | "
            f"{q['player_name']:<22} {q['stat_name']:<14} "
            f"line {q['line']:5.1f} | "
            f"{p['pick']:<5} → actual {q['actual_value']:5.1f} | "
            f"conf {conf_bar}"
        )
        if not p["correct"]:
            print(f"       diagnosis: {_diagnose(q, p)}")
        else:
            if p.get("reasoning"):
                print(f"       reasoning: {p['reasoning']}")

    w_correct = sum(window)
    w_total = len(window)
    total_correct = sum(1 for p in pick_history if p["correct"])
    print(
        f"\n  Window [{w_total} picks]: {w_correct}/{w_total} correct  |  "
        f"Overall: {total_correct}/{len(pick_history)} "
        f"({100 * total_correct // len(pick_history) if pick_history else 0}%)"
    )


def _diagnose(question: dict, pick: dict) -> str:
    """Plain-English explanation of why this pick missed."""
    actual = question["actual_value"]
    line = question["line"]
    picked = pick["pick"]
    correct = "over" if actual > line else "under"
    last3 = question.get("last_3_avg")
    rolling = question.get("rolling_avg_10")

    msg = (f"Picked {picked.upper()} on {question['stat_name']} line {line} "
           f"but actual was {actual} → {correct.upper()} was right")

    if last3 is not None and rolling is not None:
        form_diff = last3 - rolling
        if abs(form_diff) >= 2:
            direction = "above" if form_diff > 0 else "below"
            msg += (f"; last-3 avg ({last3}) was {abs(form_diff):.1f} "
                    f"{direction} rolling avg ({rolling}) — "
                    f"{'momentum missed' if picked != correct else 'regression occurred'}")
    return msg


def _print_summary(pick_history: list[dict], success: bool, best_window: int) -> None:
    total = len(pick_history)
    correct = sum(1 for p in pick_history if p["correct"])
    print(f"\n{'='*70}")
    if success:
        print(f"  SUCCESS — reached {WINDOW_SIZE - MAX_MISTAKES}/{WINDOW_SIZE} "
              f"in a {WINDOW_SIZE}-pick window")
    else:
        print(f"  FAILED — could not reach target within {MAX_TOTAL_PICKS} picks")
        print(f"  Best window: {best_window}/{WINDOW_SIZE}")
    print(f"  Total picks: {total}  |  Correct: {correct} ({100*correct//total if total else 0}%)")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Persistence + checkpointing
# ---------------------------------------------------------------------------

def _save_state(state_file: Path, next_idx: int, pick_history: list[dict],
                window: deque) -> None:
    state = {
        "next_question_idx": next_idx,
        "pick_history": pick_history,
        "window": list(window),
    }
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_file)


def _git_checkpoint(checkpoint_dir: Path, pick_num: int) -> None:
    d = str(checkpoint_dir)
    try:
        subprocess.run(["git", "-C", d, "add", "-A"], check=True)
        staged = subprocess.run(["git", "-C", d, "diff", "--cached", "--quiet"])
        if staged.returncode == 0:
            return
        subprocess.run(["git", "-C", d, "commit", "-m", f"checkpoint: pick {pick_num}"],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", d, "push", "origin", "HEAD:picks-state"],
                       check=True, stdout=subprocess.DEVNULL)
        print(f"  [checkpoint: pick {pick_num} pushed to picks-state]")
    except subprocess.CalledProcessError as exc:
        print(f"  [checkpoint warning: {exc}]", file=sys.stderr)


def _append_logs(output_dir: Path, batch_picks: list[dict],
                 questions_map: dict, batch_num: int) -> None:
    with open(output_dir / "picks_log.txt", "a") as f:
        f.write(f"\n=== Batch {batch_num} ===\n")
        for p in batch_picks:
            q = questions_map[p["question_id"]]
            mark = "OK" if p["correct"] else "XX"
            f.write(
                f"  [{mark}] Pick {p['pick_num']:3d} | "
                f"{q['player_name']:<22} {q['stat_name']:<14} "
                f"line {q['line']:5.1f} | {p['pick']:<5} → actual {q['actual_value']:.1f}"
            )
            if not p["correct"]:
                f.write(f" | {_diagnose(q, p)}")
            f.write("\n")

    with open(output_dir / "picks_log.jsonl", "a") as f:
        for p in batch_picks:
            q = questions_map[p["question_id"]]
            f.write(json.dumps({**p, "line": q["line"], "actual": q["actual_value"],
                                "stat_name": q["stat_name"], "player_name": q["player_name"],
                                "diagnosis": _diagnose(q, p) if not p["correct"] else None}) + "\n")


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run(output_dir: Path, data_dir: Path, state_file: Path,
        checkpoint_dir: Path | None = None) -> bool:
    client = make_client()
    print(f"  [LLM provider: {client.provider_name}  |  model: {client.model}]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (fetched once, then read from cache on all subsequent runs)
    df = fetch_gamelogs(data_dir)
    all_questions = build_prop_questions(df)

    if not all_questions:
        raise RuntimeError("No prop questions generated — check data fetch.")

    # Resume state or fresh start
    if state_file.exists():
        state = json.loads(state_file.read_text())
        next_idx = state["next_question_idx"]
        pick_history = state["pick_history"]
        window = deque(state["window"], maxlen=WINDOW_SIZE)
        start_pick = len(pick_history) + 1
        _print_header(len(all_questions), resuming=True, start_pick=start_pick)
    else:
        next_idx = 0
        pick_history = []
        window = deque(maxlen=WINDOW_SIZE)
        _print_header(len(all_questions), resuming=False, start_pick=1)

    success = False
    best_window = sum(window)
    batch_num = (len(pick_history) // PICKS_PER_BATCH) + 1
    questions_map = {q["question_id"]: q for q in all_questions}

    while len(pick_history) < MAX_TOTAL_PICKS and next_idx < len(all_questions):
        # Slice the next batch of questions
        batch_qs = all_questions[next_idx: next_idx + PICKS_PER_BATCH]
        if not batch_qs:
            break

        # Ask the agent for picks (with retry on transient errors)
        raw_picks = None
        last_exc = None
        for attempt in range(1, MAX_BATCH_RETRIES + 1):
            try:
                raw_picks = make_picks(client, batch_qs, pick_history)
                break
            except Exception as exc:
                last_exc = exc
                print(f"  [batch {batch_num} attempt {attempt}/{MAX_BATCH_RETRIES}: {exc}]",
                      file=sys.stderr)
                if attempt < MAX_BATCH_RETRIES:
                    time.sleep(2 ** attempt)
        if raw_picks is None:
            _save_state(state_file, next_idx, pick_history, window)
            raise last_exc

        # Evaluate picks against actual outcomes
        batch_results = []
        for rp in raw_picks:
            q = questions_map.get(rp["question_id"])
            if q is None:
                continue
            correct = rp["pick"] == q["correct_pick"]
            window.append(correct)
            best_window = max(best_window, sum(window))

            record = {
                "pick_num": len(pick_history) + 1,
                "question_id": rp["question_id"],
                "player_name": q["player_name"],
                "stat_name": q["stat_name"],
                "pick": rp["pick"],
                "confidence": rp["confidence"],
                "reasoning": rp["reasoning"],
                "correct": correct,
                "actual": q["actual_value"],
                "line": q["line"],
                "last_3_avg": q.get("last_3_avg"),
                "correct_pick": q["correct_pick"],
            }
            pick_history.append(record)
            batch_results.append(record)

        _print_batch(batch_num, batch_results, batch_qs, pick_history, window)
        _append_logs(output_dir, batch_results, questions_map, batch_num)

        next_idx += len(batch_qs)
        batch_num += 1

        # Check success condition: ≥19/20 in the sliding window
        if len(window) == WINDOW_SIZE and sum(window) >= WINDOW_SIZE - MAX_MISTAKES:
            success = True
            print(f"\n  *** TARGET REACHED: {sum(window)}/{WINDOW_SIZE} "
                  f"in the last {WINDOW_SIZE} picks ***")
            break

        # Persist after every batch
        _save_state(state_file, next_idx, pick_history, window)
        if checkpoint_dir is not None:
            _git_checkpoint(checkpoint_dir, len(pick_history))

    _print_summary(pick_history, success, best_window)

    # Save final run record
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    (output_dir / f"run_{ts}.json").write_text(json.dumps({
        "success": success,
        "total_picks": len(pick_history),
        "correct": sum(1 for p in pick_history if p["correct"]),
        "best_window": best_window,
        "pick_history": pick_history,
    }, indent=2))

    # Final save + checkpoint
    _save_state(state_file, next_idx, pick_history, window)
    if checkpoint_dir is not None:
        _git_checkpoint(checkpoint_dir, len(pick_history))

    return success


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="outputs")
    parser.add_argument("--data-dir", default="data",
                        help="Directory for cached WNBA game log data.")
    parser.add_argument("--state-file", default="outputs/state.json")
    parser.add_argument("--git-checkpoint", default=None,
                        help="Git worktree to commit+push after each batch.")
    args = parser.parse_args()

    try:
        success = run(
            output_dir=Path(args.output_dir),
            data_dir=Path(args.data_dir),
            state_file=Path(args.state_file),
            checkpoint_dir=Path(args.git_checkpoint) if args.git_checkpoint else None,
        )
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nOutcome: {'SUCCESS' if success else 'FAILED TO REACH TARGET'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
