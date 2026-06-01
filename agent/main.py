#!/usr/bin/env python3
"""
WNBA Prop Picks Optimizer — closed-loop agent (high-conviction mode).

Iteratively evaluates WNBA player props (Points, Rebounds, Assists, Pts+Reb+Ast)
where the line is a 10-game rolling average ±0.5 (PrizePicks-style). Because the
line IS the recent average, a blind guess is ~50/50 — so the agent must be
SELECTIVE: it may "pass" on coin-flip props and only commit high-confidence
picks. Only committed picks count.

Success: 16+ of any 20 COMMITTED picks correct (≤4 mistakes). Fails if it can't
reach that within MAX_QUESTIONS evaluated props.

Data: real WNBA game logs from stats.nba.com (cached on first run).
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
from picks_agent import COMMIT_CONFIDENCE, TARGET_CORRECT, WINDOW_SIZE, make_picks

PROPS_PER_BATCH = 6       # props shown to the agent per iteration
MAX_MISTAKES = WINDOW_SIZE - TARGET_CORRECT  # 4 mistakes allowed in 20 committed
MAX_QUESTIONS = 900       # fail if target not reached after this many props seen
MAX_BATCH_RETRIES = 3

# Temporal hold-out: the most-recent HOLDOUT_FRAC of props (by date) are NEVER
# shown during optimization. After the loop, the learned strategy is evaluated
# on them cold — this is the real test of generalization vs. overfitting.
HOLDOUT_FRAC = 0.20


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_header(total_questions: int, resuming: bool, start_prop: int) -> None:
    print(f"\n{'='*72}")
    print("  WNBA PROP PICKS OPTIMIZER  (high-conviction mode)")
    print(f"  Target: {TARGET_CORRECT}/{WINDOW_SIZE} correct in any window of "
          f"COMMITTED picks (≤{MAX_MISTAKES} misses)")
    print(f"  Commit threshold: confidence ≥ {COMMIT_CONFIDENCE}  |  passes don't count")
    print(f"  Budget: {MAX_QUESTIONS} props  |  Dataset: {total_questions} questions")
    if resuming:
        print(f"  >> Resuming after {start_prop - 1} props evaluated")
    print(f"{'='*72}")


def _diagnose(question: dict, pick: str) -> str:
    """Plain-English explanation of why a committed pick missed."""
    actual = question["actual_value"]
    line = question["line"]
    correct = "over" if actual > line else "under"
    last3 = question.get("last_3_avg")
    rolling = question.get("rolling_avg_10")

    msg = (f"committed {pick.upper()} on {question['stat_name']} line {line}, "
           f"actual {actual} → {correct.upper()} was right")
    if last3 is not None and rolling is not None:
        form_diff = last3 - rolling
        if abs(form_diff) >= 2:
            d = "above" if form_diff > 0 else "below"
            msg += (f"; last-3 ({last3}) was {abs(form_diff):.1f} {d} rolling avg "
                    f"— form signal misfired")
        else:
            msg += "; weak signal (last-3 ≈ line) — should have passed"
    return msg


def _print_batch(batch_num: int, evals: list[dict], questions_map: dict,
                 window: deque, committed_total: int, correct_total: int) -> None:
    committed = [e for e in evals if e["committed"]]
    passed = [e for e in evals if not e["committed"]]
    print(f"\n--- Batch {batch_num} | {len(committed)} committed, {len(passed)} passed ---")

    for e in evals:
        q = questions_map[e["question_id"]]
        if not e["committed"]:
            print(f"    · pass  | {q['player_name']:<22} {q['stat_name']:<12} "
                  f"line {q['line']:5.1f}  (conf {e['confidence']}) — {e['reasoning'][:50]}")
            continue
        mark = "✓" if e["correct"] else "✗"
        conf_bar = "●" * e["confidence"] + "○" * (5 - e["confidence"])
        print(f"  {mark} BET   | {q['player_name']:<22} {q['stat_name']:<12} "
              f"line {q['line']:5.1f} | {e['pick']:<5} → actual {q['actual_value']:5.1f} | {conf_bar}")
        if not e["correct"]:
            print(f"          diagnosis: {_diagnose(q, e['pick'])}")

    w_correct, w_total = sum(window), len(window)
    print(f"\n  Committed window [{w_total}/{WINDOW_SIZE}]: {w_correct} correct  |  "
          f"All committed: {correct_total}/{committed_total} "
          f"({100*correct_total//committed_total if committed_total else 0}%)")


def _print_summary(committed_total: int, correct_total: int, passes: int,
                   success: bool, best_window: int) -> None:
    print(f"\n{'='*72}")
    if success:
        print(f"  SUCCESS — hit {TARGET_CORRECT}/{WINDOW_SIZE} in a committed-pick window")
    else:
        print(f"  FAILED — best committed window was {best_window}/{WINDOW_SIZE} "
              f"(needed {TARGET_CORRECT})")
    rate = 100 * correct_total // committed_total if committed_total else 0
    print(f"  Committed: {committed_total} ({correct_total} correct, {rate}%)  |  "
          f"Passed: {passes}")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_state(state_file: Path, next_idx: int, committed_history: list[dict],
                window: deque, passes: int) -> None:
    state = {
        "next_question_idx": next_idx,
        "committed_history": committed_history,
        "window": list(window),
        "passes": passes,
    }
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_file)


def _git_checkpoint(checkpoint_dir: Path, marker: int) -> None:
    d = str(checkpoint_dir)
    try:
        subprocess.run(["git", "-C", d, "add", "-A"], check=True)
        if subprocess.run(["git", "-C", d, "diff", "--cached", "--quiet"]).returncode == 0:
            return
        subprocess.run(["git", "-C", d, "commit", "-m", f"checkpoint: {marker} props"],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", d, "push", "origin", "HEAD:picks-state"],
                       check=True, stdout=subprocess.DEVNULL)
        print(f"  [checkpoint: {marker} props pushed to picks-state]")
    except subprocess.CalledProcessError as exc:
        print(f"  [checkpoint warning: {exc}]", file=sys.stderr)


def _append_logs(output_dir: Path, evals: list[dict], questions_map: dict,
                 batch_num: int) -> None:
    with open(output_dir / "picks_log.txt", "a") as f:
        f.write(f"\n=== Batch {batch_num} ===\n")
        for e in evals:
            q = questions_map[e["question_id"]]
            if not e["committed"]:
                f.write(f"  [pass] {q['player_name']:<22} {q['stat_name']:<12} "
                        f"line {q['line']:.1f} (conf {e['confidence']})\n")
                continue
            mark = "OK" if e["correct"] else "XX"
            f.write(f"  [{mark}] {q['player_name']:<22} {q['stat_name']:<12} "
                    f"line {q['line']:.1f} | {e['pick']:<5} → actual {q['actual_value']:.1f}")
            if not e["correct"]:
                f.write(f" | {_diagnose(q, e['pick'])}")
            f.write("\n")

    with open(output_dir / "picks_log.jsonl", "a") as f:
        for e in evals:
            f.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------

def run(output_dir: Path, data_dir: Path, state_file: Path,
        checkpoint_dir: Path | None = None,
        players: list[str] | None = None) -> bool:
    from wnba_data import build_prop_questions, fetch_gamelogs

    client = make_client()
    print(f"  [LLM provider: {client.provider_name}  |  model: {client.model}]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Reliability scan: uses full 5-season history to find the best targets.
    # Runs before the picks loop so the report informs roster choice.
    from wnba_data import fetch_all_seasons
    from player_reliability import find_reliable_props, print_reliability_report
    from prizepicks_lines import fetch_lines
    try:
        full_df = fetch_all_seasons(data_dir)
        top_reliable = find_reliable_props(full_df, top_n=3)
        pp_lines = fetch_lines()
        print_reliability_report(top_reliable, pp_lines=pp_lines)
    except Exception as exc:
        print(f"  [reliability scan skipped: {exc}]")

    df = fetch_gamelogs(data_dir)
    full_questions = build_prop_questions(df, players=players)
    if not full_questions:
        raise RuntimeError("No prop questions generated — check data fetch.")

    # Temporal hold-out split: questions are already sorted by date, so the
    # most-recent HOLDOUT_FRAC become the test set the agent never sees during
    # optimization. This is the only honest check that the model generalizes
    # rather than overfitting the stream it trained on.
    split = int(len(full_questions) * (1 - HOLDOUT_FRAC))
    all_questions = full_questions[:split]        # train stream
    holdout_questions = full_questions[split:]    # untouched test set
    print(f"  [split] {len(all_questions)} train props  |  "
          f"{len(holdout_questions)} hold-out props "
          f"(from {holdout_questions[0]['game_date'] if holdout_questions else 'n/a'} onward)")
    questions_map = {q["question_id"]: q for q in all_questions}

    if state_file.exists():
        state = json.loads(state_file.read_text())
        next_idx = state["next_question_idx"]
        committed_history = state["committed_history"]
        window = deque(state["window"], maxlen=WINDOW_SIZE)
        passes = state.get("passes", 0)
        _print_header(len(all_questions), resuming=True, start_prop=next_idx + 1)
    else:
        next_idx, committed_history, passes = 0, [], 0
        window = deque(maxlen=WINDOW_SIZE)
        _print_header(len(all_questions), resuming=False, start_prop=1)

    success = False
    best_window = sum(window) if window else 0
    batch_num = (next_idx // PROPS_PER_BATCH) + 1

    while next_idx < MAX_QUESTIONS and next_idx < len(all_questions):
        batch_qs = all_questions[next_idx: next_idx + PROPS_PER_BATCH]
        if not batch_qs:
            break

        raw_picks, last_exc = None, None
        for attempt in range(1, MAX_BATCH_RETRIES + 1):
            try:
                raw_picks = make_picks(client, batch_qs, committed_history)
                break
            except Exception as exc:
                last_exc = exc
                print(f"  [batch {batch_num} attempt {attempt}/{MAX_BATCH_RETRIES}: {exc}]",
                      file=sys.stderr)
                if attempt < MAX_BATCH_RETRIES:
                    time.sleep(2 ** attempt)
        if raw_picks is None:
            _save_state(state_file, next_idx, committed_history, window, passes)
            raise last_exc

        evals = []
        for rp in raw_picks:
            q = questions_map.get(rp["question_id"])
            if q is None:
                continue
            committed = rp["pick"] in ("over", "under") and rp["confidence"] >= COMMIT_CONFIDENCE
            ev = {
                "question_id": rp["question_id"],
                "player_name": q["player_name"],
                "stat_name": q["stat_name"],
                "line": q["line"],
                "actual": q["actual_value"],
                "pick": rp["pick"],
                "confidence": rp["confidence"],
                "reasoning": rp["reasoning"],
                "committed": committed,
                "last_3_avg": q.get("last_3_avg"),
                "correct_pick": q["correct_pick"],
            }
            if committed:
                ev["correct"] = (rp["pick"] == q["correct_pick"])
                window.append(ev["correct"])
                best_window = max(best_window, sum(window))
                committed_history.append(ev)
            else:
                ev["correct"] = None
                passes += 1
            evals.append(ev)

        committed_total = len(committed_history)
        correct_total = sum(1 for p in committed_history if p["correct"])
        _print_batch(batch_num, evals, questions_map, window, committed_total, correct_total)
        _append_logs(output_dir, evals, questions_map, batch_num)

        next_idx += len(batch_qs)
        batch_num += 1

        if len(window) == WINDOW_SIZE and sum(window) >= TARGET_CORRECT:
            success = True
            print(f"\n  *** TARGET REACHED: {sum(window)}/{WINDOW_SIZE} in the last "
                  f"{WINDOW_SIZE} committed picks ***")
            break

        _save_state(state_file, next_idx, committed_history, window, passes)
        if checkpoint_dir is not None:
            _git_checkpoint(checkpoint_dir, next_idx)

    committed_total = len(committed_history)
    correct_total = sum(1 for p in committed_history if p["correct"])
    _print_summary(committed_total, correct_total, passes, success, best_window)

    # --- HOLD-OUT TEST: evaluate the learned strategy on unseen games -------
    holdout = _holdout_eval(client, holdout_questions, committed_history, output_dir)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    (output_dir / f"run_{ts}.json").write_text(json.dumps({
        "success": success,
        "train": {
            "committed_total": committed_total,
            "correct_total": correct_total,
            "passes": passes,
            "best_window": best_window,
        },
        "holdout": holdout,
        "committed_history": committed_history,
    }, indent=2))

    _save_state(state_file, next_idx, committed_history, window, passes)
    if checkpoint_dir is not None:
        _git_checkpoint(checkpoint_dir, next_idx)

    return success


def _holdout_eval(client: LLMClient, holdout_questions: list[dict],
                  learned_history: list[dict], output_dir: Path) -> dict:
    """
    Evaluate the agent on games it never saw during optimization. The agent
    uses its learned committed-pick history as context but we do NOT add
    hold-out outcomes back into that history — this is a frozen, cold test of
    generalization. Reports committed accuracy and the best 20-pick window.
    """
    if not holdout_questions:
        return {"committed": 0, "correct": 0, "accuracy": None, "best_window": 0}

    print(f"\n{'='*72}")
    print("  HOLD-OUT TEST  —  unseen games (the real generalization check)")
    print(f"{'='*72}")

    qmap = {q["question_id"]: q for q in holdout_questions}
    window = deque(maxlen=WINDOW_SIZE)
    committed = correct = passes = best_window = 0

    with open(output_dir / "holdout_log.txt", "w") as log:
        for start in range(0, len(holdout_questions), PROPS_PER_BATCH):
            batch = holdout_questions[start: start + PROPS_PER_BATCH]
            try:
                picks = make_picks(client, batch, learned_history)
            except Exception as exc:
                print(f"  [hold-out batch failed: {exc}]", file=sys.stderr)
                continue

            for rp in picks:
                q = qmap.get(rp["question_id"])
                if q is None:
                    continue
                is_committed = (rp["pick"] in ("over", "under")
                                and rp["confidence"] >= COMMIT_CONFIDENCE)
                if not is_committed:
                    passes += 1
                    continue
                hit = (rp["pick"] == q["correct_pick"])
                committed += 1
                correct += hit
                window.append(hit)
                best_window = max(best_window, sum(window))
                log.write(
                    f"[{'OK' if hit else 'XX'}] {q['player_name']:<20} "
                    f"{q['stat_name']:<12} line {q['line']:.1f} | "
                    f"{rp['pick']} → actual {q['actual_value']:.1f}\n"
                )

    accuracy = round(100 * correct / committed, 1) if committed else None
    hit_target = committed >= WINDOW_SIZE and best_window >= TARGET_CORRECT

    print(f"  Committed on unseen data: {committed}  ({passes} passed)")
    if accuracy is not None:
        print(f"  Hold-out accuracy: {correct}/{committed} = {accuracy}%")
        print(f"  Best {WINDOW_SIZE}-window on unseen data: {best_window}/{WINDOW_SIZE}"
              f"{'  ✓ generalizes' if hit_target else ''}")
        verdict = ("STRONG — edge holds on unseen games" if accuracy >= 56
                   else "WEAK — likely overfit; in-sample success was variance"
                   if accuracy < 52 else "MARGINAL — small/uncertain edge")
        print(f"  Verdict: {verdict}")
    else:
        print("  Agent committed to nothing on the hold-out set.")
    print(f"{'='*72}")

    return {
        "committed": committed, "correct": correct, "passes": passes,
        "accuracy": accuracy, "best_window": best_window,
        "generalizes": hit_target,
    }


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
    parser.add_argument("--players", default="",
                        help="Comma-separated player name substrings to focus on "
                             "(e.g. 'Wilson,Stewart,Collier'). Empty = default roster.")
    args = parser.parse_args()

    players = [p.strip() for p in args.players.split(",") if p.strip()] or None

    try:
        success = run(
            output_dir=Path(args.output_dir),
            data_dir=Path(args.data_dir),
            state_file=Path(args.state_file),
            checkpoint_dir=Path(args.git_checkpoint) if args.git_checkpoint else None,
            players=players,
        )
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nOutcome: {'SUCCESS' if success else 'FAILED TO REACH TARGET'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
