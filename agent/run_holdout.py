#!/usr/bin/env python3
"""
Standalone hold-out evaluator — run after a partial or complete training run.

Reads committed_history from a saved state.json, rebuilds the same train/holdout
split used during training, and evaluates the learned strategy on unseen games.

Usage:
    python run_holdout.py PATH_TO_STATE_JSON [--data-dir data] [--players ""]
"""

import argparse
import json
from pathlib import Path

from llm_client import make_client
from main import _holdout_eval, HOLDOUT_FRAC
from wnba_data import build_prop_questions, fetch_gamelogs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_file", help="Path to state.json from a training run")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/holdout")
    parser.add_argument("--players", default="",
                        help="Same --players value used during training (empty = default roster)")
    args = parser.parse_args()

    state_file = Path(args.state_file)
    if not state_file.exists():
        raise SystemExit(f"State file not found: {state_file}")

    state = json.loads(state_file.read_text())
    committed_history = state["committed_history"]
    print(f"  Loaded {len(committed_history)} committed picks from {state_file}")
    correct = sum(1 for p in committed_history if p["correct"])
    print(f"  Training accuracy: {correct}/{len(committed_history)} "
          f"({100*correct//len(committed_history) if committed_history else 0}%)")

    players = [p.strip() for p in args.players.split(",") if p.strip()] or None

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Rebuilding prop questions from {data_dir}...")
    df = fetch_gamelogs(data_dir)
    full_questions = build_prop_questions(df, players=players)

    split = int(len(full_questions) * (1 - HOLDOUT_FRAC))
    holdout_questions = full_questions[split:]
    print(f"  Hold-out set: {len(holdout_questions)} props "
          f"(from {holdout_questions[0]['game_date'] if holdout_questions else 'n/a'} onward)")

    client = make_client()
    print(f"  LLM: {client.provider_name} / {client.model}\n")

    result = _holdout_eval(client, holdout_questions, committed_history, output_dir)
    (output_dir / "holdout_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Full results saved to {output_dir}/holdout_result.json")


if __name__ == "__main__":
    main()
