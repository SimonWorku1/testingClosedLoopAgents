#!/usr/bin/env python3
"""
Autonomous closed-loop research agent — single-iteration mode.

Each GitHub Action run executes ONE iteration, saves results as an artifact,
then re-triggers itself for the next iteration. This gives a separate log and
saved artifact per iteration rather than one monolithic run.

Can also be run locally in full-loop mode (omit --iteration flag).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

from evaluator import evaluate_report
from web_researcher import run_research_agent

TARGET_SCORE = 8


# ---------------------------------------------------------------------------
# Single iteration
# ---------------------------------------------------------------------------

def run_iteration(
    client: anthropic.Anthropic,
    topic: str,
    iteration: int,
    previous_iterations: list[dict],
) -> dict:
    """Run one research agent for one iteration. Returns the result dict."""
    t0 = time.time()
    print(f"\n--- Iteration {iteration + 1} ---")
    print(f"  Researching: {topic}")

    report = run_research_agent(client, topic, 0, iteration, previous_iterations)
    elapsed_research = time.time() - t0

    print(f"  Research done ({elapsed_research:.1f}s). Evaluating...")
    score, feedback = evaluate_report(client, report, topic)
    elapsed_total = time.time() - t0

    print(f"  Score: {score}/10  |  {elapsed_total:.1f}s  |  {feedback[:100]}")
    return {
        "iteration": iteration + 1,
        "report": report,
        "score": score,
        "feedback": feedback,
        "elapsed_seconds": round(elapsed_total, 1),
    }


# ---------------------------------------------------------------------------
# Action mode: one iteration, read/write shared history file
# ---------------------------------------------------------------------------

def action_mode(topic: str, iteration: int, history_file: Path, output_dir: Path) -> None:
    """Run a single iteration. Used by the self-invoking GitHub Action."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load history from previous iterations (passed via artifact)
    previous_iterations: list[dict] = []
    if history_file.exists():
        data = json.loads(history_file.read_text())
        previous_iterations = data.get("iterations", [])

    result = run_iteration(client, topic, iteration, previous_iterations)

    # Append to history
    history = {
        "topic": topic,
        "target_score": TARGET_SCORE,
        "iterations": previous_iterations + [result],
    }
    history_file.write_text(json.dumps(history, indent=2))

    # Save this iteration's report
    report_path = output_dir / f"iteration_{iteration + 1:02d}_score{result['score']}.md"
    report_path.write_text(
        f"# Iteration {iteration + 1} Report: {topic}\n\n"
        f"> **Score:** {result['score']}/10  \n"
        f"> **Feedback:** {result['feedback']}\n\n---\n\n"
        + result["report"]
    )

    done = result["score"] >= TARGET_SCORE
    print(f"\n{'='*50}")
    print(f"Iteration {iteration + 1} complete — score {result['score']}/10")
    print(f"Status: {'DONE (target reached)' if done else 'continue'}")
    print(f"{'='*50}")

    # Write step outputs for the workflow to read
    outputs_path = output_dir / "iteration_output.json"
    outputs_path.write_text(json.dumps({
        "score": result["score"],
        "done": done,
        "next_iteration": iteration + 1,
        "report_path": str(report_path),
    }, indent=2))

    # Write to $GITHUB_OUTPUT if running in Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"score={result['score']}\n")
            f.write(f"done={'true' if done else 'false'}\n")
            f.write(f"next_iteration={iteration + 1}\n")


# ---------------------------------------------------------------------------
# Local full-loop mode (for running outside GitHub Actions)
# ---------------------------------------------------------------------------

def local_mode(topic: str, output_dir: Path, max_iterations: int = 8) -> None:
    """Run the full loop locally."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Topic: {topic}")
    print(f"Max iterations: {max_iterations}  |  Target score: {TARGET_SCORE}/10")
    print(f"{'='*60}")

    previous_iterations: list[dict] = []
    best_result: dict | None = None
    best_score = -1

    for i in range(max_iterations):
        result = run_iteration(client, topic, i, previous_iterations)
        previous_iterations.append(result)

        if result["score"] > best_score:
            best_score = result["score"]
            best_result = result

        if result["score"] >= TARGET_SCORE:
            print(f"\n✓ Target reached at iteration {i + 1}!")
            break
    else:
        print(f"\n✗ Max iterations reached. Best score: {best_score}/10")

    assert best_result is not None
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    history_path = output_dir / f"run_history_{timestamp}.json"
    history_path.write_text(json.dumps({
        "topic": topic,
        "timestamp": timestamp,
        "best_score": best_score,
        "iterations": previous_iterations,
    }, indent=2))

    report_path = output_dir / f"best_report_{timestamp}.md"
    report_path.write_text(
        f"# Research Report: {topic}\n\n"
        f"> **Score:** {best_result['score']}/10  \n"
        f"> **Feedback:** {best_result['feedback']}\n\n---\n\n"
        + best_result["report"]
    )

    summary = {
        "best_report_path": str(report_path),
        "history_path": str(history_path),
        "best_score": best_score,
        "iterations_run": len(previous_iterations),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("output_dir", nargs="?", default="outputs")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Run a single iteration (0-indexed). Omit for full local loop.")
    parser.add_argument("--history-file", default="outputs/history.json",
                        help="Path to shared history JSON (action mode)")
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    try:
        if args.iteration is not None:
            action_mode(args.topic, args.iteration, Path(args.history_file), output_dir)
        else:
            local_mode(args.topic, output_dir, args.max_iterations)
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
