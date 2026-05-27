#!/usr/bin/env python3
"""
Autonomous closed-loop research agent — single-iteration mode.

Each GitHub Action run executes ONE iteration (with NUM_AGENTS agents in
parallel), saves results as an artifact, then re-triggers itself for the
next iteration.

Can also be run locally in full-loop mode (omit --iteration flag).
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import anthropic

from evaluator import evaluate_report
from web_researcher import run_research_agent

NUM_AGENTS = 3
MAX_ITERATIONS = 5
TARGET_SCORE = 8

# Seconds between agent starts — spreads token bursts across the rate-limit window.
# At 30s stagger + 12s inter-round sleep, peak rate stays ~20-25k tokens/min
# which is safely under the Tier 1 30k/min cap.
AGENT_STAGGER_SECONDS = 30

_shutdown = threading.Event()
_executor = ThreadPoolExecutor(max_workers=NUM_AGENTS * 2)


# ---------------------------------------------------------------------------
# Single agent worker (runs in a thread)
# ---------------------------------------------------------------------------

def _agent_worker(
    client: anthropic.Anthropic,
    topic: str,
    agent_id: int,
    iteration: int,
    previous_iterations: list[dict],
) -> dict:
    if _shutdown.is_set():
        raise RuntimeError(f"Agent {agent_id + 1} cancelled before start.")

    t0 = time.time()
    print(f"  [Agent {agent_id + 1}] Starting research...")
    try:
        report = run_research_agent(
            client, topic, agent_id, iteration, previous_iterations,
            shutdown_event=_shutdown,
        )
    except Exception:
        _shutdown.set()
        raise

    if _shutdown.is_set():
        raise RuntimeError(f"Agent {agent_id + 1} cancelled after research.")

    elapsed = time.time() - t0
    print(f"  [Agent {agent_id + 1}] Research done ({elapsed:.1f}s). Evaluating...")

    try:
        score, feedback = evaluate_report(client, report, topic)
    except Exception:
        _shutdown.set()
        raise

    elapsed = time.time() - t0
    print(f"  [Agent {agent_id + 1}] Score: {score}/10  |  {elapsed:.1f}s  |  {feedback[:80]}")
    return {
        "agent_id": agent_id,
        "iteration": iteration + 1,
        "report": report,
        "score": score,
        "feedback": feedback,
        "elapsed_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# One iteration: launch NUM_AGENTS workers with staggered starts
# ---------------------------------------------------------------------------

def run_iteration(
    client: anthropic.Anthropic,
    topic: str,
    iteration: int,
    previous_iterations: list[dict],
) -> list[dict]:
    """
    Run NUM_AGENTS agents for one iteration, staggered by AGENT_STAGGER_SECONDS.
    Returns list of result dicts, one per agent.
    """
    print(f"\n--- Iteration {iteration + 1} | {NUM_AGENTS} agents ---")
    futures = []
    for agent_id in range(NUM_AGENTS):
        if agent_id > 0:
            print(f"  Waiting {AGENT_STAGGER_SECONDS}s before starting Agent {agent_id + 1}...")
            time.sleep(AGENT_STAGGER_SECONDS)
        futures.append(
            _executor.submit(
                _agent_worker, client, topic, agent_id, iteration, previous_iterations
            )
        )

    results = []
    errors = []
    for f in futures:
        try:
            results.append(f.result())
        except Exception as exc:
            errors.append(exc)

    if errors:
        raise errors[0]
    return results


# ---------------------------------------------------------------------------
# Action mode: one iteration, read/write shared history file
# ---------------------------------------------------------------------------

def action_mode(topic: str, iteration: int, history_file: Path, output_dir: Path) -> None:
    """Run a single iteration. Used by the self-invoking GitHub Action."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_iterations: list[dict] = []
    if history_file.exists():
        data = json.loads(history_file.read_text())
        previous_iterations = data.get("iterations", [])

    agent_results = run_iteration(client, topic, iteration, previous_iterations)

    # Pick best result from this iteration
    best = max(agent_results, key=lambda r: r["score"])
    scores = [r["score"] for r in agent_results]
    print(f"\nIteration {iteration + 1} scores: {scores}  |  Best: {best['score']}/10")

    iter_record = {
        "iteration": iteration + 1,
        "results": agent_results,
        "best_score": best["score"],
        "best_agent": best["agent_id"],
    }

    history = {
        "topic": topic,
        "target_score": TARGET_SCORE,
        "iterations": previous_iterations + [iter_record],
    }
    history_file.write_text(json.dumps(history, indent=2))

    report_path = output_dir / f"iteration_{iteration + 1:02d}_score{best['score']}.md"
    report_path.write_text(
        f"# Iteration {iteration + 1} Report: {topic}\n\n"
        f"> **Best Score:** {best['score']}/10 (Agent {best['agent_id'] + 1})  \n"
        f"> **All Scores:** {scores}  \n"
        f"> **Feedback:** {best['feedback']}\n\n---\n\n"
        + best["report"]
    )

    done = best["score"] >= TARGET_SCORE
    print(f"\n{'='*50}")
    print(f"Iteration {iteration + 1} complete — best score {best['score']}/10")
    print(f"Status: {'DONE (target reached)' if done else 'continuing...'}")
    print(f"{'='*50}")

    (output_dir / "iteration_output.json").write_text(json.dumps({
        "score": best["score"],
        "done": done,
        "next_iteration": iteration + 1,
        "report_path": str(report_path),
    }, indent=2))

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"score={best['score']}\n")
            f.write(f"done={'true' if done else 'false'}\n")
            f.write(f"next_iteration={iteration + 1}\n")


# ---------------------------------------------------------------------------
# Local full-loop mode
# ---------------------------------------------------------------------------

def local_mode(topic: str, output_dir: Path, max_iterations: int = MAX_ITERATIONS) -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Topic: {topic}")
    print(f"Agents/iteration: {NUM_AGENTS}  |  Max iterations: {max_iterations}  |  Target: {TARGET_SCORE}/10")
    print(f"{'='*60}")

    previous_iterations: list[dict] = []
    best_result: dict | None = None
    best_score = -1

    for i in range(max_iterations):
        agent_results = run_iteration(client, topic, i, previous_iterations)
        best = max(agent_results, key=lambda r: r["score"])
        scores = [r["score"] for r in agent_results]
        print(f"Scores: {scores}  |  Best: {best['score']}/10")

        iter_record = {"iteration": i + 1, "results": agent_results, "best_score": best["score"]}
        previous_iterations.append(iter_record)

        if best["score"] > best_score:
            best_score = best["score"]
            best_result = best

        if best["score"] >= TARGET_SCORE:
            print(f"\n✓ Target reached at iteration {i + 1}!")
            break
    else:
        print(f"\n✗ Max iterations reached. Best score: {best_score}/10")

    assert best_result is not None
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    history_path = output_dir / f"run_history_{timestamp}.json"
    history_path.write_text(json.dumps({
        "topic": topic, "timestamp": timestamp,
        "best_score": best_score, "iterations": previous_iterations,
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
                        help="Single iteration mode (0-indexed). Omit for full local loop.")
    parser.add_argument("--history-file", default="outputs/history.json")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()

    try:
        if args.iteration is not None:
            action_mode(args.topic, args.iteration, Path(args.history_file), Path(args.output_dir))
        else:
            local_mode(args.topic, Path(args.output_dir), args.max_iterations)
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
