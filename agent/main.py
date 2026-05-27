#!/usr/bin/env python3
"""
Autonomous closed-loop research agent.

Runs NUM_AGENTS parallel research agents per iteration.
Each iteration the agents receive all previous reports + scores as context.
Stops when any report scores >= TARGET_SCORE, or after MAX_ITERATIONS.
Returns the highest-scoring report.
"""

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import anthropic

from evaluator import evaluate_report
from web_researcher import run_research_agent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_AGENTS = 3
MAX_ITERATIONS = 5
TARGET_SCORE = 8

# ---------------------------------------------------------------------------
# Async helpers (Claude SDK is sync; we thread-pool it)
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=NUM_AGENTS * 2)


def _research_worker(
    client: anthropic.Anthropic,
    topic: str,
    agent_id: int,
    iteration: int,
    previous_iterations: list[dict],
) -> dict:
    """Thread worker: run one research agent and evaluate its report."""
    t0 = time.time()
    print(f"  [Agent {agent_id + 1}] Starting research (iteration {iteration + 1})...")
    report = run_research_agent(client, topic, agent_id, iteration, previous_iterations)
    elapsed_research = time.time() - t0

    print(f"  [Agent {agent_id + 1}] Research done ({elapsed_research:.1f}s). Evaluating...")
    score, feedback = evaluate_report(client, report, topic)
    elapsed_total = time.time() - t0

    print(
        f"  [Agent {agent_id + 1}] Score: {score}/10  |  "
        f"Total time: {elapsed_total:.1f}s  |  Feedback: {feedback[:80]}..."
    )
    return {
        "agent_id": agent_id,
        "iteration": iteration + 1,
        "report": report,
        "score": score,
        "feedback": feedback,
        "elapsed_seconds": round(elapsed_total, 1),
    }


async def run_iteration_async(
    loop: asyncio.AbstractEventLoop,
    client: anthropic.Anthropic,
    topic: str,
    iteration: int,
    previous_iterations: list[dict],
) -> list[dict]:
    """Launch NUM_AGENTS research workers in parallel and await all results."""
    futures = [
        loop.run_in_executor(
            _executor,
            _research_worker,
            client,
            topic,
            agent_id,
            iteration,
            previous_iterations,
        )
        for agent_id in range(NUM_AGENTS)
    ]
    return list(await asyncio.gather(*futures))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def orchestrate(topic: str, output_dir: Path) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    loop = asyncio.get_event_loop()

    all_iterations: list[dict] = []
    best_result: dict | None = None
    best_score: int = -1

    print(f"\n{'='*60}")
    print(f"Topic: {topic}")
    print(f"Agents per iteration: {NUM_AGENTS}  |  Max iterations: {MAX_ITERATIONS}")
    print(f"Target score: {TARGET_SCORE}/10")
    print(f"{'='*60}\n")

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{MAX_ITERATIONS} ---")
        iter_start = time.time()

        results = await run_iteration_async(
            loop, client, topic, iteration, all_iterations
        )

        iter_elapsed = time.time() - iter_start
        iter_record = {"iteration": iteration + 1, "results": results}
        all_iterations.append(iter_record)

        # Track best overall
        for res in results:
            if res["score"] > best_score:
                best_score = res["score"]
                best_result = res

        scores = [r["score"] for r in results]
        print(
            f"\nIteration {iteration + 1} complete in {iter_elapsed:.1f}s  |  "
            f"Scores: {scores}  |  Best so far: {best_score}/10"
        )

        # Check stopping condition
        winners = [r for r in results if r["score"] >= TARGET_SCORE]
        if winners:
            winner = max(winners, key=lambda r: r["score"])
            print(
                f"\n✓ Target score reached! Agent {winner['agent_id'] + 1} "
                f"scored {winner['score']}/10 in iteration {iteration + 1}."
            )
            best_result = winner
            break
    else:
        print(
            f"\n✗ Max iterations ({MAX_ITERATIONS}) reached. "
            f"Best score: {best_score}/10 — returning best report."
        )

    # ---------------------------------------------------------------------------
    # Persist outputs
    # ---------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Full run history (JSON)
    history_path = output_dir / f"run_history_{timestamp}.json"
    history_path.write_text(
        json.dumps(
            {
                "topic": topic,
                "timestamp": timestamp,
                "num_agents": NUM_AGENTS,
                "max_iterations": MAX_ITERATIONS,
                "target_score": TARGET_SCORE,
                "iterations_run": len(all_iterations),
                "best_score": best_score,
                "all_iterations": all_iterations,
            },
            indent=2,
        )
    )

    # Best report (Markdown)
    assert best_result is not None
    report_path = output_dir / f"best_report_{timestamp}.md"
    report_path.write_text(
        f"# Research Report: {topic}\n\n"
        f"> **Quality Score:** {best_result['score']}/10  \n"
        f"> **Produced by:** Agent {best_result['agent_id'] + 1}, "
        f"Iteration {best_result['iteration']}  \n"
        f"> **Evaluator Feedback:** {best_result['feedback']}\n\n"
        f"---\n\n"
        + best_result["report"]
    )

    # Summary to stdout (for GitHub Action logs)
    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(f"Topic         : {topic}")
    print(f"Best score    : {best_score}/10")
    print(f"Agent         : {best_result['agent_id'] + 1}")
    print(f"Iteration     : {best_result['iteration']}")
    print(f"Feedback      : {best_result['feedback']}")
    print(f"Report saved  : {report_path}")
    print(f"History saved : {history_path}")
    print(f"{'='*60}\n")

    return {
        "best_report_path": str(report_path),
        "history_path": str(history_path),
        "best_score": best_score,
        "iterations_run": len(all_iterations),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<research topic>\" [output_dir]")
        sys.exit(1)

    topic = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs")

    result = asyncio.run(orchestrate(topic, output_dir))

    # Write a machine-readable summary for the GitHub Action step
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
