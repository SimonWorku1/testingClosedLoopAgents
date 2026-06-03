#!/usr/bin/env python3
"""
Closed-loop code optimizer — entry point.

Usage: python run_agent.py

Optimizes count_unique(items) over MAX_ITERATIONS iterations using:
  - Optimizer LLM: generates candidate implementations
  - Critic LLM: reviews candidate for correctness/cheating before benchmarking
  - Evaluator: runs test suite + timeit benchmark
  - Controller: accept only if tests pass AND runtime improves >= 1%

Logs each iteration to results/iterations.jsonl.
Always keeps the best accepted version — never regresses.
Stops after MAX_ITERATIONS or MAX_STAGNANT consecutive non-improvements.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.llm_client import make_client
from agent.optimizer import generate_candidate
from agent.critic import review_candidate
from agent.evaluator import evaluate
from agent.controller import should_accept
from benchmarks.bench import BASELINE_SOURCE, measure_runtime_ms

MAX_ITERATIONS = 10
MAX_STAGNANT = 3      # stop after this many consecutive non-improvements
RESULTS_DIR = Path("results")
LOG_FILE = RESULTS_DIR / "iterations.jsonl"
BEST_FILE = RESULTS_DIR / "best_candidate.py"


def _log(record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _save_best(source: str, iteration: int, runtime_ms: float) -> None:
    header = (
        f"# Best candidate — iteration {iteration}, "
        f"runtime {runtime_ms:.3f}ms\n\n"
    )
    BEST_FILE.write_text(header + source)


def main() -> None:
    client = make_client()
    print(f"LLM provider: {client.provider_name}  model: {client.model}")

    # Measure baseline
    print("\nMeasuring baseline...")
    baseline_ms = measure_runtime_ms(BASELINE_SOURCE)
    print(f"Baseline runtime: {baseline_ms:.3f}ms")

    best_source = BASELINE_SOURCE
    best_ms = baseline_ms
    previous_results: list[dict] = []
    stagnant = 0

    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'='*55}")
        print(f"Iteration {iteration + 1}/{MAX_ITERATIONS}  |  best so far: {best_ms:.3f}ms")
        print(f"{'='*55}")

        # --- Optimizer generates a candidate ---
        print("  [Optimizer] Generating candidate...")
        try:
            candidate_source = generate_candidate(
                client,
                iteration=iteration,
                previous_results=previous_results,
                critic_feedback="",
            )
        except Exception as exc:
            print(f"  [Optimizer] ERROR: {exc}", file=sys.stderr)
            _log({
                "iteration": iteration + 1,
                "runtime_ms": None,
                "previous_runtime_ms": best_ms,
                "improvement_pct": 0.0,
                "accepted": False,
                "reason": f"optimizer_error: {exc}",
            })
            stagnant += 1
            if stagnant >= MAX_STAGNANT:
                print(f"\nStopping: {MAX_STAGNANT} consecutive non-improvements.")
                break
            continue

        print(f"  [Optimizer] Generated ({len(candidate_source)} chars)")

        # --- Critic reviews the candidate ---
        print("  [Critic] Reviewing candidate...")
        try:
            approved, critic_feedback = review_candidate(client, candidate_source)
        except Exception as exc:
            print(f"  [Critic] ERROR: {exc}", file=sys.stderr)
            approved, critic_feedback = True, f"critic_error: {exc}"

        print(f"  [Critic] {'APPROVED' if approved else 'REJECTED'}: {critic_feedback[:100]}")

        if not approved:
            # Regenerate with critic feedback
            print("  [Optimizer] Regenerating with critic feedback...")
            try:
                candidate_source = generate_candidate(
                    client,
                    iteration=iteration,
                    previous_results=previous_results,
                    critic_feedback=critic_feedback,
                )
                approved2, feedback2 = review_candidate(client, candidate_source)
                if not approved2:
                    print(f"  [Critic] Still rejected after retry: {feedback2[:100]}")
                    _log({
                        "iteration": iteration + 1,
                        "runtime_ms": None,
                        "previous_runtime_ms": best_ms,
                        "improvement_pct": 0.0,
                        "accepted": False,
                        "reason": f"critic_rejected: {feedback2}",
                    })
                    stagnant += 1
                    if stagnant >= MAX_STAGNANT:
                        print(f"\nStopping: {MAX_STAGNANT} consecutive non-improvements.")
                        break
                    continue
                critic_feedback = feedback2
            except Exception as exc:
                print(f"  [Optimizer] Retry ERROR: {exc}", file=sys.stderr)
                stagnant += 1
                continue

        # --- Evaluator: tests + benchmark ---
        print("  [Evaluator] Running tests and benchmark...")
        try:
            eval_result = evaluate(candidate_source)
        except Exception as exc:
            print(f"  [Evaluator] ERROR: {exc}", file=sys.stderr)
            _log({
                "iteration": iteration + 1,
                "runtime_ms": None,
                "previous_runtime_ms": best_ms,
                "improvement_pct": 0.0,
                "accepted": False,
                "reason": f"evaluator_error: {exc}",
            })
            stagnant += 1
            if stagnant >= MAX_STAGNANT:
                print(f"\nStopping: {MAX_STAGNANT} consecutive non-improvements.")
                break
            continue

        print(f"  [Evaluator] Tests: {'PASS' if eval_result['tests_passed'] else 'FAIL'} | "
              f"runtime: {eval_result['runtime_ms']:.3f}ms" if eval_result["runtime_ms"] else
              f"  [Evaluator] Tests: {'PASS' if eval_result['tests_passed'] else 'FAIL'}")

        # --- Controller: accept/reject ---
        accepted, reason = should_accept(eval_result, best_ms)
        runtime_ms = eval_result.get("runtime_ms") or 0.0
        improvement_pct = (best_ms - runtime_ms) / best_ms * 100 if runtime_ms else 0.0

        record = {
            "iteration": iteration + 1,
            "runtime_ms": runtime_ms if runtime_ms else None,
            "previous_runtime_ms": best_ms,
            "improvement_pct": round(improvement_pct, 2),
            "accepted": accepted,
            "reason": reason,
        }
        _log(record)

        if accepted:
            best_source = candidate_source
            best_ms = runtime_ms
            stagnant = 0
            _save_best(best_source, iteration + 1, best_ms)
            print(f"  [Controller] ACCEPTED — {reason}")
        else:
            stagnant += 1
            print(f"  [Controller] REJECTED — {reason}")

        if stagnant >= MAX_STAGNANT:
            print(f"\nStopping: {MAX_STAGNANT} consecutive non-improvements.")
            break

        previous_results.append(record)

    # Final summary
    total_improvement = (baseline_ms - best_ms) / baseline_ms * 100
    print(f"\n{'='*55}")
    print(f"DONE — {len(previous_results) + 1} iterations")
    print(f"Baseline:  {baseline_ms:.3f}ms")
    print(f"Best:      {best_ms:.3f}ms")
    print(f"Improvement: {total_improvement:.1f}%")
    print(f"Best candidate saved to: {BEST_FILE}")
    print(f"Full log: {LOG_FILE}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
