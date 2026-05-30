#!/usr/bin/env python3
"""
Autonomous Server Auto-Scaler — closed-loop control agent.

Drives a cluster's CPU utilization to a 15% target (band 14-16%) by adjusting
the server instance count each iteration. Succeeds when CPU stays in band for
3 consecutive iterations; fails if it can't stabilize within 20 iterations.

Single-job design with durable checkpointing: after each iteration it writes
state.json and (when --git-checkpoint is set) commits + pushes it, so a dead
VM resumes from the last completed iteration instead of restarting blind.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from cluster_environment import ClusterEnvironment
from llm_client import LLMClient, make_client
from scaler_agent import BAND, TARGET_CPU, decide_instances

MAX_ITERATIONS = 20
CONSECUTIVE_REQUIRED = 3
INITIAL_INSTANCES = 5
MAX_STEP_RETRIES = 3  # retries per iteration on transient API/JSON errors


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_header(resuming: bool, start_iter: int) -> None:
    print(f"\n{'='*64}")
    print("  AUTONOMOUS SERVER AUTO-SCALER")
    print(f"  Target: {TARGET_CPU:.0f}% CPU  (band {BAND[0]:.0f}-{BAND[1]:.0f}%)")
    print(f"  Stabilize: {CONSECUTIVE_REQUIRED} consecutive in-band iterations")
    print(f"  Crash if not stable within {MAX_ITERATIONS} iterations")
    if resuming:
        print(f"  >> Resuming from iteration {start_iter}")
    print(f"{'='*64}")


def _diagnose(record: dict, prev: dict | None) -> str:
    """Plain-English explanation of what went wrong with this reading."""
    cpu = record["cpu_utilization"]
    err = record["error"]

    if record["in_band"]:
        return f"on target — CPU {cpu:.2f}% is inside the {BAND[0]:.0f}-{BAND[1]:.0f}% band"

    if cpu >= 100.0:
        msg = ("CPU pinned at 100% — cluster is SATURATED and badly "
               "under-provisioned (true demand is even higher than shown)")
    elif cpu <= 1.0:
        msg = ("CPU floored at 1% — massively over-provisioned, "
               "burning budget on idle instances")
    elif err > 0:
        msg = (f"CPU {cpu:.2f}% is {err:+.2f}% ABOVE the 15% target — "
               "too few instances, still under-provisioned")
    else:
        msg = (f"CPU {cpu:.2f}% is {err:+.2f}% BELOW the 15% target — "
               "too many instances, over-provisioned")

    # Compare against the previous iteration to flag overshoot / oscillation.
    if prev is not None:
        if prev["in_band"]:
            msg += " (overshot — left the band after being inside it last iteration)"
        elif (prev["error"] > 0) != (err > 0) and abs(err) > 0.5:
            flipped = "high→low" if prev["error"] > 0 else "low→high"
            msg += f" (overshot — flipped {flipped} vs last iteration)"
    return msg


def _print_step(record: dict, consecutive: int, diagnosis: str,
                next_instances: int | None, reasoning: str | None) -> None:
    cpu = record["cpu_utilization"]
    in_band = BAND[0] <= cpu <= BAND[1]
    flag = f"IN BAND ({consecutive}/{CONSECUTIVE_REQUIRED})" if in_band else "out of band"
    if cpu >= 100.0:
        flag += "  [SATURATED]"
    elif cpu <= 1.0:
        flag += "  [over-provisioned]"
    print(
        f"\nIter {record['iteration']:2d} | "
        f"{record['instances']:>4d} instances -> CPU {cpu:6.2f}% | "
        f"error {record['error']:+6.2f}% | {flag}"
    )
    print(f"  diagnosis: {diagnosis}")
    if next_instances is not None:
        delta = next_instances - record["instances"]
        direction = "scale UP" if delta > 0 else ("scale DOWN" if delta < 0 else "hold")
        print(f"  thought: {reasoning or '(none)'}")
        print(f"  -> {direction}: {record['instances']} -> {next_instances} instances "
              f"({delta:+d})")


def _print_summary(history: list[dict], success: bool) -> None:
    print(f"\n{'='*64}")
    if success:
        print("  STABILIZED — target reached")
    else:
        print("  FAILED — cluster crashed (did not stabilize in time)")
    print(f"  Iterations used: {len(history)} / {MAX_ITERATIONS}")
    if history:
        final = history[-1]
        print(f"  Final: {final['instances']} instances -> CPU {final['cpu_utilization']:.2f}%")
    print(f"{'='*64}")


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _save_state(state_file: Path, instances: int, consecutive: int,
                history: list[dict]) -> None:
    state = {
        "next_instances": instances,
        "consecutive_in_band": consecutive,
        "history": history,
    }
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_file)


def _git_checkpoint(checkpoint_dir: Path, iteration: int) -> None:
    """Commit + push the checkpoint dir so progress survives VM death.
    Failures are logged but never abort the run — local state.json is intact."""
    d = str(checkpoint_dir)
    try:
        subprocess.run(["git", "-C", d, "add", "-A"], check=True)
        staged = subprocess.run(["git", "-C", d, "diff", "--cached", "--quiet"])
        if staged.returncode == 0:
            return
        subprocess.run(["git", "-C", d, "commit", "-m", f"checkpoint: iteration {iteration}"],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", d, "push", "origin", "HEAD:scaler-state"],
                       check=True, stdout=subprocess.DEVNULL)
        print(f"  [checkpoint: iteration {iteration} pushed to scaler-state]")
    except subprocess.CalledProcessError as exc:
        print(f"  [checkpoint warning: git push failed ({exc}); progress kept locally]",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------

def run(output_dir: Path, state_file: Path, checkpoint_dir: Path | None = None) -> bool:
    client = make_client()
    print(f"  [LLM provider: {client.provider_name}  |  model: {client.model}]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # The hidden environment is NOT durable across VMs — its internal traffic
    # state resets on resume. We rebuild knowledge from the saved reading trace,
    # which is what the agent reasons over anyway.
    env = ClusterEnvironment()

    if state_file.exists():
        state = json.loads(state_file.read_text())
        instances = state["next_instances"]
        consecutive = state["consecutive_in_band"]
        history = state["history"]
        start_iter = len(history) + 1

        # Already concluded? Don't re-run — report the saved outcome.
        if consecutive >= CONSECUTIVE_REQUIRED:
            _print_header(resuming=True, start_iter=start_iter)
            print("\n  Saved state is already stabilized — nothing to do.")
            _print_summary(history, success=True)
            return True
        if len(history) >= MAX_ITERATIONS:
            _print_header(resuming=True, start_iter=start_iter)
            print("\n  Saved state already exhausted the iteration budget.")
            _print_summary(history, success=False)
            return False
        _print_header(resuming=True, start_iter=start_iter)
    else:
        instances = INITIAL_INSTANCES
        consecutive = 0
        history = []
        start_iter = 1
        _print_header(resuming=False, start_iter=1)

    success = False
    for iteration in range(start_iter, MAX_ITERATIONS + 1):
        # --- act: provision instances, read resulting CPU -------------------
        last_exc = None
        for attempt in range(1, MAX_STEP_RETRIES + 1):
            try:
                result = env.set_instance_count(instances)
                break
            except Exception as exc:  # the env itself shouldn't throw, but be safe
                last_exc = exc
                print(f"  [iter {iteration} attempt {attempt}/{MAX_STEP_RETRIES} "
                      f"failed: {exc}]", file=sys.stderr)
                if attempt < MAX_STEP_RETRIES:
                    time.sleep(2 ** attempt)
        else:
            _save_state(state_file, instances, consecutive, history)
            raise last_exc

        cpu = result["cpu_utilization"]
        error = round(cpu - TARGET_CPU, 2)
        in_band = BAND[0] <= cpu <= BAND[1]
        consecutive = consecutive + 1 if in_band else 0

        prev = history[-1] if history else None  # reading before this one
        record = {
            "iteration": iteration,
            "instances": instances,
            "cpu_utilization": cpu,
            "error": error,
            "in_band": in_band,
            "consecutive_in_band": consecutive,
        }
        diagnosis = _diagnose(record, prev)
        record["diagnosis"] = diagnosis
        history.append(record)

        # --- check terminal condition --------------------------------------
        if consecutive >= CONSECUTIVE_REQUIRED:
            _print_step(record, consecutive, diagnosis, next_instances=None, reasoning=None)
            success = True
            _write_logs(output_dir, record, next_instances=None, reasoning="STABILIZED")
            _save_state(state_file, instances, consecutive, history)
            if checkpoint_dir is not None:
                _git_checkpoint(checkpoint_dir, iteration)
            break

        # --- decide: ask the agent for the next instance count -------------
        next_instances, reasoning = None, None
        last_exc = None
        for attempt in range(1, MAX_STEP_RETRIES + 1):
            try:
                next_instances, reasoning = decide_instances(client, history, instances)
                break
            except Exception as exc:
                last_exc = exc
                print(f"  [iter {iteration} decide attempt {attempt}/{MAX_STEP_RETRIES} "
                      f"failed: {exc}]", file=sys.stderr)
                if attempt < MAX_STEP_RETRIES:
                    time.sleep(2 ** attempt)
        else:
            _save_state(state_file, instances, consecutive, history)
            raise last_exc

        _print_step(record, consecutive, diagnosis, next_instances, reasoning)
        _write_logs(output_dir, record, next_instances, reasoning)

        instances = next_instances

        # Persist + durably checkpoint after every iteration
        _save_state(state_file, instances, consecutive, history)
        if checkpoint_dir is not None:
            _git_checkpoint(checkpoint_dir, iteration)

    _print_summary(history, success)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    (output_dir / f"run_{timestamp}.json").write_text(json.dumps({
        "success": success,
        "iterations": len(history),
        "history": history,
    }, indent=2))

    return success


def _write_logs(output_dir: Path, record: dict, next_instances: int | None,
                reasoning: str | None) -> None:
    with open(output_dir / "trace.txt", "a") as f:
        f.write(
            f"Iter {record['iteration']:2d} | {record['instances']:>4d} instances "
            f"-> CPU {record['cpu_utilization']:6.2f}% | error {record['error']:+6.2f}% | "
            f"{'in-band' if record['in_band'] else 'out'} "
            f"({record['consecutive_in_band']}/{CONSECUTIVE_REQUIRED})\n"
            f"        diagnosis: {record.get('diagnosis', '')}\n"
            + (f"        -> next {next_instances} instances ({reasoning})\n"
               if next_instances is not None else f"        -> {reasoning}\n")
        )
    with open(output_dir / "trace.jsonl", "a") as f:
        out = dict(record)
        out["next_instances"] = next_instances
        out["reasoning"] = reasoning
        f.write(json.dumps(out) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="outputs")
    parser.add_argument("--state-file", default="outputs/state.json")
    parser.add_argument("--git-checkpoint", default=None,
                        help="Directory (a git worktree on the state branch) to "
                             "commit+push after each iteration for durable resume.")
    args = parser.parse_args()

    state_file = Path(args.state_file)
    checkpoint_dir = Path(args.git_checkpoint) if args.git_checkpoint else None

    try:
        success = run(Path(args.output_dir), state_file, checkpoint_dir)
    except Exception as exc:
        # A real crash (e.g. VM/API failure) — exit non-zero so CI auto-resumes
        # from the last checkpoint.
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Reaching here means the loop ran to a conclusion (stabilized OR exhausted
    # its iteration budget). Both are clean completions — exit 0 so the workflow
    # does NOT auto-resume past the iteration limit. The stabilize/fail outcome
    # is recorded in state.json, run_*.json, and the job summary.
    print(f"\nOutcome: {'STABILIZED' if success else 'FAILED TO STABILIZE'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
