"""
Controller — accept/reject logic for candidate implementations.
Accepts a candidate only if:
  1. All tests pass (100%)
  2. Runtime improves by >= MIN_IMPROVEMENT_PCT over the current best
"""

MIN_IMPROVEMENT_PCT = 1.0  # must beat current best by at least 1%


def should_accept(
    eval_result: dict,
    current_best_ms: float,
) -> tuple[bool, str]:
    """
    Decide whether to accept a candidate.

    Args:
        eval_result: dict from evaluator.evaluate()
        current_best_ms: runtime of the current accepted best

    Returns:
        (accepted: bool, reason: str)
    """
    if not eval_result["tests_passed"]:
        return False, f"Tests failed: {eval_result['test_detail']}"

    runtime_ms = eval_result["runtime_ms"]
    if runtime_ms is None:
        return False, "No runtime measured (tests may have failed)"

    improvement_pct = (current_best_ms - runtime_ms) / current_best_ms * 100

    if improvement_pct < MIN_IMPROVEMENT_PCT:
        return False, (
            f"Insufficient improvement: {improvement_pct:.2f}% "
            f"(need >= {MIN_IMPROVEMENT_PCT}%). "
            f"candidate={runtime_ms:.3f}ms, best={current_best_ms:.3f}ms"
        )

    return True, (
        f"Accepted: {improvement_pct:.2f}% improvement "
        f"({current_best_ms:.3f}ms → {runtime_ms:.3f}ms)"
    )
