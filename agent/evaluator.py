"""
Evaluator — runs test suite and benchmark against a candidate.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_count_unique import run_tests
from benchmarks.bench import measure_runtime_ms


def evaluate(source: str) -> dict:
    """
    Run tests and benchmark the candidate source.

    Returns:
        {
            "tests_passed": bool,
            "test_detail": str,
            "runtime_ms": float | None,   # None if tests failed
        }
    """
    tests_passed, test_detail = run_tests(source)

    runtime_ms = None
    if tests_passed:
        try:
            runtime_ms = measure_runtime_ms(source)
        except Exception as exc:
            tests_passed = False
            test_detail = f"Runtime measurement failed: {exc}"

    return {
        "tests_passed": tests_passed,
        "test_detail": test_detail,
        "runtime_ms": runtime_ms,
    }
