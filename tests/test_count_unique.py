"""
Test suite for count_unique implementations.
All candidates must pass 100% of these tests.
"""

import importlib
import types


def _load_candidate(source: str) -> types.ModuleType:
    """Compile and exec source, return module-like namespace."""
    ns: dict = {}
    exec(compile(source, "<candidate>", "exec"), ns)
    mod = types.SimpleNamespace(**ns)
    if not callable(getattr(mod, "count_unique", None)):
        raise AttributeError("count_unique not defined in candidate source")
    return mod


def run_tests(source: str) -> tuple[bool, str]:
    """
    Run all test cases against the candidate source.
    Returns (all_passed: bool, detail: str).
    """
    try:
        mod = _load_candidate(source)
    except Exception as exc:
        return False, f"Compile/load error: {exc}"

    fn = mod.count_unique
    failures = []

    cases = [
        # (description, args, expected)
        ("empty list", ([],), 0),
        ("single element", ([42],), 1),
        ("all duplicates", ([7, 7, 7, 7],), 1),
        ("all unique", ([1, 2, 3, 4, 5],), 5),
        ("mixed ints", ([1, 2, 2, 3, 3, 3, 4],), 4),
        ("strings", (["a", "b", "a", "c"],), 3),
        ("mixed types (int+str)", ([1, "1", 1, "1"],), 2),
        ("single duplicate pair", ([0, 0],), 1),
        ("booleans", ([True, False, True],), 2),
        ("none values", ([None, None, 1],), 2),
        ("large list all same", ([99] * 10_000,), 1),
        ("large list all unique", (list(range(5_000)),), 5_000),
        ("large list half unique", (list(range(1_000)) * 2,), 1_000),
        ("floats", ([1.0, 2.0, 1.0, 3.0],), 3),
        ("negative numbers", ([-1, -2, -1, 0],), 3),
        ("tuples as elements", ([(1, 2), (1, 2), (3, 4)],), 2),
    ]

    for desc, args, expected in cases:
        try:
            result = fn(*args)
        except Exception as exc:
            failures.append(f"  FAIL [{desc}]: raised {type(exc).__name__}: {exc}")
            continue
        if result != expected:
            failures.append(f"  FAIL [{desc}]: expected {expected}, got {result}")

    if failures:
        return False, "Test failures:\n" + "\n".join(failures)
    return True, f"All {len(cases)} tests passed."


# Allow direct pytest usage
def _make_pytest_cases():
    from benchmarks.bench import BASELINE_SOURCE
    return BASELINE_SOURCE


if __name__ == "__main__":
    # Quick smoke test of the baseline
    baseline = """
def count_unique(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return len(result)
"""
    passed, detail = run_tests(baseline)
    print(f"Baseline: {'PASS' if passed else 'FAIL'}")
    print(detail)
