"""
Benchmark runner for count_unique candidates.
Uses timeit.repeat() for reliable measurements.
"""

import timeit
import types

BASELINE_SOURCE = """
def count_unique(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return len(result)
"""

# Benchmark workload: mix of sizes and duplicate ratios
_WORKLOADS = [
    list(range(500)),                      # 500 unique
    [x % 100 for x in range(1_000)],       # 1000 items, 100 unique
    [x % 10 for x in range(2_000)],        # high-dup
    list(range(200)) * 3,                  # repeat with 200 unique
]

_REPEAT = 5
_NUMBER = 20


def _load(source: str) -> types.SimpleNamespace:
    ns: dict = {}
    exec(compile(source, "<bench>", "exec"), ns)
    if "count_unique" not in ns:
        raise AttributeError("count_unique not defined")
    return types.SimpleNamespace(**ns)


def measure_runtime_ms(source: str) -> float:
    """
    Return the median wall-clock time in milliseconds to run
    count_unique across all workloads, using timeit.repeat().
    """
    mod = _load(source)
    fn = mod.count_unique

    total_times = []
    for workload in _WORKLOADS:
        # timeit.repeat returns a list of total times for `number` reps
        times = timeit.repeat(
            stmt=lambda w=workload: fn(w),
            repeat=_REPEAT,
            number=_NUMBER,
        )
        # per-call time in ms (best of repeats / number of calls)
        best_per_call_ms = min(times) / _NUMBER * 1000
        total_times.append(best_per_call_ms)

    # Return mean across workloads
    return sum(total_times) / len(total_times)


if __name__ == "__main__":
    ms = measure_runtime_ms(BASELINE_SOURCE)
    print(f"Baseline runtime: {ms:.4f} ms")
