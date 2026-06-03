"""Shared rate-limit pacing for Anthropic API calls."""

import time
from datetime import datetime, timezone


def pace(headers) -> None:
    """Sleep only when the token bucket is nearly empty, based on response headers."""
    remaining = headers.get("anthropic-ratelimit-tokens-remaining")
    if remaining is None:
        return
    try:
        remaining = int(remaining)
    except (TypeError, ValueError):
        return
    if remaining < 8000:
        reset = headers.get("anthropic-ratelimit-tokens-reset")
        wait = 0.0
        if reset:
            try:
                t = datetime.fromisoformat(reset.replace("Z", "+00:00"))
                # Ensure timezone-aware comparison — Anthropic always sends UTC.
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                wait = max(0.0, (t - datetime.now(timezone.utc)).total_seconds())
            except ValueError:
                wait = 10.0
        if wait > 0:
            print(f"    [rate limit] {remaining} tokens remaining — sleeping {wait:.1f}s...")
            time.sleep(min(wait, 60))
