"""
Provider-agnostic LLM client with rate-limit self-pacing.

Why this exists
---------------
The agents in this repo were written against the Anthropic SDK, but at exam
time the API key handed to us may belong to a *different* provider (e.g.
OpenAI) and we won't know its rate-limit tier in advance. This module:

  1. Picks the provider at runtime from whichever API key is present
     (override with LLM_PROVIDER / LLM_MODEL env vars).
  2. Exposes one `complete(system=..., user=...)` call so agent code never
     touches a provider-specific SDK directly.
  3. Reads rate-limit headers off every response and *self-paces*: it honours
     `retry-after` on 429s (reactive floor) and proactively sleeps when the
     token bucket is nearly empty. The same code therefore runs correctly on
     a tiny Tier-1 key or a huge Tier-4 key with no config change.
"""

import os
import re
import time
from datetime import datetime, timezone


def _parse_duration(s: str | None) -> float:
    """Parse OpenAI reset strings like '6m0s', '1.5s', '90ms' into seconds."""
    if not s:
        return 0.0
    # Plain number → already seconds.
    try:
        return float(s)
    except (TypeError, ValueError):
        pass
    total = 0.0
    # Order matters: try 'ms' before 'm'/'s'.
    for value, unit in re.findall(r"([\d.]+)\s*(ms|h|m|s)", s):
        v = float(value)
        total += {"h": v * 3600, "m": v * 60, "s": v, "ms": v / 1000}[unit]
    return total


# ---------------------------------------------------------------------------
# Provider adapters — each knows how to call its SDK and read its headers.
# ---------------------------------------------------------------------------

class _OpenAIProvider:
    name = "openai"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str):
        from openai import OpenAI
        import openai
        self._client = OpenAI(api_key=api_key)
        self._rate_limit_error = openai.RateLimitError

    def create(self, system: str, user: str, model: str, max_tokens: int):
        resp = self._client.chat.completions.with_raw_response.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        msg = resp.parse()
        return msg.choices[0].message.content, resp.headers

    def is_rate_limit(self, exc: Exception) -> bool:
        return isinstance(exc, self._rate_limit_error)

    def remaining_tokens(self, headers) -> int | None:
        v = headers.get("x-ratelimit-remaining-tokens")
        return int(v) if v is not None else None

    def seconds_until_reset(self, headers) -> float:
        return _parse_duration(headers.get("x-ratelimit-reset-tokens"))


class _AnthropicProvider:
    name = "anthropic"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._rate_limit_error = anthropic.RateLimitError

    def create(self, system: str, user: str, model: str, max_tokens: int):
        resp = self._client.messages.with_raw_response.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        msg = resp.parse()
        return msg.content[0].text, resp.headers

    def is_rate_limit(self, exc: Exception) -> bool:
        return isinstance(exc, self._rate_limit_error)

    def remaining_tokens(self, headers) -> int | None:
        v = headers.get("anthropic-ratelimit-tokens-remaining")
        return int(v) if v is not None else None

    def seconds_until_reset(self, headers) -> float:
        reset = headers.get("anthropic-ratelimit-tokens-reset")
        if not reset:
            return 0.0
        # Anthropic returns an RFC-3339 timestamp, not a duration.
        try:
            t = datetime.fromisoformat(reset.replace("Z", "+00:00"))
            return max(0.0, (t - datetime.now(timezone.utc)).total_seconds())
        except ValueError:
            return _parse_duration(reset)


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class LLMClient:
    """A provider-agnostic chat client that paces itself to the key's limits."""

    MAX_RETRIES = 6

    def __init__(self, provider, model: str):
        self._provider = provider
        self._model = model

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system: str, user: str,
                 model: str | None = None, max_tokens: int = 512) -> str:
        """Send one system+user turn and return the assistant's text."""
        model = model or self._model
        for attempt in range(self.MAX_RETRIES):
            try:
                text, headers = self._provider.create(system, user, model, max_tokens)
            except Exception as exc:
                # Reactive floor: never exceed the limit even if pacing is wrong.
                if self._provider.is_rate_limit(exc) and attempt < self.MAX_RETRIES - 1:
                    wait = self._retry_after(exc) or (2 ** attempt)
                    time.sleep(min(wait, 60))
                    continue
                raise
            # Proactive: if the next call might not fit, wait for the bucket.
            self._maybe_throttle(headers, max_tokens)
            return text
        raise RuntimeError(f"{self._provider.name}: rate limited after {self.MAX_RETRIES} retries")

    def _maybe_throttle(self, headers, max_tokens: int) -> None:
        remaining = self._provider.remaining_tokens(headers)
        if remaining is None:
            return
        # Leave headroom: if we can't comfortably fit another call, glide in.
        if remaining < max_tokens * 2:
            reset = self._provider.seconds_until_reset(headers)
            if reset > 0:
                time.sleep(min(reset, 60))

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        try:
            return float(exc.response.headers.get("retry-after"))
        except (AttributeError, TypeError, ValueError):
            return None


def make_client() -> LLMClient:
    """
    Build an LLMClient from the environment.

    Selection order:
      1. LLM_PROVIDER env var ('openai' | 'anthropic'), if set.
      2. Otherwise whichever API key is present (OpenAI wins if both are).
    Model defaults per provider; override with LLM_MODEL.
    """
    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if forced == "openai" or (not forced and has_openai):
        provider = _OpenAIProvider(os.environ["OPENAI_API_KEY"])
    elif forced == "anthropic" or (not forced and has_anthropic):
        provider = _AnthropicProvider(os.environ["ANTHROPIC_API_KEY"])
    else:
        raise RuntimeError(
            "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
            "(and optionally LLM_PROVIDER / LLM_MODEL)."
        )

    model = os.environ.get("LLM_MODEL", "").strip() or provider.DEFAULT_MODEL
    return LLMClient(provider, model)
