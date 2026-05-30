# Closed-Loop Agent Builder

Set up or update a self-improving closed-loop agent system for a given goal.

This skill encodes hard-won lessons. The "Failure modes to prevent" section below is non-negotiable — every generated project must defend against every item listed there.

## Step 1 — Gather inputs

Ask the user (in a single AskUserQuestion call) for anything not already provided:

1. **Goal** (required): What should the agents accomplish?
2. **Agents per iteration** (default 3)
3. **Max iterations** (default 5)
4. **Target score** (default 8 out of 10)

If the user already provided any of these inline, skip those questions.

## Step 2 — Detect repo state

Check whether `agent/main.py` exists.

- **Exists** → adapt the files in-place.
- **Does not exist** → scaffold the full project.

## Step 3 — Analyse the goal

Decide:

- **What does a worker need to do?** (search the web, call an API, read files, run code, etc.)
- **What tools?** Design schemas (name, description, input_schema). If web search is needed, reuse the DuckDuckGo HTML + urllib pattern. Prefer pure-Python helpers; otherwise add deps to requirements.txt.
- **What is a "good result"?** Write a strict 5-dimension rubric (see Step 4 → evaluator.py for the bar — it must be brutal).
- **What system prompt?** Tell the worker exactly what to do, the output format, and what high quality looks like.

## Step 4 — Write the files

### `agent/worker.py`

Public API:
- `build_system_prompt() -> str`
- `build_user_prompt(goal, agent_id, iteration, previous_iterations) -> str` — must include condensed summaries of previous iterations so each run improves on the last
- `TOOLS = [...]` (empty list if none needed)
- `_execute_tool(tool_name, tool_input) -> str`
- `run_worker_agent(client, goal, agent_id, iteration, previous_iterations, shutdown_event=None) -> str`

The agent loop MUST:
1. Go through the provider-agnostic `llm_client.py` (see below) — never import an SDK directly and never hardcode the model name. The model comes from the client (`LLM_MODEL` env / per-provider default).
2. Self-pace via the client's rate-limit handling — do NOT hardcode a fixed sleep between rounds. The client reads response headers and sleeps only when the token bucket is actually low (failure mode #11).
3. Always make one final no-tools call at the end with a large `max_tokens` and an instruction like "write your final output now, no more tools" so the agent always produces text — never trust that the tool loop terminated with a text block.
4. Accept `shutdown_event: threading.Event | None`; check it before each API call and return `"[Cancelled]"` if set.
5. Rely on the client's built-in `retry-after` + exponential backoff (up to 6 retries) rather than a per-call backoff helper of its own.

### `agent/llm_client.py`

Provider-agnostic LLM client (copy from the budget-optimizer example). Responsibilities:
- `make_client()` selects OpenAI or Anthropic from whichever key is present (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`); override with `LLM_PROVIDER` / `LLM_MODEL`.
- One `complete(system=, user=, max_tokens=)` method so agent code is SDK-free.
- Self-pacing: honour `retry-after` on 429s, proactively sleep when `*-remaining-tokens` is low. Handles both providers' header names and reset formats. This is the single defence for failure mode #11.

### `agent/evaluator.py`

Must contain:
- All LLM calls go through `llm_client.py` (which owns retry/backoff and pacing) — the evaluator does not need its own backoff helper.
- `RUBRIC` — a brutally strict 5-dimension rubric, 0–2 per dimension, total 0–10. The rubric MUST:
  - Open with "You reject 80% of submissions on first review. Default to the LOWER score when uncertain."
  - Define each score 0/1/2 with **specific verifiable evidence** required for 2 (counts, named sources, quantitative claims, etc.) — not adjectives like "thorough" or "well-organized".
  - Include a CRITICAL CALIBRATION block with **hard caps**: "If X is missing, the dimension cannot exceed 1." Hard caps are the single most effective lever — every previous version of this skill passed on first try until hard caps were added.
  - Forbid "AI-isms" ("delve into", "it is important to note", "in conclusion") under the Structure dimension so prose quality is enforced.
  - Require feedback to quote specific phrases from the output (4–6 sentences).
- `evaluate_<thing>(client, output, goal) -> tuple[int, str]` — returns `(score, feedback)`. Name the function specifically (e.g. `evaluate_report`, `evaluate_code`); main.py imports this name.

Evaluator output format (instruct the model to return JSON only):
```json
{"scores": {"dim1": 0-2, ...}, "total": 0-10, "feedback": "..."}
```

### `agent/main.py`

The orchestrator. Copy the existing structure. Critical requirements (see Failure Modes section for the bugs these prevent):

- `NUM_AGENTS`, `MAX_ITERATIONS`, `TARGET_SCORE` per user input.
- `AGENT_STAGGER_SECONDS = 30` (spreads token bursts).
- `_agent_worker` returns `dict | None`. On any exception during research or evaluation, log the error and **return None — do NOT set the shutdown event**. The shutdown event is for cooperative cancellation, never a poison pill.
- `run_iteration` collects results, filters out `None` returns, and only raises if **zero** agents succeeded. Partial success is success — a transient rate limit on Agent 2 must not discard Agent 1's good result.
- Fresh `threading.Event()` per iteration (no cross-iteration poisoning).
- `action_mode` writes `score` and `done` to `$GITHUB_OUTPUT`, atomically replaces history file via tmp + os.replace.
- `argparse` matches the workflow invocation exactly.

### `agent/requirements.txt`

Both SDKs so either key works at runtime: `anthropic>=0.40.0` and `openai>=1.40.0`, plus whatever the worker actually imports.

### `.github/workflows/loop_agent.yml`

Copy the existing `research_agent.yml` structure. Critical requirements:

- Inputs: `goal` (or `topic`), `iteration` (default "0"), `run_id` (default ""), `prev_run_id` (default ""), `max_iterations` (default "5").
- `permissions: actions: write` (needed to self-dispatch AND to read cross-run artifacts).
- Download step MUST include all of:
  - `if: ${{ github.event.inputs.iteration != '0' && github.event.inputs.prev_run_id != '' }}`
  - `continue-on-error: true`
  - `run-id: ${{ github.event.inputs.prev_run_id }}`
  - `github-token: ${{ secrets.GITHUB_TOKEN }}`
  - These four together let manual mid-chain dispatches work and tolerate transient artifact lookup failures. `download-artifact@v4` only searches the current run by default — `run-id` + `github-token` are mandatory for cross-run download.
- Upload step uses `if: always()` and the same artifact name `<workflow>-history-${{ github.event.inputs.run_id || github.run_id }}` so iteration 0 establishes the shared key.
- Trigger-next step:
  - `if: steps.research.outputs.done == 'false' && steps.research.outcome == 'success'` — the outcome guard is mandatory; without it any Python crash silently keeps chaining.
  - `parseInt(maxIterations) || 5` — fallback for NaN/empty input, else `nextIteration >= NaN` is always false and the cap never fires.
  - Passes `prev_run_id: '${{ github.run_id }}'` and `run_id: <preserved>` to the next iteration.

### If adapting an existing project

- If `worker.py` already exists with a different name (e.g. `web_researcher.py`), either rename it or replace it. Don't leave both — main.py imports must be unambiguous.
- Re-check the rubric against the current goal; an old rubric for a different goal is worse than no rubric.

## Step 5 — Failure modes to prevent (READ THIS, every time)

Every one of these has bitten this project. The generated code must defend against each.

1. **Shutdown poisoning.** One agent's `RateLimitError` cancels every sibling and the iteration crashes with successful results discarded. → Per-agent exceptions return `None` and never call `shutdown.set()`. `run_iteration` keeps partial results.

2. **First-try pass.** A "strict" rubric without hard caps still hands out 8/10 on the first iteration because the model rewards verbosity. → Rubric MUST have explicit hard caps ("missing inline citations → max 1 on Accuracy regardless of other quality").

3. **Cross-run artifact download.** `download-artifact@v4` defaults to the current run. Iteration 1 looks for iteration 0's artifact and finds nothing. → `run-id` + `github-token` + `prev_run_id` input + propagate it via the trigger script.

4. **Infinite chain on crash.** Default `if: outputs.done == 'false'` treats a crashed step (empty outputs) the same as score-below-target. → Add `&& steps.research.outcome == 'success'`.

5. **Max-iterations bypass.** Manual dispatch with blank `max_iterations` → `parseInt('') === NaN` → `nextIteration >= NaN` is false forever. → `|| 5` fallback.

6. **Mid-chain manual dispatch fails hard.** User re-runs iteration 3 without filling in `prev_run_id` and the whole step explodes. → `if:` requires `prev_run_id != ''`, plus `continue-on-error: true` for transient artifact failures.

7. **Worker exits without text.** Tool-use loop hits `max_tokens` mid-tool, never emits a text block, evaluator gets empty output → 0/10. → Always make one final no-tools call after the tool loop.

8. **Stale shutdown event.** Reusing a `threading.Event` across iterations means a previous failure permanently cancels future runs. → Construct a fresh `Event()` inside `run_iteration`.

9. **History file write torn by crash.** Direct `write_text` on the history file leaves a half-written JSON if the process is killed between iterations. → Write to `.tmp` then `os.replace`.

10. **Workflow input name mismatch.** Skill says "rename `topic` to `goal`" but the trigger script's `createWorkflowDispatch` inputs still pass `topic`. → After renaming, grep the whole workflow file for the old name.

11. **Hardcoded provider / hardcoded rate limit.**
    - *Symptom:* code authenticates fine in dev but fails at exam/eval time with `AuthenticationError`, or runs into `429`s and dies because it assumed a tier it doesn't have. The key handed to you at runtime may be a *different provider* (OpenAI vs Anthropic) and an *unknown tier*.
    - *Root cause:* the agent imports one SDK directly (`anthropic.Anthropic(...)`) and/or sleeps a fixed number of seconds between calls.
    - *Fix:* put a provider-agnostic `llm_client.py` between the agent and any SDK. Select the provider at runtime from whichever key is present (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, override via `LLM_PROVIDER`/`LLM_MODEL`). Expose one `complete(system=, user=)` method. Self-pace off the response's rate-limit headers — honour `retry-after` on 429s (reactive floor) **and** proactively sleep when `*-remaining-tokens` is low until `*-reset`. Header names differ per provider (`x-ratelimit-*` vs `anthropic-ratelimit-*`) and reset formats differ (OpenAI duration `"6m0s"` vs Anthropic RFC-3339 timestamp) — parse both. The workflow forwards *both* keys so the same code runs on either. Never hardcode a sleep interval; the headers tell you the real budget.

## Step 6 — Verify consistency

Before committing, check:
- `main.py` import names match `worker.py` and `evaluator.py` exports.
- Workflow `python main.py` invocation matches `argparse` in main.py.
- All five workflow input names appear consistently in: `on.workflow_dispatch.inputs`, every `github.event.inputs.X` reference, and the `createWorkflowDispatch` inputs object.
- `requirements.txt` covers every import.
- Re-read the rubric. If a "well-written but generic" output would score above 6, tighten it.

## Step 7 — Commit and push

Single commit on the current branch with a message naming the goal and configuration.

## Step 8 — Tell the user

- Scaffold vs adapt mode.
- What tools the worker has.
- Hardest rubric dimension and what separates a 2 from a 1.
- How to trigger: Actions → workflow → Run workflow → enter the goal.
- One-line note: "If a manual mid-chain dispatch is needed, fill in both `run_id` (the run ID of iteration 0) and `prev_run_id` (the run ID of the previous iteration); otherwise the new run starts without history."
