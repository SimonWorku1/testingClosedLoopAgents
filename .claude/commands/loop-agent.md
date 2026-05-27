# Closed-Loop Agent Builder

Set up or update a self-improving closed-loop agent system for a given goal.

## Step 1 — Gather inputs

Ask the user (in a single AskUserQuestion call with up to 4 questions) for anything not already provided:

1. **Goal** (required): What should the agents accomplish? (e.g. "research a topic and write a report", "generate and refine a Python script that solves X", "summarise and critique a set of documents")
2. **Agents per iteration** (default 3): How many agents run in parallel each iteration?
3. **Max iterations** (default 5): Maximum improvement cycles before stopping.
4. **Target score** (default 8): Score out of 10 at which the loop stops early.

If the user already provided any of these in the invocation message, skip asking for them.

## Step 2 — Detect repo state

Check whether `agent/main.py` exists in the current working directory.

- **Exists** → this is an existing closed-loop agent project. You will **adapt** the files in-place.
- **Does not exist** → this is a fresh repo. You will **scaffold** the full project.

## Step 3 — Analyse the goal

Before writing any code, think carefully about what the goal requires:

- **What does a worker agent need to do?** (search the web, call an API, read files, generate code, call a database, etc.)
- **What tools does it need?** Design the tool schemas (name, description, input_schema) for any tools required. If web search is needed, use the DuckDuckGo HTML + urllib pattern from the existing codebase. If other resources are needed, implement them as pure-Python helpers with no extra pip dependencies where possible; otherwise add them to requirements.txt.
- **What is a "good result"?** Translate the goal into a 5-dimension rubric (0–2 each, total 0–10) where each dimension is specific, measurable, and hard to game. Each score of 2 must require genuinely exceptional work. A score of 8+ should take multiple iterations to achieve.
- **What system prompt should the worker use?** Write a system prompt that tells the worker exactly what to do, what format to return the result in, and what constitutes a high-quality output.

## Step 4 — Write the files

### Always write these files (create or overwrite):

#### `agent/worker.py`
Implements the agent that performs the task. Pattern:
- `build_system_prompt() -> str` — the worker's system prompt tailored to the goal
- `build_user_prompt(goal, agent_id, iteration, previous_iterations) -> str` — includes previous results as context so each iteration improves on the last
- `TOOLS = [...]` — list of tool dicts the worker can call (empty list if no tools needed)
- `_execute_tool(tool_name, tool_input) -> str` — dispatches tool calls
- `run_worker_agent(client, goal, agent_id, iteration, previous_iterations, shutdown_event=None) -> str` — the main entry point; returns the agent's output as a string

The agent loop must:
1. Use `model="claude-sonnet-4-6"` and `max_tokens=4096` for tool-use rounds
2. Sleep 12 seconds between tool-use rounds (rate limit guard)
3. After the tool loop ends (any reason), make one final no-tools call with `max_tokens=8192` and "write your final output now, no more tools" instruction so the agent always produces text
4. Support a `shutdown_event: threading.Event` parameter; check it before each API call and return `"[Cancelled]"` if set
5. Use `_api_call_with_backoff` (copy the pattern from evaluator.py) for all API calls

#### `agent/evaluator.py`
Implements the LLM evaluator. Must contain:
- `RUBRIC` — a strict, goal-specific rubric. Write the rubric so that:
  - Score 2 on each dimension requires specific, verifiable evidence — not vague claims
  - Score 9–10 should be rare; explicitly instruct the evaluator to be harsh
  - Feedback must name specific gaps (3–4 sentences)
- `evaluate_output(client, output, goal) -> tuple[int, str]` — returns `(score, feedback)`

The evaluator must ask the model to return JSON only:
```json
{
  "scores": {"dim1": 0-2, "dim2": 0-2, ...},
  "total": 0-10,
  "feedback": "..."
}
```

#### `agent/main.py`
Orchestrator. Copy the structure from the existing `main.py` exactly, changing:
- Import `run_worker_agent` from `worker` instead of `web_researcher`
- Import `evaluate_output` from `evaluator`
- Set `NUM_AGENTS`, `MAX_ITERATIONS`, `TARGET_SCORE` to the user's chosen values
- Keep `AGENT_STAGGER_SECONDS = 30`
- Keep all the action_mode / local_mode / _agent_worker / run_iteration logic intact
- Update the report filename and content to reflect the goal

#### `agent/requirements.txt`
`anthropic>=0.40.0` plus any additional packages the worker needs.

#### `.github/workflows/loop_agent.yml`
Copy the structure from the existing `research_agent.yml` exactly, changing:
- `workflow_id` in the self-trigger script to `loop_agent.yml`
- The `topic` input renamed to `goal` with an appropriate description
- Job name updated to reflect the goal type
- The `python main.py` call to pass `goal` instead of `topic`

### If adapting an existing project:
- Also delete or overwrite `agent/web_researcher.py` if the new goal no longer needs web search (replace its functionality in `worker.py`).
- If `agent/web_researcher.py` is still needed, keep it and have `worker.py` import from it.

## Step 5 — Verify consistency

Before committing, check:
- `main.py` imports match what `worker.py` and `evaluator.py` export
- The `evaluate_output` function signature matches how `main.py` calls it
- The workflow `python main.py` invocation passes the right `--goal` or `--topic` flag (update argparse in main.py if needed)
- `requirements.txt` includes every import used across all agent files

## Step 6 — Commit and push

Stage all changed files and commit with a message describing the goal and configuration. Push to the current branch.

## Step 7 — Tell the user

Report:
- What mode was used (scaffold vs adapt)
- What tools the worker has access to
- A one-paragraph summary of the rubric (what makes a 2 vs a 1 on the hardest dimension)
- How to trigger the workflow: go to Actions → loop_agent.yml → Run workflow → enter the goal
