# Autonomous Closed-Loop Research Agent

An agentic system that researches a topic by browsing the web and iteratively improves a report until it passes a quality bar — evaluated by a separate LLM judge.

## Architecture

```
GitHub Action (workflow_dispatch)
         │
         ▼
  Orchestrator (main.py)
         │
   ┌─────┴─────────────────────────────────────────┐
   │           Iteration loop (max 5)               │
   │                                                │
   │   ┌──────────┐  ┌──────────┐  ┌──────────┐   │
   │   │ Agent 1  │  │ Agent 2  │  │ Agent 3  │   │
   │   │(parallel)│  │(parallel)│  │(parallel)│   │
   │   └────┬─────┘  └────┬─────┘  └────┬─────┘   │
   │        │              │              │          │
   │   web search + fetch  │              │          │
   │   ─────────────────── │              │          │
   │   write report        │              │          │
   │        │              │              │          │
   │   ┌────▼──────────────▼──────────────▼──────┐  │
   │   │         LLM Evaluator (0–10 rubric)     │  │
   │   └────────────────────┬────────────────────┘  │
   │                        │                        │
   │            score ≥ 8? ─┤                        │
   │            YES → done  │                        │
   │            NO  → pass all 3 reports + scores    │
   │                  to next iteration              │
   └─────────────────────────────────────────────────┘
         │
         ▼
  Upload best report as artifact
```

## How It Works

1. **3 parallel research agents** are launched each iteration.
2. Each agent uses **Claude with tool use** (web search + page fetch) to gather sources, then writes a structured report.
3. A separate **LLM evaluator** scores each report 0–10 on a 5-dimension rubric.
4. If any report scores **≥ 8/10**, the loop stops immediately.
5. If not, **all 3 reports + scores + feedback** are passed as context to the next iteration's agents so they can improve.
6. After **5 iterations**, the highest-scoring report is returned regardless.

## Quality Rubric (Evaluator)

| Dimension | Points |
|-----------|--------|
| Accuracy & Factual Correctness | 0–2 |
| Depth & Comprehensiveness | 0–2 |
| Structure & Clarity | 0–2 |
| Multiple Perspectives | 0–2 |
| Actionable Conclusions | 0–2 |
| **Total** | **0–10** |

## Running via GitHub Actions

1. Add `ANTHROPIC_API_KEY` to your repository secrets.
2. Go to **Actions → Autonomous Research Agent → Run workflow**.
3. Enter your research topic (e.g. `"The impact of large language models on scientific research"`).
4. The workflow uploads two artifacts:
   - `best_report_<timestamp>.md` — the winning report in Markdown
   - `run_history_<timestamp>.json` — full iteration-by-iteration history

## Running Locally

```bash
cd agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python main.py "Your research topic here" ./outputs
```

## File Structure

```
├── .github/workflows/research_agent.yml   # GitHub Action
├── agent/
│   ├── main.py           # Orchestrator: iteration loop, parallel execution
│   ├── web_researcher.py # Research agent with web search + fetch tools
│   ├── evaluator.py      # LLM judge with structured rubric
│   └── requirements.txt
└── README.md
```
