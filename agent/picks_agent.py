"""
LLM-based WNBA prop picks agent.

Receives a batch of player-prop questions (without answers) and its full
pick history, then makes over/under predictions. Between batches it sees
which picks it got wrong, allowing it to refine its strategy across iterations.
"""

import json
import re

from llm_client import LLMClient

WINDOW_SIZE = 20
TARGET_CORRECT = 19  # 19/20 = success

_SYSTEM_PROMPT = f"""You are a WNBA sports analyst building a prop-picks model.

TARGET: achieve {TARGET_CORRECT} or more correct in any {WINDOW_SIZE} consecutive picks.

THE LINES EXPLAINED:
Each line is the 10-game rolling average for that player+stat, nudged ±0.5 to avoid ties.
Because the line IS the rolling average, you cannot just always pick "over" — you need to
identify when a player will beat or miss their own recent average.

FACTORS THAT PREDICT DEVIATION FROM THE ROLLING AVERAGE:

1. Recent form vs line:
   - last_3_avg >> line  (>+2.0): momentum is real → lean OVER
   - last_3_avg << line  (<-2.0): cold streak → lean UNDER
   - last_3_avg ≈ line: toss-up, look at other factors

2. Home vs away:
   - Stars often perform slightly better at home (energy, comfort)
   - Away back-to-backs dampen output

3. Stat stability by type:
   - PRA (combined) is more stable than raw blocks or steals
   - Points for top scorers → more predictable than role players

4. Opponent signal:
   - Weak defensive teams → lean OVER on points/PRA for stars
   - Elite defenses → lean UNDER even for stars

5. Season trajectory:
   - First 10 games: lines are volatile (small prior sample)
   - After 30+ games: rolling avg is reliable; trust the form signal

WHAT GOES WRONG (common mistakes to avoid):
- Chasing hot streaks when the line already reflects them
- Over-trusting season_avg when recent form diverges sharply
- Picking low-confidence spots (confidence 1-2) when better options exist
- Ignoring that PRA is 3 stats combined — volatility is lower

Respond with ONLY a JSON array — one object per question in the same order:
[{{"question_id": "...", "pick": "over" or "under", "confidence": 1-5, "reasoning": "<one sentence>"}}]"""


def _build_history_summary(pick_history: list[dict]) -> str:
    if not pick_history:
        return "No pick history yet — this is the first batch."

    total = len(pick_history)
    correct = sum(1 for p in pick_history if p["correct"])
    recent = pick_history[-WINDOW_SIZE:] if len(pick_history) >= WINDOW_SIZE else pick_history
    recent_correct = sum(1 for p in recent if p["correct"])

    lines = [
        f"Overall: {correct}/{total} correct ({100*correct//total}%)",
        f"Last {len(recent)}: {recent_correct}/{len(recent)} | "
        f"Need {TARGET_CORRECT}/{WINDOW_SIZE} in any window to succeed",
    ]

    # Show the last 3 mistakes with full context so the agent can learn
    mistakes = [p for p in pick_history[-30:] if not p["correct"]][-3:]
    if mistakes:
        lines.append("\nMost recent mistakes (study these):")
        for m in mistakes:
            hint = ""
            if m.get("last_3_avg") is not None:
                diff = m["last_3_avg"] - m["line"]
                hint = f" (last-3 avg was {m['last_3_avg']}, {diff:+.1f} vs line)"
            lines.append(
                f"  ✗ {m['player_name']} {m['stat_name']} line {m['line']} | "
                f"picked {m['pick'].upper()}{hint} | "
                f"actual {m['actual']:.1f} → correct was {m['correct_pick'].upper()}"
            )

    # Also show last 3 correct picks for calibration
    wins = [p for p in pick_history[-20:] if p["correct"]][-3:]
    if wins:
        lines.append("\nRecent correct picks (what worked):")
        for w in wins:
            lines.append(
                f"  ✓ {w['player_name']} {w['stat_name']} line {w['line']} | "
                f"picked {w['pick'].upper()} | actual {w['actual']:.1f}"
            )

    return "\n".join(lines)


def make_picks(
    client: LLMClient,
    questions: list[dict],
    pick_history: list[dict],
) -> list[dict]:
    """
    Ask the model to make over/under picks for each question.
    Returns a list of dicts: {question_id, pick, confidence, reasoning}.
    """
    history_text = _build_history_summary(pick_history)

    questions_text = "\n\n".join(
        f"Pick {j + 1} | ID: {q['question_id']}\n"
        f"  {q['player_name']} ({q['team']})  |  {q['home_away']} vs {q['opponent']}  |  {q['game_date']}\n"
        f"  Stat: {q['stat_name']}  |  Line: {q['line']}\n"
        f"  Rolling avg (last 10): {q['rolling_avg_10']}  |  Season avg: {q['season_avg']}\n"
        f"  Last 3 avg: {q.get('last_3_avg', 'N/A')}  |  Last 5 avg: {q.get('last_5_avg', 'N/A')}"
        for j, q in enumerate(questions)
    )

    user_message = (
        f"--- YOUR PICK HISTORY ---\n{history_text}\n\n"
        f"--- MAKE PICKS FOR THESE {len(questions)} PROPS ---\n\n"
        f"{questions_text}\n\n"
        "JSON array only."
    )

    raw = client.complete(
        system=_SYSTEM_PROMPT,
        user=user_message,
        max_tokens=1024,
    ).strip()

    # Parse the response
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    raw = re.sub(r":\s*NaN\b", ": 0", raw)
    raw = re.sub(r":\s*Infinity\b", ": 0", raw)

    parsed = json.loads(raw)

    # Build result, one entry per question in original order
    by_id = {p.get("question_id"): p for p in parsed if p.get("question_id")}
    result = []
    for q in questions:
        p = by_id.get(q["question_id"], {})
        pick = str(p.get("pick", "over")).lower().strip()
        if pick not in ("over", "under"):
            pick = "over"
        result.append({
            "question_id": q["question_id"],
            "pick": pick,
            "confidence": max(1, min(5, int(p.get("confidence", 3)))),
            "reasoning": str(p.get("reasoning", "")).strip() or "(no reasoning)",
        })

    return result
