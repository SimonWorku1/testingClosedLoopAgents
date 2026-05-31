"""
LLM-based WNBA prop picks agent (high-conviction / cherry-pick mode).

Receives a batch of player-prop questions (without answers) plus its history of
COMMITTED picks. For each prop it may pick "over", "under", or "pass". Only
high-confidence picks count toward the success window, so the agent is rewarded
for betting selectively — exactly how a real sharp operates on PrizePicks.
"""

import json
import re

from llm_client import LLMClient

WINDOW_SIZE = 20
COMMIT_CONFIDENCE = 4   # min confidence (1-5) for a pick to count as committed
TARGET_CORRECT = 16     # 16/20 committed = success (≤4 mistakes)

_SYSTEM_PROMPT = f"""You are a WNBA sports analyst running a HIGH-CONVICTION prop-picks model.

YOUR EDGE COMES FROM SELECTIVITY. You do not have to pick every prop.
For each prop you may answer "over", "under", or "pass".

SCORING:
- Only picks with confidence {COMMIT_CONFIDENCE} or 5 are COMMITTED (they count).
- Picks with confidence 1-3, or "pass", are skipped and do NOT count.
- Success = {TARGET_CORRECT} or more correct out of any {WINDOW_SIZE} COMMITTED picks.
- A wrong committed pick hurts you. A pass costs nothing. PASS when unsure.

THE LINES EXPLAINED:
Each line is the player's own 10-game rolling average, nudged ±0.5 to break ties.
Because the line IS the recent average, a random game is ~50/50 to go over/under.
You CANNOT beat this by guessing — you must only commit when a specific, strong
signal tips the odds clearly in one direction. Most props should be a PASS.

SIGNALS THAT JUSTIFY COMMITTING (need at least one STRONG one):

1. MEAN REVERSION (often the strongest learnable edge):
   - If overs_last3 = 3 (player went OVER the line all 3 recent games) AND
     last_3_avg is well above the line, they are "hot" and DUE TO REGRESS →
     consider committing UNDER.
   - If overs_last3 = 0 (went UNDER all 3) AND last_3_avg is well below the
     line, they are "cold" and due to bounce back → consider committing OVER.
   - Watch your own committed history: if your reversion bets keep missing for
     a player, that player is trending (not reverting) — adjust.

2. Recent form divergence:
   - last_3_avg >= 3.0 ABOVE the line AND last_5 agrees → trend OVER
   - last_3_avg >= 3.0 BELOW the line AND last_5 agrees → trend UNDER
   - Reversion vs. trend conflict? When unsure which dominates → PASS.

3. Volatility filter (use the `volatility` / `cv` fields):
   - LOW volatility (cv < 0.25): the line is a reliable anchor → safe to commit
     when a form/reversion signal points clearly one way.
   - HIGH volatility (cv > 0.45): the player is erratic, the line means little
     → PASS unless the signal is extreme.

4. Stat stability:
   - PRA (Pts+Reb+Ast) combines 3 stats → lower variance → more trustworthy
   - Points for a high-usage star → more predictable than role players

5. Sample maturity:
   - games_played >= 15 → rolling average is reliable, trust the signals
   - games_played < 10 → small sample, lines are volatile → lean PASS

6. Context (tie-breakers, never commit on these alone):
   - Home games slightly favor stars; away back-to-backs dampen output

DISCIPLINE:
- It is correct to pass on most of a batch. Quality over quantity.
- Only use confidence {COMMIT_CONFIDENCE}-5 when a clear form signal is present.
- Study your recent mistakes below and stop committing to similar weak spots.

Respond with ONLY a JSON array — one object per question, same order:
[{{"question_id": "...", "pick": "over"|"under"|"pass", "confidence": 1-5, "reasoning": "<one sentence>"}}]"""


def _build_history_summary(committed_history: list[dict]) -> str:
    if not committed_history:
        return "No committed picks yet — this is the first batch. Be selective."

    total = len(committed_history)
    correct = sum(1 for p in committed_history if p["correct"])
    recent = committed_history[-WINDOW_SIZE:]
    recent_correct = sum(1 for p in recent if p["correct"])

    lines = [
        f"Committed picks: {correct}/{total} correct ({100*correct//total}%)",
        f"Last {len(recent)} committed: {recent_correct}/{len(recent)} | "
        f"Need {TARGET_CORRECT}/{WINDOW_SIZE} in a window to succeed",
    ]

    mistakes = [p for p in committed_history[-30:] if not p["correct"]][-4:]
    if mistakes:
        lines.append("\nRecent COMMITTED mistakes — find the pattern and avoid it:")
        for m in mistakes:
            hint = ""
            if m.get("last_3_avg") is not None:
                diff = m["last_3_avg"] - m["line"]
                hint = f" [last-3 was {m['last_3_avg']}, {diff:+.1f} vs line]"
            lines.append(
                f"  ✗ {m['player_name']} {m['stat_name']} line {m['line']} | "
                f"committed {m['pick'].upper()}{hint} | "
                f"actual {m['actual']:.1f} → {m['correct_pick'].upper()} was right"
            )

    wins = [p for p in committed_history[-20:] if p["correct"]][-3:]
    if wins:
        lines.append("\nRecent committed WINS — keep doing this:")
        for w in wins:
            hint = ""
            if w.get("last_3_avg") is not None:
                diff = w["last_3_avg"] - w["line"]
                hint = f" [last-3 {diff:+.1f} vs line]"
            lines.append(
                f"  ✓ {w['player_name']} {w['stat_name']} line {w['line']} | "
                f"{w['pick'].upper()}{hint} | actual {w['actual']:.1f}"
            )

    return "\n".join(lines)


def make_picks(
    client: LLMClient,
    questions: list[dict],
    committed_history: list[dict],
) -> list[dict]:
    """
    Ask the model to make over/under/pass picks for each question.
    Returns a list of dicts: {question_id, pick, confidence, reasoning}.
    """
    history_text = _build_history_summary(committed_history)

    questions_text = "\n\n".join(
        f"Prop {j + 1} | ID: {q['question_id']}\n"
        f"  {q['player_name']} ({q['team']})  |  {q['home_away']} vs {q['opponent']}  |  {q['game_date']}\n"
        f"  Stat: {q['stat_name']}  |  Line: {q['line']}  |  games played: {q.get('games_played', '?')}\n"
        f"  Rolling avg (10): {q['rolling_avg_10']}  |  Season avg: {q['season_avg']}\n"
        f"  Last 3 avg: {q.get('last_3_avg', 'N/A')}  |  Last 5 avg: {q.get('last_5_avg', 'N/A')}\n"
        f"  Volatility: {q.get('volatility', 'N/A')} (cv {q.get('cv', 'N/A')})  |  "
        f"overs in last 3: {q.get('overs_last3', 'N/A')}/3"
        for j, q in enumerate(questions)
    )

    user_message = (
        f"--- YOUR COMMITTED-PICK HISTORY ---\n{history_text}\n\n"
        f"--- EVALUATE THESE {len(questions)} PROPS (pass freely) ---\n\n"
        f"{questions_text}\n\n"
        "JSON array only."
    )

    raw = client.complete(
        system=_SYSTEM_PROMPT,
        user=user_message,
        max_tokens=1024,
    ).strip()

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

    by_id = {p.get("question_id"): p for p in parsed if p.get("question_id")}
    result = []
    for q in questions:
        p = by_id.get(q["question_id"], {})
        pick = str(p.get("pick", "pass")).lower().strip()
        if pick not in ("over", "under", "pass"):
            pick = "pass"
        result.append({
            "question_id": q["question_id"],
            "pick": pick,
            "confidence": max(1, min(5, int(p.get("confidence", 1)))),
            "reasoning": str(p.get("reasoning", "")).strip() or "(no reasoning)",
        })

    return result
