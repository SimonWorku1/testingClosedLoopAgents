"""
WNBA data pipeline: fetches player game logs from stats.nba.com via nba_api,
generates PrizePicks-style rolling-average prop lines, and caches everything to
disk so subsequent runs never re-fetch.

Lines use a 10-game rolling average (the same anchor PP uses), nudged ±0.5 to
avoid ties. The nudge is deterministic per question so resume produces the same
dataset.
"""

import hashlib
import json
import random
import time
from pathlib import Path

import pandas as pd

LEAGUE_ID = "10"          # WNBA in nba_api
SEASONS = ["2024", "2023"]  # tried in order; 2025 starts mid-May each year
TOP_N_PLAYERS = 60        # most-active players by total minutes
ROLLING_WINDOW = 10       # games averaged for the line
MIN_PRIOR_GAMES = 5       # minimum prior games before generating a line
MIN_MINUTES = 10          # ignore DNPs and garbage-time entries

# Stat types → (column or formula, display name)
PROPS = [
    ("PTS",  "Points"),
    ("REB",  "Rebounds"),
    ("AST",  "Assists"),
    ("PRA",  "Pts+Reb+Ast"),
]


# ---------------------------------------------------------------------------
# Fetching + caching
# ---------------------------------------------------------------------------

def _parse_minutes(val) -> float:
    """stats.nba.com returns MIN as '33:24' or a float — handle both."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and ":" in val:
        m, s = val.split(":", 1)
        return float(m) + float(s) / 60
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _fetch_season(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs
    time.sleep(2)  # be polite to stats.nba.com
    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        league_id_nullable=LEAGUE_ID,
        season_type_nullable="Regular Season",
        timeout=60,
    )
    df = logs.get_data_frames()[0]
    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
            "GAME_DATE", "MATCHUP", "WL", "MIN",
            "PTS", "REB", "AST", "STL", "BLK", "FG3M"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["MIN"] = df["MIN"].apply(_parse_minutes)
    df = df[df["MIN"] >= MIN_MINUTES]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)


def fetch_gamelogs(cache_dir: Path) -> pd.DataFrame:
    """Fetch WNBA player game logs (cached after the first successful fetch)."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        cache_file = cache_dir / f"wnba_gamelogs_{season}.json"
        if cache_file.exists():
            print(f"  [data] Loading cached {season} logs ({cache_file.name})")
            df = pd.read_json(cache_file, orient="records")
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            return df

        print(f"  [data] Fetching WNBA {season} game logs from stats.nba.com...")
        try:
            df = _fetch_season(season)
        except Exception as exc:
            print(f"  [data] {season} fetch failed: {exc}; trying older season")
            continue

        print(f"  [data] {len(df)} records, {df['PLAYER_NAME'].nunique()} players")
        df.to_json(cache_file, orient="records", date_format="iso")
        return df

    raise RuntimeError(
        "Could not fetch WNBA game logs for any season. "
        "Check network access or place wnba_gamelogs_YYYY.json in the data cache dir."
    )


# ---------------------------------------------------------------------------
# Prop question generation
# ---------------------------------------------------------------------------

def _stat_value(row: dict, key: str) -> float:
    if key == "PRA":
        return float(row.get("PTS", 0) or 0) + float(row.get("REB", 0) or 0) + float(row.get("AST", 0) or 0)
    return float(row.get(key, 0) or 0)


def _deterministic_nudge(question_id: str) -> float:
    """Reproducible ±0.5 nudge so resume produces the same lines."""
    h = int(hashlib.md5(question_id.encode()).hexdigest()[:8], 16)
    return -0.5 if h % 2 == 0 else 0.5


def build_prop_questions(df: pd.DataFrame) -> list[dict]:
    """
    For each player-game (after MIN_PRIOR_GAMES prior games), generate one
    prop question per stat. Strictly temporally ordered — no lookahead.

    The agent NEVER sees `actual_value` or `correct_pick`; those are
    evaluated by main.py after the agent submits its pick.
    """
    top_ids = (
        df.groupby("PLAYER_ID")["MIN"].sum()
        .sort_values(ascending=False)
        .head(TOP_N_PLAYERS)
        .index.tolist()
    )
    df = df[df["PLAYER_ID"].isin(top_ids)].copy()

    questions = []
    for pid in top_ids:
        pdf = df[df["PLAYER_ID"] == pid].sort_values("GAME_DATE").reset_index(drop=True)
        rows = pdf.to_dict("records")
        name = rows[0]["PLAYER_NAME"]
        team = rows[0]["TEAM_ABBREVIATION"]

        for stat_key, stat_name in PROPS:
            series = [_stat_value(r, stat_key) for r in rows]

            for i in range(MIN_PRIOR_GAMES, len(rows)):
                row = rows[i]
                prior = series[max(0, i - ROLLING_WINDOW): i]
                if len(prior) < MIN_PRIOR_GAMES:
                    continue

                date_str = pd.Timestamp(row["GAME_DATE"]).strftime("%Y%m%d")
                qid = f"{pid}_{stat_key}_{date_str}"
                rolling_avg = sum(prior) / len(prior)
                line = round(rolling_avg + _deterministic_nudge(qid), 1)
                actual = round(series[i], 1)

                matchup = str(row.get("MATCHUP", ""))
                home_away = "away" if "@" in matchup else "home"
                opponent = matchup.split()[-1] if matchup else "UNK"

                last3 = series[max(0, i - 3): i]
                last5 = series[max(0, i - 5): i]

                questions.append({
                    "question_id": qid,
                    "player_name": name,
                    "team": team,
                    "game_date": pd.Timestamp(row["GAME_DATE"]).strftime("%Y-%m-%d"),
                    "opponent": opponent,
                    "home_away": home_away,
                    "stat_key": stat_key,
                    "stat_name": stat_name,
                    "line": line,
                    # --- hidden from agent ---
                    "actual_value": actual,
                    "correct_pick": "over" if actual > line else "under",
                    # --- context the agent can use ---
                    "rolling_avg_10": round(rolling_avg, 1),
                    "season_avg": round(sum(series[:i]) / i, 1),
                    "last_3_avg": round(sum(last3) / len(last3), 1) if last3 else None,
                    "last_5_avg": round(sum(last5) / len(last5), 1) if last5 else None,
                    "games_played": i,
                })

    questions.sort(key=lambda q: q["game_date"])
    print(f"  [data] {len(questions)} prop questions for {len(PROPS)} stat types")
    return questions
