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
import statistics
import time
from pathlib import Path

import pandas as pd

LEAGUE_ID = "10"          # WNBA in nba_api
# All five seasons used for reliability analysis; the picks loop uses only the
# most-recent cached season so the training set stays temporally coherent.
SEASONS = ["2026", "2025", "2024", "2023", "2022"]
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


def fetch_all_seasons(cache_dir: Path) -> pd.DataFrame:
    """
    Fetch and concatenate all SEASONS into one DataFrame for multi-year analysis
    (reliability scoring, player discovery). Each season is cached individually.
    Seasons that fail to fetch are skipped with a warning.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in SEASONS:
        cache_file = cache_dir / f"wnba_gamelogs_{season}.json"
        if cache_file.exists():
            print(f"  [data] Loading cached {season} logs")
            df = pd.read_json(cache_file, orient="records")
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            frames.append(df)
            continue
        print(f"  [data] Fetching WNBA {season} game logs from stats.nba.com...")
        try:
            df = _fetch_season(season)
        except Exception as exc:
            print(f"  [data] {season} fetch failed: {exc} — skipping")
            continue
        print(f"  [data] {len(df)} records, {df['PLAYER_NAME'].nunique()} players")
        df.to_json(cache_file, orient="records", date_format="iso")
        frames.append(df)

    if not frames:
        raise RuntimeError(
            "Could not fetch WNBA game logs for any season. "
            "Check network access or place wnba_gamelogs_YYYY.json in the cache dir."
        )
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["PLAYER_ID", "GAME_DATE"]
    ).reset_index(drop=True)
    print(f"  [data] Multi-season dataset: {len(combined)} records across "
          f"{combined['PLAYER_NAME'].nunique()} players")
    return combined


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


# Default small roster: high-volume, low-variance stars (most predictable props).
# Matched case-insensitively as substrings against PLAYER_NAME.
DEFAULT_ROSTER = ["Wilson", "Stewart", "Collier"]


def build_prop_questions(df: pd.DataFrame,
                         players: list[str] | None = None,
                         stats: list[str] | None = None) -> list[dict]:
    """
    For each player-game (after MIN_PRIOR_GAMES prior games), generate one
    prop question per stat. Strictly temporally ordered — no lookahead.

    `players`: list of name substrings to focus on. None → DEFAULT_ROSTER.
               Pass ["*"] for the top-N-by-minutes broad set.
    `stats`: list of stat keys to include e.g. ["AST"] or ["PTS","PRA"].
             None → all PROPS.

    The agent NEVER sees `actual_value` or `correct_pick`; those are
    evaluated by main.py after the agent submits its pick.
    """
    if players is None:
        players = DEFAULT_ROSTER
    if players == ["*"]:
        players = None

    if players:
        names = df["PLAYER_NAME"]
        mask = pd.Series(False, index=df.index)
        for sub in players:
            mask |= names.str.contains(sub, case=False, na=False)
        matched = df[mask]["PLAYER_NAME"].unique().tolist()
        if not matched:
            raise RuntimeError(
                f"No players matched {players}. Available example names: "
                f"{sorted(df['PLAYER_NAME'].unique())[:10]}"
            )
        print(f"  [data] Roster focus: {matched}")
        top_ids = df[mask]["PLAYER_ID"].unique().tolist()
    else:
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

        active_props = [(k, n) for k, n in PROPS if stats is None or k in stats]
        for stat_key, stat_name in active_props:
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

                # Volatility from PRIOR games only (no lookahead): how erratic is
                # this player on this stat? High volatility => the line is a poor
                # predictor => the agent should lean toward passing.
                volatility = statistics.stdev(prior) if len(prior) >= 2 else 0.0
                cv = (volatility / rolling_avg) if rolling_avg > 0 else 0.0
                # Streak: how many of the last 3 prior games were over the line?
                overs_last3 = sum(1 for v in last3 if v > line)

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
                    "volatility": round(volatility, 1),       # std dev of prior games
                    "cv": round(cv, 2),                        # coefficient of variation
                    "overs_last3": overs_last3,                # of last 3, how many over line
                    "games_played": i,
                })

    questions.sort(key=lambda q: q["game_date"])
    stat_labels = stats if stats else [k for k, _ in PROPS]
    print(f"  [data] {len(questions)} prop questions for {stat_labels}")
    return questions
