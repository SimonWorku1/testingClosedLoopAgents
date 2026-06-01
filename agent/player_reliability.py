"""
Player reliability scorer — finds WNBA players whose stat lines are historically
predictable, making them the best targets for prop picks.

Metrics computed per (player, stat) from all available history:
  - cv          : coefficient of variation (std / mean) — lower = more stable
  - hit_rate    : how often the 10-game rolling-avg line predicts the outcome
  - reversion   : after overs_last3=3, P(next < line); after =0, P(next > line)
  - autocorr    : lag-1 autocorrelation of the stat series (negative = reverter)
  - n_games     : sample size (more = more trustworthy)

A composite reliability_score (0–1) weighs all five. Players are ranked across
all stat types, then the top N are returned with their best stat and name.

Typical use:
    df = fetch_gamelogs(data_dir)          # uses SEASONS (all 5 years)
    top = find_reliable_players(df, top_n=3)
    for p in top:
        print(p["player_name"], p["best_stat"], p["reliability_score"])
"""

import statistics
from collections import defaultdict

# Minimum games required before a player-stat pair is scored.
MIN_GAMES = 20


def _lag1_autocorr(series: list[float]) -> float | None:
    """Pearson autocorrelation at lag 1. Returns None if underdetermined."""
    if len(series) < 4:
        return None
    x = series[:-1]
    y = series[1:]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = (
        (sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)) ** 0.5
    )
    return num / denom if denom > 0 else None


def _score_player_stat(values: list[float], rolling_window: int = 10,
                       min_games: int = MIN_GAMES) -> dict | None:
    """
    Compute reliability metrics for a single (player, stat) time series.
    `values` must be in chronological order.
    Returns None if there are too few games.
    """
    if len(values) < min_games:
        return None

    mean = statistics.mean(values)
    if mean <= 0:
        return None

    cv = statistics.stdev(values) / mean

    # Rolling-window line prediction: line = mean of prior `rolling_window` games.
    # hit_rate: when the rolling avg is hot (> mean) predict under; when cold
    # (< mean) predict over. Measures how reliably this player mean-reverts.
    # Baseline is 0.5 (random); a reliable reverter exceeds 0.52+.
    hits = total = 0
    over_streaks_correct = under_streaks_correct = streak_total = 0
    stdev = statistics.stdev(values)

    for i in range(rolling_window, len(values)):
        prior = values[i - rolling_window: i]
        line = sum(prior) / len(prior)
        actual = values[i]
        over = actual > line

        # Only count when the line has deviated meaningfully from the mean
        # (within 0.5σ it's too noisy to signal direction).
        deviation = abs(line - mean)
        if deviation >= 0.5 * stdev and stdev > 0:
            predicted_over = line < mean  # regression-to-mean prediction
            hits += 1 if (predicted_over == over) else 0
            total += 1

        # Reversion: after 3 consecutive overs → bet under; after 3 consecutive
        # unders → bet over. How often does this actually pay off?
        last3 = values[max(0, i - 3): i]
        overs_last3 = sum(1 for v in last3 if v > line)
        if overs_last3 == 3:
            streak_total += 1
            over_streaks_correct += 0 if over else 1  # correct if under (reversion)
        elif overs_last3 == 0:
            streak_total += 1
            under_streaks_correct += 1 if over else 0  # correct if over (reversion)

    hit_rate = hits / total if total > 0 else 0.5
    reversion_rate = (
        (over_streaks_correct + under_streaks_correct) / streak_total
        if streak_total >= 5
        else None
    )
    autocorr = _lag1_autocorr(values)

    # --- composite score (0-1, higher = more reliable / predictable) ----------
    # cv:          0.1 → great, 0.5 → terrible; scale to 0-1 inverted
    cv_score = max(0.0, 1.0 - cv / 0.6)

    # hit_rate:    0.5 = baseline; we want >0.52
    hr_score = min(1.0, max(0.0, (hit_rate - 0.48) / 0.12))

    # reversion:   None = no data; >0.55 is useful
    rev_score = (
        min(1.0, max(0.0, (reversion_rate - 0.48) / 0.14))
        if reversion_rate is not None
        else 0.5  # neutral if not enough streak samples
    )

    # autocorr:    negative = reverter (predictable direction) → good
    # scale −1..+1 → 1..0
    ac_score = (1.0 - (autocorr or 0.0)) / 2.0

    # sample size bonus: asymptotes at 1.0 around 80 games
    sample_score = min(1.0, len(values) / 80)

    reliability_score = round(
        0.30 * cv_score
        + 0.25 * hr_score
        + 0.20 * rev_score
        + 0.15 * ac_score
        + 0.10 * sample_score,
        4,
    )

    player_type = "random"
    if autocorr is not None:
        if autocorr < -0.10:
            player_type = "reverter"
        elif autocorr > 0.10:
            player_type = "streaker"

    # --- next-game prediction -------------------------------------------
    # The line PrizePicks posts ≈ rolling_avg_10 ± 0.5.
    # We predict the player's actual output by applying a form adjustment:
    #   reverter: pull last-3 deviation back toward the rolling avg
    #   streaker: nudge slightly in the direction of recent form
    #   random  : prediction = rolling avg (no edge)
    #
    # Reversion strength is calibrated to the player's historical rate:
    #   rate 0.50 → no adjustment; rate 0.70 → pull back 40% of the deviation
    recent_window = values[-rolling_window:]
    rolling_avg = sum(recent_window) / len(recent_window)
    last3 = values[-3:]
    last3_avg = sum(last3) / len(last3)
    form_diff = last3_avg - rolling_avg  # positive = running hot

    if player_type == "reverter":
        rev = reversion_rate if reversion_rate is not None else 0.55
        pull = (rev - 0.50) * 2          # 0.70 rate → 0.40 pull strength
        predicted = rolling_avg - form_diff * pull
    elif player_type == "streaker":
        predicted = rolling_avg + form_diff * 0.15  # slight continuation
    else:
        predicted = rolling_avg

    predicted = round(max(0.0, predicted), 1)
    # The PrizePicks line is rolling_avg ± 0.5; we compare our prediction
    # against that to advise over or under.
    approx_line = round(rolling_avg, 1)
    implied_pick = "over" if predicted > approx_line else "under"

    return {
        "n_games": len(values),
        "mean": round(mean, 2),
        "cv": round(cv, 3),
        "hit_rate": round(hit_rate, 3),
        "reversion_rate": round(reversion_rate, 3) if reversion_rate is not None else None,
        "autocorr": round(autocorr, 3) if autocorr is not None else None,
        "player_type": player_type,
        "reliability_score": reliability_score,
        "rolling_avg": round(rolling_avg, 1),
        "last3_avg": round(last3_avg, 1),
        "predicted": predicted,
        "approx_line": approx_line,
        "implied_pick": implied_pick,
    }


def _stat_values(rows: list[dict], stat_key: str) -> list[float]:
    if stat_key == "PRA":
        return [
            float(r.get("PTS", 0) or 0)
            + float(r.get("REB", 0) or 0)
            + float(r.get("AST", 0) or 0)
            for r in rows
        ]
    return [float(r.get(stat_key, 0) or 0) for r in rows]


MIN_VS_TEAM_GAMES = 8   # minimum matchups to score a player vs a specific opponent


def _opponent_from_matchup(matchup: str) -> str:
    """Extract opponent abbreviation from 'SEA @ LAS' or 'SEA vs. LAS'."""
    parts = str(matchup).split()
    return parts[-1] if parts else "UNK"


def score_all_players(df, stat_keys=("PTS", "REB", "AST", "PRA")) -> list[dict]:
    """
    Score every (player, stat, context) triple in the DataFrame, where context
    is either "all" (career-wide) or a specific opponent abbreviation.

    For each player-stat pair we produce:
      - one "all" entry using the full chronological series
      - one entry per opponent where the player has MIN_VS_TEAM_GAMES+ matchups,
        using only that subset (chronological within the subset)

    Returns a flat list sorted descending by reliability_score. The best version
    of a player-stat (whether overall or vs. a specific team) rises to the top.
    """
    results = []
    for pid, group in df.groupby("PLAYER_ID"):
        group = group.sort_values("GAME_DATE")
        rows = group.to_dict("records")
        name = rows[0]["PLAYER_NAME"]

        for stat_key in stat_keys:
            # --- career-wide entry ---
            all_values = _stat_values(rows, stat_key)
            metrics = _score_player_stat(all_values)
            if metrics is not None:
                results.append({
                    "player_id": pid,
                    "player_name": name,
                    "stat_key": stat_key,
                    "vs_team": "all",
                    **metrics,
                })

            # --- per-opponent entries ---
            # Group rows by opponent, preserving date order within each group.
            opp_rows: dict[str, list] = {}
            for r in rows:
                opp = _opponent_from_matchup(r.get("MATCHUP", ""))
                opp_rows.setdefault(opp, []).append(r)

            for opp, orows in opp_rows.items():
                if len(orows) < MIN_VS_TEAM_GAMES:
                    continue
                opp_values = _stat_values(orows, stat_key)
                opp_metrics = _score_player_stat(opp_values,
                                                  min_games=MIN_VS_TEAM_GAMES)
                if opp_metrics is None:
                    continue
                results.append({
                    "player_id": pid,
                    "player_name": name,
                    "stat_key": stat_key,
                    "vs_team": opp,
                    **opp_metrics,
                })

    results.sort(key=lambda r: r["reliability_score"], reverse=True)
    return results


def find_reliable_props(df, top_n: int = 3,
                        active_season: str = "2024") -> list[dict]:
    """
    Return the top_n most-bettable (player, stat[, vs_team]) props.

    Ranking is purely by reliability_score regardless of whether the best version
    of that prop is career-wide ("all") or vs. a specific opponent. A player
    whose PTS is clockwork only against a particular team still tops the list for
    that context.

    "Active" = played at least one game in active_season.

    Each entry includes: player_id, player_name, stat_key, vs_team,
    reliability_score, cv, hit_rate, reversion_rate, autocorr, player_type,
    n_games.
    """
    active_ids = set(
        df[df["GAME_DATE"].dt.year.astype(str) == active_season]["PLAYER_ID"].unique()
    )
    if not active_ids:
        cutoff = df["GAME_DATE"].max() - __import__("pandas").Timedelta(days=365)
        active_ids = set(df[df["GAME_DATE"] >= cutoff]["PLAYER_ID"].unique())

    all_scores = score_all_players(df)
    active_scores = [e for e in all_scores if e["player_id"] in active_ids]
    return active_scores[:top_n]


# Keep old name as alias so existing callers don't break
find_reliable_players = find_reliable_props


def print_reliability_report(top_props: list[dict]) -> None:
    """Pretty-print the reliability report to stdout."""
    stat_labels = {"PTS": "pts", "REB": "reb", "AST": "ast", "PRA": "PRA"}
    print(f"\n{'='*72}")
    print("  MOST BETTABLE PROPS — predicted output vs PrizePicks line")
    print(f"{'='*72}")
    for rank, p in enumerate(top_props, 1):
        rev = (f"{p['reversion_rate']:.0%}" if p["reversion_rate"] is not None
               else "n/a")
        ac = f"{p['autocorr']:+.2f}" if p["autocorr"] is not None else "n/a"
        context = f"vs {p['vs_team']}" if p.get("vs_team", "all") != "all" else "overall"
        unit = stat_labels.get(p["stat_key"], p["stat_key"].lower())
        pick_arrow = "▲ OVER " if p["implied_pick"] == "over" else "▼ UNDER"
        form_note = (
            f"  (last 3: {p['last3_avg']} — "
            + ("running HOT" if p['last3_avg'] > p['rolling_avg'] else "running COLD")
            + ")"
        )

        print(f"\n  #{rank}  {p['player_name']:<26}  {p['stat_key']}  ({context})")
        print(
            f"       {pick_arrow}  Bet they get  {p['predicted']} {unit}"
            f"  |  line ≈ {p['approx_line']}{form_note}"
        )
        print(
            f"       Reliability : {p['reliability_score']:.3f}   "
            f"CV: {p['cv']:.2f}   "
            f"Reversion rate: {rev}   "
            f"Type: {p['player_type']}"
        )
        print(
            f"       Autocorr    : {ac}   "
            f"Rolling avg (10g): {p['rolling_avg']}   "
            f"Games in sample: {p['n_games']}"
        )
    print(f"\n{'='*72}")

