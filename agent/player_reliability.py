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


def _score_player_stat(values: list[float], rolling_window: int = 10) -> dict | None:
    """
    Compute reliability metrics for a single (player, stat) time series.
    `values` must be in chronological order.
    Returns None if there are too few games.
    """
    if len(values) < MIN_GAMES:
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

    return {
        "n_games": len(values),
        "mean": round(mean, 2),
        "cv": round(cv, 3),
        "hit_rate": round(hit_rate, 3),
        "reversion_rate": round(reversion_rate, 3) if reversion_rate is not None else None,
        "autocorr": round(autocorr, 3) if autocorr is not None else None,
        "player_type": player_type,
        "reliability_score": reliability_score,
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


def score_all_players(df, stat_keys=("PTS", "REB", "AST", "PRA")) -> list[dict]:
    """
    Score every (player, stat) pair in the DataFrame.
    Returns a flat list of result dicts sorted descending by reliability_score.
    Each dict includes player_id, player_name, stat_key, and all metric fields.
    """
    results = []
    for pid, group in df.groupby("PLAYER_ID"):
        group = group.sort_values("GAME_DATE")
        rows = group.to_dict("records")
        name = rows[0]["PLAYER_NAME"]

        for stat_key in stat_keys:
            values = _stat_values(rows, stat_key)
            metrics = _score_player_stat(values)
            if metrics is None:
                continue
            results.append({
                "player_id": pid,
                "player_name": name,
                "stat_key": stat_key,
                **metrics,
            })

    results.sort(key=lambda r: r["reliability_score"], reverse=True)
    return results


def find_reliable_players(df, top_n: int = 3,
                           active_season: str = "2024") -> list[dict]:
    """
    Return the `top_n` most-reliable active WNBA players.

    "Active" = played at least one game in `active_season`.
    For each player the best-scoring stat is used as the representative entry.
    The returned list is sorted by reliability_score descending.

    Each entry:
        player_id, player_name, best_stat, reliability_score,
        cv, hit_rate, reversion_rate, autocorr, player_type, n_games
    """
    # Players active in the requested season
    active_ids = set(
        df[df["GAME_DATE"].dt.year.astype(str) == active_season]["PLAYER_ID"].unique()
    )
    if not active_ids:
        # fallback: treat any player with a game in the last 12 months as active
        cutoff = df["GAME_DATE"].max() - __import__("pandas").Timedelta(days=365)
        active_ids = set(df[df["GAME_DATE"] >= cutoff]["PLAYER_ID"].unique())

    all_scores = score_all_players(df)

    # Best stat per active player
    best_by_player: dict[int, dict] = {}
    for entry in all_scores:
        pid = entry["player_id"]
        if pid not in active_ids:
            continue
        if pid not in best_by_player:
            best_by_player[pid] = {**entry, "best_stat": entry["stat_key"]}
        # (already sorted descending, so first occurrence is best)

    ranked = sorted(best_by_player.values(),
                    key=lambda r: r["reliability_score"], reverse=True)
    return ranked[:top_n]


def print_reliability_report(top_players: list[dict]) -> None:
    """Pretty-print the reliability report to stdout."""
    print(f"\n{'='*72}")
    print("  PLAYER RELIABILITY REPORT — top predictable props targets")
    print(f"{'='*72}")
    for rank, p in enumerate(top_players, 1):
        rev = (f"{p['reversion_rate']:.0%}" if p["reversion_rate"] is not None
               else "n/a")
        ac = f"{p['autocorr']:+.2f}" if p["autocorr"] is not None else "n/a"
        print(
            f"\n  #{rank}  {p['player_name']:<26}  best stat: {p['best_stat']}"
        )
        print(
            f"       Reliability score : {p['reliability_score']:.3f}"
        )
        print(
            f"       CV (lower=stable) : {p['cv']:.2f}   "
            f"Hit rate: {p['hit_rate']:.1%}   "
            f"Reversion rate: {rev}"
        )
        print(
            f"       Autocorr (lag-1)  : {ac}   "
            f"Type: {p['player_type']}   "
            f"Games in dataset: {p['n_games']}"
        )
    print(f"\n{'='*72}")
