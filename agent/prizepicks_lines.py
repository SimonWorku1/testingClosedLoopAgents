"""
Fetch today's WNBA prop lines from the PrizePicks public API.

PrizePicks exposes a JSON:API endpoint — no auth required:
  GET https://api.prizepicks.com/projections?league_id=7&per_page=250

Returns a dict keyed by (player_name_lower, stat_key) → line_score so it can
be joined directly against the reliability scorer output.

Stat-name mapping (PrizePicks → our internal keys):
  "Points"           → PTS
  "Rebounds"         → REB
  "Assists"          → AST
  "Pts+Rebs+Asts"    → PRA   (PrizePicks spelling)
  "Pts+Reb+Ast"      → PRA   (alternate spelling seen in older responses)
"""

import time
import urllib.request
import json

PRIZEPICKS_URL = (
    "https://api.prizepicks.com/projections"
    "?league_id=7"           # 7 = WNBA
    "&per_page=250"
    "&single_stat=true"      # one line per row
)

STAT_MAP = {
    "points":           "PTS",
    "rebounds":         "REB",
    "assists":          "AST",
    "pts+rebs+asts":    "PRA",
    "pts+reb+ast":      "PRA",
    "points+rebounds+assists": "PRA",
}


def fetch_lines(retries: int = 3) -> dict[tuple[str, str], float]:
    """
    Return {(player_name_lower, stat_key): line_score} for all active WNBA
    projections on PrizePicks right now.

    Returns an empty dict if the fetch fails (so callers can degrade gracefully).
    """
    headers = {
        "Accept":     "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; wnba-prop-tool/1.0)",
        "Referer":    "https://app.prizepicks.com/",
    }

    last_exc = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(PRIZEPICKS_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
            break
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    else:
        print(f"  [prizepicks] fetch failed after {retries} attempts: {last_exc}")
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [prizepicks] JSON parse error: {exc}")
        return {}

    # JSON:API format: `included` has player objects keyed by id.
    included = {
        item["id"]: item
        for item in payload.get("included", [])
        if item.get("type") == "new_player"
    }

    lines: dict[tuple[str, str], float] = {}
    skipped = 0
    for proj in payload.get("data", []):
        if proj.get("type") != "projection":
            continue

        attrs = proj.get("attributes", {})
        stat_raw = str(attrs.get("stat_type", "")).lower().strip()
        stat_key = STAT_MAP.get(stat_raw)
        if stat_key is None:
            skipped += 1
            continue

        line_score = attrs.get("line_score")
        if line_score is None:
            continue

        # Resolve player name from the relationship → included lookup.
        rel = proj.get("relationships", {}).get("new_player", {}).get("data", {})
        player_obj = included.get(rel.get("id", ""))
        if player_obj is None:
            continue
        player_name = player_obj.get("attributes", {}).get("name", "")
        if not player_name:
            continue

        key = (player_name.lower().strip(), stat_key)
        lines[key] = float(line_score)

    print(f"  [prizepicks] {len(lines)} WNBA lines fetched "
          f"({skipped} non-target stats skipped)")
    return lines


def match_line(lines: dict, player_name: str, stat_key: str) -> float | None:
    """
    Look up a PrizePicks line for a given player + stat.
    Tries exact lowercase match first, then falls back to partial name match
    (handles 'A. Wilson' vs \"A'ja Wilson\" style mismatches).
    """
    key = (player_name.lower().strip(), stat_key)
    if key in lines:
        return lines[key]

    # Partial fallback: last-name match
    last = player_name.split()[-1].lower()
    for (pname, skey), score in lines.items():
        if skey == stat_key and last in pname:
            return score

    return None


def print_lines(lines: dict) -> None:
    """Debug helper — dump all fetched lines."""
    print(f"\n  PrizePicks WNBA lines ({len(lines)} props):")
    for (name, stat), score in sorted(lines.items()):
        print(f"    {name:<30} {stat:<5} {score}")
