"""ESPN's public NBA feeds: schedule (scoreboard) and box scores (summary).

We use ESPN because NBA.com's servers block GitHub's machines (HTTP 403).
"""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
ET = ZoneInfo("America/New_York")

# ESPN abbreviations -> NBA tricodes (the ones the rest of the engine uses)
ABBR = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX",
        "BRK": "BKN", "CHO": "CHA"}
NBA_TEAMS = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL",
             "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR",
             "UTA", "WAS"}
# ESPN season types: 1 preseason, 2 regular season, 3 postseason, 5 play-in
COUNTED_TYPES = {2, 3, 5}


def tri(abbr):
    a = str(abbr).upper()
    return ABBR.get(a, a)


def get_json(url, params=None, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=25)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def season_start_year(day_iso):
    d = date.fromisoformat(day_iso)
    return d.year if d.month >= 8 else d.year - 1


def _et_date(utc):
    dt = datetime.fromisoformat(utc.replace("Z", "+00:00"))
    return dt.astimezone(ET).date().isoformat()


def parse_scoreboard(js):
    games = []
    for ev in js.get("events", []):
        stype = int(((ev.get("season") or {}).get("type")) or 2)
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: tri(c["team"]["abbreviation"]) for c in comp["competitors"]}
        if stype not in COUNTED_TYPES or not {teams.get("home"), teams.get("away")} <= NBA_TEAMS:
            continue
        st = (comp.get("status") or ev.get("status") or {}).get("type", {})
        games.append({
            "game_id": str(ev["id"]), "date": _et_date(ev["date"]), "tipoff_utc": ev["date"],
            "completed": bool(st.get("completed")), "home": teams["home"], "away": teams["away"],
        })
    return games


def scoreboard(day_iso):
    return parse_scoreboard(get_json(f"{BASE}/scoreboard", {"dates": day_iso.replace("-", ""), "limit": 100}))


def games_between(start_iso, end_iso, workers=8):
    """All counted games with an ET date in [start, end], one request per day."""
    d0, d1 = date.fromisoformat(start_iso), date.fromisoformat(end_iso)
    days = [(d0 + timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]

    def one(day):
        try:
            return scoreboard(day)
        except Exception:
            return []

    with ThreadPoolExecutor(workers) as ex:
        out = {g["game_id"]: g for gs in ex.map(one, days) for g in gs}
    return sorted(out.values(), key=lambda g: (g["date"], g["game_id"]))


# --------------------------------------------------------------- box scores

LABEL_MAP = {"MIN": "min", "PTS": "pts", "REB": "reb", "OREB": "oreb", "AST": "ast", "STL": "stl",
             "BLK": "blk", "TO": "tov", "FG": ("fgm", "fga"), "3PT": ("fg3m", "fg3a"), "FT": ("ftm", "fta")}
KEY_MAP = {"minutes": "min", "points": "pts", "rebounds": "reb", "offensiveRebounds": "oreb",
           "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov",
           "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
           "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
           "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta")}
FIELDS = ["min", "pts", "reb", "oreb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]


def _num(v):
    try:
        return float(str(v).strip())
    except ValueError:
        return 0.0


def _minutes(v):
    s = str(v).strip()
    if ":" in s:
        m, sec = s.split(":", 1)
        return round(_num(m) + _num(sec) / 60, 2)
    return _num(s)


def parse_summary(js, game_id=None):
    """Summary JSON -> list of player-game rows, or None if the game isn't final."""
    comp = js["header"]["competitions"][0]
    if not comp.get("status", {}).get("type", {}).get("completed"):
        return None
    side = {tri(c["team"]["abbreviation"]): c["homeAway"] for c in comp["competitors"]}
    gid = str(game_id or js["header"]["id"])
    day = _et_date(comp["date"])
    teams = list(side)
    rows = []
    for tb in js.get("boxscore", {}).get("players", []):
        team = tri(tb["team"]["abbreviation"])
        opp = next((t for t in teams if t != team), None)
        for block in tb.get("statistics", []):
            keys, labels = block.get("keys") or block.get("names") or [], block.get("labels") or []
            cols = [KEY_MAP.get(k) for k in keys] if keys and any(k in KEY_MAP for k in keys) \
                else [LABEL_MAP.get(l) for l in labels]
            for a in block.get("athletes", []):
                ath = a.get("athlete", {})
                vals = a.get("stats") or []
                row = dict.fromkeys(FIELDS, 0.0)
                for col, v in zip(cols, vals):
                    if col is None:
                        continue
                    if isinstance(col, tuple):
                        made, att = (str(v).split("-") + ["0"])[:2]
                        row[col[0]], row[col[1]] = _num(made), _num(att)
                    elif col == "min":
                        row["min"] = _minutes(v)
                    else:
                        row[col] = _num(v)
                played = bool(vals) and not a.get("didNotPlay") and row["min"] > 0
                rows.append({"game_id": gid, "date": day, "team": team, "opp": opp,
                             "home": side.get(team) == "home", "player_id": int(ath["id"]),
                             "player": ath.get("displayName", ""), "starter": bool(a.get("starter")),
                             "played": played, **{k: (row[k] if k == "min" else int(row[k])) for k in FIELDS}})
    return rows


def box(game_id):
    return parse_summary(get_json(f"{BASE}/summary", {"event": game_id}), game_id)
