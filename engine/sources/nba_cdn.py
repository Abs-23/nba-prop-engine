"""NBA's public CDN: season schedule and per-game box scores.

We use cdn.nba.com (not stats.nba.com) because stats.nba.com often blocks
cloud servers like GitHub Actions, while the CDN serves static JSON reliably.
"""
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
}
ET = ZoneInfo("America/New_York")
# 002 = regular season, 004 = playoffs, 005 = play-in. Preseason/All-Star excluded.
COUNTED_PREFIXES = ("002", "004", "005")


class NotFound(Exception):
    pass


def get_json(url, retries=3, timeout=20):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code in (403, 404):
                raise NotFound(url)
            r.raise_for_status()
            return r.json()
        except NotFound:
            raise
        except Exception as e:  # network hiccup: back off and retry
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def utc_to_et_date(utc_str):
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    return dt.astimezone(ET).date().isoformat()


def fetch_schedule():
    """All counted games of the current season with ET date and teams."""
    js = get_json(SCHEDULE_URL)
    games = []
    for day in js["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            gid = g["gameId"]
            if not gid.startswith(COUNTED_PREFIXES):
                continue
            utc = g.get("gameDateTimeUTC") or g.get("gameDateUTC")
            games.append({
                "game_id": gid,
                "date": utc_to_et_date(utc),
                "tipoff_utc": utc,
                "status": int(g.get("gameStatus", 1)),
                "home": g["homeTeam"]["teamTricode"],
                "away": g["awayTeam"]["teamTricode"],
                "home_id": g["homeTeam"]["teamId"],
                "away_id": g["awayTeam"]["teamId"],
            })
    return games


def parse_minutes(s):
    """'PT34M12.00S' -> 34.2"""
    if not s:
        return 0.0
    m = re.match(r"PT(\d+)M([\d.]+)S", s)
    if not m:
        return 0.0
    return round(int(m.group(1)) + float(m.group(2)) / 60, 2)


def parse_box(js):
    """Box score JSON -> list of player-game rows. None if the game isn't final."""
    g = js["game"]
    if int(g.get("gameStatus", 0)) != 3:
        return None
    date = (g.get("gameEt") or "")[:10] or utc_to_et_date(g["gameTimeUTC"])
    rows = []
    for side, other in (("homeTeam", "awayTeam"), ("awayTeam", "homeTeam")):
        t, o = g[side], g[other]
        for p in t.get("players", []):
            s = p.get("statistics", {}) or {}
            rows.append({
                "game_id": g["gameId"], "date": date,
                "team": t["teamTricode"], "opp": o["teamTricode"],
                "home": side == "homeTeam",
                "player_id": int(p["personId"]),
                "player": p.get("name") or f'{p.get("firstName", "")} {p.get("familyName", "")}'.strip(),
                "starter": str(p.get("starter", "0")) == "1",
                "played": str(p.get("played", "0")) == "1",
                "min": parse_minutes(s.get("minutes") or s.get("minutesCalculated")),
                "pts": s.get("points", 0), "reb": s.get("reboundsTotal", 0),
                "oreb": s.get("reboundsOffensive", 0), "ast": s.get("assists", 0),
                "stl": s.get("steals", 0), "blk": s.get("blocks", 0),
                "tov": s.get("turnovers", 0),
                "fgm": s.get("fieldGoalsMade", 0), "fga": s.get("fieldGoalsAttempted", 0),
                "fg3m": s.get("threePointersMade", 0), "fg3a": s.get("threePointersAttempted", 0),
                "ftm": s.get("freeThrowsMade", 0), "fta": s.get("freeThrowsAttempted", 0),
            })
    return rows


def fetch_box(gid):
    return parse_box(get_json(BOX_URL.format(gid=gid)))
