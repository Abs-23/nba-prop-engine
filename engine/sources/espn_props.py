"""DraftKings player prop lines, as published (free) through ESPN's odds API.

Each main market comes as two entries with the same player and line: the
first is the OVER, the second the UNDER. We double-check that against
DraftKings' "milestone" ladder (e.g. "17+ points" is always an over at the
price of Over 16.5) whenever it's available.

Player ids are ESPN athlete ids, the same ids as our box-score database,
so no name matching is needed.
"""
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

URL = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{e}/competitions/{e}"
       "/odds/100/propBets?lang=en&region=us&limit=1000&page={p}")
H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
BOOK = "DraftKings"

TYPE_TO_STAT = {
    "Total Points": "pts", "Total Rebounds": "reb", "Total Assists": "ast",
    "Total 3-Point Field Goals": "fg3m", "Total Points and Rebounds": "pr",
    "Total Points and Assists": "pa", "Total Assists and Rebounds": "ra",
    "Total Points, Rebounds, and Assists": "pra", "Total Steals": "stl", "Total Blocks": "blk",
    "Total Steals and Blocks": "bs",
}
MILESTONE = {"pts": "Points Milestones", "ast": "Assists Milestones", "reb": "Rebounds Milestones",
             "fg3m": "3-Point Milestones"}


def _get(url, retries=3):
    for a in range(retries):
        try:
            r = requests.get(url, headers=H, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.5 * (a + 1))
    return None


def _prob(american):
    a = float(american)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def fetch_items(event_id):
    items, page = [], 1
    while True:
        js = _get(URL.format(e=event_id, p=page))
        if not js:
            break
        items += js.get("items", [])
        if page >= int(js.get("pageCount") or 1):
            break
        page += 1
    return items


def parse(items, event_id):
    groups = defaultdict(list)
    for x in items:
        ref = (x.get("athlete") or {}).get("$ref", "")
        if "/athletes/" not in ref:
            continue
        pid = ref.split("/athletes/")[-1].split("?")[0]
        groups[(pid, (x.get("type") or {}).get("name"))].append(x)
    rows = []
    for (pid, tname), xs in groups.items():
        stat = TYPE_TO_STAT.get(tname)
        if not stat or len(xs) != 2:
            continue
        try:
            line = float(xs[0]["odds"]["total"]["value"])
            if float(xs[1]["odds"]["total"]["value"]) != line:
                continue
            a = int(float(xs[0]["odds"]["american"]["value"]))
            b = int(float(xs[1]["odds"]["american"]["value"]))
        except (KeyError, TypeError, ValueError):
            continue
        over, under = a, b
        ms = [m for m in groups.get((pid, MILESTONE.get(stat)), [])
              if (m.get("current") or {}).get("target", {}).get("value") == line + 0.5]
        if ms:  # the milestone "line+0.5 or more" is priced like the over
            mp = _prob(ms[0]["odds"]["american"]["value"])
            if abs(_prob(b) - mp) < abs(_prob(a) - mp):
                over, under = b, a
        po, pu = _prob(over), _prob(under)
        try:
            open_line = float(xs[0]["odds"]["total"].get("open") or line)
        except ValueError:
            open_line = line
        rows.append({"game_id": str(event_id), "player_id": int(pid), "stat": stat, "line": line,
                     "over_price": over, "under_price": under, "fair_over": po / (po + pu),
                     "open_line": open_line, "book": BOOK, "updated": xs[0].get("lastUpdated")})
    return rows


def fetch_lines(event_ids, workers=6):
    """All main-line player props for these games. Missing games just return nothing."""
    out = []
    with ThreadPoolExecutor(workers) as ex:
        for eid, items in zip(event_ids, ex.map(fetch_items, event_ids)):
            out += parse(items, eid)
    return out
