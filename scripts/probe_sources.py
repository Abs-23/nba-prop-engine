"""Check data sources from the machine this runs on. Reports status as GitHub
annotations and saves small raw samples to data/debug/ for inspection."""
import json
from pathlib import Path

import requests

H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
E = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
OUT = Path("data/debug")
OUT.mkdir(parents=True, exist_ok=True)


def get(name, url, save=None):
    try:
        r = requests.get(url, headers=H, timeout=25)
        print(f"::notice title=probe {name}::HTTP {r.status_code}, {len(r.content)} bytes")
        if save and r.ok:
            (OUT / save).write_text(r.text)
        return r.json() if r.ok else None
    except Exception as e:
        print(f"::warning title=probe {name}::{type(e).__name__}: {str(e)[:200]}")


sb = get("espn_scoreboard_day", f"{E}/scoreboard?dates=20260410", "scoreboard_day.json")
rng = get("espn_scoreboard_range", f"{E}/scoreboard?dates=20251020-20251110&limit=1000")
if rng:
    print(f"::notice title=range events::{len(rng.get('events', []))} events in 2025-10-20..2025-11-10")
get("espn_scoreboard_today", f"{E}/scoreboard", "scoreboard_today.json")
if sb and sb.get("events"):
    get("espn_summary", f"{E}/summary?event={sb['events'][0]['id']}", "summary.json")
get("espn_teams", f"{E}/teams", "teams.json")
get("espn_roster", f"{E}/teams/1/roster", "roster.json")
get("espn_injuries", f"{E}/injuries")
get("nba_cdn", "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json")
