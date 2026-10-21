"""Check data sources from the machine this runs on. Reports status as GitHub
annotations, so a failed run shows which source is down without opening the log."""
import requests

H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
E = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def get(name, url, save=None):
    try:
        r = requests.get(url, headers=H, timeout=25)
        print(f"::notice title=probe {name}::HTTP {r.status_code}, {len(r.content)} bytes")
        return r.json() if r.ok else None
    except Exception as e:
        print(f"::warning title=probe {name}::{type(e).__name__}: {str(e)[:200]}")


sb = get("espn_scoreboard_day", f"{E}/scoreboard?dates=20260410", "scoreboard_day.json")
get("espn_scoreboard_today", f"{E}/scoreboard", "scoreboard_today.json")
if sb and sb.get("events"):
    get("espn_summary", f"{E}/summary?event={sb['events'][0]['id']}", "summary.json")
get("espn_teams", f"{E}/teams")
get("espn_injuries", f"{E}/injuries")
