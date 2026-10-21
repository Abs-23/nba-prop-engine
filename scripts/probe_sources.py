"""Check every data source from the machine this runs on and report status as
GitHub annotations (visible in the run summary without opening the log)."""
import requests

from engine.sources.nba_cdn import HEADERS

URLS = {
    "nba_schedule": "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
    "nba_box_2025": "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022500001.json",
    "nba_scoreboard": "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
    "stats_nba": "https://stats.nba.com/stats/scoreboardv2?GameDate=2026-10-20&LeagueID=00&DayOffset=0",
    "espn_scoreboard": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "espn_injuries": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
    "odds_api": "https://api.the-odds-api.com/v4/sports",
}
for name, url in URLS.items():
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        print(f"::notice title=probe {name}::HTTP {r.status_code}, {len(r.content)} bytes")
    except Exception as e:
        print(f"::warning title=probe {name}::{type(e).__name__}: {str(e)[:200]}")
