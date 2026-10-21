"""Current team rosters from ESPN, so traded/signed players are on the right team."""
import requests

from engine.stats import TEAM_NAMES, norm_name

TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/roster"


def fetch_rosters():
    """Returns {normalized player name: team tricode}, or None if ESPN is unreachable."""
    try:
        js = requests.get(TEAMS_URL, timeout=20).json()
        teams = [t["team"] for t in js["sports"][0]["leagues"][0]["teams"]]
    except Exception:
        return None
    out = {}
    for t in teams:
        tri = TEAM_NAMES.get(t.get("displayName"))
        if not tri:
            continue
        try:
            r = requests.get(ROSTER_URL.format(tid=t["id"]), timeout=20).json()
        except Exception:
            continue
        athletes = r.get("athletes", [])
        # ESPN sometimes groups athletes by position: [{"items": [...]}, ...]
        if athletes and isinstance(athletes[0], dict) and "items" in athletes[0]:
            athletes = [a for grp in athletes for a in grp.get("items", [])]
        for a in athletes:
            name = a.get("displayName") or a.get("fullName")
            if name:
                out[norm_name(name)] = tri
    return out or None
