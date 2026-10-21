"""Injury report from ESPN's public JSON feed."""
import requests

from engine.stats import norm_name

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def fetch_injuries():
    """Returns {normalized player name: {"status": "out", "detail": "..."}}.

    If the feed is down we return None (not {}), so the engine knows the
    report is missing and can say so instead of assuming everyone is healthy.
    """
    try:
        js = requests.get(URL, timeout=20).json()
    except Exception:
        return None
    out = {}
    for team in js.get("injuries", []):
        for inj in team.get("injuries", []):
            name = (inj.get("athlete") or {}).get("displayName")
            if not name:
                continue
            out[norm_name(name)] = {
                "name": name,
                "status": str(inj.get("status", "")).strip().lower(),
                "detail": inj.get("shortComment") or "",
            }
    return out
