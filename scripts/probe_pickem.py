"""One-off: can GitHub's servers reach pick'em line feeds (PrizePicks, Underdog)? Reports via annotations."""
import json
from pathlib import Path

import requests

H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
     "Accept": "application/json", "Referer": "https://app.prizepicks.com/", "Origin": "https://app.prizepicks.com"}
OUT = Path("data/debug"); OUT.mkdir(parents=True, exist_ok=True)
urls = {
    "prizepicks": "https://api.prizepicks.com/projections?league_id=7&per_page=250&single_stat=true&game_mode=pickem",
    "prizepicks_partner": "https://partner-api.prizepicks.com/projections?league_id=7&per_page=1000",
    "underdog": "https://api.underdogfantasy.com/beta/v5/over_under_lines",
}
for name, u in urls.items():
    try:
        r = requests.get(u, headers=H, timeout=25)
        info = f"HTTP {r.status_code}, {len(r.content)} bytes"
        if r.ok:
            js = r.json()
            n = len(js.get("data", [])) if isinstance(js, dict) and "data" in js else len(js.get("over_under_lines", [])) if isinstance(js, dict) else 0
            info += f", {n} items"
            (OUT / f"{name}.json").write_text(r.text[:300000])
        else:
            info += f", body: {r.text[:120]!r}"
        print(f"::notice title=probe {name}::{info}")
    except Exception as e:
        print(f"::warning title=probe {name}::{type(e).__name__}: {str(e)[:200]}")
