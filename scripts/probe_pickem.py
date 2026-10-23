"""One-off: does The Odds API (our key) carry pick'em books (PrizePicks, Underdog) for NBA props?"""
import os

import requests

KEY = os.environ.get("ODDS_API_KEY")
B = "https://api.the-odds-api.com/v4/sports/basketball_nba"
if not KEY:
    print("::warning title=odds api::no key in secrets")
    raise SystemExit
ev = requests.get(f"{B}/events", params={"apiKey": KEY}, timeout=20)
events = ev.json() if ev.ok else []
print(f"::notice title=events::HTTP {ev.status_code}, {len(events)} events, credits left {ev.headers.get('x-requests-remaining')}")
if events:
    e = events[0]
    r = requests.get(f"{B}/events/{e['id']}/odds", params={"apiKey": KEY, "regions": "us_dfs",
                     "markets": "player_points", "oddsFormat": "american"}, timeout=20)
    js = r.json() if r.ok else {}
    books = [b["key"] for b in js.get("bookmakers", [])] if isinstance(js, dict) else []
    n = sum(len(m.get("outcomes", [])) for b in js.get("bookmakers", []) for m in b.get("markets", [])) if books else 0
    print(f"::notice title=us_dfs region::HTTP {r.status_code}, books {books}, {n} outcomes, cost {r.headers.get('x-requests-last')}, left {r.headers.get('x-requests-remaining')}, body {str(js)[:150] if not books else ''}")
    if books:
        b = js["bookmakers"][0]; m = b["markets"][0]
        print(f"::notice title=sample::{b['key']} {m['key']} {m['outcomes'][:2]}")
