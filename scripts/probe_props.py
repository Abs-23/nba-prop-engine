"""One-off: do past games keep their DraftKings prop lines on ESPN? Reports via annotations."""
import requests

H = {"User-Agent": "Mozilla/5.0"}
B = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{e}/competitions/{e}/odds/100/propBets?lang=en&region=us&limit=1000&page={p}"
for label, eid in [("today", None), ("last night", "401909090"), ("2026-04-10", "401811026"),
                   ("2026-01-15", None), ("2025-11-05", None)]:
    if eid is None and label != "today":
        d = label.replace("-", "")
        sb = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}", headers=H, timeout=20).json()
        eid = sb["events"][0]["id"] if sb.get("events") else None
    if label == "today":
        sb = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", headers=H, timeout=20).json()
        eid = sb["events"][0]["id"]
    if not eid:
        print(f"::warning title={label}::no event"); continue
    tot = []
    for p in (1, 2):
        r = requests.get(B.format(e=eid, p=p), headers=H, timeout=30)
        js = r.json() if r.ok else {}
        tot.append(f"p{p}: HTTP {r.status_code} items {len(js.get('items', []))} count {js.get('count')}")
        if not r.ok or js.get("pageCount", 1) < 2:
            break
    print(f"::notice title=props {label} ({eid})::" + " | ".join(tot))
