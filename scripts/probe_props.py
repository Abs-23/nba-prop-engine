"""One-off: does ESPN's core API expose player prop lines? Saves small samples to data/debug/."""
import json
from pathlib import Path

import requests

H = {"User-Agent": "Mozilla/5.0"}
OUT = Path("data/debug"); OUT.mkdir(parents=True, exist_ok=True)
sb = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", headers=H, timeout=20).json()
eid = sb["events"][0]["id"]
base = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{eid}/competitions/{eid}/odds"
r = requests.get(base, headers=H, timeout=20)
print(f"::notice title=odds list::HTTP {r.status_code} {len(r.content)}b")
(OUT / "odds_list.json").write_text(r.text[:200000])
provs = []
try:
    for it in r.json().get("items", []):
        ref = it.get("$ref") or ""
        provs.append(ref.split("/odds/")[1].split("?")[0] if "/odds/" in ref else (it.get("provider") or {}).get("id"))
except Exception as e:
    print(f"::warning title=parse::{e}")
print(f"::notice title=providers::{provs}")
for pid in [p for p in provs if p][:4] + ["58", "100"]:
    u = f"{base}/{pid}/propBets?lang=en&region=us&limit=1000"
    rr = requests.get(u, headers=H, timeout=20)
    n = len(rr.json().get("items", [])) if rr.ok else 0
    print(f"::notice title=propBets provider {pid}::HTTP {rr.status_code}, {n} items, {len(rr.content)}b")
    if rr.ok and n:
        (OUT / f"props_{pid}.json").write_text(rr.text[:400000])
        it = rr.json()["items"][0]
        for k in ("athlete", "type", "property"):
            ref = (it.get(k) or {}).get("$ref") if isinstance(it.get(k), dict) else None
            if ref:
                x = requests.get(ref.replace("http://", "https://"), headers=H, timeout=20)
                (OUT / f"props_{pid}_{k}.json").write_text(x.text[:50000])
        break
