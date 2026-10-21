"""Render docs/index.html (served by GitHub Pages) and docs/picks.json."""
import json
from datetime import date, datetime, timezone

import numpy as np

from engine import DOCS, ROOT

KEEP = ["kind", "tier", "game_id", "matchup", "player", "team", "stat", "stat_label", "side", "line",
        "label", "prob", "fair", "edge", "ev", "price", "book", "proj", "l10", "l20", "reasons", "matchup_net"]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _pick(p):
    d = {k: p.get(k) for k in KEEP}
    if p.get("price") is not None:
        pr = int(p["price"])
        d["price_fmt"] = f"+{pr}" if pr > 0 else str(pr)
    return d


def render(day, games_today, safe, market, lineups, record, health, odds_enabled, out_tonight=None):
    data = {
        "date": day,
        "date_label": date.fromisoformat(day).strftime("%A, %B %-d, %Y"),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "games_today": games_today,
        "safe": [_pick(p) for p in sorted(safe, key=lambda p: -p["prob"])],
        "market": [_pick(p) for p in sorted(market, key=lambda p: -p["ev"])],
        "lineups": {f: [{"size": l["size"], "prob": l["prob"], "legs": [_pick(x) for x in l["legs"]]}
                        for l in lus] for f, lus in lineups.items()},
        "record": record, "health": health, "odds_enabled": odds_enabled, "out_tonight": out_tonight or {},
    }
    data = _clean(data)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "picks.json").write_text(json.dumps(data, indent=1, default=str))
    tpl = (ROOT / "engine" / "template.html").read_text()
    html = tpl.replace("__DATA__", json.dumps(data, default=str).replace("</", "<\\/"))
    (DOCS / "index.html").write_text(html)
    (DOCS / ".nojekyll").write_text("")
