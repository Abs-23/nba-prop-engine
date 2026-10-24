"""PrizePicks card: 3-6 picks at PrizePicks' line, priced with DraftKings' sharper market.

PrizePicks lines come from The Odds API (region "us_dfs"), the licensed route; we do
not scrape PrizePicks itself. Every PrizePicks pick pays the same, so a pick is worth
taking when the TRUE chance of the side is high at PrizePicks' number.

How the chance is worked out:
  1. DraftKings posts the same player and stat, with a price. We find the outcome
     distribution (our negative binomial, using the player's own volatility) whose
     chance of going over DraftKings' line matches DraftKings' no-vig price.
  2. That distribution gives the chance of More / Less at PrizePicks' line. When the two
     lines differ (PrizePicks 22.5, DraftKings 24.5 at -130 over), this is where an
     edge comes from.
  3. Our own model's chance is blended in with the same weights as the other cards.
If DraftKings has no line for that player and stat, the pick is skipped: our model on
its own was overconfident at real lines last season.
"""
from collections import defaultdict

import numpy as np

from engine.card import blend, _pick_legs, CARD_STATS
from engine.model import hit_rate, side_probability
from engine.picks import common_reasons, general_gate, teammates_out
from engine.stats import LABELS, MARKET_TO_STAT, TEAM_NAMES, norm_name


def tonight_events(client, games):
    """The Odds API events that match tonight's games (listing events costs no credits)."""
    by_teams = {(g["home"], g["away"]): g["game_id"] for g in games}
    return [dict(e, game_id=by_teams[(TEAM_NAMES.get(e["home_team"]), TEAM_NAMES.get(e["away_team"]))])
            for e in client.events()
            if (TEAM_NAMES.get(e["home_team"]), TEAM_NAMES.get(e["away_team"])) in by_teams]


def fetch_pickem_lines(client, events, markets):
    """PrizePicks (and Pick6) lines for these events via The Odds API. Returns rows."""
    rows = []
    for e in events:
        gid = e["game_id"]
        js = client._get(f"/events/{e['id']}/odds", regions="us_dfs", markets=",".join(markets),
                         oddsFormat="american")
        for bk in js.get("bookmakers", []):
            for mk in bk.get("markets", []):
                stat = MARKET_TO_STAT.get(mk["key"])
                if not stat:
                    continue
                seen = set()
                for oc in mk.get("outcomes", []):
                    key = (oc.get("description"), oc.get("point"))
                    if key in seen or oc.get("point") is None:
                        continue
                    seen.add(key)
                    rows.append({"game_id": gid, "player": oc.get("description"),
                                 "player_key": norm_name(oc.get("description", "")), "stat": stat,
                                 "line": float(oc["point"]), "book": bk.get("title", bk["key"]),
                                 "book_key": bk["key"]})
    return rows


def anchored_prob(line_dk, fair_over_dk, disp, line_pp, side):
    """Chance of `side` at PrizePicks' line, using a distribution matched to DraftKings' price."""
    lo, hi = 0.05, max(5.0, line_dk * 3 + 5)
    for _ in range(60):  # bisection on the mean: P(over DK line) must equal DK's no-vig over
        mid = (lo + hi) / 2
        if side_probability(mid, disp, line_dk, "over") < fair_over_dk:
            lo = mid
        else:
            hi = mid
    mu = (lo + hi) / 2
    return side_probability(mu, disp, line_pp, side), mu


def candidates(profiles, pp_lines, dk_lines, injuries, cfg):
    b = cfg["card"]["blend"]
    outs = teammates_out(profiles, injuries)
    by_name = {norm_name(p["player"]): p for p in profiles}
    dk = {(int(l["player_id"]), l["stat"]): l for l in dk_lines}
    out = []
    for ln in pp_lines:
        if ln["book_key"] != "prizepicks" or ln["stat"] not in CARD_STATS:
            continue
        prof = by_name.get(ln["player_key"])
        if prof is None or general_gate(prof, injuries, cfg):
            continue
        d = dk.get((int(prof["player_id"]), ln["stat"]))
        if d is None:
            continue  # no DraftKings anchor -> skip (model alone is overconfident)
        stat, line = ln["stat"], ln["line"]
        mean, disp = prof["means"][stat], prof["disp"][stat]
        vals = prof["history"][stat].to_numpy()
        net = (prof.get("matchup_detail", {}).get(stat) or {}).get("net", 0.0)
        for side in ("over", "under"):
            p_mkt, mu_mkt = anchored_prob(d["line"], d["fair_over"], disp, line, side)
            p_model = side_probability(mean, disp, line, side)
            p = blend(p_model, p_mkt, b)
            h10, n10 = hit_rate(vals[-10:], line, side)
            h20, n20 = hit_rate(vals[-20:], line, side)
            out.append({"prof": prof, "stat": stat, "side": side, "line": line, "dk": d,
                        "p": p, "p_model": p_model, "fair": p_mkt, "mu_mkt": mu_mkt, "mean": mean,
                        "base": prof.get("base_means", prof["means"])[stat],
                        "h10": h10, "n10": n10, "h20": h20, "n20": n20,
                        "lean": net if side == "over" else -net, "outs": outs, "price": None})
    return out


def _reasons(c, cfg, injuries, rule):
    prof, stat, side, line, d = c["prof"], c["stat"], c["side"], c["line"], c["dk"]
    lab = LABELS[stat].lower()
    word = "More" if side == "over" else "Less"
    dk_side = d["over_price"] if side == "over" else d["under_price"]
    pr = lambda p: f"+{int(p)}" if p > 0 else str(int(p))  # noqa: E731
    diff = line - d["line"]
    if diff == 0:
        cmp = f"DraftKings has the same line ({d['line']}) and prices this side at {pr(dk_side)}."
    else:
        easier = (diff < 0) == (side == "over")
        cmp = (f"DraftKings has {d['line']} (over {pr(d['over_price'])} / under {pr(d['under_price'])}); "
               f"PrizePicks' {line} is {abs(diff):g} {'easier' if easier else 'harder'} for {word}.")
    n_hist = len(prof["history"])
    r = [
        ("+" if c["p"] >= rule["p_min"] else "!",
         f'Chance it hits: {100*c["p"]:.0f}%. {cmp} Matching DraftKings\' price puts {word} {line} at '
         f'{100*c["fair"]:.0f}%; our model says {100*c["p_model"]:.0f}%.'),
        ("+" if (side == "over") == (c["mean"] > line) else "!",
         f'Projection: {c["mean"]:.1f} {lab} tonight (his usual {c["base"]:.1f}; DraftKings implies '
         f'{c["mu_mkt"]:.1f}) vs PrizePicks\' {line}.'),
        ("+" if (c["h10"] or 0) >= 0.7 else "i",
         f'Recent form: went {word.lower()} {line} in {c["n10"]} of his last {min(10, n_hist)} '
         f'and {c["n20"]} of his last {min(20, n_hist)}.'),
    ]
    return r + common_reasons(prof, stat, injuries, c["outs"], injuries is not None, side, cfg)


def build_card(cands, cfg, injuries):
    cc = cfg["card"]
    rule = cfg["pickem"]
    good = [c for c in cands if c["p"] >= rule["p_min"] and c["h10"] is not None
            and c["h10"] >= rule["form_min"] and c["lean"] >= rule["lean_min"]
            and c["p_model"] - c["fair"] <= rule["max_gap"]]
    legs = _pick_legs(good, cc["max_legs"], cc["max_per_game"])
    fill = []
    if len(legs) < cc["min_legs"]:
        backup = [c for c in cands if c not in good and c["p"] >= rule["fill_p_min"]
                  and (c["h10"] or 0) >= rule["fill_form_min"] and c["lean"] >= 0
                  and c["p_model"] - c["fair"] <= rule["max_gap"]]
        fill = _pick_legs(backup, cc["min_legs"] - len(legs), cc["max_per_game"], taken=legs)

    def to_pick(c, below=False):
        prof = c["prof"]
        word = "More" if c["side"] == "over" else "Less"
        return {"kind": "card", "tier": "PrizePicks", "card": "prizepicks", "game_id": prof["game_id"],
                "matchup": prof["matchup"], "player_id": prof["player_id"], "player": prof["player"],
                "team": prof["team"], "stat": c["stat"], "stat_label": LABELS[c["stat"]], "side": c["side"],
                "line": c["line"], "label": f'{prof["player"]} {word} {c["line"]} {LABELS[c["stat"]]}',
                "prob": c["p"], "prob_model": c["p_model"], "fair": c["fair"], "edge": c["p"] - 0.5,
                "ev": c["p"] - rule["breakeven"], "price": None, "book": "PrizePicks", "proj": c["mean"],
                "l10": c["h10"], "l20": c["h20"], "matchup_net": c["lean"], "below_bar": below,
                "dk_line": c["dk"]["line"], "reasons": _reasons(c, cfg, injuries, rule)}

    picks = [to_pick(c) for c in legs] + [to_pick(c, True) for c in fill]
    qualified = [to_pick(c) for c in sorted(good, key=lambda c: -c["p"])]
    if len(picks) < cc["min_legs"]:
        return None, qualified
    return {"name": "prizepicks", "title": "PrizePicks", "size": len(picks), "legs": picks,
            "prob": float(np.prod([p["prob"] for p in picks])), "n_qualified": len(legs),
            "n_filled": len(fill)}, qualified


# ---------------------------------------------------------------- free-credit plan + shadow test

SHADOW_STATS = {"ftm", "fta", "fgm"}


def plan_free(remaining, events, cfg, today):
    """Split today's share of the free monthly credits.

    Shadow test first: its stats on as many games as the budget allows (rotating which games
    get covered day to day). Whatever is left goes to the PrizePicks card's stats, one stat
    across all games at a time. Returns [(event, [markets])]."""
    import calendar
    import hashlib
    from datetime import date as _d
    pc, oc = cfg["pickem"], cfg["odds_api"]
    if remaining is None or not events:
        return []
    d = _d.fromisoformat(today)
    days_left = calendar.monthrange(d.year, d.month)[1] - d.day + 1
    budget = int(min(oc["max_daily_credits"], max(0, (remaining - oc["reserve_credits"]) / days_left)))
    plan = {e["id"]: [] for e in events}
    order = sorted(events, key=lambda e: hashlib.md5(f'{today}{e["id"]}'.encode()).hexdigest())
    if pc.get("mode") == "shadow":
        sm = pc["shadow_markets"]
        n = min(len(order), pc.get("shadow_max_games", 99), budget // len(sm))
        for e in order[:n]:
            plan[e["id"]] += sm
        budget -= n * len(sm)
    for m in pc["markets"]:  # card stats: only if the whole slate can be covered
        if budget < len(events):
            break
        for e in events:
            plan[e["id"]].append(m)
        budget -= len(events)
    return [(e, plan[e["id"]]) for e in events if plan[e["id"]]]


def fetch_planned(client, plan):
    rows = []
    for e, markets in plan:
        rows += fetch_pickem_lines(client, [e], markets)
    return rows


def shadow_picks(profiles, pp_lines, injuries, cfg):
    """Model-only picks on stats DraftKings doesn't post (free throws, FG made). Recorded and graded
    so we can see whether the model can be trusted on them. Never put on a card."""
    by_name = {norm_name(p["player"]): p for p in profiles}
    out = []
    for ln in pp_lines:
        if ln["book_key"] != "prizepicks" or ln["stat"] not in SHADOW_STATS:
            continue
        prof = by_name.get(ln["player_key"])
        if prof is None or general_gate(prof, injuries, cfg):
            continue
        stat, line = ln["stat"], ln["line"]
        mean, disp = prof["means"][stat], prof["disp"][stat]
        p_over = side_probability(mean, disp, line, "over")
        side, p = ("over", p_over) if p_over >= 0.5 else ("under", 1 - p_over)
        vals = prof["history"][stat].to_numpy()
        h10, _ = hit_rate(vals[-10:], line, side)
        word = "More" if side == "over" else "Less"
        out.append({"kind": "shadow", "tier": "Shadow test", "card": "shadow", "game_id": prof["game_id"],
                    "matchup": prof["matchup"], "player_id": prof["player_id"], "player": prof["player"],
                    "team": prof["team"], "stat": stat, "stat_label": LABELS[stat], "side": side, "line": line,
                    "label": f'{prof["player"]} {word} {line} {LABELS[stat]}', "prob": p, "prob_model": p,
                    "fair": None, "edge": None, "ev": 0.0, "price": None, "book": "PrizePicks", "proj": mean,
                    "l10": h10, "l20": None, "matchup_net": None, "below_bar": False, "reasons": []})
    return out
