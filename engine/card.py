"""Tonight's card: 3 to 6 props at the REAL DraftKings line.

Two cards, each with its own bar (config.yaml -> card -> cards):
  * "Most likely": the highest-chance legs at any price
  * "Fair price": only legs priced -150 or better, with a lower bar to match

How a leg qualifies (all must hold):
  * blended chance >= p_min. The blend combines our model with DraftKings' own
    price, using weights learned from last season's real lines and results
    (scripts/backtest_lines.py). The market carries most of the weight because
    it predicted results better than the model alone did.
  * recent form: the player cleared this exact line in >= form_min of his last 10
  * tonight's matchup (who's out, defense, pace, minutes) leans the pick's way
  * he passes every player gate (healthy, steady minutes, not new to his team...)

The card takes every qualifying leg, best first, up to 6 (one per player, at most
2 per game). If fewer than 3 qualify, it is filled to 3 with the next-best legs,
each clearly marked "below bar".
"""
import numpy as np

from engine.model import hit_rate, side_probability
from engine.picks import common_reasons, general_gate, teammates_out
from engine.sources.odds import american_to_decimal
from engine.stats import LABELS

CARD_STATS = ["pts", "reb", "ast", "fg3m", "pra", "pr", "pa", "ra"]


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return np.log(p / (1 - p))


def blend(p_model, fair, b):
    return float(1 / (1 + np.exp(-(b[0] + b[1] * _logit(p_model) + b[2] * _logit(fair)))))


def _price(p):
    p = int(p)
    return f"+{p}" if p > 0 else str(p)


def candidates(profiles, lines, injuries, cfg):
    """Every (player, stat, side) with a DraftKings line, scored. Gated players are dropped."""
    cc = cfg["card"]
    b = cc["blend"]
    outs = teammates_out(profiles, injuries)
    by_id = {int(p["player_id"]): p for p in profiles}
    out = []
    for ln in lines:
        prof = by_id.get(int(ln["player_id"]))
        stat = ln["stat"]
        if prof is None or stat not in CARD_STATS or general_gate(prof, injuries, cfg):
            continue
        mean, disp = prof["means"][stat], prof["disp"][stat]
        base = prof.get("base_means", prof["means"])[stat]
        vals = prof["history"][stat].to_numpy()
        net = (prof.get("matchup_detail", {}).get(stat) or {}).get("net", 0.0)
        for side in ("over", "under"):
            line = ln["line"]
            p_model = side_probability(mean, disp, line, side)
            fair = ln["fair_over"] if side == "over" else 1 - ln["fair_over"]
            p = blend(p_model, fair, b)
            price = ln["over_price"] if side == "over" else ln["under_price"]
            h10, n10 = hit_rate(vals[-10:], line, side)
            h20, n20 = hit_rate(vals[-20:], line, side)
            lean = net if side == "over" else -net
            out.append({"prof": prof, "stat": stat, "side": side, "line": line, "price": price,
                        "book": ln.get("book", "DraftKings"), "open_line": ln.get("open_line", line),
                        "p": p, "p_model": p_model, "fair": fair, "mean": mean, "base": base,
                        "h10": h10, "n10": n10, "h20": h20, "n20": n20, "lean": lean,
                        "ev": p * american_to_decimal(price) - 1, "outs": outs})
    return out


def price_ok(c, rule):
    return rule.get("price_min") is None or c["price"] >= rule["price_min"]


def qualifies(c, rule):
    return (price_ok(c, rule) and c["p"] >= rule["p_min"] and c["h10"] is not None
            and c["h10"] >= rule["form_min"] and c["lean"] >= rule["lean_min"])


def reasons(c, cfg, injuries, rule):
    prof, stat, side, line = c["prof"], c["stat"], c["side"], c["line"]
    lab = LABELS[stat].lower()
    n_hist = len(prof["history"])
    r = [
        ("+" if c["p"] >= rule["p_min"] else "!",
         f'Chance it hits: {100*c["p"]:.0f}%. DraftKings prices {side} {line} at {_price(c["price"])} '
         f'({100*c["fair"]:.0f}% after removing their margin); our model says {100*c["p_model"]:.0f}%. '
         f'The two are blended using how each predicted results last season.'),
        ("+" if side == "over" and c["mean"] > line or side == "under" and c["mean"] < line else "!",
         f'Projection: {c["mean"]:.1f} {lab} tonight (his usual {c["base"]:.1f}) vs the line of {line}.'),
        ("+" if (c["h10"] or 0) >= 0.7 else "i",
         f'Recent form: went {side} {line} in {c["n10"]} of his last {min(10, n_hist)} games '
         f'and {c["n20"]} of his last {min(20, n_hist)}.'),
    ]
    if c["open_line"] != line:
        moved_our_way = (c["open_line"] < line) if side == "over" else (c["open_line"] > line)
        r.append(("i" if moved_our_way else "+",
                  f'Line moved from {c["open_line"]} to {line} since it opened'
                  + (" (the market has moved toward this side; value is thinner)." if moved_our_way
                     else " (it moved against this side, so the number is now easier).")))
    r += common_reasons(prof, stat, injuries, c["outs"], injuries is not None, side, cfg)
    return r


def to_pick(c, cfg, injuries, rule, card_name, below_bar=False):
    prof = c["prof"]
    stat, side, line = c["stat"], c["side"], c["line"]
    return {
        "kind": "card", "tier": rule.get("title", card_name), "card": card_name,
        "game_id": prof["game_id"], "matchup": prof["matchup"], "tipoff_utc": prof.get("tipoff_utc"),
        "player_id": prof["player_id"], "player": prof["player"], "team": prof["team"],
        "stat": stat, "stat_label": LABELS[stat], "side": side, "line": line,
        "label": f'{prof["player"]} {side.title()} {line} {LABELS[stat]}',
        "prob": c["p"], "prob_model": c["p_model"], "fair": c["fair"], "edge": c["p"] - c["fair"],
        "ev": c["ev"], "price": c["price"], "book": c["book"], "proj": c["mean"],
        "l10": c["h10"], "l20": c["h20"], "matchup_net": c["lean"], "below_bar": below_bar,
        "reasons": reasons(c, cfg, injuries, rule),
    }


def _pick_legs(pool, n_max, per_game, taken=None):
    legs, players, games = [], set(), {}
    for leg in taken or []:
        players.add(leg["prof"]["player_id"])
        games[leg["prof"]["game_id"]] = games.get(leg["prof"]["game_id"], 0) + 1
    for c in sorted(pool, key=lambda c: -c["p"]):
        pid, gid = c["prof"]["player_id"], c["prof"]["game_id"]
        if pid in players or games.get(gid, 0) >= per_game:
            continue
        legs.append(c)
        players.add(pid)
        games[gid] = games.get(gid, 0) + 1
        if len(legs) >= n_max:
            break
    return legs


def build_card(cands, cfg, injuries, card_name):
    """Returns (card dict or None, list of all qualifying picks) for one card rule."""
    cc = cfg["card"]
    rule = cc["cards"][card_name]
    good = [c for c in cands if qualifies(c, rule)]
    legs = _pick_legs(good, cc["max_legs"], cc["max_per_game"])
    fill = []
    if len(legs) < cc["min_legs"]:
        backup = [c for c in cands if c not in good and price_ok(c, rule) and c["p"] >= rule["fill_p_min"]
                  and (c["h10"] or 0) >= rule["fill_form_min"] and c["lean"] >= 0]
        fill = _pick_legs(backup, cc["min_legs"] - len(legs), cc["max_per_game"], taken=legs)
    picks = ([to_pick(c, cfg, injuries, rule, card_name) for c in legs]
             + [to_pick(c, cfg, injuries, rule, card_name, True) for c in fill])
    qualified = [to_pick(c, cfg, injuries, rule, card_name) for c in sorted(good, key=lambda c: -c["p"])]
    if len(picks) < cc["min_legs"]:
        return None, qualified
    return {"name": card_name, "title": rule.get("title", card_name), "size": len(picks), "legs": picks,
            "prob": float(np.prod([p["prob"] for p in picks])), "n_qualified": len(legs),
            "n_filled": len(fill)}, qualified


def build_cards(cands, cfg, injuries):
    """All cards in config order. Returns ({name: [card] or []}, all qualifying picks)."""
    cards, qualified = {}, []
    for name in cfg["card"]["cards"]:
        card, q = build_card(cands, cfg, injuries, name)
        cards[name] = [card] if card else []
        qualified += q
    return cards, qualified
