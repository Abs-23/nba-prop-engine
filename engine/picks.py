"""Turn projections into picks — and only the picks that survive every gate.

Two kinds of picks:

  MARKET picks   Over/under a real sportsbook line. Must beat the books' own
                 no-vig probability by a margin (the "edge"), AND agree with the
                 player's recent results. Model alone is never enough.

  SAFE picks     "X or more" thresholds for any stat, including ones books rarely
                 post (FT attempts, FG attempts, 3PA). Chosen as the highest
                 threshold the model gives >= 85% AND the player cleared in
                 >= 85% of his last 20 games. This is the closest honest thing
                 to a "sure shot".

Both kinds also need tonight's MATCHUP to lean their way: the per-stat
matchup checks (engine/matchups.py), weighted by how much each one
mattered in past games, must add up in the pick's favor.

Every pick carries its reasons: each is a fact computed from the data,
labeled as support (+), risk (!) or context (i).
"""
import numpy as np

from engine.model import hit_rate, prob_at_least, side_probability
from engine.sources.odds import american_to_decimal, american_to_prob
from engine.stats import LABELS, norm_name


def _fmt_price(p):
    p = int(p)
    return f"+{p}" if p > 0 else str(p)


def general_gate(prof, injuries, cfg):
    """Reasons a player is excluded entirely tonight (empty list = OK)."""
    g = cfg["gates"]
    fails = []
    if prof["n"] < g["min_games_sample"]:
        fails.append(f'only {prof["n"]} games of history')
    if prof["proj_min"] < g["min_projected_minutes"]:
        fails.append(f'projected {prof["proj_min"]:.0f} min')
    if prof["min_cv10"] > g["max_minutes_cv"]:
        fails.append("minutes too volatile")
    inj = (injuries or {}).get(norm_name(prof["player"]))
    if inj and inj["status"] in g["exclude_injury_statuses"]:
        fails.append(f'injury status: {inj["status"]}')
    return fails


def teammates_out(profiles, injuries):
    """{team: [(name, minutes)]} for rotation players listed OUT tonight."""
    out = {}
    for p in profiles:
        inj = (injuries or {}).get(norm_name(p["player"]))
        if inj and inj["status"] == "out" and p["min_l10"] >= 20:
            out.setdefault(p["team"], []).append((p["player"], p["min_l10"]))
    return out


def common_reasons(prof, stat, injuries, outs, injuries_ok, side="over", cfg=None):
    """Context reasons. Each is marked + (helps this pick) or ! (works against it),
    which depends on the side: a generous defense helps an over and hurts an under."""
    r = []
    up = side == "over"
    stab = "stable" if prof["min_cv10"] < 0.15 else "fairly stable"
    r.append(("+" if prof["min_cv10"] < 0.15 else "i",
              f'Minutes: projected {prof["proj_min"]:.1f} (last 10 avg {prof["min_l10"]:.1f}, '
              f'{stab} at ±{prof["min_sd10"]:.1f}).'))
    r += matchup_reasons(prof, stat, side, cfg)
    if prof["b2b"]:
        r.append(("i", "Second night of a back-to-back (last season this didn't cut starters' minutes)."))
    if prof["new_team"]:
        r.append(("!", f'New team (was {prof["last_team"]}): role and minutes less certain.'))
    if prof["n_cur"] < 5:
        r.append(("i", f'Early season: {prof["n_cur"]} games this season, so last season carries more weight.'))
    for name, m in outs.get(prof["team"], []):
        if name != prof["player"]:
            r.append(("+" if up else "!",
                      f"Teammate {name} is OUT ({m:.0f} min/game): more shots and touches likely (not in the projection)."))
    if not injuries_ok:
        r.append(("!", "Injury report could not be loaded today: check status before tip-off."))
    return r


def matchup_net(prof, stat):
    m = prof.get("matchup_detail", {}).get(stat)
    return m["net"] if m else 0.0


def matchup_ok(prof, stat, side, cfg):
    """The matchup must lean the pick's way overall."""
    if stat not in prof.get("matchup_detail", {}):
        return True  # stats without matchup checks (steals, blocks...) aren't gated
    net = matchup_net(prof, stat)
    thr = cfg["matchups"]["min_net_effect"]
    return net > thr if side == "over" else net < -thr


def matchup_reasons(prof, stat, side, cfg=None):
    """One line per matchup check, strongest first, plus a one-line summary."""
    m = prof.get("matchup_detail", {}).get(stat)
    if not m:
        return []
    up = side == "over"
    ctx_min = (cfg or {}).get("matchups", {}).get("show_context_min_effect", 0.03)
    counted = [i for i in m["items"] if i["weight"]]
    ctx = [i for i in m["items"] if not i["weight"] and abs(np.exp(i["x"]) - 1) >= ctx_min and i["text"]]
    help_ = [i for i in counted if (i["effect"] > 0.005) == up and abs(i["effect"]) > 0.005]
    hurt = [i for i in counted if (i["effect"] > 0.005) != up and abs(i["effect"]) > 0.005]
    net = m["net"]
    lab = LABELS[stat].lower()
    out = [("+" if (net > 0) == up else "!",
            f'Matchup overall: {100*net:+.1f}% {lab} vs his usual output '
            f'({len(help_)} check{"s" if len(help_) != 1 else ""} for this pick, {len(hurt)} against).')]
    multi = len({i["stat"] for i in counted if i["stat"]}) > 1
    for i in sorted(counted, key=lambda i: -abs(i["effect"])):
        if abs(i["effect"]) < 0.005 or not i["text"]:
            continue
        tag = "+" if (i["effect"] > 0) == up else "!"
        comp = f'{LABELS[i["stat"]]} · ' if multi and i["stat"] else ""
        out.append((tag, f'{comp}{i["label"]}: {i["text"]} ({100*i["effect"]:+.1f}%).'))
    for i in ctx:
        out.append(("i", f'{i["label"]}: {i["text"]}. Checked, but it had no measurable effect last season, '
                         'so it is not counted.'))
    return out


def market_picks(profiles, lines, injuries, cfg):
    mc, outs = cfg["market_picks"], teammates_out(profiles, injuries)
    by_key = {}
    for p in profiles:
        by_key.setdefault(norm_name(p["player"]), []).append(p)
    picks, skipped = [], []
    for ln in lines:
        cands = [p for p in by_key.get(ln["player_key"], []) if p["team"] in (ln["home"], ln["away"])]
        if not cands:
            continue
        prof = cands[0]
        fails = general_gate(prof, injuries, cfg)
        if fails:
            skipped.append({"player": prof["player"], "stat": ln["stat"], "why": "; ".join(fails)})
            continue
        stat, line = ln["stat"], ln["line"]
        mean, disp = prof["means"][stat], prof["disp"][stat]
        vals = prof["history"][stat].to_numpy()
        for side in ("over", "under"):
            p_model = side_probability(mean, disp, line, side)
            fair = ln["fair_over"] if side == "over" else 1 - ln["fair_over"]
            w = mc["model_weight"]
            p = w * p_model + (1 - w) * fair
            price = ln[f"best_{side}_price"]
            book = ln[f"best_{side}_book"]
            edge = p - fair
            ev = p * american_to_decimal(price) - 1
            l10, l10n = hit_rate(vals[-10:], line, side)
            l20, l20n = hit_rate(vals[-20:], line, side)
            if (p < mc["min_prob"] or edge < mc["min_edge"] or ln["books"] < mc["min_books"]
                    or l10 is None or l10 < mc["min_l10_hit_rate"] or not matchup_ok(prof, stat, side, cfg)):
                continue
            lock = (p >= mc["lock_prob"] and (l20 or 0) >= mc["lock_l20_hit_rate"]
                    and edge >= mc["lock_edge"])
            reasons = [
                ("+", f'Projection: {mean:.1f} {LABELS[stat].lower()} vs line {line}. Model alone: '
                      f'{100*p_model:.0f}% for the {side}.'),
                ("+", f'Market: books price the {side} at {100*fair:.0f}% (no-vig). Blending model and market '
                      f'gives {100*p:.0f}%, an edge of +{100*edge:.1f} pts. Best price {_fmt_price(price)} at {book}.'),
                ("+" if l10 >= 0.7 else "i",
                 f'Recent form: went {side} {line} in {l10n} of last {min(10, len(vals))}, '
                 f'{l20n} of last {min(20, len(vals))}.'),
            ]
            reasons += common_reasons(prof, stat, injuries, outs, injuries is not None, side, cfg)
            picks.append({
                "kind": "market", "tier": "Lock" if lock else "Strong",
                "game_id": prof["game_id"], "matchup": prof["matchup"], "tipoff_utc": prof["tipoff_utc"],
                "player_id": prof["player_id"], "player": prof["player"], "team": prof["team"],
                "stat": stat, "stat_label": LABELS[stat], "side": side, "line": line,
                "label": f'{prof["player"]} {side.title()} {line} {LABELS[stat]}',
                "prob": p, "prob_model": p_model, "fair": fair, "edge": edge, "ev": ev, "price": price, "book": book,
                "proj": mean, "l10": l10, "l20": l20, "reasons": reasons,
                "matchup_net": matchup_net(prof, stat),
            })
    return picks, skipped


def safe_picks(profiles, injuries, cfg):
    sc, outs = cfg["safe_thresholds"], teammates_out(profiles, injuries)
    picks = []
    for prof in profiles:
        if general_gate(prof, injuries, cfg):
            continue
        for stat, kmin in sc["min_threshold"].items():
            mean, disp = prof["means"][stat], prof["disp"][stat]
            vals = prof["history"][stat].to_numpy()
            last20 = vals[-20:]
            best = None
            k = int(kmin)
            while True:
                p = prob_at_least(mean, disp, k)
                l20 = float((last20 >= k).mean())
                if p < sc["min_model_prob"] or l20 < sc["min_l20_hit_rate"]:
                    break
                best = (k, p, l20)
                k += 1
            if not best or not matchup_ok(prof, stat, "over", cfg):
                continue
            k, p, l20 = best
            l10n = int((vals[-10:] >= k).sum())
            l20n = int((last20 >= k).sum())
            season_rate = float((vals >= k).mean())
            reasons = [
                ("+", f'Projection: {mean:.1f} {LABELS[stat].lower()}; model gives {100*p:.0f}% for {k}+.'),
                ("+", f'Floor: reached {k}+ in {l20n} of last {len(last20)} and {l10n} of last {min(10, len(vals))} '
                      f'({100*season_rate:.0f}% across all {len(vals)} games tracked).'),
                ("i", f"Cushion: needs {k}, projected {mean:.1f} ({mean - k:+.1f}). "
                      "Sportsbooks sell this as an alt line at heavy juice."),
            ]
            reasons += common_reasons(prof, stat, injuries, outs, injuries is not None, "over", cfg)
            picks.append({
                "kind": "safe", "tier": "Safe",
                "game_id": prof["game_id"], "matchup": prof["matchup"], "tipoff_utc": prof["tipoff_utc"],
                "player_id": prof["player_id"], "player": prof["player"], "team": prof["team"],
                "stat": stat, "stat_label": LABELS[stat], "side": "over", "line": k - 0.5,
                "label": f'{prof["player"]} {k}+ {LABELS[stat]}',
                "prob": p, "proj": mean, "l20": l20, "l10": l10n / min(10, len(vals)),
                "cushion": (mean - k) / max(1.0, np.sqrt(mean * prof["disp"][stat])),
                "reasons": reasons, "matchup_net": matchup_net(prof, stat),
            })
    return picks
