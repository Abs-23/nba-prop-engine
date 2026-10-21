"""Build 3-, 4-, 5- and 6-leg lineups from the qualified picks.

Rules (and why):
  * one leg per player       - two legs on the same player are the same bet twice
  * <= 1 leg per team        - teammates' stats move together (pace, blowouts)
  * <= 2 legs per game       - both sides of one game share the same script
  * no forced legs           - if only 4 picks qualify, there is no 5- or 6-leg lineup

Combined probability multiplies the legs. It's the honest number: six
85% legs still lose more often than you'd guess (0.85^6 = 38%).
"""
import math


def build(picks, sizes, max_per_game, max_per_team, key):
    ranked = sorted(picks, key=key, reverse=True)
    chosen, players, per_game, per_team = [], set(), {}, {}
    for p in ranked:
        if p["player_id"] in players:
            continue
        if per_game.get(p["game_id"], 0) >= max_per_game or per_team.get(p["team"], 0) >= max_per_team:
            continue
        chosen.append(p)
        players.add(p["player_id"])
        per_game[p["game_id"]] = per_game.get(p["game_id"], 0) + 1
        per_team[p["team"]] = per_team.get(p["team"], 0) + 1
        if len(chosen) == max(sizes):
            break
    out = []
    for n in sizes:
        if len(chosen) >= n:
            legs = chosen[:n]
            out.append({"size": n, "legs": legs, "prob": math.prod(l["prob"] for l in legs)})
    return out


def build_all(safe, market, cfg):
    lc = cfg["lineups"]
    return {
        "safest": build(safe, lc["sizes"], lc["max_legs_per_game"], lc["max_legs_per_team"],
                        key=lambda p: (round(p["prob"], 3), p.get("cushion", 0))),
        "value": build(market, lc["sizes"], lc["max_legs_per_game"], lc["max_legs_per_team"],
                       key=lambda p: p["ev"]),
    }
