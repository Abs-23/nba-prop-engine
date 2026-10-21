"""End-to-end test on a simulated league (no internet needed).

Builds a fake season of box scores from players with KNOWN true talent,
fakes the schedule / rosters / injuries / sportsbook lines, runs the real
daily pipeline over several days, and then checks:
  * the pipeline runs and writes the dashboard
  * grading works
  * probabilities are calibrated (the "85%" picks hit ~85% of the time)

Run:  python tests/simulate.py
"""
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_daily  # noqa: E402
from engine import database, grading  # noqa: E402
from engine import DATA  # noqa: E402
from engine.sources import espn  # noqa: E402
from engine.stats import TEAM_NAMES, norm_name  # noqa: E402

rng = np.random.default_rng(7)
random.seed(7)
TEAMS = sorted(set(TEAM_NAMES.values()))
INV_TEAM = {v: k for k, v in TEAM_NAMES.items() if k != "LA Clippers"}

# ---------------------------------------------------------------- fake league
players = []
pid = 1000
NATO = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet", "Kilo",
        "Lima", "Mike"]
for t in TEAMS:
    defense = rng.normal(1.0, 0.04)  # team allows this multiple of everything
    for slot in range(13):
        mins = [36, 34, 32, 30, 28, 24, 20, 16, 12, 8, 4, 2, 1][slot] + rng.normal(0, 2)
        scorer = rng.uniform(0.35, 0.85)
        players.append(dict(
            pid=pid, name=f"{t} {NATO[slot]}", team=t, mins=max(1, mins), defense=defense,
            pos="GGFFC"[slot % 5],
            rate=dict(fga=scorer * 0.55, fg3a=rng.uniform(0.05, 0.25), fta=rng.uniform(0.05, 0.3),
                      reb=rng.uniform(0.12, 0.40), ast=rng.uniform(0.04, 0.28), stl=0.03, blk=0.03, tov=0.06),
        ))
        pid += 1
TEAM_DEF = {p["team"]: p["defense"] for p in players}


def sim_game(gid, day, home, away):
    rows = []
    for side, opp in ((home, away), (away, home)):
        f = TEAM_DEF[opp]
        for p in [x for x in players if x["team"] == side]:
            m = float(np.clip(rng.normal(p["mins"], 3.0), 0, 48))
            r = p["rate"]
            fga = rng.poisson(r["fga"] * m * f)
            fg3a = min(fga, rng.poisson(r["fg3a"] * m * f))
            fgm = rng.binomial(fga, 0.47)
            fg3m = min(fgm, rng.binomial(fg3a, 0.36))
            fta = rng.poisson(r["fta"] * m * f)
            ftm = rng.binomial(fta, 0.78)
            reb = rng.poisson(r["reb"] * m * f)
            oreb = rng.binomial(reb, 0.25)
            rows.append(dict(game_id=gid, date=day, team=side, opp=opp, home=side == home, player_id=p["pid"],
                             player=p["name"], pos=p["pos"], starter=p["mins"] > 26, played=m > 0, min=round(m, 2),
                             pts=2 * fgm + fg3m + ftm, reb=reb, oreb=oreb, dreb=reb - oreb,
                             pf=rng.poisson(0.07 * m),
                             ast=rng.poisson(r["ast"] * m * f), stl=rng.poisson(r["stl"] * m),
                             blk=rng.poisson(r["blk"] * m), tov=rng.poisson(r["tov"] * m),
                             fgm=fgm, fga=fga, fg3m=fg3m, fg3a=fg3a, ftm=ftm, fta=fta))
    return rows


def make_schedule(prefix, start, n_days):
    sched, n = [], 1
    for d in range(n_days):
        day = (start + timedelta(days=d)).isoformat()
        teams = random.sample(TEAMS, 14)
        for i in range(0, 14, 2):
            sched.append(dict(game_id=f"002{prefix}{n:05d}", date=day, tipoff_utc=f"{day}T23:30:00Z", status=3,
                              home=teams[i], away=teams[i + 1], home_id=0, away_id=0))
            n += 1
    return sched


prior_sched = make_schedule("25", date(2025, 10, 21), 170)  # ~1190 games
cur_sched = make_schedule("26", date(2026, 10, 20), 30)
rows = []
for g in prior_sched:
    rows += sim_game(g["game_id"], g["date"], g["home"], g["away"])
BOX = {}
for g in cur_sched:
    BOX[g["game_id"]] = sim_game(g["game_id"], g["date"], g["home"], g["away"])

# ---------------------------------------------------------------- fakes
for f in [database.PATH, grading.PICKS, grading.LINEUPS, DATA / "backfill_2025.done", database.GAMES,
          database.PLAYERS, DATA / "weights.json"]:
    if f.exists():
        f.unlink()
database.save(pd.DataFrame(rows))
pd.DataFrame([dict(game_id=g["game_id"], date=g["date"], home=g["home"], away=g["away"], spread=rng.normal(0, 6),
                   total=rng.normal(226, 8), home_score=None, away_score=None)
              for g in prior_sched + cur_sched]).to_csv(database.GAMES, index=False)
pd.DataFrame([dict(player_id=p["pid"], player=p["name"], team=p["team"], height=76 + 3 * "GFC".index(p["pos"]),
                   pos=p["pos"]) for p in players]).to_csv(database.PLAYERS, index=False)
(DATA / "backfill_2025.done").write_text("sim")
for g in cur_sched:
    g["completed"] = True
    g["spread"], g["total"] = None, None


def fake_boxes(game_ids, workers=8):
    return [r for gid in game_ids for r in BOX[gid]], 0


database.fetch_boxes = fake_boxes
espn.games_between = lambda start, end: [g for g in cur_sched if start <= g["date"] <= end]
import engine.sources.rosters as ros  # noqa: E402

ros.fetch_rosters = lambda players_out=None: {norm_name(p["name"]): p["team"] for p in players}
run_daily.fetch_injuries = lambda: {norm_name("BOS Alpha"): {"name": "BOS Alpha", "status": "out", "detail": ""},
                                    norm_name("LAL Bravo"): {"name": "LAL Bravo", "status": "questionable", "detail": ""}}


def _american(p):
    p = min(max(p, 0.02), 0.98)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def fake_lines(event_ids, workers=6):
    """A sharp sportsbook: knows each player's TRUE mean. Like real books it posts lines a little
    off center and shades the PRICE to the true probability (plus ~4.5% margin)."""
    from scipy.stats import poisson
    rows = []
    for g in [x for x in cur_sched if x["game_id"] in set(event_ids)]:
        for side, opp in ((g["home"], g["away"]), (g["away"], g["home"])):
            for p in [x for x in players if x["team"] == side and x["mins"] > 20]:
                m, f, r = p["mins"], TEAM_DEF[opp], p["rate"]
                pts = (2 * 0.47 * r["fga"] + 0.36 * r["fg3a"] + 0.78 * r["fta"]) * m * f
                true = {"pts": pts, "reb": r["reb"] * m * f, "ast": r["ast"] * m * f,
                        "fg3m": 0.36 * r["fg3a"] * m * f}
                true["pra"] = true["pts"] + true["reb"] + true["ast"]
                for stat, mu in true.items():
                    line = float(np.floor(mu + rng.normal(0, 0.12 * mu ** 0.5 + 0.4)) + 0.5)
                    po = float(poisson.sf(np.floor(line), mu))
                    rows.append({"game_id": g["game_id"], "player_id": p["pid"], "stat": stat, "line": line,
                                 "over_price": _american(po * 1.045), "under_price": _american((1 - po) * 1.045),
                                 "fair_over": po, "open_line": line, "book": "DraftKings"})
    return rows


import engine.sources.espn_props as espn_props  # noqa: E402
espn_props.fetch_lines = fake_lines


# ---------------------------------------------------------------- run 14 "days"
days = sorted({g["date"] for g in cur_sched})
for d in days[-7:-1]:  # last 6 days (each day re-learns weights, ~1 min)
    run_daily.main(d)

# ---------------------------------------------------------------- checks
ph = pd.read_csv(grading.PICKS)
lh = pd.read_csv(grading.LINEUPS)
g = ph[ph["result"].isin(["hit", "miss"])]
print("\n==== RESULTS ====")
print(f"picks recorded {len(ph)}, graded {len(g)}, void {(ph.result == 'void').sum()}")
for (kind, tier), s in g.groupby(["kind", "tier"]):
    print(f"{kind:7s} {tier:7s} n={len(s):5d}  predicted {s.prob.mean():.3f}  actual {(s.result == 'hit').mean():.3f}")
for (fl, size), s in lh[lh.result.isin(['win', 'loss'])].groupby(["flavor", "size"]):
    print(f"{fl} {size}-leg n={len(s):3d} predicted {s.prob.mean():.3f} actual {(s.result == 'win').mean():.3f}")
print(ph.groupby("stat").size().to_string())
