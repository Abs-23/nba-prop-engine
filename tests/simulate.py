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
from engine.sources import nba_cdn, odds as odds_mod  # noqa: E402
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
            rows.append(dict(game_id=gid, date=day, team=side, opp=opp, home=side == home, player_id=p["pid"],
                             player=p["name"], starter=p["mins"] > 26, played=m > 0, min=round(m, 2),
                             pts=2 * fgm + fg3m + ftm, reb=rng.poisson(r["reb"] * m * f), oreb=0,
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
for f in [database.PATH, grading.PICKS, grading.LINEUPS]:
    if f.exists():
        f.unlink()
database.save(pd.DataFrame(rows))
database.backfill_season = lambda db, prefix, max_games=1230: (db, 0)


def fake_update(db, schedule, today):
    have = set(db["game_id"])
    new = [r for g in cur_sched if g["date"] < today and g["game_id"] not in have for r in BOX[g["game_id"]]]
    return (pd.concat([db, pd.DataFrame(new)], ignore_index=True) if new else db), len({r["game_id"] for r in new})


database.update_from_schedule = fake_update
nba_cdn.fetch_schedule = lambda: cur_sched
import engine.sources.rosters as ros  # noqa: E402

ros.fetch_rosters = lambda: {norm_name(p["name"]): p["team"] for p in players}
run_daily.fetch_injuries = lambda: {norm_name("BOS Alpha"): {"name": "BOS Alpha", "status": "out", "detail": ""},
                                    norm_name("LAL Bravo"): {"name": "LAL Bravo", "status": "questionable", "detail": ""}}


class FakeOdds(odds_mod.OddsClient):
    """Sportsbook that knows the TRUE means and posts lines at them with -110/-110."""
    def __init__(self, *a, **k):
        super().__init__(api_key="fake", regions="us")
        self.remaining = 20000

    def events(self):
        return [{"id": g["game_id"], "home_team": INV_TEAM[g["home"]], "away_team": INV_TEAM[g["away"]]}
                for g in cur_sched if g["date"] == self.today]

    def event_props(self, event_id, markets):
        g = next(x for x in cur_sched if x["game_id"] == event_id)
        outs = []
        for side, opp in ((g["home"], g["away"]), (g["away"], g["home"])):
            for p in [x for x in players if x["team"] == side and x["mins"] > 20]:
                m, f, r = p["mins"], TEAM_DEF[opp], p["rate"]
                true = {"player_points": (2 * 0.47 * r["fga"] + 0.36 * r["fg3a"] + 0.78 * r["fta"]) * m * f,
                        "player_rebounds": r["reb"] * m * f, "player_assists": r["ast"] * m * f}
                for mk, mu in true.items():
                    if mk not in markets:
                        continue
                    line = np.floor(mu) + 0.5
                    outs.append((mk, p["name"], line))
        bms = []
        for book in ("BookA", "BookB", "BookC"):
            mkts = {}
            for mk, name, line in outs:
                mkts.setdefault(mk, []).extend([
                    {"name": "Over", "description": name, "price": -110, "point": line},
                    {"name": "Under", "description": name, "price": -110, "point": line}])
            bms.append({"title": book, "markets": [{"key": k, "outcomes": v} for k, v in mkts.items()]})
        return {"home_team": INV_TEAM[g["home"]], "away_team": INV_TEAM[g["away"]], "bookmakers": bms}


# ---------------------------------------------------------------- run 14 "days"
days = sorted({g["date"] for g in cur_sched})
for d in days[:-1]:
    for g in cur_sched:
        g["status"] = 3 if g["date"] < d else 1
    FakeOdds.today = d
    run_daily.OddsClient = FakeOdds
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
    print(f"lineup {fl:7s} {size}-leg n={len(s):3d} predicted {s.prob.mean():.3f} actual {(s.result == 'win').mean():.3f}")
print(ph.groupby("stat").size().to_string())
