"""Keep score. Every pick and lineup is saved, then graded once the box score is in.

data/picks_history.csv     one row per pick
data/lineups_history.csv   one row per lineup (legs stored as pick ids)

A player who doesn't play is a VOID (no win, no loss), same as a sportsbook.
"""
import json

import numpy as np
import pandas as pd

from engine import DATA
from engine.stats import add_stat_columns

PICKS = DATA / "picks_history.csv"
LINEUPS = DATA / "lineups_history.csv"
PICK_COLS = ["pick_id", "date", "kind", "tier", "game_id", "player_id", "player", "team", "stat",
             "side", "line", "prob", "fair", "edge", "price", "result", "actual"]


def pick_id(date, p):
    return f'{date}|{p["player_id"]}|{p["stat"]}|{p["side"]}|{p["line"]}|{p["kind"]}'


def _load(path, cols):
    if path.exists():
        return pd.read_csv(path, dtype={"game_id": str})
    return pd.DataFrame(columns=cols)


def record_today(date, safe, market, lineups):
    """Save today's picks and lineups. Re-running the same day replaces that day."""
    ph = _load(PICKS, PICK_COLS)
    ph = ph[ph["date"] != date]
    rows = []
    for p in safe + market:
        rows.append({"pick_id": pick_id(date, p), "date": date, "kind": p["kind"], "tier": p["tier"],
                     "game_id": p["game_id"], "player_id": p["player_id"], "player": p["player"],
                     "team": p["team"], "stat": p["stat"], "side": p["side"], "line": p["line"],
                     "prob": round(p["prob"], 4), "fair": round(p.get("fair", np.nan), 4),
                     "edge": round(p.get("edge", np.nan), 4), "price": p.get("price"),
                     "result": "pending", "actual": None})
    ph = pd.concat([ph, pd.DataFrame(rows, columns=PICK_COLS)], ignore_index=True)
    ph.to_csv(PICKS, index=False)

    lh = _load(LINEUPS, ["date", "flavor", "size", "prob", "legs", "result"])
    lh = lh[lh["date"] != date]
    lrows = [{"date": date, "flavor": flavor, "size": lu["size"], "prob": round(lu["prob"], 4),
              "legs": json.dumps([pick_id(date, l) for l in lu["legs"]]), "result": "pending"}
             for flavor, lus in lineups.items() for lu in lus]
    lh = pd.concat([lh, pd.DataFrame(lrows)], ignore_index=True)
    lh.to_csv(LINEUPS, index=False)


def grade(logs):
    """Grade every pending pick whose game is in the database."""
    ph = _load(PICKS, PICK_COLS)
    if ph.empty:
        return
    logs = add_stat_columns(logs.copy())
    finished = set(logs["game_id"].astype(str))
    idx = {(str(r.game_id), int(r.player_id)): r for r in logs.itertuples()}
    for i, r in ph[ph["result"] == "pending"].iterrows():
        gid = str(r["game_id"])
        if gid not in finished:
            continue
        row = idx.get((gid, int(r["player_id"])))
        if row is None or not row.played or row.min == 0:
            ph.at[i, "result"] = "void"
            continue
        actual = getattr(row, r["stat"])
        ph.at[i, "actual"] = actual
        if actual == r["line"]:
            ph.at[i, "result"] = "push"
        elif (actual > r["line"]) == (r["side"] == "over"):
            ph.at[i, "result"] = "hit"
        else:
            ph.at[i, "result"] = "miss"
    ph.to_csv(PICKS, index=False)

    lh = _load(LINEUPS, ["date", "flavor", "size", "prob", "legs", "result"])
    res = dict(zip(ph["pick_id"], ph["result"]))
    for i, r in lh[lh["result"] == "pending"].iterrows():
        legs = [res.get(l, "pending") for l in json.loads(r["legs"])]
        if "miss" in legs:
            lh.at[i, "result"] = "loss"
        elif "pending" in legs:
            continue
        elif all(l in ("void", "push") for l in legs):
            lh.at[i, "result"] = "void"
        else:
            lh.at[i, "result"] = "win"
    lh.to_csv(LINEUPS, index=False)


def track_record():
    """Summary numbers for the dashboard."""
    ph, lh = _load(PICKS, PICK_COLS), _load(LINEUPS, ["date", "flavor", "size", "prob", "legs", "result"])
    g = ph[ph["result"].isin(["hit", "miss"])].copy()
    out = {"graded_picks": int(len(g)), "by_tier": [], "by_stat": [], "calibration": [], "lineups": [],
           "recent": []}
    if len(g):
        g["hit"] = (g["result"] == "hit").astype(int)
        for tier, s in g.groupby("tier"):
            out["by_tier"].append({"tier": tier, "n": int(len(s)), "hits": int(s["hit"].sum()),
                                   "rate": float(s["hit"].mean()), "expected": float(s["prob"].mean())})
        for stat, s in g.groupby("stat"):
            out["by_stat"].append({"stat": stat, "n": int(len(s)), "rate": float(s["hit"].mean())})
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        g["bucket"] = pd.cut(g["prob"], bins, include_lowest=True)
        for b, s in g.groupby("bucket", observed=True):
            out["calibration"].append({"bucket": f"{int(b.left*100)}-{int(b.right*100)}%", "n": int(len(s)),
                                       "predicted": float(s["prob"].mean()), "actual": float(s["hit"].mean())})
        rec = g.sort_values("date").tail(30)
        out["recent"] = rec[["date", "player", "stat", "side", "line", "actual", "result", "tier"]].iloc[::-1] \
            .to_dict("records")
    gl = lh[lh["result"].isin(["win", "loss"])]
    for (flavor, size), s in gl.groupby(["flavor", "size"]):
        out["lineups"].append({"flavor": flavor, "size": int(size), "n": int(len(s)),
                               "wins": int((s["result"] == "win").sum()), "expected": float(s["prob"].mean())})
    return out
