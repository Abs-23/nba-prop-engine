"""Backtest: would the matchup factors have made last season's picks better?

Weights are learned on games BEFORE the split date and tested on games AFTER it,
so the test games were never seen while fitting (no peeking).

Compares, on the same test games and the same "Safe pick" rule (model >= 85%
and cleared in >= 17 of last 20):
  OLD    player's own numbers x minutes x team-level opponent adjustment (current engine)
  NEW    + per-stat matchup factors with learned weights + blowout minutes
  NEW+A  NEW, and the matchup factors must agree (several support, none strongly against)

Run:  python scripts/backtest.py [--split 2026-02-01]
"""
import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from engine import DATA, database, fit, load_config  # noqa: E402
from engine.matchups import FACTORS, POS_STATS, team_asof, team_games  # noqa: E402
from engine.model import prob_at_least  # noqa: E402

SAFE = ["pts", "reb", "ast", "fg3m", "fg3a", "fga", "fgm", "ftm", "fta", "pra", "pr", "pa", "ra"]
COMP = {**{s: [s] for s in POS_STATS}, **fit.COMBOS}


def old_factors(logs, players):
    """Team-level 'stat allowed' adjustment, as the current engine does it (as of each game)."""
    tg = team_games(logs, players)
    for s in POS_STATS:
        tg[f"allowed_{s}"] = tg[f"o_{s}"]
    ta = team_asof(tg, [f"allowed_{s}" for s in POS_STATS])
    return ta[["game_id", "team"] + [f"allowed_{s}" for s in POS_STATS]].rename(columns={"team": "opp"})


def contributions(row, stat, w):
    """{factor: % effect on this stat}, combining components for combo stats."""
    parts = COMP[stat]
    means = {c: row[f"rate_{c}"] for c in parts}
    tot = sum(means.values()) or 1.0
    out = {}
    for c in parts:
        x = fit.feature_matrix(row.to_frame().T, c)[0]
        for fac, xv in zip(FACTORS[c], x):
            wt = w["stats"][c]["factors"][fac]["weight"]
            out[fac] = out.get(fac, 0.0) + (means[c] / tot) * (np.exp(wt * float(xv)) - 1)
    return out


def agree(contrib, min_support=2, step=0.015, max_against=-0.03):
    sup = [v for v in contrib.values() if v >= step]
    against = [v for v in contrib.values() if v <= -step]
    return len(sup) >= min_support and min(against, default=0) > max_against and sum(contrib.values()) > 0


def main(split):
    cfg = load_config()
    cache = DATA.parent / ".cache_frame.pkl"
    logs, games, players = database.load(), database.load_games(), database.load_players()
    if cache.exists():
        df = pickle.load(open(cache, "rb"))
    else:
        df = fit.build_frame(logs, games, players)
        pickle.dump(df, open(cache, "wb"))
    df = df.merge(old_factors(logs, players), on=["game_id", "opp"], how="left")
    l20 = df.attrs.get("l20") or pickle.load(open(cache, "rb")).attrs["l20"]

    train = df[df["date"] < split]
    w = fit.fit_weights(train)
    test = df[(df["date"] >= split) & (df["date"] < "2026-07-01")]
    test = test[(test["n_prev"] >= cfg["gates"]["min_games_sample"]) & (test["proj_min"] >= cfg["gates"]["min_projected_minutes"])]

    disp0 = {s: float(np.nanmedian(df.loc[df["n_prev"] >= 20, f"disp_{s}"])) for s in SAFE}
    mins = cfg["safe_thresholds"]["min_threshold"]
    mm = w["minutes"]
    test = test.reset_index(drop=True)
    pm = test["proj_min"].to_numpy()
    pm_new = pm + mm["blowout_per_point"] * test["blowout"].to_numpy() * (pm / 36)
    # per base stat: old/new means and each factor's % effect
    base_old, base_new, base_contrib = {}, {}, {}
    for c in POS_STATS:
        rate = test[f"rate_{c}"].to_numpy()
        X = fit.feature_matrix(test, c)
        wts = np.array([w["stats"][c]["factors"][f]["weight"] for f in FACTORS[c]])
        base_old[c] = rate * pm * np.exp(np.nan_to_num(test[f"allowed_{c}"].to_numpy()))
        base_new[c] = rate * pm_new * np.exp(X @ wts)
        base_contrib[c] = {f: np.exp(wts[i] * X[:, i]) - 1 for i, f in enumerate(FACTORS[c])}
    res = []
    rows_idx = test["row"].astype(int).to_numpy()
    for s in SAFE:
        parts = COMP[s]
        m_old = sum(base_old[c] for c in parts)
        m_new = sum(base_new[c] for c in parts)
        share = {c: test[f"rate_{c}"].to_numpy() / sum(test[f"rate_{x}"].to_numpy() for x in parts) for c in parts}
        contrib = {}
        for c in parts:
            for f, v in base_contrib[c].items():
                contrib[f] = contrib.get(f, 0) + share[c] * v
        cm = np.column_stack(list(contrib.values()))
        support = (cm >= 0.015).sum(1)
        worst = cm.min(1)
        agree_ok = (support >= 2) & (worst > -0.03) & (cm.sum(1) > 0)
        n = np.minimum(40, test["n_prev"].to_numpy())
        d = test[f"disp_{s}"].fillna(disp0[s]).to_numpy()
        disp = np.maximum(1.0, (n * d + 10 * disp0[s]) / (n + 10))
        L = np.full((len(test), 20), np.nan)
        for i, ri in enumerate(rows_idx):
            v = np.asarray(l20[s][ri], dtype=float)
            L[i, :len(v)] = v
        cnt = (~np.isnan(L)).sum(1)
        actual = test[s].to_numpy()
        kmin = int(mins[s])
        for label, mean in (("OLD", m_old), ("NEW", m_new)):
            best_k = np.full(len(test), -1)
            best_p = np.zeros(len(test))
            alive = np.ones(len(test), bool)
            for k in range(kmin, kmin + 60):
                p = np.array([prob_at_least(mu, dd, k) if al else 0.0 for mu, dd, al in zip(mean, disp, alive)])
                l20r = np.where(cnt > 0, (L >= k).sum(1) / np.maximum(cnt, 1), 0)
                ok = alive & (p >= 0.85) & (l20r >= 0.85)
                best_k = np.where(ok, k, best_k)
                best_p = np.where(ok, p, best_p)
                alive = ok
                if not alive.any():
                    break
            sel = best_k >= 0
            for i in np.where(sel)[0]:
                rec = {"model": label, "date": test.at[i, "date"], "player": test.at[i, "player"], "stat": s,
                       "k": int(best_k[i]), "prob": float(best_p[i]), "hit": bool(actual[i] >= best_k[i]),
                       "game_id": test.at[i, "game_id"]}
                rec["effect"] = float(m_new[i] / m_old[i] - 1) if m_old[i] > 0 else 0.0
                rec["support"] = int(support[i])
                rec["worst"] = float(worst[i])
                rec["net"] = float(cm[i].sum())
                if label == "NEW":
                    rec["agree"] = bool(agree_ok[i])
                res.append(rec)
    out = pd.DataFrame(res)
    days = test["date"].nunique()
    print(f"\nTrain: games before {split} ({len(train):,} player-games). "
          f"Test: {days} days after it ({len(test):,} player-games).\n")
    rows = []
    for label, sub in (("OLD", out[out.model == "OLD"]), ("NEW", out[out.model == "NEW"]),
                       ("NEW+A", out[(out.model == "NEW") & out["agree"].fillna(False)])):
        rows.append({"model": label, "picks": len(sub), "per_day": len(sub) / days,
                     "predicted": sub["prob"].mean(), "hit_rate": sub["hit"].mean()})
    summary = pd.DataFrame(rows)
    print(summary.round(3).to_string(index=False))
    print("\nHit rate by stat (NEW+A vs OLD):")
    a = out[(out.model == "NEW") & out["agree"].fillna(False)].groupby("stat")["hit"].agg(["size", "mean"])
    o = out[out.model == "OLD"].groupby("stat")["hit"].agg(["size", "mean"])
    print(a.join(o, lsuffix="_new", rsuffix="_old").round(3).to_string())
    print("\nLearned weights (fit on training games):")
    for s, v in w["stats"].items():
        print(f"  {s:5s} " + "  ".join(f"{f}={d['weight']:.2f}" for f, d in v["factors"].items()))
    (DATA / "backtest.json").write_text(json.dumps({
        "split": split, "test_days": int(days), "summary": summary.round(4).to_dict("records")}, indent=1))
    return out, w


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2026-02-01")
    main(ap.parse_args().split)
