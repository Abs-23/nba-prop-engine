"""Backtest the daily card at REAL DraftKings lines (closing lines collected from ESPN).

Train/test split by date: weights and the model+market blend are learned on games
before the split; the card is built and graded on games after it.

    python scripts/backtest_lines.py [--split 2026-02-01]
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
from engine.model import side_probability  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_blend(pm, pf, y):
    """Logistic regression: P(side wins) from logit(model) and logit(market). Returns coefs."""
    X = np.column_stack([np.ones(len(y)), logit(pm), logit(pf)])
    b = np.zeros(3)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + np.eye(3) * 1e-3
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def fit_side(c, model_w):
    """Over/under term (Oct 25): logit(p) = b0 + model_w*logit(model) + b2*logit(market) + lean*(+1 under / -1 over),
    with a separate lean for rebounds. Model weight held fixed. Returns [b0, b2, lean_other, lean_reb]."""
    u = np.where(c["side"] == "under", 1.0, -1.0)
    X = np.column_stack([np.ones(len(c)), logit(c["fair"].to_numpy()),
                         np.where(c["stat"] != "reb", u, 0.0), np.where(c["stat"] == "reb", u, 0.0)])
    off, y, b = model_w * logit(c["p_model"].to_numpy()), c["win"].astype(float).to_numpy(), np.zeros(4)
    for _ in range(80):
        p = 1 / (1 + np.exp(-(X @ b + off)))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + np.eye(4) * 1e-3, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def blend(b, pm, pf):
    return 1 / (1 + np.exp(-(b[0] + b[1] * logit(pm) + b[2] * logit(pf))))


def candidates(df, lines, w, disp0, l20):
    """One row per (player-game, stat, side) that has a DraftKings line."""
    means, base, nets = fit.project_frame(df, w)
    key = df[["game_id", "player_id"]].astype({"game_id": str, "player_id": int}).reset_index()
    m = lines.merge(key, on=["game_id", "player_id"])
    out = []
    for r in m.itertuples():
        i, s = r.index, r.stat
        if s not in fit.PARTS:
            continue
        row = df.iloc[i]
        mu = float(means[s][i])
        n = min(40, row["n_prev"])
        d = row[f"disp_{s}"] if pd.notna(row[f"disp_{s}"]) else disp0[s]
        disp = max(1.0, (n * d + 10 * disp0[s]) / (n + 10))
        actual = float(row[s])
        last = np.asarray(l20[s][int(row["row"])], dtype=float)
        for side in ("over", "under"):
            pmod = side_probability(mu, disp, r.line, side)
            fair = r.fair_over if side == "over" else 1 - r.fair_over
            price = r.over_price if side == "over" else r.under_price
            win = (actual > r.line) if side == "over" else (actual < r.line)
            hit10 = ((last[-10:] > r.line) if side == "over" else (last[-10:] < r.line)).mean() if len(last) else np.nan
            hit20 = ((last > r.line) if side == "over" else (last < r.line)).mean() if len(last) else np.nan
            lean = float(nets[s][i]) * (1 if side == "over" else -1)
            out.append({"date": row["date"], "game_id": r.game_id, "player_id": r.player_id, "player": row["player"],
                        "team": row["team"], "stat": s, "side": side, "line": r.line, "price": price,
                        "p_model": pmod, "fair": fair, "lean": lean, "hit10": hit10, "hit20": hit20,
                        "proj_min": row["proj_min"], "n_prev": row["n_prev"], "push": actual == r.line, "win": win})
    return pd.DataFrame(out)


def build_cards(c, p_min, form_min, lean_min, max_legs=6, min_legs=3, per_game=2):
    cards = []
    for day, g in c.sort_values("p", ascending=False).groupby("date"):
        q = g[(g["p"] >= p_min) & (g["hit10"] >= form_min) & (g["lean"] >= lean_min)]
        legs, players, games = [], set(), {}
        for r in q.itertuples():
            if r.player_id in players or games.get(r.game_id, 0) >= per_game:
                continue
            legs.append(r)
            players.add(r.player_id)
            games[r.game_id] = games.get(r.game_id, 0) + 1
            if len(legs) == max_legs:
                break
        if len(legs) >= min_legs:
            cards.append({"date": day, "size": len(legs), "win": all(l.win for l in legs),
                          "leg_hits": sum(l.win for l in legs), "p": float(np.prod([l.p for l in legs])),
                          "legs": [(l.player, l.stat, l.side, l.line, round(l.p, 3), bool(l.win)) for l in legs]})
    return pd.DataFrame(cards)


def main(split):
    cfg = load_config()
    lines = pd.read_csv(DATA / "props_history.csv.gz", dtype={"game_id": str})
    cache = DATA.parent / ".cache_frame.pkl"
    df = pickle.load(open(cache, "rb")) if cache.exists() else fit.build_frame(
        database.load(), database.load_games(), database.load_players())
    df = df.reset_index(drop=True)
    l20 = df.attrs.get("l20")
    df.attrs = {}
    disp0 = {s: float(np.nanmedian(df.loc[df["n_prev"] >= 20, f"disp_{s}"])) for s in fit.PARTS}
    w = fit.fit_weights(df[df["date"] < split], cfg["matchups"]["fit_min_t"])  # matchup weights: before split only
    g = cfg["gates"]
    elig = df[(df["n_prev"] >= g["min_games_sample"]) & (df["proj_min"] >= g["min_projected_minutes"])]
    elig = elig.reset_index(drop=True)  # positions must line up with the projection arrays
    c = candidates(elig, lines, w, disp0, l20)
    c = c[~c["push"]].reset_index(drop=True)
    # Blend model + market with 2-fold cross-fitting by day: each day is scored by a blend
    # learned only on the OTHER half of the days (lines are scarce, so we can't hold out a block).
    days = sorted(c["date"].unique())
    fold = c["date"].map({d: i % 2 for i, d in enumerate(days)})
    c["p"] = np.nan
    bs = []
    for k in (0, 1):
        tr = c[fold != k]
        b = fit_blend(tr["p_model"].to_numpy(), tr["fair"].to_numpy(), tr["win"].astype(float).to_numpy())
        bs.append(b)
        msk = fold == k
        c.loc[msk, "p"] = blend(b, c.loc[msk, "p_model"].to_numpy(), c.loc[msk, "fair"].to_numpy())
    b = np.mean(bs, axis=0)
    te = c
    print(f"Lines: {lines['game_id'].nunique()} games with DraftKings lines over {te['date'].nunique()} days, "
          f"{len(te)//2:,} player-lines (matchup weights learned before {split}).")
    print("Model alone, calibration:")
    mb = pd.cut(te["p_model"], [0, .4, .5, .6, .7, .8, 1])
    print(te.groupby(mb, observed=True).agg(n=("win", "size"), predicted=("p_model", "mean"),
                                            actual=("win", "mean")).round(3).to_string())
    print(f"Blend learned on training games: logit(p) = {b[0]:+.2f} + {b[1]:.2f} x logit(model) + {b[2]:.2f} x logit(market)")
    sb = fit_side(c, b[1])
    print(f"Over/under term (model weight fixed at {b[1]:.3f}): market {sb[1]:.3f}, under lean {sb[2]:+.3f} "
          f"(rebounds {sb[3]:+.3f}). Unders hit {c.loc[c.side == 'under', 'win'].mean():.1%} vs DraftKings "
          f"{c.loc[c.side == 'under', 'fair'].mean():.1%}.")
    print("\nCalibration on test lines (all sides):")
    te = te.assign(bucket=pd.cut(te["p"], [0, .45, .5, .55, .6, .65, .7, .8, 1]))
    print(te.groupby("bucket", observed=True).agg(n=("win", "size"), predicted=("p", "mean"),
                                                  actual=("win", "mean")).round(3).to_string())
    print("\nCards (min 3, max 6 legs; one leg per player, <=2 per game):")
    res = []
    for p_min in (0.56, 0.58, 0.60, 0.62, 0.65):
        for form in (0.0, 0.6, 0.7):
            for lean in (-1.0, 0.0, 0.01):
                q = te[(te["p"] >= p_min) & (te["hit10"] >= form) & (te["lean"] >= lean)]
                cards = build_cards(te, p_min, form, lean)
                days = te["date"].nunique()
                res.append({"p_min": p_min, "form10": form, "lean": lean, "legs_per_day": len(q) / days,
                            "leg_hit": q["win"].mean() if len(q) else np.nan,
                            "card_days": len(cards) / days, "avg_size": cards["size"].mean() if len(cards) else 0,
                            "card_win": cards["win"].mean() if len(cards) else np.nan,
                            "card_pred": cards["p"].mean() if len(cards) else np.nan,
                            "card_legs_hit": (cards["leg_hits"].sum() / cards["size"].sum()) if len(cards) else np.nan})
    r = pd.DataFrame(res)
    print(r.round(3).to_string(index=False))
    (DATA / "backtest_lines.json").write_text(json.dumps({"split": split, "blend": list(map(float, b)),
                                                         "grid": r.round(4).to_dict("records")}, indent=1))
    return c, r, b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2026-03-01")
    main(ap.parse_args().split)
