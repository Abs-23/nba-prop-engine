"""Learn how much each matchup factor matters, from real games.

For every player-game we know (a) what the player usually produces per minute
before that game, (b) the minutes he actually played, (c) the matchup factors
before that game, and (d) what he actually did. A Poisson regression then
answers: "after accounting for the player's own level, how much does each
factor move the result?" That number is the factor's weight.

  weight 1.0 -> the factor's full effect shows up in the box score
  weight 0.0 -> the factor doesn't predict anything; it's ignored

Minutes (blowouts from the spread, back-to-backs) get their own small model.
"""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from engine import DATA
from engine.database import season_key
from engine.matchups import (FACTORS, POS_STATS, game_features, pos_asof, pos_table, team_asof,
                             team_games)

WEIGHTS = DATA / "weights.json"
BASE = POS_STATS
COMBOS = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}


# ------------------------------------------------------------------ player history, as of each game

def player_asof(logs, rate_half=15, min_half=5, prior_w=0.5, disp_window=40):
    """Per player-game: per-minute rates, projected minutes, recent values, dispersion,
    computed only from that player's EARLIER games."""
    df = logs[logs["played"].astype(bool) & (logs["min"] > 0)].copy()
    df["season"] = df["date"].astype(str).map(season_key)
    for c, parts in COMBOS.items():
        df[c] = df[parts].sum(axis=1)
    stats = BASE + list(COMBOS)
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    dr, dm = 0.5 ** (1 / rate_half), 0.5 ** (1 / min_half)
    out = {k: np.full(len(df), np.nan) for k in ["proj_min", "n_prev"] + [f"rate_{s}" for s in stats]
           + [f"disp_{s}" for s in stats] + [f"l20_{s}" for s in stats]}
    l20_vals = {s: [None] * len(df) for s in stats}
    for _, idx in df.groupby("player_id").indices.items():
        sub = df.iloc[idx]
        mins = sub["min"].to_numpy()
        seas = sub["season"].to_numpy()
        vals = {s: sub[s].to_numpy(dtype=float) for s in stats}
        for i in range(1, len(idx)):
            age = np.arange(i)[::-1]
            sw = np.where(seas[:i] == seas[i], 1.0, prior_w)
            wr = dr ** age * sw
            wm = dm ** age * sw
            k = idx[i]
            out["n_prev"][k] = i
            out["proj_min"][k] = (wm * mins[:i]).sum() / wm.sum()
            denom = (wr * mins[:i]).sum()
            for s in stats:
                v = vals[s][:i]
                out[f"rate_{s}"][k] = (wr * v).sum() / denom
                w = v[-disp_window:]
                out[f"disp_{s}"][k] = w.var(ddof=1) / w.mean() if len(w) > 2 and w.mean() > 0 else np.nan
                l20_vals[s][k] = v[-20:]
    for key, arr in out.items():
        df[key] = arr
    df.attrs["l20"] = l20_vals
    return df


def back_to_back(logs):
    """{(team, date): True} when the team also played the day before."""
    td = logs[["team", "date"]].drop_duplicates()
    have = set(zip(td["team"], td["date"].astype(str)))
    return {(t, d): (t, (date.fromisoformat(d) - timedelta(days=1)).isoformat()) in have
            for t, d in have}


# ------------------------------------------------------------------ training frame

def build_frame(logs, games, players):
    """One row per player-game with everything known before tip-off, plus the result."""
    tg = team_games(logs, players)
    ta = team_asof(tg)
    pa = pos_asof(pos_table(logs))
    df = player_asof(logs)
    l20 = df.attrs["l20"]
    df["row"] = np.arange(len(df))

    # Vegas lines: league averages as of each date (earlier games only)
    gm = games.copy()
    gm["season"] = gm["date"].astype(str).map(season_key)
    gm = gm.sort_values("date")
    gm["lg_total"] = gm.groupby("season")["total"].transform(lambda s: s.shift(1).expanding().mean())
    gm["lg_total"] = gm["lg_total"].fillna(gm.groupby("season")["total"].transform("mean"))
    gmap = gm.set_index("game_id")[["home", "spread", "total", "lg_total"]].to_dict("index")

    df = df.merge(ta.drop(columns=["date", "season", "opp"]).rename(columns={"team": "opp"}),
                  on=["game_id", "opp"], how="left")
    pa2 = pa.drop(columns=["date", "season"]).rename(columns={"team": "opp"})
    pa2 = pa2.rename(columns={s: f"pos_{s}" for s in POS_STATS})
    df = df.merge(pa2, on=["game_id", "opp", "pos"], how="left")
    b2b = back_to_back(logs)
    df["b2b"] = [b2b.get((t, str(d)), False) for t, d in zip(df["team"], df["date"])]

    gf = []
    for gid, team, home in zip(df["game_id"], df["team"], df["home"]):
        g = gmap.get(str(gid), {})
        lt = g.get("lg_total")
        gf.append(game_features(g.get("spread"), g.get("total"), bool(home), lt, lt / 2 if lt else None))
    gf = pd.DataFrame(gf, index=df.index)
    for c in gf.columns:
        df[c] = gf[c]
    df = df.sort_values("row")
    df.attrs["l20"] = l20
    return df


def feature_matrix(df, stat):
    """Columns of x for a base stat, in FACTORS order."""
    from engine.matchups import FACTOR_METRIC
    cols = []
    for fac in FACTORS[stat]:
        if fac == "pos":
            cols.append(df[f"pos_{stat}"].fillna(0.0).to_numpy())
        elif fac in ("implied", "total", "home"):
            cols.append(df[fac].fillna(0.0).to_numpy())
        else:
            metric, sign = FACTOR_METRIC[fac]
            cols.append(sign * df[metric].fillna(0.0).to_numpy())
    return np.column_stack(cols)


# ------------------------------------------------------------------ fitting

def poisson_fit(X, y, offset, ridge=2.0, iters=30):
    """Poisson regression with an offset and an intercept, ridge on the slopes. Returns (coef, se)."""
    n, p = X.shape
    Xi = np.column_stack([np.ones(n), X])
    beta = np.zeros(p + 1)
    pen = np.diag([0.0] + [ridge] * p)
    for _ in range(iters):
        eta = np.clip(offset + Xi @ beta, -20, 20)
        mu = np.exp(eta)
        z = (eta - offset) + (y - mu) / mu
        W = mu
        H = Xi.T @ (Xi * W[:, None]) + pen
        new = np.linalg.solve(H, Xi.T @ (W * z))
        if np.max(np.abs(new - beta)) < 1e-7:
            beta = new
            break
        beta = new
    # overdispersion-corrected standard errors
    mu = np.exp(np.clip(offset + Xi @ beta, -20, 20))
    phi = max(1.0, float(((y - mu) ** 2 / mu).sum() / max(1, n - p - 1)))
    cov = np.linalg.inv(Xi.T @ (Xi * mu[:, None]) + pen) * phi
    return beta, np.sqrt(np.diag(cov))


def eligible(df, min_prev=10, min_minutes=10):
    return df[(df["n_prev"] >= min_prev) & (df["min"] >= min_minutes) & (df["proj_min"] >= 12)]


def fit_weights(df, min_t=1.5):
    """Weights per stat and factor. A factor only gets a weight if
      * its effect is clearly bigger than noise (|t| >= min_t), and
      * it works in the direction basketball sense says it should
        (every factor is signed so that positive = good for the player).
    Anything else gets weight 0, so it can't drive picks."""
    d = eligible(df)
    out = {"stats": {}, "minutes": {}, "n_rows": int(len(d)),
           "date_range": [str(d["date"].min()), str(d["date"].max())]}
    for s in BASE:
        base = d[f"rate_{s}"].to_numpy() * d["min"].to_numpy()
        ok = base > 0.05
        X, y = feature_matrix(d, s)[ok], d[s].to_numpy(dtype=float)[ok]
        beta, se = poisson_fit(X, y, np.log(base[ok]))
        facs = {}
        for i, fac in enumerate(FACTORS[s]):
            b, e = float(beta[i + 1]), float(se[i + 1])
            keep = b > 0 and b / e >= min_t
            facs[fac] = {"weight": min(b, 1.5) if keep else 0.0, "raw": b, "t": b / e}
        out["stats"][s] = {"intercept": float(beta[0]), "factors": facs, "n": int(ok.sum())}
    # minutes: how much starters lose in expected blowouts, and on back-to-backs
    m = d[d["proj_min"] >= 20]
    y = (m["min"] - m["proj_min"]).to_numpy()
    X = np.column_stack([np.ones(len(m)), m["blowout"].to_numpy() * (m["proj_min"].to_numpy() / 36),
                         m["b2b"].astype(float).to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    out["minutes"] = {"intercept": float(coef[0]), "blowout_per_point": float(coef[1]), "b2b": float(coef[2])}
    return out


def save_weights(w):
    WEIGHTS.write_text(json.dumps(w, indent=1))


def load_weights():
    if WEIGHTS.exists():
        return json.loads(WEIGHTS.read_text())
    return None
