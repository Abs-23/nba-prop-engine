"""Matchup factors: the game-specific context around a player's projection.

Each stat has its own set of factors (2-4 each), because what moves rebounds
(opponent size, rebounding strength) is different from what moves free throws
(opponent fouling, free-throw rate allowed). Every factor is expressed as a
log-ratio vs league average:

    x = log(this opponent's value / league average)

so x = +0.10 means "about 10% above average". How much each factor actually
moves a player's output is NOT assumed: weights are learned from last season
(see engine/fit.py). A factor that turns out not to matter gets weight ~0.

All team numbers are "as of" a date (only games before it), so the same code
serves the backtest (no peeking into the future) and tonight's picks.
"""
import numpy as np
import pandas as pd

from engine.database import season_key

K_SHRINK = 10          # games before a team's own numbers are trusted over last season's
PRIOR_KEEP = 0.5       # last season's team tendencies carry over at half strength (rosters change)
POS_STATS = ["pts", "reb", "ast", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]

# Factors per base stat. "pos" = what the opponent allows to the player's position.
FACTORS = {
    "pts":  ["pos", "implied", "efg", "rim"],
    "fgm":  ["pos", "implied", "efg", "rim"],
    "fga":  ["pos", "total", "tov_forced"],
    "fg3m": ["pos", "fg3a_rate", "fg3pct", "implied"],
    "fg3a": ["pos", "fg3a_rate", "total"],
    "ftm":  ["pos", "fouls", "ftr", "home"],
    "fta":  ["pos", "fouls", "ftr", "home"],
    "reb":  ["pos", "reb_share", "size", "reb_avail"],
    "ast":  ["pos", "ast_rate", "implied", "tov_forced"],
}
FACTOR_LABELS = {
    "pos": "Position defense", "implied": "Vegas team total", "total": "Game pace (Vegas total)",
    "efg": "Shooting allowed", "rim": "Rim protection", "tov_forced": "Turnovers forced",
    "fg3a_rate": "3s allowed", "fg3pct": "3P% allowed", "fouls": "Opponent fouling",
    "ftr": "FT rate allowed", "home": "Home/away", "reb_share": "Opponent rebounding",
    "size": "Opponent size", "reb_avail": "Rebound chances", "ast_rate": "Assisted shots allowed",
}
# Team metrics, computed for the DEFENDING team from what happened in its games.
# (name, higher_is_good_for_offense)
TEAM_METRICS = ["pace", "reb_share", "reb_avail", "pf100", "ftr_allowed", "efg_allowed", "fg3a_rate_allowed",
                "fg3pct_allowed", "ast_rate_allowed", "blk100", "tov_forced100", "pts_allowed", "height"]
# factor -> (team metric, sign). Sign -1: a HIGHER opponent value means LESS for our player.
FACTOR_METRIC = {
    "efg": ("efg_allowed", 1), "rim": ("blk100", -1), "tov_forced": ("tov_forced100", -1),
    "fg3a_rate": ("fg3a_rate_allowed", 1), "fg3pct": ("fg3pct_allowed", 1), "fouls": ("pf100", 1),
    "ftr": ("ftr_allowed", 1), "reb_share": ("reb_share", -1), "size": ("height", -1),
    "reb_avail": ("reb_avail", 1), "ast_rate": ("ast_rate_allowed", 1),
}
DEFAULT_HEIGHT = {"G": 76.5, "F": 79.5, "C": 82.5}


def _fmt_height(inches):
    return f"{int(inches // 12)}'{int(round(inches % 12))}\""


# ------------------------------------------------------------------ team-game table

def team_games(logs, players=None):
    """One row per team per game, with the team's totals, the opponent's totals,
    and the per-game metrics describing this team AS A DEFENSE."""
    df = logs[logs["played"].astype(bool)].copy()
    hmap = {}
    if players is not None and len(players):
        hmap = dict(zip(players["player_id"].astype(int), pd.to_numeric(players["height"], errors="coerce")))
    df["height"] = [hmap.get(int(p)) if pd.notna(hmap.get(int(p), np.nan)) else DEFAULT_HEIGHT.get(pos, 79.5)
                    for p, pos in zip(df["player_id"], df["pos"])]
    big = df["pos"].isin(["F", "C"])
    df["hmin"] = np.where(big, df["height"] * df["min"], 0.0)
    df["bmin"] = np.where(big, df["min"], 0.0)
    cols = ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "fgm", "fga",
            "fg3m", "fg3a", "ftm", "fta", "min", "hmin", "bmin"]
    t = df.groupby(["game_id", "date", "team", "opp"], as_index=False)[cols].sum()
    o = t.rename(columns={c: f"o_{c}" for c in cols}).rename(columns={"team": "opp", "opp": "team"})
    t = t.merge(o[["game_id", "team"] + [f"o_{c}" for c in cols]], on=["game_id", "team"])
    own = t["fga"] - t["oreb"] + t["tov"] + 0.44 * t["fta"]
    opp = t["o_fga"] - t["o_oreb"] + t["o_tov"] + 0.44 * t["o_fta"]
    t["poss"] = (own + opp) / 2
    t["pace"] = t["poss"]
    t["reb_share"] = t["reb"] / (t["reb"] + t["o_reb"])
    t["reb_avail"] = t["reb"] + t["o_reb"]
    t["pf100"] = 100 * t["pf"] / t["poss"]
    t["ftr_allowed"] = t["o_fta"] / t["o_fga"]
    t["efg_allowed"] = (t["o_fgm"] + 0.5 * t["o_fg3m"]) / t["o_fga"]
    t["fg3a_rate_allowed"] = t["o_fg3a"] / t["o_fga"]
    t["fg3pct_allowed"] = t["o_fg3m"] / t["o_fg3a"].clip(lower=1)
    t["ast_rate_allowed"] = t["o_ast"] / t["o_fgm"].clip(lower=1)
    t["blk100"] = 100 * t["blk"] / t["poss"]
    t["tov_forced100"] = 100 * t["o_tov"] / t["poss"]
    t["pts_allowed"] = t["o_pts"]
    t["height"] = t["hmin"] / t["bmin"].clip(lower=1)
    t["season"] = t["date"].astype(str).map(season_key)
    return t.sort_values(["date", "game_id"]).reset_index(drop=True)


def _league_asof(t, col):
    """League average of a column over the season, using only earlier dates."""
    by_date = t.groupby(["season", "date"])[col].agg(["sum", "count"]).reset_index()
    by_date[["cs", "cc"]] = by_date.groupby("season")[["sum", "count"]].cumsum() - by_date[["sum", "count"]]
    by_date["lg"] = by_date["cs"] / by_date["cc"].replace(0, np.nan)
    return t[["season", "date"]].merge(by_date[["season", "date", "lg"]], on=["season", "date"], how="left")["lg"]


def _final_x(t, col):
    """{(season, team): log(team season average / league season average)} using full seasons."""
    g = t.groupby(["season", "team"])[col].mean()
    lg = t.groupby("season")[col].mean()
    return {k: float(np.log(v / lg[k[0]])) for k, v in g.items() if v > 0 and lg[k[0]] > 0}


def _prev_season(s):
    return f"{int(s) - 1:02d}"


def team_asof(t, metrics=None):
    """For every (game, defending team): each metric as a shrunk log-ratio, using only prior games."""
    out = t[["game_id", "date", "season", "team", "opp"]].copy()
    grp = t.groupby(["season", "team"])
    n = grp.cumcount()
    out["n_games"] = n
    for m in metrics or TEAM_METRICS:
        prev_mean = (grp[m].cumsum() - t[m]) / n.replace(0, np.nan)
        lg = _league_asof(t, m)
        x_cur = np.log(prev_mean / lg)
        fin = _final_x(t, m)
        x_pri = np.array([PRIOR_KEEP * fin.get((_prev_season(s), tm), 0.0) for s, tm in zip(t["season"], t["team"])])
        x_cur = np.nan_to_num(x_cur.to_numpy(dtype=float), nan=0.0)
        out[m] = (n * x_cur + K_SHRINK * x_pri) / (n + K_SHRINK)
    return out


def team_latest(t, season):
    """Each team's metrics right now (after every game in the database) for `season`."""
    rows = []
    cur = t[t["season"] == season]
    lg = {m: cur[m].mean() for m in TEAM_METRICS} if len(cur) else {}
    for team in sorted(set(t["team"])):
        tc = cur[cur["team"] == team]
        n = len(tc)
        row = {"team": team, "n_games": n}
        for m in TEAM_METRICS:
            fin = _final_x(t, m)
            x_pri = PRIOR_KEEP * fin.get((_prev_season(season), team), 0.0)
            x_cur = float(np.log(tc[m].mean() / lg[m])) if n and lg.get(m) else 0.0
            row[m] = (n * x_cur + K_SHRINK * x_pri) / (n + K_SHRINK)
            prev = t[(t["season"] == _prev_season(season)) & (t["team"] == team)]
            if n >= K_SHRINK or not len(prev):
                row[m + "_raw"], row[m + "_src"] = (float(tc[m].mean()) if n else np.nan), "current"
            else:  # too few games this season to quote: show last season's number
                row[m + "_raw"], row[m + "_src"] = float(prev[m].mean()), "prior"
            row[m + "_lg"] = float(lg.get(m, np.nan))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("team")
    for m in TEAM_METRICS:  # rank 1 = highest value
        df[m + "_rank"] = df[m].rank(ascending=False, method="min").astype(int)
    return df


# ------------------------------------------------------------------ position defense

def pos_table(logs):
    """Per (game, defending team, position): opponent players' totals of each stat and minutes."""
    df = logs[logs["played"].astype(bool) & (logs["min"] > 0)].copy()
    g = df.groupby(["game_id", "date", "opp", "pos"], as_index=False)[POS_STATS + ["min"]].sum()
    g = g.rename(columns={"opp": "team"})
    g["season"] = g["date"].astype(str).map(season_key)
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def pos_asof(p):
    """For every (game, defending team, position): log(allowed per-minute / league per-minute), shrunk."""
    out = p[["game_id", "date", "season", "team", "pos"]].copy()
    grp = p.groupby(["season", "team", "pos"])
    n = grp.cumcount()
    cmin = grp["min"].cumsum() - p["min"]
    # league per-minute by season+pos, earlier dates only
    lg = p.groupby(["season", "pos", "date"])[POS_STATS + ["min"]].sum().groupby(level=[0, 1]).cumsum() \
        - p.groupby(["season", "pos", "date"])[POS_STATS + ["min"]].sum()
    lg = lg.reset_index()
    fin = {}
    for s in POS_STATS:
        tot = p.groupby(["season", "team", "pos"])[[s, "min"]].sum()
        lgt = p.groupby(["season", "pos"])[[s, "min"]].sum()
        fin[s] = {k: float(np.log((v[s] / v["min"]) / (lgt.loc[(k[0], k[2]), s] / lgt.loc[(k[0], k[2]), "min"])))
                  for k, v in tot.iterrows() if v[s] > 0 and v["min"] > 0}
    m = p[["season", "pos", "date"]].merge(lg, on=["season", "pos", "date"], how="left", suffixes=("", "_lg"))
    for s in POS_STATS:
        cs = grp[s].cumsum() - p[s]
        x_cur = np.log((cs / cmin.replace(0, np.nan)) / (m[s] / m["min"].replace(0, np.nan)))
        x_cur = np.nan_to_num(x_cur.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        x_pri = np.array([PRIOR_KEEP * fin[s].get((_prev_season(se), tm, po), 0.0)
                          for se, tm, po in zip(p["season"], p["team"], p["pos"])])
        out[s] = (n * x_cur + K_SHRINK * x_pri) / (n + K_SHRINK)
    return out


def pos_latest(p, season):
    cur = p[p["season"] == season]
    out = {}
    for s in POS_STATS:
        fin_prev = {}
        prev = p[p["season"] == _prev_season(season)]
        if len(prev):
            tot = prev.groupby(["team", "pos"])[[s, "min"]].sum()
            lgt = prev.groupby("pos")[[s, "min"]].sum()
            fin_prev = {k: float(np.log((v[s] / v["min"]) / (lgt.loc[k[1], s] / lgt.loc[k[1], "min"])))
                        for k, v in tot.iterrows() if v[s] > 0 and v["min"] > 0}
        tot = cur.groupby(["team", "pos"])[[s, "min"]].sum() if len(cur) else None
        lgt = cur.groupby("pos")[[s, "min"]].sum() if len(cur) else None
        ngames = cur.groupby(["team", "pos"]).size() if len(cur) else {}
        for team in set(p["team"]):
            for pos in ("G", "F", "C"):
                n = int(ngames.get((team, pos), 0)) if len(cur) else 0
                x_cur = 0.0
                if n and tot is not None and (team, pos) in tot.index and tot.loc[(team, pos), "min"] > 0:
                    v, lv = tot.loc[(team, pos)], lgt.loc[pos]
                    if v[s] > 0 and lv[s] > 0:
                        x_cur = float(np.log((v[s] / v["min"]) / (lv[s] / lv["min"])))
                x_pri = PRIOR_KEEP * fin_prev.get((team, pos), 0.0)
                out.setdefault((team, pos), {})[s] = (n * x_cur + K_SHRINK * x_pri) / (n + K_SHRINK)
    return out


# ------------------------------------------------------------------ feature vector

def game_features(spread, total, home, league_total, league_team_pts):
    """Vegas-based features for one team in one game. Missing lines -> neutral (0)."""
    f = {"home": 0.5 if home else -0.5, "implied": 0.0, "total": 0.0, "blowout": 0.0}
    if total is not None and not pd.isna(total) and league_total:
        f["total"] = float(np.log(total / league_total))
        if spread is not None and not pd.isna(spread):
            implied = (total - spread) / 2 if home else (total + spread) / 2
            f["implied"] = float(np.log(implied / league_team_pts))
    if spread is not None and not pd.isna(spread):
        f["blowout"] = max(0.0, abs(float(spread)) - 7.0)
    return f


def stat_features(stat, team_row, pos_row, game_f):
    """Feature vector {factor: x} for one base stat."""
    x = {}
    for fac in FACTORS[stat]:
        if fac == "pos":
            x[fac] = float(pos_row.get(stat, 0.0)) if pos_row else 0.0
        elif fac in ("implied", "total", "home"):
            x[fac] = game_f[fac]
        else:
            metric, sign = FACTOR_METRIC[fac]
            x[fac] = sign * float(team_row.get(metric, 0.0)) if team_row is not None else 0.0
    return x


# ------------------------------------------------------------------ human-readable evidence

def describe(fac, stat, opp, pos, team_tbl, opp_bigs=None, game=None, x=0.0):
    """One sentence of evidence for a factor, with real numbers."""
    from engine.stats import LABELS
    lab = LABELS.get(stat, stat).lower()
    if fac == "pos":
        pct = 100 * (np.exp(x) - 1)
        who = dict(G="guards", F="forwards", C="centers")[pos]
        more = "more" if pct >= 0 else "fewer"
        return f"{opp} allow {abs(pct):.0f}% {more} {lab} per minute to {who} than average"
    if fac == "blowout":
        return (f"Spread {abs(game['spread']):.1f}: blowout risk, starters usually sit late "
                f"(about {abs(x):.1f} fewer minutes projected)") if game else ""
    if fac in ("implied", "total"):
        if not game or game.get("total") is None or pd.isna(game.get("total")):
            return "No Vegas line yet"
        if fac == "total":
            return f"Vegas game total {game['total']:.1f}"
        return f"Vegas implied team total {game['implied_pts']:.1f} points"
    if fac == "home":
        return "Playing at home" if game and game.get("home") else "Playing on the road"
    metric = FACTOR_METRIC[fac][0]
    r = team_tbl.loc[opp] if opp in team_tbl.index else None
    if r is None:
        return ""
    raw, rank = r[metric + "_raw"], int(r[metric + "_rank"])
    src = " last season" if r.get(metric + "_src") == "prior" else ""
    if pd.isna(raw):
        raw = None
    txt = {
        "reb_share": lambda v: f"{opp} grab {100*v:.1f}% of all rebounds in their games (#{rank} of 30)",
        "height": lambda v: f"{opp} frontcourt averages {_fmt_height(v)} (#{rank} tallest)",
        "reb_avail": lambda v: f"{v:.1f} total rebounds per game in {opp} games (#{rank} of 30)",
        "pf100": lambda v: f"{opp} commit {v:.1f} fouls per 100 possessions (#{rank} most)",
        "ftr_allowed": lambda v: f"{opp} allow {v:.2f} FT attempts per shot (#{rank} most)",
        "efg_allowed": lambda v: f"{opp} allow {100*v:.1f}% effective FG (#{rank} worst defense)",
        "blk100": lambda v: f"{opp} block {v:.1f} shots per 100 possessions (#{rank} most)",
        "tov_forced100": lambda v: f"{opp} force {v:.1f} turnovers per 100 possessions (#{rank} most)",
        "fg3a_rate_allowed": lambda v: f"{100*v:.0f}% of shots vs {opp} are 3s (#{rank} most)",
        "fg3pct_allowed": lambda v: f"{opp} allow {100*v:.1f}% from three (#{rank} worst)",
        "ast_rate_allowed": lambda v: f"{100*v:.0f}% of made shots vs {opp} are assisted (#{rank} most)",
    }[metric]
    s = txt(raw) if raw is not None else f"{opp}: #{rank} of 30"
    if src:
        s = s.replace(")", f",{src})", 1) if ")" in s else s + src
    if fac in ("reb_share", "size") and opp_bigs:
        s += "; " + ", ".join(opp_bigs)
    return s
