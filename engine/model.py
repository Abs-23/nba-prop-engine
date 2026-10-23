"""The projection model.

For every player and stat it answers: "what's the full range of outcomes
tonight, and how likely is each one?"

    expected stat = (production per minute) x (projected minutes) x (opponent factor)

  * production per minute: recent games count more (exponential decay);
    last season counts at half weight so early-season noise doesn't dominate.
  * projected minutes: the single biggest driver of props. Weighted to the
    last ~5 games, blended with last season while this season's sample is tiny,
    trimmed slightly on the second night of a back-to-back.
  * opponent factor: how much of this stat the opponent allows vs league
    average, shrunk toward neutral until it has enough games to trust.

The *spread* around that expectation uses a negative-binomial distribution
fitted to how volatile this specific player is (shrunk toward how volatile
players typically are for that stat). That gives an honest probability for
any line — not just a point estimate.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine.database import season_key
from engine.stats import STATS, add_stat_columns, norm_name


def prepare_logs(logs, cfg):
    df = logs.copy()
    df = df[(df["played"].astype(bool)) & (df["min"] > 0)]
    df = add_stat_columns(df)
    df["season"] = df["date"].astype(str).map(season_key)
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------- opponents

def opponent_factors(df, cfg):
    """{team: {stat: (factor, rank, n_games)}} — rank 30 = most generous defense."""
    mcfg = cfg["model"]
    cur, prior = cfg["season"]["current"], cfg["season"]["prior"]
    keys = list(STATS)
    tg = df.groupby(["game_id", "season", "team", "opp"])[keys].sum().reset_index()

    def factors(season):
        s = tg[tg["season"] == season]
        if s.empty:
            return {}, {}
        league = s[keys].mean()
        allowed = s.groupby("opp")[keys].mean()
        n = s.groupby("opp").size()
        return (allowed / league.replace(0, np.nan)).to_dict("index"), n.to_dict()

    f_cur, n_cur = factors(cur)
    f_pri, _ = factors(prior)
    k = mcfg["opponent_shrink_games"]
    lo, hi = mcfg["opponent_factor_clamp"]
    teams = set(f_cur) | set(f_pri)
    out = {t: {} for t in teams}
    for s in keys:
        vals = {}
        for t in teams:
            # last season's defense, half-trusted (rosters change over a summer)
            fp = 1 + 0.5 * (f_pri.get(t, {}).get(s, 1.0) - 1)
            n = n_cur.get(t, 0)
            fc = f_cur.get(t, {}).get(s, 1.0)
            f = (n * fc + k * fp) / (n + k)
            vals[t] = (float(np.clip(f, lo, hi)), n)
        order = sorted(vals, key=lambda t: vals[t][0])
        for rank, t in enumerate(order, 1):
            out[t][s] = (vals[t][0], rank, vals[t][1])
    return out


# ---------------------------------------------------------------- dispersion

def league_dispersion(df, min_games=20):
    """Typical variance/mean ratio per stat across regular rotation players."""
    out = {}
    g = df.groupby("player_id")
    for s in STATS:
        ratios = []
        for _, h in g:
            if len(h) < min_games:
                continue
            x = h[s].tail(40)
            if x.mean() > 0.5:
                ratios.append(x.var(ddof=1) / x.mean())
        out[s] = float(np.median(ratios)) if ratios else 1.5
    return out


# ---------------------------------------------------------------- players

def _decay_weights(n, half_life):
    ages = np.arange(n)[::-1]  # most recent game has age 0
    return 0.5 ** (ages / half_life)


def team_last_game(df):
    return df.groupby("team")["date"].max().to_dict()


def player_profile(h, team, opp, game_date, cfg, opp_f, disp0, last_game_by_team):
    """Everything the engine knows about one player for tonight's game.

    `h` is the player's game log (both seasons), oldest first.
    """
    mcfg = cfg["model"]
    cur, prior = cfg["season"]["current"], cfg["season"]["prior"]
    h = h[h["season"].isin([cur, prior])]
    n = len(h)
    if n == 0:
        return None
    is_cur = (h["season"] == cur).to_numpy()
    n_cur = int(is_cur.sum())

    # production per minute, recency-weighted, last season half weight
    w = _decay_weights(n, mcfg["rate_half_life_games"]) * np.where(is_cur, 1.0, mcfg["prior_season_weight"])
    mins = h["min"].to_numpy()
    rates = {s: float((w * h[s].to_numpy()).sum() / (w * mins).sum()) for s in STATS}

    # minutes projection
    hc = h[is_cur]
    hp = h[~is_cur]
    m_prior = float(hp["min"].mean()) if len(hp) else None
    if n_cur:
        wm = _decay_weights(n_cur, mcfg["minutes_half_life_games"])
        m_cur = float((wm * hc["min"].to_numpy()).sum() / wm.sum())
    else:
        m_cur = None
    if m_cur is None:
        proj_min = m_prior
    elif m_prior is None:
        proj_min = m_cur
    else:
        k = mcfg["minutes_prior_games"]
        proj_min = (n_cur * m_cur + k * m_prior) / (n_cur + k)

    last_team = h["team"].iloc[-1]
    new_team = last_team != team
    games_with_team = int((h["team"] == team).sum()) if not new_team else 0
    if not new_team:  # count only the current stint (consecutive games at the end of the log)
        t = h["team"].to_numpy()
        games_with_team = int(len(t) - np.flatnonzero(t != team)[-1] - 1) if (t != team).any() else len(t)
    if new_team and n_cur == 0:
        # last season's role on a different team says little about minutes now
        proj_min *= 0.95

    yesterday = (date.fromisoformat(game_date) - timedelta(days=1)).isoformat()
    b2b = last_game_by_team.get(team) == yesterday
    if b2b:
        # Last season's data showed no minutes drop on back-to-backs (+0.07 min), so the
        # default factor is 1.0. Kept as a setting in case that changes.
        proj_min *= mcfg["back_to_back_minutes_factor"]

    last10 = h.tail(10)
    m10, sd10 = float(last10["min"].mean()), float(last10["min"].std(ddof=1)) if len(last10) > 1 else 0.0
    cv10 = sd10 / m10 if m10 else 1.0

    means, disp, opp_info = {}, {}, {}
    kd = mcfg["dispersion_prior_games"]
    for s in STATS:
        f, rank, nopp = opp_f.get(opp, {}).get(s, (1.0, 15, 0))
        means[s] = rates[s] * proj_min * f
        x = h[s].tail(40)
        d_emp = x.var(ddof=1) / x.mean() if len(x) > 2 and x.mean() > 0 else disp0[s]
        nn = len(x)
        disp[s] = max(1.0, (nn * d_emp + kd * disp0[s]) / (nn + kd))
        opp_info[s] = {"factor": f, "rank": rank, "n": nopp}

    return {
        "player_id": int(h["player_id"].iloc[-1]), "player": h["player"].iloc[-1],
        "team": team, "opp_abbr": opp, "n": n, "n_cur": n_cur, "new_team": bool(new_team),
        "games_with_team": games_with_team,
        "min_cur": m_cur, "min_prior": m_prior, "min_cur_last": float(hc["min"].iloc[-1]) if n_cur else None,
        "last_team": last_team, "b2b": bool(b2b),
        "proj_min": float(proj_min), "min_l10": m10, "min_sd10": sd10, "min_cv10": cv10,
        "rates": rates, "means": means, "disp": disp, "opp": opp_info,
        "history": h,
    }


# ---------------------------------------------------------------- probabilities

def nb_params(mean, disp):
    """Negative binomial with given mean and variance = disp * mean."""
    var = max(mean * disp, mean * 1.0001)
    p = mean / var
    r = mean * mean / (var - mean)
    return r, p


def prob_at_least(mean, disp, k):
    """P(X >= k) for integer k."""
    if mean <= 0:
        return 0.0 if k > 0 else 1.0
    if disp <= 1.0001:
        return float(sps.poisson.sf(k - 1, mean))
    r, p = nb_params(mean, disp)
    return float(sps.nbinom.sf(k - 1, r, p))


def prob_exactly(mean, disp, k):
    if mean <= 0:
        return 1.0 if k == 0 else 0.0
    if disp <= 1.0001:
        return float(sps.poisson.pmf(k, mean))
    r, p = nb_params(mean, disp)
    return float(sps.nbinom.pmf(k, r, p))


def side_probability(mean, disp, line, side):
    """Probability the bet wins, given it doesn't push. Handles 24.5 and 25 lines."""
    if float(line).is_integer():
        k = int(line)
        p_push = prob_exactly(mean, disp, k)
        p_over = prob_at_least(mean, disp, k + 1)
        p_under = 1 - p_over - p_push
        denom = max(1e-9, 1 - p_push)
        return (p_over if side == "over" else p_under) / denom
    k = int(np.floor(line)) + 1  # over 24.5 means 25+
    p_over = prob_at_least(mean, disp, k)
    return p_over if side == "over" else 1 - p_over


def hit_rate(values, line, side):
    v = np.asarray(values, dtype=float)
    v = v[v != line]  # pushes don't count either way
    if len(v) == 0:
        return None, 0
    hits = (v > line) if side == "over" else (v < line)
    return float(hits.mean()), int(hits.sum())


class MatchupContext:
    """Tonight's matchup numbers for every team, plus the learned factor weights."""

    def __init__(self, logs, game_log, players, rosters, weights, season, injuries=None, frame=None,
                 today=None):
        from engine import matchups as M
        from engine.absences import ROT_MIN, SHARE_STATS, player_shares
        self.M = M
        self.w = weights
        self.today = today
        self.SHARE_STATS = SHARE_STATS
        # --- who is out tonight, and what share of their team leaves with them
        sh = player_shares(logs).sort_values("date")
        last = sh.groupby("player_id").tail(1).set_index("player_id")
        self.last_team_game = logs.groupby("team")["date"].max().to_dict()
        out_status = {"out", "doubtful"}
        self.out = {}
        for pid, r in last.iterrows():
            inj = (injuries or {}).get(norm_name(r["player"]))
            team_now = (rosters or {}).get(norm_name(r["player"]), r["team"])
            if not inj or inj["status"] not in out_status or team_now != r["team"] or r["cur_min"] < ROT_MIN:
                continue
            self.out.setdefault(r["team"], []).append({
                **{s: float(r[f"cur_sh_{s}"]) for s in SHARE_STATS},
                "player": r["player"], "pos": r["pos"], "minutes": float(r["cur_min"])})
        # --- each player's normal situation: how many teammates/opponents are usually missing.
        # Measured from THIS season's games only (rosters change over the summer, and the end of
        # last season is full of tanking/rest absences). Until a player has 10 games this season,
        # it is blended toward the typical level for this point of the season (injuries pile up:
        # ~6% of a team's production is missing on a typical October night, ~48% by March).
        self.base = {}
        if frame is not None and len(frame):
            cols = [c for c in frame.columns if c.startswith("d_vac_") or c.startswith("d_ovac_")]
            xcols = [c.replace("d_", "x_", 1) for c in cols]
            f = pd.DataFrame({"player_id": frame["player_id"].to_numpy(), "date": frame["date"].astype(str).to_numpy(),
                              **{x: (frame[c] + frame[x + "_base"]).to_numpy() for c, x in zip(cols, xcols)}})
            month = f["date"].str[5:7].astype(int)
            m_now = int((today or f["date"].max())[5:7])
            near = {10: [10, 11], 11: [10, 11, 12], 12: [11, 12, 1], 1: [12, 1, 2], 2: [1, 2, 3], 3: [2, 3, 4],
                    4: [3, 4], 5: [4, 5], 6: [5, 6]}.get(m_now, [m_now])
            pool = f[month.isin(near)]
            typical = (pool if len(pool) > 500 else f)[xcols].mean().to_dict()
            f["season"] = f["date"].map(season_key)
            cur = f[f["season"] == season].sort_values(["player_id", "date"])
            k = 10
            ew = cur.groupby("player_id")[xcols].agg(lambda x: x.ewm(halflife=15).mean().iloc[-1]) if len(cur) else None
            n = cur.groupby("player_id").size() if len(cur) else {}
            self.typical = typical
            for pid in set(f["player_id"].astype(int)):
                ncur = int(n.get(pid, 0)) if len(cur) else 0
                self.base[pid] = {x: ((ncur * float(ew.loc[pid, x]) if ncur else 0.0) + k * typical[x]) / (ncur + k)
                                  for x in xcols}
        tg = M.team_games(logs, players)
        self.team = M.team_latest(tg, season)
        self.pos = M.pos_latest(M.pos_table(logs), season)
        gl = game_log.copy()
        gl["season"] = gl["date"].astype(str).map(season_key)
        cur = gl[(gl["season"] == season) & gl["total"].notna()]
        src = cur if len(cur) >= 20 else gl[gl["total"].notna()]
        self.league_total = float(src["total"].mean()) if len(src) else 225.0
        self.player_pos = {}
        if players is not None and len(players):
            self.player_pos = dict(zip(players["player_id"].astype(int), players["pos"]))
        played = logs[logs["played"].astype(bool)]
        self.log_pos = played.groupby("player_id")["pos"].agg(lambda x: x.mode().iat[0]).to_dict()
        # each team's top rebounding bigs on the current roster (for the evidence text)
        self.bigs = {}
        on_team = {k: v for k, v in (rosters or {}).items()}
        recent = played[played["pos"].isin(["F", "C"]) & (played["min"] >= 15)]
        recent = recent.sort_values("date").groupby("player_id").tail(40)
        agg = recent.groupby("player_id").agg(player=("player", "last"), reb=("reb", "mean"), n=("reb", "size"))
        agg = agg[agg["n"] >= 8].reset_index()
        for r in agg.sort_values("reb", ascending=False).itertuples():
            team = on_team.get(norm_name(r.player))
            if team and len(self.bigs.setdefault(team, [])) < 2 and r.reb >= 5:
                self.bigs[team].append(f"{r.player.split()[-1]} {r.reb:.1f} rpg")

    def pos_of(self, pid):
        return self.player_pos.get(int(pid)) or self.log_pos.get(int(pid), "F")

    def apply(self, prof, game):
        """Re-project the 9 core stats (and their combos) with tonight's matchup."""
        M, w = self.M, self.w
        home = prof["home"]
        gf = M.game_features(game.get("spread"), game.get("total"), home, self.league_total, self.league_total / 2)
        pos = self.pos_of(prof["player_id"])
        opp = prof["opp_abbr"]
        team_row = self.team.loc[opp].to_dict() if opp in self.team.index else None
        pos_row = self.pos.get((opp, pos))
        from engine.absences import boost_x
        from engine.fit import ALTITUDE, USAGE_SRC
        pm = prof["proj_min"]
        base = self.base.get(int(prof["player_id"]), getattr(self, "typical", {}))
        team, wm = prof["team"], w["minutes"]
        mates = [o for o in self.out.get(team, []) if o["player"] != prof["player"]]
        opps = self.out.get(opp, [])
        vac = {s: min(0.8, sum(o[s] for o in mates)) for s in self.SHARE_STATS}
        ovac = {s: min(0.8, sum(o[s] for o in opps)) for s in self.SHARE_STATS}
        same = sum(o["minutes"] for o in mates if o["pos"] == pos)
        other = sum(o["minutes"] for o in mates if o["pos"] != pos)
        d_same = same - base.get("x_vac_min_same", 0.0)
        d_other = other - base.get("x_vac_min_other", 0.0)
        today = self.today or game.get("date")
        rest = (date.fromisoformat(today) - date.fromisoformat(self.last_team_game.get(team, today))).days \
            if today else 2
        opp_rest = (date.fromisoformat(today) - date.fromisoformat(self.last_team_game.get(opp, today))).days \
            if today else 2
        if rest > 10:
            rest = 2  # season opener / long break: normal rest
        if opp_rest > 10:
            opp_rest = 2
        b2b = rest == 1
        parts_min = {"blowout": wm["blowout_per_point"] * gf["blowout"] * (pm / 36),
                     "b2b": wm.get("b2b", 0.0) * b2b,
                     "freed": wm.get("vac_min_same", 0.0) * d_same + wm.get("vac_min_other", 0.0) * d_other}
        pm_adj = max(0.0, pm + sum(parts_min.values()))
        min_eff = pm_adj / pm - 1 if pm else 0.0
        eff, means = {}, {}
        prof["base_means"] = {k: v for k, v in prof["means"].items()}
        for c in M.POS_STATS:
            src = USAGE_SRC[c]
            extra = {
                "usage": float(boost_x(vac[src]) - base.get(f"x_vac_{src}", 0.0)),
                "opp_out": float(boost_x(ovac["reb" if c == "reb" else "min"])
                                 - base.get("x_ovac_reb" if c == "reb" else "x_ovac_min", 0.0)),
                "opp_tired": 1.0 if opp_rest == 1 else 0.0,
                "altitude": 1.0 if (not home and opp in ALTITUDE) else 0.0,
                "rested": 1.0 if rest >= 3 else 0.0,
            }
            x = M.stat_features(c, team_row, pos_row, gf, extra)
            tot = 0.0
            eff[c] = {}
            for fac, xv in x.items():
                wt = w["stats"][c]["factors"][fac]["weight"]
                eff[c][fac] = {"effect": float(np.exp(wt * xv) - 1), "x": xv, "weight": wt}
                tot += wt * xv
            means[c] = prof["rates"][c] * pm_adj * float(np.exp(tot))
        for c in M.POS_STATS:
            prof["means"][c] = means[c]
        for k in ("stl", "blk", "tov", "bs"):
            prof["means"][k] *= pm_adj / pm if pm else 1.0
        combos = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
        prof["matchup_detail"] = {}
        for stat in list(M.POS_STATS) + list(combos):
            parts = combos.get(stat, [stat])
            if stat in combos:
                prof["means"][stat] = sum(means[c] for c in parts)
            share = {c: means[c] / max(1e-9, sum(means[x] for x in parts)) for c in parts}
            items = []
            for c in parts:
                for fac, e in eff[c].items():
                    items.append({"stat": c, "factor": fac, "effect": share[c] * e["effect"], "x": e["x"],
                                  "weight": e["weight"]})
            if abs(min_eff) > 0.005:
                items.append({"stat": None, "factor": "minutes", "effect": min_eff, "x": pm_adj - pm, "weight": 1})
            net = float(np.prod([1 + i["effect"] for i in items if i["weight"]]) - 1)
            prof["matchup_detail"][stat] = {"net": net, "items": items}
        implied = None
        if game.get("total") is not None and game.get("spread") is not None and not pd.isna(game.get("total")):
            implied = (game["total"] - game["spread"]) / 2 if home else (game["total"] + game["spread"]) / 2
        prof["game"] = {"spread": game.get("spread"), "total": game.get("total"), "home": home,
                        "implied_pts": implied, "blowout": gf["blowout"]}
        bigs = self.bigs.get(opp, [])

        def who(lst, key):
            lst = sorted(lst, key=lambda o: -o[key])
            return ", ".join(f'{o["player"]} ({100*o[key]:.0f}%)' for o in lst[:3])

        min_bits = []
        if parts_min["freed"] >= 0.3:
            min_bits.append(f'+{parts_min["freed"]:.1f} with teammates out')
        if parts_min["blowout"] <= -0.3:
            min_bits.append(f'{parts_min["blowout"]:.1f} for blowout risk (spread {abs(game["spread"]):.1f})')
        if b2b and abs(parts_min["b2b"]) >= 0.1:
            min_bits.append(f'{parts_min["b2b"]:.1f} on a back-to-back')
        for stat, mu in prof["matchup_detail"].items():
            for it in mu["items"]:
                fac, c = it["factor"], it["stat"] or stat
                if fac == "usage":
                    src = USAGE_SRC[c]
                    it["text"] = (f"{team} without {who(mates, src)} of the team's "
                                  f"{dict(pts='points', fga='shots', fg3a='3PA', fta='FTA', reb='rebounds', ast='assists')[src]}"
                                  if mates else "No new absences vs his recent games (a returning teammate can lower this)")
                elif fac == "opp_out":
                    k = "reb" if c == "reb" else "min"
                    it["text"] = (f"{opp} without {who(opps, k)} of their {'rebounds' if k == 'reb' else 'minutes'}"
                                  if opps else f"{opp} at full strength")
                elif fac == "opp_tired":
                    it["text"] = f"{opp} played yesterday" if opp_rest == 1 else ""
                elif fac == "altitude":
                    it["text"] = f"Road game at {opp}" if it["x"] else ""
                elif fac == "rested":
                    it["text"] = f"{rest} days since last game" if it["x"] else ""
                elif fac == "minutes":
                    it["text"] = f"projected {pm_adj:.1f} minutes vs usual {pm:.1f} (" + "; ".join(min_bits) + ")" \
                        if min_bits else f"projected {pm_adj:.1f} minutes vs usual {pm:.1f}"
                else:
                    it["text"] = M.describe(fac, c, opp, pos, self.team, bigs, prof["game"], it["x"])
                it["label"] = M.FACTOR_LABELS.get(fac, "Minutes")
        prof["teammates_out"] = [o["player"] for o in mates]
        prof["pos"] = pos
        prof["opp_bigs"] = self.bigs.get(opp, [])
        prof["proj_min_adj"] = pm_adj
        return prof


def build_profiles(logs, games, rosters, cfg, mctx=None):
    """Profiles for every player on tonight's rosters who has history."""
    df = prepare_logs(logs, cfg)
    opp_f = opponent_factors(df, cfg)
    disp0 = league_dispersion(df)
    last_by_team = team_last_game(df)

    latest = df.groupby("player_id").tail(1)
    name_to_id = {norm_name(r.player): r.player_id for r in latest.itertuples()}
    if rosters:
        team_of = {name_to_id[k]: t for k, t in rosters.items() if k in name_to_id}
    else:  # fallback: whoever last played for the team
        team_of = dict(zip(latest["player_id"], latest["team"]))

    by_player = dict(tuple(df.groupby("player_id")))
    cur_games = df[df["season"] == cfg["season"]["current"]].groupby("team")["game_id"].nunique().to_dict()
    profiles = []
    for g in games:
        for team, opp, home in ((g["home"], g["away"], True), (g["away"], g["home"], False)):
            for pid, t in team_of.items():
                if t != team:
                    continue
                prof = player_profile(by_player[pid], team, opp, g["date"], cfg, opp_f, disp0, last_by_team)
                if prof:
                    prof.update(game_id=g["game_id"], home=home, tipoff_utc=g.get("tipoff_utc"),
                                matchup=f'{g["away"]} @ {g["home"]}', team_games_cur=cur_games.get(team, 0))
                    if mctx is not None:
                        mctx.apply(prof, g)
                    profiles.append(prof)
    return profiles, {"opp_factors": opp_f, "dispersion": disp0}
