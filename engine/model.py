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
    if new_team and n_cur == 0:
        # last season's role on a different team says little about minutes now
        proj_min *= 0.95

    yesterday = (date.fromisoformat(game_date) - timedelta(days=1)).isoformat()
    b2b = last_game_by_team.get(team) == yesterday
    if b2b:
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


def build_profiles(logs, games, rosters, cfg):
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
    profiles = []
    for g in games:
        for team, opp, home in ((g["home"], g["away"], True), (g["away"], g["home"], False)):
            for pid, t in team_of.items():
                if t != team:
                    continue
                prof = player_profile(by_player[pid], team, opp, g["date"], cfg, opp_f, disp0, last_by_team)
                if prof:
                    prof.update(game_id=g["game_id"], home=home, tipoff_utc=g.get("tipoff_utc"),
                                matchup=f'{g["away"]} @ {g["home"]}')
                    profiles.append(prof)
    return profiles, {"opp_factors": opp_f, "dispersion": disp0}
