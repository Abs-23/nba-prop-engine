"""Who was missing, and how much of the team's production left with them.

Box scores don't say "injured", so an absence is inferred: a ROTATION player
(averaging 18+ minutes over his last 10 games, and played in one of the team's
last 5 games) whose team played without him. Players who were traded are not
counted as absent for their old team (their next game is for a new team).

For each absent player we know his recent SHARE of the team's shots, free
throws, threes, assists, rebounds, points and minutes. Summed over everyone
missing, that's the "vacated" share: e.g. 0.22 of the team's shots walked out
the door tonight. Teammates absorb it.

The factor that matters is the CHANGE vs the player's normal situation: if a
star has been out for 3 weeks, the player's recent numbers already include
the boost, so only a NEW absence (or a RETURN) moves the projection.
"""
import numpy as np
import pandas as pd

SHARE_STATS = ["pts", "reb", "ast", "fga", "fg3a", "fta", "min"]
ROT_MIN = 18.0
RECENT_TEAM_GAMES = 5
SHARE_WINDOW = 10


def player_shares(logs):
    """Per played player-game: his share of the team's totals, and as-of averages
    (previous SHARE_WINDOW games with this team) of those shares and of minutes."""
    df = logs[logs["played"].astype(bool) & (logs["min"] > 0)].copy()
    tot = df.groupby(["game_id", "team"])[SHARE_STATS].transform("sum")
    for s in SHARE_STATS:
        df[f"sh_{s}"] = df[s] / tot[s].replace(0, np.nan)
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    g = df.groupby(["player_id", "team"])
    for s in SHARE_STATS:
        df[f"asof_sh_{s}"] = g[f"sh_{s}"].transform(lambda x: x.shift(1).rolling(SHARE_WINDOW, 1).mean())
        df[f"cur_sh_{s}"] = g[f"sh_{s}"].transform(lambda x: x.rolling(SHARE_WINDOW, 1).mean())
    df["asof_min"] = g["min"].transform(lambda x: x.shift(1).rolling(SHARE_WINDOW, 1).mean())
    df["cur_min"] = g["min"].transform(lambda x: x.rolling(SHARE_WINDOW, 1).mean())
    return df


def find_absences(logs, shares=None):
    """Rows of (game_id, date, team, player_id, player, pos, share_* , min_avg) for inferred absences."""
    sh = shares if shares is not None else player_shares(logs)
    team_games = logs[["game_id", "date", "team"]].drop_duplicates().sort_values(["team", "date"])
    tg = {t: g.reset_index(drop=True) for t, g in team_games.groupby("team")}
    out = []
    # every player's appearances, in order, with the team he played for
    apps = sh.sort_values("date")[["player_id", "player", "pos", "team", "game_id", "date"] +
                                  [f"cur_sh_{s}" for s in SHARE_STATS] + ["cur_min"]]
    for pid, a in apps.groupby("player_id"):
        a = a.reset_index(drop=True)
        for i in range(len(a)):
            row = a.iloc[i]
            if row["cur_min"] < ROT_MIN:
                continue
            team = row["team"]
            games = tg[team]
            pos_now = games.index[games["game_id"] == row["game_id"]][0]
            # team games after this appearance, until his next appearance
            nxt = a.iloc[i + 1] if i + 1 < len(a) else None
            if nxt is not None and nxt["team"] != team:
                continue  # he left the team (trade/release): not an absence
            end_date = nxt["date"] if nxt is not None else "9999-12-31"
            # same season only
            season_end = f"{int(row['date'][:4]) + (1 if int(row['date'][5:7]) >= 8 else 0)}-08-01"
            missed = games[(games.index > pos_now) & (games["date"] < min(end_date, season_end))]
            for j, (_, gm) in enumerate(missed.iterrows()):
                rec = {"game_id": gm["game_id"], "date": gm["date"], "team": team, "player_id": pid,
                       "player": row["player"], "pos": row["pos"], "streak": j + 1}
                for s in SHARE_STATS:
                    rec[f"share_{s}"] = row[f"cur_sh_{s}"]
                rec["min_avg"] = row["cur_min"]
                out.append(rec)
    return pd.DataFrame(out)


def vacated(absent):
    """Per (game_id, team): total share of each stat missing, and missing minutes by position."""
    if absent is None or absent.empty:
        cols = (["game_id", "team"] + [f"vac_{s}" for s in SHARE_STATS]
                + [f"vac_min_{p}" for p in ("G", "F", "C")] + ["vac_names"])
        return pd.DataFrame(columns=cols)
    agg = {f"share_{s}": "sum" for s in SHARE_STATS}
    v = absent.groupby(["game_id", "team"]).agg(agg).reset_index()
    for s in SHARE_STATS:
        v[f"vac_{s}"] = v.pop(f"share_{s}").clip(upper=0.8)
    for pos in ("G", "F", "C"):
        m = absent[absent["pos"] == pos].groupby(["game_id", "team"])["min_avg"].sum()
        v = v.merge(m.rename(f"vac_min_{pos}").reset_index(), on=["game_id", "team"], how="left")
    names = absent.sort_values("share_pts", ascending=False).groupby(["game_id", "team"])["player"] \
        .apply(lambda x: list(x)[:4]).rename("vac_names").reset_index()
    v = v.merge(names, on=["game_id", "team"], how="left")
    return v.fillna({f"vac_min_{p}": 0.0 for p in ("G", "F", "C")})


def boost_x(v):
    """Vacated share -> log boost if the rest of the team absorbs it proportionally."""
    return -np.log(1 - np.clip(v, 0, 0.8))
