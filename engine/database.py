"""Our own game-log database: one row per player per game, built from ESPN box scores.

Stored as data/gamelogs.csv.gz inside the repo, so every daily run starts
from yesterday's database and only downloads the games it's missing.
"""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from engine import DATA
from engine.sources import espn

PATH = DATA / "gamelogs.csv.gz"
GAMES = DATA / "games.csv"          # one row per game: spread, total, final score
PLAYERS = DATA / "players.csv"      # height and position from current rosters


def load():
    if PATH.exists():
        return pd.read_csv(PATH, dtype={"game_id": str})
    return pd.DataFrame(columns=["game_id", "date", "player_id"])


def save(df):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(["game_id", "player_id"]).sort_values(["date", "game_id"])
    df.to_csv(PATH, index=False, compression="gzip")


def season_key(day_iso):
    """'2026-11-03' -> '26' (the 2026-27 season)."""
    return str(espn.season_start_year(str(day_iso)[:10]))[2:]


def add_season(df):
    df = df.copy()
    df["season"] = df["date"].astype(str).map(season_key) if len(df) else []
    return df


def load_games():
    if GAMES.exists():
        return pd.read_csv(GAMES, dtype={"game_id": str})
    return pd.DataFrame(columns=["game_id", "date", "home", "away", "spread", "total", "home_score", "away_score"])


def save_games(gm):
    gm.drop_duplicates("game_id", keep="last").sort_values("date").to_csv(GAMES, index=False)


def fetch_boxes(game_ids, workers=8):
    def one(gid):
        try:
            return espn.box(gid)
        except Exception:
            return None

    rows, games, missing = [], [], 0
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(one, game_ids):
            if res:
                rows.extend(res[0])
                games.append(res[1])
            else:
                missing += 1
    if games:
        save_games(pd.concat([load_games(), pd.DataFrame(games)], ignore_index=True))
    return rows, missing


def add_games(df, games):
    """Download box scores for completed games we don't have yet."""
    have = set(df["game_id"].astype(str)) if len(df) else set()
    todo = [g["game_id"] for g in games if g["completed"] and g["game_id"] not in have]
    if not todo:
        return df, 0, 0
    rows, missing = fetch_boxes(todo)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df, len({r["game_id"] for r in rows}), missing


def load_players():
    if PLAYERS.exists():
        return pd.read_csv(PLAYERS)
    return pd.DataFrame(columns=["player_id", "player", "team", "height", "pos"])


def save_players(rows):
    """Keep every player we've ever seen on a roster (heights don't change)."""
    pl = pd.concat([load_players(), pd.DataFrame(rows)], ignore_index=True)
    pl.drop_duplicates("player_id", keep="last").to_csv(PLAYERS, index=False)
