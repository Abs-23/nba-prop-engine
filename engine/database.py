"""Our own game-log database: one row per player per game, built from box scores.

Stored as data/gamelogs.csv.gz inside the repo, so every daily run starts
from yesterday's database and only downloads the games it's missing.
"""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from engine import DATA
from engine.sources import nba_cdn

PATH = DATA / "gamelogs.csv.gz"


def load():
    if PATH.exists():
        df = pd.read_csv(PATH, dtype={"game_id": str})
        df["game_id"] = df["game_id"].str.zfill(10)
        return df
    return pd.DataFrame(columns=["game_id", "date", "player_id"])


def save(df):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(["game_id", "player_id"]).sort_values(["date", "game_id"])
    df.to_csv(PATH, index=False, compression="gzip")


def _fetch_many(game_ids, workers=8):
    rows, missing = [], []

    def one(gid):
        try:
            return gid, nba_cdn.fetch_box(gid)
        except nba_cdn.NotFound:
            return gid, "notfound"
        except Exception:
            return gid, None

    with ThreadPoolExecutor(workers) as ex:
        for gid, res in ex.map(one, game_ids):
            if isinstance(res, list):
                rows.extend(res)
            else:
                missing.append((gid, res))
    return rows, missing


def update_from_schedule(df, schedule, today_iso):
    """Download every finished game from the schedule that we don't have yet."""
    have = set(df["game_id"]) if len(df) else set()
    todo = [g["game_id"] for g in schedule if g["date"] < today_iso and g["game_id"] not in have]
    if not todo:
        return df, 0
    rows, _ = _fetch_many(todo)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df, len({r["game_id"] for r in rows})


def backfill_season(df, id_prefix, max_games=1230):
    """Download a past season's regular season by walking game IDs 00{yy}00001.."""
    have = set(df["game_id"]) if len(df) else set()
    ids = [f"002{id_prefix}{i:05d}" for i in range(1, max_games + 1)]
    todo = [g for g in ids if g not in have]
    if not todo:
        return df, 0
    rows, missing = _fetch_many(todo)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    print(f"backfill {id_prefix}: {len({r['game_id'] for r in rows})} games added, {len(missing)} unavailable")
    return df, len({r["game_id"] for r in rows})


def season_of(game_id):
    return str(game_id)[3:5]
