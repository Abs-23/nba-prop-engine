"""Collect DraftKings closing prop lines for every finished game in the database
(where ESPN still has them) into data/props_history.csv.gz. Used for backtesting
at real market lines. Safe to re-run: only fetches games not collected yet."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import DATA, database  # noqa: E402
from engine.sources.espn_props import fetch_lines  # noqa: E402

PATH = DATA / "props_history.csv.gz"
DONE = DATA / "props_checked.txt"   # games already tried (with or without lines)

games = database.load_games()
tried = set(DONE.read_text().split()) if DONE.exists() else set()
todo = [g for g in games["game_id"].astype(str) if g not in tried]
print(f"{len(todo)} games to check")
rows = []
for i in range(0, len(todo), 100):
    chunk = todo[i:i + 100]
    rows += fetch_lines(chunk)
    tried.update(chunk)
    print(f"  {min(i + 100, len(todo))}/{len(todo)} checked, {len(rows)} lines so far", flush=True)
new = pd.DataFrame(rows)
old = pd.read_csv(PATH, dtype={"game_id": str}) if PATH.exists() else pd.DataFrame()
allp = pd.concat([old, new], ignore_index=True)
if len(allp):
    allp = allp.drop_duplicates(["game_id", "player_id", "stat"], keep="last")
    allp.to_csv(PATH, index=False, compression="gzip")
DONE.write_text("\n".join(sorted(tried)))
n_games = allp["game_id"].nunique() if len(allp) else 0
print(f"::notice title=props history::{len(allp)} lines across {n_games} games")
