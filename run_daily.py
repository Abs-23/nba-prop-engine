"""The daily run. GitHub runs this every afternoon; you can also run it by hand:

    python run_daily.py

Steps
  1. Update the game-log database with every finished game (first run also
     downloads all of last season as a baseline).
  2. Grade yesterday's picks and lineups.
  3. Pull tonight's schedule, rosters and injury report.
  4. Project every rotation player, pull sportsbook lines, apply the gates.
  5. Build 3-6 leg lineups, save everything, publish the dashboard.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from engine import database, grading, load_config
from engine.dashboard import render
from engine.lineups import build_all
from engine.model import build_profiles
from engine.picks import market_picks, safe_picks
from engine.sources import nba_cdn
from engine.sources.injuries import fetch_injuries
from engine.sources.odds import OddsClient, consensus_lines, plan_markets
from engine.stats import TEAM_NAMES


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main(day=None):
    cfg = load_config()
    today = day or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    health = []

    # 1. database
    db = database.load()
    prior = cfg["season"]["prior_id_prefix"]
    n_prior = db.loc[db["game_id"].map(database.season_of) == prior, "game_id"].nunique() if len(db) else 0
    if n_prior < 1200:
        log(f"Backfilling {cfg['season']['prior']} (one-time, ~10 minutes)...")
        db, _ = database.backfill_season(db, prior)
    schedule = nba_cdn.fetch_schedule()
    db, added = database.update_from_schedule(db, schedule, today)
    database.save(db)
    n_games = db["game_id"].nunique()
    log(f"Database: {n_games} games ({added} new)")
    health.append({"ok": True, "text": f"{n_games:,} games in database"})

    # 2. grade
    grading.grade(db)

    # 3. tonight
    games = [g for g in schedule if g["date"] == today]
    log(f"{len(games)} games today")
    rosters = None
    injuries = None
    profiles, safe, market, lineups = [], [], [], {"safest": [], "value": []}
    odds = OddsClient(regions=cfg["odds_api"]["regions"])
    if games:
        from engine.sources.rosters import fetch_rosters
        rosters = fetch_rosters()
        health.append({"ok": rosters is not None,
                       "text": "Rosters loaded" if rosters else "Rosters unavailable (using last team played for)"})
        injuries = fetch_injuries()
        health.append({"ok": injuries is not None,
                       "text": f"Injury report: {len(injuries)} listed" if injuries is not None else "Injury report unavailable"})

        # 4. project + pick
        profiles, _ = build_profiles(db, games, rosters, cfg)
        log(f"Projected {len(profiles)} players")
        safe = safe_picks(profiles, injuries, cfg)

        lines = []
        if odds.enabled:
            try:
                events = odds.events()
                todays = {(g["home"], g["away"]) for g in games}
                events = [e for e in events
                          if (TEAM_NAMES.get(e["home_team"]), TEAM_NAMES.get(e["away_team"])) in todays]
                oc = cfg["odds_api"]
                markets = plan_markets(odds.remaining, len(events), oc["markets"], oc["reserve_credits"],
                                       oc["max_daily_credits"])
                log(f"Odds: {odds.remaining} credits left, fetching {len(markets)} markets x {len(events)} games")
                if markets:
                    for e in events:
                        lines += consensus_lines(odds.event_props(e["id"], markets))
                health.append({"ok": bool(markets),
                               "text": f"Odds: {len(markets)} markets, {int(odds.remaining or 0)} credits left"})
            except Exception as ex:
                log(f"Odds API error: {ex}")
                health.append({"ok": False, "text": "Odds API error"})
        else:
            health.append({"ok": False, "text": "No odds key: market picks off"})
        market, skipped = market_picks(profiles, lines, injuries, cfg)
        log(f"Picks: {len(safe)} safe, {len(market)} market ({len(lines)} lines checked)")

        # 5. lineups + save
        lineups = build_all(safe, market, cfg)
        grading.record_today(today, safe, market, lineups)

    render(today, len(games), safe, market, lineups, grading.track_record(), health, odds.enabled)
    log("Dashboard written to docs/index.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, US Eastern)")
    main(ap.parse_args().date)
