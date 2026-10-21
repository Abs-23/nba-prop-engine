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
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from engine import DATA, database, fit, grading, load_config
from engine.dashboard import render
from engine.lineups import build_all
from engine.model import MatchupContext, build_profiles
from engine.picks import market_picks, safe_picks
from engine.sources import espn
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
    if len(db) and not {"pos", "pf", "dreb"} <= set(db.columns):
        log("Database is missing matchup fields (position, fouls): rebuilding it from scratch.")
        db = db.iloc[0:0]
        for f in DATA.glob("backfill_*.done"):
            f.unlink()
    cur_year = espn.season_start_year(today)
    # Seasons are worked out from the date, so the engine rolls over each October by itself.
    cfg["season"] = {"current": str(cur_year)[2:], "prior": str(cur_year - 1)[2:]}
    marker = DATA / f"backfill_{cur_year - 1}.done"
    if not marker.exists():
        log(f"Backfilling the {cur_year - 1}-{str(cur_year)[2:]} season (one-time, ~10 minutes)...")
        past = espn.games_between(f"{cur_year - 1}-10-01", f"{cur_year}-06-30")
        db, added, missing = database.add_games(db, past)
        log(f"Backfill: {len(past)} games found, {added} downloaded, {missing} failed")
        database.save(db)
        if added > 1000:
            marker.write_text(f"{added} games\n")
    cur = db[db["date"].astype(str) >= f"{cur_year}-08-01"] if len(db) else db
    start = (date.fromisoformat(cur["date"].max()) - timedelta(days=3)).isoformat() if len(cur) \
        else f"{cur_year}-10-01"
    schedule = espn.games_between(start, today)
    db, added, missing = database.add_games(db, [g for g in schedule if g["date"] < today])
    database.save(db)
    n_games = db["game_id"].nunique()
    log(f"Database: {n_games} games ({added} new, {missing} not downloadable)")
    health.append({"ok": n_games > 1000, "text": f"{n_games:,} games in database"})

    # 2. grade
    grading.grade(db)

    # 3. tonight
    games = [g for g in schedule if g["date"] == today]
    log(f"{len(games)} games today")
    rosters = None
    injuries = None
    profiles, safe, market, lineups = [], [], [], {"safest": [], "value": []}
    out_tonight = {}
    odds = OddsClient(regions=cfg["odds_api"]["regions"])
    if games:
        from engine.sources.rosters import fetch_rosters
        roster_rows = []
        rosters = fetch_rosters(roster_rows)
        if roster_rows:
            database.save_players(roster_rows)
        health.append({"ok": rosters is not None,
                       "text": "Rosters loaded" if rosters else "Rosters unavailable (using last team played for)"})
        injuries = fetch_injuries()
        health.append({"ok": injuries is not None,
                       "text": f"Injury report: {len(injuries)} listed" if injuries is not None else "Injury report unavailable"})

        # 4. learn matchup weights from every game so far, then project + pick
        log("Learning matchup weights from past games...")
        frame = fit.build_frame(db, database.load_games(), database.load_players())
        weights = fit.fit_weights(frame, cfg["matchups"]["fit_min_t"])
        fit.save_weights(weights)
        used = sum(1 for v in weights["stats"].values() for f in v["factors"].values() if f["weight"])
        log(f"Weights: " + "; ".join(f"{s}: " + ", ".join(f"{k} {f['weight']:.2f}" for k, f in v["factors"].items()
                                                               if f["weight"]) for s, v in weights["stats"].items()))
        health.append({"ok": True, "text": f"Matchup checks: {used} weighted, from {weights['n_rows']:,} player-games"})
        mctx = MatchupContext(db, database.load_games(), database.load_players(), rosters, weights,
                              cfg["season"]["current"], injuries=injuries, frame=frame, today=today)
        profiles, _ = build_profiles(db, games, rosters, cfg, mctx)
        out_tonight = {t: [o["player"] for o in v] for t, v in mctx.out.items()}
        n_out = sum(len(v) for v in out_tonight.values())
        health.append({"ok": True, "text": f"{n_out} rotation players out tonight"})
        log("Out tonight: " + "; ".join(f"{t}: {', '.join(v)}" for t, v in sorted(out_tonight.items())))
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

    render(today, len(games), safe, market, lineups, grading.track_record(), health, odds.enabled,
           out_tonight if games else {})
    log("Dashboard written to docs/index.html")


if __name__ == "__main__":
    import sys
    import traceback

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, US Eastern)")
    try:
        main(ap.parse_args().date)
    except Exception:
        # Surface the full error as a GitHub annotation, readable from the run summary.
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        print("::error title=run_daily failed::" + tb[-3000:].replace("%", "%25").replace("\r", "").replace("\n", "%0A"))
        sys.exit(1)
