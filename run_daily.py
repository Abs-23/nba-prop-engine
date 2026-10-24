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
import json
import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from engine import DATA, DOCS, database, fit, grading, load_config
from engine.stats import norm_name
from engine.dashboard import render
from engine.model import MatchupContext, build_profiles

from engine.sources import espn
from engine.sources.injuries import fetch_injuries


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def save_line_snapshot(lines, day):
    """Keep tonight's lines (as seen at run time) for grading and future backtests."""
    import pandas as pd
    if not lines:
        return
    path = DATA / "props_history.csv.gz"
    new = pd.DataFrame(lines).assign(snapshot=day)
    old = pd.read_csv(path, dtype={"game_id": str}) if path.exists() else pd.DataFrame()
    allp = pd.concat([old, new], ignore_index=True).drop_duplicates(["game_id", "player_id", "stat"], keep="last")
    allp.to_csv(path, index=False, compression="gzip")


def save_pickem_snapshot(pp_lines, dk_lines, day):
    """Keep every PrizePicks line next to DraftKings' line for the same player/stat (for later study)."""
    import pandas as pd
    if not pp_lines:
        return
    path = DATA / "pickem_history.csv.gz"
    new = pd.DataFrame(pp_lines).assign(snapshot=day)
    old = pd.read_csv(path, dtype={"game_id": str}) if path.exists() else pd.DataFrame()
    allp = pd.concat([old, new], ignore_index=True).drop_duplicates(
        ["game_id", "player_key", "stat", "book_key"], keep="last")
    allp.to_csv(path, index=False, compression="gzip")


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
    # Lock: once tonight's cards exist and a game has tipped off, never rebuild them
    # (lines and injury news after tip-off would silently change the cards you already saw).
    now_utc = datetime.now(timezone.utc)
    started = [g for g in games if g.get("tipoff_utc") and
               datetime.fromisoformat(g["tipoff_utc"].replace("Z", "+00:00")) <= now_utc]
    if started:
        print("::notice title=games started::" + "; ".join(f'{g["away"]}@{g["home"]} {g["tipoff_utc"]}' for g in started)
              + f" (now {now_utc:%H:%M} UTC)")
    locked = {}
    if started and grading.has_cards(today):
        # Published cards stay exactly as they were; only cards not yet published get built.
        prev = json.loads((DOCS / "picks.json").read_text()) if (DOCS / "picks.json").exists() else {}
        if prev.get("date") == today:
            locked = {k: v for k, v in prev.get("lineups", {}).items() if v}
        wanted = set(cfg["card"]["cards"]) | ({"prizepicks"} if os.environ.get("ODDS_API_KEY") else set())
        if wanted <= set(locked):
            log(f"{len(started)} games already started and all of tonight's cards exist: locked, nothing rebuilt.")
            return
        log(f"Locked cards kept as published: {sorted(locked)}; building the rest from games not started.")
    if started:
        log(f"{len(started)} games already started: building from the {len(games) - len(started)} that haven't.")
        games = [g for g in games if g not in started]
    rosters = None
    injuries = None
    profiles, safe, market, lineups = [], [], [], {}
    out_tonight = {}
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
        # 5. tonight's DraftKings lines (free, via ESPN) -> the card
        from engine.card import build_cards, candidates
        from engine.sources.espn_props import fetch_lines
        lines = fetch_lines([g["game_id"] for g in games])
        save_line_snapshot(lines, today)
        health.append({"ok": bool(lines), "text": f"DraftKings lines: {len(lines)} for {len({l['game_id'] for l in lines})} games"
                       if lines else "DraftKings lines unavailable"})
        cands = candidates(profiles, lines, injuries, cfg)
        lineups, market = build_cards(cands, cfg, injuries)
        log(f"Lines: {len(lines)}; candidates: {len(cands)}; qualified: {len(market)}; cards: "
            + ", ".join(f"{k} {v[0]['size'] if v else 0} legs" for k, v in lineups.items()))
        # 6. PrizePicks (licensed via The Odds API): downloaded once a day to stay on the free plan
        from engine import pickem
        from engine.sources.odds import OddsClient
        client = OddsClient(regions="us_dfs")
        cache_path = DATA / "pickem_today.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        pp_lines = None
        if cache.get("date") == today:
            pp_lines = cache["lines"]
            log(f"PrizePicks: reusing today's {len(pp_lines)} lines (no credits spent)")
            health.append({"ok": True, "text": f"PrizePicks: {len(pp_lines)} lines (downloaded earlier today)"})
        elif client.enabled:
            try:
                events = pickem.tonight_events(client, games)
                plan = pickem.plan_free(client.remaining, events, cfg, today)
                pp_lines = pickem.fetch_planned(client, plan)
                cache_path.write_text(json.dumps({"date": today, "lines": pp_lines}))
                save_pickem_snapshot(pp_lines, lines, today)
                used = sum(len(m) for _, m in plan)
                health.append({"ok": True, "text": f"PrizePicks: {len(pp_lines)} lines, {used} credits used, "
                               f"{int(client.remaining or 0)} left this month"})
                log(f"PrizePicks: {len(pp_lines)} lines from {len(plan)} games; {used} credits; "
                    f"left {client.remaining}")
            except Exception as ex:
                log(f"PrizePicks lines unavailable: {ex}")
                health.append({"ok": False, "text": "PrizePicks lines unavailable"})
        else:
            health.append({"ok": False, "text": "No odds key: PrizePicks off"})
        if pp_lines:
            live = {g["game_id"] for g in games}
            pp_live = [l for l in pp_lines if l["game_id"] in live]
            shadow = pickem.shadow_picks(profiles, pp_live, injuries, cfg)
            market += shadow
            log(f"Shadow test: {len(shadow)} model-only picks recorded (free throws / FG made)")
            pcands = pickem.candidates(profiles, pp_live, lines, injuries, cfg)
            if pcands:
                pcard, pq = pickem.build_card(pcands, cfg, injuries)
                lineups["prizepicks"] = [pcard] if pcard else []
                market += pq
        if locked or started:
            # keep shadow picks already recorded today for games that have since started
            market += grading.todays_rows(today, kind="shadow", game_ids={g["game_id"] for g in started})
        if locked:
            ids = {norm_name(p["player"]): p["player_id"] for p in profiles}
            ids.update({norm_name(r.player): r.player_id for r in db.drop_duplicates("player_id", keep="last").itertuples()})
            kept = []
            for name, cs in locked.items():
                for leg in cs[0]["legs"]:
                    leg.setdefault("player_id", ids.get(norm_name(leg["player"])))
                    leg.setdefault("card", name)
                    leg.setdefault("kind", "card")
                    kept.append(leg)
                lineups[name] = cs
            market = [m for m in market if m.get("card") not in locked] + kept
        fills = [l for v in lineups.values() for c in v for l in c["legs"] if l["below_bar"]]
        grading.record_today(today, [], market + fills, lineups)

    render(today, len(games), safe, [m for m in market if m.get("kind") != "shadow"], lineups, grading.track_record(), health, True,
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
