"""Run the engine at the right moments, timed to today's first tip-off.

GitHub starts scheduled workflows late (often 2-3 hours) and sometimes skips them, so we
don't rely on its clock. The workflow wakes up several times a day; each wake-up:
  1. looks up today's first tip-off,
  2. waits until the next step that hasn't been done yet:
       build  = first tip minus 2.5 hours (cards built)
       final  = first tip minus 1 hour   (cards refreshed with late injury news and late-posted
                                          lines; PrizePicks lines downloaded, once a day)
  3. runs the engine, saves, and marks the step done so later wake-ups skip it.
A wake-up that would have to wait longer than GitHub allows (~6 hours) exits and leaves the
step to a later wake-up. Days without games just update the database and grade picks.

    python scripts/wait_and_run.py          # scheduled use
    python scripts/wait_and_run.py --now    # run once immediately (manual button)
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import DATA, load_config  # noqa: E402
from engine.sources import espn  # noqa: E402

MARKS = DATA / "run_marks.json"
MAX_WAIT = timedelta(hours=5, minutes=20)   # GitHub jobs are cut off at 6 hours


def now():
    return datetime.now(timezone.utc)


def sh(*cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def save(label):
    """Commit and push data + dashboard (GitHub Actions only)."""
    sh("git", "config", "user.name", "nba-prop-engine")
    sh("git", "config", "user.email", "actions@users.noreply.github.com")
    sh("git", "add", "data", "docs")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
        sh("git", "commit", "-q", "-m", f"Picks ({label})")
    for _ in range(3):
        sh("git", "pull", "-q", "--rebase", "-X", "theirs", check=False)
        if sh("git", "push", "-q", check=False).returncode == 0:
            return
        time.sleep(10)


def marks(today):
    m = json.loads(MARKS.read_text()) if MARKS.exists() else {}
    return m.get(today, [])


def mark(today, step):
    m = json.loads(MARKS.read_text()) if MARKS.exists() else {}
    m = {k: v for k, v in m.items() if k >= (datetime.fromisoformat(today) - timedelta(days=7)).date().isoformat()}
    m.setdefault(today, []).append(step)
    MARKS.write_text(json.dumps(m, indent=1))


def run_engine(label, push):
    ok = sh(sys.executable, "run_daily.py", check=False).returncode == 0
    if push:
        save(label)
    return ok


def main(run_now, push):
    cfg = load_config()["schedule"]
    started_at = now()
    today = started_at.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    if run_now:
        print("Manual run: running once now.")
        return run_engine("manual", push)

    games = [g for g in espn.scoreboard(today) if not g["completed"]]
    tips = sorted(datetime.fromisoformat(g["tipoff_utc"].replace("Z", "+00:00")) for g in games if g.get("tipoff_utc"))
    if not tips:
        if "nogames" not in marks(today):
            print("No games left today: updating the database and grading only.")
            mark(today, "nogames")
            return run_engine("no games", push)
        return True
    first = tips[0]
    steps = [("build", first - timedelta(minutes=cfg["build_minutes_before_first_tip"])),
             ("final", first - timedelta(minutes=cfg["final_minutes_before_first_tip"]))]
    et = ZoneInfo("America/New_York")
    print(f"First tip-off today: {first.astimezone(et):%I:%M %p} ET. "
          + "; ".join(f"{s} at {t.astimezone(et):%I:%M %p} ET" for s, t in steps))
    for step, at in steps:
        if step in marks(today):
            print(f"{step}: already done today.")
            continue
        if now() >= first:
            print(f"{step}: first game already started; running once so late games are covered.")
        elif at > now():
            wait = at - now()
            if (now() - started_at) + wait > MAX_WAIT:
                print(f"{step}: {wait} away, longer than this wake-up can wait. Leaving it to a later one.")
                return True
            print(f"{step}: waiting {wait} until {at.astimezone(et):%I:%M %p} ET")
            time.sleep(wait.total_seconds())
        mark(today, step)
        run_engine(step, push)
        if now() >= first:
            break
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true", help="run once immediately")
    ap.add_argument("--no-push", action="store_true", help="don't commit/push (local use)")
    a = ap.parse_args()
    main(a.now, not a.no_push)
