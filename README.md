# NBA Prop Engine

Every afternoon this repo builds a board of NBA player props, picks 3-, 4-, 5- and
6-leg lineups, explains every leg, and keeps an honest scoreboard of how the picks did.

**Dashboard:** `https://<your-github-username>.github.io/nba-prop-engine/`

## What you get each day

**Two cards a night, 3 to 6 props each, at the real DraftKings line.** Lines come free from
DraftKings via ESPN (points, rebounds, assists, threes, and the P+R / P+A / R+A / P+R+A combos).

| Check | Most likely | Fair price |
|---|---|---|
| DraftKings price | any | -150 or better |
| Chance it hits (our model blended with DraftKings' price) | 58%+ | 52%+ |
| Cleared this exact line recently | 7+ of last 10 | 6+ of last 10 |
| Tonight's matchup (who's out, defense, pace, minutes) | not against | +1% or better |
| Backtest: legs hit / typical price | ~67% / -175 | ~57% / -126 |

Both need the player gates: healthy, steady minutes, 3+ games with his current team.
Neither card beat DraftKings on profit by more than noise in the backtest (about +-2% per leg).

**PrizePicks card (third tab).** PrizePicks lines come through The Odds API (region `us_dfs`,
the licensed route; PrizePicks itself is not scraped). Every PrizePicks pick pays the same,
so a pick is good when the true chance of More/Less at PrizePicks' line is high. That chance
is worked out from DraftKings' price on the same player and stat (a distribution matched to
DraftKings' line and price, evaluated at PrizePicks' line), blended with our model. Edges come
mostly from PrizePicks lines that lag DraftKings'. Picks without a DraftKings line to anchor
to are skipped. Bars: 57%+ chance, 6+ of last 10, matchup not against, model within 10 pts of
the DraftKings-anchored chance. There is no PrizePicks line history to backtest yet, so the
live track record is the test. Uses about 1 Odds API credit per stat per game; the engine
spreads the monthly credits automatically (free plan: ~2 stats a night).

Each card takes every qualifying leg, best first, up to 6 (one per player, at most 2 per game).
If fewer than 3 qualify, it is filled to 3 with the next-best legs, marked **below bar**.

At the real line nothing is a sure thing: books set it near 50/50. In the backtest at real
DraftKings closing lines, "Most likely" legs hit about 67%, so a 3-leg card wins roughly 30%
of the time and a 6-leg card under 10%; "Fair price" legs hit about 57%. The dashboard shows the combined chance on every card,
and the track record shows whether the numbers hold up.

## How a pick is made

1. **Data.** Every finished NBA box score is stored in `data/gamelogs.csv.gz`
   (from NBA's public CDN). The first run downloads all of last season as a baseline.
2. **Production per minute.** Recent games weigh more; last season counts at half weight.
3. **Minutes.** The biggest driver of any prop. Weighted to the last ~5 games, blended with
   last season early on, trimmed when the spread signals a blowout. Volatile-minute players are excluded.
4. **Opponent.** Steals, blocks and turnovers use how much the opponent allows vs league average,
   shrunk toward neutral until there are enough games to trust it.
5. **Probability.** A player-specific distribution (negative binomial) turns the projection
   into a probability for any line.
6. **Market check.** The model's chance at the DraftKings line is blended with DraftKings'
   own (no-vig) price. The blend was learned from last season's real lines and results
   (`scripts/backtest_lines.py`): the market carries ~90% of the weight, because on its own
   the model was overconfident at real lines (legs it rated 60-80% hit only 50-56%).
7. **Matchup checks.** Each stat has its own (see `engine/matchups.py`):

   | Stat | Checks |
   |---|---|
   | Rebounds | position defense, opponent rebounding share, rebound chances, opponent frontcourt height |
   | Assists | position defense, assisted-shot rate allowed, Vegas team total, turnovers forced |
   | Points / FG made | position defense, shooting allowed (eFG%), rim protection, Vegas team total |
   | FG attempts | position defense, Vegas game total, turnovers forced |
   | 3PM / 3PA | position defense, 3s allowed, 3P% allowed, Vegas total |
   | FTM / FTA | position defense, FT rate allowed, opponent fouling, home/away |
   | All stats | **teammates out** (their share of shots/assists/rebounds redistributed), **opponent players out** |
   | Minutes (all) | blowout risk from the spread, minutes freed by absent teammates (more if same position), back-to-backs |

   Also tested and found to have no measurable effect last season (kept at weight 0, re-tested
   every run): altitude, extra rest (3+ days), opponent on a back-to-back.

   Weights are re-learned from the box scores every run. A check counts only if it clearly
   predicted results in past games *and* pointed the way basketball sense says it should;
   others (e.g. opponent height, opponent fouls last season) are shown as context but not counted.
   **How a Safe line is set:** from the player's *own* numbers (usual minutes x usual production).
   Tonight's matchup, including who's out, must then favor it by at least +1%. The matchup acts as
   extra cushion instead of pushing the line higher. Backtest: `python scripts/backtest.py`.
8. **Gates.** Injured / questionable / low-minute / thin-history players are dropped.
9. **Lineups.** One leg per team, at most two per game, never padded with weaker legs.

Every threshold lives in [`config.yaml`](config.yaml) with a comment explaining it.

## Validation

`python tests/simulate.py` runs the whole pipeline on a simulated league where the true
talent of every player is known. Latest result over 28 simulated days:

| Pick type | Predicted | Actual |
|---|---|---|
| Safe (X+) | 88.4% | 88.0% |
| Market, Strong | 56.2% | 54.7% |
| Market, Lock | 59.0% | 56.8% |

### Matchup backtest (last season, weights learned only from earlier games)

| Test period | Original engine | Matchups (v2) | + teammates/opponents out, line from own numbers (v3, used) |
|---|---|---|---|
| Jan 1 - Apr | 86.6% (108/day) | 88.0% (31/day) | 88.9% (38/day) |
| Feb 1 - Apr | 86.5% (95/day) | 88.3% (30/day) | 89.2% (33/day) |
| Mar 1 - Apr | 87.0% (88/day) | 89.2% (28/day) | 90.4% (29/day) |

Also tested and rejected: requiring 2+ individual checks to each pass a bar (~4 picks/day,
85.3%), and letting the matchup raise the line (87.1-88.7%).

## Setup (one time)

1. DraftKings lines come free through ESPN. The `ODDS_API_KEY` secret is only used for the
   PrizePicks card (The Odds API, region `us_dfs`).
2. **Dashboard:** *Settings → Pages →* Source: *Deploy from a branch*, Branch: `main`, folder `/docs`.
3. **First run:** *Actions → Daily picks → Run workflow*. The first run takes ~10 minutes
   (downloads last season). After that it runs by itself every day at ~5pm Eastern.

## Running cost: $0

| Part | Cost | How |
|---|---|---|
| Box scores, rosters, injuries, schedules, Vegas spreads | free | ESPN public feeds |
| DraftKings lines (Most likely and Fair price cards) | free | ESPN's odds feed, no key |
| Engine runs (3 a day) | free | GitHub Actions, unlimited for public repos |
| Dashboard | free | GitHub Pages |
| PrizePicks lines (shadow test, PrizePicks card) | free | The Odds API free plan, 500 credits/month |

Only the PrizePicks lines use credits (1 per stat per game). To stay inside the free 500:
lines are downloaded **once a day** (later runs reuse them), and each day spends only its share
of what's left in the month (about 15 credits a day). In `mode: shadow` the free-throw / FG-made
test takes those credits first, on about 5 rotating games a night; the PrizePicks card only gets
a stat when there are enough left to cover the whole slate. Settings: `config.yaml` -> `pickem`.

### Shadow test (free throws made, free throw attempts, FG made)

DraftKings doesn't post these, so there's no sharp price to check the model against, and on its
own the model was overconfident at real lines. The shadow test records the model's pick on every
PrizePicks line for these stats, grades it, and shows the results on the dashboard ("Free throw /
FG test") without ever putting them on a card. If the percentages hold over a few weeks
(~1,000 graded picks), they can earn a card; if not, they stay off.
