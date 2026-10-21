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

1. No odds key needed any more: DraftKings lines come free through ESPN.
2. **Dashboard:** *Settings → Pages →* Source: *Deploy from a branch*, Branch: `main`, folder `/docs`.
3. **First run:** *Actions → Daily picks → Run workflow*. The first run takes ~10 minutes
   (downloads last season). After that it runs by itself every day at ~5pm Eastern.

## Odds API credits

The free plan has a small monthly allowance. Each game's props cost
(markets × regions) credits, so the engine spreads what's left over the rest of the month
and drops the lowest-priority markets first (order in `config.yaml`). For full coverage
of all 8 markets every day, the paid tier is needed.

## Roadmap

- Teammate-out usage boost (with/without splits from the database)
- Blowout risk from game spreads
- Position-level defense (e.g. rebounds allowed to centers)
- Bet-size guidance once the track record proves an edge

*For entertainment and analysis. If you bet, bet only what you can afford to lose.*
