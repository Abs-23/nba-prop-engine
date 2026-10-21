# NBA Prop Engine

Every afternoon this repo builds a board of NBA player props, picks 3-, 4-, 5- and
6-leg lineups, explains every leg, and keeps an honest scoreboard of how the picks did.

**Dashboard:** `https://<your-github-username>.github.io/nba-prop-engine/`

## What you get each day

| Section | What it is |
|---|---|
| **Safest lineups** | "X or more" legs (e.g. *Jokić 9+ rebounds*, *Brunson 5+ FT attempts*). Each leg ≥85% by the model **and** cleared in ≥85% of the player's last 20 games. |
| **Value lineups** | Over/unders against real sportsbook lines that beat the books' own no-vig price. |
| **Reasons** | Every leg lists the facts behind it: **+** supports, **!** risk, **i** context. |
| **Track record** | Hit rate by tier, and a check that "85%" picks really hit ~85% of the time. |

No pick is a guarantee. Six legs at 90% each still lose almost half the time (0.9⁶ = 53%).
The dashboard always shows the combined probability so the risk is in plain sight.

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
6. **Market check.** For sportsbook lines, the final probability blends model (35%)
   and market (65%), because sportsbooks are sharp and the model's biggest disagreements
   with them are often the model's mistakes. A pick needs a 4+ point edge after blending.
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

1. **Odds key:** sign up at the-odds-api.com, then in this repo go to
   *Settings → Secrets and variables → Actions → New repository secret*,
   name it `ODDS_API_KEY`, paste the key. (Without it, Safe picks still work.)
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
