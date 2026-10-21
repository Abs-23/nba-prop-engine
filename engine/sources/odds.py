"""Player prop lines from The Odds API (the-odds-api.com).

Cost model: listing events is free; each event's props cost
(number of markets x number of regions) credits. The free plan has
a small monthly allowance, so we spread credits evenly over the
days left in the month and drop low-priority markets first.
"""
import calendar
import os
from collections import Counter, defaultdict
from datetime import date

import requests

from engine.stats import MARKET_TO_STAT, TEAM_NAMES, norm_name

BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"


def american_to_prob(price):
    price = float(price)
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def american_to_decimal(price):
    price = float(price)
    return 1 + (price / 100 if price > 0 else 100 / -price)


class OddsClient:
    def __init__(self, api_key=None, regions="us"):
        self.key = api_key or os.environ.get("ODDS_API_KEY")
        self.regions = regions
        self.remaining = None
        self.used_this_run = 0

    @property
    def enabled(self):
        return bool(self.key)

    def _get(self, path, **params):
        r = requests.get(f"{BASE}{path}", params={"apiKey": self.key, **params}, timeout=30)
        if "x-requests-remaining" in r.headers:
            self.remaining = float(r.headers["x-requests-remaining"])
        if "x-requests-last" in r.headers:
            self.used_this_run += float(r.headers["x-requests-last"])
        r.raise_for_status()
        return r.json()

    def events(self):
        """Upcoming NBA games (free call). Also reads how many credits are left."""
        return self._get("/events")

    def event_props(self, event_id, markets):
        return self._get(f"/events/{event_id}/odds", regions=self.regions,
                         markets=",".join(markets), oddsFormat="american")


def plan_markets(remaining, n_games, markets, reserve, max_daily, today=None):
    """How many markets per game we can afford today.

    Budget = credits left (minus a reserve) spread over the remaining days
    of the month, so we never run dry before credits reset on the 1st.
    """
    today = today or date.today()
    if remaining is None or n_games == 0:
        return []
    days_left = calendar.monthrange(today.year, today.month)[1] - today.day + 1
    budget = min(max_daily, max(0.0, (remaining - reserve) / days_left))
    per_game = int(budget // n_games)
    return markets[:max(0, per_game)]


def consensus_lines(event_json):
    """Collapse every bookmaker's prop prices into one consensus row per player+stat.

    - line: the point most books are posting (books occasionally differ by 0.5-1)
    - fair_over: average no-vig probability of the over across books at that line
    - best over/under price: the most generous price available at that line
    """
    home = TEAM_NAMES.get(event_json.get("home_team"), event_json.get("home_team"))
    away = TEAM_NAMES.get(event_json.get("away_team"), event_json.get("away_team"))
    quotes = defaultdict(list)  # (player, stat) -> [(book, point, over_price, under_price)]
    for bk in event_json.get("bookmakers", []):
        for mk in bk.get("markets", []):
            stat = MARKET_TO_STAT.get(mk["key"])
            if not stat:
                continue
            pairs = defaultdict(dict)
            for oc in mk.get("outcomes", []):
                player = oc.get("description")
                side = str(oc.get("name", "")).lower()
                if player and side in ("over", "under") and oc.get("point") is not None:
                    pairs[(player, float(oc["point"]))][side] = oc["price"]
            for (player, point), sides in pairs.items():
                if "over" in sides and "under" in sides:
                    quotes[(player, stat)].append((bk["title"], point, sides["over"], sides["under"]))

    rows = []
    for (player, stat), qs in quotes.items():
        line = Counter(q[1] for q in qs).most_common(1)[0][0]
        at_line = [q for q in qs if q[1] == line]
        fair = []
        for _, _, op, up in at_line:
            po, pu = american_to_prob(op), american_to_prob(up)
            fair.append(po / (po + pu))
        best_over = max(at_line, key=lambda q: american_to_decimal(q[2]))
        best_under = max(at_line, key=lambda q: american_to_decimal(q[3]))
        rows.append({
            "player": player, "player_key": norm_name(player), "stat": stat,
            "line": line, "books": len(at_line),
            "fair_over": sum(fair) / len(fair),
            "best_over_price": best_over[2], "best_over_book": best_over[0],
            "best_under_price": best_under[3], "best_under_book": best_under[0],
            "home": home, "away": away,
        })
    return rows
