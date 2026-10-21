"""Stat definitions shared by every part of the engine."""
import re
import unicodedata

# Base box-score columns we store for every player-game.
BASE_STATS = ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
              "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]

# Every stat the engine can pick, as a sum of base columns.
STATS = {
    "pts": ["pts"], "reb": ["reb"], "ast": ["ast"],
    "fg3m": ["fg3m"], "fg3a": ["fg3a"], "fga": ["fga"], "fgm": ["fgm"],
    "ftm": ["ftm"], "fta": ["fta"], "stl": ["stl"], "blk": ["blk"], "tov": ["tov"],
    "pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"],
    "ra": ["reb", "ast"], "bs": ["blk", "stl"],
}

LABELS = {
    "pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3-Pointers Made",
    "fg3a": "3-Point Attempts", "fga": "FG Attempts", "fgm": "FG Made",
    "ftm": "Free Throws Made", "fta": "Free Throw Attempts", "stl": "Steals",
    "blk": "Blocks", "tov": "Turnovers", "pra": "Pts+Reb+Ast", "pr": "Pts+Reb",
    "pa": "Pts+Ast", "ra": "Reb+Ast", "bs": "Blocks+Steals",
}

# The Odds API market key -> our stat key.
MARKET_TO_STAT = {
    "player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
    "player_threes": "fg3m", "player_points_rebounds_assists": "pra",
    "player_points_rebounds": "pr", "player_points_assists": "pa",
    "player_rebounds_assists": "ra", "player_blocks": "blk", "player_steals": "stl",
    "player_blocks_steals": "bs", "player_turnovers": "tov",
    "player_field_goals": "fgm", "player_frees_made": "ftm",
    "player_frees_attempts": "fta",
}

# Stats that get "safe threshold" picks (the ones listed in config min_threshold).
SAFE_STATS = ["pts", "reb", "ast", "fg3m", "fg3a", "fga", "fgm", "ftm", "fta",
              "pra", "pr", "pa", "ra"]

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str) -> str:
    """'Luka Dončić' / 'P.J. Washington Jr.' -> 'luka doncic' / 'pj washington'."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = s.replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(p for p in s.split() if p not in _SUFFIXES)


def add_stat_columns(df):
    """Add combo columns (pra, pr, ...) to a game-log DataFrame."""
    for key, parts in STATS.items():
        if key not in df.columns:
            df[key] = df[parts].sum(axis=1)
    return df


TEAM_NAMES = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
