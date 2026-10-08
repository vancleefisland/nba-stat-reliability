"""Central config. Each stat is a ratio num/den we want to know how fast it stabilizes."""

RAW_PATH = "data/raw/game_logs.csv"

# name -> numerator column, denominator column (one row per player-game in the CSV).
# 3P% and FT% are true binomial rates. "AST rate" uses a possessions-style denominator
# that YOU must supply (see README); beta-binomial shrinkage is only an approximation there.
STATS = {
    "3P%":      {"num": "fg3m", "den": "fg3a"},
    "FT%":      {"num": "ftm",  "den": "fta"},
    "AST rate": {"num": "ast",  "den": "poss"},
}

REQUIRED_COLS = ["player", "season", "game_date"] + sorted(
    {c for s in STATS.values() for c in (s["num"], s["den"])}
)

# Reliability study
N_GRID = [5, 10, 15, 20, 25, 30, 35]   # games per half-sample
N_SPLITS = 30                          # random split-halves per (season, N)
MIN_PLAYERS = 25                       # skip a cell with fewer eligible players

# Shrinkage evaluation: use first N games to predict the rest of the season
EVAL_CUTOFFS = [10, 20, 30, 40]
MIN_REMAINING_GAMES = 20

INTERVAL_LEVEL = 0.80
SEASON_GAMES = 70      # typical games for a rotation player, used in the trust guide
SEED = 42
