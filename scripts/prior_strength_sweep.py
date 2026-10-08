"""Which prior strength (kappa) predicts best out-of-sample?

For each stat and cutoff N: estimate skill from a player's first N games as
    (league_mean * kappa + makes) / (kappa + attempts)
and score it on the rest of the season (weighted by remaining attempts), exactly as in
src/evaluate.py, but sweeping kappa instead of fitting it. The kappa with the lowest error
is the one that actually predicts best, which is the practical tiebreaker between the
split-half and beta-binomial reliability estimates.

Run from the project root:  python scripts/prior_strength_sweep.py
"""
import numpy as np
import pandas as pd

PATH = "data/raw/game_logs.csv"
CUTOFFS = [20, 30, 40]
MIN_REMAINING_GAMES = 20
STATS = {  # stat: (makes col, attempts col, kappas to try)
    "3P%": ("fg3m", "fg3a", [50, 100, 175, 294, 366, 500, 800]),
    "FT%": ("ftm", "fta", [10, 25, 29, 61, 100, 200, 400]),
}

df = pd.read_csv(PATH, parse_dates=["game_date"]).sort_values(["season", "player", "game_date"])

for stat, (num, den, kappas) in STATS.items():
    results = {}
    for N in CUTOFFS:
        parts = []
        for (season, _), g in df.groupby(["season", "player"]):
            if len(g) >= N + MIN_REMAINING_GAMES:
                e, r = g.iloc[:N], g.iloc[N:]
                parts.append((season, e[num].sum(), e[den].sum(), r[num].sum(), r[den].sum()))
        t = pd.DataFrame(parts, columns=["season", "en", "ed", "rn", "rd"])
        t = t[(t.ed > 0) & (t.rd > 0)]
        for k in kappas:
            se, w = [], []
            for _, s in t.groupby("season"):
                m = s.en.sum() / s.ed.sum()                    # league mean from first N games only
                pred = (m * k + s.en) / (k + s.ed)
                se.append((s.rn / s.rd - pred) ** 2)
                w.append(s.rd)
            results[(k, N)] = np.average(pd.concat(se), weights=pd.concat(w))
    table = pd.Series(results).unstack()
    table.index.name = "prior strength"
    table.columns = [f"{c} games" for c in table.columns]
    print(f"\n{stat}: out-of-sample error (lower is better); best per column marked *")
    out = table.map(lambda v: f"{v:.5f}")
    for col in table.columns:
        out.loc[table[col].idxmin(), col] += " *"
    print(out.to_string())