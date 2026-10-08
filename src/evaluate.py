"""Does shrinkage actually predict better than raw numbers? Out-of-sample test.

For each season and cutoff N: estimate each player's skill from their first N games,
then score against their performance over the REST of the season (never seen when
estimating). Prior is fit on the first-N-games data only, so there is no leakage.
Errors are weighted by remaining attempts. Remaining-season rates are themselves noisy,
so absolute MSE is inflated; compare methods against each other, not to zero.
"""
import numpy as np
import pandas as pd
from .shrinkage import BetaPrior


def evaluate_shrinkage(df: pd.DataFrame, stat: str, num: str, den: str,
                       cutoffs, min_remaining_games: int) -> pd.DataFrame:
    out = []
    for N in cutoffs:
        early, rest = [], []
        for (season, _), g in df.groupby(["season", "player"]):
            g = g.sort_values("game_date")
            if len(g) >= N + min_remaining_games:
                e, r = g.iloc[:N], g.iloc[N:]
                early.append((season, e[num].sum(), e[den].sum()))
                rest.append((r[num].sum(), r[den].sum()))
        if len(early) < 30:
            continue
        E = pd.DataFrame(early, columns=["season", "en", "ed"])
        R = pd.DataFrame(rest, columns=["rn", "rd"])
        t = pd.concat([E, R], axis=1)
        sq = {"raw": [], "shrunk": [], "league": []}
        w = []
        for _, s in t.groupby("season"):
            s = s[(s.ed > 0) & (s.rd > 0)]
            if len(s) < 30:
                continue
            prior = BetaPrior.fit(s.en, s.ed)
            actual = (s.rn / s.rd).to_numpy()
            preds = {"raw": (s.en / s.ed).to_numpy(),
                     "shrunk": prior.posterior_mean(s.en.to_numpy(), s.ed.to_numpy()),
                     "league": np.full(len(s), s.en.sum() / s.ed.sum())}
            for k, p in preds.items():
                sq[k].append((actual - p) ** 2)
            w.append(s.rd.to_numpy())
        if not w:
            continue
        w = np.concatenate(w)
        mse = {k: float(np.average(np.concatenate(v), weights=w)) for k, v in sq.items()}
        out.append({"stat": stat, "cutoff_games": N, "n_players": len(w),
                    "mse_raw": mse["raw"], "mse_shrunk": mse["shrunk"],
                    "mse_league_avg": mse["league"],
                    "pct_better_than_raw": 100 * (1 - mse["shrunk"] / mse["raw"])})
    return pd.DataFrame(out)
