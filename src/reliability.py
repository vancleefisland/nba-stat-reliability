"""Split-half reliability: how much of a stat's player-to-player spread is real skill?

For sample size N games: randomly draw two disjoint N-game samples from each player's
season, compute the stat in each, and correlate across players. That correlation is the
reliability r_N of an N-game sample. Under classical test theory r_N = N / (N + k), where
k is the sample size at which reliability is 50% (signal variance = noise variance).
"""
import numpy as np
import pandas as pd
from . import config as C


def split_half_curve(df: pd.DataFrame, num: str, den: str, n_grid, n_splits, rng,
                     min_players: int = C.MIN_PLAYERS) -> pd.DataFrame:
    rows = []
    for N in n_grid:
        cell_r, cell_w = [], []
        for _, sdf in df.groupby("season"):
            players = [(g[num].to_numpy(float), g[den].to_numpy(float))
                       for _, g in sdf.groupby("player") if len(g) >= 2 * N]
            if len(players) < min_players:
                continue
            rs = []
            for _ in range(n_splits):
                x1, x2 = [], []
                for a, b in players:
                    idx = rng.permutation(len(a))
                    i1, i2 = idx[:N], idx[N:2 * N]
                    d1, d2 = b[i1].sum(), b[i2].sum()
                    if d1 > 0 and d2 > 0:
                        x1.append(a[i1].sum() / d1)
                        x2.append(a[i2].sum() / d2)
                if len(x1) >= min_players:
                    rs.append(np.corrcoef(x1, x2)[0, 1])
            if rs:
                cell_r.append(np.mean(rs)); cell_w.append(len(players))
        if cell_r:
            rows.append({"games": N, "reliability": np.average(cell_r, weights=cell_w),
                         "n_players": int(np.sum(cell_w))})
    return pd.DataFrame(rows)


def fit_half_reliability_point(games, reliability) -> float:
    """Least-squares fit of r = N / (N + k); returns k (games to 50% reliability)."""
    games = np.asarray(games, float)
    r = np.asarray(reliability, float)
    ks = np.logspace(-1, 4, 2000)
    sse = [np.sum((games / (games + k) - r) ** 2) for k in ks]
    return float(ks[int(np.argmin(sse))])


def games_for_reliability(k: float, target: float) -> float:
    """Sample size needed to reach a target reliability: N = k * target / (1 - target)."""
    return k * target / (1 - target)
