"""Generates FAKE game logs with KNOWN true talent, purely to test the pipeline.
Never report results from this data."""
import numpy as np, pandas as pd
rng = np.random.default_rng(0)

def beta_mean_sd(m, sd, size):
    kappa = m * (1 - m) / sd**2 - 1
    return rng.beta(m * kappa, (1 - m) * kappa, size)

P = 350
t3, tft, tast = beta_mean_sd(.35, .03, P), beta_mean_sd(.77, .07, P), beta_mean_sd(.10, .05, P)
a3 = rng.gamma(4, 1.0, P) + .5; aft = rng.gamma(3, 1.0, P) + .5; ps = rng.normal(45, 8, P).clip(25, 70)

rows = []
for season in (2022, 2023, 2024):
    dates = pd.date_range(f"{season}-10-20", periods=82, freq="2D")
    for i in range(P):
        g = int(rng.integers(30, 82))
        days = np.sort(rng.choice(82, g, replace=False))
        fg3a = rng.poisson(a3[i], g); fta = rng.poisson(aft[i], g)
        poss = rng.poisson(ps[i], g)
        rows.append(pd.DataFrame({
            "player": f"P{i}", "season": season, "game_date": dates[days],
            "fg3a": fg3a, "fg3m": rng.binomial(fg3a, t3[i]),
            "fta": fta, "ftm": rng.binomial(fta, tft[i]),
            "poss": poss, "ast": rng.binomial(poss, tast[i])}))
pd.concat(rows).to_csv("data/raw/game_logs.csv", index=False)
print("wrote SYNTHETIC data/raw/game_logs.csv")
print("TRUE prior strengths (a+b): 3P%%=%.0f  FT%%=%.0f  AST=%.0f" % (
    .35*.65/.03**2-1, .77*.23/.07**2-1, .10*.90/.05**2-1))
