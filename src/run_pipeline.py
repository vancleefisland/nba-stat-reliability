"""End-to-end: load -> reliability curves -> shrinkage evaluation -> priors -> trust guide.
Usage: python -m src.run_pipeline [path/to/game_logs.csv]"""
import json, sys
import numpy as np
import pandas as pd
from . import config as C
from .data import load_game_logs
from .reliability import split_half_curve, fit_half_reliability_point
from .evaluate import evaluate_shrinkage
from .shrinkage import BetaPrior
from .trust_guide import build_guide


def main(path=C.RAW_PATH):
    df = load_game_logs(path)
    rng = np.random.default_rng(C.SEED)
    curves, summary, evals, priors = [], [], [], {}

    for stat, spec in C.STATS.items():
        num, den = spec["num"], spec["den"]
        curve = split_half_curve(df, num, den, C.N_GRID, C.N_SPLITS, rng)
        curve.insert(0, "stat", stat)
        curves.append(curve)
        k = fit_half_reliability_point(curve.games, curve.reliability)
        per_game = df[den].mean()
        totals = df.groupby(["season", "player"])[[num, den]].sum()
        prior = BetaPrior.fit(totals[num], totals[den])
        priors[stat] = {"alpha": prior.alpha, "beta": prior.beta,
                        "mean": prior.mean, "strength": prior.strength}
        # Two independent estimates of 'attempts until 50% reliability':
        #  (1) split-half curve + Spearman-Brown (games -> attempts via mean attempts/game)
        #  (2) beta-binomial prior strength (a + b), already in attempts
        summary.append({"stat": stat, "k_games": k, "attempts_per_game": per_game,
                        "attempts_to_50_splithalf": k * per_game,
                        "attempts_to_50_betabinom": prior.strength})
        evals.append(evaluate_shrinkage(df, stat, num, den, C.EVAL_CUTOFFS, C.MIN_REMAINING_GAMES))

    curves = pd.concat(curves); summary = pd.DataFrame(summary); evals = pd.concat(evals)
    curves.to_csv("outputs/reliability_curves.csv", index=False)
    summary.round(3).to_csv("outputs/reliability_summary.csv", index=False)
    evals.round(5).to_csv("outputs/shrinkage_eval.csv", index=False)
    json.dump(priors, open("outputs/priors.json", "w"), indent=2)
    open("outputs/trust_guide.md", "w").write(build_guide(summary, evals))

    print(summary.round(1).to_string(index=False)); print()
    print(evals.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
