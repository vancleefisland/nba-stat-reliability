"""Turn results into a plain-English one-pager for non-technical readers."""
import pandas as pd
from . import config as C
from .reliability import games_for_reliability


def verdict(k_attempts: float, season_attempts: float) -> str:
    share = k_attempts / season_attempts
    if share < 0.25:
        return "Stabilizes fast. Early-season numbers are worth taking seriously."
    if share < 1.0:
        return "Moderate. Needs a big chunk of a season; be careful with month-long hot or cold streaks."
    return "Slow. Even a full season is mostly noise plus signal in equal parts; lean on the prior."


def build_guide(summary: pd.DataFrame, shrink_eval: pd.DataFrame) -> str:
    lines = ["# Stat Trust Guide: how long until a number means something?", "",
             "**Reliability** is the share of the differences between players that reflects real skill rather than luck. "
             "50% reliability means signal and noise are equal; 70% means the number is mostly signal.", "",
             "| Stat | Attempts to 50% | Attempts to 70% | ~Games to 70% | How to read it |",
             "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        a50 = r.attempts_to_50_betabinom
        a70 = games_for_reliability(a50, 0.7)
        season = r.attempts_per_game * C.SEASON_GAMES
        lines.append(f"| {r.stat} | {a50:.0f} | {a70:.0f} | {a70 / r.attempts_per_game:.0f} | {verdict(a50, season)} |")
    lines += ["", "_Headline numbers come from a beta-binomial model of league-wide talent; "
              "a split-half reliability study is reported in the repo as a cross-check._"]
    lines += ["", "## Does shrinkage help?",
              "Estimates from the first N games, pulled toward the league average in proportion to how little we know, "
              "were compared with raw numbers on the rest of the season (lower error is better).", ""]
    if shrink_eval is not None and len(shrink_eval):
        lines += ["| Stat | After N games | Improvement over raw |", "|---|---|---|"]
        for _, r in shrink_eval.iterrows():
            lines.append(f"| {r.stat} | {int(r.cutoff_games)} | {r.pct_better_than_raw:.1f}% |")
    lines += ["", "## Rule of thumb",
              "Before reacting to a stat, ask: how many attempts is this, and how many does this stat need? "
              "If the sample is well below the 50% mark, treat the raw number as mostly noise.", "",
              "_Limitations: players with more games are over-represented; skill changes within a season are ignored; "
              "see README for details._"]
    return "\n".join(lines)
