# Stat Trust Guide: how long until a number means something?

**Reliability** is the share of the differences between players that reflects real skill rather than luck. 50% reliability means signal and noise are equal; 70% means the number is mostly signal.

| Stat | Attempts to 50% | Attempts to 70% | ~Games to 70% | How to read it |
|---|---|---|---|---|
| 3P% | 294 | 685 | 204 | Slow. Even a full season is mostly noise plus signal in equal parts; lean on the prior. |
| FT% | 25 | 58 | 28 | Stabilizes fast. Early-season numbers are worth taking seriously. |
| AST rate | 75 | 176 | 4 | Stabilizes fast. Early-season numbers are worth taking seriously. |

_Headline numbers come from a beta-binomial model of league-wide talent; a split-half reliability study is reported in the repo as a cross-check._

## Does shrinkage help?
Estimates from the first N games, pulled toward the league average in proportion to how little we know, were compared with raw numbers on the rest of the season (lower error is better).

| Stat | After N games | Improvement over raw |
|---|---|---|
| 3P% | 10 | 76.3% |
| 3P% | 20 | 55.7% |
| 3P% | 30 | 39.8% |
| 3P% | 40 | 29.7% |
| FT% | 10 | 60.4% |
| FT% | 20 | 41.0% |
| FT% | 30 | 24.3% |
| FT% | 40 | 16.7% |
| AST rate | 10 | 9.1% |
| AST rate | 20 | 2.6% |
| AST rate | 30 | 2.4% |
| AST rate | 40 | 1.7% |

## Rule of thumb
Before reacting to a stat, ask: how many attempts is this, and how many does this stat need? If the sample is well below the 50% mark, treat the raw number as mostly noise.

_Limitations: players with more games are over-represented; skill changes within a season are ignored; see README for details._