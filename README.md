# NBA Stat Reliability Guide
**How many games until you can trust a number?**

For common NBA stats, how much of a player's observed number reflects real skill versus luck at a given sample size, and does accounting for that improve prediction?

**Status:** pipeline complete and run on six seasons of real data. Tested on Python 3.9 on macOS.

## Key findings
- **FT% stabilizes fast, 3P% does not.** About 25 attempts to reach 50% reliability for FT% vs. about 294 for 3P%. The average player takes about 235 threes in a full season, so a typical full-season 3P% is still less than half signal.
- **Early 3P% is worse than useless.** After 10 games, using a player's raw 3P% has about 4x the error of just guessing the league average. Shrinkage fixes this, but it only beats the league average by about 1% at that point, growing to about 9% by game 40.
- **FT% shrinkage helps a lot** (about 35% lower error than the league average after 10 games, about 54% after 40), because players genuinely differ.
- **The ordering is robust, the exact thresholds are not.** Two reliability methods disagree by up to about 2x on the numbers (see "Two methods, two answers"), but agree FT% is far faster than 3P%.

## Method
1. **Beta-binomial empirical Bayes.** Each player's true talent comes from a league-wide Beta distribution fit by maximum likelihood. The prior strength (a + b) is the number of "ghost" league-average attempts mixed into each record, which doubles as the attempts needed for 50% reliability.
2. **Split-half reliability (cross-check).** Random disjoint N-game samples from each player-season are correlated across players, and a Spearman-Brown curve r = N/(N+k) is fit to find where reliability reaches 50%.
3. **Out-of-sample test.** Skill is estimated from the first N games and scored against the rest of the season (prior fit only on the first N games, so no leakage), comparing raw, shrunk, and league-average estimates.
4. **Prior-strength sweep.** Several candidate prior strengths are scored on held-out games, as a tiebreaker between the two reliability estimates.
5. **Plain-English output.** `outputs/trust_guide.md` is a one-page guide, and `app.py` is an interactive version.

## Run it
```bash
pip install -r requirements.txt
python scripts/collect_game_logs.py          # downloads data/raw/game_logs.csv (needs internet)
python -m src.run_pipeline                   # reliability, shrinkage evaluation, trust guide
python scripts/prior_strength_sweep.py       # which prior strength predicts best
python -m pytest                             # unit tests
streamlit run app.py                         # interactive tool
# optional: python -m src.make_synthetic     # FAKE data for testing only; delete before real runs
```
The notebook `notebooks/results.ipynb` draws the plots from the files in `outputs/`.

## Project structure
```
src/        core code (data checks, shrinkage, reliability, evaluation, trust guide)
scripts/    collect_game_logs.py, prior_strength_sweep.py
notebooks/  results.ipynb
tests/      unit tests
outputs/    generated results (trust guide, CSVs)
app.py      Streamlit tool
```

## Data
- **Source:** player and team game logs from `nba_api` (regular-season logs, 2021-22 through 2024-25).
- **Size:** 104,262 player-games, 925 players, about 26,000 rows per season.
- **Not included in the repo.** Run the collection script to download it. Check the data provider's terms of use before sharing the data itself.
- **Sanity checks:** league 3P% of 35.4% to 36.6% and FT% of 77.5% to 78.4% by season; no missing values; each season runs October to April.
- **Schema** (`data/raw/game_logs.csv`, one row per player-game): `player, player_name, season, game_date, fg3m, fg3a, ftm, fta, ast, poss`. Seasons are labeled by starting year (2023 = 2023-24).
- **Possessions are estimated,** since they aren't in standard game logs. Team possessions are approximated as FGA + 0.44*FTA - OREB + TOV, then scaled by the player's share of team minutes.

## Results

### How many attempts until a stat is trustworthy?

| Stat | Attempts to 50% reliability | Attempts to 70% | Typical full-season attempts* |
|---|---|---|---|
| FT% | ~25 | ~58 | ~150 |
| AST rate (per possession) | ~75 | ~176 | ~3,400 possessions |
| 3P% | ~294 | ~685 | ~235 |

*League-wide average per player-game x 70 games. High-volume shooters take far more.

Assist rate looks very reliable, but a lot of that is likely role (guards vs. big men) rather than skill, and it relies on estimated possessions, so I treat it as exploratory.

### Does shrinkage beat the alternatives?

Skill was estimated from each player's first N games and scored on the rest of the season. Positive means lower error than guessing the league average.

| After N games | 3P%: shrunk vs league avg | FT%: shrunk vs league avg |
|---|---|---|
| 10 | +0.9% | +34.5% |
| 20 | +4.6% | +44.8% |
| 30 | +8.1% | +50.1% |
| 40 | +8.8% | +53.7% |

### Two methods, two answers
Split-half reliability and the beta-binomial model disagree on how many attempts a stat needs to reach 50% reliability: for 3P%, about 175 vs. about 294 (after correcting split-half for the higher shot volume of full-season players), and for FT%, about 61 vs. about 25. I tested two explanations, shot volume of qualifying players and fitting the beta-binomial on the same players (70+ games only, which gave 366 and 29), and neither explained the gap, so the cause remains unresolved.

To decide which to use, I swept prior strengths and scored each on held-out games (`scripts/prior_strength_sweep.py`). The beta-binomial values predicted best for both stats (294 for 3P%, about 29 for FT%). For 3P% the error is nearly flat between about 175 and 500, so the exact threshold is uncertain; for FT% it is more sensitive. I therefore use the beta-binomial values as the headline numbers and report split-half as a cross-check. I did not compute confidence intervals on these differences.

## Limitations
- **One shared talent distribution for all players.** I suspect high-volume shooters are better shooters, which a single prior would miss, but I have not tested this. It may be part of why shrinkage barely beats the league average for 3P%.
- **Assist rate reflects role, not just skill,** and uses estimated possessions. The role explanation is my inference; the data has no positions to test it.
- **Possessions are estimated** from team box-score totals scaled by minutes, not measured from play-by-play.
- **Split-half uses only players with at least 2N games,** so it over-represents rotation players.
- **Talent is assumed constant within a season.** Role, health, and skill change.
- **The out-of-sample target is noisy,** since rest-of-season rates are themselves small samples, so absolute errors are inflated. Compare methods to each other, not to zero.
- **No confidence intervals** on the reliability estimates or on the sweep results.
- **Possible in-season tournament games.** One player (Buddy Hield) shows 84 games in a season, which exceeds the 82-game schedule. 
- **Six seasons only,** during a period when 3-point volume was changing.

### What I'd do differently

- **Compare methods in the same units from the start.** My split half study measures sample size in games and the beta-binomial model measures it in attempts. Converting between them with league average shot volume hid part of the disagreement, and it took several extra tests to find out the conversion wasn't the main cause. Next time I would build both estimates in attempts from the beginning.
- **Fit priors by shot volume or position.** Assist rate and 3-point shooting clearly depend on role, and a single league wide prior ignores that. This is the first extension I would try, and I'd test whether it makes shrinkage beat the league average by more for 3P%.
- **Put uncertainty on the headline numbers.** The prior-strength sweep suggests the 3P% threshold is fairly flat between about 175 and 500 attempts, but I only compared point estimates. Bootstrapping by player would show which differences are real.
- **Use real possessions.** My possession counts are estimates from box-score totals. Play-by-play data would remove a source of error from the assist-rate results.
- **Clean the data at collection time.** Buddy Hield shows 84 games in a season, which suggests non regular season games slipped in. I'd filter these in the download script instead of discovering it afterward.
- **Test on a season the model never saw.** I tuned and compared methods on the same four seasons. A later season held out until the end would be a cleaner final check.
- **Check results against published stabilization research earlier.** [Add what you found when you compared, and whether it agreed.]


