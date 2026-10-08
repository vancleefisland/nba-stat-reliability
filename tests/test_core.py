import numpy as np
from src.shrinkage import BetaPrior
from src.reliability import fit_half_reliability_point, games_for_reliability


def test_prior_recovers_true_parameters():
    rng = np.random.default_rng(1)
    p = rng.beta(35, 65, 2000)
    n = rng.integers(50, 400, 2000)
    prior = BetaPrior.fit(rng.binomial(n, p), n)
    assert abs(prior.mean - 0.35) < 0.01
    assert 70 < prior.strength < 140          # truth is 100


def test_posterior_lies_between_raw_and_prior_mean():
    prior = BetaPrior(35, 65)
    est = float(prior.posterior_mean(5, 8))   # raw 62.5%, prior mean 35%
    assert 0.35 < est < 0.625


def test_more_data_means_less_shrinkage():
    prior = BetaPrior(35, 65)
    small = float(prior.posterior_mean(6, 10)); large = float(prior.posterior_mean(600, 1000))
    assert abs(large - 0.6) < abs(small - 0.6)


def test_interval_narrows_with_sample_size():
    prior = BetaPrior(35, 65)
    lo1, hi1 = prior.posterior_interval(10, 30); lo2, hi2 = prior.posterior_interval(100, 300)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_spearman_brown_fit_recovers_k():
    games = np.array([5, 10, 20, 40, 80])
    k_true = 25.0
    assert abs(fit_half_reliability_point(games, games / (games + k_true)) - k_true) < 1.0


def test_games_for_reliability():
    assert games_for_reliability(30, 0.5) == 30
    assert abs(games_for_reliability(30, 0.7) - 70) < 1e-9
