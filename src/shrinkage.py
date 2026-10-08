"""Empirical Bayes (beta-binomial) shrinkage.

Idea: every player has a true talent p drawn from a league-wide Beta(a, b).
Observing x successes in n trials updates it to Beta(a + x, b + n - x).
The prior strength (a + b) is the number of 'ghost attempts' of league-average
performance mixed into every player's record: an intuitive way to explain it.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta as beta_dist, betabinom


@dataclass
class BetaPrior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        return self.alpha + self.beta

    @classmethod
    def fit(cls, successes, trials) -> "BetaPrior":
        """Maximum-likelihood beta-binomial fit across players (handles unequal n)."""
        x = np.asarray(successes, dtype=int)
        n = np.asarray(trials, dtype=int)
        keep = n > 0
        x, n = x[keep], n[keep]
        if len(x) < 10:
            raise ValueError("Need at least 10 players with attempts to fit a prior.")

        def nll(theta):
            m, kappa = expit(theta[0]), np.exp(theta[1])
            return -betabinom.logpmf(x, n, m * kappa, (1 - m) * kappa).sum()

        m0 = x.sum() / n.sum()
        res = minimize(nll, x0=[np.log(m0 / (1 - m0)), np.log(100.0)], method="Nelder-Mead")
        m, kappa = expit(res.x[0]), np.exp(res.x[1])
        return cls(alpha=m * kappa, beta=(1 - m) * kappa)

    def posterior_mean(self, successes, trials):
        return (self.alpha + np.asarray(successes)) / (self.strength + np.asarray(trials))

    def posterior_interval(self, successes, trials, level: float = 0.8):
        a = self.alpha + np.asarray(successes, dtype=float)
        b = self.beta + np.asarray(trials, dtype=float) - np.asarray(successes, dtype=float)
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        return beta_dist.ppf(lo, a, b), beta_dist.ppf(hi, a, b)
