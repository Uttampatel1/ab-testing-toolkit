"""Bayesian A/B analysis for conversion rates (Beta-Binomial conjugate model).

With a ``Beta(a, b)`` prior and ``k`` conversions in ``n`` trials, the posterior
is ``Beta(a + k, b + n - k)``. From posterior samples of each arm we compute the
quantities a decision-maker actually wants:

* **P(treatment > control)** — probability the variant is genuinely better.
* **Expected uplift** — average relative improvement.
* **Expected loss** — risk of shipping the wrong arm (the decision metric in the
  "expected-loss < threshold" stopping rule).
* **Credible intervals** — the Bayesian analogue of a confidence interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BayesianResult:
    prob_b_beats_a: float
    expected_uplift: float        # relative, E[(pb - pa)/pa]
    expected_loss_choosing_b: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]

    def as_dict(self) -> dict:
        return {
            "prob_treatment_beats_control": round(self.prob_b_beats_a, 4),
            "expected_uplift_pct": round(self.expected_uplift * 100, 3),
            "expected_loss_choosing_treatment": round(self.expected_loss_choosing_b, 6),
            "credible_interval_control": [round(x, 5) for x in self.ci_a],
            "credible_interval_treatment": [round(x, 5) for x in self.ci_b],
        }


def beta_binomial_analysis(
    conv_a: int,
    n_a: int,
    conv_b: int,
    n_b: int,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    n_samples: int = 100_000,
    cred_mass: float = 0.95,
    seed: int = 42,
) -> BayesianResult:
    """Monte-Carlo Bayesian comparison of two conversion rates."""
    rng = np.random.default_rng(seed)
    post_a = rng.beta(prior_a + conv_a, prior_b + n_a - conv_a, n_samples)
    post_b = rng.beta(prior_a + conv_b, prior_b + n_b - conv_b, n_samples)

    prob_b = float(np.mean(post_b > post_a))
    uplift = float(np.mean((post_b - post_a) / post_a))
    # Expected loss if we choose B but A is actually better.
    loss_b = float(np.mean(np.maximum(post_a - post_b, 0.0)))

    lo = (1 - cred_mass) / 2
    hi = 1 - lo
    ci_a = (
        float(stats.beta.ppf(lo, prior_a + conv_a, prior_b + n_a - conv_a)),
        float(stats.beta.ppf(hi, prior_a + conv_a, prior_b + n_a - conv_a)),
    )
    ci_b = (
        float(stats.beta.ppf(lo, prior_a + conv_b, prior_b + n_b - conv_b)),
        float(stats.beta.ppf(hi, prior_a + conv_b, prior_b + n_b - conv_b)),
    )
    return BayesianResult(
        prob_b_beats_a=prob_b,
        expected_uplift=uplift,
        expected_loss_choosing_b=loss_b,
        ci_a=ci_a,
        ci_b=ci_b,
    )
