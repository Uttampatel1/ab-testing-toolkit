from __future__ import annotations

from src.bayesian import beta_binomial_analysis


def test_treatment_clearly_better_gives_high_probability():
    res = beta_binomial_analysis(100, 1000, 150, 1000, n_samples=50_000, seed=1)
    assert res.prob_b_beats_a > 0.99
    assert res.expected_uplift > 0
    # expected loss of choosing the (correct) better arm is tiny
    assert res.expected_loss_choosing_b < 1e-3


def test_no_difference_gives_probability_near_half():
    res = beta_binomial_analysis(120, 1000, 120, 1000, n_samples=50_000, seed=2)
    assert 0.4 < res.prob_b_beats_a < 0.6


def test_credible_intervals_bracket_rates():
    res = beta_binomial_analysis(120, 1000, 150, 1000, n_samples=20_000, seed=3)
    assert res.ci_a[0] < 0.12 < res.ci_a[1]
    assert res.ci_b[0] < 0.15 < res.ci_b[1]


def test_result_is_deterministic_with_seed():
    a = beta_binomial_analysis(100, 1000, 150, 1000, n_samples=20_000, seed=7)
    b = beta_binomial_analysis(100, 1000, 150, 1000, n_samples=20_000, seed=7)
    assert a.prob_b_beats_a == b.prob_b_beats_a
