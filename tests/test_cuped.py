import numpy as np

from src.config import Settings
from src.cuped import cuped_adjust, cuped_theta, cuped_welch_ttest
from src.generate_data import generate


def test_theta_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = 3.0 * x + rng.normal(scale=0.5, size=5000)  # slope 3
    assert abs(cuped_theta(y, x) - 3.0) < 0.1


def test_adjust_removes_covariate_variance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = 2.0 * x + rng.normal(scale=0.3, size=5000)
    theta = cuped_theta(y, x)
    y_cuped = cuped_adjust(y, x, theta, x.mean())
    assert y_cuped.var() < y.var()


def test_zero_variance_covariate_is_safe():
    y = np.array([1.0, 2.0, 3.0])
    x = np.array([5.0, 5.0, 5.0])  # no variance
    assert cuped_theta(y, x) == 0.0


def test_cuped_reduces_variance_on_experiment_data():
    df = generate(Settings(n_per_arm=6000, seed=3))
    a = df[df["group"] == "control"]
    b = df[df["group"] == "treatment"]
    res = cuped_welch_ttest(
        a["revenue"].to_numpy(), a["pre_revenue"].to_numpy(),
        b["revenue"].to_numpy(), b["pre_revenue"].to_numpy(),
    )
    # the covariate is predictive, so variance should drop and the CI should tighten
    assert res.variance_reduction > 0.05
    naive_width = res.naive.ci_high - res.naive.ci_low
    cuped_width = res.adjusted.ci_high - res.adjusted.ci_low
    assert cuped_width < naive_width


def test_cuped_keeps_effect_estimate_roughly_unbiased():
    df = generate(Settings(n_per_arm=8000, seed=7))
    a = df[df["group"] == "control"]
    b = df[df["group"] == "treatment"]
    res = cuped_welch_ttest(
        a["revenue"].to_numpy(), a["pre_revenue"].to_numpy(),
        b["revenue"].to_numpy(), b["pre_revenue"].to_numpy(),
    )
    # adjustment shifts the point estimate only a little (X is balanced across arms)
    assert abs(res.adjusted.estimate - res.naive.estimate) < 0.5
