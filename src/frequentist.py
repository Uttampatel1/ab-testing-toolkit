"""Frequentist A/B analysis: hypothesis tests, confidence intervals, power.

Covers the two workhorses of online experimentation:

* **Conversion (proportions):** two-proportion z-test, Wald CIs for each rate and
  for their difference.
* **Revenue (continuous):** Welch's t-test (unequal variances).

Plus the planning tools every honest experiment needs *before* it starts:
sample-size, achieved-power, and minimum-detectable-effect calculators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    statistic: float
    p_value: float
    estimate: float          # observed effect (difference)
    ci_low: float
    ci_high: float
    significant: bool

    def as_dict(self) -> dict:
        return {
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 5),
            "estimate": round(self.estimate, 5),
            "ci_low": round(self.ci_low, 5),
            "ci_high": round(self.ci_high, 5),
            "significant": self.significant,
        }


def proportion_ci(conversions: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wald confidence interval for a single conversion rate."""
    p = conversions / n
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / n)
    return (p - z * se, p + z * se)


def two_proportion_ztest(
    conv_a: int, n_a: int, conv_b: int, n_b: int, alpha: float = 0.05
) -> TestResult:
    """Two-sided two-proportion z-test (B − A), with a Wald CI on the difference."""
    p_a, p_b = conv_a / n_a, conv_b / n_b
    diff = p_b - p_a

    # Pooled proportion for the test statistic (H0: p_a == p_b).
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled SE for the CI on the difference.
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    zc = stats.norm.ppf(1 - alpha / 2)
    return TestResult(
        statistic=float(z),
        p_value=float(p_value),
        estimate=float(diff),
        ci_low=float(diff - zc * se_diff),
        ci_high=float(diff + zc * se_diff),
        significant=bool(p_value < alpha),
    )


def welch_ttest(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Welch's two-sample t-test on a continuous metric (B − A)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t, p = stats.ttest_ind(b, a, equal_var=False)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = _welch_df(a, b)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return TestResult(
        statistic=float(t),
        p_value=float(p),
        estimate=float(diff),
        ci_low=float(diff - tc * se),
        ci_high=float(diff + tc * se),
        significant=bool(p < alpha),
    )


def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    num = (va + vb) ** 2
    den = va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)
    return num / den if den > 0 else len(a) + len(b) - 2


def sample_size_proportion(
    p1: float, mde: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Required sample size **per arm** to detect an absolute lift ``mde``."""
    p2 = p1 + mde
    p_bar = (p1 + p2) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    num = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(np.ceil(num / mde**2))


def power_proportion(
    p1: float, p2: float, n_per_arm: int, alpha: float = 0.05
) -> float:
    """Achieved power for detecting p2 vs p1 at sample size ``n_per_arm``."""
    p_bar = (p1 + p2) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se_pool = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_arm)
    se_alt = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_arm)
    z_beta = (abs(p2 - p1) - z_alpha * se_pool) / se_alt
    return float(stats.norm.cdf(z_beta))


def minimum_detectable_effect(
    p1: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest absolute lift detectable at the given sample size (binary search)."""
    lo, hi = 1e-6, 1.0 - p1
    for _ in range(100):
        mid = (lo + hi) / 2
        if power_proportion(p1, p1 + mid, n_per_arm, alpha) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
