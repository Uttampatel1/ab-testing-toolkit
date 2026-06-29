"""CUPED — variance reduction with a pre-experiment covariate.

CUPED (Controlled-experiment Using Pre-Experiment Data, Microsoft 2013) is the
standard trick top experimentation teams use to get **more power from the same
traffic**. If you have a pre-period covariate ``X`` (e.g. each visitor's spend the
month *before* the test) that correlates with the metric ``Y``, you can subtract
off the part of ``Y`` that ``X`` already explains::

    Y_cuped = Y - theta * (X - mean(X)),   theta = cov(X, Y) / var(X)

Because ``X`` is measured *before* assignment it's balanced across arms, so the
adjustment leaves the treatment effect's expectation unchanged while shrinking its
variance by roughly ``corr(X, Y)**2``. Less variance → tighter CIs → the test ends
sooner, with no extra users.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frequentist import TestResult, welch_ttest


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """Optimal CUPED coefficient ``cov(x, y) / var(x)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    var_x = x.var(ddof=1)
    if var_x < 1e-12:
        return 0.0
    return float(np.cov(x, y, ddof=1)[0, 1] / var_x)


def cuped_adjust(
    y: np.ndarray, x: np.ndarray, theta: float, x_mean: float
) -> np.ndarray:
    """Return the CUPED-adjusted metric ``y - theta * (x - x_mean)``."""
    return np.asarray(y, dtype=float) - theta * (np.asarray(x, dtype=float) - x_mean)


@dataclass
class CupedResult:
    theta: float
    variance_reduction: float   # fraction of variance removed (0..1)
    naive: TestResult           # Welch t-test on the raw metric
    adjusted: TestResult        # Welch t-test on the CUPED-adjusted metric

    def as_dict(self) -> dict:
        return {
            "theta": round(self.theta, 4),
            "variance_reduction_pct": round(self.variance_reduction * 100, 2),
            "naive": self.naive.as_dict(),
            "cuped": self.adjusted.as_dict(),
        }


def cuped_welch_ttest(
    a_y: np.ndarray,
    a_x: np.ndarray,
    b_y: np.ndarray,
    b_x: np.ndarray,
    alpha: float = 0.05,
) -> CupedResult:
    """Welch's t-test (B − A) before and after CUPED adjustment.

    ``theta`` and the covariate mean are estimated on the **pooled** data so the
    same transform is applied to both arms (keeping the estimate unbiased).
    """
    a_y = np.asarray(a_y, dtype=float)
    b_y = np.asarray(b_y, dtype=float)
    a_x = np.asarray(a_x, dtype=float)
    b_x = np.asarray(b_x, dtype=float)

    pooled_y = np.concatenate([a_y, b_y])
    pooled_x = np.concatenate([a_x, b_x])
    theta = cuped_theta(pooled_y, pooled_x)
    x_mean = float(pooled_x.mean())

    a_adj = cuped_adjust(a_y, a_x, theta, x_mean)
    b_adj = cuped_adjust(b_y, b_x, theta, x_mean)

    naive = welch_ttest(a_y, b_y, alpha)
    adjusted = welch_ttest(a_adj, b_adj, alpha)

    var_before = pooled_y.var(ddof=1)
    var_after = np.concatenate([a_adj, b_adj]).var(ddof=1)
    reduction = 0.0 if var_before < 1e-12 else 1.0 - var_after / var_before

    return CupedResult(
        theta=theta,
        variance_reduction=float(reduction),
        naive=naive,
        adjusted=adjusted,
    )
