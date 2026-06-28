"""Sequential & multi-variant testing — peek safely, and test many arms at once.

Two additions that take the toolkit beyond a single fixed-horizon A/B test:

* **Always-valid confidence sequences** (asymptotic CS, Waudby-Smith & Ramdas,
  2023): an interval you may inspect after *every* visitor and stop the moment it
  excludes your null — without inflating the false-positive rate. The price for
  this "anytime validity" is a wider interval than the fixed-``n`` Wald CI.
* **Multi-variant correction** (Holm-Bonferroni): compare A/B/C/... against a
  control while controlling the family-wise error rate, so running five variants
  doesn't quietly turn a 5% error rate into 23%.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .frequentist import TestResult, two_proportion_ztest


def always_valid_ci(
    successes: int, n: int, alpha: float = 0.05, rho: float | None = None
) -> tuple[float, float]:
    """Asymptotic confidence sequence for a single proportion.

    Unlike a fixed-sample CI, this interval is valid *uniformly over time*: you
    may recompute it after every observation and act on it the first time it
    excludes your null, with the Type-I error still bounded by ``alpha``.

    ``rho`` tunes where the sequence is tightest; by default it is set near the
    width-optimal value for the current sample size.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    var = max(p * (1.0 - p), 1e-12)
    if rho is None:
        # Width-optimal tuning around the current n (WSR 2023, eq. for rho*).
        rho = np.sqrt(max((-2 * np.log(alpha) + np.log(-2 * np.log(alpha) + 1)) / n, 1e-12))
    nr2 = n * rho**2
    radius = np.sqrt(var) * np.sqrt(
        (2 * (nr2 + 1)) / (n**2 * rho**2) * np.log(np.sqrt(nr2 + 1) / alpha)
    )
    return (max(0.0, p - radius), min(1.0, p + radius))


def holm_bonferroni(
    pvalues, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Holm's step-down correction.

    Returns ``(adjusted_pvalues, rejected)`` where ``rejected[i]`` is True if
    hypothesis ``i`` is rejected at family-wise level ``alpha``. Holm is
    uniformly more powerful than plain Bonferroni while controlling the same
    family-wise error rate.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adj[idx] = min(running_max, 1.0)
    return adj, adj < alpha


@dataclass
class VariantComparison:
    name: str
    result: TestResult
    p_adjusted: float
    significant_adjusted: bool

    def as_dict(self) -> dict:
        d = self.result.as_dict()
        d.update(
            name=self.name,
            p_adjusted=round(self.p_adjusted, 5),
            significant_adjusted=self.significant_adjusted,
        )
        return d


def compare_variants(
    control: tuple[int, int],
    variants: Mapping[str, tuple[int, int]],
    alpha: float = 0.05,
) -> list[VariantComparison]:
    """Compare each variant against the control with Holm FWER control.

    ``control`` and each value of ``variants`` are ``(conversions, n)`` tuples.
    Returns one :class:`VariantComparison` per variant, carrying both the raw
    z-test and the multiplicity-adjusted verdict.
    """
    conv_c, n_c = control
    names = list(variants)
    results = {
        nm: two_proportion_ztest(conv_c, n_c, variants[nm][0], variants[nm][1], alpha)
        for nm in names
    }
    adj, rejected = holm_bonferroni([results[nm].p_value for nm in names], alpha)
    return [
        VariantComparison(nm, results[nm], float(adj[i]), bool(rejected[i]))
        for i, nm in enumerate(names)
    ]
