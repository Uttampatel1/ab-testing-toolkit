"""Sample-Ratio-Mismatch (SRM) — the experiment's smoke alarm.

You designed a 50/50 split but observed 50,100 vs 49,000. Is that just noise, or is
the randomiser/logging broken? An SRM means the arms aren't comparable, so *every*
downstream p-value is suspect — you must fix the pipeline, not read the result.

This is a chi-square goodness-of-fit test of the observed arm counts against the
intended allocation. Because it runs on every experiment, the convention is a very
strict threshold (``alpha = 0.001``) to avoid crying wolf — a real SRM produces a
p-value many orders of magnitude below that anyway.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SRMResult:
    chi_square: float
    p_value: float
    passed: bool                # True = no mismatch detected (safe to analyse)
    observed: list[int]
    expected: list[float]

    def as_dict(self) -> dict:
        return {
            "chi_square": round(self.chi_square, 4),
            "p_value": self.p_value,
            "srm_detected": not self.passed,
            "observed": self.observed,
            "expected": [round(e, 1) for e in self.expected],
        }


def srm_test(
    observed,
    expected_ratio=None,
    alpha: float = 0.001,
) -> SRMResult:
    """Chi-square test that arm counts match the intended split.

    ``observed`` is the per-arm visitor counts; ``expected_ratio`` is the intended
    allocation (defaults to an equal split). An SRM is *detected* when
    ``p_value < alpha`` — in which case ``passed`` is False and the experiment
    should not be trusted until the cause is found.
    """
    observed = [int(c) for c in observed]
    k = len(observed)
    if k < 2:
        raise ValueError("need at least two arms")
    if expected_ratio is None:
        expected_ratio = [1.0] * k
    if len(expected_ratio) != k:
        raise ValueError("expected_ratio length must match number of arms")

    total = sum(observed)
    ratio = np.asarray(expected_ratio, dtype=float)
    expected = (ratio / ratio.sum() * total).tolist()

    chi2, p = stats.chisquare(observed, f_exp=expected)
    return SRMResult(
        chi_square=float(chi2),
        p_value=float(p),
        passed=bool(p >= alpha),
        observed=observed,
        expected=expected,
    )
