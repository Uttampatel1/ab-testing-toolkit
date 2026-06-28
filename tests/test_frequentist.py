from __future__ import annotations

import numpy as np
import pytest

from src.frequentist import (
    minimum_detectable_effect,
    power_proportion,
    proportion_ci,
    sample_size_proportion,
    two_proportion_ztest,
    welch_ttest,
)


def test_ztest_detects_clear_difference():
    # 100/1000 vs 150/1000 is a large, obvious lift
    res = two_proportion_ztest(100, 1000, 150, 1000)
    assert res.estimate == pytest.approx(0.05)
    assert res.p_value < 0.01
    assert res.significant
    assert res.ci_low > 0  # CI excludes zero


def test_ztest_no_difference_is_not_significant():
    res = two_proportion_ztest(120, 1000, 122, 1000)
    assert not res.significant
    assert res.ci_low < 0 < res.ci_high


def test_proportion_ci_brackets_point_estimate():
    lo, hi = proportion_ci(120, 1000)
    assert lo < 0.12 < hi


def test_welch_ttest_on_shifted_normals():
    rng = np.random.default_rng(0)
    a = rng.normal(50, 10, 2000)
    b = rng.normal(53, 10, 2000)
    res = welch_ttest(a, b)
    assert res.estimate > 0
    assert res.significant


def test_sample_size_increases_for_smaller_effect():
    big = sample_size_proportion(0.10, 0.05)
    small = sample_size_proportion(0.10, 0.01)
    assert small > big > 0


def test_power_and_sample_size_are_consistent():
    # At the sample size computed for 80% power, achieved power should be ~0.80.
    n = sample_size_proportion(0.10, 0.02, alpha=0.05, power=0.80)
    achieved = power_proportion(0.10, 0.12, n, alpha=0.05)
    assert abs(achieved - 0.80) < 0.03


def test_mde_recovers_effect_used_for_sizing():
    n = sample_size_proportion(0.10, 0.02, power=0.80)
    mde = minimum_detectable_effect(0.10, n, power=0.80)
    assert abs(mde - 0.02) < 0.003
