from __future__ import annotations

import numpy as np

from src.frequentist import proportion_ci
from src.sequential import always_valid_ci, compare_variants, holm_bonferroni


def test_confidence_sequence_brackets_point_estimate():
    lo, hi = always_valid_ci(120, 1000)
    assert lo < 0.12 < hi


def test_confidence_sequence_is_wider_than_fixed_ci():
    # Anytime validity costs width: the CS must be wider than the fixed-n Wald CI.
    cs_lo, cs_hi = always_valid_ci(120, 1000)
    w_lo, w_hi = proportion_ci(120, 1000)
    assert (cs_hi - cs_lo) > (w_hi - w_lo)


def test_confidence_sequence_shrinks_with_more_data():
    small = always_valid_ci(120, 1000)
    large = always_valid_ci(1200, 10000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_holm_is_more_conservative_than_raw():
    pvals = [0.001, 0.02, 0.04]
    adj, rejected = holm_bonferroni(pvals, alpha=0.05)
    assert np.all(adj >= np.array(pvals) - 1e-12)
    assert adj.shape == (3,)


def test_holm_rejects_only_strong_effects():
    # One clearly significant, two borderline that shouldn't survive correction.
    adj, rejected = holm_bonferroni([0.0001, 0.03, 0.04], alpha=0.05)
    assert rejected[0]
    assert not rejected[2]


def test_compare_variants_flags_the_winner():
    comparisons = compare_variants(
        control=(100, 2000),
        variants={"B": (180, 2000), "C": (105, 2000)},
    )
    by_name = {c.name: c for c in comparisons}
    assert by_name["B"].significant_adjusted        # big lift survives correction
    assert not by_name["C"].significant_adjusted    # noise does not
    assert by_name["B"].p_adjusted >= by_name["B"].result.p_value
