import pytest

from src.srm import srm_test


def test_balanced_split_passes():
    res = srm_test([50000, 49800])
    assert res.passed
    assert not res.as_dict()["srm_detected"]
    assert res.p_value > 0.001


def test_clear_mismatch_is_detected():
    # 52k vs 48k on a 50/50 design is a blatant SRM
    res = srm_test([52000, 48000])
    assert not res.passed
    assert res.p_value < 0.001


def test_respects_expected_ratio():
    # a 2:1 design with matching counts should pass
    res = srm_test([20000, 10000], expected_ratio=[2, 1])
    assert res.passed
    # the same counts judged against 1:1 should fail
    res2 = srm_test([20000, 10000], expected_ratio=[1, 1])
    assert not res2.passed


def test_expected_counts_sum_to_total():
    res = srm_test([30, 70])
    assert sum(res.expected) == pytest.approx(100)


def test_requires_two_arms():
    with pytest.raises(ValueError):
        srm_test([100])


def test_ratio_length_must_match():
    with pytest.raises(ValueError):
        srm_test([100, 100], expected_ratio=[1, 1, 1])
