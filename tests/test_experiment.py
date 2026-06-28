from __future__ import annotations

from src.config import Settings
from src.experiment import analyze
from src.generate_data import generate


def _settings() -> Settings:
    return Settings(
        n_per_arm=8000, baseline_rate=0.12, treatment_rate=0.145,
        n_mc_samples=20_000, seed=42,
    )


def test_report_recovers_positive_lift():
    settings = _settings()
    report = analyze(generate(settings), settings=settings)
    assert report.rate_treatment > report.rate_control
    assert report.conversion_test.estimate > 0
    # a ~2.5pp lift at 8k/arm should be detectable
    assert report.conversion_test.significant
    assert report.bayesian.prob_b_beats_a > 0.9


def test_report_dict_structure():
    settings = _settings()
    report = analyze(generate(settings), settings=settings)
    d = report.as_dict()
    assert set(d) >= {
        "samples", "conversion_rate", "frequentist_conversion",
        "frequentist_revenue_per_visitor", "bayesian_conversion",
    }
    assert d["conversion_rate"]["relative_lift_pct"] > 0
