from __future__ import annotations

from src.config import Settings
from src.generate_data import generate


def _settings() -> Settings:
    return Settings(n_per_arm=4000, baseline_rate=0.10, treatment_rate=0.14, seed=9)


def test_shape_and_groups():
    df = generate(_settings())
    assert len(df) == 8000
    assert set(df["group"].unique()) == {"control", "treatment"}
    assert df.groupby("group").size().tolist() == [4000, 4000]


def test_observed_rates_close_to_truth():
    df = generate(_settings())
    rates = df.groupby("group")["converted"].mean()
    assert abs(rates["control"] - 0.10) < 0.02
    assert abs(rates["treatment"] - 0.14) < 0.02


def test_revenue_only_for_converters():
    df = generate(_settings())
    non_converters = df[df["converted"] == 0]
    assert (non_converters["revenue"] == 0).all()
    assert (df[df["converted"] == 1]["revenue"] > 0).all()
