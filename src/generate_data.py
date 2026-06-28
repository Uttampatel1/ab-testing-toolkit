"""Synthetic A/B experiment generator.

Simulates a two-arm online experiment: each visitor is assigned to *control* or
*treatment* and may convert (Bernoulli) and, if so, generate revenue. The true
effect is known (set in config), so we can check that the analysis recovers it.
Everything is seed-driven — no real users.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import Settings, get_settings


def generate(settings: Settings | None = None) -> pd.DataFrame:
    """Return a tidy frame: one row per visitor (group, converted, revenue)."""
    settings = settings or get_settings()
    rng = np.random.default_rng(settings.seed)

    def arm(group: str, rate: float, aov: float) -> pd.DataFrame:
        n = settings.n_per_arm
        converted = rng.random(n) < rate
        # Revenue is positive only for converters; gamma keeps it non-negative.
        revenue = np.where(
            converted, rng.gamma(shape=4.0, scale=aov / 4.0, size=n), 0.0
        )
        return pd.DataFrame(
            {"group": group, "converted": converted.astype(int),
             "revenue": np.round(revenue, 2)}
        )

    control = arm("control", settings.baseline_rate, settings.aov_control)
    treatment = arm("treatment", settings.treatment_rate, settings.aov_treatment)
    df = pd.concat([control, treatment], ignore_index=True)
    return df.sample(frac=1.0, random_state=settings.seed).reset_index(drop=True)


def main() -> None:
    settings = get_settings()
    df = generate(settings)
    os.makedirs(settings.data_dir, exist_ok=True)
    path = os.path.join(settings.data_dir, "experiment.csv")
    df.to_csv(path, index=False)
    rates = df.groupby("group")["converted"].mean()
    print(f"Wrote {len(df)} visitors to {path}")
    print("Observed conversion rates:")
    print(rates.round(4).to_string())


if __name__ == "__main__":
    main()
