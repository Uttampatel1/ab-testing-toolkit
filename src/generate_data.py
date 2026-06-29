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
        # Latent per-visitor "spend propensity" (mean 1). It drives both the
        # pre-experiment covariate and in-experiment revenue, so the covariate is
        # predictive — which is exactly what CUPED exploits to cut variance.
        latent = rng.gamma(shape=4.0, scale=1.0 / 4.0, size=n)
        pre_revenue = np.clip(
            latent * settings.pre_aov + rng.normal(0.0, settings.pre_aov * 0.15, n),
            0.0, None,
        )

        # High-propensity visitors both convert more and spend more, so the
        # pre-period covariate predicts conversion too. Weights are normalised to
        # mean 1 so the per-arm conversion rate stays at the configured `rate`.
        weight = latent ** 0.8
        weight = weight / weight.mean()
        p_convert = np.clip(rate * weight, 0.0, 0.99)
        converted = rng.random(n) < p_convert
        # Revenue is positive only for converters; gamma keeps it non-negative.
        revenue = np.where(
            converted, latent * rng.gamma(shape=4.0, scale=aov / 4.0, size=n), 0.0
        )
        return pd.DataFrame(
            {"group": group, "converted": converted.astype(int),
             "revenue": np.round(revenue, 2),
             "pre_revenue": np.round(pre_revenue, 2)}
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
