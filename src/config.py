"""Configuration from environment / ``.env``."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Defaults for the synthetic experiment and the analysis."""

    data_dir: str = _get("DATA_DIR", "data")

    # Synthetic experiment: per-arm sample size and true conversion rates.
    n_per_arm: int = int(_get("N_PER_ARM", "5000"))
    baseline_rate: float = float(_get("BASELINE_RATE", "0.12"))   # control
    treatment_rate: float = float(_get("TREATMENT_RATE", "0.138"))  # +1.8pp lift
    # Mean order value per converting user (for the revenue metric).
    aov_control: float = float(_get("AOV_CONTROL", "60.0"))
    aov_treatment: float = float(_get("AOV_TREATMENT", "61.0"))
    # Mean pre-experiment spend (a pre-period covariate used for CUPED).
    pre_aov: float = float(_get("PRE_AOV", "50.0"))

    # Analysis settings.
    alpha: float = float(_get("ALPHA", "0.05"))   # significance level
    power: float = float(_get("POWER", "0.80"))   # target power for sample sizing
    n_mc_samples: int = int(_get("N_MC_SAMPLES", "100000"))  # Bayesian Monte Carlo

    seed: int = int(_get("SEED", "42"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
