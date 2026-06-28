"""Tie the analyses together: from a tidy experiment frame to a full report."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .bayesian import BayesianResult, beta_binomial_analysis
from .config import Settings, get_settings
from .frequentist import TestResult, two_proportion_ztest, welch_ttest


@dataclass
class ExperimentReport:
    n_control: int
    n_treatment: int
    rate_control: float
    rate_treatment: float
    conversion_test: TestResult       # frequentist z-test on conversion
    revenue_test: TestResult          # Welch t-test on revenue per visitor
    bayesian: BayesianResult

    def as_dict(self) -> dict:
        return {
            "samples": {"control": self.n_control, "treatment": self.n_treatment},
            "conversion_rate": {
                "control": round(self.rate_control, 4),
                "treatment": round(self.rate_treatment, 4),
                "absolute_lift": round(self.rate_treatment - self.rate_control, 4),
                "relative_lift_pct": round(
                    (self.rate_treatment / self.rate_control - 1) * 100, 2
                ),
            },
            "frequentist_conversion": self.conversion_test.as_dict(),
            "frequentist_revenue_per_visitor": self.revenue_test.as_dict(),
            "bayesian_conversion": self.bayesian.as_dict(),
        }


def analyze(
    df: pd.DataFrame,
    control: str = "control",
    treatment: str = "treatment",
    settings: Settings | None = None,
) -> ExperimentReport:
    settings = settings or get_settings()
    a = df[df["group"] == control]
    b = df[df["group"] == treatment]

    conv_a, n_a = int(a["converted"].sum()), len(a)
    conv_b, n_b = int(b["converted"].sum()), len(b)

    conv_test = two_proportion_ztest(conv_a, n_a, conv_b, n_b, settings.alpha)
    rev_test = welch_ttest(a["revenue"].to_numpy(), b["revenue"].to_numpy(), settings.alpha)
    bayes = beta_binomial_analysis(
        conv_a, n_a, conv_b, n_b,
        n_samples=settings.n_mc_samples, seed=settings.seed,
    )

    return ExperimentReport(
        n_control=n_a,
        n_treatment=n_b,
        rate_control=conv_a / n_a,
        rate_treatment=conv_b / n_b,
        conversion_test=conv_test,
        revenue_test=rev_test,
        bayesian=bayes,
    )
