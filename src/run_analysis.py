"""Run the full A/B analysis on the synthetic experiment and print a report.

Usage::

    python -m src.run_analysis
"""
from __future__ import annotations

import json
import os

from .config import get_settings
from .experiment import analyze
from .frequentist import minimum_detectable_effect, sample_size_proportion
from .generate_data import generate


def run() -> dict:
    settings = get_settings()
    df = generate(settings)
    report = analyze(df, settings=settings)

    # Planning context: what we *could* have detected, and what we'd need.
    planning = {
        "target_alpha": settings.alpha,
        "target_power": settings.power,
        "mde_at_current_n": round(
            minimum_detectable_effect(
                settings.baseline_rate, settings.n_per_arm, settings.alpha, settings.power
            ),
            5,
        ),
        "n_per_arm_for_2pp_lift": sample_size_proportion(
            settings.baseline_rate, 0.02, settings.alpha, settings.power
        ),
    }

    out = {"report": report.as_dict(), "planning": planning}

    os.makedirs(settings.data_dir, exist_ok=True)
    with open(os.path.join(settings.data_dir, "report.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(out, indent=2))
    verdict = (
        "SHIP treatment"
        if report.conversion_test.significant and report.conversion_test.estimate > 0
        else "INCONCLUSIVE / keep control"
    )
    print(f"\nDecision (frequentist, alpha={settings.alpha}): {verdict}")
    print(
        f"Bayesian P(treatment > control) = "
        f"{report.bayesian.prob_b_beats_a:.3f}"
    )
    return out


if __name__ == "__main__":
    run()
