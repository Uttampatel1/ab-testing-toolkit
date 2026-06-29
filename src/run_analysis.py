"""Run the full A/B analysis on the synthetic experiment and print a report.

Usage::

    python -m src.run_analysis
"""
from __future__ import annotations

import json
import os

from .config import get_settings
from .cuped import cuped_welch_ttest
from .experiment import analyze
from .frequentist import minimum_detectable_effect, sample_size_proportion
from .generate_data import generate
from .logging_utils import get_logger, log_timing
from .sequential import always_valid_ci
from .srm import srm_test

log = get_logger(__name__)


def run() -> dict:
    settings = get_settings()
    log.info(
        "Analysing experiment: %d/arm, baseline=%.3f, alpha=%.3f",
        settings.n_per_arm, settings.baseline_rate, settings.alpha,
    )
    df = generate(settings)
    with log_timing(log, "frequentist + Bayesian analysis"):
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

    # Anytime-valid CIs: safe to inspect mid-experiment without inflating error.
    ctrl = df[df["group"] == "control"]
    trt = df[df["group"] == "treatment"]
    anytime = {
        "control_cs": [round(x, 5) for x in always_valid_ci(
            int(ctrl["converted"].sum()), len(ctrl), settings.alpha)],
        "treatment_cs": [round(x, 5) for x in always_valid_ci(
            int(trt["converted"].sum()), len(trt), settings.alpha)],
        "note": "Confidence sequences — valid even if you peek after every visitor.",
    }

    # Guardrail: are the arms even the size we intended? (run before trusting anything)
    srm = srm_test([len(ctrl), len(trt)])

    # CUPED: use the pre-experiment covariate to cut variance on the revenue metric.
    cuped = cuped_welch_ttest(
        ctrl["revenue"].to_numpy(), ctrl["pre_revenue"].to_numpy(),
        trt["revenue"].to_numpy(), trt["pre_revenue"].to_numpy(),
        settings.alpha,
    )

    out = {
        "srm_guardrail": srm.as_dict(),
        "report": report.as_dict(),
        "cuped_revenue": cuped.as_dict(),
        "planning": planning,
        "anytime_valid": anytime,
    }

    os.makedirs(settings.data_dir, exist_ok=True)
    with open(os.path.join(settings.data_dir, "report.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(out, indent=2))
    if not srm.passed:
        print(f"\n⚠️  SRM DETECTED (p={srm.p_value:.2e}) — arms aren't comparable; "
              "fix the pipeline before trusting any result.")
    print(
        f"\nCUPED variance reduction on revenue: {cuped.variance_reduction * 100:.1f}% "
        f"(naive p={cuped.naive.p_value:.4f} -> CUPED p={cuped.adjusted.p_value:.4f})"
    )
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
