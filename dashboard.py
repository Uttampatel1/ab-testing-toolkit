"""Streamlit A/B testing calculator and result explorer.

Run with::

    streamlit run dashboard.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from src.bayesian import beta_binomial_analysis
from src.frequentist import (
    minimum_detectable_effect,
    sample_size_proportion,
    two_proportion_ztest,
)

st.set_page_config(page_title="A/B Testing Toolkit", layout="wide")
st.title("🧪 A/B Testing Toolkit")

tab_analyze, tab_plan = st.tabs(["Analyse a result", "Plan a test"])

with tab_analyze:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Control")
        n_a = st.number_input("Visitors (A)", 100, 10_000_000, 5000, step=100)
        conv_a = st.number_input("Conversions (A)", 0, int(n_a), 600, step=10)
    with c2:
        st.subheader("Treatment")
        n_b = st.number_input("Visitors (B)", 100, 10_000_000, 5000, step=100)
        conv_b = st.number_input("Conversions (B)", 0, int(n_b), 690, step=10)

    alpha = st.slider("Significance level α", 0.01, 0.10, 0.05, 0.01)
    res = two_proportion_ztest(int(conv_a), int(n_a), int(conv_b), int(n_b), alpha)
    bayes = beta_binomial_analysis(int(conv_a), int(n_a), int(conv_b), int(n_b))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rate A", f"{conv_a / n_a:.2%}")
    m2.metric("Rate B", f"{conv_b / n_b:.2%}")
    m3.metric("p-value", f"{res.p_value:.4f}",
              "significant" if res.significant else "not significant")
    m4.metric("P(B > A)", f"{bayes.prob_b_beats_a:.1%}")

    st.write(
        f"**Absolute lift:** {res.estimate:+.3%}  "
        f"(95% CI {res.ci_low:+.3%} … {res.ci_high:+.3%})"
    )

    # Posterior densities of the two conversion rates.
    xs = np.linspace(
        max(0, min(conv_a / n_a, conv_b / n_b) - 0.05),
        min(1, max(conv_a / n_a, conv_b / n_b) + 0.05),
        400,
    )
    dens = pd.DataFrame(
        {
            "rate": xs,
            "Control": stats.beta.pdf(xs, 1 + conv_a, 1 + n_a - conv_a),
            "Treatment": stats.beta.pdf(xs, 1 + conv_b, 1 + n_b - conv_b),
        }
    ).set_index("rate")
    st.line_chart(dens)

with tab_plan:
    st.subheader("Sample size & minimum detectable effect")
    base = st.slider("Baseline conversion rate", 0.01, 0.50, 0.12, 0.01)
    mde = st.slider("Absolute lift to detect (MDE)", 0.005, 0.10, 0.02, 0.005)
    power = st.slider("Target power", 0.70, 0.95, 0.80, 0.05)
    alpha2 = st.slider("α (planning)", 0.01, 0.10, 0.05, 0.01)

    n_needed = sample_size_proportion(base, mde, alpha2, power)
    st.metric("Required sample size per arm", f"{n_needed:,}")

    n_have = st.number_input("If you only have N per arm", 100, 10_000_000, n_needed, step=100)
    detectable = minimum_detectable_effect(base, int(n_have), alpha2, power)
    st.metric("Smallest lift you could detect", f"{detectable:.3%}")
