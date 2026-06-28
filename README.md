# 🧪 A/B Testing Toolkit

**Business question:** *Variant B's conversion rate looks higher than A's — but is the difference real, how confident are we, and should we ship it? And before we even start: how big a sample do we actually need?*

This toolkit answers all of that with **both** a frequentist and a Bayesian lens, plus the experiment-design calculators (sample size, power, minimum detectable effect) that keep a test honest *before* data collection begins.

---

## Key result (synthetic experiment: 5,000 visitors/arm)

| | Control | Treatment | Lift |
|--|--------:|----------:|-----:|
| **Conversion rate** | 11.98% | 14.00% | **+2.02pp (+16.9% relative)** |

| Method | Verdict |
|--------|---------|
| **Frequentist** two-proportion z-test | p = **0.003** → significant; 95% CI on lift excludes 0 |
| **Frequentist** Welch t-test (revenue/visitor) | p = 0.004 → significant |
| **Bayesian** Beta-Binomial | **P(treatment > control) = 99.8%**, expected uplift +17.0%, expected loss ≈ 3e-6 |
| **Decision** | ✅ **Ship treatment** — both frameworks agree |

*(Reproducible from `python -m src.run_analysis` on the default synthetic data and seed.)*

**Planning context the toolkit also reports:**
- **MDE at this sample size:** ~1.9pp — the smallest lift this test could reliably catch at 80% power. Our +2.0pp effect sits just above it, which is *why* the result is significant.
- **Sample size for a 2pp lift:** ~4,438 visitors/arm needed for 80% power — so 5,000/arm was adequately powered. This is the check that prevents underpowered "we saw nothing" tests.

**A data scientist's read on the result:**
- Frequentist and Bayesian **agreeing** is the comfortable case. They answer different questions — *"how surprising is this data if there's no effect?"* (p-value) vs *"what's the probability B is actually better?"* (posterior) — and the Bayesian **expected loss** is what you'd actually wire into an automated ship/no-ship rule.
- **Power before p-values.** An A/B result is only interpretable if the test was designed to detect a business-meaningful effect. The sizing tools make that explicit.
- **Don't peek.** Repeatedly checking significance inflates false positives; decide the sample size up front (or use a proper sequential method) — see "Extensions".

## How it works

```
synthetic experiment ─► per-arm conversions & revenue
        │
        ├─ Frequentist ─► two-proportion z-test (conversion)
        │                 Welch t-test (revenue)  + Wald CIs
        │
        ├─ Bayesian ────► Beta-Binomial posteriors ─► P(B>A),
        │                 expected uplift, expected loss, credible intervals
        │
        └─ Design ──────► sample size · power · minimum detectable effect
```

## Beyond the basic A/B test

Two upgrades (in `src/sequential.py`) take this past a single fixed-horizon test:

- **Always-valid confidence sequences** (`always_valid_ci`) — an interval you can
  inspect after *every* visitor and stop the moment it excludes your null, without
  inflating the false-positive rate. This is the principled answer to "can I peek?".
  The cost of anytime-validity is a wider interval than the fixed-`n` Wald CI.
- **Multi-variant testing with Holm-Bonferroni** (`compare_variants`) — run A/B/C/…
  against a control while controlling the family-wise error rate, so five variants
  don't silently turn a 5% error budget into ~23%.

```python
from src.sequential import always_valid_ci, compare_variants
always_valid_ci(120, 1000)                       # (lo, hi) — safe to peek
compare_variants((100, 2000), {"B": (180, 2000), "C": (105, 2000)})
```

## Tech stack

- **Stats:** SciPy (`norm`, `t`, `beta`, `ttest_ind`), NumPy, pandas
- **App:** Streamlit (result analyser + experiment-design calculator)
- **Sequential / multi-arm:** asymptotic confidence sequences + Holm-Bonferroni
- **Observability:** structured logging via `src/logging_utils.py` (`LOG_LEVEL` env)
- **Deploy:** `Dockerfile` + `docker-compose.yml`; GitHub Actions CI runs the suite
- **Tests:** pytest (25 tests, including power/sample-size self-consistency, confidence-sequence width, and multiplicity correction)

## Setup & run

```bash
cd 09-ab-testing-toolkit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.generate_data    # synthetic experiment.csv
python -m src.run_analysis     # full frequentist + Bayesian report (+ report.json)
streamlit run dashboard.py     # interactive analyser & sample-size calculator
pytest -q                      # run tests
```

## Project structure

```
09-ab-testing-toolkit/
├── dashboard.py            # Streamlit analyser + design calculator
├── src/
│   ├── config.py           # typed settings from .env
│   ├── generate_data.py    # synthetic two-arm experiment
│   ├── frequentist.py      # z-test, t-test, CIs, power, sample size, MDE
│   ├── bayesian.py         # Beta-Binomial: P(B>A), uplift, expected loss
│   ├── sequential.py       # always-valid confidence sequences + Holm multi-arm
│   ├── experiment.py       # end-to-end report from a data frame
│   ├── logging_utils.py    # structured logging + timing
│   └── run_analysis.py     # CLI report + ship/no-ship verdict
├── tests/                  # 25 pytest tests
├── Dockerfile              # containerised Streamlit app
├── docker-compose.yml
├── .github/workflows/ci.yml
├── .env.example
├── requirements.txt
└── .gitignore
```

## Possible extensions

- **CUPED variance reduction** using pre-experiment covariates to reach significance with fewer users.
- **Segmentation / heterogeneous treatment effects** (does B help new users but hurt returning ones?).
- **Ratio & count metrics** (revenue-per-user with the delta method, Poisson tests).
```
