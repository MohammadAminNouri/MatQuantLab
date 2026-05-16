# MatQuantLab

**Critical Materials Alpha Research Lab**

MatQuantLab is a Python research project that tests whether **industrial metals, energy shocks, macro stress, and critical-materials supply-chain signals** can predict future returns and drawdown risk in materials, mining, aerospace, defense, industrial, and additive-manufacturing-linked equities.

This is not a generic “stock prediction” project. The goal is to combine:

- materials engineering intuition
- commodity and energy markets
- aerospace / defense / industrial supply-chain exposure
- machine learning
- signal decay analysis
- regime detection
- realistic backtesting with transaction costs
- overfitting checks
- an interactive dashboard

> Research question: **Can materials-engineering knowledge improve cross-asset financial prediction?**

---

## What makes this project different

Most beginner quant projects try to predict Apple or the S&P 500 with an LSTM. MatQuantLab instead asks a more niche question:

> Do copper, aluminum, energy, USD, rates, and volatility shocks contain delayed information about mining, aerospace, defense, industrial, and additive-manufacturing equities?

The core custom feature is the **Critical Materials Stress Index (CMSI)**, a rolling stress proxy built from commodity, energy, macro, and volatility shocks.

---

## Project pipeline

```text
data download
    ↓
feature engineering
    ↓
Critical Materials Stress Index
    ↓
signal decay analysis
    ↓
walk-forward ML models
    ↓
long-short backtest with costs
    ↓
regime analysis
    ↓
interactive Streamlit dashboard
```

---

## Repository structure

```text
MatQuantLab/
├── app/
│   └── streamlit_app.py
├── config/
│   └── universe.yaml
├── docs/
│   ├── BEGINNER_UPLOAD_STEPS.md
│   ├── RESEARCH_PLAN.md
│   ├── NOTEBOOK_MARKDOWNS.md
│   └── LIMITATIONS_AND_ETHICS.md
├── notebooks/
│   ├── 00_RUN_ME_FIRST_COLAB.ipynb
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_signal_decay_analysis.ipynb
│   ├── 04_ml_prediction_models.ipynb
│   ├── 05_backtest_with_costs.ipynb
│   ├── 06_regime_detection.ipynb
│   └── 07_final_report.ipynb
├── scripts/
│   ├── run_daily_update.py
│   ├── generate_research_report.py
│   └── run_all.py
├── src/matquantlab/
│   ├── data_sources.py
│   ├── features.py
│   ├── models.py
│   ├── backtest.py
│   ├── regimes.py
│   ├── overfitting.py
│   └── visualization.py
└── tests/
    └── test_pipeline.py
```

---

## Beginner path: run everything

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/run_all.py
```

### Interactive dashboard

```bash
streamlit run app/streamlit_app.py
```

### Google Colab

Open:

```text
notebooks/00_RUN_ME_FIRST_COLAB.ipynb
```

Then click **Open in Colab** from GitHub.

---

## Main ML models

MatQuantLab starts with transparent and robust models before using complex ones:

1. Ridge Regression
2. Elastic Net
3. Random Forest
4. Gradient Boosting
5. optional XGBoost / LightGBM later

The project uses **walk-forward validation**, not random train/test split.

---

## Key outputs

The scripts create:

```text
outputs/figures/cmsi.png
outputs/figures/signal_decay_heatmap.png
outputs/figures/model_leaderboard.png
outputs/figures/feature_importance.png
outputs/figures/backtest_equity_curve.png
outputs/tables/model_leaderboard.csv
outputs/tables/signal_decay.csv
outputs/tables/backtest_summary.csv
outputs/research_summary.md
```

---

## Current research questions

1. Does the Critical Materials Stress Index predict future returns of mining, aerospace, industrial, and additive-manufacturing equities?
2. Which forecast horizon is strongest: 5D, 20D, or 60D?
3. Do commodity shocks work differently for miners versus aerospace and defense firms?
4. Does ML outperform linear baselines?
5. Does the strategy survive transaction costs?
6. Is the signal stable across regimes?
7. Is the result real, or just overfitting?

---

## Important warning

This is a research and education project, **not financial advice**. Free market data can contain missing data, survivorship bias, corporate-action issues, and timestamp problems. Any result must be treated as a hypothesis, not a trading recommendation.

---

## Status

- [x] data pipeline
- [x] feature engineering
- [x] Critical Materials Stress Index
- [x] signal decay analysis
- [x] walk-forward ML
- [x] transaction-cost backtest
- [x] regime analysis
- [x] Streamlit dashboard
- [x] GitHub Actions workflow
## Automated Research Outputs

### Critical Materials Stress Index
![CMSI](outputs/figures/cmsi.png)

### Lead-Lag Signal Heatmap
![Heatmap](outputs/figures/signal_decay_heatmap.png)

### ML Model Comparison
![Models](outputs/figures/model_leaderboard.png)

### Strategy Equity Curve
![Backtest](outputs/figures/backtest_equity_curve.png)

### Feature Importance
![Importance](outputs/figures/feature_importance.png)
## How to use this repo without local installation

1. Open the Actions tab.
2. Select Run MatQuantLab Research Pipeline.
3. Click Run workflow.
4. After it finishes, open outputs/figures and outputs/research_summary.md.
5. Use the generated charts to inspect critical-materials signals and ML model behavior.
