# MatQuantLab V4

## Global Critical Materials Intelligence System

MatQuantLab is not a generic stock-prediction app.

It is a **materials-informed industrial stress radar** that tests whether metals, energy, macro, USD, volatility, geopolitical news, and exposure logic can provide early-warning signals for:

- mining and metals
- aerospace and defense
- industrials
- global equity indices
- additive manufacturing
- semiconductor/critical minerals exposure

## Live App

https://matquantlab-gegdszewybwbbbdvm5kbnc.streamlit.app/

Main app file:

```text
app/live_dashboard.py
```

## Core Message

> MatQuantLab transforms critical-materials, macro, and geopolitical stress into interpretable early-warning signals for industrial, mining, aerospace, defense, and additive-manufacturing assets.

## V4 Layers

1. **Materials prices** — metals, battery materials, rare earths, steel chain, energy, precious/stress materials.
2. **Global indices** — USA, Europe, China/Hong Kong, Japan, global and emerging markets.
3. **Macro stress** — USD, rates, VIX, credit proxies, FX, freight proxy.
4. **Geopolitical news** — critical minerals, sanctions, export controls, shipping disruption, defense spending.
5. **Exposure engine** — asset/material sensitivity scores.
6. **Prediction and validation** — prediction tool for all selected assets, regime detection, and toy backtest.

## What The App Shows

### Executive Radar
Global Critical Materials Stress Index, current regime, strongest stress drivers, and project interpretation.

### Prediction Tool for All
Trains a research model per selected asset and outputs:

- prediction direction
- latest predicted pressure
- confidence score
- test correlation
- RMSE
- top drivers
- warning summary

### Material Exposure Matrix
Maps assets to exposures such as copper, aluminum, energy, aerospace, defense, additive manufacturing, China sensitivity, and macro beta.

### Global Stress Map
Tracks USA, Europe, China/Hong Kong, Japan, global, and emerging-market indices.

### Geopolitical News
Uses a GDELT news query to monitor themes such as rare earths, sanctions, export controls, tariffs, shipping disruption, and defense spending.

### Regime Detection
Labels the environment as:

- Normal / mixed
- Commodity shock
- Energy-cost shock
- USD / commodity pressure
- Industrial slowdown
- Risk-off industrial stress
- Materials stress easing

### Validation / Backtest
Shows a simple out-of-sample validation table and toy sign-based equity curve.

## How To Understand The App

### GCMSI
Global Critical Materials Stress Index.

- High value: materials/macro stress is rising.
- Low value: materials stress is easing.
- High GCMSI + high VIX: possible industrial risk-off regime.

### Spearman IC
Measures whether a signal ranks future returns correctly.

- Positive: signal may lead stronger future return.
- Negative: signal may lead weakness.
- Near zero: weak or no relationship.

### Prediction Direction
The ML model's latest pressure estimate.

- Positive: model sees upward pressure.
- Negative: model sees downward pressure.

This is not a trading signal. It is a research indicator.

### Confidence
A normalized estimate of how large the prediction is relative to historical target volatility.

### Top Drivers
The stress variables most responsible for the model output.

## Data Sources

- Yahoo Finance via `yfinance`
- GDELT public news API
- User-defined exposure scores in `data/exposure_score_table.csv`

## Important Warning

This is a research and portfolio project, not financial advice.

The app does not include professional-grade transaction costs, slippage, liquidity, survivorship-bias controls, or audited market data.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/live_dashboard.py
```
