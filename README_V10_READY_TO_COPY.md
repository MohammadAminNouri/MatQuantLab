# MatQuantLab V10 — Universal Interval Critical Materials Terminal

V10 fixes the V9 empty-feature crash.

## Main fixes

- Works from 1-minute monitoring to 10-year daily mode
- Automatically adjusts invalid period/interval combinations
- Adaptive shock window for short and long datasets
- Fallback price features if material/future configs do not produce enough usable features
- Raw Data diagnostics show actual period, feature status, feature rows and price coverage
- Futures/commodities can be predicted directly
- ML signals still show valid-until estimate, data coverage and model validation

## Best settings

Long-term ML:
```text
Interval: 1d
Lookback: 5y or 10y
Model: AutoBest
Prediction horizon: 5–20
Shock window: 20
```

Fast terminal:
```text
Interval: 1m or 5m
Lookback: 1d or 5d
Model: AutoBest
Prediction horizon: 3–12
Shock window: 5–20
```

BUY/SELL labels are ML-only research labels, not financial advice.
