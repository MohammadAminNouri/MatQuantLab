# MatQuantLab V2 — Live / Interactive Dashboard Guide

This V2 patch adds interactivity without damaging the current working pipeline.

## What V2 adds

- `app/live_dashboard.py`
- `config/experiment_config.yml`
- `scripts/intraday_update.py`
- `.github/workflows/live_snapshot.yml`

## Important limitation

This is not high-frequency trading. Free Yahoo/yfinance data can be delayed, incomplete, or rate-limited.

GitHub Actions cannot truly run every minute. The practical minimum scheduled interval is around 5 minutes, and GitHub can delay scheduled jobs. Therefore:

- Use Streamlit for interactive refresh.
- Use GitHub Actions for 5-minute or daily snapshots.
- Use audited paid market data for real professional trading research.

## How to use on GitHub only

1. Upload the V2 patch into the same repository.
2. Go to **Actions**.
3. Open **MatQuantLab Intraday Snapshot**.
4. Click **Run workflow**.
5. After it succeeds, open:
   - `data/live/latest_intraday_prices.csv`
   - `data/live/latest_intraday_metadata.json`

## How to use as an interactive dashboard

Deploy this file on Streamlit Community Cloud:

```text
app/live_dashboard.py
```

The app lets you change:

- asset
- interval
- period
- ML model
- prediction horizon
- shock window
- refresh behavior

## Good first experiments

Try:

| Asset | Interval | Period | Model | Why |
|---|---:|---:|---|---|
| XME | 5m | 5d | Ridge | mining/materials signal |
| COPX | 5m | 5d | Ridge | copper miners |
| ITA | 15m | 1mo | RandomForest | aerospace/defense |
| PRNT | 15m | 1mo | Ridge | additive manufacturing |
| XLI | 1d | 1y | RandomForest | industrial macro behavior |

## What to look at

- Latest CMSI value
- Spearman IC table
- Prediction correlation
- Feature importance
- Toy equity curve

## What not to claim

Do not claim this is a trading bot or a profitable live strategy. Say:

> This is an interactive research dashboard for testing whether commodity and macro stress signals have predictive relationships with industrial and materials-linked assets.
