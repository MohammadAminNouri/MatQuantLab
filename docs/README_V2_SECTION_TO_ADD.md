## V2 — Interactive Live Research Dashboard

MatQuantLab now includes an interactive dashboard for testing critical-materials signals with intraday or daily market data.

### Run the dashboard

```bash
streamlit run app/live_dashboard.py
```

### Dashboard controls

The dashboard lets you choose:

- target asset: `XME`, `COPX`, `ITA`, `XAR`, `PRNT`, etc.
- interval: `1m`, `5m`, `15m`, `30m`, `60m`, `1d`
- period: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`
- model: `Ridge` or `RandomForest`
- prediction horizon
- shock window
- manual refresh / auto-refresh

### Intraday snapshot workflow

The repository also includes:

```text
.github/workflows/live_snapshot.yml
```

This can create periodic intraday snapshots in:

```text
data/live/
```

### Research warning

This is not a live trading system. Free market data can be delayed, rate-limited, or incomplete. The dashboard is for research and portfolio demonstration only.
