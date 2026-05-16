"""
MatQuantLab V2 — Intraday snapshot updater

This script downloads recent intraday data and saves snapshots into data/live/.
It does not replace the existing GitHub Actions research pipeline.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

import pandas as pd
import yfinance as yf
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment_config.yml"
LIVE_DIR = ROOT / "data" / "live"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default=None)
    parser.add_argument("--interval", default=None)
    args = parser.parse_args()

    config = load_config()
    assets = config["assets"]
    commodities = config["commodity_features"]
    interval = args.interval or config.get("default_interval", "5m")
    period = args.period or config.get("default_period", "5d")

    # Yahoo restrictions: keep 1m data short.
    if interval == "1m" and period not in {"1d", "5d"}:
        period = "5d"

    tickers = sorted(set(assets + list(commodities.values())))
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()

    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="all")
    prices.to_csv(LIVE_DIR / "latest_intraday_prices.csv")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": period,
        "interval": interval,
        "rows": int(len(prices)),
        "columns": list(prices.columns),
    }
    (LIVE_DIR / "latest_intraday_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved intraday snapshot:", metadata)


if __name__ == "__main__":
    main()
