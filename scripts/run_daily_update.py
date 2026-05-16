from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running directly without pip install -e .
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matquantlab.data_sources import (
    download_fred_series,
    download_yfinance_prices,
    get_all_yfinance_tickers,
    load_universe,
    make_demo_prices,
    save_frame,
)
from matquantlab.features import make_feature_matrix, safe_pct_change


def main() -> None:
    parser = argparse.ArgumentParser(description="Update MatQuantLab market data and features.")
    parser.add_argument("--config", default="config/universe.yaml")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data. Do not use for real research.")
    args = parser.parse_args()

    universe = load_universe(args.config)
    start = universe.get("project", {}).get("start_date", "2010-01-01")
    tickers = get_all_yfinance_tickers(universe)

    if args.demo:
        prices = make_demo_prices(universe, start="2015-01-01")
        macro = None
        source = "DEMO synthetic data"
    else:
        prices = download_yfinance_prices(tickers, start=start)
        fred = universe.get("fred_series", {})
        macro = download_fred_series(fred.keys(), start=start) if isinstance(fred, dict) else None
        source = "Yahoo Finance + FRED"

    features = make_feature_matrix(prices, universe)
    returns = safe_pct_change(prices)

    save_frame(prices, "data/raw/prices_yfinance.parquet")
    if macro is not None and not macro.empty:
        save_frame(macro, "data/raw/macro_fred.parquet")
    save_frame(returns, "data/processed/returns.parquet")
    save_frame(features, "data/processed/features.parquet")

    print("MatQuantLab data update complete")
    print(f"Source: {source}")
    print(f"Prices shape: {prices.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")
    print("Saved data/raw/prices_yfinance.parquet")
    print("Saved data/processed/features.parquet")


if __name__ == "__main__":
    main()
