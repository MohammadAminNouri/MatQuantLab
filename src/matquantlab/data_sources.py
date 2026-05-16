from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_universe(path: str | Path = "config/universe.yaml") -> dict:
    """Load the project universe YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_ticker_dict(d: dict) -> Dict[str, str]:
    """Flatten a nested YAML ticker dictionary into ticker -> description."""
    out: Dict[str, str] = {}
    for _, group in d.items():
        if isinstance(group, dict):
            for k, v in group.items():
                out[str(k)] = str(v)
    return out


def get_asset_tickers(universe: dict) -> List[str]:
    return sorted(flatten_ticker_dict(universe.get("assets", {})).keys())


def get_market_tickers(universe: dict) -> List[str]:
    return sorted(flatten_ticker_dict(universe.get("commodities_and_macro", {})).keys())


def get_all_yfinance_tickers(universe: dict) -> List[str]:
    tickers = get_asset_tickers(universe) + get_market_tickers(universe)
    return sorted(dict.fromkeys(tickers))


def download_yfinance_prices(
    tickers: Iterable[str],
    start: str = "2010-01-01",
    end: str | None = None,
    min_non_null: int = 120,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance via yfinance.

    Some tickers may fail. The function drops columns with too little data.
    """
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        raise RuntimeError("yfinance returned no data. Check internet connection or tickers.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
        elif "Adj Close" in data.columns.get_level_values(0):
            close = data["Adj Close"].copy()
        else:
            raise RuntimeError(f"Could not find Close prices in columns: {data.columns}")
    else:
        close = data.to_frame(tickers[0]) if isinstance(data, pd.Series) else data.copy()

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    close = close.dropna(axis=1, thresh=min_non_null)
    close = close.loc[:, ~close.columns.duplicated()]
    return close


def download_fred_series(series: Iterable[str], start: str = "2010-01-01") -> pd.DataFrame:
    """Download selected FRED series using pandas-datareader.

    FRED does not require an API key for this simple use case.
    """
    try:
        from pandas_datareader import data as pdr
    except Exception:
        return pd.DataFrame()

    frames = []
    for s in series:
        try:
            x = pdr.DataReader(s, "fred", start=start)
            x.columns = [s]
            frames.append(x)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.ffill()


def save_frame(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception:
        # Fallback so the project still works on minimal Python installs.
        df.to_pickle(path)


def load_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def make_demo_prices(universe: dict, start: str = "2015-01-01", seed: int = 7) -> pd.DataFrame:
    """Create synthetic demo data for testing only.

    This is not used for real research unless --demo is passed.
    """
    rng = np.random.default_rng(seed)
    tickers = get_all_yfinance_tickers(universe)
    dates = pd.bdate_range(start=start, end=pd.Timestamp.today().normalize())
    n = len(dates)

    market = rng.normal(0.00025, 0.010, size=n)
    metal_cycle = np.zeros(n)
    for i in range(1, n):
        metal_cycle[i] = 0.96 * metal_cycle[i-1] + rng.normal(0, 0.006)

    prices = {}
    for t in tickers:
        noise = rng.normal(0, 0.012, size=n)
        beta_metal = 0.0
        if t in {"XME", "COPX", "PICK", "FCX", "SCCO", "AA", "XLB"}:
            beta_metal = 0.45
        if t in {"ITA", "XAR", "LMT", "RTX", "NOC", "GD", "BA"}:
            beta_metal = -0.12
        if t in {"HG=F", "ALI=F", "SI=F", "PL=F", "PA=F"}:
            beta_metal = 0.80
        rets = market + beta_metal * np.roll(metal_cycle, 5) + noise
        prices[t] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(prices, index=dates)
