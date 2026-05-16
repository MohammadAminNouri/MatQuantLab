from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def safe_pct_change(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return df.pct_change(periods=periods, fill_method=None).replace([np.inf, -np.inf], np.nan)


def rolling_zscore(x: pd.Series | pd.DataFrame, window: int = 252, min_periods: int = 60):
    mean = x.rolling(window, min_periods=min_periods).mean()
    std = x.rolling(window, min_periods=min_periods).std()
    return (x - mean) / std.replace(0, np.nan)


def realized_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)


def max_drawdown_rolling(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    roll_max = prices.rolling(window, min_periods=min(window, max(3, window // 3))).max()
    return prices / roll_max - 1.0


def make_global_features(
    prices: pd.DataFrame,
    commodity_macro_tickers: Iterable[str],
    windows: Iterable[int] = (5, 20, 60),
) -> pd.DataFrame:
    """Create global commodity/macro shock features."""
    commodity_macro_tickers = [t for t in commodity_macro_tickers if t in prices.columns]
    returns = safe_pct_change(prices[commodity_macro_tickers])
    out = pd.DataFrame(index=prices.index)

    for w in windows:
        mom = safe_pct_change(prices[commodity_macro_tickers], periods=w)
        z = rolling_zscore(mom, window=252, min_periods=80)
        for col in mom.columns:
            clean = col.replace("^", "").replace("=", "_").replace(".", "_").replace("-", "_")
            out[f"ret_{w}d__{clean}"] = mom[col]
            out[f"shock_z_{w}d__{clean}"] = z[col]

    # CMSI: average absolute standardized stress across key commodity/macro shocks.
    shock20 = [c for c in out.columns if c.startswith("shock_z_20d__")]
    if shock20:
        raw_stress = out[shock20].abs().mean(axis=1)
        out["critical_materials_stress_index"] = rolling_zscore(raw_stress, 252, 80)
    else:
        out["critical_materials_stress_index"] = np.nan

    return out.replace([np.inf, -np.inf], np.nan)


def make_asset_features(prices: pd.DataFrame, asset_tickers: Iterable[str]) -> pd.DataFrame:
    """Create a wide dataframe of asset-specific momentum, volatility, and drawdown features."""
    asset_tickers = [t for t in asset_tickers if t in prices.columns]
    rets = safe_pct_change(prices[asset_tickers])
    out = pd.DataFrame(index=prices.index)
    for w in (5, 20, 60):
        mom = safe_pct_change(prices[asset_tickers], periods=w)
        vol = realized_volatility(rets, window=w)
        dd = max_drawdown_rolling(prices[asset_tickers], window=w)
        for t in asset_tickers:
            clean = t.replace("^", "").replace("=", "_").replace(".", "_").replace("-", "_")
            out[f"asset_mom_{w}d__{clean}"] = mom[t]
            out[f"asset_vol_{w}d__{clean}"] = vol[t]
            out[f"asset_drawdown_{w}d__{clean}"] = dd[t]
    return out.replace([np.inf, -np.inf], np.nan)


def make_feature_matrix(prices: pd.DataFrame, universe: dict) -> pd.DataFrame:
    from .data_sources import get_asset_tickers, get_market_tickers

    assets = get_asset_tickers(universe)
    markets = get_market_tickers(universe)
    global_features = make_global_features(prices, markets)
    asset_features = make_asset_features(prices, assets)
    features = pd.concat([global_features, asset_features], axis=1)
    return features.sort_index()


def make_forward_returns(prices: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int]) -> dict[int, pd.DataFrame]:
    assets = [a for a in assets if a in prices.columns]
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        out[int(h)] = prices[assets].pct_change(periods=int(h), fill_method=None).shift(-int(h))
    return out


def make_asset_panel(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    assets: Iterable[str],
    horizon: int = 20,
    max_rows_per_asset: int | None = None,
) -> pd.DataFrame:
    """Create long supervised panel: date, asset, global features, asset-local features, target."""
    assets = [a for a in assets if a in prices.columns]
    global_cols = [c for c in features.columns if not c.startswith("asset_")]
    rows = []
    for asset in assets:
        clean = asset.replace("^", "").replace("=", "_").replace(".", "_").replace("-", "_")
        local_cols = [c for c in features.columns if c.endswith(f"__{clean}")]
        cols = global_cols + local_cols
        target = prices[asset].pct_change(periods=horizon, fill_method=None).shift(-horizon)
        frame = features[cols].copy()
        rename = {c: c.split("__")[0] if c.endswith(f"__{clean}") else c for c in local_cols}
        frame = frame.rename(columns=rename)
        frame["target_fwd_return"] = target
        frame["date"] = frame.index
        frame["asset"] = asset
        frame = frame.dropna(subset=["target_fwd_return"])
        if max_rows_per_asset is not None and len(frame) > max_rows_per_asset:
            frame = frame.iloc[-max_rows_per_asset:]
        rows.append(frame.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    # Keep rows with at least some non-missing features, then fill remaining with expanding/zero-safe median later in model pipeline.
    feature_cols = [c for c in panel.columns if c not in {"date", "asset", "target_fwd_return"}]
    panel = panel.dropna(subset=feature_cols, how="all")
    return panel.sort_values(["date", "asset"]).reset_index(drop=True)


def make_signal_decay_table(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    assets: Iterable[str],
    feature_names: Iterable[str],
    horizons: Iterable[int] = (5, 20, 60),
) -> pd.DataFrame:
    """Compute Spearman IC for feature vs future return by asset and horizon."""
    from scipy.stats import spearmanr

    rows = []
    for h in horizons:
        fwd = make_forward_returns(prices, assets, [h])[int(h)]
        for asset in fwd.columns:
            y = fwd[asset]
            for feat in feature_names:
                if feat not in features.columns:
                    continue
                joined = pd.concat([features[feat], y], axis=1).dropna()
                if len(joined) < 80:
                    continue
                ic = spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).correlation
                rows.append({"horizon": int(h), "asset": asset, "feature": feat, "spearman_ic": ic, "n_obs": len(joined)})
    return pd.DataFrame(rows)
