"""
MatQuantLab V2 — Interactive Live Dashboard

This app is designed to be added to the existing MatQuantLab repository without
breaking the current GitHub Actions pipeline.

Run locally:
    streamlit run app/live_dashboard.py

Deploy:
    Streamlit Community Cloud -> app/live_dashboard.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment_config.yml"


def read_config() -> dict:
    import yaml

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "assets": ["XME", "COPX", "ITA", "XAR", "PRNT", "SPY"],
        "commodity_features": {
            "Copper": "HG=F",
            "Aluminum": "ALI=F",
            "Oil": "CL=F",
            "Gold": "GC=F",
            "USD": "DX-Y.NYB",
            "VIX": "^VIX",
        },
        "intervals": ["1m", "5m", "15m", "1d"],
        "periods": ["1d", "5d", "1mo", "1y"],
    }


def safe_period_for_interval(interval: str, requested_period: str) -> str:
    """Yahoo has restrictions for intraday data. Keep choices safe."""
    if interval == "1m":
        return requested_period if requested_period in {"1d", "5d"} else "5d"
    if interval in {"2m", "5m", "15m", "30m", "60m", "90m"}:
        return requested_period if requested_period in {"1d", "5d", "1mo"} else "1mo"
    return requested_period


@st.cache_data(ttl=60, show_spinner=False)
def download_prices(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.xs("Close", axis=1, level=1, drop_level=True).copy()
    else:
        prices = raw[["Close"]].copy()
        if len(tickers) == 1:
            prices.columns = tickers

    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="all")
    return prices


def build_features(prices: pd.DataFrame, commodity_map: dict[str, str], window: int = 20) -> pd.DataFrame:
    features = pd.DataFrame(index=prices.index)

    for name, ticker in commodity_map.items():
        if ticker in prices.columns:
            ret = prices[ticker].pct_change(window)
            roll_mean = ret.rolling(max(window * 4, 20), min_periods=max(window, 5)).mean()
            roll_std = ret.rolling(max(window * 4, 20), min_periods=max(window, 5)).std()
            features[f"{name}_shock"] = (ret - roll_mean) / roll_std

    shock_cols = list(features.columns)
    if shock_cols:
        features["CMSI"] = features[shock_cols].mean(axis=1)

    return features.replace([np.inf, -np.inf], np.nan).dropna()


def make_forward_return(prices: pd.DataFrame, asset: str, horizon_steps: int) -> pd.Series:
    return prices[asset].pct_change(horizon_steps).shift(-horizon_steps).rename("future_return")


def train_model(df: pd.DataFrame, feature_cols: list[str], model_name: str):
    if len(df) < 80:
        return None, None, None, "Not enough rows for ML. Try longer period or daily interval."

    split = int(len(df) * 0.75)
    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[feature_cols].fillna(0.0)
    y_train = train["future_return"].astype(float)
    X_test = test[feature_cols].fillna(0.0)
    y_test = test["future_return"].astype(float)

    if model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators=150, max_depth=4, random_state=42)
    else:
        model = Ridge(alpha=10.0)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = float(mean_squared_error(y_test, pred) ** 0.5)
    corr = pd.Series(pred).corr(pd.Series(y_test.values))
    corr = float(corr) if not pd.isna(corr) else 0.0

    result = test.copy()
    result["prediction"] = pred

    metrics = {"rmse": rmse, "prediction_corr": corr, "n_train": len(train), "n_test": len(test)}
    return model, result, metrics, None


def long_short_backtest(result: pd.DataFrame) -> pd.DataFrame:
    # Single-asset toy sign strategy: long if prediction positive, short if negative.
    out = result[["future_return", "prediction"]].copy().dropna()
    if out.empty:
        return pd.DataFrame(columns=["strategy_return", "equity"])

    out["position"] = np.where(out["prediction"] > 0, 1.0, -1.0)
    out["strategy_return"] = out["position"] * out["future_return"]
    out["equity"] = (1.0 + out["strategy_return"].fillna(0.0)).cumprod()
    return out


def feature_importance(model, feature_cols: list[str]) -> pd.Series:
    if model is None:
        return pd.Series(dtype=float)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(getattr(model, "coef_", np.zeros(len(feature_cols))))
    return pd.Series(values, index=feature_cols).sort_values(ascending=False)


st.set_page_config(page_title="MatQuantLab Live", layout="wide")
st.title("MatQuantLab V2 — Interactive Critical Materials Signal Lab")
st.caption("Research-only dashboard. This is not investment advice and not a live trading system.")

config = read_config()
assets = config.get("assets", ["XME", "COPX", "ITA", "XAR", "PRNT", "SPY"])
commodity_map = config.get("commodity_features", {})
intervals = config.get("intervals", ["1m", "5m", "15m", "1d"])
periods = config.get("periods", ["1d", "5d", "1mo", "1y"])

with st.sidebar:
    st.header("Experiment Controls")

    asset = st.selectbox("Target asset", assets, index=assets.index(config.get("default_asset", assets[0])) if config.get("default_asset", assets[0]) in assets else 0)
    interval = st.selectbox("Data interval", intervals, index=intervals.index(config.get("default_interval", intervals[0])) if config.get("default_interval", intervals[0]) in intervals else 0)
    period = st.selectbox("Data period", periods, index=periods.index(config.get("default_period", periods[0])) if config.get("default_period", periods[0]) in periods else 0)
    model_name = st.selectbox("ML model", config.get("models", ["Ridge", "RandomForest"]))

    horizon_steps = st.slider("Prediction horizon in bars", min_value=1, max_value=120, value=12, step=1)
    z_window = st.slider("Shock window in bars", min_value=3, max_value=80, value=20, step=1)

    auto_refresh = st.toggle("Auto-refresh every 60 seconds", value=False)
    refresh = st.button("Refresh latest data")

if auto_refresh:
    time.sleep(1)
    st.rerun()

safe_period = safe_period_for_interval(interval, period)

all_tickers = sorted(set([asset] + list(commodity_map.values())))
with st.spinner("Downloading latest market data..."):
    prices = download_prices(all_tickers, safe_period, interval)

if prices.empty or asset not in prices.columns:
    st.error("No valid data returned. Try a different interval/period or asset.")
    st.stop()

features = build_features(prices, commodity_map, window=z_window)
if features.empty:
    st.error("Features could not be created. Try a longer period or larger interval.")
    st.stop()

future_return = make_forward_return(prices, asset, horizon_steps)
df = pd.concat([features, future_return], axis=1, join="inner").dropna()

latest_time = prices.index[-1]
st.success(f"Latest update loaded. Last timestamp: {latest_time}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Asset", asset)
k2.metric("Last price", f"{prices[asset].dropna().iloc[-1]:.2f}")
k3.metric("Latest CMSI", f"{features['CMSI'].iloc[-1]:.3f}" if "CMSI" in features else "N/A")
k4.metric("Rows for ML", len(df))

tab1, tab2, tab3, tab4 = st.tabs(["Live Signals", "ML Model", "Backtest", "Data"])

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"{asset} price")
        st.line_chart(prices[asset].dropna())

    with c2:
        st.subheader("Critical Materials Stress Index")
        if "CMSI" in features:
            st.line_chart(features["CMSI"])

    st.subheader("Signal correlation with future return")
    rows = []
    for col in features.columns:
        aligned = pd.concat([features[col], future_return], axis=1, join="inner").dropna()
        if len(aligned) > 20:
            rows.append({
                "feature": col,
                "spearman_ic": aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"),
                "pearson_corr": aligned.iloc[:, 0].corr(aligned.iloc[:, 1]),
            })
    corr_df = pd.DataFrame(rows).sort_values("spearman_ic", ascending=False)
    st.dataframe(corr_df, use_container_width=True)

with tab2:
    feature_cols = list(features.columns)
    model, result, metrics, err = train_model(df, feature_cols, model_name)

    if err:
        st.warning(err)
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", model_name)
        m2.metric("RMSE", f"{metrics['rmse']:.5f}")
        m3.metric("Prediction corr", f"{metrics['prediction_corr']:.3f}")
        m4.metric("Test rows", metrics["n_test"])

        st.subheader("Prediction vs realized future return")
        plot_df = result[["future_return", "prediction"]].copy()
        st.line_chart(plot_df)

        st.subheader("Feature importance")
        imp = feature_importance(model, feature_cols)
        st.bar_chart(imp)

with tab3:
    if "result" in locals() and result is not None:
        bt = long_short_backtest(result)
        if bt.empty:
            st.warning("Backtest table is empty.")
        else:
            st.subheader("Toy sign-based equity curve")
            st.line_chart(bt["equity"])
            st.dataframe(bt.tail(20), use_container_width=True)
    else:
        st.warning("Run the ML tab first.")

with tab4:
    st.subheader("Downloaded prices")
    st.dataframe(prices.tail(30), use_container_width=True)
    st.subheader("Created features")
    st.dataframe(features.tail(30), use_container_width=True)

st.divider()
st.caption(
    "V2 is designed for interactive research. For real institutional use, replace free Yahoo data "
    "with audited market data and add strict timestamp validation, transaction-cost modeling, and survivorship-bias controls."
)
