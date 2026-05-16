"""
MatQuantLab V3 — Critical Materials Intelligence Dashboard

A materials-informed, interactive ML research app for testing whether metals,
energy, USD, volatility, and industrial stress signals contain lead-lag
information for mining, aerospace, defense, industrials, and additive
manufacturing assets.

Run:
    streamlit run app/live_dashboard.py
"""

from __future__ import annotations

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment_config.yml"


def load_config() -> dict:
    import yaml
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def safe_period(interval: str, period: str) -> str:
    if interval == "1m":
        return period if period in {"1d", "5d"} else "5d"
    if interval in {"5m", "15m", "30m", "60m"}:
        return period if period in {"1d", "5d", "1mo"} else "1mo"
    return period


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
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        if len(tickers) == 1:
            prices.columns = tickers

    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="all")
    return prices


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std()
    return (series - mean) / std


def build_features(
    prices: pd.DataFrame,
    commodity_map: dict[str, str],
    weights: dict[str, float],
    shock_window: int,
) -> pd.DataFrame:
    features = pd.DataFrame(index=prices.index)

    for name, ticker in commodity_map.items():
        if ticker in prices.columns:
            ret = prices[ticker].pct_change(shock_window)
            features[f"{name}_shock"] = zscore(ret, max(20, shock_window * 4))

    shock_cols = [c for c in features.columns if c.endswith("_shock")]
    if not shock_cols:
        return pd.DataFrame()

    weighted = []
    total_w = 0.0
    for name, w in weights.items():
        col = f"{name}_shock"
        if col in features.columns:
            weighted.append(features[col] * float(w))
            total_w += float(w)

    if weighted and total_w > 0:
        features["CMSI_v2_weighted"] = sum(weighted) / total_w
    else:
        features["CMSI_v2_weighted"] = features[shock_cols].mean(axis=1)

    # Simple sub-indexes for storytelling.
    metal_cols = [c for c in ["Copper_shock", "Aluminum_shock", "Gold_shock"] if c in features]
    macro_cols = [c for c in ["USD_shock", "VIX_shock", "Oil_shock"] if c in features]
    if metal_cols:
        features["Metals_Stress"] = features[metal_cols].mean(axis=1)
    if macro_cols:
        features["Macro_Risk_Stress"] = features[macro_cols].mean(axis=1)

    return features.replace([np.inf, -np.inf], np.nan).dropna()


def classify_stress(value: float) -> tuple[str, str]:
    if value >= 2.0:
        return "Extreme stress", "Supply-chain / macro stress is unusually high."
    if value >= 1.0:
        return "Elevated stress", "Stress is above normal; watch industrial drawdown risk."
    if value <= -1.0:
        return "Relief / easing", "Stress indicators are below normal."
    return "Normal", "No major critical-materials stress signal."


def forward_return(prices: pd.DataFrame, asset: str, horizon: int) -> pd.Series:
    return prices[asset].pct_change(horizon).shift(-horizon).rename("future_return")


def correlations(features: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        aligned = pd.concat([features[col], y], axis=1, join="inner").dropna()
        if len(aligned) >= 20:
            rows.append({
                "feature": col,
                "spearman_ic": aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"),
                "pearson_corr": aligned.iloc[:, 0].corr(aligned.iloc[:, 1]),
                "rows": len(aligned),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("spearman_ic", ascending=False)


def train_model(df: pd.DataFrame, feature_cols: list[str], model_name: str):
    if len(df) < 80:
        return None, None, None, "Not enough rows for ML. Use longer period or daily interval."

    split = int(len(df) * 0.75)
    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[feature_cols].fillna(0.0)
    y_train = train["future_return"].astype(float)
    X_test = test[feature_cols].fillna(0.0)
    y_test = test["future_return"].astype(float)

    if model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
    else:
        model = Ridge(alpha=10.0)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    out = test.copy()
    out["prediction"] = pred

    rmse = float(mean_squared_error(y_test, pred) ** 0.5)
    corr = pd.Series(pred).corr(pd.Series(y_test.values))
    corr = float(corr) if not pd.isna(corr) else 0.0

    return model, out, {"rmse": rmse, "prediction_corr": corr, "train_rows": len(train), "test_rows": len(test)}, None


def feature_importance(model, feature_cols: list[str]) -> pd.Series:
    if model is None:
        return pd.Series(dtype=float)
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        vals = np.abs(getattr(model, "coef_", np.zeros(len(feature_cols))))
    return pd.Series(vals, index=feature_cols).sort_values(ascending=False)


def warning_logic(asset: str, features: pd.DataFrame, corr_df: pd.DataFrame) -> list[str]:
    warnings_out = []
    latest = features.iloc[-1]

    cmsi = latest.get("CMSI_v2_weighted", np.nan)
    metals = latest.get("Metals_Stress", np.nan)
    macro = latest.get("Macro_Risk_Stress", np.nan)

    if pd.notna(cmsi) and cmsi > 1.0:
        warnings_out.append("CMSI is elevated: industrial and materials-linked drawdown risk may be higher.")
    if pd.notna(cmsi) and cmsi < -1.0:
        warnings_out.append("CMSI is easing: critical-materials stress is below normal.")

    if asset in {"ITA", "XAR"} and pd.notna(macro) and macro > 1.0:
        warnings_out.append("Aerospace/defense watch: macro/energy stress is elevated, which may pressure supplier margins.")
    if asset in {"XME", "COPX", "PICK"} and pd.notna(metals) and abs(metals) > 1.0:
        warnings_out.append("Mining/metals watch: metals stress is large enough to deserve lead-lag testing.")
    if asset in {"PRNT", "DDD", "SSYS", "MTLS", "NNDM"} and pd.notna(cmsi) and abs(cmsi) > 1.0:
        warnings_out.append("Additive manufacturing watch: AM-linked equities may be sensitive to industrial risk appetite and input-cost stress.")

    if not corr_df.empty:
        top = corr_df.iloc[0]
        warnings_out.append(f"Top current lead-lag candidate: `{top['feature']}` with Spearman IC {top['spearman_ic']:.3f}.")

    if not warnings_out:
        warnings_out.append("No strong stress warning. Treat this as neutral research state.")

    return warnings_out


def sign_backtest(result: pd.DataFrame) -> pd.DataFrame:
    bt = result[["prediction", "future_return"]].dropna().copy()
    if bt.empty:
        return pd.DataFrame()
    bt["position"] = np.where(bt["prediction"] > 0, 1.0, -1.0)
    bt["strategy_return"] = bt["position"] * bt["future_return"]
    bt["equity"] = (1.0 + bt["strategy_return"].fillna(0.0)).cumprod()
    return bt


st.set_page_config(page_title="MatQuantLab V3", layout="wide")
st.title("MatQuantLab V3 — Critical Materials Intelligence Layer")
st.caption("Materials-informed ML dashboard for cross-asset stress, lead-lag signals, and industrial equity risk. Research only, not investment advice.")

config = load_config()
assets = config.get("assets", ["XME", "COPX", "ITA", "XAR", "PRNT", "SPY"])
commodity_map = config.get("commodity_features", {})
weights = config.get("critical_material_weights", {})
exposure_map = config.get("exposure_map", {})
am_basket = config.get("additive_manufacturing_basket", ["PRNT", "DDD", "SSYS", "MTLS", "NNDM"])
intervals = config.get("intervals", ["1m", "5m", "15m", "1d"])
periods = config.get("periods", ["1d", "5d", "1mo", "1y"])
models = config.get("models", ["Ridge", "RandomForest"])

with st.sidebar:
    st.header("Experiment Controls")
    asset = st.selectbox("Target asset", assets, index=assets.index(config.get("default_asset", assets[0])) if config.get("default_asset", assets[0]) in assets else 0)
    interval = st.selectbox("Interval", intervals, index=intervals.index(config.get("default_interval", intervals[0])) if config.get("default_interval", intervals[0]) in intervals else 0)
    period = st.selectbox("Period", periods, index=periods.index(config.get("default_period", periods[0])) if config.get("default_period", periods[0]) in periods else 0)
    model_name = st.selectbox("Model", models, index=models.index(config.get("default_model", models[0])) if config.get("default_model", models[0]) in models else 0)
    horizon = st.slider("Prediction horizon in bars", 1, 120, 12)
    shock_window = st.slider("Shock window in bars", 3, 80, 20)
    auto_refresh = st.toggle("Auto-refresh every 60 seconds", value=False)
    if st.button("Refresh latest data"):
        st.cache_data.clear()
        st.rerun()

if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()

period = safe_period(interval, period)
tickers = sorted(set([asset] + list(commodity_map.values()) + am_basket))
prices = download_prices(tickers, period, interval)

if prices.empty or asset not in prices.columns:
    st.error("Could not download enough data. Try another period, interval, or asset.")
    st.stop()

features = build_features(prices, commodity_map, weights, shock_window)
if features.empty:
    st.error("Could not build features. Try longer period or bigger interval.")
    st.stop()

y = forward_return(prices, asset, horizon)
df = pd.concat([features, y], axis=1, join="inner").dropna()
corr_df = correlations(features, y)

latest_price = prices[asset].dropna().iloc[-1]
latest_cmsi = features["CMSI_v2_weighted"].iloc[-1]
stress_label, stress_text = classify_stress(float(latest_cmsi))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Target asset", asset)
c2.metric("Latest price", f"{latest_price:.2f}")
c3.metric("CMSI v2", f"{latest_cmsi:.2f}", help="Weighted Critical Materials Stress Index.")
c4.metric("Stress regime", stress_label)

st.info(stress_text)

asset_info = exposure_map.get(asset, {"theme": "Custom asset", "logic": "No exposure logic configured."})
st.markdown(f"**Exposure logic — {asset_info.get('theme','')}:** {asset_info.get('logic','')}")

tabs = st.tabs([
    "Industrial Stress Radar",
    "Lead-Lag Lab",
    "ML Explainability",
    "AM Watchlist",
    "Backtest",
    "Raw Data",
])

with tabs[0]:
    st.subheader("Industrial Stress Radar")
    r1, r2 = st.columns([1.2, 1])

    with r1:
        radar_cols = [c for c in ["CMSI_v2_weighted", "Metals_Stress", "Macro_Risk_Stress"] if c in features.columns]
        st.line_chart(features[radar_cols])

    with r2:
        st.markdown("### Early-warning notes")
        for w in warning_logic(asset, features, corr_df):
            st.write("• " + w)

        latest = features.iloc[-1]
        driver_cols = [c for c in features.columns if c.endswith("_shock")]
        if driver_cols:
            drivers = latest[driver_cols].sort_values(key=lambda s: s.abs(), ascending=False)
            st.markdown("### Current strongest stress drivers")
            st.dataframe(drivers.rename("latest_z_score").to_frame(), use_container_width=True)

with tabs[1]:
    st.subheader("Lead-Lag Signal Lab")
    st.write("This table tests whether each stress feature has a relationship with future return at the selected horizon.")
    if corr_df.empty:
        st.warning("Not enough aligned rows.")
    else:
        st.dataframe(corr_df, use_container_width=True)
        st.bar_chart(corr_df.set_index("feature")["spearman_ic"])

with tabs[2]:
    st.subheader("ML Explainability")
    feature_cols = list(features.columns)
    model, result, metrics, err = train_model(df, feature_cols, model_name)

    if err:
        st.warning(err)
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", model_name)
        m2.metric("RMSE", f"{metrics['rmse']:.5f}")
        m3.metric("Prediction corr", f"{metrics['prediction_corr']:.3f}")
        m4.metric("Test rows", metrics["test_rows"])

        st.markdown("### Prediction vs realized future return")
        st.line_chart(result[["prediction", "future_return"]])

        imp = feature_importance(model, feature_cols)
        st.markdown("### Main signal drivers")
        st.bar_chart(imp)
        st.dataframe(imp.rename("importance").to_frame(), use_container_width=True)

        latest_features = features.iloc[[-1]][feature_cols].fillna(0.0)
        latest_prediction = float(model.predict(latest_features)[0])
        direction = "positive" if latest_prediction > 0 else "negative"
        st.markdown("### Interpretable signal summary")
        st.write(f"Latest model-implied {horizon}-bar pressure for **{asset}** is **{direction}** ({latest_prediction:.4f}).")
        st.write("Main drivers are the highest-importance features above. Treat this as a research signal, not a trade instruction.")

with tabs[3]:
    st.subheader("Additive Manufacturing Watchlist")
    available = [x for x in am_basket if x in prices.columns]
    if not available:
        st.warning("No AM basket tickers were downloaded.")
    else:
        am_prices = prices[available].dropna(how="all")
        am_returns = am_prices.pct_change().dropna()
        st.markdown("AM basket tickers:")
        st.write(", ".join(available))
        st.line_chart(am_prices / am_prices.iloc[0])
        st.markdown("### Recent AM basket return snapshot")
        st.dataframe(am_returns.tail(20), use_container_width=True)

        if "CMSI_v2_weighted" in features:
            rows = []
            for t in available:
                yy = prices[t].pct_change(horizon).shift(-horizon).rename("future_return")
                aligned = pd.concat([features["CMSI_v2_weighted"], yy], axis=1, join="inner").dropna()
                if len(aligned) > 20:
                    rows.append({"asset": t, "CMSI_spearman_ic": aligned.iloc[:,0].corr(aligned.iloc[:,1], method="spearman")})
            if rows:
                st.markdown("### CMSI relation to AM-linked future returns")
                st.dataframe(pd.DataFrame(rows).sort_values("CMSI_spearman_ic", ascending=False), use_container_width=True)

with tabs[4]:
    st.subheader("Toy Research Backtest")
    if "result" not in locals() or result is None:
        st.warning("Open ML Explainability tab first.")
    else:
        bt = sign_backtest(result)
        if bt.empty:
            st.warning("Backtest is empty.")
        else:
            st.line_chart(bt["equity"])
            st.dataframe(bt.tail(30), use_container_width=True)
            st.caption("Toy sign-based backtest: long when prediction > 0, short when prediction < 0. No transaction-cost or slippage model.")

with tabs[5]:
    st.subheader("Downloaded price data")
    st.dataframe(prices.tail(50), use_container_width=True)
    st.subheader("Feature data")
    st.dataframe(features.tail(50), use_container_width=True)

st.divider()
st.caption(
    "Data source: Yahoo Finance through yfinance. This app is a research/portfolio demonstration. "
    "It is not a trading system and does not provide financial advice."
)
