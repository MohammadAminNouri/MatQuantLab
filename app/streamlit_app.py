from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import spearmanr

from matquantlab.data_sources import get_asset_tickers, load_frame, load_universe
from matquantlab.features import make_forward_returns

st.set_page_config(page_title="MatQuantLab", layout="wide")

st.title("MatQuantLab — Critical Materials Alpha Explorer")
st.caption("Interactive research dashboard for commodities, macro stress, and industrial equity returns.")

prices_path = Path("data/raw/prices_yfinance.parquet")
features_path = Path("data/processed/features.parquet")

if not prices_path.exists() or not features_path.exists():
    st.warning("Data files not found. Run: python scripts/run_all.py")
    st.stop()

universe = load_universe("config/universe.yaml")
prices = load_frame(prices_path)
features = load_frame(features_path)
assets = [a for a in get_asset_tickers(universe) if a in prices.columns]

st.sidebar.header("Controls")
asset = st.sidebar.selectbox("Asset", assets, index=assets.index("XME") if "XME" in assets else 0)
feature = st.sidebar.selectbox("Feature", list(features.columns), index=list(features.columns).index("critical_materials_stress_index") if "critical_materials_stress_index" in features.columns else 0)
horizon = st.sidebar.selectbox("Forward return horizon", [5, 20, 60], index=1)

fwd = make_forward_returns(prices, [asset], [horizon])[horizon][asset]
df = pd.concat([features[feature], fwd], axis=1).dropna()
df.columns = [feature, f"future_{horizon}d_return"]

if df.empty:
    st.error("No overlapping data for this asset/feature/horizon.")
    st.stop()

ic = spearmanr(df[feature], df[f"future_{horizon}d_return"]).correlation

c1, c2, c3 = st.columns(3)
c1.metric("Asset", asset)
c2.metric("Horizon", f"{horizon}D")
c3.metric("Spearman IC", f"{ic:.4f}")

st.subheader("Feature through time")
fig1 = px.line(df.reset_index(), x=df.index.name or "index", y=feature, title=f"{feature}")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Feature vs future return")
plot_df = df.reset_index().rename(columns={df.index.name or "index": "date"})
fig2 = px.scatter(plot_df, x=feature, y=f"future_{horizon}d_return", trendline="ols", title=f"{asset}: {feature} vs future {horizon}D return")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Interpretation")
st.write(
    "A positive IC means higher feature values tend to be followed by higher future returns. "
    "A negative IC means higher feature values tend to be followed by lower future returns. "
    "In financial data, even small IC values can be interesting only if they are stable, explainable, and survive transaction costs."
)
