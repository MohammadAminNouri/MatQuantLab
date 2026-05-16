from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from matquantlab.backtest import backtest_from_predictions
from matquantlab.data_sources import get_asset_tickers, get_market_tickers, load_frame, load_universe
from matquantlab.features import make_asset_panel, make_signal_decay_table
from matquantlab.models import model_leaderboard, permutation_feature_importance_simple
from matquantlab.overfitting import simple_overfit_warning, permutation_ic_test
from matquantlab.regimes import detect_regimes, regime_performance
from matquantlab.visualization import (
    ensure_output_dirs,
    save_backtest_plot,
    save_cmsi_plot,
    save_feature_importance_plot,
    save_model_leaderboard_plot,
    save_signal_decay_heatmap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MatQuantLab ML report and figures.")
    parser.add_argument("--config", default="config/universe.yaml")
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args()

    ensure_output_dirs()
    universe = load_universe(args.config)
    prices = load_frame("data/raw/prices_yfinance.parquet")
    features = load_frame("data/processed/features.parquet")

    assets_all = [a for a in get_asset_tickers(universe) if a in prices.columns]
    priority_assets = ["SPY", "XLB", "XME", "COPX", "PICK", "XLI", "ITA", "XAR", "VIS", "PRNT", "DDD", "SSYS", "MTLS", "FCX", "AA"]
    assets = [a for a in priority_assets if a in assets_all] or assets_all[:15]
    horizons = universe.get("horizons", [5, 20, 60])

    feature_candidates = [
        c for c in features.columns
        if c.startswith("shock_z_20d") or c == "critical_materials_stress_index"
    ]
    decay = make_signal_decay_table(prices, features, assets, feature_candidates, horizons=horizons)
    decay.to_csv("outputs/tables/signal_decay.csv", index=False)

    panel = make_asset_panel(prices, features, assets, horizon=args.horizon, max_rows_per_asset=300)
    try:
        panel.to_parquet("data/processed/model_panel.parquet")
    except Exception:
        panel.to_pickle("data/processed/model_panel.parquet")

    leaderboard, preds = model_leaderboard(panel)
    leaderboard.to_csv("outputs/tables/model_leaderboard.csv", index=False)
    if not preds.empty:
        try:
            preds.to_parquet("data/processed/predictions.parquet")
        except Exception:
            preds.to_pickle("data/processed/predictions.parquet")

    best_model = leaderboard.iloc[0]["model"] if not leaderboard.empty else None
    best_preds = preds[preds["model"] == best_model].copy() if best_model is not None and not preds.empty else pd.DataFrame()
    bt, bt_summary = backtest_from_predictions(best_preds, cost_bps=float(universe.get("transaction_cost_bps", 10)))
    if not bt.empty:
        bt.to_csv("outputs/tables/backtest_period_returns.csv", index=False)
    pd.DataFrame([bt_summary]).to_csv("outputs/tables/backtest_summary.csv", index=False)

    importance = permutation_feature_importance_simple(panel, best_preds)
    importance.to_csv("outputs/tables/feature_importance_proxy.csv", index=False)

    regimes = detect_regimes(features)
    if not regimes.empty:
        regimes.to_csv("outputs/tables/regimes.csv")
        reg_perf = regime_performance(best_preds, regimes)
        reg_perf.to_csv("outputs/tables/regime_performance.csv", index=False)
    else:
        reg_perf = pd.DataFrame()

    if not best_preds.empty:
        perm = permutation_ic_test(best_preds["target_fwd_return"], best_preds["y_pred"], n_perm=500)
    else:
        perm = {}
    pd.DataFrame([perm]).to_csv("outputs/tables/permutation_test.csv", index=False)

    save_cmsi_plot(features)
    save_signal_decay_heatmap(decay)
    save_model_leaderboard_plot(leaderboard)
    save_feature_importance_plot(importance)
    save_backtest_plot(bt)

    warning = simple_overfit_warning(leaderboard)
    summary = f"""# MatQuantLab Research Summary

## Research question

Do critical-materials, commodity, energy, macro, and volatility shocks predict future returns of materials, mining, aerospace, defense, industrial, and additive-manufacturing-linked equities?

## Data

- Price columns downloaded: {len(prices.columns)}
- Available assets tested: {len(assets)}
- Date range: {prices.index.min().date()} to {prices.index.max().date()}
- ML forecast horizon: {args.horizon} trading days

## Best model

{best_model if best_model else "No model produced predictions"}

## Model leaderboard

{leaderboard.to_markdown(index=False) if not leaderboard.empty else "No leaderboard available."}

## Backtest summary

{pd.DataFrame([bt_summary]).to_markdown(index=False) if bt_summary else "No backtest available."}

## Permutation overfitting check

{pd.DataFrame([perm]).to_markdown(index=False) if perm else "No permutation test available."}

## Quant warning

{warning}

## Important limitation

This is research-only. Results from free data are hypotheses, not trading recommendations.
"""
    Path("outputs/research_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
