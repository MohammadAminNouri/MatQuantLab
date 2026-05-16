from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_output_dirs() -> None:
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)


def save_cmsi_plot(features: pd.DataFrame, path: str = "outputs/figures/cmsi.png") -> None:
    ensure_output_dirs()
    if "critical_materials_stress_index" not in features.columns:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    features["critical_materials_stress_index"].dropna().plot(ax=ax)
    ax.axhline(0, linewidth=1)
    ax.set_title("Critical Materials Stress Index")
    ax.set_ylabel("Rolling z-score")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_signal_decay_heatmap(decay: pd.DataFrame, path: str = "outputs/figures/signal_decay_heatmap.png") -> None:
    ensure_output_dirs()
    if decay.empty:
        return
    pivot = decay.pivot_table(index="feature", columns="horizon", values="spearman_ic", aggfunc="mean")
    pivot = pivot.reindex(pivot.abs().mean(axis=1).sort_values(ascending=False).head(20).index)
    fig, ax = plt.subplots(figsize=(9, max(5, len(pivot) * 0.28)))
    im = ax.imshow(pivot.fillna(0).values, aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}D" for c in pivot.columns])
    ax.set_title("Average Signal Decay: Feature IC by Horizon")
    fig.colorbar(im, ax=ax, label="Spearman IC")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_model_leaderboard_plot(leaderboard: pd.DataFrame, path: str = "outputs/figures/model_leaderboard.png") -> None:
    ensure_output_dirs()
    if leaderboard.empty or "spearman_ic" not in leaderboard.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = leaderboard["model"].astype(str)
    y = leaderboard["spearman_ic"].astype(float)
    ax.bar(x, y)
    ax.axhline(0, linewidth=1)
    ax.set_title("Walk-Forward ML Leaderboard")
    ax.set_ylabel("Spearman IC")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_feature_importance_plot(importance: pd.DataFrame, path: str = "outputs/figures/feature_importance.png") -> None:
    ensure_output_dirs()
    if importance.empty:
        return
    imp = importance.sort_values("abs_ic", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, max(5, len(imp) * 0.28)))
    ax.barh(imp["feature"], imp["abs_ic"])
    ax.set_title("Feature Importance Proxy: Absolute IC")
    ax.set_xlabel("Absolute Spearman IC")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_backtest_plot(bt: pd.DataFrame, path: str = "outputs/figures/backtest_equity_curve.png") -> None:
    ensure_output_dirs()
    if bt.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    bt.set_index("date")[["gross_equity", "net_equity"]].plot(ax=ax)
    ax.set_title("Long-Short Backtest: Gross vs Net Equity")
    ax.set_ylabel("Equity, start = 1")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
