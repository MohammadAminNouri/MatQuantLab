from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    returns = pd.Series(returns).dropna()
    if returns.std() == 0 or len(returns) < 3:
        return float("nan")
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def max_drawdown(equity: pd.Series) -> float:
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def backtest_from_predictions(
    preds: pd.DataFrame,
    cost_bps: float = 10.0,
    top_quantile: float = 0.25,
    bottom_quantile: float = 0.25,
) -> tuple[pd.DataFrame, dict]:
    """Simple long-short period backtest using model predictions.

    Each prediction date forms a long/short portfolio. The realized period return
    is computed from target_fwd_return. This is not an execution engine; it is an
    economic-value test for a signal.
    """
    if preds.empty:
        return pd.DataFrame(), {}

    clean_preds = preds.dropna(subset=["y_pred", "target_fwd_return"]).copy()
    clean_preds = (
        clean_preds.groupby(["date", "asset"], as_index=False)[["y_pred", "target_fwd_return"]]
        .mean()
        .sort_values(["date", "asset"])
    )

    frames = []
    last_weights = None
    for date, g in clean_preds.groupby("date"):
        g = g.copy().sort_values("y_pred")
        if len(g) < 4:
            continue
        n_long = max(1, int(np.ceil(len(g) * top_quantile)))
        n_short = max(1, int(np.ceil(len(g) * bottom_quantile)))
        shorts = g.head(n_short)
        longs = g.tail(n_long)
        weights = pd.Series(0.0, index=g["asset"])
        weights.loc[longs["asset"].values] = 1.0 / n_long
        weights.loc[shorts["asset"].values] = -1.0 / n_short

        gross = float((weights.reindex(g["asset"].values).values * g["target_fwd_return"].values).sum())
        if last_weights is None:
            turnover = float(weights.abs().sum())
        else:
            union = weights.index.union(last_weights.index)
            turnover = float((weights.reindex(union).fillna(0) - last_weights.reindex(union).fillna(0)).abs().sum())
        cost = turnover * cost_bps / 10000.0
        net = gross - cost
        frames.append({"date": pd.Timestamp(date), "gross_return": gross, "turnover": turnover, "cost": cost, "net_return": net})
        last_weights = weights

    bt = pd.DataFrame(frames).sort_values("date")
    if bt.empty:
        return bt, {}
    bt["gross_equity"] = (1.0 + bt["gross_return"]).cumprod()
    bt["net_equity"] = (1.0 + bt["net_return"]).cumprod()
    periods_per_year = max(1, int(round(252 / 20)))
    summary = {
        "n_periods": int(len(bt)),
        "gross_total_return": float(bt["gross_equity"].iloc[-1] - 1.0),
        "net_total_return": float(bt["net_equity"].iloc[-1] - 1.0),
        "gross_sharpe": sharpe_ratio(bt["gross_return"], periods_per_year),
        "net_sharpe": sharpe_ratio(bt["net_return"], periods_per_year),
        "max_drawdown_net": max_drawdown(bt["net_equity"]),
        "avg_turnover": float(bt["turnover"].mean()),
        "avg_cost": float(bt["cost"].mean()),
    }
    return bt, summary
