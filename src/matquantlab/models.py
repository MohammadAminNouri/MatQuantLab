from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def prepare_model_frame(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode asset names and return feature columns."""
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    asset_dummies = pd.get_dummies(df["asset"], prefix="asset", dtype=float)
    df = pd.concat([df.drop(columns=["asset"]), asset_dummies], axis=1)
    feature_cols = [c for c in df.columns if c not in {"date", "target_fwd_return"}]
    return df, feature_cols


def make_model(name: str, random_state: int = 7):
    name = name.lower()
    if name == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ])
    if name == "elastic_net":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.25, max_iter=10000, random_state=random_state)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=30, max_depth=4, min_samples_leaf=30, random_state=random_state, n_jobs=-1)),
        ])
    if name == "gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(n_estimators=60, learning_rate=0.04, max_depth=2, random_state=random_state)),
        ])
    raise ValueError(f"Unknown model: {name}")


def walk_forward_predictions(
    panel: pd.DataFrame,
    model_name: str = "ridge",
    initial_train_years: int = 4,
    test_months: int = 6,
    min_train_rows: int = 500,
) -> pd.DataFrame:
    """Chronological walk-forward predictions.

    No random shuffling. Each test period only sees past data.
    """
    df, feature_cols = prepare_model_frame(panel)
    df = df.dropna(subset=["target_fwd_return"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    start_date = df["date"].min() + pd.DateOffset(years=initial_train_years)
    if start_date >= df["date"].max():
        # fallback for short demo datasets
        start_date = df["date"].quantile(0.60)

    preds = []
    test_start = pd.Timestamp(start_date)
    model_template = make_model(model_name)

    while test_start < df["date"].max():
        test_end = test_start + pd.DateOffset(months=test_months)
        train = df[df["date"] < test_start]
        test = df[(df["date"] >= test_start) & (df["date"] < test_end)]
        if len(train) >= min_train_rows and len(test) > 0:
            model = clone(model_template)
            X_train = train[feature_cols]
            y_train = train["target_fwd_return"]
            X_test = test[feature_cols]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            out = test[["date", "target_fwd_return"]].copy()
            # recover original asset from one-hot dummies
            dummy_cols = [c for c in out.columns if c.startswith("asset_")]
            out = test[["date", "target_fwd_return"]].copy()
            asset_cols = [c for c in test.columns if c.startswith("asset_")]
            out["asset"] = test[asset_cols].idxmax(axis=1).str.replace("asset_", "", regex=False) if asset_cols else "UNKNOWN"
            out["y_pred"] = y_pred
            out["model"] = model_name
            preds.append(out)
        test_start = test_end

    if not preds:
        return pd.DataFrame()
    return pd.concat(preds, ignore_index=True)


def evaluate_predictions(preds: pd.DataFrame) -> Dict[str, float]:
    if preds.empty:
        return {"n": 0}
    y = preds["target_fwd_return"].astype(float)
    p = preds["y_pred"].astype(float)
    ok = y.notna() & p.notna()
    y = y[ok]
    p = p[ok]
    if len(y) < 3:
        return {"n": int(len(y))}
    ic = spearmanr(p, y).correlation
    directional_acc = float((np.sign(p) == np.sign(y)).mean())
    rmse = float(mean_squared_error(y, p) ** 0.5)
    mae = float(mean_absolute_error(y, p))
    return {
        "n": int(len(y)),
        "spearman_ic": float(ic) if pd.notna(ic) else np.nan,
        "directional_accuracy": directional_acc,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2_score(y, p)),
        "pred_std": float(np.std(p)),
    }


def model_leaderboard(panel: pd.DataFrame, model_names: Iterable[str] = ("ridge", "elastic_net", "random_forest")) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_preds = []
    rows = []
    for name in model_names:
        preds = walk_forward_predictions(panel, model_name=name)
        if preds.empty:
            continue
        metrics = evaluate_predictions(preds)
        metrics["model"] = name
        rows.append(metrics)
        all_preds.append(preds)
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty and "spearman_ic" in leaderboard.columns:
        leaderboard = leaderboard.sort_values("spearman_ic", ascending=False)
    pred_frame = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return leaderboard, pred_frame


def permutation_feature_importance_simple(panel: pd.DataFrame, preds: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Approximate importance by absolute Spearman correlation with target.

    This is intentionally simple and stable for a beginner repo.
    """
    from scipy.stats import spearmanr

    df, feature_cols = prepare_model_frame(panel)
    rows = []
    for c in feature_cols:
        if c.startswith("asset_"):
            continue
        joined = df[[c, "target_fwd_return"]].dropna()
        if len(joined) < 100:
            continue
        corr = spearmanr(joined[c], joined["target_fwd_return"]).correlation
        if pd.notna(corr):
            rows.append({"feature": c, "abs_ic": abs(corr), "signed_ic": corr})
    out = pd.DataFrame(rows).sort_values("abs_ic", ascending=False).head(top_n)
    return out
