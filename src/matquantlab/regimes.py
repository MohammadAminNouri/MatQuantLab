from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def detect_regimes(features: pd.DataFrame, n_regimes: int = 3, random_state: int = 7) -> pd.DataFrame:
    """Detect market regimes from global features using KMeans."""
    global_cols = [c for c in features.columns if not c.startswith("asset_")]
    X = features[global_cols].copy()
    X = X.dropna(how="all")
    if X.empty or len(X) < 50:
        return pd.DataFrame()
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_regimes, n_init=20, random_state=random_state)),
    ])
    labels = pipe.fit_predict(X)
    out = pd.DataFrame({"date": X.index, "regime": labels}).set_index("date")
    if "critical_materials_stress_index" in features.columns:
        cmsi = features["critical_materials_stress_index"].reindex(out.index)
        means = cmsi.groupby(out["regime"]).mean().sort_values()
        mapping = {old: rank for rank, old in enumerate(means.index)}
        out["regime_ordered_by_cmsi"] = out["regime"].map(mapping)
    return out


def regime_performance(preds: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    if preds.empty or regimes.empty:
        return pd.DataFrame()
    tmp = preds.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    r = regimes.reset_index().rename(columns={"index": "date"})
    r["date"] = pd.to_datetime(r["date"])
    tmp = pd.merge_asof(tmp.sort_values("date"), r.sort_values("date"), on="date")
    rows = []
    for reg, g in tmp.groupby("regime"):
        if len(g) < 20:
            continue
        corr = g[["y_pred", "target_fwd_return"]].corr(method="spearman").iloc[0, 1]
        rows.append({"regime": int(reg), "n": len(g), "spearman_ic": corr, "avg_target_return": g["target_fwd_return"].mean()})
    return pd.DataFrame(rows)
