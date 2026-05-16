from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def permutation_ic_test(y_true: pd.Series, y_pred: pd.Series, n_perm: int = 1000, seed: int = 7) -> dict:
    """Permutation test for Spearman IC.

    Tests whether the observed rank correlation is stronger than correlations
    obtained after shuffling predictions.
    """
    rng = np.random.default_rng(seed)
    joined = pd.concat([pd.Series(y_true), pd.Series(y_pred)], axis=1).dropna()
    joined.columns = ["y", "p"]
    if len(joined) < 20:
        return {"n": len(joined), "observed_ic": np.nan, "p_value": np.nan}
    obs = spearmanr(joined["y"], joined["p"]).correlation
    perm = []
    p = joined["p"].to_numpy().copy()
    y = joined["y"].to_numpy().copy()
    for _ in range(n_perm):
        rng.shuffle(p)
        perm.append(spearmanr(y, p).correlation)
    perm = np.array(perm, dtype=float)
    p_value = float((np.abs(perm) >= abs(obs)).mean())
    return {"n": len(joined), "observed_ic": float(obs), "p_value": p_value, "perm_mean": float(np.nanmean(perm)), "perm_std": float(np.nanstd(perm))}


def simple_overfit_warning(leaderboard: pd.DataFrame) -> str:
    if leaderboard.empty or "spearman_ic" not in leaderboard.columns:
        return "No model results available."
    best = leaderboard.iloc[0]
    ic = best.get("spearman_ic", np.nan)
    if pd.isna(ic):
        return "IC is missing; check data quality."
    if abs(ic) > 0.15:
        return "Warning: very high IC for daily/weekly financial data. Check leakage, survivorship bias, and overfitting."
    if abs(ic) < 0.02:
        return "Weak signal. This may be noise unless it is stable across assets/regimes and survives costs."
    return "Signal is non-trivial. Continue with robustness, transaction costs, and regime tests."
