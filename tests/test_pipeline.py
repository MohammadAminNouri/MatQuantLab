from __future__ import annotations

import numpy as np
import pandas as pd

from matquantlab.features import rolling_zscore, make_global_features, make_asset_panel
from matquantlab.models import model_leaderboard


def test_rolling_zscore_basic():
    s = pd.Series(np.arange(100, dtype=float))
    z = rolling_zscore(s, window=20, min_periods=10)
    assert len(z) == 100
    assert z.notna().sum() > 0


def test_feature_and_panel_shapes():
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame({
        "XME": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
        "ITA": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
        "HG=F": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
        "CL=F": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
    }, index=dates)
    features = make_global_features(prices, ["HG=F", "CL=F"])
    panel = make_asset_panel(prices, features, ["XME", "ITA"], horizon=20)
    assert "critical_materials_stress_index" in features.columns
    assert not panel.empty
    assert {"date", "asset", "target_fwd_return"}.issubset(panel.columns)


def test_model_leaderboard_runs_on_demo_panel():
    dates = pd.bdate_range("2015-01-01", periods=900)
    rng = np.random.default_rng(3)
    shock = rng.normal(0, 1, len(dates))
    xme = 100 * np.exp(np.cumsum(0.0002 + 0.001 * np.roll(shock, 5) + rng.normal(0, 0.01, len(dates))))
    ita = 100 * np.exp(np.cumsum(0.0002 - 0.0005 * np.roll(shock, 5) + rng.normal(0, 0.01, len(dates))))
    hg = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    prices = pd.DataFrame({"XME": xme, "ITA": ita, "HG=F": hg}, index=dates)
    features = make_global_features(prices, ["HG=F"])
    panel = make_asset_panel(prices, features, ["XME", "ITA"], horizon=20)
    lb, preds = model_leaderboard(panel, model_names=["ridge"])
    assert not lb.empty
    assert not preds.empty
