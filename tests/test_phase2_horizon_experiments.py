"""Tests for Phase 2 horizon / target representation experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.phase2_horizon_experiments_lib import (
    EXPERIMENTS,
    assign_horizon_bucket,
    augment_horizon_features,
    compare_to_baseline,
    get_experiment_feature_cols,
)
from scripts.validation_audit_lib import build_rolling_folds, check_fold_disjoint


def _mini_frame() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01 13:00:00", periods=6, freq="h")
    issue = ts - pd.Timedelta(hours=14)
    return pd.DataFrame(
        {
            "forecast_kst_dtm": ts,
            "data_available_kst_dtm": issue,
            "group_id": [1, 2, 3, 1, 2, 3],
            "power_kwh": [1000.0, 2000.0, 1500.0, 1100.0, 2100.0, 1600.0],
            "ldaps_ws10": np.linspace(5, 10, 6),
            "hour": ts.hour,
            "month": ts.month,
        }
    )


def test_horizon_bucket_boundaries():
    lh = np.array([12, 18, 19, 24, 25, 35], dtype=float)
    buckets = assign_horizon_bucket(lh)
    assert buckets.tolist() == [0, 0, 1, 1, 2, 2]


def test_augment_adds_lead_features():
    out = augment_horizon_features(_mini_frame())
    for col in ["lead_hours", "lead_hours_norm", "lead_sin", "lead_cos", "horizon_bucket", "hb_12_18"]:
        assert col in out.columns
    assert out["lead_hours"].iloc[0] == pytest.approx(14.0)


def test_lead_features_included_for_b1():
    spec = next(e for e in EXPERIMENTS if e.name == "B1_lead_features")
    df = augment_horizon_features(_mini_frame())
    cols = get_experiment_feature_cols(df, spec)
    assert "lead_hours" in cols
    assert "horizon_bucket" not in cols


def test_rolling_folds_reused_for_oof():
    folds = build_rolling_folds()
    assert len(folds) == 7
    assert check_fold_disjoint(folds)["all_disjoint"]


def test_compare_to_baseline_shape():
    rows = [
        {"experiment": "A0_v13_baseline", "fold_id": 1, "score": 0.60, "1_minus_nmae": 0.85, "ficr": 0.35},
        {"experiment": "A0_v13_baseline", "fold_id": 2, "score": 0.61, "1_minus_nmae": 0.86, "ficr": 0.36},
        {"experiment": "B1_lead_features", "fold_id": 1, "score": 0.605, "1_minus_nmae": 0.851, "ficr": 0.349},
        {"experiment": "B1_lead_features", "fold_id": 2, "score": 0.612, "1_minus_nmae": 0.861, "ficr": 0.358},
    ]
    cmp = compare_to_baseline(rows)
    assert len(cmp) == 1
    assert cmp.iloc[0]["experiment"] == "B1_lead_features"
    assert cmp.iloc[0]["positive_score_folds"] == 2


@pytest.mark.slow
def test_a0_fold_matches_phase1_baseline():
    from scripts.phase2_horizon_experiments_lib import evaluate_fold_experiment
    from scripts.validation_audit_lib import evaluate_fold
    from src.data_loader import load_labels

    labels = load_labels()
    fold = build_rolling_folds()[-1]
    a0 = evaluate_fold_experiment(labels, fold, next(e for e in EXPERIMENTS if e.is_baseline))
    ref = evaluate_fold(labels, fold, "v13_best")
    assert a0["score"] == pytest.approx(ref["score"], abs=1e-4)
