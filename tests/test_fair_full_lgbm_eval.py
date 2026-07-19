"""Tests for fair full-LGBM A0 vs B2 evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fair_full_lgbm_eval_lib import (
    FAIR_CONFIGS,
    GroupBlendModelCF,
    SPECS,
    build_fold_bundle,
    document_lgbm_config,
    summarize_config,
)
from src.config import GROUP_CAPACITY_KWH, OUTPUT_DIR


def test_fair_config_names():
    assert FAIR_CONFIGS == ["A0_FULL_RAW", "A0_FULL_CAL", "B2_FULL_RAW", "B2_FULL_CAL"]


def test_only_target_differs_between_a0_and_b2():
    assert SPECS["A0_FULL_RAW"].use_capacity_factor is False
    assert SPECS["B2_FULL_RAW"].use_capacity_factor is True
    assert SPECS["A0_FULL_CAL"].apply_calibration is True
    assert SPECS["B2_FULL_CAL"].apply_calibration is True


def test_lgbm_full_tree_budget_documented():
    doc = document_lgbm_config()
    assert doc["mean_regressor"]["n_estimators"] == 900
    assert doc["quantile_regressor"]["n_estimators"] == 700
    assert doc["mean_regressor"]["early_stopping"] is False
    assert doc["quantile_regressor"]["learning_rate"] == pytest.approx(0.035)
    assert doc["mean_regressor"]["random_state"] == 42


def test_capacity_factor_roundtrip_full_trees():
    cap = GROUP_CAPACITY_KWH["kpx_group_1"]
    m = GroupBlendModelCF(q_weight=0.6)
    X = pd.DataFrame({"x": [1.0, 2.0]})
    y_cf = pd.Series([0.5, 0.8])
    m.fit(X, y_cf, cap)
    pred = m.predict(X)
    assert np.all(pred >= 0)
    assert np.all(pred <= cap)


def test_feature_parity_a0_b2():
    doc = document_lgbm_config()
    assert doc["features"]["count"] > 40
    assert "lead_hours" not in doc["features"]["columns"]


def test_summarize_config_shape():
    rows = []
    for cfg in FAIR_CONFIGS:
        for fid in range(1, 4):
            rows.append({"config": cfg, "fold_id": fid, "score": 0.6, "1_minus_nmae": 0.85, "ficr": 0.34})
    cmp = summarize_config(rows)
    assert len(cmp) == 4


def test_config_json_exists_after_run():
    path = OUTPUT_DIR / "fair_full_lgbm_config.json"
    if not path.exists():
        pytest.skip("run run_fair_full_lgbm_eval.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["mean_regressor"]["n_estimators"] == 900


def test_fold_results_have_four_configs():
    path = OUTPUT_DIR / "fair_full_lgbm_fold_results.csv"
    if not path.exists():
        pytest.skip("run run_fair_full_lgbm_eval.py first")
    df = pd.read_csv(path)
    assert set(df["config"].unique()) == set(FAIR_CONFIGS)
    assert df["fold_id"].nunique() == 7


def test_no_nan_in_results_if_present():
    path = OUTPUT_DIR / "fair_full_lgbm_fold_results.csv"
    if not path.exists():
        pytest.skip("run run_fair_full_lgbm_eval.py first")
    df = pd.read_csv(path)
    assert df[["score", "1_minus_nmae", "ficr"]].notna().all().all()
    assert np.isfinite(df[["score", "1_minus_nmae", "ficr"]].to_numpy()).all()


@pytest.mark.slow
def test_build_fold_bundle_keys():
    from src.data_loader import load_labels
    from scripts.validation_audit_lib import build_rolling_folds

    labels = load_labels()
    bundle = build_fold_bundle(labels, build_rolling_folds()[0])
    assert "feature_cols" in bundle
    assert len(bundle["fit_near"]) > 0
