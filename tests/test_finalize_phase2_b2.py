"""Tests for Phase 2 B2 final decision."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.finalize_phase2_b2 import (
    B2_RAW_SPEC,
    assess_b7_need,
    attach_oof,
    detect_missing_folds,
    evaluate_b2_submission_criteria,
    load_oof_metrics,
    summarize_vs_a0,
    validate_submission,
)
from scripts.phase2_horizon_experiments_lib import GroupBlendModelCF
from src.config import GROUP_CAPACITY_KWH, OUTPUT_DIR
from src.data_loader import load_submission_template


@pytest.fixture
def fold_df() -> pd.DataFrame:
    path = OUTPUT_DIR / "phase2_horizon_fold_results.csv"
    if not path.exists():
        pytest.skip("phase2 fold results missing")
    return pd.read_csv(path)


@pytest.fixture
def oof_df() -> pd.DataFrame:
    path = OUTPUT_DIR / "phase2_horizon_oof_metrics.csv"
    if not path.exists():
        pytest.skip("phase2 oof metrics missing")
    return load_oof_metrics(path)


def test_b2_submission_criteria_on_existing_results(fold_df, oof_df):
    b2 = attach_oof(summarize_vs_a0(fold_df, "B2_capacity_factor"), oof_df)
    result = evaluate_b2_submission_criteria(b2)
    assert "all_pass" in result
    assert result["checks"]["mean_delta_score_gt_0001"] is True
    assert result["checks"]["score_folds_gte_5_of_7"] is False
    assert result["checks"]["mean_delta_nmae_gt_0"] is False
    assert result["all_pass"] is False


def test_missing_fold_detection(fold_df):
    missing_b7 = detect_missing_folds(fold_df, "B7_full_stack")
    assert missing_b7 == list(range(1, 8))


def test_b7_should_not_run():
    d = assess_b7_need()
    assert d.should_run is False
    assert "horizon_split" in d.reason or "B3" in d.reason


def test_a0_b1_b2_summary_table(fold_df, oof_df):
    rows = []
    for exp in ["A0_v13_baseline", "B1_lead_features", "B2_capacity_factor"]:
        rows.append(attach_oof(summarize_vs_a0(fold_df, exp), oof_df))
    b2 = next(r for r in rows if r["experiment"] == "B2_capacity_factor")
    assert b2["positive_score_folds"] == 4
    assert b2["oof_score"] > b2["a0_oof_score"]


def test_capacity_factor_roundtrip():
    from scripts.train_v03 import GroupBlendModel

    cap = GROUP_CAPACITY_KWH["kpx_group_1"]
    m = GroupBlendModelCF(q_weight=0.6)
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y_kwh = pd.Series([cap * 0.5, cap * 0.8, cap * 0.2])
    m.fit(X, y_kwh, cap)
    pred = m.predict(X)
    assert np.all(pred >= 0)
    assert np.all(pred <= cap)


def test_group_capacity_constants():
    assert GROUP_CAPACITY_KWH["kpx_group_1"] == 21600
    assert GROUP_CAPACITY_KWH["kpx_group_3"] == 21000


def test_b2_raw_spec_has_no_calibration():
    assert B2_RAW_SPEC.use_v13_postprocess is False
    assert B2_RAW_SPEC.target_capacity_factor is True


def test_sample_submission_schema():
    t = load_submission_template()
    assert len(t) == 8760
    assert "forecast_id" in t.columns


def test_decision_json_exists_after_run():
    path = OUTPUT_DIR / "phase2_b2_final_decision.json"
    if not path.exists():
        pytest.skip("run finalize_phase2_b2.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["verdict"] in {
        "READY_B2_RAW",
        "READY_B2_V13_CALIBRATION",
        "RUN_B7_FIRST",
        "NOT_READY",
    }


def test_no_submission_when_not_ready():
    path = OUTPUT_DIR / "phase2_b2_final_decision.json"
    if not path.exists():
        pytest.skip("run finalize_phase2_b2.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["verdict"] != "NOT_READY":
        pytest.skip("submission expected for ready verdict")
    assert data.get("submission_created") is None


def test_validate_submission_template_self():
    t = load_submission_template()
    tmp = OUTPUT_DIR / "_tmp_template_validation.csv"
    t.to_csv(tmp, index=False)
    try:
        v = validate_submission(tmp)
        assert v["all_pass"]
    finally:
        tmp.unlink(missing_ok=True)
