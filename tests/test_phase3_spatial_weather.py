"""Tests for Phase 3 spatial / weather features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.spatial_feature_lib import (
    CONFIGS,
    PHASE3_CONFIGS,
    add_disagreement_features,
    add_physical_features,
    aggregate_scalar_for_timestamp,
    aggregate_wd_for_timestamp,
    aggregate_ws_for_timestamp,
    angular_difference_deg,
    audit_spatial_aggregation,
    build_grid_context,
    check_feature_leakage,
    circular_mean_components,
    fit_wind_bin_edges,
    get_phase3_feature_columns,
    grade_candidate,
    idw_weights,
)
from src.data_loader import load_weather


def test_idw_weights_sum_to_one():
    dist = np.array([0.0, 1.0, 2.0, 5.0])
    w = idw_weights(dist, k=3)
    assert pytest.approx(w.sum()) == 1.0
    assert w[3] == 0.0


def test_idw_zero_distance():
    dist = np.array([0.0, 2.0, 3.0])
    w = idw_weights(dist, k=None)
    assert np.isfinite(w).all()
    assert pytest.approx(w.sum()) == 1.0


def test_circular_mean_north():
    sin_m, cos_m, rvl, var = circular_mean_components(np.array([0.0, 0.0]))
    assert cos_m > 0.9
    assert rvl > 0.9


def test_angular_difference():
    assert angular_difference_deg(10, 350) == pytest.approx(20.0)
    assert angular_difference_deg(0, 180) == pytest.approx(180.0)


def test_spatial_std_range():
    from src.features import _add_wind_features, LDAPS_WIND_COLS
    weather = load_weather("ldaps", "train")
    weather = _add_wind_features(weather, LDAPS_WIND_COLS, "ldaps")
    ctx = build_grid_context(weather, "ldaps")
    row = weather.iloc[0:16]
    stats = aggregate_ws_for_timestamp(row, ctx, 1, "ldaps_ws10", full_multi=True)
    assert "spatial_std" in stats
    assert "spatial_range" in stats
    assert stats["spatial_range"] >= 0


def test_ldaps_gfs_timestamp_alignment():
    audit = audit_spatial_aggregation()
    assert audit["timestamp_alignment_ldaps_gfs"] == 1


def test_disagreement_features():
    df = pd.DataFrame({
        "ldaps_ws10": [5.0, 6.0],
        "gfs_ws10": [4.0, 7.0],
        "sp_ldaps_ws10_nearest": [5.0, 6.0],
        "sp_gfs_ws10_nearest": [4.0, 7.0],
        "sp_ldaps_ws10_idw5": [5.1, 6.1],
        "sp_gfs_ws10_idw5": [4.1, 7.1],
        "sp_ldaps_ws10_spatial_mean": [5.2, 6.2],
        "sp_gfs_ws10_spatial_mean": [4.2, 7.2],
        "hour": [1, 2],
        "month": [1, 2],
        "group_id": [1, 2],
        "scada_prior_kwh": [1000, 2000],
        "ws10_diff_ldaps_gfs": [1.0, -1.0],
    })
    out = add_disagreement_features(df)
    assert "dagr_ws10_nearest_diff" in out.columns
    assert "dagr_ws10_nearest_abs" in out.columns


def test_air_density_units():
    df = pd.DataFrame({
        "blend_ws10": [8.0],
        "ldaps_temp2m": [280.0],  # Kelvin
        "ldaps_sp": [90000.0],  # Pa
    })
    out = add_physical_features(df, np.array([0, 4, 8, 12, 20]))
    rho = out["phy_air_density"].iloc[0]
    assert 0.5 < rho < 2.0  # kg/m3 scale


def test_no_target_leakage_in_spatial_cols():
    df = pd.DataFrame({"power_kwh": [100], "sp_ldaps_ws10_nearest": [5.0]})
    r = check_feature_leakage(df, pd.Timestamp("2024-01-01"))
    assert r["passed"]


def test_phase3_feature_s0_count():
    from src.data_loader import load_labels
    from src.power_curve import build_scada_monthly_curve
    from scripts.spatial_feature_lib import build_phase3_dataset

    labels = load_labels()
    scada = build_scada_monthly_curve()
    df = build_phase3_dataset(labels, "idw", pd.Timestamp("2024-01-01"), scada, "train", CONFIGS["S0"])
    cols = get_phase3_feature_columns(df, CONFIGS["S0"])
    assert len(cols) == 51


def test_s1_has_more_features_than_s0():
    from src.data_loader import load_labels
    from src.power_curve import build_scada_monthly_curve
    from scripts.spatial_feature_lib import build_phase3_dataset

    labels = load_labels()
    scada = build_scada_monthly_curve()
    df = build_phase3_dataset(labels, "idw", pd.Timestamp("2024-01-01"), scada, "train", CONFIGS["S1"])
    assert len(get_phase3_feature_columns(df, CONFIGS["S1"])) > 51


def test_grade_candidate_strong():
    g = grade_candidate({
        "mean_delta_score": 0.002,
        "positive_score_folds": 6,
        "mean_delta_nmae": 0.002,
        "worst_fold_delta_score": -0.0005,
        "oof_delta_score": 0.001,
        "oof_delta_nmae": 0.001,
        "mean_delta_ficr": 0.0,
        "leakage_passed": True,
        "max_fold_contribution": 0.3,
    })
    assert g == "strong_candidate"


def test_grade_candidate_reject():
    g = grade_candidate({"mean_delta_score": -0.001, "positive_score_folds": 2})
    assert g == "reject"


def test_checkpoint_path_format():
    from scripts.eval_phase3_spatial_weather import _pred_path
    p = _pred_path("S1", 3)
    assert "fold3_S1" in str(p)


def test_oof_key_uniqueness():
    from scripts.validation_audit_lib import build_rolling_folds
    folds = build_rolling_folds()
    assert len(folds) == 7
    starts = [f.valid_start for f in folds]
    assert len(starts) == len(set(starts))


def test_wd_circular_aggregation():
    weather = load_weather("gfs", "train")
    from src.features import _add_wind_features, GFS_WIND_COLS
    weather = _add_wind_features(weather, GFS_WIND_COLS, "gfs")
    ctx = build_grid_context(weather, "gfs")
    g = weather.iloc[:9]
    stats = aggregate_wd_for_timestamp(g, ctx, 1, GFS_WIND_COLS["u10"], GFS_WIND_COLS["v10"])
    assert "idw_sin" in stats
    assert "nearest_idw_angdiff" in stats
