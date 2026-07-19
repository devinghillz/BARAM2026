"""Tests for FICR boundary analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import FICR_TIER_2, GENERATION_FLOOR_RATIO
from scripts.ficr_boundary_lib import (
    MIN_SAMPLES,
    compute_row_ficr_fields,
    count_ficr_transitions,
    ficr_row_success,
    filter_candidate_slices,
    row_unit_price as lib_row_unit_price,
)


def test_row_unit_price_matches_metrics():
    assert lib_row_unit_price(0.05) == 4.0
    assert lib_row_unit_price(0.07) == 3.0
    assert lib_row_unit_price(0.09) == 0.0


def test_inside_ficr_classification():
    cap = 21600.0
    actual = 10000.0
    pred = 10400.0  # 400/21600 = 1.85% error
    r = compute_row_ficr_fields(actual, pred, cap)
    assert r["ficr_class"] == "inside_ficr"
    assert r["ficr_success"] is True


def test_under_near_boundary_classification():
    cap = 21600.0
    actual = 10000.0
    # error 9% under: pred = 10000 - 0.09*21600 = 8056
    pred = 8056.0
    r = compute_row_ficr_fields(actual, pred, cap)
    assert r["ficr_class"] == "under_near_boundary"
    assert r["is_under"] is True
    assert r["distance_to_nearest_ficr_boundary"] == pytest.approx(0.01, abs=1e-4)
    assert r["boundary_band"] == "0_1pct"


def test_under_far_classification():
    cap = 21600.0
    actual = 10000.0
    pred = 5000.0
    r = compute_row_ficr_fields(actual, pred, cap)
    assert r["ficr_class"] == "under_far"
    assert r["boundary_band"] == "5pct_plus"


def test_over_near_boundary_classification():
    cap = 21600.0
    actual = 10000.0
    pred = 10000 + 0.09 * cap
    r = compute_row_ficr_fields(actual, pred, cap)
    assert r["ficr_class"] == "over_near_boundary"
    assert r["is_over"] is True


def test_inactive_excluded():
    cap = 21600.0
    actual = 1000.0  # below 10% floor
    r = compute_row_ficr_fields(actual, 1000.0, cap)
    assert r["ficr_class"] == "inactive_or_excluded"
    assert r["active"] is False


def test_min_multiplier_under_near_boundary():
    cap = 21600.0
    actual = 10000.0
    pred = 8056.0
    r = compute_row_ficr_fields(actual, pred, cap)
    target = actual - FICR_TIER_2 * cap
    assert r["min_multiplier_to_8pct"] == pytest.approx(target / pred, rel=1e-6)


def test_boundary_distance_calculation():
    cap = 10000.0
    actual = 5000.0
    pred = 4100.0  # error 900/10000 = 9%
    r = compute_row_ficr_fields(actual, pred, cap)
    assert r["error_rate"] == pytest.approx(0.09)
    assert r["distance_to_nearest_ficr_boundary"] == pytest.approx(0.01)


def test_ficr_success_matches_row_unit_price():
    cap = 21600.0
    actual = 10000.0
    for er, expected in [(0.05, True), (0.07, True), (0.09, False)]:
        pred = actual - er * cap
        r = compute_row_ficr_fields(actual, pred, cap)
        assert r["ficr_success"] == expected
        assert (r["unit_price"] > 0) == expected


def test_count_ficr_transitions():
    df = pd.DataFrame({
        "actual": [10000.0, 10000.0],
        "prediction": [8056.0, 10000.0],
        "capacity": [21600.0, 21600.0],
        "ficr_success": [False, True],
    })
    new_pred = pd.Series([8272.0, 10000.0])  # first at 8% boundary exactly
    t = count_ficr_transitions(df, new_pred)
    assert t["fail_to_success"] >= 1
    assert t["net_ficr_row_gain"] >= 1


def test_simulation_does_not_mutate_original():
    df = pd.DataFrame({
        "forecast_kst_dtm": pd.to_datetime(["2024-01-01 01:00:00"]),
        "group_id": [2],
        "month": [1],
        "hour": [1],
        "actual": [10000.0],
        "prediction": [9000.0],
        "capacity": [21600.0],
        "active": [True],
        "ficr_success": [False],
        "group_col": ["kpx_group_2"],
    })
    orig = df["prediction"].iloc[0]
    _ = df.copy()
    df_copy = df.copy()
    df_copy.loc[0, "prediction"] = orig * 1.02
    assert df["prediction"].iloc[0] == orig


def test_filter_candidate_slices_min_sample():
    stats = {
        "2024_full": [
            {
                "group_id": 2, "month": 2, "hour": 19,
                "row_count": 30, "under_near_count": 5, "over_near_count": 0,
                "far_error_count": 1, "median_min_multiplier": 1.02,
                "is_pipeline_hotspot": True,
            }
        ],
    }
    out = filter_candidate_slices(stats, min_n=MIN_SAMPLES["strict"])
    assert len(out) == 0
    out2 = filter_candidate_slices(stats, min_n=MIN_SAMPLES["exploratory"])
    assert len(out2) == 1


def test_key_alignment_no_duplicates():
    from src.data_loader import load_labels
    from scripts.eval_v13_comprehensive import build_split_bundle
    from scripts.ficr_boundary_lib import build_aligned_long_frame, validate_period_keys
    from scripts.submission_diff_lib import V13_BEST_CONFIG
    from scripts.v13_factorial_lib import PERIOD_DEFS

    labels = load_labels()
    bundle = build_split_bundle(labels, PERIOD_DEFS["2024_full"]["start"])
    df = build_aligned_long_frame(
        bundle, V13_BEST_CONFIG,
        PERIOD_DEFS["2024_full"]["start"], None, "2024_full",
    )
    assert df["key"].duplicated().sum() == 0
    rep = validate_period_keys({"2024_full": df})
    assert rep["2024_full"]["duplicate_keys"] == 0


def test_period_split_no_overlap():
    from scripts.v13_factorial_lib import PERIOD_DEFS

    h1_end = PERIOD_DEFS["2024_h1"]["end"]
    h2_start = PERIOD_DEFS["2024_h2"]["start"]
    assert h1_end == h2_start
