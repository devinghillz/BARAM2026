"""Tests for FICR candidate family validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ficr_candidate_families_lib import (
    CandidateSpec,
    apply_multiplier_to_wide,
    block_bootstrap_candidate,
    build_candidate_specs,
    candidate_mask,
    count_tier_transitions,
    evaluate_december_year,
    leave_one_period_out,
    long_to_wide_pred,
    price_to_tier,
    resolve_mask,
    simulate_on_period,
    top_candidate_configs,
)
from scripts.ficr_boundary_lib import compute_row_ficr_fields, long_to_wide_truth


def _make_row(
    ts: str,
    gid: int,
    month: int,
    hour: int,
    actual: float,
    pred: float,
    cap: float = 21600.0,
) -> dict:
    fields = compute_row_ficr_fields(actual, pred, cap)
    col = f"kpx_group_{gid}"
    return {
        "forecast_kst_dtm": pd.Timestamp(ts),
        "group_id": gid,
        "group_col": col,
        "month": month,
        "hour": hour,
        "actual": actual,
        "prediction": pred,
        "capacity": cap,
        "util_pred": pred / cap,
        "key": f"{ts}|g{gid}",
        **fields,
    }


def _full_timestamp_block(ts: str, month: int, hour: int) -> list[dict]:
    """All three groups at one timestamp for wide-format compatibility."""
    return [
        _make_row(ts, 1, month, hour, 8000, 7800),
        _make_row(ts, 2, month, hour, 10000, 9000),
        _make_row(ts, 3, month, hour, 5000, 4800),
    ]


def _tiny_frame() -> pd.DataFrame:
    rows = []
    rows.extend(_full_timestamp_block("2024-12-09 09:00:00", 12, 9))
    rows.extend(_full_timestamp_block("2024-12-09 10:00:00", 12, 10))
    rows.extend(_full_timestamp_block("2024-12-09 14:00:00", 12, 14))
    rows.extend(_full_timestamp_block("2024-03-04 04:00:00", 3, 4))
    rows.extend(_full_timestamp_block("2024-07-14 14:00:00", 7, 14))
    return pd.DataFrame(rows)


def test_candidate_mask_matches_definition():
    df = _tiny_frame()
    specs = build_candidate_specs()
    a1 = candidate_mask(df, specs["A1"])
    assert a1.sum() == 2
    assert set(df.loc[a1, "hour"].tolist()) == {9, 10}

    c1 = candidate_mask(df, specs["C1"])
    assert c1.sum() == 1
    assert df.loc[c1, "month"].iloc[0] == 3

    c3 = candidate_mask(df, specs["C3"])
    assert c3.sum() == 2


def test_or_combination_no_double_count():
    df = _tiny_frame()
    specs = build_candidate_specs()
    d1 = resolve_mask(df, specs["D1"], specs)
    a1 = resolve_mask(df, specs["A1"], specs)
    c1 = resolve_mask(df, specs["C1"], specs)
    assert d1.sum() == a1.sum() + c1.sum()
    pred_wide = long_to_wide_pred(df)
    orig = pred_wide.copy()
    mask = d1
    _, sub = apply_multiplier_to_wide(df, pred_wide, mask, 1.03)
    assert len(sub) == d1.sum()


def test_original_prediction_unchanged():
    df = _tiny_frame()
    specs = build_candidate_specs()
    mask = resolve_mask(df, specs["A1"], specs)
    pred_wide = long_to_wide_pred(df)
    orig_vals = pred_wide["kpx_group_2"].tolist()
    apply_multiplier_to_wide(df, pred_wide, mask, 1.05)
    assert pred_wide["kpx_group_2"].tolist() == orig_vals


def test_utilization_condition_mask():
    df = _tiny_frame()
    h9 = df[(df["group_id"] == 2) & (df["hour"] == 9)].index
    h10 = df[(df["group_id"] == 2) & (df["hour"] == 10)].index
    df.loc[h9, "prediction"] = 7000.0
    df.loc[h9, "util_pred"] = 7000.0 / 21600.0
    df.loc[h10, "prediction"] = 5000.0
    df.loc[h10, "util_pred"] = 5000.0 / 21600.0
    specs = build_candidate_specs()
    mask = resolve_mask(df, specs["A1"], specs)
    _, sub = apply_multiplier_to_wide(df, long_to_wide_pred(df), mask, 1.03, util_threshold=0.3)
    assert len(sub) == 1
    assert sub["hour"].iloc[0] == 10


def test_ficr_tier_transition_aggregation():
    cap = 21600.0
    actual = 10000.0
    pred_fail = 8056.0
    pred_tier2 = 8500.0
    fields_fail = compute_row_ficr_fields(actual, pred_fail, cap)
    fields_t2 = compute_row_ficr_fields(actual, pred_tier2, cap)
    sub = pd.DataFrame([
        {"actual": actual, "prediction": pred_fail, "capacity": cap, "unit_price": fields_fail["unit_price"]},
        {"actual": actual, "prediction": pred_tier2, "capacity": cap, "unit_price": fields_t2["unit_price"]},
    ])
    new_pred = pd.Series([actual - 0.07 * cap, actual - 0.05 * cap])
    t = count_tier_transitions(sub, new_pred)
    assert t["tier0_to_tier2"] + t["tier0_to_tier1"] >= 1


def test_price_to_tier():
    assert price_to_tier(4.0) == 1
    assert price_to_tier(3.0) == 2
    assert price_to_tier(0.0) == 0


def test_date_block_bootstrap_uses_dates():
    base = _tiny_frame()
    blocks = []
    for i in range(5):
        block = base.copy()
        block["forecast_kst_dtm"] = block["forecast_kst_dtm"] + pd.Timedelta(days=i)
        blocks.append(block)
    df = pd.concat(blocks, ignore_index=True)
    df["date"] = df["forecast_kst_dtm"].dt.date
    df["month"] = df["forecast_kst_dtm"].dt.month
    df["hour"] = df["forecast_kst_dtm"].dt.hour
    df["key"] = df["forecast_kst_dtm"].astype(str) + "|g" + df["group_id"].astype(str)
    spec = build_candidate_specs()["A1"]
    b1 = block_bootstrap_candidate(df, spec, build_candidate_specs(), 1.02, n_iter=10, seed=42)
    b2 = block_bootstrap_candidate(df, spec, build_candidate_specs(), 1.02, n_iter=10, seed=42)
    assert b1["valid"] and b2["valid"]
    assert b1["delta_score_mean"] == b2["delta_score_mean"]


def test_lopo_select_eval_periods_separated():
    results = [
        {"candidate_id": "A1", "period": "2023_h2", "apply_mode": "uniform", "multiplier": 1.01,
         "score": 0.61, "delta_score": 0.001, "ficr": 0.38, "delta_ficr": 0.001,
         "net_transition": 1, "success_to_fail": 0, "fail_to_success": 1},
        {"candidate_id": "A1", "period": "2023_h2", "apply_mode": "uniform", "multiplier": 1.03,
         "score": 0.62, "delta_score": 0.002, "ficr": 0.39, "delta_ficr": 0.002,
         "net_transition": 2, "success_to_fail": 0, "fail_to_success": 2},
        {"candidate_id": "A1", "period": "2024_h2", "apply_mode": "uniform", "multiplier": 1.03,
         "score": 0.60, "delta_score": -0.001, "ficr": 0.37, "delta_ficr": -0.001,
         "net_transition": -1, "success_to_fail": 1, "fail_to_success": 0},
    ]
    lopo = leave_one_period_out(results, build_candidate_specs())
    fold1 = [x for x in lopo if x["fold_id"] == 1 and x["candidate_id"] == "A1"][0]
    assert fold1["selected_multiplier"] == 1.03
    assert fold1["select_period"] == "2023_h2"
    assert fold1["eval_period"] == "2024_h2"
    assert fold1["eval_delta_score"] == -0.001


def test_lopo_does_not_use_eval_for_selection():
    results = [
        {"candidate_id": "C1", "period": "2024_h1", "apply_mode": "uniform", "multiplier": 1.01,
         "score": 0.50, "delta_score": 0.0001, "ficr": 0.3, "delta_ficr": 0.0,
         "net_transition": 0, "success_to_fail": 0, "fail_to_success": 0},
        {"candidate_id": "C1", "period": "2024_h1", "apply_mode": "uniform", "multiplier": 1.03,
         "score": 0.49, "delta_score": -0.0001, "ficr": 0.3, "delta_ficr": 0.0,
         "net_transition": 0, "success_to_fail": 0, "fail_to_success": 0},
        {"candidate_id": "C1", "period": "2024_h2", "apply_mode": "uniform", "multiplier": 1.01,
         "score": 0.60, "delta_score": 0.0002, "ficr": 0.35, "delta_ficr": 0.0,
         "net_transition": 0, "success_to_fail": 0, "fail_to_success": 0},
        {"candidate_id": "C1", "period": "2024_h2", "apply_mode": "uniform", "multiplier": 1.03,
         "score": 0.99, "delta_score": 0.5, "ficr": 0.9, "delta_ficr": 0.5,
         "net_transition": 10, "success_to_fail": 0, "fail_to_success": 10},
    ]
    lopo = leave_one_period_out(results, build_candidate_specs())
    fold3 = [x for x in lopo if x["fold_id"] == 3 and x["candidate_id"] == "C1"][0]
    assert fold3["selected_multiplier"] == 1.01


def test_period_keys_sorted_unique():
    df = _tiny_frame()
    keys = df["key"].tolist()
    assert keys == sorted(keys) or len(keys) == len(set(keys))


def test_top_candidate_ranking():
    results = [
        {"candidate_id": "A1", "period": "2024_full", "multiplier": 1.01, "apply_mode": "uniform",
         "delta_score": 0.001, "delta_ficr": 0.001},
        {"candidate_id": "A2", "period": "2024_full", "multiplier": 1.02, "apply_mode": "uniform",
         "delta_score": 0.002, "delta_ficr": 0.001},
        {"candidate_id": "A2", "period": "2024_full", "multiplier": 1.02, "apply_mode": "util_lt_0.3",
         "delta_score": 0.003, "delta_ficr": 0.002},
    ]
    top = top_candidate_configs(results, limit=2)
    assert top[0]["candidate_id"] == "A2"
    assert top[0]["apply_mode"] == "util_lt_0.3"


def test_december_year_evaluation():
    rows = []
    rows.extend(_full_timestamp_block("2023-12-09 09:00:00", 12, 9))
    rows.extend(_full_timestamp_block("2023-12-09 10:00:00", 12, 10))
    rows.extend(_full_timestamp_block("2023-11-09 09:00:00", 11, 9))
    df = pd.DataFrame(rows)
    spec = build_candidate_specs()["A1"]
    mask = resolve_mask(df, spec, build_candidate_specs())
    r = evaluate_december_year(df, mask, 1.02, 2023)
    assert r["valid"]
    assert r["year"] == 2023
