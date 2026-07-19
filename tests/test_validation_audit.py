"""Tests for rolling validation audit framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validation_audit_lib import (
    PUBLIC_LB,
    RollingFold,
    build_rolling_folds,
    check_fold_disjoint,
    fold_similarity_to_public,
    issue_cycle_disjoint_masks,
    public_lb_delta,
)
from src.config import VALID_START
from src.split_utils import KEYS


def test_rolling_folds_chronological():
    folds = build_rolling_folds()
    assert 6 <= len(folds) <= 8
    for f in folds:
        assert f.valid_end > f.valid_start
    for a, b in zip(folds, folds[1:]):
        assert a.valid_end == b.valid_start
        assert a.valid_start < b.valid_start


def test_fold_validation_length_at_least_2_months():
    folds = build_rolling_folds()
    for f in folds:
        days = (f.valid_end - f.valid_start).days
        assert days >= 60


def test_folds_are_disjoint():
    result = check_fold_disjoint(build_rolling_folds())
    assert result["all_disjoint"]
    assert result["overlapping_pairs"] == []


def test_issue_cycle_disjoint_no_overlap_after_filter():
    df = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(
                ["2023-03-31 23:00:00", "2023-04-01 01:00:00", "2023-04-02 01:00:00"]
            ),
            "data_available_kst_dtm": pd.to_datetime(
                ["2023-03-31 11:00:00", "2023-03-31 11:00:00", "2023-04-01 11:00:00"]
            ),
        }
    )
    vs = pd.Timestamp("2023-04-01 01:00:00")
    ve = pd.Timestamp("2023-05-01 01:00:00")
    train_mask, valid_mask, spanning = issue_cycle_disjoint_masks(df, vs, ve)
    assert len(spanning) == 1
    train_issues = set(df.loc[train_mask, "data_available_kst_dtm"])
    valid_issues = set(df.loc[valid_mask, "data_available_kst_dtm"])
    assert train_issues.isdisjoint(valid_issues)


def test_public_lb_delta_values():
    pub = public_lb_delta()
    assert pub["delta_score"] == pytest.approx(
        PUBLIC_LB["v13"]["score"] - PUBLIC_LB["v12"]["score"], abs=1e-10
    )
    assert pub["delta_nmae"] == pytest.approx(
        PUBLIC_LB["v13"]["1_minus_nmae"] - PUBLIC_LB["v12"]["1_minus_nmae"], abs=1e-10
    )
    assert pub["delta_ficr"] == pytest.approx(
        PUBLIC_LB["v13"]["ficr"] - PUBLIC_LB["v12"]["ficr"], abs=1e-10
    )


def test_fold_similarity_identical_delta():
    pub = public_lb_delta()
    row = {"fold_id": 1, "score": 0.62, "1_minus_nmae": 0.85, "ficr": 0.38}
    base = {
        "fold_id": 1,
        "score": 0.62 - pub["delta_score"],
        "1_minus_nmae": 0.85 - pub["delta_nmae"],
        "ficr": 0.38 - pub["delta_ficr"],
    }
    sim = fold_similarity_to_public(row, base, pub)
    assert sim["direction_match_all"] == 1
    assert sim["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert sim["normalized_euclidean_distance"] == pytest.approx(0.0, abs=1e-6)


def test_lead_hours_calculation():
    issue = pd.Timestamp("2024-01-01 11:00:00")
    target = pd.Timestamp("2024-01-02 01:00:00")
    lead = (target - issue).total_seconds() / 3600.0
    assert lead == pytest.approx(14.0)


def test_merge_keys_are_explicit():
    assert KEYS == ["forecast_kst_dtm", "group_id"]


def test_validation_audit_json_exists_after_run():
    path = ROOT / "outputs" / "validation_audit.json"
    if not path.exists():
        pytest.skip("Run scripts/audit_validation_and_horizon.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "train_label_range" in data
    assert "holdout_v13_best" in data
    assert data["validation_holdout"]["valid_start"] == VALID_START


@pytest.mark.slow
def test_holdout_v13_reproduction():
    from scripts.validation_audit_lib import reproduce_holdout_v13
    from src.data_loader import load_labels

    scores = reproduce_holdout_v13(load_labels())
    assert scores["score"] > 0.61
    assert scores["n_rows"] > 8000


def test_fold_similarity_opposite_direction():
    pub = public_lb_delta()
    v13 = {"fold_id": 2, "score": 0.61, "1_minus_nmae": 0.84, "ficr": 0.39}
    v12 = {"fold_id": 2, "score": 0.62, "1_minus_nmae": 0.85, "ficr": 0.38}
    sim = fold_similarity_to_public(v13, v12, pub)
    assert sim["cosine_similarity"] < 0
