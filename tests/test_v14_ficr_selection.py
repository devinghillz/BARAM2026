"""Tests for v14 FICR final selection."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ficr_candidate_families_lib import build_candidate_specs
from scripts.v14_ficr_selection_lib import (
    SUBMISSION_FILENAMES,
    apply_spec_to_submission,
    build_ablation_specs,
    build_robust_ranking_table,
    composite_rank_key,
    decide_final_selection,
    load_prior_results,
    passes_mandatory,
    rank_eligible_candidates,
    selection_spec_for_choice,
)


def _sample_ranking_row(**kwargs) -> pd.Series:
    base = {
        "candidate_id": "A1",
        "delta_score_2024_full": 0.0002,
        "delta_score_2024_h2": 0.0003,
        "delta_score_2023_h2": 0.0003,
        "lopo_positive_folds": 2,
        "bootstrap_p_delta_score_positive": 0.95,
        "bootstrap_delta_score_p05": 0.00005,
        "net_transition": 5,
        "fail_to_success": 5,
        "success_to_fail": 0,
        "multiplier": 1.03,
        "cross_period_min": 0.0002,
        "simplicity_score": 1,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_mandatory_filter_rejects_low_bootstrap():
    row = _sample_ranking_row(bootstrap_p_delta_score_positive=0.5)
    ok, fails = passes_mandatory(row)
    assert not ok
    assert "bootstrap_prob_low" in fails


def test_mandatory_filter_rejects_serious_negative_p05():
    row = _sample_ranking_row(bootstrap_delta_score_p05=-0.0002)
    ok, fails = passes_mandatory(row)
    assert not ok
    assert "bootstrap_p05_serious_negative" in fails


def test_composite_ranking_order():
    a1 = _sample_ranking_row(candidate_id="A1", cross_period_min=0.0003, bootstrap_p_delta_score_positive=0.96)
    d1 = _sample_ranking_row(candidate_id="D1", cross_period_min=0.00035, bootstrap_p_delta_score_positive=0.98)
    assert composite_rank_key(d1) > composite_rank_key(a1)


def test_tie_prefers_simpler_candidate():
    a1 = _sample_ranking_row(
        candidate_id="A1",
        simplicity_score=1,
        delta_score_2024_full=0.00031,
        cross_period_min=0.0003,
        bootstrap_p_delta_score_positive=0.96,
        bootstrap_delta_score_p05=0.00005,
    )
    d1 = _sample_ranking_row(
        candidate_id="D1",
        simplicity_score=4,
        delta_score_2024_full=0.00031,
        cross_period_min=0.0003,
        bootstrap_p_delta_score_positive=0.96,
        bootstrap_delta_score_p05=0.00005,
    )
    ranked = rank_eligible_candidates(pd.DataFrame([d1.to_dict(), a1.to_dict()]))
    assert ranked.iloc[0]["candidate_id"] == "A1"


def test_b1_hour_masks():
    specs = build_ablation_specs()
    assert specs["B1"].hours == {9, 10, 14, 15}


def test_b1_minus_h14_mask():
    specs = build_ablation_specs()
    assert specs["B1_minus_h14"].hours == {9, 10, 15}


def test_b1_minus_h15_mask():
    specs = build_ablation_specs()
    assert specs["B1_minus_h15"].hours == {9, 10, 14}


def test_multiplier_applied_only_on_slice():
    base = pd.DataFrame({
        "forecast_id": [1, 2, 3],
        "forecast_kst_dtm": [
            "2025-12-09 09:00:00",
            "2025-12-09 10:00:00",
            "2025-03-04 04:00:00",
        ],
        "kpx_group_1": [100.0, 100.0, 200.0],
        "kpx_group_2": [1000.0, 1000.0, 1000.0],
        "kpx_group_3": [50.0, 50.0, 50.0],
    })
    spec = build_candidate_specs()["A1"]
    out = apply_spec_to_submission(base, spec, 1.03)
    assert out.loc[0, "kpx_group_2"] == pytest.approx(1030.0)
    assert out.loc[2, "kpx_group_2"] == 1000.0


def test_d1_applies_two_slices():
    base = pd.DataFrame({
        "forecast_id": [1, 2],
        "forecast_kst_dtm": ["2025-12-09 09:00:00", "2025-03-04 04:00:00"],
        "kpx_group_1": [200.0, 200.0],
        "kpx_group_2": [1000.0, 1000.0],
        "kpx_group_3": [50.0, 50.0],
    })
    sel = selection_spec_for_choice("SELECT_D1")
    out = apply_spec_to_submission(base, sel.spec, 1.03)
    assert out.loc[0, "kpx_group_2"] == pytest.approx(1030.0)
    assert out.loc[1, "kpx_group_1"] == pytest.approx(206.0)


def test_selection_reproducible_from_prior():
    prior = load_prior_results()
    ranking = build_robust_ranking_table(prior)
    r1 = rank_eligible_candidates(ranking)
    r2 = rank_eligible_candidates(ranking)
    assert r1["candidate_id"].tolist() == r2["candidate_id"].tolist()


def test_submission_filename_mapping():
    assert "h9h10" in SUBMISSION_FILENAMES["SELECT_A1"]


def test_keep_v13_best_when_no_eligible():
    ablation = pd.DataFrame([
        {"candidate_id": "B1", "period": "2024_full", "delta_score": 0.0003},
        {"candidate_id": "B1", "period": "bootstrap_summary", "bootstrap_p05": 0.0001, "bootstrap_p_positive": 0.9},
    ])
    sel = decide_final_selection(pd.DataFrame(), {"passes": True}, ablation)
    assert sel["choice"] == "KEEP_V13_BEST"


def test_capacity_safe_skips_near_cap_rows():
    base = pd.DataFrame({
        "forecast_id": [1],
        "forecast_kst_dtm": ["2025-03-04 04:00:00"],
        "kpx_group_1": [21000.0],
        "kpx_group_2": [1000.0],
        "kpx_group_3": [50.0],
    })
    sel = selection_spec_for_choice("SELECT_D1")
    out = apply_spec_to_submission(base, sel.spec, 1.03)
    assert out.loc[0, "kpx_group_1"] == 21000.0


@pytest.mark.skipif(not (ROOT / "outputs" / "v13_ficr_candidate_families.json").exists(), reason="prior results")
def test_prior_robust_count():
    prior = load_prior_results()
    ranking = build_robust_ranking_table(prior)
    assert len(ranking) == 6
