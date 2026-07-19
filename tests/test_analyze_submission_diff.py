"""Tests for submission diff analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.submission_diff_lib import (
    G3FEB_KEEP_CONFIG,
    OLD_HYBRID_CONFIG,
    V13_BEST_CONFIG,
    check_key_integrity,
    clipping_analysis,
    compute_performance_recommendation,
    infer_submission_config,
    is_hotspot_row,
    merge_submissions,
    submission_verdict,
    wide_to_long,
)


def _verdict_inputs(
    integrity=None,
    overall=None,
    hotspot=None,
    clipping=None,
    merged=None,
    base_config=None,
    candidate_config=None,
):
    integrity = integrity or {"keys_fully_match": True, "duplicate_keys_base": 0, "duplicate_keys_candidate": 0}
    overall = overall or {
        "changed_ratio": 0.99,
        "global_mean_change_pct": 0.05,
        "absolute_normalized_delta_median": 0.0001,
        "absolute_normalized_delta_quantiles": {"p95": 0.001},
        "pct_abs_norm_delta_ge_1pct": 0.0,
        "pct_abs_norm_delta_ge_2pct": 0.0,
    }
    hotspot = hotspot or {
        "abs_delta_hotspot_share": 0.06,
        "non_hotspot_changed_ratio": 0.99,
    }
    clipping = clipping or {
        "candidate_negative": 0,
        "candidate_above_capacity": 0,
        "base_negative": 0,
        "candidate_nan": 0,
        "candidate_inf": 0,
        "newly_clipped_to_zero": 0,
        "newly_clipped_to_capacity": 0,
    }
    return submission_verdict(
        integrity,
        overall,
        hotspot,
        clipping,
        base_config or V13_BEST_CONFIG,
        candidate_config or OLD_HYBRID_CONFIG,
        schema={"schema_valid": True, "row_count_match": True, "columns_match": True},
        merged=merged,
    )


def _wide(ids, ts, g1, g2, g3):
    return pd.DataFrame({
        "forecast_id": ids,
        "forecast_kst_dtm": pd.to_datetime(ts),
        "kpx_group_1": g1,
        "kpx_group_2": g2,
        "kpx_group_3": g3,
    })


def test_merge_aligns_by_key_not_row_order():
    base = _wide(
        ["a", "b"],
        ["2025-01-01 01:00:00", "2025-01-01 02:00:00"],
        [100.0, 200.0],
        [100.0, 200.0],
        [100.0, 200.0],
    )
    cand = _wide(
        ["b", "a"],
        ["2025-01-01 02:00:00", "2025-01-01 01:00:00"],
        [210.0, 110.0],
        [210.0, 110.0],
        [210.0, 110.0],
    )
    merged = merge_submissions(base, cand)
    assert len(merged) == 6
    row = merged[(merged["forecast_id"] == "a") & (merged["group_id"] == 1)].iloc[0]
    assert row["p_base"] == 100.0
    assert row["p_candidate"] == 110.0
    assert row["delta"] == 10.0


def test_duplicate_key_detection():
    base = _wide(["a", "a"], ["2025-01-01 01:00:00", "2025-01-01 01:00:00"], [1, 2], [1, 2], [1, 2])
    cand = _wide(["a"], ["2025-01-01 01:00:00"], [1], [1], [1])
    bl = wide_to_long(base)
    cl = wide_to_long(cand)
    integrity = check_key_integrity(bl, cl)
    assert integrity["duplicate_keys_base"] >= 3
    assert integrity["keys_fully_match"] is False


def test_missing_key_detection():
    base = _wide(["a", "b"], ["2025-01-01 01:00:00", "2025-01-01 02:00:00"], [1, 2], [1, 2], [1, 2])
    cand = _wide(["a"], ["2025-01-01 01:00:00"], [1], [1], [1])
    bl = wide_to_long(base)
    cl = wide_to_long(cand)
    integrity = check_key_integrity(bl, cl)
    assert integrity["missing_in_candidate"] == 3
    assert integrity["keys_fully_match"] is False


def test_capacity_normalization():
    base = _wide(["a"], ["2025-07-01 00:00:00"], [2160.0], [0.0], [0.0])
    cand = _wide(["a"], ["2025-07-01 00:00:00"], [2160.0 + 216.0], [0.0], [0.0])
    merged = merge_submissions(base, cand)
    g1 = merged[merged["group_id"] == 1].iloc[0]
    assert g1["capacity"] == 21600
    assert g1["normalized_delta"] == pytest.approx(216.0 / 21600)


def test_hotspot_flag():
    assert is_hotspot_row(1, 7, 0) is True
    assert is_hotspot_row(3, 11, 4) is True
    assert is_hotspot_row(1, 3, 10) is False


def test_pct_change_metrics_global_vs_rowwise():
    from scripts.submission_diff_lib import compute_pct_change_metrics

    base = pd.Series([0.0, 100.0, 200.0])
    cand = pd.Series([0.0, 110.0, 190.0])
    m = compute_pct_change_metrics(base, cand)
    # global: (0+110+190)/(3) / (0+100+200)/3 - 1 = 100/100 - 1 = 0? 
    # mean base = 100, mean cand = 100, global = 0
    assert m["global_mean_change_pct"] == pytest.approx(0.0, abs=1e-9)
    # rowwise on base>0: mean(10%, -5%) = 2.5%
    assert m["rowwise_mean_change_pct"] == pytest.approx(2.5, abs=1e-9)
    assert m["rowwise_excluded_zero_base_rows"] == 1


def test_pct_change_positive_global_negative_rowwise():
    """mean delta>0 but rowwise negative when low-base rows excluded."""
    from scripts.submission_diff_lib import compute_pct_change_metrics

    base = pd.Series([0.0, 100.0, 10000.0])
    cand = pd.Series([0.0, 90.0, 10100.0])
    m = compute_pct_change_metrics(base, cand)
    assert m["global_mean_change_pct"] > 0
    assert m["rowwise_mean_change_pct"] < 0


def test_clipping_nan_and_capacity():
    base = _wide(["a", "b"], ["2025-01-01 01:00:00", "2025-01-01 02:00:00"], [0.0, 21600.0], [0.0, 0.0], [0.0, 0.0])
    cand = _wide(["a", "b"], ["2025-01-01 01:00:00", "2025-01-01 02:00:00"], [100.0, 25000.0], [0.0, 0.0], [0.0, 0.0])
    merged = merge_submissions(base, cand)
    clip = clipping_analysis(merged)
    assert clip["base_at_zero"] >= 1
    assert clip["base_at_capacity"] >= 1
    assert clip["candidate_above_capacity"] >= 1


def test_global_micro_blend_99pct_changed_not_reject():
    """99% 행 변경 + 작은 p95 → reject 아님."""
    v = _verdict_inputs()
    assert v["submission_safety_verdict"] in {"safe", "safe_with_warnings"}
    assert v["submission_safety_verdict"] != "reject"
    assert v["change_pattern"]["global_micro_change"] is True


def test_key_mismatch_reject():
    v = _verdict_inputs(integrity={"keys_fully_match": False, "duplicate_keys_base": 0, "duplicate_keys_candidate": 0})
    assert v["submission_safety_verdict"] == "reject"
    assert any(r["rule_id"] == "keys_fully_match" for r in v["triggered_rules"])


def test_nan_reject():
    clipping = {
        "candidate_negative": 0,
        "candidate_above_capacity": 0,
        "base_negative": 0,
        "candidate_nan": 3,
        "candidate_inf": 0,
        "newly_clipped_to_zero": 0,
        "newly_clipped_to_capacity": 0,
    }
    v = _verdict_inputs(clipping=clipping)
    assert v["submission_safety_verdict"] == "reject"
    assert any(r["rule_id"] == "candidate_nan" for r in v["triggered_rules"])


def test_capacity_overflow_reject():
    clipping = {
        "candidate_negative": 0,
        "candidate_above_capacity": 2,
        "base_negative": 0,
        "candidate_nan": 0,
        "candidate_inf": 0,
        "newly_clipped_to_zero": 0,
        "newly_clipped_to_capacity": 0,
    }
    v = _verdict_inputs(clipping=clipping)
    assert v["submission_safety_verdict"] == "reject"
    assert any(r["rule_id"] == "candidate_above_capacity" for r in v["triggered_rules"])


def _merged_with_hotspot_large_change(pct_delta: float = 0.015):
    """g2 m2 h19 hotspot 1행에만 capacity 비율 변화."""
    ts = pd.to_datetime(["2025-02-01 19:00:00", "2025-03-01 10:00:00"])
    delta_kwh = 21600 * pct_delta
    micro = 1.0
    base = _wide(
        ["f0", "f1"],
        ts,
        [1000.0, 1000.0],
        [1000.0, 1000.0],
        [1000.0, 1000.0],
    )
    cand = base.copy()
    cand.loc[0, "kpx_group_2"] = 1000.0 + delta_kwh
    cand.loc[1, "kpx_group_1"] = 1000.0 + micro
    merged = merge_submissions(base, cand)
    merged["is_pipeline_hotspot"] = merged.apply(
        lambda r: is_hotspot_row(int(r["group_id"]), int(r["month"]), int(r["hour"])), axis=1
    )
    return merged


def test_hotspot_1_to_2pct_safe_with_warnings_or_review():
    merged = _merged_with_hotspot_large_change(pct_delta=0.015)
    overall = {
        "changed_ratio": 1.0,
        "global_mean_change_pct": 0.1,
        "absolute_normalized_delta_median": 0.0005,
        "absolute_normalized_delta_quantiles": {"p95": 0.015},
        "pct_abs_norm_delta_ge_1pct": float((merged["absolute_normalized_delta"] >= 0.01).mean()),
        "pct_abs_norm_delta_ge_2pct": float((merged["absolute_normalized_delta"] >= 0.02).mean()),
    }
    v = _verdict_inputs(overall=overall, merged=merged)
    assert v["submission_safety_verdict"] in {"safe_with_warnings", "review", "safe"}
    assert v["submission_safety_verdict"] != "reject"


def test_performance_recommendation_default_not_evaluated():
    v = _verdict_inputs()
    assert v["performance_recommendation"] == "not_evaluated"
    perf = compute_performance_recommendation(None)
    assert perf["recommendation"] == "not_evaluated"


def test_triggered_rules_in_verdict_structure():
    v = _verdict_inputs(integrity={"keys_fully_match": False, "duplicate_keys_base": 1, "duplicate_keys_candidate": 0})
    assert "triggered_rules" in v
    assert "rule_results" in v
    assert len(v["triggered_rules"]) >= 1
    for r in v["triggered_rules"]:
        assert "rule_id" in r
        assert "measured_value" in r
        assert "threshold" in r
        assert "reason" in r


def test_infer_submission_config_g3feb_keep():
    cfg = infer_submission_config("submissions/v13_hybrid_g105_g3feb102_v0305.csv")
    assert cfg == G3FEB_KEEP_CONFIG


def test_infer_submission_config_old_hybrid():
    cfg = infer_submission_config("submissions/v13_hybrid_g105_g1_v0305.csv")
    assert cfg == OLD_HYBRID_CONFIG


@pytest.mark.skipif(
    not (ROOT / "submissions" / "v13_best.csv").exists()
    or not (ROOT / "submissions" / "v13_hybrid_g105_g3feb102_v0305.csv").exists(),
    reason="submission files not present",
)
def test_triggered_rules_in_json_and_markdown_output(tmp_path):
    import scripts.analyze_submission_diff as mod

    out = tmp_path / "test_diff.json"
    md = tmp_path / "test_diff.md"
    argv = [
        "analyze_submission_diff.py",
        "--base", str(ROOT / "submissions" / "v13_best.csv"),
        "--candidate", str(ROOT / "submissions" / "v13_hybrid_g105_g3feb102_v0305.csv"),
        "--output", str(out),
        "--markdown", str(md),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "triggered_rules" in data["verdict"]
    assert "submission_safety_verdict" in data["verdict"]
    assert data["verdict"]["performance_recommendation"] == "not_evaluated"

    md_text = md.read_text(encoding="utf-8")
    assert "submission_safety_verdict" in md_text
    assert "Triggered rules" in md_text
    assert "performance_recommendation" in md_text
