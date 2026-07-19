"""Tests for g3feb_keep hypothesis split."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.submission_diff_lib import (
    G3FEB_KEEP_CONFIG,
    HOTSPOT_G23,
    OLD_HYBRID_CONFIG,
    V13_BEST_CONFIG,
    merge_submissions,
    wide_to_long,
)


def test_g3feb_keep_config_feb_multiplier():
    assert G3FEB_KEEP_CONFIG["g3_season"][2] == 1.02
    assert OLD_HYBRID_CONFIG["g3_season"][2] == 1.0
    assert G3FEB_KEEP_CONFIG["blend_v03"] == 0.05
    assert G3FEB_KEEP_CONFIG["g2_mult"] == 1.05


def test_b_vs_c_diff_only_on_g3_feb_hotspot():
    """Simulate: identical except g3 feb hotspot rows."""
    n = 4
    ts = pd.to_datetime(["2025-02-01 07:00:00", "2025-02-01 22:00:00", "2025-03-01 10:00:00", "2025-08-01 05:00:00"])
    ids = [f"f{i}" for i in range(n)]
    base_g3 = [100.0, 200.0, 300.0, 400.0]
    cand_g3 = [102.0, 204.0, 300.0, 400.0]  # only first two differ (feb hs hours 7,22)

    def wide(g3):
        return pd.DataFrame({
            "forecast_id": ids,
            "forecast_kst_dtm": ts,
            "kpx_group_1": [1.0] * n,
            "kpx_group_2": [1.0] * n,
            "kpx_group_3": g3,
        })

    merged = merge_submissions(wide(base_g3), wide(cand_g3))
    feb_hs = (
        (merged["group_id"] == 3)
        & (merged["month"] == 2)
        & merged["hour"].isin(HOTSPOT_G23[3][2])
    )
    non = merged[~feb_hs & (merged["group_id"] == 3)]
    assert (non["delta"].abs() <= 1e-9).all()
    feb = merged[feb_hs]
    assert (feb["delta"].abs() > 0).all()


@pytest.mark.skipif(
    not (ROOT / "submissions" / "v13_hybrid_g105_g1_v0305.csv").exists()
    or not (ROOT / "submissions" / "v13_hybrid_g105_g3feb102_v0305.csv").exists(),
    reason="submission files not generated yet",
)
def test_real_submissions_b_vs_c_non_feb_identical():
    from scripts.eval_v13_g3feb_keep import check_b_vs_c_only_g3_feb

    old = ROOT / "submissions" / "v13_hybrid_g105_g1_v0305.csv"
    new = ROOT / "submissions" / "v13_hybrid_g105_g3feb102_v0305.csv"
    r = check_b_vs_c_only_g3_feb(old, new)
    assert r["identical_non_g3_feb"] is True
    assert r["max_abs_delta_non_g3_feb"] < 1e-4


@pytest.mark.skipif(
    not (ROOT / "submissions" / "v13_hybrid_g105_g3feb102_v0305.csv").exists(),
    reason="submission not generated",
)
def test_submission_matches_sample_rows():
    from scripts.eval_v13_g3feb_keep import verify_sample_submission

    path = ROOT / "submissions" / "v13_hybrid_g105_g3feb102_v0305.csv"
    v = verify_sample_submission(path)
    assert v["row_match"] is True
    assert v["id_match"] is True
    assert v["columns_match"] is True
