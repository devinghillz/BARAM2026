"""Tests for v13 v03 × g2 factorial experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.submission_diff_lib import G3FEB_KEEP_CONFIG, V13_BEST_CONFIG
from scripts.v13_factorial_lib import (
    FACTORIAL_CONFIGS,
    FACTORIAL_PATHS,
    PERIOD_DEFS,
    build_bundles,
    config_diff_only,
    factorial_effects,
    g2_only_prediction_diff,
    submissions_identical,
    validate_period_keys,
    verify_sample_submission,
)


def test_a_config_values():
    cfg = FACTORIAL_CONFIGS["A_v13_best"]
    assert cfg["blend_v03"] == 0.07
    assert cfg["g2_mult"] == 1.06
    assert cfg["g3_season"][2] == 1.02
    assert cfg == V13_BEST_CONFIG


def test_b_config_values():
    cfg = FACTORIAL_CONFIGS["B_v03_5_only"]
    assert cfg["blend_v03"] == 0.05
    assert cfg["g2_mult"] == 1.06
    assert cfg["g3_season"][2] == 1.02


def test_c_config_values():
    cfg = FACTORIAL_CONFIGS["C_g2_105_only"]
    assert cfg["blend_v03"] == 0.07
    assert cfg["g2_mult"] == 1.05
    assert cfg["g3_season"][2] == 1.02


def test_d_config_values():
    cfg = FACTORIAL_CONFIGS["D_both"]
    assert cfg["blend_v03"] == 0.05
    assert cfg["g2_mult"] == 1.05
    assert cfg["g3_season"][2] == 1.02
    assert cfg == G3FEB_KEEP_CONFIG


def test_b_diff_only_v03_blend():
    a = FACTORIAL_CONFIGS["A_v13_best"]
    b = FACTORIAL_CONFIGS["B_v03_5_only"]
    assert config_diff_only(a, b, {"blend_v03"})
    assert a["blend_v03"] != b["blend_v03"]
    assert a["g2_mult"] == b["g2_mult"]


def test_c_diff_only_g2_mult():
    a = FACTORIAL_CONFIGS["A_v13_best"]
    c = FACTORIAL_CONFIGS["C_g2_105_only"]
    assert config_diff_only(a, c, {"g2_mult"})
    assert a["g2_mult"] != c["g2_mult"]
    assert a["blend_v03"] == c["blend_v03"]


def test_factorial_v03_main_effect_formula():
    scores = {
        "A_v13_best": {"score": 0.620, "1_minus_nmae": 0.863, "ficr": 0.377},
        "B_v03_5_only": {"score": 0.619, "1_minus_nmae": 0.862, "ficr": 0.376},
        "C_g2_105_only": {"score": 0.618, "1_minus_nmae": 0.861, "ficr": 0.375},
        "D_both": {"score": 0.617, "1_minus_nmae": 0.860, "ficr": 0.374},
    }
    fx = factorial_effects(scores)
    assert fx["score"]["v03_main"] == pytest.approx(((0.619 + 0.617) / 2) - ((0.620 + 0.618) / 2))
    assert fx["score"]["g2_main"] == pytest.approx(((0.618 + 0.617) / 2) - ((0.620 + 0.619) / 2))
    assert fx["score"]["interaction"] == pytest.approx(0.617 - 0.619 - 0.618 + 0.620)


def test_factorial_interaction_effect_formula():
    scores = {
        "A_v13_best": {"score": 1.0, "1_minus_nmae": 1.0, "ficr": 1.0},
        "B_v03_5_only": {"score": 2.0, "1_minus_nmae": 2.0, "ficr": 2.0},
        "C_g2_105_only": {"score": 3.0, "1_minus_nmae": 3.0, "ficr": 3.0},
        "D_both": {"score": 10.0, "1_minus_nmae": 10.0, "ficr": 10.0},
    }
    fx = factorial_effects(scores)
    assert fx["score"]["interaction"] == pytest.approx(10 - 2 - 3 + 1)
    assert fx["1_minus_nmae"]["interaction"] == pytest.approx(10 - 2 - 3 + 1)
    assert fx["ficr"]["interaction"] == pytest.approx(10 - 2 - 3 + 1)


@pytest.mark.skipif(
    not FACTORIAL_PATHS["A_v13_best"].exists(),
    reason="v13_best.csv missing",
)
def test_a_matches_v13_best_file():
    r = submissions_identical(FACTORIAL_PATHS["A_v13_best"], FACTORIAL_PATHS["A_v13_best"])
    assert r["identical"] is True


@pytest.mark.skipif(
    not FACTORIAL_PATHS["D_both"].exists(),
    reason="g3feb_keep missing",
)
def test_d_matches_g3feb_keep_file():
    r = submissions_identical(FACTORIAL_PATHS["D_both"], FACTORIAL_PATHS["D_both"])
    assert r["identical"] is True


@pytest.mark.skipif(
    not FACTORIAL_PATHS["A_v13_best"].exists()
    or not FACTORIAL_PATHS["C_g2_105_only"].exists(),
    reason="submissions not generated",
)
def test_c_vs_a_g2_hotspot_only_diff():
    r = g2_only_prediction_diff(FACTORIAL_PATHS["A_v13_best"], FACTORIAL_PATHS["C_g2_105_only"])
    assert r["non_g2_hotspot_identical"] is True


@pytest.mark.skipif(
    not all(FACTORIAL_PATHS[k].exists() for k in FACTORIAL_CONFIGS),
    reason="not all factorial submissions present",
)
def test_all_submissions_match_sample():
    for name, path in FACTORIAL_PATHS.items():
        v = verify_sample_submission(path)
        assert v["row_match"] is True, name
        assert v["id_match"] is True, name
        assert v["columns_match"] is True, name


@pytest.mark.slow
def test_period_splits_no_duplicate_ts():
    from src.data_loader import load_labels

    labels = load_labels()
    bundles = build_bundles(labels)
    report = validate_period_keys(bundles)
    for pname, info in report.items():
        assert info["duplicate_ts"] == 0, pname
        assert info["n_rows"] > 0, pname
        assert info["n_unique_ts"] == info["n_rows"], pname


def test_period_definitions_cover_four_periods():
    assert set(PERIOD_DEFS) == {"2024_full", "2024_h1", "2024_h2", "2023_h2"}


def test_submission_paths_mapping():
    assert FACTORIAL_PATHS["B_v03_5_only"].name == "v13_v0305_g2106_g3feb102.csv"
    assert FACTORIAL_PATHS["C_g2_105_only"].name == "v13_v0307_g2105_g3feb102.csv"
