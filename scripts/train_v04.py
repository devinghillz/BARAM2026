"""
BARAM 2026 v0.4 — FICR 집중 2차 개선

v0.3 LB 0.611 대비:
  - 그룹별 quantile blend (group_3 가중)
  - 풍속 5~12 m/s 검증 기반 승수 보정
  - sweet-spot 피처 추가
  - LB 오프셋 보정 (v0.3: local - LB = +0.008)

사용법:
  python scripts/train_v04.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import apply_ws_band_multipliers, tune_ws_band_multipliers
from src.config import GROUP_COLUMNS, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.train_v03 import (
    blend_wide,
    build_dataset,
    get_feature_columns,
    predict_wide,
    GroupBlendModel,
    _filter_group_train,
)

# v0.3 LB 실측: local 0.619 -> LB 0.611
LB_OFFSET_V03 = -0.008

Q_GRID_G3 = [0.70, 0.75, 0.80]
Q_GRID_G12 = [0.55, 0.60]
W_IDW_GRID = [0.7, 0.8, 0.9]


def tune_q_weights_fast(
    train_near: pd.DataFrame,
    train_idw: pd.DataFrame,
    valid_near: pd.DataFrame,
    valid_idw: pd.DataFrame,
    feature_cols: list[str],
    labels: pd.DataFrame,
    valid_start: pd.Timestamp,
    w_idw: float,
) -> tuple[dict[int, float], dict]:
    """group_1/2는 보수적으로, group_3만 quantile 비중 확대."""
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]]
    best_q = {1: 0.55, 2: 0.60, 3: 0.75}
    best_scores = {"score": -1.0}

    for q1 in Q_GRID_G12:
        for q2 in Q_GRID_G12:
            for q3 in Q_GRID_G3:
                trial = {1: q1, 2: q2, 3: q3}
                print(f"    q=({q1},{q2},{q3})...", flush=True)
                m_near = train_all_groups_qmap(train_near, feature_cols, trial)
                m_idw = train_all_groups_qmap(train_idw, feature_cols, trial)
                pred = blend_wide(
                    predict_wide(m_near, valid_near, feature_cols),
                    predict_wide(m_idw, valid_idw, feature_cols),
                    w_idw,
                )
                scores = evaluate_submission(valid_true, pred, time_col="kst_dtm")
                if scores["score"] > best_scores["score"]:
                    best_scores = scores
                    best_q = trial

    return best_q, best_scores


def train_all_groups_qmap(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    q_weights: dict[int, float],
) -> dict[int, GroupBlendModel]:
    from src.config import GROUP_CAPACITY_KWH

    models = {}
    for gid in [1, 2, 3]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        m = GroupBlendModel(q_weight=q_weights[gid])
        m.fit(part[feature_cols], part["power_kwh"], cap)
        models[gid] = m
    return models


def tune_w_idw_fast(
    train_near: pd.DataFrame,
    train_idw: pd.DataFrame,
    valid_near: pd.DataFrame,
    valid_idw: pd.DataFrame,
    feature_cols: list[str],
    labels: pd.DataFrame,
    valid_start: pd.Timestamp,
    q_weights: dict[int, float],
) -> tuple[float, dict]:
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]]
    m_near = train_all_groups_qmap(train_near, feature_cols, q_weights)
    m_idw = train_all_groups_qmap(train_idw, feature_cols, q_weights)
    p_near = predict_wide(m_near, valid_near, feature_cols)
    p_idw = predict_wide(m_idw, valid_idw, feature_cols)

    best_w, best_scores = 0.8, {"score": -1.0}
    for i, w in enumerate(W_IDW_GRID, start=1):
        print(f"    w_idw {w:.1f} ({i}/{len(W_IDW_GRID)})...", flush=True)
        merged = blend_wide(p_near, p_idw, w)
        scores = evaluate_submission(valid_true, merged, time_col="kst_dtm")
        if scores["score"] > best_scores["score"]:
            best_scores = scores
            best_w = w
    return best_w, best_scores


def run(valid_start: str) -> None:
    labels = load_labels()
    scada_curve = build_scada_monthly_curve()
    valid_start_ts = pd.Timestamp(valid_start)

    print("[1/7] Build datasets...")
    train_near = build_dataset(labels, "nearest", valid_start_ts, scada_curve, "train")
    train_idw = build_dataset(labels, "idw", valid_start_ts, scada_curve, "train")
    test_near = build_dataset(labels, "nearest", None, scada_curve, "test")
    test_idw = build_dataset(labels, "idw", None, scada_curve, "test")
    feature_cols = get_feature_columns(train_idw)
    print(f"  features: {len(feature_cols)}")

    fit_mask = train_idw["forecast_kst_dtm"] < valid_start_ts
    valid_mask = train_idw["forecast_kst_dtm"] >= valid_start_ts
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start_ts, ["kst_dtm", *GROUP_COLUMNS]]

    print("[2/7] Tune per-group q_weight (12 combos, ~4분)...")
    w_idw_init = 0.8
    q_weights, _ = tune_q_weights_fast(
        train_near.loc[fit_mask],
        train_idw.loc[fit_mask],
        train_near.loc[valid_mask],
        train_idw.loc[valid_mask],
        feature_cols,
        labels,
        valid_start_ts,
        w_idw_init,
    )
    print(f"  q_weights: {q_weights}")

    print("[3/7] Tune w_idw...")
    w_idw, val_scores = tune_w_idw_fast(
        train_near.loc[fit_mask],
        train_idw.loc[fit_mask],
        train_near.loc[valid_mask],
        train_idw.loc[valid_mask],
        feature_cols,
        labels,
        valid_start_ts,
        q_weights,
    )
    print(f"  w_idw: {w_idw}")
    print(f"  pre-calib Score: {val_scores['score']:.6f}  FICR: {val_scores['ficr']:.6f}")

    print("[4/7] Tune wind-band multipliers (5~12 m/s)...")
    m_near = train_all_groups_qmap(train_near.loc[fit_mask], feature_cols, q_weights)
    m_idw = train_all_groups_qmap(train_idw.loc[fit_mask], feature_cols, q_weights)
    val_pred = blend_wide(
        predict_wide(m_near, train_near.loc[valid_mask], feature_cols),
        predict_wide(m_idw, train_idw.loc[valid_mask], feature_cols),
        w_idw,
    )
    ws_mult, cal_scores = tune_ws_band_multipliers(
        val_pred,
        train_idw.loc[valid_mask],
        valid_true,
    )
    print(f"  ws_multipliers: {ws_mult}")
    print(f"  1-NMAE : {cal_scores['1_minus_nmae']:.6f}")
    print(f"  FICR   : {cal_scores['ficr']:.6f}")
    print(f"  Score  : {cal_scores['score']:.6f}")

    print("[5/7] Retrain on full data...")
    full_near = train_all_groups_qmap(train_near, feature_cols, q_weights)
    full_idw = train_all_groups_qmap(train_idw, feature_cols, q_weights)

    print("[6/7] Predict test + apply calibration...")
    test_pred = blend_wide(
        predict_wide(full_near, test_near, feature_cols),
        predict_wide(full_idw, test_idw, feature_cols),
        w_idw,
    )
    test_pred = apply_ws_band_multipliers(test_pred, test_idw, ws_mult)

    print("[7/7] Save submission...")
    submission = load_submission_template().drop(columns=GROUP_COLUMNS)
    submission = submission.merge(test_pred, on="forecast_kst_dtm", how="left")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSION_DIR / "v04_pergroup_ws_calib.csv"
    submission.to_csv(out, index=False, encoding="utf-8-sig")

    projected_lb = cal_scores["score"] + LB_OFFSET_V03
    print(f"Saved: {out}")
    print("\n--- v0.4 summary (2024 hold-out) ---")
    print(f"v0.3 LB        : 0.611")
    print(f"v0.4 local     : {cal_scores['score']:.6f}")
    print(f"projected LB   : ~{projected_lb:.3f}  (offset {LB_OFFSET_V03:+.3f})")
    print(f"q_weights      : {q_weights}")
    print(f"w_idw          : {w_idw}")
    print(f"ws_multipliers : {ws_mult}")
    print(
        f"제출 메모: v0.4 | 그룹별 분위수 블렌드 + 풍속5~12m/s 보정 | "
        f"로컬 {cal_scores['score']:.3f}, LB 예상 ~{projected_lb:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-start", default=VALID_START)
    run(valid_start=parser.parse_args().valid_start)
