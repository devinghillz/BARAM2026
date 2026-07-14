"""
BARAM 2026 v0.6 — group_3 집중 + 겨울 슬롯

v0.4/05b 기반:
  - group_3 quantile 강화 (q=0.85~0.90, alpha=0.62)
  - FICR 경계 구간 sample weight
  - group_3 풍속 5~12m/s 추가 보정
  - 1~3월 + 10/20/22시 슬롯 보정 (11월 제외)

사용법:
  python scripts/train_v06.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import (
    WINTER_MONTHS,
    apply_v04_calibrations,
    apply_ws_band_multipliers,
    tune_slot_multipliers,
)
from src.config import GROUP_COLUMNS, GROUP_CAPACITY_KWH, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.train_v03 import (
    blend_wide,
    build_dataset,
    ficr_sample_weights,
    get_feature_columns,
    make_lgbm_mean,
    make_lgbm_quantile,
    predict_wide,
    _filter_group_train,
)

LB_OFFSET = -0.008
W_IDW = 0.9
Q12 = 0.6
Q3_GRID = [0.80, 0.85, 0.90]
WS_G12 = {1: 1.10, 2: 1.08}
WS_G3_GRID = [1.0, 1.05, 1.08, 1.10]
Q_ALPHA = {1: 0.58, 2: 0.58, 3: 0.62}


def ficr_weights_v06(y: np.ndarray, capacity: float, gid: int) -> np.ndarray:
    util = y / capacity
    w = ficr_sample_weights(y, capacity)
    boundary = (util >= 0.25) & (util <= 0.75)
    w[boundary] *= 1.25
    if gid == 3:
        w[util >= 0.35] *= 1.2
    return w


class GroupBlendModelV06:
    def __init__(self, q_weight: float, q_alpha: float = 0.58, gid: int = 1):
        self.q_weight = q_weight
        self.q_alpha = q_alpha
        self.gid = gid
        self.mean_model = make_lgbm_mean()
        self.q_model = make_lgbm_quantile(alpha=q_alpha)
        self.capacity = 21600.0

    def fit(self, X, y, capacity: float):
        self.capacity = capacity
        w = ficr_weights_v06(y.to_numpy(), capacity, self.gid)
        self.mean_model.fit(X, y, regressor__sample_weight=w)
        self.q_model.fit(X, y, regressor__sample_weight=w)
        return self

    def predict(self, X) -> np.ndarray:
        pred = (1 - self.q_weight) * self.mean_model.predict(X) + self.q_weight * self.q_model.predict(X)
        return np.clip(pred, 0, self.capacity)


def train_groups_v06(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    q_weights: dict[int, float],
) -> dict[int, GroupBlendModelV06]:
    models = {}
    for gid in [1, 2, 3]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        m = GroupBlendModelV06(
            q_weight=q_weights[gid],
            q_alpha=Q_ALPHA[gid],
            gid=gid,
        )
        m.fit(part[feature_cols], part["power_kwh"], cap)
        models[gid] = m
    return models


def predict_blend(
    train_near, train_idw, valid_near, valid_idw, test_near, test_idw,
    feat, q_weights, fit_mask, valid_mask,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m_near = train_groups_v06(train_near.loc[fit_mask], feat, q_weights)
    m_idw = train_groups_v06(train_idw.loc[fit_mask], feat, q_weights)
    val = blend_wide(
        predict_wide(m_near, valid_near, feat),
        predict_wide(m_idw, valid_idw, feat),
        W_IDW,
    )
    f_near = train_groups_v06(train_near, feat, q_weights)
    f_idw = train_groups_v06(train_idw, feat, q_weights)
    test = blend_wide(
        predict_wide(f_near, test_near, feat),
        predict_wide(f_idw, test_idw, feat),
        W_IDW,
    )
    return val, test


def tune_q3(
    train_near, train_idw, valid_near, valid_idw, feat, labels, vs,
) -> tuple[dict[int, float], dict]:
    valid_true = labels.loc[labels["kst_dtm"] >= vs, ["kst_dtm", *GROUP_COLUMNS]]
    fit_near = train_near["forecast_kst_dtm"] < vs
    fit_idw = train_idw["forecast_kst_dtm"] < vs
    best_q = {1: Q12, 2: Q12, 3: 0.85}
    best_s = {"score": -1.0}

    for q3 in Q3_GRID:
        qw = {1: Q12, 2: Q12, 3: q3}
        print(f"    q3={q3:.2f}...", flush=True)
        m_near = train_groups_v06(train_near.loc[fit_near], feat, qw)
        m_idw = train_groups_v06(train_idw.loc[fit_idw], feat, qw)
        pred = blend_wide(
            predict_wide(m_near, valid_near, feat),
            predict_wide(m_idw, valid_idw, feat),
            W_IDW,
        )
        s = evaluate_submission(valid_true, pred, time_col="kst_dtm")
        if s["score"] > best_s["score"]:
            best_s = s
            best_q = qw
    return best_q, best_s


def tune_ws_g3(
    val_pred, long_valid, valid_true, ws_g12: dict[int, float],
) -> tuple[dict[int, float], dict]:
    best = {**ws_g12, 3: 1.0}
    best_s = {"score": -1.0}
    for m3 in WS_G3_GRID:
        trial = {**ws_g12, 3: m3}
        adj = apply_ws_band_multipliers(val_pred, long_valid, trial)
        s = evaluate_submission(valid_true, adj, time_col="kst_dtm")
        print(f"    ws_g3={m3:.2f} -> Score {s['score']:.6f}", flush=True)
        if s["score"] > best_s["score"]:
            best_s = s
            best[3] = m3
    return best, best_s


def run(valid_start: str) -> None:
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(valid_start)

    print("[1/7] Build datasets...")
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)
    fit = train_idw["forecast_kst_dtm"] < vs
    valid = train_idw["forecast_kst_dtm"] >= vs
    valid_true = labels.loc[labels["kst_dtm"] >= vs, ["kst_dtm", *GROUP_COLUMNS]]

    print("[2/7] Tune group_3 quantile weight...")
    q_weights, mid = tune_q3(
        train_near, train_idw,
        train_near.loc[valid], train_idw.loc[valid],
        feat, labels, vs,
    )
    print(f"  q_weights: {q_weights}  (mid Score {mid['score']:.6f})")

    print("[3/7] Train + tune group_3 ws multiplier...")
    m_near = train_groups_v06(train_near.loc[fit], feat, q_weights)
    m_idw = train_groups_v06(train_idw.loc[fit], feat, q_weights)
    val_pred = blend_wide(
        predict_wide(m_near, train_near.loc[valid], feat),
        predict_wide(m_idw, train_idw.loc[valid], feat),
        W_IDW,
    )
    ws_mult, ws_scores = tune_ws_g3(
        val_pred, train_idw.loc[valid], valid_true, WS_G12
    )
    val_ws = apply_ws_band_multipliers(val_pred, train_idw.loc[valid], ws_mult)
    print(f"  ws_multipliers: {ws_mult}")

    print("[4/7] Tune winter slot (1~3월, 10/20/22시)...")
    slot_mult, cal_scores = tune_slot_multipliers(
        val_ws, valid_true, worst_months=WINTER_MONTHS
    )
    print(f"  slot_multipliers: {slot_mult}")
    print(f"  1-NMAE : {cal_scores['1_minus_nmae']:.6f}")
    print(f"  FICR   : {cal_scores['ficr']:.6f}")
    print(f"  Score  : {cal_scores['score']:.6f}")

    print("[5/7] Retrain full data...")
    f_near = train_groups_v06(train_near, feat, q_weights)
    f_idw = train_groups_v06(train_idw, feat, q_weights)

    print("[6/7] Predict test + calibrate...")
    test_pred = blend_wide(
        predict_wide(f_near, test_near, feat),
        predict_wide(f_idw, test_idw, feat),
        W_IDW,
    )
    test_pred = apply_v04_calibrations(
        test_pred, test_idw, ws_mult, slot_mult, winter_only=True
    )

    print("[7/7] Save...")
    out = SUBMISSION_DIR / "v06_g3_winter_calib.csv"
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    sub = load_submission_template().drop(columns=GROUP_COLUMNS)
    sub.merge(test_pred, on="forecast_kst_dtm", how="left").to_csv(
        out, index=False, encoding="utf-8-sig"
    )

    projected = cal_scores["score"] + LB_OFFSET
    print(f"Saved: {out}")
    print("\n--- v0.6 summary ---")
    print(f"v0.3 LB     : 0.611")
    print(f"v05b local  : 0.633")
    print(f"v0.6 local  : {cal_scores['score']:.6f}")
    print(f"LB 예상     : ~{projected:.3f}")
    print(f"q_weights   : {q_weights}")
    print(f"ws_mult     : {ws_mult}")
    print(f"slot_mult   : {slot_mult}")
    print(
        f"제출 메모: v0.6 | group3 강화 + 겨울슬롯 + 풍속보정 | "
        f"로컬 {cal_scores['score']:.3f}, LB 예상 ~{projected:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-start", default=VALID_START)
    run(valid_start=parser.parse_args().valid_start)
