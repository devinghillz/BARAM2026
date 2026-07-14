"""
BARAM 2026 v0.5 — FICR 3차 개선

v0.4 대비:
  - LDAPS 전용 브랜치 앙상블 (상관 0.72 풍속 신호 강화)
  - SCADA prior 조건부 풍속 보정 (과보정 완화)
  - 겨울·전환시간대(1~3·11월, 10·20·22시) 슬롯 보정
  - group_3 quantile alpha 상향 (0.62)
  - v0.4 q_weight 고정 + group_3만 미세 튜닝

사용법:
  python scripts/train_v05.py
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import (
    apply_all_calibrations,
    tune_slot_multipliers,
    tune_ws_band_multipliers_conditional,
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

LB_OFFSET = -0.008  # v0.3 실측: local - LB

# v0.4 최적값 기준
Q_BASE = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW = 0.9
Q_ALPHA = {1: 0.58, 2: 0.58, 3: 0.62}
Q3_GRID = [0.75, 0.80, 0.85]
W_LDAPS_GRID = [0.0, 0.15, 0.25, 0.35]


def ficr_sample_weights_v05(y: np.ndarray, capacity: float) -> np.ndarray:
    util = y / capacity
    w = ficr_sample_weights(y, capacity)
    boundary = (util >= 0.25) & (util <= 0.75)
    w[boundary] *= 1.25
    return w


class GroupBlendModelV05:
    def __init__(self, q_weight: float = 0.6, q_alpha: float = 0.58):
        self.q_weight = q_weight
        self.q_alpha = q_alpha
        self.mean_model = make_lgbm_mean()
        self.q_model = make_lgbm_quantile(alpha=q_alpha)
        self.capacity = 21600.0

    def fit(self, X, y, capacity: float):
        self.capacity = capacity
        w = ficr_sample_weights_v05(y.to_numpy(), capacity)
        self.mean_model.fit(X, y, regressor__sample_weight=w)
        self.q_model.fit(X, y, regressor__sample_weight=w)
        return self

    def predict(self, X) -> np.ndarray:
        p_mean = self.mean_model.predict(X)
        p_q = self.q_model.predict(X)
        pred = (1 - self.q_weight) * p_mean + self.q_weight * p_q
        return np.clip(pred, 0, self.capacity)


def train_groups(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    q_weights: dict[int, float],
    q_alphas: dict[int, float] | None = None,
) -> dict[int, GroupBlendModelV05]:
    if q_alphas is None:
        q_alphas = Q_ALPHA
    models = {}
    for gid in [1, 2, 3]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        m = GroupBlendModelV05(q_weight=q_weights[gid], q_alpha=q_alphas[gid])
        m.fit(part[feature_cols], part["power_kwh"], cap)
        models[gid] = m
    return models


def blend_ldaps_into(
    base: pd.DataFrame,
    ldaps: pd.DataFrame,
    w_ldaps: float,
) -> pd.DataFrame:
    out = base.copy()
    for col in GROUP_COLUMNS:
        out[col] = w_ldaps * ldaps[col] + (1.0 - w_ldaps) * base[col]
    return out


def tune_q3_only(
    train_near: pd.DataFrame,
    train_idw: pd.DataFrame,
    valid_near: pd.DataFrame,
    valid_idw: pd.DataFrame,
    feat_all: list[str],
    feat_ldaps: list[str],
    labels: pd.DataFrame,
    valid_start: pd.Timestamp,
    w_ldaps: float,
    w_idw: float,
) -> tuple[dict[int, float], dict]:
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]]
    best_q = dict(Q_BASE)
    best_scores = {"score": -1.0}

    for q3 in Q3_GRID:
        trial = dict(Q_BASE)
        trial[3] = q3
        m_near = train_groups(train_near, feat_all, trial)
        m_idw = train_groups(train_idw, feat_all, trial)
        m_ldaps = train_groups(train_idw, feat_ldaps, trial)
        base = blend_wide(
            predict_wide(m_near, valid_near, feat_all),
            predict_wide(m_idw, valid_idw, feat_all),
            w_idw,
        )
        ldaps = predict_wide(m_ldaps, valid_idw, feat_ldaps)
        pred = blend_ldaps_into(base, ldaps, w_ldaps)
        scores = evaluate_submission(valid_true, pred, time_col="kst_dtm")
        print(f"    q3={q3:.2f} -> Score {scores['score']:.6f}", flush=True)
        if scores["score"] > best_scores["score"]:
            best_scores = scores
            best_q = trial
    return best_q, best_scores


def tune_w_ldaps(
    train_near: pd.DataFrame,
    train_idw: pd.DataFrame,
    valid_near: pd.DataFrame,
    valid_idw: pd.DataFrame,
    feat_all: list[str],
    feat_ldaps: list[str],
    labels: pd.DataFrame,
    valid_start: pd.Timestamp,
    q_weights: dict[int, float],
    w_idw: float,
) -> tuple[float, dict]:
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]]
    m_near = train_groups(train_near, feat_all, q_weights)
    m_idw = train_groups(train_idw, feat_all, q_weights)
    m_ldaps = train_groups(train_idw, feat_ldaps, q_weights)
    base = blend_wide(
        predict_wide(m_near, valid_near, feat_all),
        predict_wide(m_idw, valid_idw, feat_all),
        w_idw,
    )
    ldaps = predict_wide(m_ldaps, valid_idw, feat_ldaps)

    best_w, best_scores = 0.25, {"score": -1.0}
    for w in W_LDAPS_GRID:
        pred = blend_ldaps_into(base, ldaps, w)
        scores = evaluate_submission(valid_true, pred, time_col="kst_dtm")
        print(f"    w_ldaps={w:.2f} -> Score {scores['score']:.6f}", flush=True)
        if scores["score"] > best_scores["score"]:
            best_scores = scores
            best_w = w
    return best_w, best_scores


def run(valid_start: str) -> None:
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(valid_start)

    print("[1/8] Build datasets...")
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat_all = get_feature_columns(train_idw)
    feat_ldaps = get_feature_columns(train_idw, prefixes=["ldaps"])
    print(f"  features: all={len(feat_all)}, ldaps={len(feat_ldaps)}")

    fit = train_idw["forecast_kst_dtm"] < vs
    valid = train_idw["forecast_kst_dtm"] >= vs
    valid_true = labels.loc[labels["kst_dtm"] >= vs, ["kst_dtm", *GROUP_COLUMNS]]

    print("[2/8] Tune LDAPS branch weight...")
    w_ldaps, _ = tune_w_ldaps(
        train_near.loc[fit], train_idw.loc[fit],
        train_near.loc[valid], train_idw.loc[valid],
        feat_all, feat_ldaps, labels, vs, Q_BASE, W_IDW,
    )
    print(f"  w_ldaps: {w_ldaps}")

    print("[3/8] Tune group_3 quantile weight...")
    q_weights, mid_scores = tune_q3_only(
        train_near.loc[fit], train_idw.loc[fit],
        train_near.loc[valid], train_idw.loc[valid],
        feat_all, feat_ldaps, labels, vs, w_ldaps, W_IDW,
    )
    print(f"  q_weights: {q_weights}")
    print(f"  mid Score: {mid_scores['score']:.6f}  FICR: {mid_scores['ficr']:.6f}")

    print("[4/8] Train for calibration...")
    m_near = train_groups(train_near.loc[fit], feat_all, q_weights)
    m_idw = train_groups(train_idw.loc[fit], feat_all, q_weights)
    m_ldaps = train_groups(train_idw.loc[fit], feat_ldaps, q_weights)
    val_base = blend_wide(
        predict_wide(m_near, train_near.loc[valid], feat_all),
        predict_wide(m_idw, train_idw.loc[valid], feat_all),
        W_IDW,
    )
    val_ldaps = predict_wide(m_ldaps, train_idw.loc[valid], feat_ldaps)
    val_pred = blend_ldaps_into(val_base, val_ldaps, w_ldaps)

    print("[5/8] Tune conditional ws-band multipliers...")
    ws_mult, ws_scores = tune_ws_band_multipliers_conditional(
        val_pred, train_idw.loc[valid], valid_true
    )
    val_ws = apply_all_calibrations(
        val_pred, train_idw.loc[valid], ws_mult, {1: 1.0, 2: 1.0, 3: 1.0}
    )
    print(f"  ws_multipliers: {ws_mult}")

    print("[6/8] Tune slot multipliers (겨울·10/20/22시)...")
    slot_mult, cal_scores = tune_slot_multipliers(val_ws, valid_true)
    print(f"  slot_multipliers: {slot_mult}")
    print(f"  1-NMAE : {cal_scores['1_minus_nmae']:.6f}")
    print(f"  FICR   : {cal_scores['ficr']:.6f}")
    print(f"  Score  : {cal_scores['score']:.6f}")

    print("[7/8] Retrain full + predict test...")
    f_near = train_groups(train_near, feat_all, q_weights)
    f_idw = train_groups(train_idw, feat_all, q_weights)
    f_ldaps = train_groups(train_idw, feat_ldaps, q_weights)
    test_base = blend_wide(
        predict_wide(f_near, test_near, feat_all),
        predict_wide(f_idw, test_idw, feat_all),
        W_IDW,
    )
    test_ldaps = predict_wide(f_ldaps, test_idw, feat_ldaps)
    test_pred = blend_ldaps_into(test_base, test_ldaps, w_ldaps)
    test_pred = apply_all_calibrations(test_pred, test_idw, ws_mult, slot_mult)

    print("[8/8] Save submission...")
    submission = load_submission_template().drop(columns=GROUP_COLUMNS)
    submission = submission.merge(test_pred, on="forecast_kst_dtm", how="left")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSION_DIR / "v05_ldaps_slot_calib.csv"
    submission.to_csv(out, index=False, encoding="utf-8-sig")

    projected = cal_scores["score"] + LB_OFFSET
    print(f"Saved: {out}")
    print("\n--- v0.5 summary (2024 hold-out) ---")
    print(f"v0.3 LB        : 0.611")
    print(f"v0.4 local     : 0.633 (미제출)")
    print(f"v0.5 local     : {cal_scores['score']:.6f}")
    print(f"LB 예상        : ~{projected:.3f}  (offset {LB_OFFSET:+.3f})")
    print(f"w_ldaps        : {w_ldaps}")
    print(f"q_weights      : {q_weights}")
    print(f"ws_multipliers : {ws_mult}")
    print(f"slot_multipliers: {slot_mult}")
    print(
        f"제출 메모: v0.5 | LDAPS브랜치 + 조건부풍속보정 + 슬롯보정 | "
        f"로컬 {cal_scores['score']:.3f}, LB 예상 ~{projected:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-start", default=VALID_START)
    run(valid_start=parser.parse_args().valid_start)
