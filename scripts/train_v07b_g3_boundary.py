"""
BARAM 2026 v0.7b — group_3 경계 가중만 (구조는 v05b 유지)

v0.7 분리 모델이 g3 예측 악화 → v05b 골격 + g3 학습 가중치만 강화.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_v04_calibrations, tune_slot_multipliers
from src.config import GROUP_COLUMNS, GROUP_CAPACITY_KWH, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.train_v03 import (
    GroupBlendModel,
    blend_wide,
    build_dataset,
    ficr_sample_weights,
    get_feature_columns,
    make_lgbm_mean,
    make_lgbm_quantile,
    predict_wide,
    _filter_group_train,
)
from scripts.train_v07 import ficr_boundary_weights_g3

LB_OFFSET = -0.008
Q_WEIGHTS = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW = 0.9
WS_MULT = {1: 1.10, 2: 1.08, 3: 1.0}


class GroupBlendModelG3Boundary:
    """v05b q=0.8 유지 + 경계 가중 + quantile α=0.60."""

    def __init__(self, q_weight: float = 0.8, q_alpha: float = 0.60):
        self.q_weight = q_weight
        self.mean_model = make_lgbm_mean()
        self.q_model = make_lgbm_quantile(alpha=q_alpha)
        self.capacity = GROUP_CAPACITY_KWH["kpx_group_3"]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        w = ficr_boundary_weights_g3(y.to_numpy(), self.capacity, X)
        self.mean_model.fit(X, y, regressor__sample_weight=w)
        self.q_model.fit(X, y, regressor__sample_weight=w)
        return self

    def predict(self, X) -> np.ndarray:
        pred = (1 - self.q_weight) * self.mean_model.predict(X) + self.q_weight * self.q_model.predict(X)
        return np.clip(pred, 0, self.capacity)


def train_groups_v07b(train_df: pd.DataFrame, feat: list[str]) -> dict:
    models = {}
    for gid in [1, 2]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        m = GroupBlendModel(q_weight=Q_WEIGHTS[gid])
        m.fit(part[feat], part["power_kwh"], cap)
        models[gid] = m
    part3 = _filter_group_train(train_df, 3)
    m3 = GroupBlendModelG3Boundary(q_weight=Q_WEIGHTS[3])
    m3.fit(part3[feat], part3["power_kwh"])
    models[3] = m3
    return models


def predict_wide_mixed(models, infer_df, feat):
    rows = []
    for gid, m in models.items():
        p = infer_df[infer_df.group_id == gid].copy()
        p["pred_kwh"] = m.predict(p[feat])
        rows.append(p)
    long = pd.concat(rows, ignore_index=True)
    return (
        long.pivot(index="forecast_kst_dtm", columns="group_id", values="pred_kwh")
        .rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
        .reset_index()
    )


def run() -> None:
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)

    print("[1/5] Build datasets...")
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)
    fit_near = train_near["forecast_kst_dtm"] < vs
    fit_idw = train_idw["forecast_kst_dtm"] < vs
    valid_near = train_near["forecast_kst_dtm"] >= vs
    valid_idw = train_idw["forecast_kst_dtm"] >= vs
    valid_true = labels.loc[labels["kst_dtm"] >= vs, ["kst_dtm", *GROUP_COLUMNS]]

    print("[2/5] Train (g1/g2 표준, g3 경계가중)...")
    m_near = train_groups_v07b(train_near.loc[fit_near], feat)
    m_idw = train_groups_v07b(train_idw.loc[fit_idw], feat)
    val_pred = blend_wide(
        predict_wide_mixed(m_near, train_near.loc[valid_near], feat),
        predict_wide_mixed(m_idw, train_idw.loc[valid_idw], feat),
        W_IDW,
    )
    val_pred = apply_v04_calibrations(val_pred, train_idw.loc[valid_idw], WS_MULT)
    base_s = evaluate_submission(valid_true, val_pred, time_col="kst_dtm")
    print(f"  pre-slot Score: {base_s['score']:.6f}  FICR: {base_s['ficr']:.6f}")

    print("[3/5] Tune slot...")
    slot_mult, scores = tune_slot_multipliers(val_pred, valid_true)
    print(f"  slot: {slot_mult}")
    print(f"  Score: {scores['score']:.6f}  FICR: {scores['ficr']:.6f}")

    print("[4/5] Full retrain + test...")
    f_near = train_groups_v07b(train_near, feat)
    f_idw = train_groups_v07b(train_idw, feat)
    test_pred = blend_wide(
        predict_wide_mixed(f_near, test_near, feat),
        predict_wide_mixed(f_idw, test_idw, feat),
        W_IDW,
    )
    test_pred = apply_v04_calibrations(test_pred, test_idw, WS_MULT, slot_mult)

    print("[5/5] Save...")
    out = SUBMISSION_DIR / "v07b_g3_boundary.csv"
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    sub = load_submission_template().drop(columns=GROUP_COLUMNS)
    sub.merge(test_pred, on="forecast_kst_dtm", how="left").to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Saved: {out}")
    print(f"\n--- v0.7b summary ---")
    print(f"v05b local  : 0.633")
    print(f"v0.7b local : {scores['score']:.6f}")
    print(f"LB 예상     : ~{scores['score'] + LB_OFFSET:.3f}")
    print(f"slot_mult   : {slot_mult}")
    print(
        f"제출 메모: v0.7b | g3 FICR경계가중 + v05b구조 | "
        f"로컬 {scores['score']:.3f}"
    )


if __name__ == "__main__":
    run()
