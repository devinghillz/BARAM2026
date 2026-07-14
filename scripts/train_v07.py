"""
BARAM 2026 v0.7 — group_3 전용 + FICR 경계 가중

전략:
  - group_1/2: v05b 동일 (q=0.6, ws×1.10/1.08, slot×1.06/1.04)
  - group_3: 전용 모델 (2023~, quantile α=0.60, 경계 가중)
  - group_3 q_weight·ws·slot만 검증 튜닝

사용법:
  python scripts/train_v07.py
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
    apply_slot_multipliers,
    apply_ws_band_multipliers,
    apply_v04_calibrations,
)
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

LB_OFFSET = -0.008
W_IDW = 0.9
Q_G12 = 0.6
Q3_GRID = [0.75, 0.80, 0.85, 0.90]
WS_G12 = {1: 1.10, 2: 1.08}
WS_G3_GRID = [1.0, 1.05, 1.08]
SLOT_G12 = {1: 1.06, 2: 1.04}
SLOT_G3_GRID = [1.0, 1.02, 1.04, 1.06]
G3_Q_ALPHA = 0.60


def ficr_boundary_weights_g3(
    y: np.ndarray,
    capacity: float,
    X: pd.DataFrame | None = None,
) -> np.ndarray:
    """FICR 6~8% 경계 민감 구간 + sweet-spot 가중 (group_3 전용)."""
    util = y / capacity
    w = ficr_sample_weights(y, capacity)
    boundary = (util >= 0.28) & (util <= 0.72)
    w[boundary] *= 1.5
    w[util >= 0.45] *= 1.15
    if X is not None and "in_ws_sweet_spot" in X.columns:
        sweet = X["in_ws_sweet_spot"].to_numpy(dtype=bool)
        w[sweet & boundary] *= 1.25
    return w


class GroupBlendModelG3:
    def __init__(self, q_weight: float, q_alpha: float = G3_Q_ALPHA):
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


def train_g12(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[int, GroupBlendModel]:
    models = {}
    for gid in [1, 2]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        m = GroupBlendModel(q_weight=Q_G12)
        m.fit(part[feature_cols], part["power_kwh"], cap)
        models[gid] = m
    return models


def train_g3(train_df: pd.DataFrame, feature_cols: list[str], q_weight: float) -> GroupBlendModelG3:
    part = _filter_group_train(train_df, 3)
    m = GroupBlendModelG3(q_weight=q_weight)
    m.fit(part[feature_cols], part["power_kwh"])
    return m


def predict_wide_split(
    models: dict[int, GroupBlendModel | GroupBlendModelG3],
    infer_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for gid in [1, 2, 3]:
        part = infer_df[infer_df["group_id"] == gid].copy()
        part["pred_kwh"] = models[gid].predict(part[feature_cols])
        rows.append(part)
    long_pred = pd.concat(rows, ignore_index=True)
    return (
        long_pred.pivot(index="forecast_kst_dtm", columns="group_id", values="pred_kwh")
        .rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
        .reset_index()
    )


def blend_near_idw(
    models_near: dict,
    models_idw: dict,
    valid_near: pd.DataFrame,
    valid_idw: pd.DataFrame,
    feat: list[str],
) -> pd.DataFrame:
    return blend_wide(
        predict_wide_split(models_near, valid_near, feat),
        predict_wide_split(models_idw, valid_idw, feat),
        W_IDW,
    )


def apply_calib_g12_then_tune_g3_ws(
    pred: pd.DataFrame,
    long_df: pd.DataFrame,
    valid_true: pd.DataFrame,
    ws_g3_grid: list[float],
) -> tuple[dict[int, float], dict]:
    base_ws = {**WS_G12, 3: 1.0}
    pred_g12 = apply_ws_band_multipliers(pred, long_df, base_ws)
    best_ws3 = 1.0
    best_s = evaluate_submission(valid_true, pred_g12, time_col="kst_dtm")

    for m3 in ws_g3_grid:
        trial = {**WS_G12, 3: m3}
        adj = apply_ws_band_multipliers(pred, long_df, trial)
        s = evaluate_submission(valid_true, adj, time_col="kst_dtm")
        print(f"    ws_g3={m3:.2f} -> Score {s['score']:.6f}", flush=True)
        if s["score"] > best_s["score"]:
            best_s = s
            best_ws3 = m3

    return {**WS_G12, 3: best_ws3}, best_s


def tune_slot_g3_only(
    pred: pd.DataFrame,
    valid_true: pd.DataFrame,
    slot_g12: dict[int, float],
) -> tuple[dict[int, float], dict]:
    best_g3 = 1.0
    best_s = {"score": -1.0}
    for m3 in SLOT_G3_GRID:
        trial = {**slot_g12, 3: m3}
        adj = apply_slot_multipliers(pred, trial)
        s = evaluate_submission(valid_true, adj, time_col="kst_dtm")
        print(f"    slot_g3={m3:.2f} -> Score {s['score']:.6f}", flush=True)
        if s["score"] > best_s["score"]:
            best_s = s
            best_g3 = m3
    return {**slot_g12, 3: best_g3}, best_s


def run(valid_start: str) -> None:
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(valid_start)

    print("[1/8] Build datasets...")
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

    print("[2/8] Train group_1/2 (v05b params)...")
    g12_near = train_g12(train_near.loc[fit_near], feat)
    g12_idw = train_g12(train_idw.loc[fit_idw], feat)

    print("[3/8] Tune group_3 quantile weight...")
    best_q3 = 0.80
    best_mid = {"score": -1.0}
    for q3 in Q3_GRID:
        print(f"    q3={q3:.2f}...", flush=True)
        models_near = {**g12_near, 3: train_g3(train_near.loc[fit_near], feat, q3)}
        models_idw = {**g12_idw, 3: train_g3(train_idw.loc[fit_idw], feat, q3)}
        pred = blend_near_idw(
            models_near, models_idw,
            train_near.loc[valid_near], train_idw.loc[valid_idw], feat,
        )
        s = evaluate_submission(valid_true, pred, time_col="kst_dtm")
        if s["score"] > best_mid["score"]:
            best_mid = s
            best_q3 = q3
    print(f"  best q3: {best_q3}  (raw Score {best_mid['score']:.6f})")

    print("[4/8] Retrain with best q3...")
    models_near = {**g12_near, 3: train_g3(train_near.loc[fit_near], feat, best_q3)}
    models_idw = {**g12_idw, 3: train_g3(train_idw.loc[fit_idw], feat, best_q3)}
    val_pred = blend_near_idw(
        models_near, models_idw,
        train_near.loc[valid_near], train_idw.loc[valid_idw], feat,
    )

    print("[5/8] Tune group_3 ws multiplier (g1/g2 fixed)...")
    ws_mult, ws_scores = apply_calib_g12_then_tune_g3_ws(
        val_pred, train_idw.loc[valid_idw], valid_true, WS_G3_GRID
    )
    val_ws = apply_ws_band_multipliers(val_pred, train_idw.loc[valid_idw], ws_mult)
    print(f"  ws_multipliers: {ws_mult}")

    print("[6/8] Tune group_3 slot (g1/g2 fixed)...")
    slot_mult, cal_scores = tune_slot_g3_only(val_ws, valid_true, SLOT_G12)
    print(f"  slot_multipliers: {slot_mult}")
    print(f"  1-NMAE : {cal_scores['1_minus_nmae']:.6f}")
    print(f"  FICR   : {cal_scores['ficr']:.6f}")
    print(f"  Score  : {cal_scores['score']:.6f}")

    print("[7/8] Retrain full + predict test...")
    full_near = {**train_g12(train_near, feat), 3: train_g3(train_near, feat, best_q3)}
    full_idw = {**train_g12(train_idw, feat), 3: train_g3(train_idw, feat, best_q3)}
    test_pred = blend_wide(
        predict_wide_split(full_near, test_near, feat),
        predict_wide_split(full_idw, test_idw, feat),
        W_IDW,
    )
    test_pred = apply_v04_calibrations(test_pred, test_idw, ws_mult, slot_mult)

    print("[8/8] Save...")
    out = SUBMISSION_DIR / "v07_g3_dedicated.csv"
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    sub = load_submission_template().drop(columns=GROUP_COLUMNS)
    sub.merge(test_pred, on="forecast_kst_dtm", how="left").to_csv(
        out, index=False, encoding="utf-8-sig"
    )

    projected = cal_scores["score"] + LB_OFFSET
    print(f"Saved: {out}")
    print("\n--- v0.7 summary ---")
    print(f"v05b local  : 0.633")
    print(f"v0.7 local  : {cal_scores['score']:.6f}")
    print(f"LB 예상     : ~{projected:.3f}")
    print(f"q3          : {best_q3}")
    print(f"ws_mult     : {ws_mult}")
    print(f"slot_mult   : {slot_mult}")
    print(
        f"제출 메모: v0.7 | group3 전용모델 + 경계가중 | "
        f"로컬 {cal_scores['score']:.3f}, LB 예상 ~{projected:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-start", default=VALID_START)
    run(valid_start=parser.parse_args().valid_start)
