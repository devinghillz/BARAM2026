"""
BARAM 2026 v0.3 — 모델 자체 개선

개선:
  - IDW 격자 집계 (EDA 상관 +0.72)
  - 파워커브 피처 (ws^2, ws^3, hub blend)
  - FICR 인식 sample weight (고발전 구간 가중)
  - Mean LGBM + Quantile(0.58) 앙상블 (과소예측 완화)
  - 검증 기반 quantile blend 비율 튜닝

사용법:
  python scripts/train_v03.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import GROUP_COLUMNS, GROUP_CAPACITY_KWH, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template, load_weather
from src.features import (
    aggregate_weather_to_groups,
    build_group_frame,
    get_feature_columns,
    merge_weather_frames,
)
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve

Q_WEIGHT_GRID = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]


def build_dataset(labels, method, clim_before, scada_curve, split):
    frames = []
    for source in ["ldaps", "gfs"]:
        weather = load_weather(source, split)
        frames.append(aggregate_weather_to_groups(weather, source=source, method=method))
    merged = merge_weather_frames(frames)
    return build_group_frame(
        merged,
        labels=labels if split == "train" else None,
        clim_labels=labels,
        clim_before=clim_before,
        scada_curve=scada_curve,
    )


def ficr_sample_weights(y: np.ndarray, capacity: float) -> np.ndarray:
    """고발전·FICR 민감 구간에 학습 가중치 부여."""
    util = y / capacity
    w = np.ones(len(y), dtype=float)
    active = util >= 0.10
    w[active] += 1.5 * util[active]
    w[util >= 0.40] += 1.0
    w[util >= 0.60] += 1.5
    return w


def make_lgbm_mean():
    import lightgbm as lgb

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                lgb.LGBMRegressor(
                    objective="regression",
                    n_estimators=900,
                    learning_rate=0.03,
                    num_leaves=63,
                    min_child_samples=25,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=0.2,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_lgbm_quantile(alpha: float = 0.58):
    import lightgbm as lgb

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    n_estimators=700,
                    learning_rate=0.035,
                    num_leaves=47,
                    min_child_samples=30,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_alpha=0.15,
                    reg_lambda=0.25,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                ),
            ),
        ]
    )


class GroupBlendModel:
    def __init__(self, q_weight: float = 0.35):
        self.q_weight = q_weight
        self.mean_model = make_lgbm_mean()
        self.q_model = make_lgbm_quantile()
        self.capacity = 21600.0

    def fit(self, X, y, capacity: float):
        self.capacity = capacity
        w = ficr_sample_weights(y.to_numpy(), capacity)
        self.mean_model.fit(X, y, regressor__sample_weight=w)
        self.q_model.fit(X, y, regressor__sample_weight=w)
        return self

    def predict(self, X) -> np.ndarray:
        p_mean = self.mean_model.predict(X)
        p_q = self.q_model.predict(X)
        pred = (1 - self.q_weight) * p_mean + self.q_weight * p_q
        return np.clip(pred, 0, self.capacity)


def _filter_group_train(df: pd.DataFrame, gid: int) -> pd.DataFrame:
    part = df[df["group_id"] == gid].dropna(subset=["power_kwh"])
    if gid == 3:
        part = part[part["forecast_kst_dtm"] >= "2023-01-01 01:00:00"]
    return part


def train_all_groups(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    q_weight: float,
) -> dict[int, GroupBlendModel]:
    models = {}
    for gid in [1, 2, 3]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        m = GroupBlendModel(q_weight=q_weight)
        m.fit(part[feature_cols], part["power_kwh"], cap)
        models[gid] = m
    return models


def predict_wide(models, infer_df, feature_cols) -> pd.DataFrame:
    rows = []
    for gid, model in models.items():
        part = infer_df[infer_df["group_id"] == gid].copy()
        part["pred_kwh"] = model.predict(part[feature_cols])
        rows.append(part)
    long_pred = pd.concat(rows, ignore_index=True)
    return (
        long_pred.pivot(index="forecast_kst_dtm", columns="group_id", values="pred_kwh")
        .rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
        .reset_index()
    )


def tune_q_weight(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    labels: pd.DataFrame,
    valid_start: pd.Timestamp,
) -> tuple[float, dict]:
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]]
    best_qw, best_scores = 0.35, {"score": -1.0}

    for i, qw in enumerate(Q_WEIGHT_GRID, start=1):
        print(f"    q_weight {qw:.2f} ({i}/{len(Q_WEIGHT_GRID)})...", flush=True)
        models = train_all_groups(train_df, feature_cols, qw)
        pred = predict_wide(models, valid_df, feature_cols)
        scores = evaluate_submission(valid_true, pred, time_col="kst_dtm")
        if scores["score"] > best_scores["score"]:
            best_scores = scores
            best_qw = qw

    return best_qw, best_scores


def tune_ensemble_weight(
    train_near: pd.DataFrame,
    train_idw: pd.DataFrame,
    valid_near: pd.DataFrame,
    valid_idw: pd.DataFrame,
    feature_cols: list[str],
    labels: pd.DataFrame,
    valid_start: pd.Timestamp,
    q_weight: float,
) -> tuple[float, dict]:
    """nearest vs IDW 예측 블렌드 비율 튜닝."""
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]]
    models_near = train_all_groups(train_near, feature_cols, q_weight)
    models_idw = train_all_groups(train_idw, feature_cols, q_weight)
    pred_near = predict_wide(models_near, valid_near, feature_cols)
    pred_idw = predict_wide(models_idw, valid_idw, feature_cols)

    best_w, best_scores = 0.6, {"score": -1.0}
    grid = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    for i, w_idw in enumerate(grid, start=1):
        print(f"    w_idw {w_idw:.1f} ({i}/{len(grid)})...", flush=True)
        merged = pred_near.copy()
        for col in GROUP_COLUMNS:
            merged[col] = (1 - w_idw) * pred_near[col] + w_idw * pred_idw[col]
        scores = evaluate_submission(valid_true, merged, time_col="kst_dtm")
        if scores["score"] > best_scores["score"]:
            best_scores = scores
            best_w = w_idw
    return best_w, best_scores, models_near, models_idw


def blend_wide(pred_a: pd.DataFrame, pred_b: pd.DataFrame, w_b: float) -> pd.DataFrame:
    out = pred_a.copy()
    for col in GROUP_COLUMNS:
        out[col] = (1 - w_b) * pred_a[col] + w_b * pred_b[col]
    return out


def run(valid_start: str, method: str = "idw") -> None:
    labels = load_labels()
    scada_curve = build_scada_monthly_curve()
    valid_start_ts = pd.Timestamp(valid_start)

    print("[1/6] Build datasets (nearest + IDW, power-curve features)...")
    train_near = build_dataset(labels, "nearest", valid_start_ts, scada_curve, "train")
    train_idw = build_dataset(labels, "idw", valid_start_ts, scada_curve, "train")
    test_near = build_dataset(labels, "nearest", None, scada_curve, "test")
    test_idw = build_dataset(labels, "idw", None, scada_curve, "test")
    feature_cols = get_feature_columns(train_idw)
    print(f"  features: {len(feature_cols)}")

    fit_mask = train_idw["forecast_kst_dtm"] < valid_start_ts
    valid_mask = train_idw["forecast_kst_dtm"] >= valid_start_ts

    print("[2/6] Tune quantile blend weight (IDW hold-out, ~2분)...")
    best_qw, _ = tune_q_weight(
        train_idw.loc[fit_mask],
        train_idw.loc[valid_mask],
        feature_cols,
        labels,
        valid_start_ts,
    )
    print(f"  best q_weight: {best_qw}")

    print("[3/6] Tune nearest/IDW ensemble weight (~3분)...")
    w_idw, val_scores, _, _ = tune_ensemble_weight(
        train_near.loc[fit_mask],
        train_idw.loc[fit_mask],
        train_near.loc[valid_mask],
        train_idw.loc[valid_mask],
        feature_cols,
        labels,
        valid_start_ts,
        best_qw,
    )
    print(f"  ensemble w_idw: {w_idw}")
    print(f"  1-NMAE : {val_scores['1_minus_nmae']:.6f}")
    print(f"  FICR   : {val_scores['ficr']:.6f}")
    print(f"  Score  : {val_scores['score']:.6f}")

    print("[4/6] Retrain on full data...")
    full_near = train_all_groups(train_near, feature_cols, best_qw)
    full_idw = train_all_groups(train_idw, feature_cols, best_qw)

    print("[5/6] Predict test...")
    test_pred = blend_wide(
        predict_wide(full_near, test_near, feature_cols),
        predict_wide(full_idw, test_idw, feature_cols),
        w_idw,
    )

    print("[6/6] Save submission...")
    submission = load_submission_template().drop(columns=GROUP_COLUMNS)
    submission = submission.merge(test_pred, on="forecast_kst_dtm", how="left")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSION_DIR / "v03_ficr_weighted_quantile_blend.csv"
    submission.to_csv(out, index=False, encoding="utf-8-sig")

    projected_lb = val_scores["score"] + 0.010
    print(f"Saved: {out}")
    print("\n--- v0.3 summary (2024 hold-out) ---")
    print(f"v0.2 LB        : 0.588")
    print(f"v0.3 local     : {val_scores['score']:.6f}")
    print(f"projected LB   : ~{projected_lb:.3f}")
    print(f"q_weight       : {best_qw}")
    print(f"w_idw          : {w_idw}")
    print(
        f"제출 메모: v0.3 | FICR 가중학습 + 분위수 앙상블 + nearest/IDW 혼합 | "
        f"로컬 {val_scores['score']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-start", default=VALID_START)
    parser.add_argument("--method", default="idw", choices=["idw", "nearest"])
    args = parser.parse_args()
    run(valid_start=args.valid_start, method=args.method)
