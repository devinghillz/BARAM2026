"""
BARAM 2026 v0.2 파이프라인

개선 사항:
  1) FICR 의식 후처리 (climatology shrink)
  2) IDW 격자 집계
  3) SCADA 월별 파워커브 prior + climatology 피처
  4) LDAPS/GFS/Combined LGBM 앙상블 + 하이퍼파라미터 튜닝

사용법:
  python scripts/train_v02.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
from src.postprocess import apply_ficr_postprocess, shrink_toward_climatology, tune_shrink_on_validation
from src.power_curve import build_month_hour_climatology, build_scada_monthly_curve


def make_lgbm(tuned: bool = True):
    import lightgbm as lgb

    params = dict(
        objective="regression",
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    if tuned:
        params.update(
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=47,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=0.1,
        )
    else:
        params.update(n_estimators=300, learning_rate=0.05, num_leaves=31)

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", lgb.LGBMRegressor(**params)),
        ]
    )


def build_dataset(
    labels: pd.DataFrame,
    method: str,
    clim_before: pd.Timestamp | None,
    scada_curve: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    weather_frames = []
    for source in ["ldaps", "gfs"]:
        weather = load_weather(source, split)
        weather_frames.append(
            aggregate_weather_to_groups(weather, source=source, method=method)
        )
    merged = merge_weather_frames(weather_frames)
    label_arg = labels if split == "train" else None
    return build_group_frame(
        merged,
        labels=label_arg,
        clim_labels=labels,
        clim_before=clim_before,
        scada_curve=scada_curve,
    )


def train_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[int, Pipeline]:
    models = {}
    for gid in [1, 2, 3]:
        part = train_df[train_df["group_id"] == gid].dropna(subset=["power_kwh"])
        if gid == 3:
            part = part[part["forecast_kst_dtm"] >= "2023-01-01 01:00:00"]
        model = make_lgbm(tuned=True)
        model.fit(part[feature_cols], part["power_kwh"])
        models[gid] = model
    return models


def predict_wide(
    models: dict[int, Pipeline],
    infer_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for gid, model in models.items():
        part = infer_df[infer_df["group_id"] == gid].copy()
        part["pred_kwh"] = model.predict(part[feature_cols])
        rows.append(part)
    long_pred = pd.concat(rows, ignore_index=True)
    wide = (
        long_pred.pivot(index="forecast_kst_dtm", columns="group_id", values="pred_kwh")
        .rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
        .reset_index()
    )
    return wide


def clip_wide(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for col in GROUP_COLUMNS:
        out[col] = out[col].clip(0, GROUP_CAPACITY_KWH[col])
    return out


def _score_on_valid(pred_wide: pd.DataFrame, labels: pd.DataFrame, valid_start: pd.Timestamp) -> dict:
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]]
    return evaluate_submission(valid_true, pred_wide, time_col="kst_dtm")


def find_ensemble_weights(
    preds: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    valid_start: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, float]]:
    keys = list(preds.keys())
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]]
    best_w = {k: 1 / len(keys) for k in keys}
    best_scores = {"score": -1.0}
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    if len(keys) == 4:
        for w_a in weight_options:
            for w_b in weight_options:
                for w_c in weight_options:
                    w_d = 1.0 - w_a - w_b - w_c
                    if w_d < -1e-9 or w_d > 1.0:
                        continue
                    weights = {keys[0]: w_a, keys[1]: w_b, keys[2]: w_c, keys[3]: w_d}
                    merged = preds[keys[0]].copy()
                    for col in GROUP_COLUMNS:
                        merged[col] = sum(weights[k] * preds[k][col].values for k in keys)
                    merged = clip_wide(merged)
                    scores = evaluate_submission(valid_true, merged, time_col="kst_dtm")
                    if scores["score"] > best_scores["score"]:
                        best_scores = scores
                        best_w = weights
    else:
        grid = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 1.0]
        for w_ldaps in grid:
            for w_gfs in grid:
                w_comb = 1.0 - w_ldaps - w_gfs
                if w_comb < -1e-9:
                    continue
                weights = {"ldaps": w_ldaps, "gfs": w_gfs, "both": max(w_comb, 0.0)}
                merged = preds["both"].copy()
                for col in GROUP_COLUMNS:
                    merged[col] = (
                        weights["ldaps"] * preds["ldaps"][col].values
                        + weights["gfs"] * preds["gfs"][col].values
                        + weights["both"] * preds["both"][col].values
                    )
                merged = clip_wide(merged)
                scores = evaluate_submission(valid_true, merged, time_col="kst_dtm")
                if scores["score"] > best_scores["score"]:
                    best_scores = scores
                    best_w = weights

    return best_w, best_scores


def blend_predictions(preds: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    base_key = next(iter(preds))
    out = preds[base_key].copy()
    for col in GROUP_COLUMNS:
        out[col] = sum(weights.get(k, 0.0) * preds[k][col].values for k in preds)
    return clip_wide(out)


def run(valid_start: str, method: str = "idw") -> None:
    labels = load_labels()
    scada_curve = build_scada_monthly_curve()
    valid_start_ts = pd.Timestamp(valid_start)

    print("[1/6] Build datasets (nearest base + IDW ensemble member)...")
    train_near = build_dataset(labels, "nearest", clim_before=valid_start_ts, scada_curve=scada_curve, split="train")
    test_near = build_dataset(labels, "nearest", clim_before=None, scada_curve=scada_curve, split="test")
    train_idw = build_dataset(labels, "idw", clim_before=valid_start_ts, scada_curve=scada_curve, split="train")
    test_idw = build_dataset(labels, "idw", clim_before=None, scada_curve=scada_curve, split="test")

    train_df = train_near
    test_df = test_near

    feature_all = get_feature_columns(train_df)
    feature_ldaps = get_feature_columns(train_df, prefixes=["ldaps"])
    feature_gfs = get_feature_columns(train_df, prefixes=["gfs"])

    fit_mask = train_df["forecast_kst_dtm"] < valid_start_ts
    valid_mask = train_df["forecast_kst_dtm"] >= valid_start_ts
    idw_fit_mask = train_idw["forecast_kst_dtm"] < valid_start_ts
    idw_valid_mask = train_idw["forecast_kst_dtm"] >= valid_start_ts

    print(f"  features: all={len(feature_all)}, ldaps={len(feature_ldaps)}, gfs={len(feature_gfs)}")

    print("[2/6] Train ensemble members (LDAPS / GFS / Combined / IDW-Combined)...")
    model_sets = {
        "ldaps": (feature_ldaps, train_models(train_df.loc[fit_mask], feature_ldaps)),
        "gfs": (feature_gfs, train_models(train_df.loc[fit_mask], feature_gfs)),
        "both": (feature_all, train_models(train_df.loc[fit_mask], feature_all)),
        "idw": (
            feature_all,
            train_models(train_idw.loc[idw_fit_mask], feature_all),
        ),
    }

    print("[3/6] Validation predictions + ensemble weight search...")
    valid_preds = {}
    for name, (cols, models) in model_sets.items():
        infer = train_idw.loc[idw_valid_mask] if name == "idw" else train_df.loc[valid_mask]
        valid_preds[name] = clip_wide(predict_wide(models, infer, cols))

    weights, ens_scores = find_ensemble_weights(valid_preds, labels, valid_start_ts)
    print(f"  ensemble weights: {weights}")
    print(f"  pre-postprocess Score: {ens_scores['score']:.6f} (NMAE {ens_scores['1_minus_nmae']:.4f}, FICR {ens_scores['ficr']:.4f})")

    ens_valid = blend_predictions(valid_preds, weights)

    print("[4/6] Tune FICR postprocess shrink...")
    best_shrink, shrink_scores = tune_shrink_on_validation(
        ens_valid,
        labels.loc[labels["kst_dtm"] >= valid_start_ts, ["kst_dtm", *GROUP_COLUMNS]],
        labels,
        valid_start_ts,
    )
    clim = build_month_hour_climatology(labels, before=valid_start_ts)
    final_valid = shrink_toward_climatology(ens_valid, clim, shrink=best_shrink)
    final_scores = _score_on_valid(final_valid, labels, valid_start_ts)
    print(f"  best shrink: {best_shrink}")
    print(f"  1-NMAE : {final_scores['1_minus_nmae']:.6f}")
    print(f"  FICR   : {final_scores['ficr']:.6f}")
    print(f"  Score  : {final_scores['score']:.6f}")

    print("[5/6] Retrain on full data...")
    full_sets = {
        "ldaps": (feature_ldaps, train_models(train_df, feature_ldaps)),
        "gfs": (feature_gfs, train_models(train_df, feature_gfs)),
        "both": (feature_all, train_models(train_df, feature_all)),
        "idw": (feature_all, train_models(train_idw, feature_all)),
    }
    test_preds = {}
    for name, (cols, models) in full_sets.items():
        infer = test_idw if name == "idw" else test_df
        test_preds[name] = clip_wide(predict_wide(models, infer, cols))

    ens_test = blend_predictions(test_preds, weights)
    clim_full = build_month_hour_climatology(labels, before=valid_start_ts)
    final_test, _ = apply_ficr_postprocess(ens_test, labels, valid_start_ts, shrink=best_shrink)
    # apply with fixed shrink + train climatology
    final_test = shrink_toward_climatology(ens_test, clim_full, shrink=best_shrink)

    print("[6/6] Save submission...")
    submission = load_submission_template().drop(columns=GROUP_COLUMNS)
    submission = submission.merge(final_test, on="forecast_kst_dtm", how="left")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSION_DIR / "v02_nearest_ensemble.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_path}")
    print("\n--- v0.2 summary ---")
    print(f"v0.1 Public LB : 0.582 (local 0.580)")
    print(f"v0.2 local     : {final_scores['score']:.6f} (shrink={best_shrink}, weights={weights})")
    print(f"제출 메모: v0.2 | IDW + SCADA/clim + LGBM ensemble + FICR shrink | local {final_scores['score']:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--valid-start", default=VALID_START)
    p.add_argument("--method", default="idw", choices=["idw", "nearest", "mean"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(valid_start=args.valid_start, method=args.method)
