"""
베이스라인: 기상예보 -> KPX 그룹 집계 -> 그룹별 회귀 모델

사용법:
  python scripts/baseline_train_predict.py
  python scripts/baseline_train_predict.py --source both --model lgbm
  python scripts/baseline_train_predict.py --valid-start "2024-07-01 01:00:00"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import GROUP_COLUMNS, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template, load_weather
from src.features import (
    aggregate_weather_to_groups,
    build_group_frame,
    get_feature_columns,
)
from src.metrics import evaluate_submission


def _make_model(name: str):
    if name == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", Ridge(alpha=1.0)),
            ]
        )
    if name == "lgbm":
        import lightgbm as lgb

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    lgb.LGBMRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model: {name}")


def _build_frames(sources: list[str], method: str, labels):
    frames = []
    for source in sources:
        weather_train = load_weather(source, "train")
        weather_test = load_weather(source, "test")
        train_groups = aggregate_weather_to_groups(
            weather_train, source=source, method=method
        )
        test_groups = aggregate_weather_to_groups(
            weather_test, source=source, method=method
        )
        frames.append(
            (
                build_group_frame(train_groups, labels=labels),
                build_group_frame(test_groups, labels=None),
            )
        )

    if len(frames) == 1:
        return frames[0]

    train_base, test_base = frames[0]
    for train_add, test_add in frames[1:]:
        key = ["forecast_kst_dtm", "group_id", "data_available_kst_dtm"]
        train_base = train_base.merge(
            train_add.drop(columns=["power_kwh"], errors="ignore"),
            on=key,
            how="left",
            suffixes=("", "_dup"),
        )
        test_base = test_base.merge(
            test_add,
            on=key,
            how="left",
            suffixes=("", "_dup"),
        )
    return train_base, test_base


def clip_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    from src.config import GROUP_CAPACITY_KWH

    out = pred.copy()
    for col in GROUP_COLUMNS:
        out[col] = out[col].clip(lower=0, upper=GROUP_CAPACITY_KWH[col])
    return out


def train_group_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> dict[int, Pipeline]:
    models: dict[int, Pipeline] = {}
    for group_id in [1, 2, 3]:
        part = train_df[train_df["group_id"] == group_id].copy()
        part = part.dropna(subset=["power_kwh"])
        if group_id == 3:
            part = part[part["forecast_kst_dtm"] >= "2023-01-01 01:00:00"]

        model = _make_model(model_name)
        model.fit(part[feature_cols], part["power_kwh"])
        models[group_id] = model
    return models


def predict_group_models(
    models: dict[int, Pipeline],
    infer_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for group_id, model in models.items():
        part = infer_df[infer_df["group_id"] == group_id].copy()
        part["pred_kwh"] = model.predict(part[feature_cols])
        rows.append(part)
    long_pred = pd.concat(rows, ignore_index=True)

    wide = (
        long_pred.pivot(index="forecast_kst_dtm", columns="group_id", values="pred_kwh")
        .rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
        .reset_index()
    )
    return wide


def run(sources: list[str], valid_start: str, method: str, model_name: str) -> None:
    source_tag = "_".join(sources)
    labels = load_labels()

    print(f"[1/4] Build features (sources={sources}, method={method})...")
    train_frame, test_frame = _build_frames(sources, method, labels)
    feature_cols = get_feature_columns(train_frame)
    print(f"  features: {len(feature_cols)}")

    valid_start_ts = pd.Timestamp(valid_start)
    fit_mask = train_frame["forecast_kst_dtm"] < valid_start_ts
    valid_mask = train_frame["forecast_kst_dtm"] >= valid_start_ts

    print(f"[2/4] Train {model_name} (fit<{valid_start_ts}, valid>={valid_start_ts})...")
    models = train_group_models(train_frame.loc[fit_mask], feature_cols, model_name)

    print("[3/4] Local validation (공식 metric)...")
    valid_pred = clip_predictions(
        predict_group_models(models, train_frame.loc[valid_mask], feature_cols)
    )
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start_ts, ["kst_dtm", *GROUP_COLUMNS]]
    scores = evaluate_submission(valid_true, valid_pred, time_col="kst_dtm")
    print(f"  1-NMAE : {scores['1_minus_nmae']:.6f}")
    print(f"  FICR   : {scores['ficr']:.6f}")
    print(f"  Score  : {scores['score']:.6f}  (rows={scores['n_rows']})")

    print("[4/4] Retrain on full labels and predict test...")
    full_models = train_group_models(train_frame, feature_cols, model_name)
    test_pred = clip_predictions(
        predict_group_models(full_models, test_frame, feature_cols)
    )

    submission = load_submission_template()
    submission = submission.drop(columns=GROUP_COLUMNS).merge(
        test_pred, on="forecast_kst_dtm", how="left"
    )

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSION_DIR / f"baseline_{source_tag}_{model_name}_{method}.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved submission: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BARAM 2026 baseline pipeline")
    parser.add_argument(
        "--source",
        choices=["ldaps", "gfs", "both"],
        default="both",
        help="기상 소스 (both=LDAPS+GFS 병합)",
    )
    parser.add_argument("--model", choices=["ridge", "lgbm"], default="lgbm")
    parser.add_argument("--method", choices=["nearest", "mean"], default="nearest")
    parser.add_argument("--valid-start", default=VALID_START)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sources = ["ldaps", "gfs"] if args.source == "both" else [args.source]
    run(
        sources=sources,
        valid_start=args.valid_start,
        method=args.method,
        model_name=args.model,
    )
