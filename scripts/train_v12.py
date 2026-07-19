"""
v12 — 그룹별 SCADA prior + 교차 피처 + v05b 후처리.

개선:
  - vestas/unison 터빈을 KPX 그룹에 맞게 합산한 SCADA 곡선
  - prior_util, worst_slot 교차, 풍향×풍속 등 피처
  - v05b와 5% 블렌드 옵션 (로컬 안정성)

사용법:
  python scripts/train_v12.py
  python scripts/train_v12.py --blend-v05b 0.05
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_v04_calibrations
from src.config import GROUP_COLUMNS, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template, load_weather
from src.features import (
    aggregate_weather_to_groups,
    build_group_frame,
    get_feature_columns,
    merge_weather_frames,
)
from src.metrics import evaluate_submission
from src.power_curve import build_group_scada_curves, build_group_type_curves, build_scada_monthly_curve
from scripts.train_v03 import blend_wide, build_dataset, predict_wide
from scripts.train_v04 import train_all_groups_qmap
from src.split_utils import valid_split_frames

LB_OFFSET = -0.018
Q_WEIGHTS = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW = 0.9
WS_MULT = {1: 1.10, 2: 1.08, 3: 1.0}
SLOT_MULT = {1: 1.06, 2: 1.04, 3: 1.06}


def build_dataset_v12(labels, method, clim_before, split, enhanced=True):
    group_curve = build_group_scada_curves()
    type_curves = build_group_type_curves() if enhanced else None
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
        scada_curve=group_curve,
        type_curves=type_curves,
        enhanced=enhanced,
    )


def predict_v12(labels, enhanced=True, full_train=False):
    vs = pd.Timestamp(VALID_START)
    train_near = build_dataset_v12(labels, "nearest", vs, "train", enhanced=enhanced)
    train_idw = build_dataset_v12(labels, "idw", vs, "train", enhanced=enhanced)
    test_near = build_dataset_v12(labels, "nearest", None, "test", enhanced=enhanced)
    test_idw = build_dataset_v12(labels, "idw", None, "test", enhanced=enhanced)
    feat = get_feature_columns(train_idw)

    if full_train:
        fit_near, fit_idw = train_near, train_idw
        valid = train_idw.iloc[0:0]
    else:
        fit_near = train_near[train_near["forecast_kst_dtm"] < vs]
        fit_idw = train_idw[train_idw["forecast_kst_dtm"] < vs]
        valid = train_idw[train_idw["forecast_kst_dtm"] >= vs]

    m_near = train_all_groups_qmap(fit_near, feat, Q_WEIGHTS)
    m_idw = train_all_groups_qmap(fit_idw, feat, Q_WEIGHTS)

    val_pred = None
    if len(valid) > 0:
        _, _, near_val, idw_val = valid_split_frames(train_near, train_idw, vs)
        val_pred = blend_wide(
            predict_wide(m_near, near_val, feat),
            predict_wide(m_idw, idw_val, feat),
            W_IDW,
        )
        val_pred = apply_v04_calibrations(val_pred, idw_val, WS_MULT, SLOT_MULT)

    test_pred = blend_wide(
        predict_wide(m_near, test_near, feat),
        predict_wide(m_idw, test_idw, feat),
        W_IDW,
    )
    test_pred = apply_v04_calibrations(test_pred, test_idw, WS_MULT, SLOT_MULT)
    return val_pred, test_pred, train_idw


def predict_v05b_baseline(labels, full_train=False):
    """비교·블렌드용 v05b (레거시 prior)."""
    vs = pd.Timestamp(VALID_START)
    scada = build_scada_monthly_curve()
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)

    if full_train:
        fit_near, fit_idw = train_near, train_idw
        valid = train_idw.iloc[0:0]
    else:
        fit_near = train_near[train_near["forecast_kst_dtm"] < vs]
        fit_idw = train_idw[train_idw["forecast_kst_dtm"] < vs]
        valid = train_idw[train_idw["forecast_kst_dtm"] >= vs]

    m_near = train_all_groups_qmap(fit_near, feat, Q_WEIGHTS)
    m_idw = train_all_groups_qmap(fit_idw, feat, Q_WEIGHTS)

    val_pred = None
    if len(valid) > 0:
        _, _, near_val, idw_val = valid_split_frames(train_near, train_idw, vs)
        val_pred = blend_wide(
            predict_wide(m_near, near_val, feat),
            predict_wide(m_idw, idw_val, feat),
            W_IDW,
        )
        val_pred = apply_v04_calibrations(val_pred, idw_val, WS_MULT, SLOT_MULT)

    test_pred = blend_wide(
        predict_wide(m_near, test_near, feat),
        predict_wide(m_idw, test_idw, feat),
        W_IDW,
    )
    test_pred = apply_v04_calibrations(test_pred, test_idw, WS_MULT, SLOT_MULT)
    return val_pred, test_pred


def blend_predictions(pred_a, pred_b, w_b: float):
    out = pred_a.copy()
    for col in GROUP_COLUMNS:
        out[col] = (1 - w_b) * pred_a[col] + w_b * pred_b[col]
    return out


def save_submission(test_pred: pd.DataFrame, filename: str) -> Path:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSION_DIR / filename
    sub = load_submission_template().drop(columns=GROUP_COLUMNS)
    sub.merge(test_pred, on="forecast_kst_dtm", how="left").to_csv(
        out, index=False, encoding="utf-8-sig"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend-v05b", type=float, default=0.0, help="v05b 블렌드 비율 (0~0.15)")
    parser.add_argument("--no-enhanced", action="store_true", help="교차 피처 비활성")
    args = parser.parse_args()

    labels = load_labels()
    vs = pd.Timestamp(VALID_START)
    valid_true = labels.loc[labels["kst_dtm"] >= vs, ["kst_dtm", *GROUP_COLUMNS]]
    enhanced = not args.no_enhanced

    print("Train v12 (group SCADA + enhanced features)...")
    val_v12, test_v12, _ = predict_v12(labels, enhanced=enhanced, full_train=False)
    s_v12 = evaluate_submission(valid_true, val_v12, time_col="kst_dtm")
    print(f"  v12 local: {s_v12['score']:.6f}  NMAE={s_v12['1_minus_nmae']:.4f}  FICR={s_v12['ficr']:.4f}")

    test_out = test_v12
    tag = "v12_group_scada"
    if args.blend_v05b > 0:
        _, test_v05b = predict_v05b_baseline(labels, full_train=True)
        test_out = blend_predictions(test_v12, test_v05b, args.blend_v05b)
        tag = f"v12_blend{int(args.blend_v05b*100):02d}"

    path = save_submission(test_out, f"{tag}.csv")
    print(f"Saved: {path}")
    print(f"LB 예상 (v12 기준): ~{s_v12['score'] + LB_OFFSET:.3f}")


if __name__ == "__main__":
    main()
