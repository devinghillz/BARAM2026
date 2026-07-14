"""
v0.4 최적 파라미터 + 슬롯 보정만 추가 (빠른 검증용).

v0.5 LDAPS/조건부 보정이 v0.4 대비 하락 → v0.4 골격 유지 + 슬롯만 탐색.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_slot_multipliers, apply_ws_band_multipliers, tune_slot_multipliers
from src.config import GROUP_COLUMNS, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.train_v04 import train_all_groups_qmap
from scripts.train_v03 import blend_wide, build_dataset, get_feature_columns, predict_wide

LB_OFFSET = -0.008
Q_WEIGHTS = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW = 0.9
WS_MULT = {1: 1.10, 2: 1.08, 3: 1.0}


def run() -> None:
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)

    print("Build + train (v0.4 params)...")
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)
    fit = train_idw["forecast_kst_dtm"] < vs
    valid = train_idw["forecast_kst_dtm"] >= vs
    valid_true = labels.loc[labels["kst_dtm"] >= vs, ["kst_dtm", *GROUP_COLUMNS]]

    m_near = train_all_groups_qmap(train_near.loc[fit], feat, Q_WEIGHTS)
    m_idw = train_all_groups_qmap(train_idw.loc[fit], feat, Q_WEIGHTS)
    val_pred = blend_wide(
        predict_wide(m_near, train_near.loc[valid], feat),
        predict_wide(m_idw, train_idw.loc[valid], feat),
        W_IDW,
    )
    val_pred = apply_ws_band_multipliers(val_pred, train_idw.loc[valid], WS_MULT)
    print(f"v0.4 replay Score: {evaluate_submission(valid_true, val_pred, time_col='kst_dtm')['score']:.6f}")

    slot_mult, scores = tune_slot_multipliers(val_pred, valid_true)
    print(f"slot_multipliers: {slot_mult}")
    print(f"v0.4+slot Score: {scores['score']:.6f}  FICR: {scores['ficr']:.6f}")

    f_near = train_all_groups_qmap(train_near, feat, Q_WEIGHTS)
    f_idw = train_all_groups_qmap(train_idw, feat, Q_WEIGHTS)
    test_pred = blend_wide(
        predict_wide(f_near, test_near, feat),
        predict_wide(f_idw, test_idw, feat),
        W_IDW,
    )
    test_pred = apply_ws_band_multipliers(test_pred, test_idw, WS_MULT)
    test_pred = apply_slot_multipliers(test_pred, slot_mult)

    out = SUBMISSION_DIR / "v05b_v04_slot.csv"
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    sub = load_submission_template().drop(columns=GROUP_COLUMNS)
    sub.merge(test_pred, on="forecast_kst_dtm", how="left").to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved: {out}")
    print(f"LB 예상: ~{scores['score'] + LB_OFFSET:.3f}")


if __name__ == "__main__":
    run()
