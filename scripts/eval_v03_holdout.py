"""v0.3 hold-out 상세 검증 (제출 전 점수 확인용)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import GROUP_COLUMNS, GROUP_CAPACITY_KWH, VALID_START
from src.data_loader import load_labels
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.train_v03 import (
    blend_wide,
    build_dataset,
    get_feature_columns,
    predict_wide,
    train_all_groups,
)

GEN_FLOOR = 0.10


def band_stats(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in GROUP_COLUMNS:
        cap = GROUP_CAPACITY_KWH[col]
        actual = y_true[col].to_numpy(dtype=float)
        forecast = y_pred[col].to_numpy(dtype=float)
        mask = actual >= cap * GEN_FLOOR
        actual = actual[mask]
        forecast = forecast[mask]
        err = np.abs(forecast - actual) / cap
        rows.append(
            {
                "group": col,
                "n": int(mask.sum()),
                "le_6pct": float((err <= 0.06).mean()),
                "pct_6_8": float(((err > 0.06) & (err <= 0.08)).mean()),
                "gt_8pct": float((err > 0.08).mean()),
                "under_pct": float((forecast < actual).mean()),
                "mean_err": float(err.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    labels = load_labels()
    scada = build_scada_monthly_curve()
    valid_start = pd.Timestamp(VALID_START)

    print("Loading data...")
    train_near = build_dataset(labels, "nearest", valid_start, scada, "train")
    train_idw = build_dataset(labels, "idw", valid_start, scada, "train")
    feature_cols = get_feature_columns(train_idw)

    fit_mask = train_idw["forecast_kst_dtm"] < valid_start
    valid_mask = train_idw["forecast_kst_dtm"] >= valid_start
    valid_true = labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]]

    print("Training v0.3 (fixed q_weight=0.65, w_idw=0.8)...")
    models_near = train_all_groups(train_near.loc[fit_mask], feature_cols, 0.65)
    models_idw = train_all_groups(train_idw.loc[fit_mask], feature_cols, 0.65)
    pred_near = predict_wide(models_near, train_near.loc[valid_mask], feature_cols)
    pred_idw = predict_wide(models_idw, train_idw.loc[valid_mask], feature_cols)
    v03 = blend_wide(pred_near, pred_idw, 0.8)
    scores = evaluate_submission(valid_true, v03, time_col="kst_dtm")

    aligned_true, aligned_pred = valid_true.set_index("kst_dtm"), v03.set_index("forecast_kst_dtm")
    common = aligned_true.index.intersection(aligned_pred.index)
    bands = band_stats(aligned_true.loc[common], aligned_pred.loc[common])

    print()
    print("=" * 58)
    print("2024 HOLD-OUT 검증 (제출 전 로컬 점수)")
    print("=" * 58)
    print(f"검증 구간   : {valid_start} ~ {common.max()}")
    print(f"검증 행 수   : {len(common):,} 시간")
    print()
    print(f"Score       : {scores['score']:.6f}")
    print(f"1-NMAE      : {scores['1_minus_nmae']:.6f}")
    print(f"FICR        : {scores['ficr']:.6f}")
    print()
    print("FICR 오차 밴드 (평가 대상 시간만):")
    print(bands.to_string(index=False))
    avg = bands[["le_6pct", "pct_6_8", "gt_8pct", "under_pct"]].mean()
    print()
    print(
        f"전체 평균    ≤6%={avg['le_6pct']:.1%}  "
        f"6~8%={avg['pct_6_8']:.1%}  "
        f">8%={avg['gt_8pct']:.1%}  "
        f"과소예측={avg['under_pct']:.1%}"
    )
    print()
    print("Public LB 예상 (과거 v0.1/v0.2 오프셋 기준):")
    print(f"  보수적 (+0.008): {scores['score'] + 0.008:.3f}")
    print(f"  중앙값 (+0.010): {scores['score'] + 0.010:.3f}")
    print(f"  낙관적 (+0.012): {scores['score'] + 0.012:.3f}")
    print()
    print("비교 기준:")
    print("  v0.2 LB       : 0.588 (local 0.578)")
    print("  팀 최고 LB    : 0.608")
    print("  상위 10위 LB  : ~0.653")


if __name__ == "__main__":
    main()
