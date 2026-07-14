"""
전 버전 로컬 hold-out 비교 검증 (2024).

사용법:
  python scripts/eval_local_all.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import WINTER_MONTHS, apply_v04_calibrations
from src.config import GROUP_COLUMNS, GROUP_CAPACITY_KWH, OUTPUT_DIR, VALID_START
from src.data_loader import load_labels
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.train_v03 import blend_wide, build_dataset, get_feature_columns, predict_wide, train_all_groups
from scripts.train_v04 import train_all_groups_qmap

LB_OFFSET = -0.008
GEN_FLOOR = 0.10


def band_stats(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    rows = []
    for col in GROUP_COLUMNS:
        cap = GROUP_CAPACITY_KWH[col]
        a = y_true[col].to_numpy(float)
        f = y_pred[col].to_numpy(float)
        m = a >= cap * GEN_FLOOR
        err = np.abs(f[m] - a[m]) / cap
        rows.append(
            {
                "group": col,
                "le_6pct": float((err <= 0.06).mean()),
                "gt_8pct": float((err > 0.08).mean()),
                "under_pct": float((f[m] < a[m]).mean()),
            }
        )
    df = pd.DataFrame(rows)
    return {
        "le_6pct": float(df["le_6pct"].mean()),
        "gt_8pct": float(df["gt_8pct"].mean()),
        "under_pct": float(df["under_pct"].mean()),
        "by_group": rows,
    }


def score_version(name, pred, valid_true) -> dict:
    s = evaluate_submission(valid_true, pred, time_col="kst_dtm")
    aligned = valid_true.set_index("kst_dtm")
    common = aligned.index.intersection(pred.set_index("forecast_kst_dtm").index)
    bands = band_stats(aligned.loc[common], pred.set_index("forecast_kst_dtm").loc[common])
    return {
        "version": name,
        "score": s["score"],
        "1_minus_nmae": s["1_minus_nmae"],
        "ficr": s["ficr"],
        "projected_lb": s["score"] + LB_OFFSET,
        **bands,
    }


def build_predictions(labels, scada, vs):
    valid_true = labels.loc[labels["kst_dtm"] >= vs, ["kst_dtm", *GROUP_COLUMNS]]

    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    feat = get_feature_columns(train_idw)
    fit_mask = train_idw["forecast_kst_dtm"] < vs
    valid_mask = train_idw["forecast_kst_dtm"] >= vs

    print("  학습 v0.3 (q=0.65, w_idw=0.8)...")
    m3n = train_all_groups(train_near.loc[fit_mask], feat, 0.65)
    m3i = train_all_groups(train_idw.loc[fit_mask], feat, 0.65)
    v03 = blend_wide(
        predict_wide(m3n, train_near.loc[valid_mask], feat),
        predict_wide(m3i, train_idw.loc[valid_mask], feat),
        0.8,
    )

    print("  학습 v0.4 (q={0.6,0.6,0.8}, w_idw=0.9)...")
    q4 = {1: 0.6, 2: 0.6, 3: 0.8}
    m4n = train_all_groups_qmap(train_near.loc[fit_mask], feat, q4)
    m4i = train_all_groups_qmap(train_idw.loc[fit_mask], feat, q4)
    v04_base = blend_wide(
        predict_wide(m4n, train_near.loc[valid_mask], feat),
        predict_wide(m4i, train_idw.loc[valid_mask], feat),
        0.9,
    )
    v04 = apply_v04_calibrations(
        v04_base, train_idw.loc[valid_mask], {1: 1.10, 2: 1.08, 3: 1.0}
    )

    print("  학습 v05b (v0.4 + 슬롯)...")
    v05b = apply_v04_calibrations(
        v04_base,
        train_idw.loc[valid_mask],
        {1: 1.10, 2: 1.08, 3: 1.0},
        slot_mult={1: 1.06, 2: 1.04, 3: 1.06},
        winter_only=False,
    )

    return valid_true, {"v0.3": v03, "v0.4": v04, "v05b": v05b, "v05b_raw": v04_base}


def main() -> None:
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)

    print("=" * 60)
    print("로컬 hold-out 검증 (2024-01-01 ~)")
    print("=" * 60)

    valid_true, preds = build_predictions(labels, scada, vs)
    results = [score_version(k, v, valid_true) for k, v in preds.items() if k != "v05b_raw"]

    print()
    print(f"{'버전':<8} {'Score':>8} {'1-NMAE':>8} {'FICR':>8} {'>8%':>7} {'LB예상':>8}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['version']:<8} {r['score']:8.4f} {r['1_minus_nmae']:8.4f} "
            f"{r['ficr']:8.4f} {r['gt_8pct']:6.1%} {r['projected_lb']:8.3f}"
        )

    print()
    print("그룹별 >8% 비율:")
    for r in results:
        gtxt = ", ".join(f"{g['group'].split('_')[-1]}:{g['gt_8pct']:.1%}" for g in r["by_group"])
        print(f"  {r['version']}: {gtxt}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "local_validation_compare.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
