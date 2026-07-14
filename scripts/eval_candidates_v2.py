"""
제출 후보 검증 v2 — eval_local_all 동일 학습 + 후처리 비교.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import (
    WINTER_MONTHS,
    WORST_HOURS,
    apply_slot_multipliers,
    apply_ws_band_multipliers,
    apply_ws_band_multipliers_conditional,
)
from src.config import OUTPUT_DIR, VALID_START
from src.data_loader import load_labels
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.eval_local_all import build_predictions
from scripts.train_v03 import build_dataset

LB_V03, LB_V05B = -0.008, -0.018
COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]


def blend(a, b, alpha):
    out = a.copy()
    for c in COLS:
        out[c] = alpha * a[c] + (1 - alpha) * b[c]
    return out


def sc(pred, truth, name, offset):
    s = evaluate_submission(truth, pred, time_col="kst_dtm")
    return dict(name=name, score=s["score"], nmae=s["1_minus_nmae"], ficr=s["ficr"], lb=s["score"] + offset)


def main():
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)
    truth = labels.loc[labels.kst_dtm >= vs, ["kst_dtm", *COLS]]

    print("=" * 60)
    print("학습 (eval_local_all 동일 파이프라인)")
    print("=" * 60)
    _, preds = build_predictions(labels, scada, vs)
    v03, v05b = preds["v0.3"], preds["v05b"]
    raw = preds["v05b_raw"]

    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    long_valid = train_idw[train_idw.forecast_kst_dtm >= vs]

    R = []
    R.append(sc(v03, truth, "A_v03", LB_V03))
    R.append(sc(raw, truth, "B_v05b_보정전", LB_V05B))
    R.append(sc(v05b, truth, "C_v05b_제출본★", LB_V05B))

    for a in [0.20, 0.30, 0.40, 0.50, 0.60]:
        off = a * LB_V03 + (1 - a) * LB_V05B
        R.append(sc(blend(v03, v05b, a), truth, f"D_blend_{int(a*100)}pct_v03+v05b", off))

    for a in [0.30, 0.40, 0.50]:
        off = a * LB_V03 + (1 - a) * LB_V05B
        R.append(sc(blend(v03, raw, a), truth, f"E_blend_{int(a*100)}pct_v03+raw", off))

    ws_sets = {
        "mild": {1: 1.08, 2: 1.06, 3: 1.0},
        "mild2": {1: 1.06, 2: 1.05, 3: 1.0},
        "full": {1: 1.10, 2: 1.08, 3: 1.0},
    }
    slot_sets = {
        "mild": {1: 1.04, 2: 1.03, 3: 1.04},
        "full": {1: 1.06, 2: 1.04, 3: 1.06},
    }
    for wn, ws in ws_sets.items():
        for sn, sl in slot_sets.items():
            p = apply_slot_multipliers(apply_ws_band_multipliers(raw, long_valid, ws), sl)
            R.append(sc(p, truth, f"F_{wn}_ws+{sn}_slot", LB_V05B))

    mild_cal = apply_slot_multipliers(
        apply_ws_band_multipliers(raw, long_valid, ws_sets["mild"]), slot_sets["mild"]
    )
    for a in [0.25, 0.35, 0.45]:
        off = a * LB_V03 + (1 - a) * LB_V05B
        R.append(sc(blend(v03, mild_cal, a), truth, f"G_blend_{int(a*100)}pct_v03+mild", off))

    for w in [(1.06, 1.05), (1.08, 1.06)]:
        ws = {1: w[0], 2: w[1], 3: 1.0}
        R.append(sc(apply_ws_band_multipliers_conditional(v03, long_valid, ws), truth, f"H_v03_cond_{w[0]}", LB_V03))
        R.append(sc(apply_ws_band_multipliers_conditional(raw, long_valid, ws), truth, f"H_raw_cond_{w[0]}", LB_V05B))

    winter = apply_slot_multipliers(
        apply_ws_band_multipliers(raw, long_valid, ws_sets["mild"]),
        slot_sets["mild"],
        worst_months=WINTER_MONTHS,
        worst_hours=WORST_HOURS,
    )
    R.append(sc(winter, truth, "I_winter_mild", LB_V05B))
    R.append(sc(blend(v03, winter, 0.35), truth, "J_blend35_v03+winter", 0.35 * LB_V03 + 0.65 * LB_V05B))

    by_local = sorted(R, key=lambda x: x["score"], reverse=True)
    by_lb = sorted(R, key=lambda x: x["lb"], reverse=True)

    print("\n" + "=" * 60)
    print("TOP 15 (로컬 Score)")
    print("=" * 60)
    print(f"{'후보':<34} {'Local':>7} {'NMAE':>7} {'FICR':>7} {'LB예상':>7}")
    print("-" * 60)
    for r in by_local[:15]:
        print(f"{r['name']:<34} {r['score']:7.4f} {r['nmae']:7.4f} {r['ficr']:7.4f} {r['lb']:7.4f}")

    # pick: best LB that's different strategy from #1
    p1 = by_lb[0]
    p2 = next((r for r in by_lb if r["name"] != p1["name"] and r["name"][0] != p1["name"][0]), by_lb[1])

    print("\n" + "=" * 60)
    print("★ 오늘 제출 추천 (2회)")
    print("=" * 60)
    for i, p in enumerate([p1, p2], 1):
        print(f"  [{i}] {p['name']}")
        print(f"      로컬 {p['score']:.4f} | NMAE {p['nmae']:.4f} | FICR {p['ficr']:.4f} | LB~{p['lb']:.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "submission_candidates_v2.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"all": R, "top": by_local[:15], "pick1": p1, "pick2": p2}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
