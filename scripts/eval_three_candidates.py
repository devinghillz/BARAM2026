"""
남은 제출 후보 3종 + v05b 기준 로컬 검증.

  1. fullws + mild slot
  2. mild calib (ws + slot 모두 약화)
  3. v03 10% + v05b blend

사용법:
  python scripts/eval_three_candidates.py
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

from src.calibration import apply_slot_multipliers, apply_v04_calibrations, apply_ws_band_multipliers
from src.config import GROUP_COLUMNS, OUTPUT_DIR, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.eval_local_all import build_predictions
from scripts.train_v03 import blend_wide, build_dataset

COLS = GROUP_COLUMNS
WS_FULL = {1: 1.10, 2: 1.08, 3: 1.0}
WS_MILD = {1: 1.08, 2: 1.06, 3: 1.0}
SLOT_FULL = {1: 1.06, 2: 1.04, 3: 1.06}
SLOT_MILD = {1: 1.04, 2: 1.03, 3: 1.04}
LB = {"v05b": 0.6149, "v11": 0.6128, "v03": 0.6113}


def blend(a, b, alpha):
    out = a.copy()
    for c in COLS:
        out[c] = alpha * a[c] + (1 - alpha) * b[c]
    return out


def sc(name, pred, truth, lb_key=None):
    s = evaluate_submission(truth, pred, time_col="kst_dtm")
    row = {
        "case": name,
        "local": round(s["score"], 6),
        "nmae": round(s["1_minus_nmae"], 6),
        "ficr": round(s["ficr"], 6),
    }
    if lb_key and lb_key in LB:
        row["lb_actual"] = LB[lb_key]
        row["gap"] = round(row["local"] - LB[lb_key], 4)
    return row


def main():
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)
    truth = labels.loc[labels.kst_dtm >= vs, ["kst_dtm", *COLS]]

    print("=" * 62)
    print("베이스 학습 (eval_local_all)")
    print("=" * 62)
    _, preds = build_predictions(labels, scada, vs)
    v03, v05b, raw = preds["v0.3"], preds["v05b"], preds["v05b_raw"]
    long_val = build_dataset(labels, "idw", vs, scada, "train")
    long_val = long_val[long_val.forecast_kst_dtm >= vs]

    candidates = [
        ("ref_v05b", v05b, "v05b"),
        (
            "1_fullws_mildslot",
            apply_slot_multipliers(
                apply_ws_band_multipliers(raw, long_val, WS_FULL), SLOT_MILD
            ),
            None,
        ),
        (
            "2_mild_calib",
            apply_slot_multipliers(
                apply_ws_band_multipliers(raw, long_val, WS_MILD), SLOT_MILD
            ),
            None,
        ),
        ("3_blend10_v03", blend(v03, v05b, 0.10), None),
    ]

    R = [sc(name, p, truth, lb) for name, p, lb in candidates]
    ranked = sorted(R, key=lambda x: x["local"], reverse=True)

    print("\n" + "=" * 62)
    print("로컬 hold-out (2024) 결과")
    print("=" * 62)
    print(f"{'케이스':<22} {'Score':>7} {'NMAE':>7} {'FICR':>7} {'LB실측':>8} {'갭':>7}")
    print("-" * 62)
    for r in ranked:
        lb = f"{r['lb_actual']:.4f}" if "lb_actual" in r else "-"
        gap = f"{r['gap']:+.4f}" if "gap" in r else "-"
        print(f"{r['case']:<22} {r['local']:7.4f} {r['nmae']:7.4f} {r['ficr']:7.4f} {lb:>8} {gap:>7}")

    # v11 참고 (LB만)
    print(f"\n참고 v11 (LB 실측): Score=0.6128  NMAE=0.8456  FICR=0.3799  갭=-0.0219")

    # LB 갭 추정 (v05b 기준)
    gap_v05b = ranked[0]["local"] - LB["v05b"] if ranked[0]["case"] != "ref_v05b" else LB["v05b"] - LB["v05b"]
    ref = next(r for r in R if r["case"] == "ref_v05b")
    print("\n--- LB 추정 (v05b 갭 -0.0186 적용) ---")
    for r in ranked:
        if r["case"] == "ref_v05b":
            est = LB["v05b"]
        else:
            est = round(r["local"] - 0.0186, 4)
        print(f"  {r['case']:<22} LB~{est:.4f}  (로컬 {r['local']:.4f})")

    print("\n--- v03 갭(-0.0091) 기준 LB 추정 (과적합 적은 쪽) ---")
    for r in ranked:
        if r["case"] == "3_blend10_v03":
            est = round(r["local"] - 0.0091, 4)
            print(f"  {r['case']:<22} LB~{est:.4f}")

    best = ranked[0]
    # 추천: LB 갭 작은 mild/fullws vs 로컬 1위
    mild = next(r for r in R if r["case"] == "2_mild_calib")
    fullws = next(r for r in R if r["case"] == "1_fullws_mildslot")
    b10 = next(r for r in R if r["case"] == "3_blend10_v03")

    print("\n" + "=" * 62)
    print("제출 추천 요약")
    print("=" * 62)
    print(f"  로컬 1위: {best['case']} ({best['local']:.4f})")
    print(f"  v05b 대비 fullws_mildslot: {fullws['local'] - ref['local']:+.4f}")
    print(f"  v05b 대비 mild_calib:      {mild['local'] - ref['local']:+.4f}")
    print(f"  v05b 대비 blend10:         {b10['local'] - ref['local']:+.4f}")
    print("  v11 교훈: 로컬↑ != LB↑ → 갭 작은 mild/fullws 우선")

    out = OUTPUT_DIR / "three_candidates_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": R, "ranked": ranked, "ref_v05b": ref}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
