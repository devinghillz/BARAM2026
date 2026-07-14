"""
v1.1 — P1_08 이용률구간 보정 그리드 + 최적 조합 탐색.

사용법:
  python scripts/eval_v11_grid.py
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_slot_multipliers, apply_ws_band_multipliers
from src.config import GROUP_COLUMNS, OUTPUT_DIR, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.eval_local_all import build_predictions
from scripts.eval_v10_priorities import (
    LB_V05B,
    NON_WINTER_MONTHS,
    SLOT_MILD,
    WS_FULL,
    apply_util_tier_boost,
)
from scripts.train_v03 import blend_wide, build_dataset, get_feature_columns, predict_wide, train_all_groups
from scripts.train_v04 import train_all_groups_qmap

COLS = GROUP_COLUMNS
Q_V05 = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW = 0.9


def score(name, pred, truth, **extra) -> dict:
    s = evaluate_submission(truth, pred, time_col="kst_dtm")
    row = {
        "case": name,
        "local": round(s["score"], 6),
        "nmae": round(s["1_minus_nmae"], 6),
        "ficr": round(s["ficr"], 6),
    }
    row.update(extra)
    return row


def high_mult(scale: float) -> dict[int, float]:
    return {1: round(1.03 + scale, 4), 2: round(1.02 + scale, 4), 3: round(1.01 + scale, 4)}


def apply_pipeline(raw, long_df, high_s, mid_m, high_lo, slot=None, slot_months=None):
    base = apply_ws_band_multipliers(raw, long_df, WS_FULL)
    out = apply_util_tier_boost(
        base,
        high_mult=high_mult(high_s),
        mid_mult={1: mid_m, 2: mid_m, 3: 1.0},
        high_lo=high_lo,
    )
    if slot:
        out = apply_slot_multipliers(out, slot, worst_months=slot_months)
    return out


def save_sub(pred, fname):
    sub = load_submission_template().drop(columns=COLS)
    sub.merge(pred, on="forecast_kst_dtm", how="left").to_csv(
        SUBMISSION_DIR / fname, index=False, encoding="utf-8-sig"
    )


def main():
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)
    truth = labels.loc[labels.kst_dtm >= vs, ["kst_dtm", *COLS]]

    print("=" * 64)
    print("[1/3] 베이스 예측 (eval_local_all)")
    print("=" * 64)
    _, preds = build_predictions(labels, scada, vs)
    raw = preds["v05b_raw"]
    v05b = preds["v05b"]
    long_val = build_dataset(labels, "idw", vs, scada, "train")
    long_val = long_val[long_val.forecast_kst_dtm >= vs]

    R: list[dict] = []
    R.append(score("ref_v05b", v05b, truth))

    print("\n[2/3] P1_08 그리드 (high_scale x mid_mult x high_lo)")
    high_scales = [0.0, 0.01, 0.02, 0.03]
    mid_mults = [0.96, 0.98, 1.0]
    high_los = [0.60, 0.65, 0.70]

    for hs, mm, hl in itertools.product(high_scales, mid_mults, high_los):
        name = f"grid_hs{hs:.2f}_mm{mm:.2f}_hl{hl:.2f}"
        p = apply_pipeline(raw, long_val, hs, mm, hl)
        R.append(score(name, p, truth, high_s=hs, mid_m=mm, high_lo=hl, slot="none"))

    grid_only = [r for r in R if r["case"].startswith("grid_")]
    top_grid = sorted(grid_only, key=lambda x: x["local"], reverse=True)[:5]
    print(f"  그리드 {len(grid_only)}개 완료, 상위 5개 조합에 slot 변형 추가...")

    combos = []
    for g in top_grid:
        hs, mm, hl = g["high_s"], g["mid_m"], g["high_lo"]
        tag = f"hs{hs:.2f}_mm{mm:.2f}_hl{hl:.2f}"
        variants = [
            (f"best_{tag}", None, None, "none"),
            (f"best_{tag}_mildslot", SLOT_MILD, None, "mild"),
            (f"best_{tag}_mildslot_nowinter", SLOT_MILD, NON_WINTER_MONTHS, "mild_nowinter"),
            (f"best_{tag}_mildslot_winter_only", SLOT_MILD, {11}, "mild_w11"),
        ]
        for name, slot, months, st in variants:
            p = apply_pipeline(raw, long_val, hs, mm, hl, slot=slot, slot_months=months)
            combos.append(
                score(name, p, truth, parent=tag, high_s=hs, mid_m=mm, high_lo=hl, slot_type=st)
            )

    R.extend(combos)
    ranked = sorted(R, key=lambda x: x["local"], reverse=True)

    print("\n" + "=" * 64)
    print("[3/3] 결과 TOP 15")
    print("=" * 64)
    print(f"{'케이스':<42} {'Score':>7} {'NMAE':>7} {'FICR':>7}")
    print("-" * 68)
    for r in ranked[:15]:
        print(f"{r['case']:<42} {r['local']:7.4f} {r['nmae']:7.4f} {r['ficr']:7.4f}")

    best = ranked[0]
    ref = next(r["local"] for r in R if r["case"] == "ref_v05b")
    gap = LB_V05B - ref
    print(f"\n★ 1위: {best['case']}  Score={best['local']:.4f}  FICR={best['ficr']:.4f}")
    print(f"  v05b 대비: {best['local'] - ref:+.4f}")
    print(f"  LB 추정 (v05b 갭 {gap:+.4f}): {best['local'] + gap:.4f}")

    def build_test_pred(row: dict):
        hs, mm, hl = row["high_s"], row["mid_m"], row["high_lo"]
        slot = SLOT_MILD if row.get("slot_type", "none") in {"mild", "mild_nowinter", "mild_w11"} else None
        months = None
        st = row.get("slot_type", "none")
        if st == "mild_nowinter":
            months = NON_WINTER_MONTHS
        elif st == "mild_w11":
            months = {11}
        return apply_pipeline(raw_test, test_idw, hs, mm, hl, slot=slot, slot_months=months)

    print("\n테스트 CSV 생성...")
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)
    f5n = train_all_groups_qmap(train_near, feat, Q_V05)
    f5i = train_all_groups_qmap(train_idw, feat, Q_V05)
    raw_test = blend_wide(predict_wide(f5n, test_near, feat), predict_wide(f5i, test_idw, feat), W_IDW)

    SUBMISSION_DIR.mkdir(exist_ok=True)
    exported = []
    for r in ranked[:3]:
        case = r["case"]
        if case == "ref_v05b":
            from src.calibration import apply_v04_calibrations
            from scripts.eval_v10_priorities import SLOT_FULL

            pred = apply_v04_calibrations(raw_test, test_idw, WS_FULL, slot_mult=SLOT_FULL)
            fname = "v11_ref_v05b.csv"
        elif "high_s" in r:
            pred = build_test_pred(r)
            fname = f"v11_{case}.csv"
        else:
            continue
        save_sub(pred, fname)
        exported.append(fname)
        print(f"  -> submissions/{fname}")

    out = OUTPUT_DIR / "v11_grid_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ranked_top20": ranked[:20],
                "best": best,
                "top_grid_params": top_grid,
                "gap_v05b": round(gap, 4),
                "exports": exported,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
