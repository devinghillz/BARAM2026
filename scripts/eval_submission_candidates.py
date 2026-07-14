"""
제출 전 후보 일괄 로컬 검증 (2024 hold-out).

v03+v05b 블렌드, 완화 보정, 조건부 FICR 후처리 등을 비교.

사용법:
  python scripts/eval_submission_candidates.py
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
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
from src.config import GROUP_COLUMNS, GROUP_CAPACITY_KWH, OUTPUT_DIR, VALID_START
from src.data_loader import load_labels
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.train_v03 import blend_wide, build_dataset, get_feature_columns, predict_wide, train_all_groups
from scripts.train_v04 import train_all_groups_qmap

LB_OFFSET_V03 = -0.008
LB_OFFSET_V05B = -0.018


def blend_preds(a: pd.DataFrame, b: pd.DataFrame, alpha_a: float) -> pd.DataFrame:
    out = a.copy()
    for col in GROUP_COLUMNS:
        out[col] = alpha_a * a[col] + (1 - alpha_a) * b[col]
    return out


def score_pred(pred: pd.DataFrame, valid_true: pd.DataFrame, name: str, offset: float) -> dict:
    s = evaluate_submission(valid_true, pred, time_col="kst_dtm")
    return {
        "name": name,
        "score": s["score"],
        "1_minus_nmae": s["1_minus_nmae"],
        "ficr": s["ficr"],
        "projected_lb": s["score"] + offset,
    }


def main() -> None:
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)
    valid_true = labels.loc[labels["kst_dtm"] >= vs, ["kst_dtm", *GROUP_COLUMNS]]

    print("=" * 65)
    print("데이터 로드 + 모델 학습 (v03, v05b 베이스)...")
    print("=" * 65)

    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    feat = get_feature_columns(train_idw)
    fit_near = train_near["forecast_kst_dtm"] < vs
    fit_idw = train_idw["forecast_kst_dtm"] < vs
    valid_near = train_near["forecast_kst_dtm"] >= vs
    valid_idw = train_idw["forecast_kst_dtm"] >= vs
    long_valid = train_idw.loc[valid_idw]

    # v03 base
    print("  v03 학습...")
    m3n = train_all_groups(train_near.loc[fit_near], feat, 0.65)
    m3i = train_all_groups(train_idw.loc[fit_idw], feat, 0.65)
    v03 = blend_wide(
        predict_wide(m3n, train_near.loc[valid_near], feat),
        predict_wide(m3i, long_valid, feat),
        0.8,
    )

    # v05b base (보정 전)
    print("  v05b 학습...")
    q5 = {1: 0.6, 2: 0.6, 3: 0.8}
    m5n = train_all_groups_qmap(train_near.loc[fit_near], feat, q5)
    m5i = train_all_groups_qmap(train_idw.loc[fit_idw], feat, q5)
    v05b_raw = blend_wide(
        predict_wide(m5n, train_near.loc[valid_near], feat),
        predict_wide(m5i, long_valid, feat),
        0.9,
    )

    results: list[dict] = []

    # --- 기준선 ---
    results.append(score_pred(v03, valid_true, "A_v03_기준(LB0.611)", LB_OFFSET_V03))
    results.append(score_pred(v05b_raw, valid_true, "B_v05b_보정전", LB_OFFSET_V05B))

    v05b_full = apply_slot_multipliers(
        apply_ws_band_multipliers(v05b_raw, long_valid, {1: 1.10, 2: 1.08, 3: 1.0}),
        {1: 1.06, 2: 1.04, 3: 1.06},
    )
    results.append(score_pred(v05b_full, valid_true, "C_v05b_풀보정(제출본)", LB_OFFSET_V05B))

    # --- v03+v05b 블렌드 ---
    print("  블렌드 탐색...")
    for alpha in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        blended = blend_preds(v03, v05b_full, alpha)
        offset = alpha * LB_OFFSET_V03 + (1 - alpha) * LB_OFFSET_V05B
        results.append(
            score_pred(blended, valid_true, f"D_blend_v03_{alpha:.0%}_v05b", offset)
        )

    # 블렌드 + raw v05b (보정 전)
    for alpha in [0.30, 0.50]:
        blended = blend_preds(v03, v05b_raw, alpha)
        offset = alpha * LB_OFFSET_V03 + (1 - alpha) * LB_OFFSET_V05B
        results.append(
            score_pred(blended, valid_true, f"E_blend_v03_{alpha:.0%}_v05b_raw", offset)
        )

    # --- 완화 ws/slot ---
    print("  완화 보정 탐색...")
    ws_mild_opts = [
        ({1: 1.06, 2: 1.05, 3: 1.0}, "mild1"),
        ({1: 1.08, 2: 1.06, 3: 1.0}, "mild2"),
        ({1: 1.05, 2: 1.04, 3: 1.0}, "mild3"),
    ]
    slot_mild_opts = [
        ({1: 1.03, 2: 1.02, 3: 1.03}, "s1"),
        ({1: 1.04, 2: 1.03, 3: 1.04}, "s2"),
        ({1: 1.06, 2: 1.04, 3: 1.06}, "s_full"),
    ]
    for (ws, wsname), (slot, sname) in product(ws_mild_opts, slot_mild_opts):
        pred = apply_slot_multipliers(
            apply_ws_band_multipliers(v05b_raw, long_valid, ws), slot
        )
        results.append(
            score_pred(pred, valid_true, f"F_v05b_{wsname}_{sname}", LB_OFFSET_V05B)
        )

    # --- v03 + 완화 보정 ---
    for ws, wsname in ws_mild_opts:
        boosted = apply_ws_band_multipliers(v03, long_valid, ws)
        results.append(
            score_pred(boosted, valid_true, f"G_v03_ws_{wsname}", LB_OFFSET_V03)
        )

    # --- 조건부 보정 (SCADA prior) ---
    print("  조건부 보정 탐색...")
    for base_name, base_pred, offset in [
        ("v03", v03, LB_OFFSET_V03),
        ("v05b_raw", v05b_raw, LB_OFFSET_V05B),
    ]:
        for mult_g in [(1.05, 1.05, 1.0), (1.08, 1.06, 1.0)]:
            ws = {1: mult_g[0], 2: mult_g[1], 3: mult_g[2]}
            pred = apply_ws_band_multipliers_conditional(
                base_pred, long_valid, ws, prior_ratio=0.92
            )
            results.append(
                score_pred(
                    pred, valid_true,
                    f"H_{base_name}_cond_ws_{mult_g[0]}_{mult_g[1]}",
                    offset,
                )
            )

    # --- 블렌드 + mild 보정 최적 조합 ---
    best_mild = max(
        [r for r in results if r["name"].startswith("F_v05b_mild2")],
        key=lambda x: x["score"],
    )
    for alpha in [0.35, 0.45, 0.55]:
        mild_pred = apply_slot_multipliers(
            apply_ws_band_multipliers(v05b_raw, long_valid, {1: 1.08, 2: 1.06, 3: 1.0}),
            {1: 1.04, 2: 1.03, 3: 1.04},
        )
        blended = blend_preds(v03, mild_pred, alpha)
        offset = alpha * LB_OFFSET_V03 + (1 - alpha) * LB_OFFSET_V05B
        results.append(
            score_pred(blended, valid_true, f"I_blend_mild_{alpha:.0%}", offset)
        )

    # --- 겨울 슬롯만 (1~3월) ---
    for tag, base_pred, offset in [
        ("v05b_raw", v05b_raw, LB_OFFSET_V05B),
        ("v03", v03, LB_OFFSET_V03),
    ]:
        pred = apply_ws_band_multipliers(
            base_pred, long_valid, {1: 1.08, 2: 1.06, 3: 1.0},
        )
        pred = apply_slot_multipliers(
            pred, {1: 1.04, 2: 1.03, 3: 1.04},
            worst_months=WINTER_MONTHS,
            worst_hours=WORST_HOURS,
        )
        results.append(
            score_pred(pred, valid_true, f"J_{tag}_winter_slot", offset)
        )

    # 정렬
    by_local = sorted(results, key=lambda x: x["score"], reverse=True)
    by_lb = sorted(results, key=lambda x: x["projected_lb"], reverse=True)

    print()
    print("=" * 65)
    print("TOP 10 (로컬 Score)")
    print("=" * 65)
    print(f"{'후보':<35} {'Local':>7} {'NMAE':>7} {'FICR':>7} {'LB예상':>7}")
    print("-" * 65)
    for r in by_local[:10]:
        print(
            f"{r['name']:<35} {r['score']:7.4f} {r['1_minus_nmae']:7.4f} "
            f"{r['ficr']:7.4f} {r['projected_lb']:7.4f}"
        )

    print()
    print("=" * 65)
    print("TOP 10 (LB 예상)")
    print("=" * 65)
    for r in by_lb[:10]:
        print(
            f"{r['name']:<35} {r['score']:7.4f} {r['1_minus_nmae']:7.4f} "
            f"{r['ficr']:7.4f} {r['projected_lb']:7.4f}"
        )

    # 제출 추천
    print()
    print("=" * 65)
    print("제출 추천 (2회)")
    print("=" * 65)

    # LB 예상 top 중 v05b 변형과 blend 다양하게
    rec = []
    seen_types = set()
    for r in by_lb:
        t = r["name"][0]
        if t not in seen_types or len(rec) < 2:
            rec.append(r)
            seen_types.add(t)
        if len(rec) >= 5:
            break

    # 수동 추천 로직: 1) LB예상 1위, 2) 다른 전략 1위
    pick1 = by_lb[0]
    pick2 = None
    for r in by_lb[1:]:
        if not r["name"].startswith(pick1["name"][:2]):
            pick2 = r
            break
    if pick2 is None:
        pick2 = by_lb[1]

    print(f"  [1순위] {pick1['name']}")
    print(f"         로컬 {pick1['score']:.4f} | FICR {pick1['ficr']:.4f} | NMAE {pick1['1_minus_nmae']:.4f} | LB~{pick1['projected_lb']:.4f}")
    print(f"  [2순위] {pick2['name']}")
    print(f"         로컬 {pick2['score']:.4f} | FICR {pick2['ficr']:.4f} | NMAE {pick2['1_minus_nmae']:.4f} | LB~{pick2['projected_lb']:.4f}")

    out = OUTPUT_DIR / "submission_candidates_ranked.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"all": results, "top_local": by_local[:10], "top_lb": by_lb[:10], "recommend": [pick1, pick2]}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
