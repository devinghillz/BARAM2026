"""
심층 EDA v2 — 오차 케이스 분석 (그룹/월/풍속/풍향/과소·과대).

사용법:
  python scripts/deep_eda_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, OUTPUT_DIR, VALID_START
from src.data_loader import load_labels
from src.metrics import FICR_TIER_1, FICR_TIER_2
from src.power_curve import build_scada_monthly_curve
from scripts.eval_local_all import build_predictions

GEN_FLOOR = 0.10


def analyze_errors(long_df: pd.DataFrame, pred_col: str = "pred_kwh") -> dict:
    """long format: forecast_kst_dtm, group_id, power_kwh, pred_kwh, features."""
    df = long_df.copy()
    gid_to_col = {1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"}
    df["capacity"] = df["group_id"].map({g: GROUP_CAPACITY_KWH[c] for g, c in zip([1, 2, 3], GROUP_COLUMNS)})
    df["actual"] = df["power_kwh"]
    df["pred"] = df[pred_col]
    df["util"] = df["actual"] / df["capacity"]
    active = df[df["util"] >= GEN_FLOOR].copy()
    active["err_rate"] = (active["pred"] - active["actual"]).abs() / active["capacity"]
    active["err_signed"] = (active["pred"] - active["actual"]) / active["capacity"]
    active["under"] = active["pred"] < active["actual"]
    active["tier"] = np.select(
        [active["err_rate"] <= FICR_TIER_1, active["err_rate"] <= FICR_TIER_2],
        ["le_6", "6_8"],
        default="gt_8",
    )
    active["month"] = active["forecast_kst_dtm"].dt.month
    active["hour"] = active["forecast_kst_dtm"].dt.hour

    if "ldaps_ws_hub_blend" not in active.columns and "ldaps_ws10" in active.columns:
        active["ldaps_ws_hub_blend"] = active["ldaps_ws10"]
    active["ws_bin"] = active["ldaps_ws_hub_blend"].fillna(0).round().astype(int)

    def agg_slice(sub, name):
        if len(sub) == 0:
            return None
        return {
            "name": name,
            "n": int(len(sub)),
            "mean_err": float(sub["err_rate"].mean()),
            "gt_8_share": float((sub["tier"] == "gt_8").mean()),
            "under_share": float(sub["under"].mean()),
            "mean_util": float(sub["util"].mean()),
        }

    report = {
        "n_active": int(len(active)),
        "overall": agg_slice(active, "all"),
        "by_group": [],
        "by_month": [],
        "by_hour_worst": [],
        "by_ws_bin": [],
        "by_util_bin": [],
        "under_vs_over": {},
        "sweet_spot": {},
        "ldaps_gfs_diff": {},
        "actionable_cases": [],
    }

    for gid in [1, 2, 3]:
        sub = active[active["group_id"] == gid]
        report["by_group"].append(agg_slice(sub, f"group_{gid}"))

    for m in range(1, 13):
        report["by_month"].append(agg_slice(active[active["month"] == m], f"month_{m}"))

    hour_stats = []
    for h in range(24):
        s = active[active["hour"] == h]
        if len(s) > 0:
            hour_stats.append((h, (s["tier"] == "gt_8").mean(), s["under"].mean()))
    hour_stats.sort(key=lambda x: -x[1])
    report["by_hour_worst"] = [
        {"hour": h, "gt_8_share": float(g), "under_share": float(u)} for h, g, u in hour_stats[:8]
    ]

    for ws in sorted(active["ws_bin"].unique()):
        if 0 <= ws <= 15:
            report["by_ws_bin"].append(agg_slice(active[active["ws_bin"] == ws], f"ws_{ws}"))

    util_bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    for lo, hi in util_bins:
        sub = active[(active["util"] >= lo) & (active["util"] < hi)]
        report["by_util_bin"].append(agg_slice(sub, f"util_{lo}_{hi}"))

    for tier in ["le_6", "6_8", "gt_8"]:
        sub = active[active["tier"] == tier]
        report["under_vs_over"][tier] = {
            "under_share": float(sub["under"].mean()) if len(sub) else 0,
            "n": int(len(sub)),
        }

    sweet = active[(active["ldaps_ws_hub_blend"] >= 5) & (active["ldaps_ws_hub_blend"] <= 12)]
    report["sweet_spot"] = agg_slice(sweet, "ws_5_12")

    if "ws10_diff_ldaps_gfs" in active.columns:
        for label, cond in [
            ("ldaps_higher", active["ws10_diff_ldaps_gfs"] > 1.0),
            ("gfs_higher", active["ws10_diff_ldaps_gfs"] < -1.0),
            ("similar", active["ws10_diff_ldaps_gfs"].abs() <= 1.0),
        ]:
            report["ldaps_gfs_diff"][label] = agg_slice(active[cond], label)

    # 과소예측 비중 높은 케이스
    cases = []
    for gid in [1, 2, 3]:
        for m in [1, 2, 3, 12]:
            sub = active[(active["group_id"] == gid) & (active["month"] == m)]
            if len(sub) < 30:
                continue
            cases.append({
                "case": f"g{gid}_m{m}",
                "n": len(sub),
                "gt_8": float((sub["tier"] == "gt_8").mean()),
                "under": float(sub["under"].mean()),
                "mean_util": float(sub["util"].mean()),
            })
    cases.sort(key=lambda x: -x["gt_8"])
    report["worst_cases_top10"] = cases[:10]

    # 권장 방향
    report["actionable_cases"] = [
        "풍속 5~12m/s + 과소예측 비중 높음 → sweet-spot 조건부 상향 보정",
        "group_3 + 겨울(1~3월) gt_8 최고 → g3 겨울 전용 보정 (학습 분리보다 후처리)",
        "이용률 30~70% 구간 FICR 민감 → 경계 가중 학습 (q=0.8 유지, α=0.58)",
        "LDAPS>GFS 차이 클 때 오차 패턴 상이 → ws10_diff 피처 강화",
        "10·20·22시 전환대 오차 집중 → 시간대 슬롯 보정 (약하게)",
        "v05b 보정 과적합(LB갭 -0.018) → mild ws/slot 또는 v03 블렌드",
    ]

    return report


def main():
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)

    print("예측 생성 (v05b 베이스)...")
    valid_true, preds = build_predictions(labels, scada, vs)
    v05b = preds["v05b"]

    from scripts.train_v03 import build_dataset
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    valid_long = train_idw[train_idw["forecast_kst_dtm"] >= vs].dropna(subset=["power_kwh"])

    wide = v05b.set_index("forecast_kst_dtm")
    rows = []
    for gid in [1, 2, 3]:
        col = f"kpx_group_{gid}"
        part = valid_long[valid_long["group_id"] == gid].copy()
        part["pred_kwh"] = part["forecast_kst_dtm"].map(wide[col])
        rows.append(part)
    merged = pd.concat(rows, ignore_index=True)

    print("오차 케이스 분석...")
    report = analyze_errors(merged)

    # 라벨 기초 통계
    lab = labels.copy()
    lab["year"] = lab["kst_dtm"].dt.year
    yearly = {}
    for col in GROUP_COLUMNS:
        for y in lab["year"].unique():
            sub = lab[lab["year"] == y][col].dropna()
            cap = GROUP_CAPACITY_KWH[col]
            if len(sub) == 0:
                continue
            yearly[f"{col}_{y}"] = {
                "mean_util": float((sub / cap).mean()),
                "active_ratio": float((sub >= cap * 0.1).mean()),
            }
    report["yearly_label_stats"] = yearly

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "deep_eda_v2_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {out}")
    print("\n=== 핵심 요약 ===")
    print(f"평가 대상: {report['n_active']:,} 시간")
    g = report["by_group"]
    for x in g:
        print(f"  {x['name']}: >8%={x['gt_8_share']:.1%}, 과소={x['under_share']:.1%}")
    print(f"  sweet-spot(5~12m/s): >8%={report['sweet_spot']['gt_8_share']:.1%}, 과소={report['sweet_spot']['under_share']:.1%}")
    print("\n권장 방향:")
    for a in report["actionable_cases"]:
        print(f"  - {a}")


if __name__ == "__main__":
    main()
