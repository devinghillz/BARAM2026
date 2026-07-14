"""v0.3 score projection before implementation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, OUTPUT_DIR, VALID_START
from src.metrics import FICR_TIER_1, FICR_TIER_2, metric
from src.power_curve import build_month_hour_climatology

# Reuse step3 pipeline
from scripts.run_eda_step3 import build_error_detail, build_validation_frame, train_and_predict


def compute_group_metrics(detail: pd.DataFrame) -> dict:
    rows = []
    for col in GROUP_COLUMNS:
        gid = int(col.split("_")[-1])
        part = detail[detail["group_col"] == col]
        answer = part[["actual_kwh"]].rename(columns={"actual_kwh": col})
        pred = part[["pred_kwh"]].rename(columns={"pred_kwh": col})
        if len(part) == 0:
            continue
        total, nmae, ficr = metric(answer, pred)
        rows.append({"group": col, "1_minus_nmae": nmae, "ficr": ficr, "score": total})
    df = pd.DataFrame(rows)
    return {
        "1_minus_nmae": float(df["1_minus_nmae"].mean()),
        "ficr": float(df["ficr"].mean()),
        "score": float(df["score"].mean()),
    }


def apply_clim_shrink(detail: pd.DataFrame, labels: pd.DataFrame, shrink: float, before: pd.Timestamp) -> pd.DataFrame:
    clim = build_month_hour_climatology(labels, before=before)
    out = detail.drop(columns=["clim_power_kwh"], errors="ignore").merge(
        clim, on=["group_id", "month", "hour"], how="left"
    )
    cap_map = {int(c.split("_")[-1]): GROUP_CAPACITY_KWH[c] for c in GROUP_COLUMNS}
    new_pred = []
    for _, row in out.iterrows():
        cap = cap_map[row["group_id"]]
        climv = row["clim_power_kwh"] if pd.notna(row["clim_power_kwh"]) else row["actual_kwh"]
        p = (1 - shrink) * row["pred_kwh"] + shrink * climv
        new_pred.append(np.clip(p, 0, cap))
    out["pred_kwh"] = new_pred
    return _refresh_error_rate(out)


def _refresh_error_rate(out: pd.DataFrame) -> pd.DataFrame:
    cap_series = out["group_col"].map({c: GROUP_CAPACITY_KWH[c] for c in GROUP_COLUMNS})
    out["error_rate"] = (out["pred_kwh"] - out["actual_kwh"]).abs() / cap_series
    return out


def apply_under_bias_fix(detail: pd.DataFrame, factor: float, ws_range: tuple[float, float] | None = None) -> pd.DataFrame:
    out = detail.copy()
    mask = out["pred_kwh"] < out["actual_kwh"]
    if ws_range:
        lo, hi = ws_range
        ws = out["blend_ws10"].fillna(out["ldaps_ws10"])
        mask &= (ws >= lo) & (ws <= hi)
    out.loc[mask, "pred_kwh"] *= factor
    for col in GROUP_COLUMNS:
        cap = GROUP_CAPACITY_KWH[col]
        gmask = out["group_col"] == col
        out.loc[gmask, "pred_kwh"] = out.loc[gmask, "pred_kwh"].clip(0, cap)
    return _refresh_error_rate(out)


def apply_slot_shrink(detail: pd.DataFrame, labels: pd.DataFrame, slots: list[tuple[int, int]], shrink: float, before: pd.Timestamp):
    clim = build_month_hour_climatology(labels, before=before)
    out = detail.drop(columns=["clim_power_kwh"], errors="ignore").merge(
        clim, on=["group_id", "month", "hour"], how="left"
    )
    cap_map = {int(c.split("_")[-1]): GROUP_CAPACITY_KWH[c] for c in GROUP_COLUMNS}
    preds = []
    for _, row in out.iterrows():
        cap = cap_map[row["group_id"]]
        p = row["pred_kwh"]
        if (int(row["month"]), int(row["hour"])) in slots:
            climv = row["clim_power_kwh"] if pd.notna(row["clim_power_kwh"]) else row["actual_kwh"]
            p = (1 - shrink) * p + shrink * climv
        preds.append(np.clip(p, 0, cap))
    out["pred_kwh"] = preds
    return _refresh_error_rate(out)


def tier_stats(detail: pd.DataFrame) -> dict:
    er = detail["error_rate"]
    return {
        "le_6pct": float((er <= FICR_TIER_1).mean()),
        "pct_6_8": float(((er > FICR_TIER_1) & (er <= FICR_TIER_2)).mean()),
        "gt_8pct": float((er > FICR_TIER_2).mean()),
        "mean_error_rate": float(er.mean()),
    }


def score_from_detail(detail: pd.DataFrame) -> dict:
    wide_a, wide_p = [], []
    for ts, part in detail.groupby("forecast_kst_dtm"):
        row_a = {"forecast_kst_dtm": ts}
        row_p = {"forecast_kst_dtm": ts}
        for _, r in part.iterrows():
            row_a[r["group_col"]] = r["actual_kwh"]
            row_p[r["group_col"]] = r["pred_kwh"]
        wide_a.append(row_a)
        wide_p.append(row_p)
    answer = pd.DataFrame(wide_a).sort_values("forecast_kst_dtm")
    pred = pd.DataFrame(wide_p).sort_values("forecast_kst_dtm")
    total, nmae, ficr = metric(answer[GROUP_COLUMNS], pred[GROUP_COLUMNS])
    return {"score": total, "1_minus_nmae": nmae, "ficr": ficr, **tier_stats(detail)}


def main():
    valid_start = pd.Timestamp(VALID_START)
    labels = pd.read_csv(ROOT / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])

    print("Building 2024 hold-out predictions...")
    fit, valid = build_validation_frame(labels, valid_start)
    valid_pred = train_and_predict(fit, valid)
    base_detail = build_error_detail(valid_pred)
    base = score_from_detail(base_detail)

    scenarios = {"v0.2_baseline (현재)": base_detail}

    # Scenario A: global clim shrink
    for shrink in [0.05, 0.10, 0.15, 0.20]:
        scenarios[f"A_clim_shrink_{int(shrink*100)}pct"] = apply_clim_shrink(
            base_detail, labels, shrink=shrink, before=valid_start
        )

    # Scenario B: under-prediction fix mid-high wind
    for factor in [1.05, 1.10, 1.15]:
        scenarios[f"B_under_fix_ws5_12_x{factor}"] = apply_under_bias_fix(
            base_detail, factor=factor, ws_range=(5, 12)
        )

    # Scenario C: worst month-hour slots targeted
    worst_slots = [(1, 21), (1, 20), (2, 8), (3, 11), (7, 20), (10, 10), (8, 22)]
    scenarios["C_worst_slots_shrink_15pct"] = apply_slot_shrink(
        base_detail, labels, worst_slots, shrink=0.15, before=valid_start
    )

    # Scenario D: combined best guess
    d = apply_under_bias_fix(base_detail, factor=1.10, ws_range=(5, 12))
    d = apply_clim_shrink(d, labels, shrink=0.10, before=valid_start)
    d = apply_slot_shrink(d, labels, worst_slots, shrink=0.10, before=valid_start)
    scenarios["D_combined_v03_estimate"] = d

    # Scenario E: optimistic - move 15% of gt8 hours into tier2 (theoretical ceiling)
    e = base_detail.copy()
    gt8_idx = e[e["error_rate"] > FICR_TIER_2].index
    n_move = int(len(gt8_idx) * 0.15)
    rng = np.random.default_rng(42)
    move_idx = rng.choice(gt8_idx, size=n_move, replace=False)
    for idx in move_idx:
        row = e.loc[idx]
        cap = GROUP_CAPACITY_KWH[row["group_col"]]
        # pull prediction toward actual to land at 7% error
        target_err = 0.07
        direction = 1 if row["pred_kwh"] < row["actual_kwh"] else -1
        e.loc[idx, "pred_kwh"] = row["actual_kwh"] + direction * target_err * cap
        e.loc[idx, "pred_kwh"] = np.clip(e.loc[idx, "pred_kwh"], 0, cap)
    e["error_rate"] = (e["pred_kwh"] - e["actual_kwh"]).abs() / e["group_col"].map(
        {c: GROUP_CAPACITY_KWH[c] for c in GROUP_COLUMNS}
    )
    scenarios["E_theoretical_15pct_gt8_fixed"] = _refresh_error_rate(e)

    results = []
    for name, detail in scenarios.items():
        s = score_from_detail(detail)
        results.append({"scenario": name, **s})

    res_df = pd.DataFrame(results).sort_values("score", ascending=False)

    # LB calibration: local 0.578 -> LB 0.588 for v0.2 (offset +0.010)
    lb_offset = 0.010
    res_df["projected_lb"] = res_df["score"] + lb_offset

    report = {
        "baseline_local": base,
        "lb_calibration_note": "v0.2 local 0.578 -> Public LB 0.588 (+0.010 offset applied)",
        "current_lb_reference": {"v0.1": 0.582, "v0.2": 0.588},
        "top10_target": {"score": 0.653, "ficr": 0.43, "nmae": 0.873},
        "scenarios": results,
        "ranked": res_df.to_dict("records"),
    }

    out = OUTPUT_DIR / "v03_score_projection.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== v0.3 Score Projection (2024 hold-out) ===\n")
    print(res_df[["scenario", "score", "projected_lb", "1_minus_nmae", "ficr", "gt_8pct"]].to_string(index=False))
    print(f"\nSaved: {out}")
    return report


if __name__ == "__main__":
    main()
