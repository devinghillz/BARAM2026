"""FICR near-boundary failure analysis — testable core logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, VALID_START
from src.metrics import (
    FICR_TIER_1,
    FICR_TIER_2,
    GENERATION_FLOOR_RATIO,
    PRICE_TIER_1,
    PRICE_TIER_2,
    evaluate_submission,
    metric,
)
from scripts.eval_v13_comprehensive import SPLITS, apply_pipeline_fixed, build_split_bundle
from scripts.submission_diff_lib import V13_BEST_CONFIG, is_hotspot_row
from scripts.v13_factorial_lib import PERIOD_DEFS

FICR_DEFINITION = {
    "error_rate_formula": "|forecast - actual| / capacity",
    "active_threshold": f"actual >= {GENERATION_FLOOR_RATIO:.0%} * capacity",
    "tier_1_max_error_rate": FICR_TIER_1,
    "tier_2_max_error_rate": FICR_TIER_2,
    "tier_1_unit_price": PRICE_TIER_1,
    "tier_2_unit_price": PRICE_TIER_2,
    "failed_unit_price": 0.0,
    "group_specific_tiers": False,
    "normalization_basis": "group capacity (kWh), same for all groups per column",
    "ficr_group_formula": "sum(actual * unit_price) / sum(actual * 4.0)",
    "ficr_aggregate": "nanmean across groups",
    "score_formula": "0.5 * (1 - NMAE) + 0.5 * FICR",
    "note": (
        "FICR은 signed error가 아니라 절대 오차율 기준이다. "
        "6% 이하=4원, 6~8%=3원, 8% 초과=0원. "
        "±6%/8%는 과대/과소 방향과 무관하게 |error|/capacity에 적용된다."
    ),
}

BOUNDARY_BANDS = [
    ("0_1pct", 0.0, 0.01),
    ("1_2pct", 0.01, 0.02),
    ("2_3pct", 0.02, 0.03),
    ("3_5pct", 0.03, 0.05),
    ("5pct_plus", 0.05, np.inf),
]

UTIL_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
UTIL_LABELS = ["0~0.1", "0.1~0.3", "0.3~0.5", "0.5~0.7", "0.7~1.0"]
WS_BINS = [0.0, 3.0, 5.0, 8.0, 12.0, np.inf]
WS_LABELS = ["0~3", "3~5", "5~8", "8~12", "12+"]

SIM_MULTIPLIERS = [1.005, 1.01, 1.015, 1.02, 1.025, 1.03]
MIN_SAMPLES = {"strict": 100, "medium": 50, "exploratory": 24}


def row_unit_price(error_rate: float) -> float:
    if error_rate <= FICR_TIER_1:
        return PRICE_TIER_1
    if error_rate <= FICR_TIER_2:
        return PRICE_TIER_2
    return 0.0


def ficr_row_success(error_rate: float) -> bool:
    return error_rate <= FICR_TIER_2


def compute_row_ficr_fields(actual: float, prediction: float, capacity: float) -> dict[str, Any]:
    floor = capacity * GENERATION_FLOOR_RATIO
    if actual < floor or np.isnan(actual):
        return {
            "active": False,
            "ficr_class": "inactive_or_excluded",
            "error_rate": np.nan,
            "signed_error": np.nan,
            "normalized_signed_error": np.nan,
            "ficr_lower_bound_rate": FICR_TIER_1,
            "ficr_upper_bound_rate": FICR_TIER_2,
            "distance_to_nearest_ficr_boundary": np.nan,
            "boundary_band": None,
            "min_pred_delta_to_8pct": np.nan,
            "min_multiplier_to_8pct": np.nan,
            "is_under": np.nan,
            "is_over": np.nan,
            "unit_price": 0.0,
            "ficr_success": False,
        }

    signed_error = prediction - actual
    error_rate = abs(signed_error) / capacity
    is_under = signed_error < -1e-9
    is_over = signed_error > 1e-9
    unit_price = row_unit_price(error_rate)
    success = ficr_row_success(error_rate)

    if success:
        if error_rate <= FICR_TIER_1:
            dist = FICR_TIER_1 - error_rate
            nearest = "tier1_inner"
        else:
            dist = error_rate - FICR_TIER_1
            nearest = "tier2_to_tier1"
        ficr_class = "inside_ficr"
        boundary_band = None
        min_pred_delta = np.nan
        min_mult = np.nan
        if error_rate > FICR_TIER_1:
            if is_under:
                target = actual - FICR_TIER_1 * capacity
                min_pred_delta = target - prediction
                min_mult = target / prediction if prediction > 1e-9 else np.nan
            elif is_over:
                target = actual + FICR_TIER_1 * capacity
                min_pred_delta = target - prediction
                min_mult = target / prediction if prediction > 1e-9 else np.nan
    else:
        dist = error_rate - FICR_TIER_2
        nearest = "outside_8pct"
        boundary_band = _boundary_band(dist)
        if is_under:
            target = actual - FICR_TIER_2 * capacity
            min_pred_delta = target - prediction
            min_mult = target / prediction if prediction > 1e-9 else np.nan
            ficr_class = "under_near_boundary" if dist <= 0.05 else "under_far"
        elif is_over:
            target = actual + FICR_TIER_2 * capacity
            min_pred_delta = target - prediction
            min_mult = target / prediction if prediction > 1e-9 else np.nan
            ficr_class = "over_near_boundary" if dist <= 0.05 else "over_far"
        else:
            ficr_class = "over_near_boundary" if dist <= 0.05 else "over_far"
            min_pred_delta = 0.0
            min_mult = 1.0

    return {
        "active": True,
        "ficr_class": ficr_class,
        "error_rate": error_rate,
        "signed_error": signed_error,
        "normalized_signed_error": signed_error / capacity,
        "ficr_lower_bound_rate": FICR_TIER_1,
        "ficr_upper_bound_rate": FICR_TIER_2,
        "distance_to_nearest_ficr_boundary": dist,
        "nearest_boundary": nearest,
        "boundary_band": boundary_band,
        "min_pred_delta_to_8pct": min_pred_delta,
        "min_multiplier_to_8pct": min_mult,
        "is_under": is_under,
        "is_over": is_over,
        "unit_price": unit_price,
        "ficr_success": success,
    }


def _boundary_band(distance_outside_8pct: float) -> str | None:
    if distance_outside_8pct <= 0:
        return None
    if distance_outside_8pct <= 0.01:
        return "0_1pct"
    if distance_outside_8pct <= 0.02:
        return "1_2pct"
    if distance_outside_8pct <= 0.03:
        return "2_3pct"
    if distance_outside_8pct <= 0.05:
        return "3_5pct"
    return "5pct_plus"


def build_aligned_long_frame(
    bundle: dict,
    config: dict,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    period_name: str,
) -> pd.DataFrame:
    pred_wide = apply_pipeline_fixed(bundle["raw"], bundle["v03"], bundle["long_val"], config)
    truth = bundle["truth"].copy()
    long_val = bundle["long_val"].copy()
    long_val["forecast_kst_dtm"] = pd.to_datetime(long_val["forecast_kst_dtm"])

    mask = truth["kst_dtm"] >= start
    if end is not None:
        mask &= truth["kst_dtm"] < end
    truth = truth.loc[mask, ["kst_dtm", *GROUP_COLUMNS]].copy()

    rows = []
    for col in GROUP_COLUMNS:
        gid = int(col.split("_")[-1])
        cap = GROUP_CAPACITY_KWH[col]
        t_col = truth[["kst_dtm", col]].rename(columns={"kst_dtm": "forecast_kst_dtm", col: "actual"})
        p_col = pred_wide[["forecast_kst_dtm", col]].rename(columns={col: "prediction"})
        merged = t_col.merge(p_col, on="forecast_kst_dtm", how="inner")
        lv = long_val.loc[long_val["group_id"] == gid, ["forecast_kst_dtm", "ldaps_ws_hub_blend"]]
        merged = merged.merge(lv, on="forecast_kst_dtm", how="left")
        merged["group_id"] = gid
        merged["group_col"] = col
        merged["capacity"] = cap
        merged["period"] = period_name
        merged["month"] = merged["forecast_kst_dtm"].dt.month
        merged["hour"] = merged["forecast_kst_dtm"].dt.hour
        merged["key"] = merged["forecast_kst_dtm"].astype(str) + "|g" + merged["group_id"].astype(str)
        rows.append(merged)

    df = pd.concat(rows, ignore_index=True)
    df = df[df["actual"].notna()].copy()

    ficr_fields = df.apply(
        lambda r: compute_row_ficr_fields(float(r["actual"]), float(r["prediction"]), float(r["capacity"])),
        axis=1,
    )
    ficr_df = pd.DataFrame(ficr_fields.tolist())
    df = pd.concat([df.reset_index(drop=True), ficr_df], axis=1)

    df["util_pred"] = df["prediction"] / df["capacity"]
    df["util_actual"] = df["actual"] / df["capacity"]
    df["util_pred_bin"] = pd.cut(df["util_pred"], bins=UTIL_BINS, labels=UTIL_LABELS, include_lowest=True)
    df["util_actual_bin"] = pd.cut(df["util_actual"], bins=UTIL_BINS, labels=UTIL_LABELS, include_lowest=True)
    df["ws_bin"] = pd.cut(df["ldaps_ws_hub_blend"], bins=WS_BINS, labels=WS_LABELS, include_lowest=True)
    df["is_pipeline_hotspot"] = df.apply(
        lambda r: is_hotspot_row(int(r["group_id"]), int(r["month"]), int(r["hour"])), axis=1
    )
    return df


def classify_summary(df: pd.DataFrame) -> dict[str, Any]:
    active = df[df["active"]]
    n = len(df)
    na = len(active)
    counts = active["ficr_class"].value_counts().to_dict()
    failed = active[~active["ficr_success"]]
    near_under = active[active["ficr_class"] == "under_near_boundary"]
    near_over = active[active["ficr_class"] == "over_near_boundary"]
    return {
        "total_rows": n,
        "active_rows": na,
        "inactive_rows": n - na,
        "class_counts": counts,
        "failed_rows": len(failed),
        "failed_ratio": len(failed) / na if na else 0.0,
        "near_boundary_failed_ratio": (len(near_under) + len(near_over)) / na if na else 0.0,
        "under_near_boundary_ratio": len(near_under) / na if na else 0.0,
        "over_near_boundary_ratio": len(near_over) / na if na else 0.0,
        "inside_ficr_ratio": counts.get("inside_ficr", 0) / na if na else 0.0,
    }


def aggregate_slice_stats(df: pd.DataFrame, group_cols: list[str], slice_label: str | None = None) -> list[dict]:
    active = df[df["active"]].copy()
    if not group_cols:
        return [_one_slice(active, {}, slice_label or "overall")]

    out = []
    for keys, g in active.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = slice_label or "_".join(f"{c}={v}" for c, v in zip(group_cols, keys))
        dims = dict(zip(group_cols, keys))
        out.append(_one_slice(g, dims, label))
    return out


def _one_slice(g: pd.DataFrame, dims: dict, label: str) -> dict:
    failed = g[~g["ficr_success"]]
    near_u = g[g["ficr_class"] == "under_near_boundary"]
    near_o = g[g["ficr_class"] == "over_near_boundary"]
    far = g[g["ficr_class"].isin(["under_far", "over_far"])]
    mults = near_u["min_multiplier_to_8pct"].dropna()
    mults = mults[(mults > 0) & (mults < 10)]

    earned = (g["actual"] * g["unit_price"]).sum()
    max_settle = (g["actual"] * PRICE_TIER_1).sum()
    ficr = float(earned / max_settle) if max_settle > 0 else np.nan
    nmae = float(g["error_rate"].mean()) if len(g) else np.nan

    return {
        "slice": label,
        **dims,
        "row_count": int(len(g)),
        "failed_count": int(len(failed)),
        "under_near_count": int(len(near_u)),
        "over_near_count": int(len(near_o)),
        "far_error_count": int(len(far)),
        "near_boundary_ratio": (len(near_u) + len(near_o)) / len(g) if len(g) else 0.0,
        "under_near_ratio": len(near_u) / len(g) if len(g) else 0.0,
        "over_near_ratio": len(near_o) / len(g) if len(g) else 0.0,
        "mean_boundary_distance": float(failed["distance_to_nearest_ficr_boundary"].mean()) if len(failed) else 0.0,
        "median_min_multiplier": float(mults.median()) if len(mults) else np.nan,
        "p90_min_multiplier": float(mults.quantile(0.9)) if len(mults) else np.nan,
        "mean_bias": float(g["signed_error"].mean()) if len(g) else np.nan,
        "nmae": nmae,
        "ficr": ficr,
        "is_pipeline_hotspot": bool(dims.get("is_pipeline_hotspot", g["is_pipeline_hotspot"].mean() > 0.5 if len(g) else False)),
    }


def filter_candidate_slices(
    slice_stats_by_period: dict[str, list[dict]],
    min_n: int = 50,
) -> list[dict]:
    """Aggregate g×m×h slices across periods and score candidacy."""
    key_cols = ["group_id", "month", "hour"]
    pooled: dict[tuple, dict] = {}

    for period, slices in slice_stats_by_period.items():
        for s in slices:
            if not all(k in s for k in key_cols):
                continue
            key = (s["group_id"], s["month"], s["hour"])
            if key not in pooled:
                pooled[key] = {
                    "group_id": s["group_id"],
                    "month": s["month"],
                    "hour": s["hour"],
                    "slice": f"g{key[0]}_m{key[1]}_h{key[2]}",
                    "periods": {},
                    "total_rows": 0,
                    "total_under_near": 0,
                    "total_over_near": 0,
                    "total_far": 0,
                }
            pooled[key]["periods"][period] = s
            pooled[key]["total_rows"] += s["row_count"]
            pooled[key]["total_under_near"] += s["under_near_count"]
            pooled[key]["total_over_near"] += s["over_near_count"]
            pooled[key]["total_far"] += s["far_error_count"]

    candidates = []
    for key, p in pooled.items():
        n = p["total_rows"]
        if n < min_n:
            continue
        under_ratio = p["total_under_near"] / n
        over_ratio = p["total_over_near"] / n
        far_ratio = p["total_far"] / n
        mults = [
            p["periods"][per]["median_min_multiplier"]
            for per in p["periods"]
            if not np.isnan(p["periods"][per].get("median_min_multiplier", np.nan))
        ]
        med_mult = float(np.median(mults)) if mults else np.nan

        period_under_positive = sum(
            1 for per, s in p["periods"].items() if s["under_near_count"] > s["over_near_count"]
        )

        hotspot = any(s.get("is_pipeline_hotspot") for s in p["periods"].values())

        score = 0.0
        if under_ratio >= 0.05:
            score += 2
        if over_ratio <= 0.01:
            score += 1
        if 1.01 <= med_mult <= 1.03:
            score += 2
        elif 1.005 <= med_mult <= 1.05:
            score += 1
        if period_under_positive >= 2:
            score += 2
        if under_ratio > far_ratio:
            score += 1
        if not hotspot:
            score += 1

        candidates.append({
            **p,
            "under_near_ratio": under_ratio,
            "over_near_ratio": over_ratio,
            "far_ratio": far_ratio,
            "median_min_multiplier": med_mult,
            "periods_with_under_dominant": period_under_positive,
            "is_pipeline_hotspot": hotspot,
            "candidate_score": score,
        })

    return sorted(candidates, key=lambda x: (-x["candidate_score"], -x["under_near_ratio"], -x["total_rows"]))


def long_to_wide_pred(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(
        index="forecast_kst_dtm", columns="group_col", values="prediction", aggfunc="first"
    ).reset_index()
    return wide[["forecast_kst_dtm", *GROUP_COLUMNS]]


def long_to_wide_truth(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(
        index="forecast_kst_dtm", columns="group_col", values="actual", aggfunc="first"
    ).reset_index()
    out = wide.rename(columns={"forecast_kst_dtm": "kst_dtm"})
    return out[["kst_dtm", *GROUP_COLUMNS]]


def count_ficr_transitions(df: pd.DataFrame, new_pred: pd.Series) -> dict[str, int]:
    old_success = df["ficr_success"].to_numpy()
    new_fields = [
        compute_row_ficr_fields(float(a), float(p), float(c))
        for a, p, c in zip(df["actual"], new_pred, df["capacity"])
    ]
    new_success = np.array([f["ficr_success"] for f in new_fields])
    return {
        "fail_to_success": int((~old_success & new_success).sum()),
        "success_to_fail": int((old_success & ~new_success).sum()),
        "net_ficr_row_gain": int(new_success.sum() - old_success.sum()),
        "changed_rows": int((np.abs(new_pred.to_numpy() - df["prediction"].to_numpy()) > 1e-9).sum()),
    }


GROUP_ID_TO_COL = {1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"}


def simulate_slice_multiplier(
    df: pd.DataFrame,
    truth_wide: pd.DataFrame,
    group_id: int,
    month: int,
    hour: int,
    multiplier: float,
    baseline_metrics: dict,
) -> dict[str, Any]:
    mask = (
        (df["group_id"] == group_id)
        & (df["month"] == month)
        & (df["hour"] == hour)
        & (df["active"])
    )
    pred_wide = long_to_wide_pred(df)
    pred_copy = pred_wide.copy()
    col = GROUP_ID_TO_COL[group_id]

    affected = df.loc[mask, ["forecast_kst_dtm", "prediction"]].copy()
    if affected.empty:
        return {"valid": False}

    affected["new_prediction"] = affected["prediction"] * multiplier
    for _, row in affected.iterrows():
        ts = row["forecast_kst_dtm"]
        pred_copy.loc[pred_copy["forecast_kst_dtm"] == ts, col] = row["new_prediction"]

    metrics = evaluate_submission(truth_wide, pred_copy, time_col="kst_dtm")
    sub = df.loc[mask].copy()
    sub_new = sub.copy()
    sub_new["prediction"] = sub_new["prediction"] * multiplier
    trans = count_ficr_transitions(sub, sub_new["prediction"])

    return {
        "valid": True,
        "group_id": group_id,
        "month": month,
        "hour": hour,
        "multiplier": multiplier,
        "score": metrics["score"],
        "1_minus_nmae": metrics["1_minus_nmae"],
        "ficr": metrics["ficr"],
        "delta_score": metrics["score"] - baseline_metrics["score"],
        "delta_nmae": metrics["1_minus_nmae"] - baseline_metrics["1_minus_nmae"],
        "delta_ficr": metrics["ficr"] - baseline_metrics["ficr"],
        **trans,
    }


def grade_candidate(sim_results: list[dict], candidate: dict) -> str:
    by_period: dict[str, list[dict]] = {}
    for r in sim_results:
        if not r.get("valid"):
            continue
        by_period.setdefault(r["period"], []).append(r)

    if not by_period:
        return "reject"

    best_per_period = {
        p: max(rs, key=lambda x: (x["delta_score"], x["delta_ficr"]))
        for p, rs in by_period.items()
    }
    full = best_per_period.get("2024_full")
    if full is None or full["delta_score"] < 0:
        return "reject"
    if full["net_ficr_row_gain"] <= 0:
        return "reject"
    if full["success_to_fail"] > full["fail_to_success"] * 0.5:
        return "reject"

    period_improve = sum(1 for r in best_per_period.values() if r["delta_score"] > 0)
    h1 = best_per_period.get("2024_h1")
    h2 = best_per_period.get("2024_h2")
    h23 = best_per_period.get("2023_h2")

    n = candidate["total_rows"]
    mult = full["multiplier"]

    if mult > 1.03:
        return "reject"
    if n >= MIN_SAMPLES["strict"] and period_improve >= 3 and full["delta_ficr"] > 0:
        if h23 is None or h23["delta_score"] >= -0.0001:
            return "robust_candidate"
    if n >= MIN_SAMPLES["medium"] and period_improve >= 2:
        if h23 is None or h23["delta_score"] >= -0.0002:
            return "conditional_candidate"
    if n >= MIN_SAMPLES["exploratory"]:
        return "exploratory_only"
    return "reject"


def validate_period_keys(long_frames: dict[str, pd.DataFrame]) -> dict:
    report = {}
    for pname, df in long_frames.items():
        ts = df["forecast_kst_dtm"]
        keys = df["key"]
        report[pname] = {
            "n_rows": len(df),
            "n_unique_keys": keys.nunique(),
            "duplicate_keys": int(keys.duplicated().sum()),
            "duplicate_ts_group": int(df.duplicated(subset=["forecast_kst_dtm", "group_id"]).sum()),
        }
    return report


def build_all_period_frames(bundles: dict, config: dict | None = None) -> dict[str, pd.DataFrame]:
    config = config or V13_BEST_CONFIG
    frames = {}
    for pname, pdef in PERIOD_DEFS.items():
        bundle = bundles[pdef["bundle_key"]]
        frames[pname] = build_aligned_long_frame(
            bundle, config, pdef["start"], pdef["end"], pname
        )
    return frames


def baseline_metrics_by_period(frames: dict[str, pd.DataFrame]) -> dict[str, dict]:
    out = {}
    for pname, df in frames.items():
        truth = long_to_wide_truth(df)
        pred = long_to_wide_pred(df)
        out[pname] = evaluate_submission(truth, pred, time_col="kst_dtm")
    return out
