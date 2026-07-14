"""Step 3: FICR error-band analysis on 2024 hold-out predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, OUTPUT_DIR, VALID_START
from src.data_loader import load_labels, load_weather
from src.features import (
    aggregate_weather_to_groups,
    build_group_frame,
    get_feature_columns,
    merge_weather_frames,
)
from src.metrics import FICR_TIER_1, FICR_TIER_2, evaluate_submission
from src.power_curve import build_scada_monthly_curve

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIER_LABELS = {
    "tier1_le_6pct": "≤6% (4원)",
    "tier2_6_8pct": "6~8% (3원)",
    "tier3_gt_8pct": ">8% (0원)",
}


def _error_tier(error_rate: float) -> str:
    if error_rate <= FICR_TIER_1:
        return "tier1_le_6pct"
    if error_rate <= FICR_TIER_2:
        return "tier2_6_8pct"
    return "tier3_gt_8pct"


def build_validation_frame(labels: pd.DataFrame, valid_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    scada_curve = build_scada_monthly_curve()
    frames = []
    for source in ["ldaps", "gfs"]:
        w = load_weather(source, "train")
        g = aggregate_weather_to_groups(w, source=source, method="nearest")
        frames.append(g)
    merged = merge_weather_frames(frames)
    frame = build_group_frame(
        merged,
        labels=labels,
        clim_labels=labels,
        clim_before=valid_start,
        scada_curve=scada_curve,
    )
    fit = frame[frame["forecast_kst_dtm"] < valid_start].dropna(subset=["power_kwh"])
    valid = frame[frame["forecast_kst_dtm"] >= valid_start].dropna(subset=["power_kwh"])
    return fit, valid


def train_and_predict(fit: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    import lightgbm as lgb

    feature_cols = get_feature_columns(fit)
    rows = []
    for gid in [1, 2, 3]:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    lgb.LGBMRegressor(
                        n_estimators=600,
                        learning_rate=0.04,
                        num_leaves=47,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )
        tr = fit[fit["group_id"] == gid]
        va = valid[valid["group_id"] == gid].copy()
        model.fit(tr[feature_cols], tr["power_kwh"])
        va["pred_kwh"] = model.predict(va[feature_cols])
        rows.append(va)
    return pd.concat(rows, ignore_index=True)


def build_error_detail(valid_pred: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in valid_pred.iterrows():
        gid = int(row["group_id"])
        col = f"kpx_group_{gid}"
        cap = GROUP_CAPACITY_KWH[col]
        actual = float(row["power_kwh"])
        pred = float(row["pred_kwh"])
        if actual < cap * 0.10:
            continue
        err = abs(pred - actual) / cap
        records.append(
            {
                "forecast_kst_dtm": row["forecast_kst_dtm"],
                "group_id": gid,
                "group_col": col,
                "actual_kwh": actual,
                "pred_kwh": pred,
                "error_rate": err,
                "error_kwh": abs(pred - actual),
                "tier": _error_tier(err),
                "month": row["forecast_kst_dtm"].month,
                "hour": row["forecast_kst_dtm"].hour,
                "ldaps_ws10": row.get("ldaps_ws10", np.nan),
                "ldaps_ws50": row.get("ldaps_ws50", np.nan),
                "blend_ws10": row.get("blend_ws10", np.nan),
            }
        )
    df = pd.DataFrame(records)
    df["ws_bin"] = (df["blend_ws10"].fillna(df["ldaps_ws10"]) // 1).clip(0, 25).astype(int)
    return df


def tier_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    counts = df["tier"].value_counts()
    return {
        "n_active_hours": total,
        "tier_share": {TIER_LABELS[k]: float(counts.get(k, 0) / total) for k in TIER_LABELS},
        "tier_count": {TIER_LABELS[k]: int(counts.get(k, 0)) for k in TIER_LABELS},
    }


def breakdown(df: pd.DataFrame, key: str, top_n: int = 12) -> list[dict]:
    g = df.groupby(key)
    out = []
    for name, part in g:
        n = len(part)
        gt8 = (part["tier"] == "tier3_gt_8pct").mean()
        le6 = (part["tier"] == "tier1_le_6pct").mean()
        out.append(
            {
                key: int(name) if key != "group_col" else str(name),
                "n": n,
                "mean_error_rate": float(part["error_rate"].mean()),
                "share_le_6pct": float(le6),
                "share_gt_8pct": float(gt8),
                "mean_actual_mwh": float(part["actual_kwh"].mean() / 1000),
            }
        )
    return sorted(out, key=lambda x: -x["share_gt_8pct"])[:top_n]


def cross_breakdown(df: pd.DataFrame) -> list[dict]:
    g = df.groupby(["month", "hour"])
    rows = []
    for (month, hour), part in g:
        if len(part) < 30:
            continue
        rows.append(
            {
                "month": int(month),
                "hour": int(hour),
                "n": len(part),
                "share_gt_8pct": float((part["tier"] == "tier3_gt_8pct").mean()),
                "mean_error_rate": float(part["error_rate"].mean()),
            }
        )
    return sorted(rows, key=lambda x: -x["share_gt_8pct"])[:15]


def over_under_analysis(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["bias"] = np.where(df["pred_kwh"] > df["actual_kwh"], "over", "under")
    rows = []
    for tier in ["tier3_gt_8pct", "tier2_6_8pct", "tier1_le_6pct"]:
        part = df[df["tier"] == tier]
        if len(part) == 0:
            continue
        rows.append(
            {
                "tier": TIER_LABELS[tier],
                "over_ratio": float((part["bias"] == "over").mean()),
                "under_ratio": float((part["bias"] == "under").mean()),
                "mean_error_rate": float(part["error_rate"].mean()),
            }
        )
    return {"bias_by_tier": rows}


def improvement_hints(df: pd.DataFrame, summary: dict) -> list[str]:
    hints = []
    gt8_share = summary["tier_share"][TIER_LABELS["tier3_gt_8pct"]]
    hints.append(
        f"평가 대상 시간의 {gt8_share*100:.1f}%가 8% 초과 오차 → FICR 병목 (상위권은 이 비율을 크게 낮춤)"
    )

    worst_month = breakdown(df, "month", top_n=3)
    if worst_month:
        m = worst_month[0]
        hints.append(
            f"8% 초과 최다 월: {m['month']}월 ({m['share_gt_8pct']*100:.1f}%) → 여름/가을 저풍·저이용 구간 후처리 필요"
        )

    worst_hour = breakdown(df, "hour", top_n=3)
    if worst_hour:
        h = worst_hour[0]
        hints.append(
            f"8% 초과 최다 시간: {h['hour']}시 ({h['share_gt_8pct']*100:.1f}%) → 일변동 climatology 보정"
        )

    worst_ws = breakdown(df, "ws_bin", top_n=3)
    if worst_ws:
        w = worst_ws[0]
        hints.append(
            f"8% 초과 최다 풍속 bin: {w['ws_bin']} m/s ({w['share_gt_8pct']*100:.1f}%) → 파워커브 비선형 구간 보정"
        )

    for g in breakdown(df, "group_col", top_n=3):
        hints.append(
            f"{g['group_col']}: 8% 초과 {g['share_gt_8pct']*100:.1f}%, 평균오차 {g['mean_error_rate']*100:.1f}%"
        )

    return hints


def main():
    valid_start = pd.Timestamp(VALID_START)
    labels = load_labels()

    print("[3-1] Train on pre-2024, predict 2024 hold-out...")
    fit, valid = build_validation_frame(labels, valid_start)
    valid_pred = train_and_predict(fit, valid)

    pred_wide = (
        valid_pred.pivot(index="forecast_kst_dtm", columns="group_id", values="pred_kwh")
        .rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
        .reset_index()
    )
    pred_wide["forecast_kst_dtm"] = pd.to_datetime(pred_wide["forecast_kst_dtm"])
    true_wide = labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]]
    scores = evaluate_submission(true_wide, pred_wide, time_col="kst_dtm")
    print(f"  Score={scores['score']:.4f}, NMAE={scores['1_minus_nmae']:.4f}, FICR={scores['ficr']:.4f}")

    print("[3-2] Error-band breakdown...")
    detail = build_error_detail(valid_pred)
    overall = tier_summary(detail)
    by_group = breakdown(detail, "group_col", top_n=5)
    by_month = breakdown(detail, "month", top_n=12)
    by_hour = breakdown(detail, "hour", top_n=12)
    by_ws = breakdown(detail, "ws_bin", top_n=12)
    worst_slots = cross_breakdown(detail)
    bias = over_under_analysis(detail)
    hints = improvement_hints(detail, overall)

    report = {
        "validation_period": f">={valid_start}",
        "model": "nearest + LDAPS/GFS LGBM (v0.2 유사)",
        "scores": scores,
        "overall_tier_distribution": overall,
        "by_group": by_group,
        "by_month_sorted_by_gt8": by_month,
        "by_hour_sorted_by_gt8": by_hour,
        "by_wind_speed_bin_sorted_by_gt8": by_ws,
        "worst_month_hour_slots": worst_slots,
        "over_under_bias": bias,
        "improvement_hints": hints,
    }

    out = OUTPUT_DIR / "eda_step3_ficr_error_bands.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return report


if __name__ == "__main__":
    main()
