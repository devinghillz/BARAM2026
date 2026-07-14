"""Step 1-2: EDA + feature candidate extraction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, OUTPUT_DIR, VALID_START
from src.data_loader import load_labels, load_turbine_info, load_weather
from src.features import aggregate_weather_to_groups, build_group_frame, get_feature_columns
from src.power_curve import build_scada_monthly_curve, build_month_hour_climatology

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_labels(labels: pd.DataFrame) -> dict:
    rows = []
    for col in GROUP_COLUMNS:
        cap = GROUP_CAPACITY_KWH[col]
        s = labels[col]
        valid = s.dropna()
        rows.append(
            {
                "group": col,
                "label_start": str(valid.index.min()) if len(valid) else None,
                "first_valid_date": str(labels.loc[valid.index.min(), "kst_dtm"]) if len(valid) else None,
                "missing_ratio": float(s.isna().mean()),
                "zero_ratio": float((valid == 0).mean()),
                "active_ge_10pct_ratio": float((valid >= cap * 0.10).mean()),
                "utilization_mean": float((valid / cap).mean()),
                "utilization_p50": float((valid / cap).median()),
                "utilization_p90": float((valid / cap).quantile(0.9)),
                "capacity_kwh": cap,
            }
        )

    labels = labels.copy()
    labels["month"] = labels["kst_dtm"].dt.month
    labels["hour"] = labels["kst_dtm"].dt.hour
    seasonal = {}
    for col in GROUP_COLUMNS:
        cap = GROUP_CAPACITY_KWH[col]
        by_month = labels.groupby("month")[col].apply(lambda x: (x.dropna() / cap).mean())
        seasonal[col] = by_month.round(3).to_dict()

    hourly = {}
    for col in GROUP_COLUMNS:
        cap = GROUP_CAPACITY_KWH[col]
        by_hour = labels.groupby("hour")[col].apply(lambda x: (x.dropna() / cap).mean())
        hourly[col] = by_hour.round(3).to_dict()

    return {"summary": rows, "seasonal_utilization": seasonal, "hourly_utilization": hourly}


def analyze_scada() -> dict:
    from src.config import PATHS

    vestas = pd.read_csv(PATHS["scada_vestas"], encoding="utf-8-sig")
    unison = pd.read_csv(PATHS["scada_unison"], encoding="utf-8-sig")
    vestas_ws = vestas.filter(regex="_ws$").mean(axis=1)
    vestas_pw = vestas.filter(regex="power").mean(axis=1)
    unison_ws = unison.filter(regex="_ws$").mean(axis=1)
    unison_pw = unison.filter(regex="power").mean(axis=1)

    def curve_stats(ws, pw, name):
        df = pd.DataFrame({"ws": ws, "pw": pw}).dropna()
        df["ws_bin"] = (df["ws"] // 1).clip(0, 25)
        binned = df.groupby("ws_bin")["pw"].median()
        cut_in = float(binned[binned > 10].index.min()) if (binned > 10).any() else None
        rated_ws = float(binned.idxmax()) if len(binned) else None
        return {
            "source": name,
            "n_rows": len(df),
            "ws_mean": float(df["ws"].mean()),
            "ws_p90": float(df["ws"].quantile(0.9)),
            "pw_mean_kw": float(df["pw"].mean()),
            "cut_in_ws_bin_approx": cut_in,
            "peak_ws_bin": rated_ws,
        }

    return {
        "vestas": curve_stats(vestas_ws, vestas_pw, "vestas"),
        "unison": curve_stats(unison_ws, unison_pw, "unison"),
    }


def analyze_weather_structure() -> dict:
    out = {}
    for source in ["ldaps", "gfs"]:
        w = load_weather(source, "train")
        grids_per_ts = w.groupby("forecast_kst_dtm").size()
        out[source] = {
            "rows": len(w),
            "period_start": str(w["forecast_kst_dtm"].min()),
            "period_end": str(w["forecast_kst_dtm"].max()),
            "grids_per_timestamp_median": float(grids_per_ts.median()),
            "unique_grids": int(w["grid_id"].nunique()),
        }
    return out


def analyze_feature_correlations(labels: pd.DataFrame) -> dict:
    scada_curve = build_scada_monthly_curve()
    valid_start = pd.Timestamp(VALID_START)

    results = {}
    for method in ["nearest", "idw"]:
        frames = []
        for source in ["ldaps", "gfs"]:
            w = load_weather(source, "train")
            g = aggregate_weather_to_groups(w, source=source, method=method)
            frames.append(g)
        from src.features import merge_weather_frames

        merged = merge_weather_frames(frames)
        frame = build_group_frame(
            merged,
            labels=labels,
            clim_labels=labels,
            clim_before=valid_start,
            scada_curve=scada_curve,
        )
        frame = frame[frame["forecast_kst_dtm"] < valid_start].dropna(subset=["power_kwh"])

        feature_cols = get_feature_columns(frame)
        corrs = []
        for col in feature_cols:
            c = frame[[col, "power_kwh", "group_id"]].corr(numeric_only=True).loc[col, "power_kwh"]
            if not np.isnan(c):
                corrs.append({"feature": col, "corr_power": float(c)})

        corr_df = pd.DataFrame(corrs).sort_values("corr_power", key=abs, ascending=False)
        results[method] = {
            "n_features": len(feature_cols),
            "top15": corr_df.head(15).to_dict("records"),
            "by_group_top5": {},
        }
        for gid in [1, 2, 3]:
            part = frame[frame["group_id"] == gid]
            gcorrs = []
            for col in feature_cols:
                c = part[[col, "power_kwh"]].corr(numeric_only=True).iloc[0, 1]
                if not np.isnan(c):
                    gcorrs.append({"feature": col, "corr_power": float(c)})
            gdf = pd.DataFrame(gcorrs).sort_values("corr_power", key=abs, ascending=False)
            results[method][f"group_{gid}_top5"] = gdf.head(5).to_dict("records")

    return results


def analyze_group_correlations(labels: pd.DataFrame) -> dict:
    cols = [c for c in GROUP_COLUMNS]
    df = labels[cols].dropna(how="all")
    corr = df.corr().round(3)
    return corr.to_dict()


def recommend_features(corr_results: dict) -> list[dict]:
    nearest_top = {r["feature"]: r["corr_power"] for r in corr_results["nearest"]["top15"]}
    idw_top = {r["feature"]: r["corr_power"] for r in corr_results["idw"]["top15"]}

    candidates = [
        ("풍속 10m", ["ldaps_ws10", "gfs_ws10", "blend_ws10"], "필수"),
        ("풍속 허브고도", ["ldaps_ws50", "gfs_ws80", "gfs_ws100", "blend_ws_hub"], "필수"),
        ("풍향", ["ldaps_wd10_sin", "ldaps_wd10_cos", "gfs_wd10_sin", "gfs_wd10_cos"], "권장"),
        ("시간/계절", ["hour_sin", "hour_cos", "month_sin", "month_cos", "hour", "month"], "필수"),
        ("기후학", ["clim_power_kwh"], "FICR 안정"),
        ("SCADA prior", ["scada_prior_kwh"], "FICR/곡선"),
        ("예보 차이", ["ws10_diff_ldaps_gfs", "ws10_ratio_ldaps_gfs"], "보조"),
        ("기온/습도/기압", ["ldaps_temp2m", "gfs_temp2m", "ldaps_rh2m", "gfs_rh2m", "ldaps_sp", "gfs_sp"], "보조"),
        ("돌풍", ["gfs_gust"], "보조"),
    ]

    recs = []
    for name, feats, priority in candidates:
        scores = []
        for f in feats:
            if f in nearest_top:
                scores.append(nearest_top[f])
            if f in idw_top:
                scores.append(idw_top[f])
        recs.append(
            {
                "category": name,
                "features": feats,
                "priority": priority,
                "max_abs_corr_seen": float(max([abs(s) for s in scores], default=0)),
            }
        )
    return sorted(recs, key=lambda x: -x["max_abs_corr_seen"])


def main():
    print("Loading data...")
    labels = load_labels()
    turbines = load_turbine_info()

    print("Step 1: EDA...")
    label_eda = analyze_labels(labels)
    scada_eda = analyze_scada()
    weather_eda = analyze_weather_structure()
    group_corr = analyze_group_correlations(labels)

    print("Step 2: Feature correlations...")
    feat_corr = analyze_feature_correlations(labels)
    recommendations = recommend_features(feat_corr)

    report = {
        "step1_labels": label_eda,
        "step1_scada": scada_eda,
        "step1_weather": weather_eda,
        "step1_turbines": {
            "count": len(turbines),
            "by_kpx_group": turbines.groupby("kpx_group").size().to_dict(),
            "hub_height_m": turbines["hub_height_m"].dropna().unique().tolist(),
        },
        "step1_group_corr": group_corr,
        "step2_feature_corr": feat_corr,
        "step2_recommendations": recommendations,
    }

    out_json = OUTPUT_DIR / "eda_step12_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out_json}")
    return report


if __name__ == "__main__":
    main()
