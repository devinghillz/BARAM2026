from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS
from src.data_loader import haversine_km, load_group_centroids
from src.power_curve import (
    GROUP_TURBINE_COUNT,
    build_group_scada_curves,
    build_month_hour_climatology,
    build_scada_monthly_curve,
    build_group_type_curves,
    scada_prior_from_wind,
    type_prior_from_wind,
)

LDAPS_WIND_COLS = {
    "u10": "heightAboveGround_10_10u",
    "v10": "heightAboveGround_10_10v",
    "u50_max": "heightAboveGround_50_50MUmax",
    "v50_max": "heightAboveGround_50_50MVmax",
    "temp2m": "heightAboveGround_2_t",
    "rh2m": "heightAboveGround_2_r",
    "sp": "surface_0_sp",
}

GFS_WIND_COLS = {
    "u10": "heightAboveGround_10_10u",
    "v10": "heightAboveGround_10_10v",
    "u80": "heightAboveGround_80_u",
    "v80": "heightAboveGround_80_v",
    "u100": "heightAboveGround_100_100u",
    "v100": "heightAboveGround_100_100v",
    "temp2m": "heightAboveGround_2_2t",
    "rh2m": "heightAboveGround_2_2r",
    "sp": "surface_0_sp",
    "gust": "surface_0_gust",
}


def _wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    return np.hypot(u, v)


def _wind_dir_deg(u: pd.Series, v: pd.Series) -> pd.Series:
    return (270 - np.degrees(np.arctan2(v, u))) % 360


def _add_wind_features(df: pd.DataFrame, mapping: dict[str, str], prefix: str) -> pd.DataFrame:
    out = df.copy()
    if mapping["u10"] in out.columns and mapping["v10"] in out.columns:
        out[f"{prefix}_ws10"] = _wind_speed(out[mapping["u10"]], out[mapping["v10"]])
        out[f"{prefix}_wd10"] = _wind_dir_deg(out[mapping["u10"]], out[mapping["v10"]])
        out[f"{prefix}_wd10_sin"] = np.sin(np.radians(out[f"{prefix}_wd10"]))
        out[f"{prefix}_wd10_cos"] = np.cos(np.radians(out[f"{prefix}_wd10"]))

    if "u50_max" in mapping and mapping["u50_max"] in out.columns:
        out[f"{prefix}_ws50"] = _wind_speed(
            out[mapping["u50_max"]], out[mapping["v50_max"]]
        )

    if "u80" in mapping and mapping["u80"] in out.columns:
        out[f"{prefix}_ws80"] = _wind_speed(out[mapping["u80"]], out[mapping["v80"]])
        out[f"{prefix}_ws100"] = _wind_speed(
            out[mapping["u100"]], out[mapping["v100"]]
        )

    if mapping["temp2m"] in out.columns:
        out[f"{prefix}_temp2m"] = out[mapping["temp2m"]]
    if mapping["rh2m"] in out.columns:
        out[f"{prefix}_rh2m"] = out[mapping["rh2m"]]
    if mapping["sp"] in out.columns:
        out[f"{prefix}_sp"] = out[mapping["sp"]]
    if "gust" in mapping and mapping["gust"] in out.columns:
        out[f"{prefix}_gust"] = out[mapping["gust"]]
    return out


def _precompute_idw_weights(
    weather: pd.DataFrame,
    centroids: pd.DataFrame,
    power: float = 2.0,
) -> dict[int, pd.DataFrame]:
    grids = (
        weather[["grid_id", "latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    weights: dict[int, pd.DataFrame] = {}
    for _, cent in centroids.iterrows():
        gid = int(cent["group_id"])
        dist = haversine_km(
            cent["latitude"],
            cent["longitude"],
            grids["latitude"].to_numpy(),
            grids["longitude"].to_numpy(),
        )
        w = 1.0 / (dist + 0.1) ** power
        w = w / w.sum()
        weights[gid] = pd.DataFrame({"grid_id": grids["grid_id"], "weight": w})
    return weights


def _aggregate_idw(
    weather: pd.DataFrame,
    feature_cols: list[str],
    weights: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    parts = []
    meta_cols = ["forecast_kst_dtm", "data_available_kst_dtm"]
    for gid, wdf in weights.items():
        merged = weather.merge(wdf, on="grid_id", how="inner")
        for col in feature_cols:
            merged[f"__{col}"] = merged[col] * merged["weight"]
        agg_map = {f"__{col}": "sum" for col in feature_cols}
        agg_map["data_available_kst_dtm"] = "first"
        grouped = merged.groupby("forecast_kst_dtm", as_index=False).agg(agg_map)
        rename = {f"__{col}": col for col in feature_cols}
        grouped = grouped.rename(columns=rename)
        grouped["group_id"] = gid
        parts.append(grouped[meta_cols + ["group_id"] + feature_cols])
    return pd.concat(parts, ignore_index=True)


def _assign_nearest_grid(weather: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    grid_meta = (
        weather[["grid_id", "latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    matches = []
    for _, centroid in centroids.iterrows():
        dist = haversine_km(
            centroid["latitude"],
            centroid["longitude"],
            grid_meta["latitude"].to_numpy(),
            grid_meta["longitude"].to_numpy(),
        )
        nearest = int(grid_meta.iloc[int(np.argmin(dist))]["grid_id"])
        matches.append({"group_id": int(centroid["group_id"]), "grid_id": nearest})
    return pd.DataFrame(matches)


def aggregate_weather_to_groups(
    weather: pd.DataFrame,
    source: str,
    method: str = "idw",
) -> pd.DataFrame:
    """
    기상 격자 -> KPX 그룹 집계.

    method: 'idw' | 'nearest' | 'mean'
    """
    if source == "ldaps":
        mapping = LDAPS_WIND_COLS
        prefix = "ldaps"
    elif source == "gfs":
        mapping = GFS_WIND_COLS
        prefix = "gfs"
    else:
        raise ValueError("source must be 'ldaps' or 'gfs'")

    weather = _add_wind_features(weather, mapping, prefix)
    feature_cols = [c for c in weather.columns if c.startswith(f"{prefix}_")]
    centroids = load_group_centroids()

    if method == "mean":
        agg = weather.groupby("forecast_kst_dtm", as_index=False)[feature_cols].mean()
        parts = []
        for group_id in [1, 2, 3]:
            part = agg.copy()
            part["group_id"] = group_id
            part["data_available_kst_dtm"] = weather.groupby("forecast_kst_dtm")[
                "data_available_kst_dtm"
            ].first().values
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    if method == "idw":
        weights = _precompute_idw_weights(weather, centroids)
        return _aggregate_idw(weather, feature_cols, weights)

    grid_map = _assign_nearest_grid(weather, centroids)
    merged = weather.merge(grid_map, on="grid_id", how="inner")
    meta_cols = ["forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "group_id"]
    picked = merged[meta_cols + feature_cols].drop_duplicates(
        subset=["forecast_kst_dtm", "group_id"]
    )
    return picked.sort_values(["forecast_kst_dtm", "group_id"]).reset_index(drop=True)


def add_time_features(df: pd.DataFrame, time_col: str = "forecast_kst_dtm") -> pd.DataFrame:
    out = df.copy()
    ts = out[time_col]
    out["hour"] = ts.dt.hour
    out["month"] = ts.dt.month
    out["dayofweek"] = ts.dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["season"] = ((out["month"] % 12 + 3) // 3).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_climatology_features(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    clim = build_month_hour_climatology(labels, before=before)
    out = add_time_features(df) if "hour" not in df.columns else df.copy()
    return out.merge(clim, on=["group_id", "month", "hour"], how="left")


def add_power_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """풍속 비선형(파워커브) 파생 피처."""
    out = df.copy()
    ws_cols = [
        c
        for c in out.columns
        if c.endswith("_ws10")
        or c.endswith("_ws50")
        or c.endswith("_ws80")
        or c.endswith("_ws100")
        or c in {"blend_ws10", "blend_ws_hub"}
    ]
    for col in ws_cols:
        s = out[col].fillna(0).clip(lower=0)
        out[f"{col}_sq"] = s ** 2
        out[f"{col}_cu"] = s ** 3
    if "ldaps_ws10" in out.columns and "ldaps_ws50" in out.columns:
        out["ldaps_ws_hub_blend"] = 0.4 * out["ldaps_ws10"] + 0.6 * out["ldaps_ws50"]
        out["ldaps_ws_hub_blend_sq"] = out["ldaps_ws_hub_blend"] ** 2
    if "ldaps_ws_hub_blend" in out.columns:
        ws = out["ldaps_ws_hub_blend"]
        out["in_ws_sweet_spot"] = ((ws >= 5.0) & (ws <= 12.0)).astype(np.int8)
        out["ws_sweet_spot_x_prior"] = out["in_ws_sweet_spot"] * out.get(
            "scada_prior_kwh", pd.Series(0, index=out.index)
        )
    return out


def _safe_series(s: pd.Series, clip: float = 1e6) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-clip, clip)


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """v12: 용량·계절·풍속 교차 피처."""
    out = df.copy()
    caps = out["group_id"].map(
        {gid: GROUP_CAPACITY_KWH[col] for gid, col in enumerate(GROUP_COLUMNS, 1)}
    )
    if "scada_prior_kwh" in out.columns:
        out["prior_util"] = _safe_series(out["scada_prior_kwh"] / caps)
        if "clim_power_kwh" in out.columns:
            out["clim_prior_ratio"] = _safe_series(
                out["clim_power_kwh"] / (out["scada_prior_kwh"] + 100.0), clip=100
            )
    if "group_prior_kwh" in out.columns:
        out["group_prior_util"] = _safe_series(out["group_prior_kwh"] / caps)
        out["prior_gap"] = _safe_series(out["group_prior_kwh"] - out["scada_prior_kwh"])
    if "clim_power_kwh" in out.columns:
        out["clim_util"] = _safe_series(out["clim_power_kwh"] / caps)
    ws = out.get("ldaps_ws_hub_blend", out.get("ldaps_ws10", pd.Series(0, index=out.index)))
    ws = _safe_series(ws.fillna(0), clip=50)
    out["winter_hour"] = (
        out["month"].isin([1, 2, 3, 11]).astype(int) * out["hour"].isin([10, 20, 22]).astype(int)
    )
    out["winter_x_ws"] = out["month"].isin([1, 2, 3]).astype(int) * ws
    out["worst_slot_x_ws"] = out["winter_hour"] * ws
    if "ws10_diff_ldaps_gfs" in out.columns and "in_ws_sweet_spot" in out.columns:
        out["diff_x_sweet"] = out["ws10_diff_ldaps_gfs"] * out["in_ws_sweet_spot"]
    if "scada_prior_kwh" in out.columns:
        out["prior_x_hub_ws"] = _safe_series(out["scada_prior_kwh"] * ws, clip=1e6)
        out["prior_x_ws_sq"] = _safe_series(out["scada_prior_kwh"] * (ws ** 2), clip=1e6)
    if "ldaps_wd10_sin" in out.columns:
        out["wd_sin_x_ws"] = _safe_series(out["ldaps_wd10_sin"] * ws)
        out["wd_cos_x_ws"] = _safe_series(out["ldaps_wd10_cos"] * ws)
    out["n_turbines"] = out["group_id"].map(GROUP_TURBINE_COUNT)
    out["is_group3"] = (out["group_id"] == 3).astype(np.int8)
    return out


def add_blend_and_scada_features(
    df: pd.DataFrame,
    scada_curve: pd.DataFrame,
    type_curves: dict[str, pd.DataFrame] | None = None,
    enhanced: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    ws_cols = [c for c in out.columns if c.endswith("_ws10")]
    if ws_cols:
        out["blend_ws10"] = out[ws_cols].mean(axis=1)
    hub_cols = [c for c in out.columns if c.endswith("_ws50") or c.endswith("_ws80") or c.endswith("_ws100")]
    if hub_cols:
        out["blend_ws_hub"] = out[hub_cols].mean(axis=1)

    if "ldaps_ws10" in out.columns and "gfs_ws10" in out.columns:
        out["ws10_diff_ldaps_gfs"] = out["ldaps_ws10"] - out["gfs_ws10"]
        if enhanced:
            out["ws10_ratio_ldaps_gfs"] = _safe_series(
                out["ldaps_ws10"] / (out["gfs_ws10"].abs() + 0.5), clip=20
            )
        else:
            out["ws10_ratio_ldaps_gfs"] = out["ldaps_ws10"] / (out["gfs_ws10"] + 1e-3)

    ws_for_prior = "blend_ws10"
    if "blend_ws10" in out.columns:
        if "ldaps_ws10" in out.columns and "gfs_ws10" not in out.columns:
            ws_for_prior = "ldaps_ws10"
        elif "gfs_ws10" in out.columns and "ldaps_ws10" not in out.columns:
            ws_for_prior = "gfs_ws10"

    if "blend_ws10" in out.columns or ws_for_prior in out.columns:
        # v05b 호환: 레거시 prior 유지
        legacy_curve = scada_curve
        if "group_id" in scada_curve.columns:
            from src.power_curve import build_scada_monthly_curve

            legacy_curve = build_scada_monthly_curve()
        out["scada_prior_kwh"] = scada_prior_from_wind(out, legacy_curve, ws_col=ws_for_prior)
        if enhanced and "group_id" in scada_curve.columns:
            out["group_prior_kwh"] = scada_prior_from_wind(out, scada_curve, ws_col=ws_for_prior)
    if type_curves is not None and "group_id" in out.columns:
        tp = type_prior_from_wind(out, type_curves, ws_col=ws_for_prior if "blend_ws10" in out.columns else "ldaps_ws10")
        for c in ["prior_vestas_kwh", "prior_unison_kwh", "type_prior_kwh"]:
            if c in tp.columns:
                out[c] = tp[c].to_numpy()
    out = add_power_curve_features(out)
    if enhanced:
        return add_enhanced_features(out)
    return out


def build_group_frame(
    weather_groups: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    clim_labels: pd.DataFrame | None = None,
    clim_before: pd.Timestamp | None = None,
    scada_curve: pd.DataFrame | None = None,
    type_curves: dict[str, pd.DataFrame] | None = None,
    enhanced: bool = False,
) -> pd.DataFrame:
    df = add_time_features(weather_groups)
    if clim_labels is not None:
        df = add_climatology_features(df, clim_labels, before=clim_before)
    if scada_curve is not None:
        tc = type_curves if enhanced else None
        df = add_blend_and_scada_features(df, scada_curve, type_curves=tc, enhanced=enhanced)

    if labels is not None:
        long_labels = labels.melt(
            id_vars=["kst_dtm"],
            value_vars=["kpx_group_1", "kpx_group_2", "kpx_group_3"],
            var_name="group_col",
            value_name="power_kwh",
        )
        long_labels["group_id"] = long_labels["group_col"].str.extract(r"(\d+)").astype(int)
        df = df.merge(
            long_labels,
            left_on=["forecast_kst_dtm", "group_id"],
            right_on=["kst_dtm", "group_id"],
            how="left",
        ).drop(columns=["kst_dtm", "group_col"])
    return df


def merge_weather_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) == 1:
        return frames[0]
    base = frames[0]
    time_keys = ["forecast_kst_dtm", "group_id", "data_available_kst_dtm"]
    for add in frames[1:]:
        overlap = set(base.columns) & set(add.columns) - set(time_keys)
        add = add.drop(columns=[c for c in overlap if c in add.columns], errors="ignore")
        base = base.merge(add, on=time_keys, how="left")
    return base


def get_feature_columns(df: pd.DataFrame, prefixes: list[str] | None = None) -> list[str]:
    exclude = {
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "group_id",
        "power_kwh",
        "kst_dtm",
    }
    shared = {
        "hour", "month", "clim_power_kwh", "scada_prior_kwh",
        "dayofweek", "is_weekend", "season",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "ldaps_ws_hub_blend", "ldaps_ws_hub_blend_sq",
        "in_ws_sweet_spot", "ws_sweet_spot_x_prior",
        "prior_util", "clim_util", "clim_prior_ratio",
        "winter_hour", "winter_x_ws", "worst_slot_x_ws",
        "diff_x_sweet", "prior_x_hub_ws", "prior_x_ws_sq",
        "wd_sin_x_ws", "wd_cos_x_ws", "n_turbines", "is_group3",
        "prior_vestas_kwh", "prior_unison_kwh", "type_prior_kwh",
        "group_prior_kwh", "group_prior_util", "prior_gap",
    }
    both_only = {
        "ws10_diff_ldaps_gfs", "ws10_ratio_ldaps_gfs",
        "blend_ws10", "blend_ws_hub",
    }

    cols = []
    for col in df.columns:
        if col in exclude or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if prefixes is None:
            cols.append(col)
            continue
        if col in both_only:
            if prefixes == ["ldaps"] or prefixes == ["gfs"]:
                continue
            cols.append(col)
            continue
        if col in shared or any(col.startswith(p) for p in prefixes):
            cols.append(col)
            continue
        if any(col.startswith(p) and (col.endswith("_sq") or col.endswith("_cu")) for p in prefixes):
            cols.append(col)
    return cols
