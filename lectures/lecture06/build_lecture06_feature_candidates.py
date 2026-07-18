from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


TIME_KEYS = ["forecast_kst_dtm", "data_available_kst_dtm"]
GROUP_KEYS = [*TIME_KEYS, "group_id"]
EARTH_RADIUS_KM = 6371.0088
RD = 287.05

LDAPS_VECTOR_PAIRS = [
    ("ws10", "heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    ("ws_bl5", "heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS"),
]
LDAPS_PROXY_PAIRS = [
    ("ws50_component_max_proxy", "heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax"),
    ("ws50_component_min_proxy", "heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin"),
]
GFS_VECTOR_PAIRS = [
    ("ws10", "heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    ("ws80", "heightAboveGround_80_u", "heightAboveGround_80_v"),
    ("ws100", "heightAboveGround_100_100u", "heightAboveGround_100_100v"),
    ("ws_pbl", "planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"),
    ("ws850", "isobaricInhPa_850_u", "isobaricInhPa_850_v"),
    ("ws700", "isobaricInhPa_700_u", "isobaricInhPa_700_v"),
    ("ws500", "isobaricInhPa_500_u", "isobaricInhPa_500_v"),
]
DIFFERENCE_PAIRS = [
    ("10u", "heightAboveGround_10_10u", "heightAboveGround_10_10u"),
    ("10v", "heightAboveGround_10_10v", "heightAboveGround_10_10v"),
    ("ws10", "ws10", "ws10"),
    ("t2m", "heightAboveGround_2_t", "heightAboveGround_2_2t"),
    ("dpt2m", "heightAboveGround_2_dpt", "heightAboveGround_2_2d"),
    ("rh2m", "heightAboveGround_2_r", "heightAboveGround_2_2r"),
    ("surface_pressure", "surface_0_sp", "surface_0_sp"),
    ("sea_level_pressure", "meanSea_0_prmsl", "meanSea_0_prmsl"),
]
EXPECTED_SHAPES = {
    "time_features": {"train": (26_304, 13), "test": (8_760, 13)},
    "calendar_features": {"train": (26_304, 5), "test": (8_760, 5)},
    "wind_grid_features": {"train": (26_304, 573), "test": (8_760, 573)},
    "grid_statistics": {"train": (26_304, 372), "test": (8_760, 372)},
    "physical_grid_features": {"train": (26_304, 404), "test": (8_760, 404)},
    "center_nearest": {"train": (78_912, 119), "test": (26_280, 119)},
    "turbine_nearest": {"train": (78_912, 119), "test": (26_280, 119)},
    "idw_p1": {"train": (78_912, 119), "test": (26_280, 119)},
    "idw_p2": {"train": (78_912, 119), "test": (26_280, 119)},
    "model_difference_center_nearest": {"train": (78_912, 11), "test": (26_280, 11)},
    "model_difference_turbine_nearest": {"train": (78_912, 11), "test": (26_280, 11)},
    "model_difference_idw_p1": {"train": (78_912, 11), "test": (26_280, 11)},
    "model_difference_idw_p2": {"train": (78_912, 11), "test": (26_280, 11)},
}


def load_lecture05_builder(project_root: Path):
    path = project_root / "lectures" / "lecture05" / "build_lecture05_master_data.py"
    spec = importlib.util.spec_from_file_location("lecture05_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_turbine_metadata(turbine_metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"KPX그룹", "turbine_latitude", "turbine_longitude"}
    missing = required - set(turbine_metadata.columns)
    if missing:
        raise ValueError(f"터빈 메타데이터 필수 컬럼이 없습니다: {sorted(missing)}")
    result = turbine_metadata.rename(columns={"KPX그룹": "group_id"}).copy()
    result["group_id"] = pd.to_numeric(result["group_id"], errors="raise").astype("int8")
    if result[["group_id", "turbine_latitude", "turbine_longitude"]].isna().any().any():
        raise ValueError("터빈 그룹 또는 좌표에 결측이 있습니다.")
    return result


def build_time_features(forecast_index: pd.DataFrame) -> pd.DataFrame:
    result = forecast_index.copy()
    ft = result["forecast_kst_dtm"]
    at = result["data_available_kst_dtm"]
    result["year"] = ft.dt.year.astype("int16")
    result["month"] = ft.dt.month.astype("int8")
    result["hour"] = ft.dt.hour.astype("int8")
    result["dayofyear"] = ft.dt.dayofyear.astype("int16")
    result["lead_hour"] = ((ft - at).dt.total_seconds() / 3600.0).astype("float32")
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24.0)
    result["dayofyear_sin"] = np.sin(2 * np.pi * (result["dayofyear"] - 1) / 365.2425)
    result["dayofyear_cos"] = np.cos(2 * np.pi * (result["dayofyear"] - 1) / 365.2425)
    result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12.0)
    return result


def build_calendar_features(forecast_index: pd.DataFrame) -> pd.DataFrame:
    result = forecast_index.copy()
    ft = result["forecast_kst_dtm"]
    result["day_of_month"] = ft.dt.day.astype("int8")
    result["dayofweek"] = ft.dt.dayofweek.astype("int8")
    result["is_weekend"] = result["dayofweek"].ge(5).astype("int8")
    return result


def add_wind_candidates(weather: pd.DataFrame, vector_pairs: list[tuple], proxy_pairs: list[tuple] | None = None):
    result = weather.copy()
    added = []
    for name, u_col, v_col in vector_pairs:
        speed = np.hypot(result[u_col], result[v_col])
        result[name] = speed
        result[f"{name}_sq"] = speed**2
        result[f"{name}_cube"] = speed**3
        result[f"{name}_unit_u"] = result[u_col] / speed.where(speed > 1e-12)
        result[f"{name}_unit_v"] = result[v_col] / speed.where(speed > 1e-12)
        added += [name, f"{name}_sq", f"{name}_cube", f"{name}_unit_u", f"{name}_unit_v"]
    for name, u_col, v_col in proxy_pairs or []:
        speed = np.hypot(result[u_col], result[v_col])
        result[name] = speed
        result[f"{name}_sq"] = speed**2
        result[f"{name}_cube"] = speed**3
        added += [name, f"{name}_sq", f"{name}_cube"]
    return result, added


def candidate_values_to_wide(weather: pd.DataFrame, value_columns: list[str], prefix: str, make_grid_token) -> pd.DataFrame:
    indexed = weather.set_index([*TIME_KEYS, "grid_id"])[value_columns]
    if indexed.index.duplicated().any():
        raise ValueError("wide 변환 전 중복 키가 있습니다.")
    wide = indexed.unstack("grid_id")
    wide.columns = [f"{prefix}_g{make_grid_token(grid_id)}_{variable}" for variable, grid_id in wide.columns]
    return wide.reset_index().sort_values(TIME_KEYS).reset_index(drop=True)


def build_grid_statistics(weather: pd.DataFrame, value_columns: list[str], prefix: str) -> pd.DataFrame:
    grouped = weather.groupby(TIME_KEYS, sort=True)[value_columns]
    complete = grouped.count().eq(weather["grid_id"].nunique())
    mean = grouped.mean().where(complete)
    std = grouped.std(ddof=0).where(complete)
    minimum = grouped.min().where(complete)
    maximum = grouped.max().where(complete)
    result = pd.concat(
        [
            mean.add_suffix("__mean"),
            std.add_suffix("__std"),
            minimum.add_suffix("__min"),
            maximum.add_suffix("__max"),
            (maximum - minimum).add_suffix("__range"),
        ],
        axis=1,
    )
    result.columns = [f"{prefix}_{column}" for column in result.columns]
    return result.reset_index()


def haversine_distance_km(lat1, lon1, lat2, lon2):
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(value))


def build_center_nearest_weights(turbines: pd.DataFrame, grid: pd.DataFrame, source: str) -> pd.DataFrame:
    records = []
    grid_lat, grid_lon = grid["latitude"].to_numpy(), grid["longitude"].to_numpy()
    for group_id, group in turbines.groupby("group_id"):
        distances = haversine_distance_km(group["turbine_latitude"].mean(), group["turbine_longitude"].mean(), grid_lat, grid_lon)
        nearest = int(np.argmin(distances))
        for index, grid_id in enumerate(grid["grid_id"]):
            records.append({"source": source, "method": "center_nearest", "group_id": int(group_id), "grid_id": grid_id, "weight": float(index == nearest)})
    return pd.DataFrame(records)


def build_turbine_nearest_weights(turbines: pd.DataFrame, grid: pd.DataFrame, source: str) -> pd.DataFrame:
    records = []
    grid_lat, grid_lon = grid["latitude"].to_numpy(), grid["longitude"].to_numpy()
    for group_id, group in turbines.groupby("group_id"):
        nearest_ids = []
        for _, turbine in group.iterrows():
            distances = haversine_distance_km(turbine["turbine_latitude"], turbine["turbine_longitude"], grid_lat, grid_lon)
            nearest_ids.append(grid["grid_id"].iloc[int(np.argmin(distances))])
        weights = pd.Series(nearest_ids).value_counts(normalize=True)
        for grid_id in grid["grid_id"]:
            records.append({"source": source, "method": "turbine_nearest", "group_id": int(group_id), "grid_id": grid_id, "weight": float(weights.get(grid_id, 0.0))})
    return pd.DataFrame(records)


def build_turbine_idw_weights(turbines: pd.DataFrame, grid: pd.DataFrame, source: str, power: float, epsilon: float = 1e-6) -> pd.DataFrame:
    records = []
    grid_lat, grid_lon = grid["latitude"].to_numpy(), grid["longitude"].to_numpy()
    for group_id, group in turbines.groupby("group_id"):
        vectors = []
        for _, turbine in group.iterrows():
            distances = haversine_distance_km(turbine["turbine_latitude"], turbine["turbine_longitude"], grid_lat, grid_lon)
            if (distances < epsilon).any():
                weights = (distances < epsilon).astype("float64")
            else:
                weights = distances ** (-power)
            vectors.append(weights / weights.sum())
        group_weights = np.vstack(vectors).mean(axis=0)
        group_weights /= group_weights.sum()
        for grid_id, weight in zip(grid["grid_id"], group_weights):
            records.append({"source": source, "method": f"idw_p{power:g}", "group_id": int(group_id), "grid_id": grid_id, "weight": float(weight)})
    return pd.DataFrame(records)


def weighted_project_long(weather: pd.DataFrame, weights: pd.DataFrame, value_columns: list[str], prefix: str) -> pd.DataFrame:
    positive = weights.loc[weights["weight"] > 0, ["group_id", "grid_id", "weight"]].copy()
    merged = weather[[*TIME_KEYS, "grid_id", *value_columns]].merge(positive, on="grid_id", how="inner", validate="many_to_many")
    weighted_values = merged[value_columns].multiply(merged["weight"], axis=0)
    weighted_table = pd.concat([merged[GROUP_KEYS], weighted_values], axis=1)
    weighted_sum = weighted_table.groupby(GROUP_KEYS, sort=True)[value_columns].sum(min_count=1)
    valid_counts = merged.groupby(GROUP_KEYS, sort=True)[value_columns].count()
    expected_counts = positive.groupby("group_id").size()
    expected = valid_counts.index.get_level_values("group_id").map(expected_counts).to_numpy()
    weighted_sum = weighted_sum.where(valid_counts.eq(expected[:, None]))
    weighted_sum.columns = [f"{prefix}_{column}" for column in weighted_sum.columns]
    return weighted_sum.reset_index()


def nullable_flag(source: pd.Series | pd.DataFrame, condition: pd.Series) -> pd.Series:
    missing = source.isna().any(axis=1) if isinstance(source, pd.DataFrame) else source.isna()
    return pd.Series(np.where(missing, np.nan, condition.astype("float32")), index=condition.index, dtype="float32")


def build_physical_ldaps(weather: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = weather[TIME_KEYS + ["grid_id"]].copy()
    rh = weather["heightAboveGround_2_r"]
    result = result.assign(
        t2m_c=weather["heightAboveGround_2_t"] - 273.15,
        dpt2m_c=weather["heightAboveGround_2_dpt"] - 273.15,
        temp_dewpoint_gap=weather["heightAboveGround_2_t"] - weather["heightAboveGround_2_dpt"],
        pressure_delta=weather["meanSea_0_prmsl"] - weather["surface_0_sp"],
        dry_air_density=weather["surface_0_sp"] / (RD * weather["heightAboveGround_2_t"]),
        rh_over_100_flag=nullable_flag(rh, rh > 100),
    )
    for column in ["etc_0_hcc", "etc_0_mcc", "etc_0_lcc", "etc_0_VLCDC"]:
        valid = weather[column].dropna()
        if not valid.between(0.0, 1.0001).all():
            raise ValueError(f"{column}: LDAPS 운량이 0~1 범위를 벗어났습니다.")
        result[f"{column}_fraction"] = weather[column].astype("float32")
    for column in ["surface_0_avg_lsprate", "surface_0_lssrate", "surface_0_ncpcp", "surface_0_snol", "surface_0_SNOM"]:
        result[f"{column}_positive_flag"] = nullable_flag(weather[column], weather[column] > 0)
    return result, [column for column in result.columns if column not in [*TIME_KEYS, "grid_id"]]


def build_physical_gfs(weather: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = weather[TIME_KEYS + ["grid_id"]].copy()
    virtual_temp = weather["heightAboveGround_2_2t"] * (1 + 0.61 * weather["heightAboveGround_2_2sh"])
    moist_density = weather["surface_0_sp"] / (RD * virtual_temp)
    result = result.assign(
        t2m_c=weather["heightAboveGround_2_2t"] - 273.15,
        dpt2m_c=weather["heightAboveGround_2_2d"] - 273.15,
        temp_dewpoint_gap=weather["heightAboveGround_2_2t"] - weather["heightAboveGround_2_2d"],
        pressure_delta=weather["meanSea_0_prmsl"] - weather["surface_0_sp"],
        dry_air_density=weather["surface_0_sp"] / (RD * weather["heightAboveGround_2_2t"]),
        moist_air_density=moist_density,
        gust_ws10_gap=weather["surface_0_gust"] - weather["ws10"],
        ws80_ws10_gap=weather["ws80"] - weather["ws10"],
        ws100_ws80_gap=weather["ws100"] - weather["ws80"],
        ws100_ws10_gap=weather["ws100"] - weather["ws10"],
        density_x_ws100_cube=moist_density * weather["ws100_cube"],
    )
    for column in ["lowCloudLayer_0_lcc", "middleCloudLayer_0_mcc", "highCloudLayer_0_hcc", "atmosphere_0_tcc"]:
        result[f"{column}_fraction"] = weather[column] / 100.0
    precip_sources = weather[["surface_0_prate", "surface_0_tp"]]
    result["precip_positive_flag"] = nullable_flag(precip_sources, (weather["surface_0_prate"] > 0) | (weather["surface_0_tp"] > 0))
    result["surface_0_prate_log1p"] = np.log1p(weather["surface_0_prate"].clip(lower=0))
    result["surface_0_tp_log1p"] = np.log1p(weather["surface_0_tp"].clip(lower=0))
    return result, [column for column in result.columns if column not in [*TIME_KEYS, "grid_id"]]


def build_model_difference(block: pd.DataFrame, method: str) -> pd.DataFrame:
    result = block[GROUP_KEYS].copy()
    for name, ldaps_col, gfs_col in DIFFERENCE_PAIRS:
        result[f"diff_{name}"] = block[f"ldaps_{ldaps_col}"] - block[f"gfs_{gfs_col}"]
    return result


def merge_sources(ldaps: pd.DataFrame, gfs: pd.DataFrame, keys: list[str] = TIME_KEYS) -> pd.DataFrame:
    return ldaps.merge(gfs, on=keys, validate="one_to_one")


def build_spatial_weights(turbines: pd.DataFrame, ldaps_grid: pd.DataFrame, gfs_grid: pd.DataFrame) -> pd.DataFrame:
    builders = [
        lambda grid, source: build_center_nearest_weights(turbines, grid, source),
        lambda grid, source: build_turbine_nearest_weights(turbines, grid, source),
        lambda grid, source: build_turbine_idw_weights(turbines, grid, source, 1),
        lambda grid, source: build_turbine_idw_weights(turbines, grid, source, 2),
    ]
    return pd.concat(
        [builder(grid, source) for builder in builders for grid, source in [(ldaps_grid, "ldaps"), (gfs_grid, "gfs")]],
        ignore_index=True,
    )


def project_group_block(ldaps: pd.DataFrame, gfs: pd.DataFrame, weights: pd.DataFrame, method: str, value_columns: dict[str, list[str]]) -> pd.DataFrame:
    source_blocks = []
    for source, weather in [("ldaps", ldaps), ("gfs", gfs)]:
        source_weights = weights[(weights["source"] == source) & (weights["method"] == method)]
        source_blocks.append(weighted_project_long(weather, source_weights, value_columns[source], source))
    return merge_sources(source_blocks[0], source_blocks[1], GROUP_KEYS)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def feature_columns(frame: pd.DataFrame, keys: list[str]) -> list[str]:
    return [column for column in frame.columns if column not in keys]


def registry_rows(block_name: str, frame: pd.DataFrame, keys: list[str], source: str, row_scope: str, spatial_method: str, group_specific: bool) -> list[dict[str, object]]:
    scope = "forecast_group" if row_scope == "group" else "forecast"
    parent_columns = registry_parent_columns(block_name)
    formula = registry_formula(block_name, spatial_method)
    base = {"block_name": block_name, "source": source, "row_scope": scope, "spatial_method": spatial_method, "group_specific": group_specific, "parent_columns": parent_columns, "formula": formula, "unit_status": "candidate", "fit_required": False, "label_used": False, "status": "candidate", "caveat": ""}
    return [{"feature_name": column, **base} for column in feature_columns(frame, keys)]


def registry_parent_columns(block_name: str) -> str:
    if block_name == "time_features":
        return "forecast_kst_dtm,data_available_kst_dtm"
    if block_name == "calendar_features":
        return "forecast_kst_dtm"
    if block_name == "wind_grid_features":
        return "u_component,v_component,grid_id"
    if block_name == "grid_statistics":
        return "weather_variables,wind_speed_candidates,grid_id"
    if block_name == "physical_grid_features":
        return "weather_variables,wind_speed_candidates,grid_id"
    if block_name.startswith("model_difference_"):
        return "ldaps_matched_feature,gfs_matched_feature"
    return "weather_variables,wind_speed_candidates,spatial_weights"


def registry_formula(block_name: str, spatial_method: str) -> str:
    if block_name == "time_features":
        return "calendar_and_lead_time"
    if block_name == "calendar_features":
        return "calendar_candidate_values"
    if block_name == "wind_grid_features":
        return "sqrt(u^2+v^2),unit_vector,speed_powers"
    if block_name == "grid_statistics":
        return "mean,std,min,max,range_across_grids"
    if block_name == "physical_grid_features":
        return "unit_conversion,density,cloud_fraction,precip_flags"
    if block_name.startswith("model_difference_"):
        return "ldaps_feature-gfs_feature"
    return f"weighted_projection:{spatial_method}"


def build_feature_candidates(project_root: Path):
    lecture05 = load_lecture05_builder(project_root)
    package = lecture05.build_raw_master_package(project_root)
    tables = package["tables"]
    make_grid_token = lecture05.make_grid_token

    turbines = normalize_turbine_metadata(tables["turbine_metadata"])
    ldaps_grid = tables["ldaps_grid_metadata"]
    gfs_grid = tables["gfs_grid_metadata"]
    ldaps_value_columns = package["manifest"]["ldaps_value_columns"]
    gfs_value_columns = package["manifest"]["gfs_value_columns"]

    time_train = build_time_features(tables["train_forecast_index"])
    time_test = build_time_features(tables["test_forecast_index"])
    calendar_train = build_calendar_features(tables["train_forecast_index"])
    calendar_test = build_calendar_features(tables["test_forecast_index"])

    ldaps_train_wind, ldaps_wind_cols = add_wind_candidates(tables["ldaps_train_long"], LDAPS_VECTOR_PAIRS, LDAPS_PROXY_PAIRS)
    ldaps_test_wind, _ = add_wind_candidates(tables["ldaps_test_long"], LDAPS_VECTOR_PAIRS, LDAPS_PROXY_PAIRS)
    gfs_train_wind, gfs_wind_cols = add_wind_candidates(tables["gfs_train_long"], GFS_VECTOR_PAIRS)
    gfs_test_wind, _ = add_wind_candidates(tables["gfs_test_long"], GFS_VECTOR_PAIRS)

    wind_grid_train = merge_sources(
        candidate_values_to_wide(ldaps_train_wind, ldaps_wind_cols, "ldaps", make_grid_token),
        candidate_values_to_wide(gfs_train_wind, gfs_wind_cols, "gfs", make_grid_token),
    )
    wind_grid_test = merge_sources(
        candidate_values_to_wide(ldaps_test_wind, ldaps_wind_cols, "ldaps", make_grid_token),
        candidate_values_to_wide(gfs_test_wind, gfs_wind_cols, "gfs", make_grid_token),
    )

    ldaps_stats_cols = [column for column in ldaps_value_columns if column not in {"surface_0_lsm", "surface_0_h"}] + [name for name, _, _ in LDAPS_VECTOR_PAIRS + LDAPS_PROXY_PAIRS]
    gfs_stats_cols = gfs_value_columns + [name for name, _, _ in GFS_VECTOR_PAIRS]
    grid_stats_train = merge_sources(
        build_grid_statistics(ldaps_train_wind, ldaps_stats_cols, "ldaps"),
        build_grid_statistics(gfs_train_wind, gfs_stats_cols, "gfs"),
    )
    grid_stats_test = merge_sources(
        build_grid_statistics(ldaps_test_wind, ldaps_stats_cols, "ldaps"),
        build_grid_statistics(gfs_test_wind, gfs_stats_cols, "gfs"),
    )

    ldaps_phys_train, ldaps_phys_cols = build_physical_ldaps(ldaps_train_wind)
    ldaps_phys_test, _ = build_physical_ldaps(ldaps_test_wind)
    gfs_phys_train, gfs_phys_cols = build_physical_gfs(gfs_train_wind)
    gfs_phys_test, _ = build_physical_gfs(gfs_test_wind)
    physical_train = merge_sources(
        candidate_values_to_wide(ldaps_phys_train, ldaps_phys_cols, "ldaps", make_grid_token),
        candidate_values_to_wide(gfs_phys_train, gfs_phys_cols, "gfs", make_grid_token),
    )
    physical_test = merge_sources(
        candidate_values_to_wide(ldaps_phys_test, ldaps_phys_cols, "ldaps", make_grid_token),
        candidate_values_to_wide(gfs_phys_test, gfs_phys_cols, "gfs", make_grid_token),
    )

    weights = build_spatial_weights(turbines, ldaps_grid, gfs_grid)

    group_train, group_test, diff_train, diff_test = {}, {}, {}, {}
    spatial_value_columns = {
        "ldaps": ldaps_value_columns + ldaps_wind_cols,
        "gfs": gfs_value_columns + gfs_wind_cols,
    }
    for method in ["center_nearest", "turbine_nearest", "idw_p1", "idw_p2"]:
        group_train[method] = project_group_block(ldaps_train_wind, gfs_train_wind, weights, method, spatial_value_columns)
        group_test[method] = project_group_block(ldaps_test_wind, gfs_test_wind, weights, method, spatial_value_columns)
        diff_train[method] = build_model_difference(group_train[method], method)
        diff_test[method] = build_model_difference(group_test[method], method)

    blocks = {
        "time_features": (time_train, time_test, TIME_KEYS, "shared", "time", "none", False),
        "calendar_features": (calendar_train, calendar_test, TIME_KEYS, "shared", "time", "none", False),
        "wind_grid_features": (wind_grid_train, wind_grid_test, TIME_KEYS, "shared", "weather", "raw_grid", False),
        "grid_statistics": (grid_stats_train, grid_stats_test, TIME_KEYS, "shared", "weather", "all_grid", False),
        "physical_grid_features": (physical_train, physical_test, TIME_KEYS, "shared", "weather", "raw_grid", False),
    }
    for method in group_train:
        blocks[method] = (group_train[method], group_test[method], GROUP_KEYS, "group", "weather", method, True)
        blocks[f"model_difference_{method}"] = (diff_train[method], diff_test[method], GROUP_KEYS, "group", "ldaps-gfs", method, True)

    audit_blocks(blocks, weights)
    registry = pd.DataFrame([row for name, (train, _, keys, scope, source, method, is_group) in blocks.items() for row in registry_rows(name, train, keys, source, scope, method, is_group)])
    manifest = build_manifest(blocks)
    audit = build_audit(blocks, weights, registry)
    return blocks, weights, registry, manifest, audit


def audit_blocks(blocks: dict, weights: pd.DataFrame) -> None:
    for name, (train, test, keys, *_rest) in blocks.items():
        if tuple(train.shape) != EXPECTED_SHAPES[name]["train"] or tuple(test.shape) != EXPECTED_SHAPES[name]["test"]:
            raise ValueError(f"{name}: 예상 shape와 다릅니다.")
        if train.duplicated(keys).any() or test.duplicated(keys).any():
            raise ValueError(f"{name}: key 중복이 있습니다.")
        if list(train.columns) != list(test.columns):
            raise ValueError(f"{name}: Train/Test 열 순서가 다릅니다.")
        if np.isinf(train.select_dtypes(include="number").to_numpy()).any() or np.isinf(test.select_dtypes(include="number").to_numpy()).any():
            raise ValueError(f"{name}: 무한값이 존재합니다.")
    weight_sums = weights.groupby(["source", "method", "group_id"])["weight"].sum()
    if not np.isclose(weight_sums.to_numpy(), 1.0, atol=1e-10).all():
        raise ValueError("공간 weight 합이 1이 아닙니다.")


def build_manifest(blocks: dict) -> dict[str, dict[str, object]]:
    manifest = {}
    for name, (train, _test, keys, scope, _source, _method, _is_group) in blocks.items():
        manifest[name] = {"row_scope": "forecast_group" if scope == "group" else "forecast", "key_columns": keys, "feature_count": len(feature_columns(train, keys))}
    return manifest


def build_audit(blocks: dict, weights: pd.DataFrame, registry: pd.DataFrame) -> dict[str, object]:
    return {
        "blocks": {
            name: {
                "train_shape": list(train.shape),
                "test_shape": list(test.shape),
                "train_missing_cells": int(train.isna().sum().sum()),
                "test_missing_cells": int(test.isna().sum().sum()),
                "test_missing_rows": int(test.isna().any(axis=1).sum()),
            }
            for name, (train, test, *_rest) in blocks.items()
        },
        "weight_sum_min": float(weights.groupby(["source", "method", "group_id"])["weight"].sum().min()),
        "weight_sum_max": float(weights.groupby(["source", "method", "group_id"])["weight"].sum().max()),
        "registry_rows": int(len(registry)),
        "registry_label_used_any": bool(registry["label_used"].any()),
        "registry_fit_required_any": bool(registry["fit_required"].any()),
        "expected_shapes_checked": True,
        "missing_policy": "preserve_nan_in_flags_and_require_complete_grid_statistics",
    }


def write_outputs(blocks: dict, weights: pd.DataFrame, registry: pd.DataFrame, manifest: dict, audit: dict, output_dir: Path, metadata_only: bool = False) -> None:
    shared_dir = output_dir / "shared"
    group_dir = output_dir / "group"
    metadata_dir = output_dir / "metadata"
    reports_dir = output_dir / "reports"
    for directory in [shared_dir, group_dir, metadata_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if not metadata_only:
        for name, (train, test, _keys, scope, *_rest) in blocks.items():
            target_dir = group_dir if scope == "group" else shared_dir
            save_csv(train, target_dir / f"{name}_train.csv")
            save_csv(test, target_dir / f"{name}_test.csv")
    save_csv(weights, metadata_dir / "spatial_weights.csv")
    save_csv(registry, metadata_dir / "feature_registry.csv")
    (metadata_dir / "block_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (reports_dir / "audit_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_audit_markdown(audit, reports_dir / "audit_summary.md")


def write_audit_markdown(audit: dict, path: Path) -> None:
    lines = ["# Lecture 06 Feature Candidate Audit", "", "| block | train shape | test shape | test missing rows | test missing cells |", "|---|---:|---:|---:|---:|"]
    for name, data in audit["blocks"].items():
        lines.append(f"| `{name}` | {data['train_shape']} | {data['test_shape']} | {data['test_missing_rows']:,} | {data['test_missing_cells']:,} |")
    lines += [
        "",
        f"- Weight sum min: {audit['weight_sum_min']:.12f}",
        f"- Weight sum max: {audit['weight_sum_max']:.12f}",
        f"- Registry rows: {audit['registry_rows']:,}",
        f"- Registry label_used any: {audit['registry_label_used_any']}",
        f"- Registry fit_required any: {audit['registry_fit_required_any']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Lecture 06 feature candidate blocks.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "lecture06_feature_candidates")
    parser.add_argument("--metadata-only", action="store_true", help="Write only metadata and audit files, not large feature block CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocks, weights, registry, manifest, audit = build_feature_candidates(args.project_root)
    write_outputs(blocks, weights, registry, manifest, audit, args.output_dir, metadata_only=args.metadata_only)
    print(json.dumps({name: data["train_shape"] for name, data in audit["blocks"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
