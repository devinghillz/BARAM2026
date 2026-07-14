from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PATHS

INFO_HEADER_ROW = 3
INFO_COL = {
    "turbine_no": 5,
    "coord": 6,
    "kpx_group": 7,
    "hub_height": 8,
    "capacity_mw": 10,
    "group_capacity_mw": 11,
}


def read_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_labels() -> pd.DataFrame:
    df = read_csv(PATHS["labels"])
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    return df.sort_values("kst_dtm").reset_index(drop=True)


def load_submission_template() -> pd.DataFrame:
    df = read_csv(PATHS["submission"])
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    return df


def _parse_dms_pair(coord: str) -> tuple[float, float]:
    """Google 좌표 문자열(도분초)을 (lat, lon) 십진수로 변환."""
    if not isinstance(coord, str):
        raise ValueError(f"Invalid coordinate: {coord}")

    parts = re.findall(
        r"(\d+)[°\u00b0](\d+)'([\d.]+)\"([NS])\s+(\d+)[°\u00b0](\d+)'([\d.]+)\"([EW])",
        coord,
    )
    if not parts:
        raise ValueError(f"Could not parse coordinate: {coord}")

    lat_d, lat_m, lat_s, lat_h, lon_d, lon_m, lon_s, lon_h = parts[0]
    lat = int(lat_d) + int(lat_m) / 60 + float(lat_s) / 3600
    lon = int(lon_d) + int(lon_m) / 60 + float(lon_s) / 3600
    if lat_h == "S":
        lat *= -1
    if lon_h == "W":
        lon *= -1
    return lat, lon


def load_turbine_info() -> pd.DataFrame:
    raw = pd.read_excel(PATHS["info"], header=INFO_HEADER_ROW)
    df = pd.DataFrame(
        {
            "turbine_no": raw.iloc[:, INFO_COL["turbine_no"]],
            "coord_raw": raw.iloc[:, INFO_COL["coord"]],
            "kpx_group": raw.iloc[:, INFO_COL["kpx_group"]].ffill().astype(int),
            "hub_height_m": raw.iloc[:, INFO_COL["hub_height"]],
            "capacity_mw": raw.iloc[:, INFO_COL["capacity_mw"]],
            "group_capacity_mw": raw.iloc[:, INFO_COL["group_capacity_mw"]].ffill(),
        }
    )
    lat_lon = df["coord_raw"].map(_parse_dms_pair)
    df["latitude"] = lat_lon.map(lambda x: x[0])
    df["longitude"] = lat_lon.map(lambda x: x[1])
    return df.drop(columns=["coord_raw"]).reset_index(drop=True)


def load_group_centroids() -> pd.DataFrame:
    info = load_turbine_info()
    centroids = (
        info.groupby("kpx_group", as_index=False)[["latitude", "longitude"]]
        .mean()
        .rename(columns={"kpx_group": "group_id"})
    )
    return centroids


def load_weather(source: str, split: str) -> pd.DataFrame:
    if source not in {"ldaps", "gfs"}:
        raise ValueError("source must be 'ldaps' or 'gfs'")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    path = PATHS[f"{source}_{split}"]
    df = read_csv(path)
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["data_available_kst_dtm"] = pd.to_datetime(df["data_available_kst_dtm"])
    return df.sort_values(["forecast_kst_dtm", "grid_id"]).reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """벡터화 haversine 거리(km)."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
