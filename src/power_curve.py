from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, PATHS
from src.data_loader import load_turbine_info, read_csv

# vestas 12기: 그룹1=wtg01-06, 그룹2=wtg07-12 / unison 5기: 그룹3
VESTAS_GROUP_WTGS = {1: list(range(1, 7)), 2: list(range(7, 13))}
UNISON_GROUP_WTGS = {3: list(range(1, 6))}
GROUP_TURBINE_COUNT = {1: 6, 2: 6, 3: 5}


def build_month_hour_climatology(
    labels: pd.DataFrame,
    before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """그룹×월×시간 발전량 중앙값 (학습 기간 통계)."""
    df = labels.copy()
    if before is not None:
        df = df[df["kst_dtm"] < before]
    df["month"] = df["kst_dtm"].dt.month
    df["hour"] = df["kst_dtm"].dt.hour

    rows = []
    for col in GROUP_COLUMNS:
        gid = int(col.split("_")[-1])
        part = df[["month", "hour", col]].rename(columns={col: "power_kwh"})
        part["group_id"] = gid
        rows.append(part)

    long_df = pd.concat(rows, ignore_index=True).dropna(subset=["power_kwh"])
    clim = (
        long_df.groupby(["group_id", "month", "hour"], as_index=False)["power_kwh"]
        .median()
        .rename(columns={"power_kwh": "clim_power_kwh"})
    )
    return clim


def _turbine_cols(prefix: str, wtg_no: int) -> tuple[str, str]:
    tag = f"{prefix}_wtg{wtg_no:02d}"
    return f"{tag}_ws", f"{tag}_power_kw10m"


def _group_scada_long(
    df: pd.DataFrame,
    prefix: str,
    group_wtgs: dict[int, list[int]],
) -> pd.DataFrame:
    """터빈별 SCADA를 그룹 합산 출력(kW) long 포맷으로 변환."""
    parts = []
    for gid, wtgs in group_wtgs.items():
        ws_cols, pw_cols = [], []
        for wtg in wtgs:
            ws_c, pw_c = _turbine_cols(prefix, wtg)
            if ws_c in df.columns and pw_c in df.columns:
                ws_cols.append(ws_c)
                pw_cols.append(pw_c)
        if not pw_cols:
            continue
        part = pd.DataFrame(
            {
                "kst_dtm": df["kst_dtm"],
                "group_id": gid,
                "ws": df[ws_cols].mean(axis=1),
                "power_kw": df[pw_cols].sum(axis=1),
            }
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_group_scada_curves() -> pd.DataFrame:
    """
    그룹×월×풍속bin SCADA 출력 중앙값 (kWh 스케일 prior용).
    vestas/unison 터빈을 KPX 그룹에 맞게 합산 후 통계 산출.
    """
    vestas = read_csv(PATHS["scada_vestas"])
    unison = read_csv(PATHS["scada_unison"])
    vestas["kst_dtm"] = pd.to_datetime(vestas["kst_dtm"])
    unison["kst_dtm"] = pd.to_datetime(unison["kst_dtm"])

    long_df = pd.concat(
        [
            _group_scada_long(vestas, "vestas", VESTAS_GROUP_WTGS),
            _group_scada_long(unison, "unison", UNISON_GROUP_WTGS),
        ],
        ignore_index=True,
    )
    long_df = long_df.dropna()
    long_df["month"] = long_df["kst_dtm"].dt.month
    long_df["ws_bin"] = (long_df["ws"] // 1).clip(0, 25)

    curve = (
        long_df.groupby(["group_id", "month", "ws_bin"], as_index=False)["power_kw"]
        .median()
        .rename(columns={"power_kw": "scada_group_power_kw"})
    )
    return curve


def build_scada_monthly_curve() -> pd.DataFrame:
    """
    레거시 호환: 전체 터빈 평균 월×풍속bin 곡선.
    신규 코드는 build_group_scada_curves() 사용 권장.
    """
    group_curve = build_group_scada_curves()
    legacy = (
        group_curve.groupby(["month", "ws_bin"], as_index=False)["scada_group_power_kw"]
        .mean()
        .rename(columns={"scada_group_power_kw": "scada_power_kw"})
    )
    return legacy


def scada_prior_from_wind(
    frame: pd.DataFrame,
    curve: pd.DataFrame,
    ws_col: str = "blend_ws10",
    group_col: str = "group_id",
) -> pd.Series:
    """월×풍속 bin×그룹 SCADA prior (kWh, 그룹 합산 출력)."""
    tmp = frame.copy()
    tmp["month"] = tmp["forecast_kst_dtm"].dt.month
    tmp["ws_bin"] = (tmp[ws_col].fillna(0) // 1).clip(0, 25)

    if "group_id" in curve.columns:
        merged = tmp.merge(
            curve,
            on=[group_col, "month", "ws_bin"],
            how="left",
        )
        return merged["scada_group_power_kw"].fillna(0)

    merged = tmp.merge(curve, on=["month", "ws_bin"], how="left")
    n_turb = tmp[group_col].map(GROUP_TURBINE_COUNT).fillna(17)
    return merged["scada_power_kw"].fillna(0) * n_turb


def build_group_type_curves() -> dict[str, pd.DataFrame]:
    """vestas / unison 타입별 월×풍속bin 터빈평균 곡선 (교차 피처용)."""
    curves = {}
    vestas = read_csv(PATHS["scada_vestas"])
    unison = read_csv(PATHS["scada_unison"])
    for name, df, prefix in [
        ("vestas", vestas, "vestas"),
        ("unison", unison, "unison"),
    ]:
        df = df.copy()
        df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
        ws_cols = [c for c in df.columns if c.endswith("_ws")]
        pw_cols = [c for c in df.columns if "power" in c]
        part = pd.DataFrame(
            {
                "kst_dtm": df["kst_dtm"],
                "ws": df[ws_cols].mean(axis=1),
                "power_kw": df[pw_cols].mean(axis=1),
            }
        ).dropna()
        part["month"] = part["kst_dtm"].dt.month
        part["ws_bin"] = (part["ws"] // 1).clip(0, 25)
        curves[name] = (
            part.groupby(["month", "ws_bin"], as_index=False)["power_kw"]
            .median()
            .rename(columns={"power_kw": f"{prefix}_type_power_kw"})
        )
    return curves


def type_prior_from_wind(
    frame: pd.DataFrame,
    type_curves: dict[str, pd.DataFrame],
    ws_col: str = "blend_ws10",
) -> pd.DataFrame:
    """그룹 터빈 타입(vestas/unison)에 맞는 타입별 prior."""
    tmp = frame.copy()
    tmp["month"] = tmp["forecast_kst_dtm"].dt.month
    tmp["ws_bin"] = (tmp[ws_col].fillna(0) // 1).clip(0, 25)
    tmp["turbine_type"] = np.where(tmp["group_id"].isin([1, 2]), "vestas", "unison")
    tmp["n_turbines"] = tmp["group_id"].map(GROUP_TURBINE_COUNT)

    out = tmp[["forecast_kst_dtm", "group_id"]].copy()
    for ttype, curve in type_curves.items():
        col = f"prior_{ttype}_kwh"
        merged = tmp.merge(curve, on=["month", "ws_bin"], how="left")
        power_col = [c for c in curve.columns if c.endswith("_power_kw")][0]
        scaled = np.where(
            merged["turbine_type"] == ttype,
            merged[power_col].fillna(0) * merged["n_turbines"],
            0.0,
        )
        out[col] = scaled
    out["type_prior_kwh"] = out["prior_vestas_kwh"] + out["prior_unison_kwh"]
    return out
