from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, PATHS
from src.data_loader import read_csv


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


def build_scada_monthly_curve() -> pd.DataFrame:
    """
    SCADA 풍속-출력 월별 보정 계수.
    test 예측 입력으로 SCADA는 쓰지 않고, 학습 기간 통계만 피처/후처리에 활용.
    """
    vestas = read_csv(PATHS["scada_vestas"])
    unison = read_csv(PATHS["scada_unison"])
    vestas["kst_dtm"] = pd.to_datetime(vestas["kst_dtm"])
    unison["kst_dtm"] = pd.to_datetime(unison["kst_dtm"])

    def _aggregate(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        ws_cols = [c for c in df.columns if c.endswith("_ws")]
        pw_cols = [c for c in df.columns if "power" in c]
        ws = df[ws_cols].mean(axis=1)
        pw = df[pw_cols].mean(axis=1)
        out = pd.DataFrame(
            {
                "kst_dtm": df["kst_dtm"],
                f"{prefix}_ws": ws,
                f"{prefix}_power": pw,
            }
        )
        return out

    scada = pd.concat(
        [_aggregate(vestas, "vestas"), _aggregate(unison, "unison")],
        ignore_index=True,
    )
    scada["month"] = scada["kst_dtm"].dt.month
    scada["ws"] = scada[["vestas_ws", "unison_ws"]].mean(axis=1)
    scada["power"] = scada[["vestas_power", "unison_power"]].mean(axis=1)
    scada = scada.dropna()
    scada["ws_bin"] = (scada["ws"] // 1).clip(0, 25)

    curve = (
        scada.groupby(["month", "ws_bin"], as_index=False)["power"]
        .median()
        .rename(columns={"power": "scada_power_kw"})
    )
    return curve


def scada_prior_from_wind(
    frame: pd.DataFrame,
    curve: pd.DataFrame,
    ws_col: str = "blend_ws10",
) -> pd.Series:
    """월×풍속 bin 기준 SCADA 출력 prior (kW, 터빈 평균)."""
    tmp = frame.copy()
    tmp["month"] = tmp["forecast_kst_dtm"].dt.month
    tmp["ws_bin"] = (tmp[ws_col].fillna(0) // 1).clip(0, 25)
    merged = tmp.merge(curve, on=["month", "ws_bin"], how="left")
    # 17터빈 평균 kW -> 대략 그룹 스케일 prior (경험적 스케일)
    return merged["scada_power_kw"].fillna(0) * 17.0
