"""학습/검증 프레임 정렬 유틸."""

from __future__ import annotations

import pandas as pd

KEYS = ["forecast_kst_dtm", "group_id"]


def valid_split_frames(
    train_near: pd.DataFrame,
    train_idw: pd.DataFrame,
    valid_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """nearest/idw valid 구간을 (시간, 그룹) 키로 정렬해 동일 행끼리 맞춤."""
    fit_mask = train_idw["forecast_kst_dtm"] < valid_start
    val_mask = ~fit_mask
    idw_val = train_idw.loc[val_mask].sort_values(KEYS).reset_index(drop=True)
    near_val = (
        train_near.loc[train_near["forecast_kst_dtm"] >= valid_start]
        .merge(idw_val[KEYS], on=KEYS, how="inner")
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    return (
        train_near.loc[fit_mask].reset_index(drop=True),
        train_idw.loc[fit_mask].reset_index(drop=True),
        near_val,
        idw_val,
    )
