"""풍속 구간 기반 FICR 보정 (과소예측 완화)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GROUP_COLUMNS, GROUP_CAPACITY_KWH
from src.metrics import evaluate_submission

WS_LOW = 5.0
WS_HIGH = 12.0
DEFAULT_WS_COL = "ldaps_ws_hub_blend"
MULT_GRID = [1.0, 1.03, 1.05, 1.08, 1.10, 1.12]
MULT_GRID_CONSERVATIVE = [1.0, 1.03, 1.05, 1.08]

# EDA 3단계: FICR 최악 월·시간
WORST_MONTHS = {1, 2, 3, 11}
WINTER_MONTHS = {1, 2, 3}
WORST_HOURS = {10, 20, 22}
SLOT_MULT_GRID = [1.0, 1.02, 1.04, 1.06]


def _ws_series(long_df: pd.DataFrame, ws_col: str) -> pd.Series:
    if ws_col in long_df.columns:
        return long_df[ws_col]
    if "ldaps_ws10" in long_df.columns:
        return long_df["ldaps_ws10"]
    if "blend_ws10" in long_df.columns:
        return long_df["blend_ws10"]
    raise KeyError(f"wind speed column not found: {ws_col}")


def apply_ws_band_multipliers(
    pred_wide: pd.DataFrame,
    long_df: pd.DataFrame,
    multipliers: dict[int, float],
    ws_col: str = DEFAULT_WS_COL,
) -> pd.DataFrame:
    """풍속 5~12 m/s 구간에서 그룹별 승수 적용."""
    out = pred_wide.copy()
    ws = _ws_series(long_df, ws_col)
    long = long_df[["forecast_kst_dtm", "group_id"]].copy()
    long["ws"] = ws.to_numpy()

    for gid, mult in multipliers.items():
        if mult == 1.0:
            continue
        col = f"kpx_group_{gid}"
        cap = GROUP_CAPACITY_KWH[col]
        mask_long = (long["group_id"] == gid) & (long["ws"] >= WS_LOW) & (long["ws"] <= WS_HIGH)
        boost_times = long.loc[mask_long, "forecast_kst_dtm"].unique()
        idx = out["forecast_kst_dtm"].isin(boost_times)
        out.loc[idx, col] = np.clip(out.loc[idx, col] * mult, 0, cap)
    return out


def tune_ws_band_multipliers(
    pred_wide: pd.DataFrame,
    long_df: pd.DataFrame,
    valid_true: pd.DataFrame,
    ws_col: str = DEFAULT_WS_COL,
    mult_grid: list[float] | None = None,
) -> tuple[dict[int, float], dict[str, float]]:
    """그룹별 greedy 탐색으로 풍속 밴드 승수 튜닝."""
    if mult_grid is None:
        mult_grid = MULT_GRID

    best = {1: 1.0, 2: 1.0, 3: 1.0}
    best_scores = {"score": -1.0}

    for gid in [1, 2, 3]:
        for mult in mult_grid:
            trial = dict(best)
            trial[gid] = mult
            adjusted = apply_ws_band_multipliers(pred_wide, long_df, trial, ws_col=ws_col)
            scores = evaluate_submission(valid_true, adjusted, time_col="kst_dtm")
            if scores["score"] > best_scores["score"]:
                best_scores = scores
                best[gid] = mult

    return best, best_scores


def apply_ws_band_multipliers_conditional(
    pred_wide: pd.DataFrame,
    long_df: pd.DataFrame,
    multipliers: dict[int, float],
    ws_col: str = DEFAULT_WS_COL,
    prior_col: str = "scada_prior_kwh",
    prior_ratio: float = 0.92,
) -> pd.DataFrame:
    """풍속 5~12 m/s + 예측이 SCADA prior보다 낮을 때만 승수 적용 (과보정 완화)."""
    out = pred_wide.copy()
    ws = _ws_series(long_df, ws_col)
    long = long_df[["forecast_kst_dtm", "group_id"]].copy()
    long["ws"] = ws.to_numpy()
    if prior_col in long_df.columns:
        long["prior"] = long_df[prior_col].to_numpy()
    else:
        return apply_ws_band_multipliers(out, long_df, multipliers, ws_col=ws_col)

    for gid, mult in multipliers.items():
        if mult == 1.0:
            continue
        col = f"kpx_group_{gid}"
        cap = GROUP_CAPACITY_KWH[col]
        mask_long = (
            (long["group_id"] == gid)
            & (long["ws"] >= WS_LOW)
            & (long["ws"] <= WS_HIGH)
        )
        for _, row in long.loc[mask_long].iterrows():
            t = row["forecast_kst_dtm"]
            idx = out["forecast_kst_dtm"] == t
            if not idx.any():
                continue
            raw = float(out.loc[idx, col].iloc[0])
            if raw < row["prior"] * prior_ratio:
                out.loc[idx, col] = np.clip(raw * mult, 0, cap)
    return out


def tune_ws_band_multipliers_conditional(
    pred_wide: pd.DataFrame,
    long_df: pd.DataFrame,
    valid_true: pd.DataFrame,
    ws_col: str = DEFAULT_WS_COL,
    mult_grid: list[float] | None = None,
) -> tuple[dict[int, float], dict[str, float]]:
    if mult_grid is None:
        mult_grid = MULT_GRID_CONSERVATIVE
    best = {1: 1.0, 2: 1.0, 3: 1.0}
    best_scores = {"score": -1.0}
    for gid in [1, 2, 3]:
        for mult in mult_grid:
            trial = dict(best)
            trial[gid] = mult
            adjusted = apply_ws_band_multipliers_conditional(
                pred_wide, long_df, trial, ws_col=ws_col
            )
            scores = evaluate_submission(valid_true, adjusted, time_col="kst_dtm")
            if scores["score"] > best_scores["score"]:
                best_scores = scores
                best[gid] = mult
    return best, best_scores


def apply_slot_multipliers(
    pred_wide: pd.DataFrame,
    multipliers: dict[int, float],
    worst_months: set[int] | None = None,
    worst_hours: set[int] | None = None,
) -> pd.DataFrame:
    """FICR 최악 월·시간대에 그룹별 승수 적용."""
    if worst_months is None:
        worst_months = WORST_MONTHS
    if worst_hours is None:
        worst_hours = WORST_HOURS

    out = pred_wide.copy()
    ts = pd.to_datetime(out["forecast_kst_dtm"])
    slot_mask = ts.dt.month.isin(worst_months) & ts.dt.hour.isin(worst_hours)

    for gid, mult in multipliers.items():
        if mult == 1.0:
            continue
        col = f"kpx_group_{gid}"
        cap = GROUP_CAPACITY_KWH[col]
        out.loc[slot_mask, col] = np.clip(out.loc[slot_mask, col] * mult, 0, cap)
    return out


def tune_slot_multipliers(
    pred_wide: pd.DataFrame,
    valid_true: pd.DataFrame,
    mult_grid: list[float] | None = None,
    worst_months: set[int] | None = None,
    worst_hours: set[int] | None = None,
) -> tuple[dict[int, float], dict[str, float]]:
    if mult_grid is None:
        mult_grid = SLOT_MULT_GRID
    best = {1: 1.0, 2: 1.0, 3: 1.0}
    best_scores = {"score": -1.0}
    for gid in [1, 2, 3]:
        for mult in mult_grid:
            trial = dict(best)
            trial[gid] = mult
            adjusted = apply_slot_multipliers(
                pred_wide, trial, worst_months=worst_months, worst_hours=worst_hours
            )
            scores = evaluate_submission(valid_true, adjusted, time_col="kst_dtm")
            if scores["score"] > best_scores["score"]:
                best_scores = scores
                best[gid] = mult
    return best, best_scores


def apply_v04_calibrations(
    pred_wide: pd.DataFrame,
    long_df: pd.DataFrame,
    ws_mult: dict[int, float],
    slot_mult: dict[int, float] | None = None,
    winter_only: bool = False,
) -> pd.DataFrame:
    out = apply_ws_band_multipliers(pred_wide, long_df, ws_mult)
    if not slot_mult:
        return out
    months = WINTER_MONTHS if winter_only else WORST_MONTHS
    return apply_slot_multipliers(out, slot_mult, worst_months=months)
