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
