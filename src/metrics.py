"""
대회 공식 평가 산식 (평가_산식 코드.ipynb 와 동일 로직)

Score = 0.5 * (1 - NMAE) + 0.5 * FICR
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS

GENERATION_FLOOR_RATIO = 0.10
FICR_TIER_1 = 0.06  # 6% 이하 → 4원/kWh
FICR_TIER_2 = 0.08  # 8% 이하 → 3원/kWh
PRICE_TIER_1 = 4.0
PRICE_TIER_2 = 3.0


def metric(answer_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    공식 metric 함수.

    answer_df, pred_df: kpx_group_1/2/3 컬럼 포함 (행 순서·길이 동일 가정)
    """
    group_nmae = []
    group_ficr = []

    for col in GROUP_COLUMNS:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        capacity = GROUP_CAPACITY_KWH[col]

        valid = actual >= capacity * GENERATION_FLOOR_RATIO
        actual = actual[valid]
        forecast = forecast[valid]

        if len(actual) == 0:
            group_nmae.append(np.nan)
            group_ficr.append(np.nan)
            continue

        error_rate = np.abs(forecast - actual) / capacity
        group_nmae.append(float(np.mean(error_rate)))

        unit_price = np.select(
            [error_rate <= FICR_TIER_1, error_rate <= FICR_TIER_2],
            [PRICE_TIER_1, PRICE_TIER_2],
            default=0.0,
        )
        earned_settlement = np.sum(actual * unit_price)
        max_settlement = np.sum(actual * PRICE_TIER_1)
        group_ficr.append(float(earned_settlement / max_settlement))

    one_minus_nmae = 1.0 - float(np.nanmean(group_nmae))
    ficr = float(np.nanmean(group_ficr))
    total_score = 0.5 * one_minus_nmae + 0.5 * ficr
    return total_score, one_minus_nmae, ficr


def _align_wide(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    time_col_true: str,
    time_col_pred: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    true = y_true.copy()
    pred = y_pred.copy()
    true[time_col_true] = pd.to_datetime(true[time_col_true])
    pred[time_col_pred] = pd.to_datetime(pred[time_col_pred])

    true = true.set_index(time_col_true).sort_index()
    pred = pred.set_index(time_col_pred).sort_index()
    common = true.index.intersection(pred.index)
    return true.loc[common, GROUP_COLUMNS], pred.loc[common, GROUP_COLUMNS]


def evaluate_submission(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    time_col: str = "kst_dtm",
) -> dict[str, float]:
    """시간 컬럼 기준 align 후 공식 metric 호출."""
    true_time_col = time_col if time_col in y_true.columns else "forecast_kst_dtm"
    pred_time_col = "forecast_kst_dtm" if "forecast_kst_dtm" in y_pred.columns else true_time_col

    answer, pred = _align_wide(y_true, y_pred, true_time_col, pred_time_col)
    # group_3 등 결측 라벨 제거
    keep = answer.notna().all(axis=1)
    answer = answer.loc[keep].reset_index(drop=True)
    pred = pred.loc[keep].reset_index(drop=True)

    total_score, one_minus_nmae, ficr = metric(answer, pred)
    return {
        "score": total_score,
        "1_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "n_rows": len(answer),
    }
