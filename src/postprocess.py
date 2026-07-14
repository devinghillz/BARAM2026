from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS
from src.metrics import evaluate_submission, metric
from src.power_curve import build_month_hour_climatology


def shrink_toward_climatology(
    pred_wide: pd.DataFrame,
    climatology: pd.DataFrame,
    shrink: float = 0.12,
) -> pd.DataFrame:
    """예측을 월×시간 기후학적 중앙값 쪽으로 수축 (큰 오차/FICR 페널티 완화)."""
    out = pred_wide.copy()
    out["month"] = out["forecast_kst_dtm"].dt.month
    out["hour"] = out["forecast_kst_dtm"].dt.hour

    for col in GROUP_COLUMNS:
        gid = int(col.split("_")[-1])
        cap = GROUP_CAPACITY_KWH[col]
        clim = climatology[climatology["group_id"] == gid]
        merged = out.merge(clim, on=["month", "hour"], how="left")
        raw = merged[col].to_numpy(dtype=float)
        climv = merged["clim_power_kwh"].to_numpy(dtype=float).copy()
        missing = np.isnan(climv)
        climv[missing] = raw[missing]
        adjusted = (1.0 - shrink) * raw + shrink * climv
        out[col] = np.clip(adjusted, 0, cap)

    return out.drop(columns=["month", "hour"])


def tune_shrink_on_validation(
    pred_wide: pd.DataFrame,
    y_true: pd.DataFrame,
    labels_for_clim: pd.DataFrame,
    valid_start: pd.Timestamp,
    shrink_grid: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """검증 구간에서 FICR/Score 최대화 shrink 탐색."""
    if shrink_grid is None:
        shrink_grid = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22]

    clim = build_month_hour_climatology(labels_for_clim, before=valid_start)
    best_shrink = 0.0
    best_scores = {"score": -1.0}

    true = y_true.copy()
    true["kst_dtm"] = pd.to_datetime(true["kst_dtm"])
    pred_base = pred_wide.copy()
    pred_base["forecast_kst_dtm"] = pd.to_datetime(pred_base["forecast_kst_dtm"])

    for shrink in shrink_grid:
        adjusted = shrink_toward_climatology(pred_base, clim, shrink=shrink)
        scores = evaluate_submission(true, adjusted, time_col="kst_dtm")
        total = scores["score"]
        if total > best_scores["score"]:
            best_scores = {**scores, "shrink": shrink}
            best_shrink = shrink

    return best_shrink, best_scores


def apply_ficr_postprocess(
    pred_wide: pd.DataFrame,
    labels: pd.DataFrame,
    valid_start: pd.Timestamp,
    shrink: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """학습 기간 기후학적 프로파일 기반 후처리."""
    if shrink is None:
        shrink, _ = tune_shrink_on_validation(
            pred_wide,
            labels.loc[labels["kst_dtm"] >= valid_start, ["kst_dtm", *GROUP_COLUMNS]],
            labels,
            valid_start,
        )
    clim = build_month_hour_climatology(labels, before=None)
    # 전체 라벨 기반 climatology로 test 후처리 (미래 라벨 미사용)
    clim_train = build_month_hour_climatology(labels, before=valid_start)
    out = shrink_toward_climatology(pred_wide, clim_train, shrink=shrink)
    return out, shrink
