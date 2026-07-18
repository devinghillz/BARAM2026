from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP_IDS = [1, 2, 3]
METRIC_REQUIRED_COLUMNS = {"forecast_kst_dtm", "group_id", "y_true", "y_pred"}
CAPACITY_REQUIRED_COLUMNS = {"group_id", "그룹설비용량(MW)"}


@dataclass(frozen=True)
class GroupMetric:
    group_id: int
    capacity: float
    label_count: int
    eligible_count: int
    nmae: float
    one_minus_nmae: float
    ficr: float
    score: float
    rate_4_count: int
    rate_3_count: int
    rate_0_count: int


def extract_capacity_by_group(turbine_metadata: pd.DataFrame) -> dict[int, float]:
    missing = CAPACITY_REQUIRED_COLUMNS - set(turbine_metadata.columns)
    if missing:
        raise ValueError(f"설비용량 컬럼이 없습니다: {sorted(missing)}")

    capacity_table = (
        turbine_metadata[["group_id", "그룹설비용량(MW)"]]
        .drop_duplicates()
        .sort_values("group_id")
        .reset_index(drop=True)
    )
    if capacity_table["group_id"].duplicated().any():
        raise ValueError("한 그룹에 여러 그룹설비용량이 있습니다.")

    capacity_table["group_id"] = pd.to_numeric(capacity_table["group_id"], errors="raise").astype(int)
    if set(capacity_table["group_id"]) != set(GROUP_IDS):
        raise ValueError("group_id가 1, 2, 3이 아닙니다.")

    capacity_table["capacity"] = pd.to_numeric(capacity_table["그룹설비용량(MW)"], errors="raise").mul(1000.0)
    return dict(zip(capacity_table["group_id"], capacity_table["capacity"]))


def evaluate_baram_score(
    prediction_frame: pd.DataFrame,
    capacity_by_group: dict[int, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    missing = METRIC_REQUIRED_COLUMNS - set(prediction_frame.columns)
    if missing:
        raise ValueError(f"평가 필수 컬럼이 없습니다: {sorted(missing)}")

    frame = prediction_frame.copy()
    frame["group_id"] = pd.to_numeric(frame["group_id"], errors="raise").astype(int)
    if frame.duplicated(["forecast_kst_dtm", "group_id"]).any():
        raise ValueError("시각·그룹 예측이 중복됩니다.")
    if frame["group_id"].isna().any():
        raise ValueError("group_id에 결측이 있습니다.")
    if frame["y_pred"].isna().any():
        raise ValueError("예측값에 결측이 있습니다.")
    if not np.isfinite(frame["y_pred"].to_numpy(dtype="float64")).all():
        raise ValueError("예측값에 inf 또는 -inf가 있습니다.")

    group_results = []
    for group_id in GROUP_IDS:
        if group_id not in capacity_by_group:
            raise ValueError(f"group {group_id} 설비용량이 없습니다.")

        capacity = float(capacity_by_group[group_id])
        group_frame = frame.loc[frame["group_id"].eq(group_id)].copy()
        group_frame = group_frame.loc[group_frame["y_true"].notna()].copy()
        if group_frame.empty:
            raise ValueError(f"group {group_id}: 사용 가능한 Label이 없습니다.")

        evaluated = group_frame.loc[group_frame["y_true"].ge(0.10 * capacity)].copy()
        if evaluated.empty:
            raise ValueError(f"group {group_id}: 10% 이상 평가행이 없습니다.")

        normalized_error = evaluated["y_pred"].sub(evaluated["y_true"]).abs().div(capacity)
        settlement_rate = np.select(
            [normalized_error.le(0.06), normalized_error.le(0.08)],
            [4.0, 3.0],
            default=0.0,
        )
        actual = evaluated["y_true"].to_numpy(dtype="float64")
        maximum_settlement = float(np.sum(actual * 4.0))
        if maximum_settlement <= 0:
            raise ValueError(f"group {group_id}: 최대 정산금이 0 이하입니다.")

        nmae = float(normalized_error.mean())
        ficr = float(np.sum(actual * settlement_rate) / maximum_settlement)
        group_results.append(
            GroupMetric(
                group_id=group_id,
                capacity=capacity,
                label_count=int(len(group_frame)),
                eligible_count=int(len(evaluated)),
                nmae=nmae,
                one_minus_nmae=float(1.0 - nmae),
                ficr=ficr,
                score=float(0.5 * (1.0 - nmae) + 0.5 * ficr),
                rate_4_count=int(np.sum(settlement_rate == 4.0)),
                rate_3_count=int(np.sum(settlement_rate == 3.0)),
                rate_0_count=int(np.sum(settlement_rate == 0.0)),
            )
        )

    group_metric_frame = pd.DataFrame([result.__dict__ for result in group_results])
    mean_nmae = float(group_metric_frame["nmae"].mean())
    mean_ficr = float(group_metric_frame["ficr"].mean())
    one_minus_nmae = float(1.0 - mean_nmae)
    return (
        {
            "score": float(0.5 * one_minus_nmae + 0.5 * mean_ficr),
            "nmae": mean_nmae,
            "one_minus_nmae": one_minus_nmae,
            "ficr": mean_ficr,
        },
        group_metric_frame,
    )


def audit_prediction_coverage(prediction_frame: pd.DataFrame, expected_forecast_times: pd.Series) -> None:
    actual_group_ids = set(prediction_frame["group_id"].dropna().astype(int).unique())
    if actual_group_ids != set(GROUP_IDS):
        raise ValueError("예측 group_id가 1, 2, 3과 다릅니다.")

    expected = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(expected_forecast_times).unique(), GROUP_IDS],
        names=["forecast_kst_dtm", "group_id"],
    )
    actual = pd.MultiIndex.from_frame(prediction_frame[["forecast_kst_dtm", "group_id"]])
    if actual.has_duplicates:
        raise ValueError("시각·그룹 예측이 중복됩니다.")
    missing = expected.difference(actual)
    unexpected = actual.difference(expected)
    if len(missing) > 0:
        raise ValueError(f"예측 누락 key: {len(missing):,}개")
    if len(unexpected) > 0:
        raise ValueError(f"예상 밖 key: {len(unexpected):,}개")


def evaluate_concatenated_oof(
    fold_prediction_frames: list[pd.DataFrame],
    expected_forecast_times: pd.Series,
    capacity_by_group: dict[int, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    oof = pd.concat(fold_prediction_frames, axis=0, ignore_index=True)
    audit_prediction_coverage(oof, expected_forecast_times)
    return evaluate_baram_score(oof, capacity_by_group)
