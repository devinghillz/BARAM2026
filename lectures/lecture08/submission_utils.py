from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lecture08_data import GROUP_IDS, GROUP_KEYS


SUBMISSION_COLUMNS = ["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]


def prediction_long_to_wide(prediction_long: pd.DataFrame) -> pd.DataFrame:
    wide = prediction_long.pivot(index="forecast_kst_dtm", columns="group_id", values="y_pred").rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"}).reset_index()
    expected_columns = ["forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]
    if list(wide.columns) != expected_columns or len(wide) != 8760:
        raise ValueError("Wide prediction 열 또는 행 수가 다릅니다.")
    return wide


def build_submission(sample_submission_path: Path, prediction_long: pd.DataFrame) -> pd.DataFrame:
    sample = pd.read_csv(sample_submission_path, encoding="utf-8-sig")
    original_columns = list(sample.columns)
    original_forecast_id = sample["forecast_id"].copy()
    original_forecast_time = sample["forecast_kst_dtm"].copy()
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"], errors="raise")
    submission = sample[["forecast_id", "forecast_kst_dtm"]].merge(prediction_long_to_wide(prediction_long), on="forecast_kst_dtm", how="left", validate="one_to_one", sort=False)
    if list(submission.columns) != SUBMISSION_COLUMNS or original_columns != SUBMISSION_COLUMNS:
        raise ValueError("제출 열 또는 순서가 다릅니다.")
    if not submission["forecast_id"].reset_index(drop=True).equals(original_forecast_id.reset_index(drop=True)):
        raise ValueError("forecast_id가 변경됐습니다.")
    time_string = submission["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if not time_string.equals(original_forecast_time.astype(str).reset_index(drop=True)):
        raise ValueError("forecast_kst_dtm 또는 행 순서가 변경됐습니다.")
    submission["forecast_kst_dtm"] = time_string
    return submission


def audit_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame, capacity_by_group: dict[int, float]) -> dict[str, object]:
    if list(submission.columns) != list(sample_submission.columns) or len(submission) != 8760:
        raise ValueError("제출 열 또는 행 수가 다릅니다.")
    if not submission["forecast_id"].equals(sample_submission["forecast_id"]):
        raise ValueError("forecast_id가 다릅니다.")
    if not submission["forecast_kst_dtm"].astype(str).equals(sample_submission["forecast_kst_dtm"].astype(str)):
        raise ValueError("forecast_kst_dtm 또는 순서가 다릅니다.")
    if submission["forecast_id"].duplicated().any():
        raise ValueError("forecast_id가 중복됩니다.")
    numeric = submission[SUBMISSION_COLUMNS[2:]].apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("제출 예측값에 NaN 또는 inf가 있습니다.")
    report = {"row_count": int(len(submission)), "columns": list(submission.columns), "forecast_id_unchanged": True, "forecast_time_unchanged": True, "nan_count": 0, "inf_count": 0, "prediction_clipping": False, "by_group": {}}
    for group_id in GROUP_IDS:
        values = numeric[f"kpx_group_{group_id}"]
        report["by_group"][str(group_id)] = {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "below_zero_count": int(values.lt(0).sum()), "above_capacity_count": int(values.gt(capacity_by_group[group_id]).sum())}
    return report


def write_submission(submission: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.6f")
