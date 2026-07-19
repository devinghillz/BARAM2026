from __future__ import annotations

import numpy as np
import pandas as pd

from lecture08_data import GROUP_KEYS
from lecture08_models import MODEL_NAMES


def _merge_predictions(prediction_by_model: dict[str, pd.DataFrame], key_columns: list[str]) -> pd.DataFrame:
    result = None
    for model_name in MODEL_NAMES:
        part = prediction_by_model[model_name][[*key_columns, "y_pred"]].rename(columns={"y_pred": f"pred_{model_name}"})
        result = part if result is None else result.merge(part, on=key_columns, how="inner", validate="one_to_one")
    return result


def add_disagreement_columns(result: pd.DataFrame, capacity_by_group: dict[int, float]) -> pd.DataFrame:
    prediction_columns = [f"pred_{model_name}" for model_name in MODEL_NAMES]
    matrix = result[prediction_columns].to_numpy(dtype=np.float64)
    result["prediction_mean"] = matrix.mean(axis=1)
    result["prediction_std"] = matrix.std(axis=1, ddof=0)
    result["prediction_range"] = matrix.max(axis=1) - matrix.min(axis=1)
    result["abs_lgbm_xgb"] = result["pred_lightgbm"].sub(result["pred_xgboost"]).abs()
    result["abs_lgbm_cat"] = result["pred_lightgbm"].sub(result["pred_catboost"]).abs()
    result["abs_xgb_cat"] = result["pred_xgboost"].sub(result["pred_catboost"]).abs()
    capacity = result["group_id"].map(capacity_by_group).astype(float)
    result["normalized_prediction_std"] = result["prediction_std"].div(capacity)
    result["normalized_prediction_range"] = result["prediction_range"].div(capacity)
    return result


def build_oof_disagreement(oof_by_model: dict[str, pd.DataFrame], capacity_by_group: dict[int, float]) -> pd.DataFrame:
    result = add_disagreement_columns(_merge_predictions(oof_by_model, [*GROUP_KEYS, "fold_name", "y_true"]), capacity_by_group)
    capacity = result["group_id"].map(capacity_by_group).astype(float)
    for model_name in MODEL_NAMES:
        result[f"normalized_error_{model_name}"] = result[f"pred_{model_name}"].sub(result["y_true"]).abs().div(capacity)
    result["normalized_error_mean_prediction"] = result["prediction_mean"].sub(result["y_true"]).abs().div(capacity)
    return result


def build_test_disagreement(prediction_by_model: dict[str, pd.DataFrame], capacity_by_group: dict[int, float]) -> pd.DataFrame:
    return add_disagreement_columns(_merge_predictions(prediction_by_model, GROUP_KEYS), capacity_by_group)


def summarize_test_disagreement_by_issue(test_disagreement: pd.DataFrame) -> pd.DataFrame:
    return test_disagreement.groupby("data_available_kst_dtm", as_index=False).agg(
        mean_normalized_std=("normalized_prediction_std", "mean"),
        p90_normalized_std=("normalized_prediction_std", lambda values: values.quantile(0.90)),
        max_normalized_std=("normalized_prediction_std", "max"),
        mean_normalized_range=("normalized_prediction_range", "mean"),
        p90_normalized_range=("normalized_prediction_range", lambda values: values.quantile(0.90)),
    )


def disagreement_summary(oof_disagreement: pd.DataFrame | None, test_disagreement: pd.DataFrame | None) -> dict[str, object]:
    summary = {}
    if oof_disagreement is not None:
        summary["oof_rows"] = int(len(oof_disagreement))
        summary["oof_corr_std_error_mean_prediction"] = float(oof_disagreement["normalized_prediction_std"].corr(oof_disagreement["normalized_error_mean_prediction"]))
    if test_disagreement is not None:
        summary["test_rows"] = int(len(test_disagreement))
        summary["test_mean_normalized_std"] = float(test_disagreement["normalized_prediction_std"].mean())
        summary["test_p90_normalized_std"] = float(test_disagreement["normalized_prediction_std"].quantile(0.90))
    return summary
