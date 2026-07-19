from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from lecture08_data import FORECAST_KEYS, GROUP_IDS, GROUP_KEYS


RANDOM_SEED = 42
MAX_BOOSTING_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 200
MODEL_NAMES = ["lightgbm", "xgboost", "catboost"]


def require_model_dependencies(model_names: list[str] = MODEL_NAMES) -> None:
    module_by_model = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}
    missing = []
    for model_name in model_names:
        try:
            __import__(module_by_model[model_name])
        except Exception:
            missing.append(module_by_model[model_name])
    try:
        __import__("joblib")
    except Exception:
        missing.append("joblib")
    if missing:
        raise ModuleNotFoundError(f"제8강 모델 실행 의존성이 없습니다: {sorted(set(missing))}")


def create_lightgbm(n_estimators: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=1.00,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def create_xgboost(n_estimators: int, use_early_stopping: bool):
    import xgboost as xgb

    params = {
        "objective": "reg:squarederror",
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5.0,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
        "tree_method": "hist",
        "eval_metric": "mae",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    if use_early_stopping:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return xgb.XGBRegressor(**params)


def create_catboost(iterations: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        random_seed=RANDOM_SEED,
        allow_writing_files=False,
        verbose=False,
    )


def fit_model(model_name: str, n_iterations: int, x_train, y_train, x_valid=None, y_valid=None):
    if model_name == "lightgbm":
        import lightgbm as lgb

        model = create_lightgbm(n_iterations)
        if x_valid is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric="mae", callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)])
        return model, int(getattr(model, "best_iteration_", n_iterations) or n_iterations)
    if model_name == "xgboost":
        model = create_xgboost(n_iterations, use_early_stopping=x_valid is not None)
        fit_kwargs = {"verbose": False} if x_valid is not None else {}
        if x_valid is not None:
            fit_kwargs["eval_set"] = [(x_valid, y_valid)]
        model.fit(x_train, y_train, **fit_kwargs)
        return model, int(getattr(model, "best_iteration", n_iterations - 1) + 1)
    if model_name == "catboost":
        model = create_catboost(n_iterations)
        if x_valid is None:
            model.fit(x_train, y_train, verbose=False)
        else:
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True, early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        return model, int((model.get_best_iteration() if x_valid is not None else n_iterations - 1) + 1)
    raise ValueError(f"알 수 없는 모델: {model_name}")


def to_float32_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return frame[feature_columns].to_numpy(dtype=np.float32, copy=True)


def audit_model_fold_time(train_part: pd.DataFrame, validation_part: pd.DataFrame) -> None:
    if train_part.empty or validation_part.empty:
        raise ValueError("Fold Train 또는 Validation이 비어 있습니다.")
    min_validation_available = validation_part["data_available_kst_dtm"].min()
    if not train_part["data_available_kst_dtm"].max() < min_validation_available:
        raise ValueError("Validation 발행 이후 예보가 Train에 포함됐습니다.")
    if not train_part["forecast_kst_dtm"].max() < min_validation_available:
        raise ValueError("Validation 시작 이후 target Label이 Train에 포함됐습니다.")
    if set(train_part["data_available_kst_dtm"]) & set(validation_part["data_available_kst_dtm"]):
        raise ValueError("같은 예보 발행 묶음이 Train/Validation에 나뉘었습니다.")


def audit_fold_features(train_part: pd.DataFrame, feature_columns: list[str], fold_name: str, group_id: int) -> None:
    all_missing = [column for column in feature_columns if train_part[column].isna().all()]
    if all_missing:
        raise ValueError(f"{fold_name}, group {group_id}: Fold Train 전체가 NaN인 feature가 있습니다: {all_missing[:10]}")


def make_fold_parts(group_frame: pd.DataFrame, assignment: pd.DataFrame, fold_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_assignment = assignment.loc[assignment["fold_name"].eq(fold_name)].copy()
    selected = {}
    for role in ["train", "validation"]:
        times = fold_assignment.loc[fold_assignment["role"].eq(role), FORECAST_KEYS].drop_duplicates()
        selected[role] = group_frame.merge(times.assign(_selected=True), on=FORECAST_KEYS, how="inner", validate="many_to_one").drop(columns="_selected")
    audit_model_fold_time(selected["train"], selected["validation"])
    return selected["train"].loc[selected["train"]["y_true"].notna()].copy(), selected["validation"]


def run_primary_cv(model_name: str, train_frame: pd.DataFrame, assignment: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows, training_rows = [], []
    for fold_name in assignment["fold_name"].drop_duplicates().sort_values():
        for group_id in GROUP_IDS:
            group_frame = train_frame.loc[train_frame["group_id"].eq(group_id)].copy()
            train_part, validation_part = make_fold_parts(group_frame, assignment, fold_name)
            audit_fold_features(train_part, feature_columns, fold_name, group_id)
            x_train, y_train = to_float32_matrix(train_part, feature_columns), train_part["y_true"].to_numpy(dtype=np.float32)
            x_valid = to_float32_matrix(validation_part, feature_columns)
            valid_label_mask = validation_part["y_true"].notna()
            if not valid_label_mask.any():
                raise ValueError(f"{fold_name}, group {group_id}: Validation Label이 없습니다.")
            start = perf_counter()
            model, best_iteration = fit_model(model_name, MAX_BOOSTING_ROUNDS, x_train, y_train, x_valid[valid_label_mask.to_numpy()], validation_part.loc[valid_label_mask, "y_true"].to_numpy(dtype=np.float32))
            prediction = model.predict(x_valid)
            if not np.isfinite(prediction).all():
                raise ValueError("Validation 예측에 NaN 또는 inf가 있습니다.")
            part = validation_part[[*GROUP_KEYS, "y_true"]].copy()
            part["fold_name"], part["model_name"], part["y_pred"] = fold_name, model_name, prediction.astype(np.float64)
            prediction_rows.append(part)
            training_rows.append({"model_name": model_name, "fold_name": fold_name, "group_id": group_id, "train_label_count": int(len(train_part)), "validation_row_count": int(len(validation_part)), "validation_label_count": int(valid_label_mask.sum()), "best_iteration": best_iteration, "fit_seconds": float(perf_counter() - start)})
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(training_rows)


def choose_final_iterations(training_log: pd.DataFrame) -> pd.DataFrame:
    result = training_log.groupby(["model_name", "group_id"], as_index=False)["best_iteration"].median()
    result["final_iteration"] = result["best_iteration"].round().clip(lower=1).astype(int)
    return result.drop(columns="best_iteration")


def fit_fixed_iteration_model(model_name: str, n_iterations: int, x_train: np.ndarray, y_train: np.ndarray):
    return fit_model(model_name, n_iterations, x_train, y_train)[0]


def train_final_models(model_name: str, train_frame: pd.DataFrame, feature_columns: list[str], final_iterations: pd.DataFrame) -> dict[int, object]:
    models = {}
    for group_id in GROUP_IDS:
        group_train = train_frame.loc[train_frame["group_id"].eq(group_id) & train_frame["y_true"].notna()].copy()
        n_iterations = int(final_iterations.loc[final_iterations["model_name"].eq(model_name) & final_iterations["group_id"].eq(group_id), "final_iteration"].iloc[0])
        models[group_id] = fit_fixed_iteration_model(model_name, n_iterations, to_float32_matrix(group_train, feature_columns), group_train["y_true"].to_numpy(dtype=np.float32))
    return models


def predict_test_long(model_name: str, models: dict[int, object], test_frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for group_id in GROUP_IDS:
        group_test = test_frame.loc[test_frame["group_id"].eq(group_id)].sort_values("forecast_kst_dtm").copy()
        if len(group_test) != 8760:
            raise ValueError(f"group {group_id}: Test 행 수가 8,760이 아닙니다.")
        prediction = models[group_id].predict(to_float32_matrix(group_test, feature_columns))
        if not np.isfinite(prediction).all():
            raise ValueError("Test 예측에 NaN 또는 inf가 있습니다.")
        output = group_test[GROUP_KEYS].copy()
        output["model_name"], output["y_pred"] = model_name, prediction.astype(np.float64)
        rows.append(output)
    result = pd.concat(rows, ignore_index=True)
    if len(result) != 26280:
        raise ValueError("Test long prediction 행 수가 26,280이 아닙니다.")
    return result


def save_final_models(models: dict[int, object], model_name: str, model_dir) -> None:
    import joblib

    output_dir = model_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for group_id, model in models.items():
        joblib.dump(model, output_dir / f"group_{group_id}.joblib")


def load_final_models(model_name: str, model_dir) -> dict[int, object]:
    import joblib

    return {group_id: joblib.load(model_dir / model_name / f"group_{group_id}.joblib") for group_id in GROUP_IDS}
