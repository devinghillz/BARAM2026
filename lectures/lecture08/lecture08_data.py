from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_KEYS = ["forecast_kst_dtm", "data_available_kst_dtm"]
GROUP_KEYS = [*FORECAST_KEYS, "group_id"]
GROUP_IDS = [1, 2, 3]
TARGET_COLUMNS = {1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"}
FEATURE_SPEC = "lecture08_baseline_v1"
EXPECTED_FEATURE_COUNT = 130


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_lecture06_package(project_root: Path) -> None:
    root = project_root / "lectures" / "lecture06" / "lecture06_feature_candidates"
    required = [
        root / "metadata" / "block_manifest.json",
        root / "reports" / "audit_summary.json",
        root / "shared" / "time_features_train.csv",
        root / "shared" / "time_features_test.csv",
        root / "shared" / "calendar_features_train.csv",
        root / "shared" / "calendar_features_test.csv",
        root / "group" / "center_nearest_train.csv",
        root / "group" / "center_nearest_test.csv",
    ]
    if all(path.exists() for path in required):
        return
    subprocess.run([sys.executable, str(project_root / "lectures" / "lecture06" / "build_lecture06_feature_candidates.py"), "--project-root", str(project_root)], check=True)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"제6강 산출물이 없습니다: {missing}")


def ensure_lecture07_package(project_root: Path) -> None:
    root = project_root / "lectures" / "lecture07" / "lecture07_validation"
    protocol_path = root / "metadata" / "validation_protocol.json"
    required = [protocol_path, root / "folds" / "primary_fold_assignments.csv", root / "folds" / "year_block_assignments.csv"]
    if all(path.exists() for path in required):
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol["protocol_version"] != "lecture07_v2":
            raise ValueError("Validation protocol이 lecture07_v2가 아닙니다.")
        return
    subprocess.run([sys.executable, str(project_root / "lectures" / "lecture07" / "build_lecture07_validation.py"), "--project-root", str(project_root)], check=True)


def feature_paths(project_root: Path) -> dict[str, dict[str, Path]]:
    root = project_root / "lectures" / "lecture06" / "lecture06_feature_candidates"
    return {
        "train": {
            "time": root / "shared" / "time_features_train.csv",
            "calendar": root / "shared" / "calendar_features_train.csv",
            "center_nearest": root / "group" / "center_nearest_train.csv",
        },
        "test": {
            "time": root / "shared" / "time_features_test.csv",
            "calendar": root / "shared" / "calendar_features_test.csv",
            "center_nearest": root / "group" / "center_nearest_test.csv",
        },
    }


def read_feature_csv(path: Path, key_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    for column in FORECAST_KEYS:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    if frame[key_columns].isna().any().any():
        raise ValueError(f"{path.name}: key 결측")
    if frame.duplicated(key_columns).any():
        raise ValueError(f"{path.name}: key 중복")
    return frame


def merge_shared_blocks(time_frame: pd.DataFrame, calendar_frame: pd.DataFrame) -> pd.DataFrame:
    result = time_frame.copy()
    overlap = (set(result.columns) & set(calendar_frame.columns)) - set(FORECAST_KEYS)
    if overlap:
        raise ValueError(f"calendar: 중복 feature {sorted(overlap)[:10]}")
    before_rows = len(result)
    result = result.merge(calendar_frame, on=FORECAST_KEYS, how="left", validate="one_to_one")
    if len(result) != before_rows:
        raise ValueError("calendar: merge 후 행 수 변경")
    if result.shape[1] != 16:
        raise ValueError(f"Shared block 열 수가 다릅니다: {result.shape[1]} != 16")
    return result


def build_group_feature_table(shared_frame: pd.DataFrame, center_nearest_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    before_rows = len(center_nearest_frame)
    result = center_nearest_frame.merge(shared_frame, on=FORECAST_KEYS, how="left", validate="many_to_one").sort_values(GROUP_KEYS).reset_index(drop=True)
    if len(result) != before_rows:
        raise ValueError("Group feature merge 후 행 수가 바뀌었습니다.")
    if set(result["group_id"].unique()) != set(GROUP_IDS):
        raise ValueError("group_id가 1, 2, 3이 아닙니다.")
    feature_columns = [column for column in result.columns if column not in GROUP_KEYS]
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Baseline feature 수가 다릅니다: {len(feature_columns)} != {EXPECTED_FEATURE_COUNT}")
    return result, feature_columns


def build_feature_tables(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    paths = feature_paths(project_root)
    tables = {}
    for split in ["train", "test"]:
        shared = merge_shared_blocks(
            read_feature_csv(paths[split]["time"], FORECAST_KEYS),
            read_feature_csv(paths[split]["calendar"], FORECAST_KEYS),
        )
        tables[split], feature_columns = build_group_feature_table(shared, read_feature_csv(paths[split]["center_nearest"], GROUP_KEYS))
    audit_train_test_schema(tables["train"], tables["test"], feature_columns)
    return tables["train"], tables["test"], feature_columns


def attach_labels(train_feature_frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_frame = labels.copy()
    label_frame["forecast_kst_dtm"] = pd.to_datetime(label_frame["kst_dtm"], errors="raise")
    long_labels = label_frame.melt(
        id_vars=["forecast_kst_dtm"],
        value_vars=list(TARGET_COLUMNS.values()),
        var_name="target_column",
        value_name="y_true",
    )
    long_labels["group_id"] = long_labels["target_column"].map({v: k for k, v in TARGET_COLUMNS.items()}).astype(int)
    result = train_feature_frame.merge(long_labels[["forecast_kst_dtm", "group_id", "y_true"]], on=["forecast_kst_dtm", "group_id"], how="left", validate="one_to_one")
    expected = {1: 26200, 2: 26201, 3: 17538}
    actual = result.groupby("group_id")["y_true"].count().to_dict()
    if actual != expected:
        raise ValueError(f"그룹별 Label 수가 다릅니다: {actual}")
    return result


def audit_train_test_schema(train_frame: pd.DataFrame, test_frame: pd.DataFrame, feature_columns: list[str]) -> None:
    train_features = [column for column in train_frame.columns if column not in GROUP_KEYS]
    test_features = [column for column in test_frame.columns if column not in GROUP_KEYS]
    if train_features != test_features or train_features != feature_columns:
        raise ValueError("Train/Test feature 열 또는 순서가 다릅니다.")
    non_numeric = [column for column in feature_columns if not (pd.api.types.is_numeric_dtype(train_frame[column]) and pd.api.types.is_numeric_dtype(test_frame[column]))]
    if non_numeric:
        raise ValueError(f"비수치 feature가 있습니다: {non_numeric[:10]}")
    all_missing_train = [column for column in feature_columns if train_frame[column].isna().all()]
    all_missing_test = [column for column in feature_columns if test_frame[column].isna().all()]
    if all_missing_train or all_missing_test:
        raise ValueError(f"전체 NaN feature가 있습니다: train={all_missing_train[:10]}, test={all_missing_test[:10]}")


def audit_forecast_availability(forecast_index: pd.DataFrame, expected_rows: int) -> dict[str, object]:
    frame = forecast_index.copy()
    for column in FORECAST_KEYS:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    if len(frame) != expected_rows:
        raise ValueError(f"예상 forecast 행 수와 다릅니다: {len(frame):,} != {expected_rows:,}")
    if frame[FORECAST_KEYS].isna().any().any() or frame.duplicated(FORECAST_KEYS).any():
        raise ValueError("시간키 결측 또는 중복이 있습니다.")
    lead_hour = frame["forecast_kst_dtm"].sub(frame["data_available_kst_dtm"]).dt.total_seconds().div(3600.0)
    if not lead_hour.between(12, 35, inclusive="both").all() or set(lead_hour.astype(int).unique()) != set(range(12, 36)):
        raise ValueError("lead_hour 12~35 조건을 만족하지 않습니다.")
    rows_per_issue = frame.groupby("data_available_kst_dtm").size()
    issue_per_target = frame.groupby("forecast_kst_dtm")["data_available_kst_dtm"].nunique()
    if not rows_per_issue.eq(24).all() or not issue_per_target.eq(1).all():
        raise ValueError("예보 발행 묶음 구조가 예상과 다릅니다.")
    if not frame["data_available_kst_dtm"].lt(frame["forecast_kst_dtm"]).all():
        raise ValueError("예측 대상 이후 이용 가능한 예보가 있습니다.")
    return {
        "forecast_rows": int(len(frame)),
        "forecast_times": int(frame["forecast_kst_dtm"].nunique()),
        "issue_batches": int(frame["data_available_kst_dtm"].nunique()),
        "issue_batch_size": 24,
        "lead_hour_min": int(lead_hour.min()),
        "lead_hour_max": int(lead_hour.max()),
    }


def load_primary_assignments(project_root: Path) -> pd.DataFrame:
    path = project_root / "lectures" / "lecture07" / "lecture07_validation" / "folds" / "primary_fold_assignments.csv"
    assignment = pd.read_csv(path, encoding="utf-8-sig")
    for column in FORECAST_KEYS:
        assignment[column] = pd.to_datetime(assignment[column], errors="raise")
    missing = {"fold_name", "role", *FORECAST_KEYS} - set(assignment.columns)
    if missing:
        raise ValueError(f"Fold assignment 컬럼 누락: {sorted(missing)}")
    return assignment


def load_year_block_assignments(project_root: Path) -> pd.DataFrame:
    path = project_root / "lectures" / "lecture07" / "lecture07_validation" / "folds" / "year_block_assignments.csv"
    assignment = pd.read_csv(path, encoding="utf-8-sig")
    for column in FORECAST_KEYS:
        assignment[column] = pd.to_datetime(assignment[column], errors="raise")
    return assignment


def expected_oof_times(assignment: pd.DataFrame) -> pd.Series:
    return assignment.loc[assignment["role"].eq("validation"), "forecast_kst_dtm"].drop_duplicates().sort_values().reset_index(drop=True)


def missing_report(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "missing_cells": int(frame[feature_columns].isna().sum().sum()),
        "rows_with_missing": int(frame[feature_columns].isna().any(axis=1).sum()),
    }


def feature_spec() -> dict[str, object]:
    return {
        "feature_spec": FEATURE_SPEC,
        "shared_blocks": ["time_features", "calendar_features"],
        "group_blocks": ["center_nearest"],
        "feature_count": EXPECTED_FEATURE_COUNT,
        "scada_used": False,
        "label_feature_used": False,
        "fit_required_before_model": False,
    }


def experiment_config() -> dict[str, object]:
    return {
        "experiment_id": FEATURE_SPEC,
        "validation_protocol": "lecture07_v2",
        "models": ["lightgbm", "xgboost", "catboost"],
        "model_structure": "separate_by_group",
        "seed": 42,
        "feature_spec": FEATURE_SPEC,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "missing_policy": "native_missing",
        "scada_used": False,
        "prediction_clipping": False,
        "primary_metric": "concatenated_oof_official_score",
        "final_iteration_policy": "median_of_four_fold_best_iterations",
        "test_disagreement_used_for_training": False,
        "submission_count": 3,
    }


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for module_name, key in [("sklearn", "scikit_learn"), ("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("catboost", "catboost"), ("joblib", "joblib")]:
        try:
            module = __import__(module_name)
            versions[key] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[key] = "not_installed"
    return versions


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
