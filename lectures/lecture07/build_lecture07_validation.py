from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baram_metric import GROUP_IDS, evaluate_baram_score, extract_capacity_by_group


TIME_KEYS = ["forecast_kst_dtm", "data_available_kst_dtm"]
PROTOCOL_VERSION = "lecture07_v1"
EXPECTED_CAPACITY = {1: 21600.0, 2: 21600.0, 3: 21000.0}


def load_lecture05_builder(project_root: Path):
    path = project_root / "lectures" / "lecture05" / "build_lecture05_master_data.py"
    spec = importlib.util.spec_from_file_location("lecture05_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_turbine_metadata(turbine_metadata: pd.DataFrame) -> pd.DataFrame:
    result = turbine_metadata.rename(columns={"KPX그룹": "group_id"}).copy()
    result["group_id"] = pd.to_numeric(result["group_id"], errors="raise").astype(int)
    return result


def build_fold_frame(forecast_index: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    frame = forecast_index.copy().sort_values(TIME_KEYS).reset_index(drop=True)
    if frame[TIME_KEYS].isna().any().any():
        raise ValueError("시간키에 결측이 있습니다.")
    if frame.duplicated(TIME_KEYS).any():
        raise ValueError("시간키 중복이 있습니다.")
    frame["row_id"] = np.arange(len(frame), dtype="int64")
    label_frame = labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}).copy()
    label_frame["forecast_kst_dtm"] = pd.to_datetime(label_frame["forecast_kst_dtm"])
    return frame.merge(label_frame, on="forecast_kst_dtm", how="left", validate="one_to_one")


def build_issue_quarter_folds(frame: pd.DataFrame, validation_year: int = 2024):
    periods = pd.period_range(start=f"{validation_year}Q1", end=f"{validation_year}Q4", freq="Q")
    issue_quarter = frame["data_available_kst_dtm"].dt.to_period("Q")
    folds, rows = {}, []
    for fold_number, period in enumerate(periods, start=1):
        validation_mask = issue_quarter.eq(period)
        if not validation_mask.any():
            raise ValueError(f"{period}: Validation 행이 없습니다.")
        validation_issue_start = frame.loc[validation_mask, "data_available_kst_dtm"].min()
        validation_issue_end = frame.loc[validation_mask, "data_available_kst_dtm"].max()
        train_mask = frame["data_available_kst_dtm"].lt(validation_issue_start) & frame["forecast_kst_dtm"].lt(validation_issue_start)
        fold_name = f"fold_{fold_number}_{period}"
        folds[fold_name] = {"train_mask": train_mask, "validation_mask": validation_mask}
        rows.append(
            {
                "fold_name": fold_name,
                "validation_period": str(period),
                "validation_issue_start": validation_issue_start,
                "validation_issue_end": validation_issue_end,
                "train_row_count": int(train_mask.sum()),
                "validation_row_count": int(validation_mask.sum()),
            }
        )
    manifest = pd.DataFrame(rows)
    if not manifest["train_row_count"].is_monotonic_increasing:
        raise ValueError("Expanding Train 크기가 증가하지 않습니다.")
    return folds, manifest


def build_year_block_fold(frame: pd.DataFrame, validation_year: int = 2024) -> dict[str, object]:
    validation_mask = frame["data_available_kst_dtm"].dt.year.eq(validation_year)
    validation_issue_start = frame.loc[validation_mask, "data_available_kst_dtm"].min()
    train_mask = frame["data_available_kst_dtm"].lt(validation_issue_start) & frame["forecast_kst_dtm"].lt(validation_issue_start)
    return {
        "fold_name": f"stress_{validation_year}_year_block",
        "validation_period": str(validation_year),
        "validation_issue_start": validation_issue_start,
        "validation_issue_end": frame.loc[validation_mask, "data_available_kst_dtm"].max(),
        "train_row_count": int(train_mask.sum()),
        "validation_row_count": int(validation_mask.sum()),
        "primary_cv_included": False,
    }


def audit_issue_batch_size(frame: pd.DataFrame, expected_size: int = 24) -> None:
    rows_per_issue = frame.groupby("data_available_kst_dtm").size()
    if not rows_per_issue.eq(expected_size).all():
        raise ValueError(f"발행시각당 행 수가 {expected_size}가 아닙니다.")


def audit_fold(frame: pd.DataFrame, train_mask: pd.Series, validation_mask: pd.Series) -> None:
    if (train_mask & validation_mask).any():
        raise ValueError("Train/Validation 행이 겹칩니다.")
    if set(frame.loc[train_mask, "data_available_kst_dtm"]) & set(frame.loc[validation_mask, "data_available_kst_dtm"]):
        raise ValueError("Train/Validation에 같은 예보 발행시각이 있습니다.")
    if not frame.loc[train_mask, "forecast_kst_dtm"].max() < frame.loc[validation_mask, "data_available_kst_dtm"].min():
        raise ValueError("Validation 시작 이후의 target Label이 Train에 포함됐습니다.")


def label_coverage(frame: pd.DataFrame, folds: dict[str, dict[str, pd.Series]], capacity_by_group: dict[int, float]) -> pd.DataFrame:
    rows = []
    for fold_name, masks in folds.items():
        for role, mask_name in [("train", "train_mask"), ("validation", "validation_mask")]:
            for group_id in GROUP_IDS:
                y_true = frame.loc[masks[mask_name], f"kpx_group_{group_id}"]
                rows.append(
                    {
                        "fold_name": fold_name,
                        "role": role,
                        "group_id": group_id,
                        "row_count": int(len(y_true)),
                        "label_count": int(y_true.notna().sum()),
                        "eligible_count": int(y_true.ge(0.10 * capacity_by_group[group_id]).sum()),
                    }
                )
    return pd.DataFrame(rows)


def fold_assignments(frame: pd.DataFrame, folds: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    rows = []
    for fold_name, masks in folds.items():
        for role, mask_name in [("train", "train_mask"), ("validation", "validation_mask")]:
            part = frame.loc[masks[mask_name], ["row_id", *TIME_KEYS]].copy()
            part.insert(0, "fold_name", fold_name)
            part.insert(1, "role", role)
            rows.append(part)
    return pd.concat(rows, ignore_index=True).sort_values(["fold_name", "role", "row_id"]).reset_index(drop=True)


def protocol() -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "split_axis": "data_available_kst_dtm",
        "validation_year": 2024,
        "primary_folds": ["2024Q1", "2024Q2", "2024Q3", "2024Q4"],
        "train_policy": "expanding",
        "label_purge": True,
        "metric": "official_score",
        "group_weighting": "equal",
        "eligibility_threshold": 0.10,
        "settlement_thresholds": [0.06, 0.08],
        "settlement_rates": [4, 3, 0],
        "prediction_clipping": False,
    }


def unit_test_frame(error_ratio: float, capacity_by_group: dict[int, float]) -> pd.DataFrame:
    rows = []
    for group_id, capacity in capacity_by_group.items():
        for hour, actual_ratio in enumerate([0.10, 0.30, 0.60], start=1):
            y_true = capacity * actual_ratio
            rows.append({"forecast_kst_dtm": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour), "group_id": group_id, "y_true": y_true, "y_pred": y_true + capacity * error_ratio})
    return pd.DataFrame(rows)


def run_metric_unit_tests(capacity_by_group: dict[int, float]) -> list[dict[str, object]]:
    cases = [
        ("perfect", 0.00, {"score": 1.0, "one_minus_nmae": 1.0, "ficr": 1.0}),
        ("seven_percent", 0.07, {"score": 0.84, "one_minus_nmae": 0.93, "ficr": 0.75}),
        ("eight_percent_boundary", 0.08, {"score": 0.835, "one_minus_nmae": 0.92, "ficr": 0.75}),
        ("eight_percent_exceeded", 0.080001, {"ficr": 0.0}),
    ]
    results = []
    for name, error_ratio, expected in cases:
        summary, _ = evaluate_baram_score(unit_test_frame(error_ratio, capacity_by_group), capacity_by_group)
        passed = all(np.isclose(summary[key], value, atol=1e-10) for key, value in expected.items())
        if not passed:
            raise ValueError(f"metric unit test 실패: {name}")
        results.append({"test_name": name, "passed": True, "summary": summary})

    boundary = unit_test_frame(0.0, capacity_by_group)
    boundary.loc[boundary.groupby("group_id").head(1).index, "y_true"] = boundary["group_id"].map(capacity_by_group).mul(0.10)
    _, group_metrics = evaluate_baram_score(boundary, capacity_by_group)
    if not group_metrics["eligible_count"].eq(3).all():
        raise ValueError("10% 평가대상 경계 테스트 실패")
    results.append({"test_name": "ten_percent_boundary_included", "passed": True})

    rows = []
    for group_id, row_count in {1: 1, 2: 10, 3: 10}.items():
        capacity = capacity_by_group[group_id]
        for row_number in range(row_count):
            y_true = capacity * 0.50
            error_ratio = 0.00 if group_id == 1 else 0.08
            rows.append(
                {
                    "forecast_kst_dtm": pd.Timestamp("2024-02-01") + pd.Timedelta(hours=row_number),
                    "group_id": group_id,
                    "y_true": y_true,
                    "y_pred": y_true + capacity * error_ratio,
                }
            )
    summary, group_metrics = evaluate_baram_score(pd.DataFrame(rows), capacity_by_group)
    expected_equal_weight_score = float(group_metrics["score"].mean())
    row_weighted_score = float(np.average(group_metrics["score"], weights=group_metrics["eligible_count"]))
    if not np.isclose(summary["score"], expected_equal_weight_score, atol=1e-10):
        raise ValueError("그룹 동일 가중 테스트 실패")
    if np.isclose(summary["score"], row_weighted_score, atol=1e-10):
        raise ValueError("그룹 동일 가중 테스트가 행 가중 평균과 구분되지 않습니다.")
    results.append({"test_name": "group_equal_weighting", "passed": True, "row_weighted_score": row_weighted_score})
    return results


def build_validation_package(project_root: Path) -> dict[str, object]:
    lecture05 = load_lecture05_builder(project_root)
    package = lecture05.build_raw_master_package(project_root)
    tables = package["tables"]
    turbine_metadata = normalize_turbine_metadata(tables["turbine_metadata"])
    capacity_by_group = extract_capacity_by_group(turbine_metadata)
    if capacity_by_group != EXPECTED_CAPACITY:
        raise ValueError("검증된 그룹 설비용량과 다릅니다.")

    frame = build_fold_frame(tables["train_forecast_index"], tables["labels"])
    audit_issue_batch_size(frame)
    folds, primary_manifest = build_issue_quarter_folds(frame)
    for masks in folds.values():
        audit_fold(frame, masks["train_mask"], masks["validation_mask"])

    coverage = label_coverage(frame, folds, capacity_by_group)
    unit_tests = run_metric_unit_tests(capacity_by_group)
    assignments = fold_assignments(frame, folds)
    year_block = build_year_block_fold(frame)
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "forecast_rows": int(len(frame)),
        "issue_batch_count": int(frame["data_available_kst_dtm"].nunique()),
        "issue_batch_size": 24,
        "primary_fold_count": int(len(folds)),
        "capacity_by_group": capacity_by_group,
        "metric_unit_tests_passed": all(item["passed"] for item in unit_tests),
        "no_shared_issue_batch": True,
        "label_purge_checked": True,
        "prediction_clipping": False,
    }
    return {
        "capacity_by_group": capacity_by_group,
        "primary_manifest": primary_manifest,
        "year_block": year_block,
        "coverage": coverage,
        "unit_tests": unit_tests,
        "assignments": assignments,
        "protocol": protocol(),
        "audit": audit,
    }


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_outputs(package: dict[str, object], output_dir: Path) -> None:
    folds_dir = output_dir / "folds"
    metadata_dir = output_dir / "metadata"
    reports_dir = output_dir / "reports"
    for directory in [folds_dir, metadata_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    package["assignments"].to_csv(folds_dir / "primary_fold_assignments.csv", index=False, encoding="utf-8-sig")
    package["primary_manifest"].to_csv(folds_dir / "primary_fold_manifest.csv", index=False, encoding="utf-8-sig")
    write_json(package["year_block"], folds_dir / "year_block_manifest.json")
    write_json(package["protocol"], metadata_dir / "validation_protocol.json")
    write_json(package["capacity_by_group"], metadata_dir / "capacity_by_group.json")
    write_json(package["unit_tests"], reports_dir / "metric_unit_tests.json")
    package["coverage"].to_csv(reports_dir / "label_coverage_by_fold.csv", index=False, encoding="utf-8-sig")
    write_json(package["audit"], reports_dir / "validation_audit.json")
    write_audit_markdown(package, reports_dir / "validation_audit.md")


def write_audit_markdown(package: dict[str, object], path: Path) -> None:
    audit = package["audit"]
    manifest = package["primary_manifest"]
    lines = [
        "# Lecture 07 Validation Audit",
        "",
        f"- Protocol version: `{audit['protocol_version']}`",
        f"- Forecast rows: {audit['forecast_rows']:,}",
        f"- Issue batch count: {audit['issue_batch_count']:,}",
        f"- Issue batch size: {audit['issue_batch_size']}",
        f"- Metric unit tests passed: {audit['metric_unit_tests_passed']}",
        f"- Prediction clipping in metric: {audit['prediction_clipping']}",
        "",
        "| fold | validation period | train rows | validation rows | issue start | issue end |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in manifest.to_dict("records"):
        lines.append(
            f"| `{row['fold_name']}` | {row['validation_period']} | {row['train_row_count']:,} | "
            f"{row['validation_row_count']:,} | {row['validation_issue_start']} | {row['validation_issue_end']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Lecture 07 validation protocol and metric audit.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "lecture07_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = build_validation_package(args.project_root)
    write_outputs(package, args.output_dir)
    print(json.dumps(package["audit"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
