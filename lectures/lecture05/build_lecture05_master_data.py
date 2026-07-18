from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


TIME_KEYS = ["forecast_kst_dtm", "data_available_kst_dtm"]
SPATIAL_KEYS = ["grid_id", "latitude", "longitude"]
WEATHER_KEY = [*TIME_KEYS, "grid_id"]
META_COLUMNS = [*TIME_KEYS, *SPATIAL_KEYS]
TARGET_COLUMNS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

INFO_COLUMNS = [
    "단계",
    "명칭",
    "제작사",
    "모델명",
    "호기",
    "좌표(Google)",
    "KPX그룹",
    "Hub Height(m)",
    "Rotor Diameter(m)",
    "설비용량(MW)",
    "그룹설비용량(MW)",
]


def read_weather_raw(path: str | Path) -> pd.DataFrame:
    weather = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = set(META_COLUMNS) - set(weather.columns)
    if missing_columns:
        raise ValueError(f"{path}: 필수 컬럼이 없습니다: {sorted(missing_columns)}")

    result = weather.copy()
    for column in TIME_KEYS:
        result[column] = pd.to_datetime(result[column], errors="raise")

    if result[WEATHER_KEY].isna().any().any():
        raise ValueError(f"{path}: 시간 또는 grid key에 결측이 있습니다.")

    return result.sort_values(WEATHER_KEY).reset_index(drop=True)


def read_labels(path: str | Path) -> pd.DataFrame:
    labels = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = {"kst_dtm", *TARGET_COLUMNS} - set(labels.columns)
    if missing_columns:
        raise ValueError(f"{path}: Label 필수 컬럼이 없습니다: {sorted(missing_columns)}")

    result = labels.copy()
    result["kst_dtm"] = pd.to_datetime(result["kst_dtm"], errors="raise")
    if result["kst_dtm"].duplicated().any():
        raise ValueError("Label 시각 중복이 있습니다.")

    return result.sort_values("kst_dtm").reset_index(drop=True)


def read_turbine_metadata(path: str | Path) -> pd.DataFrame:
    try:
        info = pd.read_excel(path, sheet_name="info", header=3)
    except ImportError:
        info = read_info_xlsx_without_openpyxl(path)

    info = info.dropna(axis=1, how="all")
    missing_columns = set(INFO_COLUMNS) - set(info.columns)
    if missing_columns:
        raise ValueError(f"info.xlsx 필수 컬럼이 없습니다: {sorted(missing_columns)}")

    result = info[INFO_COLUMNS].copy().reset_index(drop=True).replace("", pd.NA)
    result["KPX그룹"] = result["KPX그룹"].ffill()
    result["그룹설비용량(MW)"] = result["그룹설비용량(MW)"].ffill()
    coordinates = result["좌표(Google)"].map(parse_google_coordinate)
    result["turbine_latitude"] = coordinates.map(lambda value: value[0])
    result["turbine_longitude"] = coordinates.map(lambda value: value[1])
    return result


def parse_google_coordinate(value: object) -> tuple[float, float]:
    pattern = r"""^\s*(\d+(?:\.\d+)?)°(\d+(?:\.\d+)?)'(\d+(?:\.\d+)?)"([NS])\s+(\d+(?:\.\d+)?)°(\d+(?:\.\d+)?)'(\d+(?:\.\d+)?)"([EW])\s*$"""
    match = re.match(pattern, str(value).strip())
    if not match:
        raise ValueError(f"좌표(Google)를 파싱할 수 없습니다: {value}")
    values = match.groups()
    to_decimal = lambda d, m, s, direction: (float(d) + float(m) / 60 + float(s) / 3600) * (-1 if direction in {"S", "W"} else 1)
    return to_decimal(*values[:4]), to_decimal(*values[4:])


def read_info_xlsx_without_openpyxl(path: str | Path) -> pd.DataFrame:
    """Minimal xlsx reader for this workbook, used when openpyxl is unavailable."""
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        shared_strings = parse_shared_strings(workbook, namespace)
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    rows: dict[int, dict[int, object]] = {}
    for row in sheet.find("x:sheetData", namespace).findall("x:row", namespace):
        row_index = int(row.attrib["r"])
        values: dict[int, object] = {}
        for cell in row.findall("x:c", namespace):
            value_node = cell.find("x:v", namespace)
            if value_node is None:
                value: object = ""
            elif cell.attrib.get("t") == "s":
                value = shared_strings[int(value_node.text or 0)]
            else:
                value = coerce_excel_scalar(value_node.text or "")
            values[column_number(cell.attrib["r"])] = value
        rows[row_index] = values

    header_row = rows[4]
    headers = [str(header_row[column]) for column in sorted(header_row)]
    records = []
    for row_index in sorted(index for index in rows if index > 4):
        row = rows[row_index]
        record = {header: row.get(column, "") for header, column in zip(headers, sorted(header_row))}
        if any(value != "" for value in record.values()):
            records.append(record)

    return pd.DataFrame(records)


def parse_shared_strings(workbook: zipfile.ZipFile, namespace: dict[str, str]) -> list[str]:
    shared = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings = []
    for item in shared.findall("x:si", namespace):
        texts = [node.text or "" for node in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
        strings.append("".join(texts))
    return strings


def column_number(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value


def coerce_excel_scalar(value: str) -> object:
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return int(numeric)
    return numeric


def audit_weather_long(weather: pd.DataFrame, expected_grid_count: int, name: str) -> dict[str, object]:
    duplicated_count = int(weather.duplicated(WEATHER_KEY).sum())
    if duplicated_count:
        raise ValueError(f"{name}: 시간·격자 중복이 있습니다: {duplicated_count}")

    rows_per_forecast = weather.groupby(TIME_KEYS).size()
    invalid = rows_per_forecast[~rows_per_forecast.eq(expected_grid_count)]
    if len(invalid):
        raise ValueError(f"{name}: 시각별 격자 수 오류\n{invalid.head()}")

    return {
        "rows": int(len(weather)),
        "columns": int(len(weather.columns)),
        "unique_forecast_times": int(weather["forecast_kst_dtm"].nunique()),
        "expected_grid_count": int(expected_grid_count),
        "duplicated_weather_keys": duplicated_count,
        "missing_cells": int(weather.isna().sum().sum()),
    }


def audit_grid_coordinates(weather: pd.DataFrame, name: str) -> None:
    coordinate_counts = weather.groupby("grid_id")[["latitude", "longitude"]].nunique(dropna=False)
    if not coordinate_counts.eq(1).all().all():
        raise ValueError(f"{name}: grid_id 좌표가 시간에 따라 변합니다.")


def audit_availability_time(weather: pd.DataFrame, name: str) -> None:
    availability_counts = weather.groupby("forecast_kst_dtm")["data_available_kst_dtm"].nunique()
    if not availability_counts.eq(1).all():
        raise ValueError(f"{name}: 하나의 예측시각에 여러 이용 가능 시각이 있습니다.")


def audit_forecast_availability(forecast_index: pd.DataFrame, name: str) -> dict[str, object]:
    missing_columns = set(TIME_KEYS) - set(forecast_index.columns)
    if missing_columns:
        raise ValueError(f"{name}: 시간키가 없습니다: {sorted(missing_columns)}")
    if forecast_index[TIME_KEYS].isna().any().any():
        raise ValueError(f"{name}: 시간키에 결측이 있습니다.")
    if forecast_index.duplicated(TIME_KEYS).any():
        raise ValueError(f"{name}: 시간키가 중복됩니다.")
    if not forecast_index["data_available_kst_dtm"].lt(forecast_index["forecast_kst_dtm"]).all():
        raise ValueError(f"{name}: 예측 대상 이후 이용 가능한 예보가 있습니다.")

    lead_hour = (
        forecast_index["forecast_kst_dtm"]
        .sub(forecast_index["data_available_kst_dtm"])
        .dt.total_seconds()
        .div(3600.0)
    )
    if not lead_hour.between(12, 35, inclusive="both").all():
        raise ValueError(f"{name}: lead_hour가 12~35를 벗어납니다.")
    if set(lead_hour.astype(int).unique()) != set(range(12, 36)):
        raise ValueError(f"{name}: lead_hour 12~35가 모두 존재하지 않습니다.")

    rows_per_issue = forecast_index.groupby("data_available_kst_dtm").size()
    if not rows_per_issue.eq(24).all():
        raise ValueError(f"{name}: 발행시각당 target 수가 24가 아닙니다.")

    return {
        "forecast_rows": int(len(forecast_index)),
        "forecast_times": int(forecast_index["forecast_kst_dtm"].nunique()),
        "issue_batches": int(forecast_index["data_available_kst_dtm"].nunique()),
        "issue_batch_size": 24,
        "lead_hour_min": int(lead_hour.min()),
        "lead_hour_max": int(lead_hour.max()),
    }


def extract_grid_metadata(weather: pd.DataFrame, weather_source: str) -> pd.DataFrame:
    metadata = (
        weather[["grid_id", "latitude", "longitude"]]
        .drop_duplicates()
        .sort_values("grid_id")
        .reset_index(drop=True)
    )
    if metadata["grid_id"].duplicated().any():
        raise ValueError(f"{weather_source}: 같은 grid_id에 여러 좌표가 있습니다.")

    metadata.insert(0, "weather_source", weather_source)
    return metadata


def audit_source_train_test_match(train: pd.DataFrame, test: pd.DataFrame, weather_source: str) -> None:
    train_value_columns = [column for column in train.columns if column not in META_COLUMNS]
    test_value_columns = [column for column in test.columns if column not in META_COLUMNS]
    if train_value_columns != test_value_columns:
        raise ValueError(f"{weather_source}: Train/Test 변수 목록 또는 순서가 다릅니다.")

    train_grid = extract_grid_metadata(train, weather_source=weather_source)
    test_grid = extract_grid_metadata(test, weather_source=weather_source)
    if not train_grid.equals(test_grid):
        raise ValueError(f"{weather_source}: Train/Test 격자 좌표가 다릅니다.")


def extract_forecast_index(weather: pd.DataFrame) -> pd.DataFrame:
    return weather[TIME_KEYS].drop_duplicates().sort_values("forecast_kst_dtm").reset_index(drop=True)


def make_grid_token(grid_id: object) -> str:
    try:
        numeric = float(grid_id)
        if numeric.is_integer():
            return f"{int(numeric):02d}"
    except (TypeError, ValueError):
        pass

    token = re.sub(r"[^0-9A-Za-z]+", "_", str(grid_id)).strip("_")
    if not token:
        raise ValueError(f"grid_id를 열 이름으로 바꿀 수 없습니다: {grid_id}")
    return token


def raw_weather_to_wide(
    weather: pd.DataFrame,
    prefix: str,
    expected_grid_ids: list[object],
    expected_value_columns: list[str],
) -> pd.DataFrame:
    actual_value_columns = [column for column in weather.columns if column not in META_COLUMNS]
    if actual_value_columns != expected_value_columns:
        raise ValueError(f"{prefix}: 예상 변수 schema와 다릅니다.")

    indexed = weather.set_index([*TIME_KEYS, "grid_id"])[expected_value_columns]
    if indexed.index.duplicated().any():
        raise ValueError(f"{prefix}: wide 변환 전 중복 키가 있습니다.")

    wide = indexed.unstack("grid_id")
    expected_columns = pd.MultiIndex.from_product([expected_value_columns, expected_grid_ids])
    wide = wide.reindex(columns=expected_columns)
    wide.columns = [
        f"{prefix}_g{make_grid_token(grid_id)}_{variable}"
        for variable, grid_id in wide.columns
    ]
    return wide.reset_index().sort_values(TIME_KEYS).reset_index(drop=True)


def build_label_availability(labels: pd.DataFrame) -> pd.DataFrame:
    availability = labels[["kst_dtm"]].copy()
    for target in TARGET_COLUMNS:
        mask_name = target.replace("kpx_", "") + "_label_available"
        availability[mask_name] = labels[target].notna()
    return availability


def save_table(frame: pd.DataFrame, path_without_suffix: Path) -> list[str]:
    csv_path = path_without_suffix.with_suffix(".csv")
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return [str(csv_path)]


def build_raw_master_package(project_root: Path) -> dict[str, object]:
    ldaps_train = read_weather_raw(project_root / "train" / "ldaps_train.csv")
    gfs_train = read_weather_raw(project_root / "train" / "gfs_train.csv")
    ldaps_test = read_weather_raw(project_root / "test" / "ldaps_test.csv")
    gfs_test = read_weather_raw(project_root / "test" / "gfs_test.csv")

    audit = {
        "ldaps_train": audit_weather_long(ldaps_train, 16, "ldaps_train"),
        "gfs_train": audit_weather_long(gfs_train, 9, "gfs_train"),
        "ldaps_test": audit_weather_long(ldaps_test, 16, "ldaps_test"),
        "gfs_test": audit_weather_long(gfs_test, 9, "gfs_test"),
    }

    for name, frame in [
        ("ldaps_train", ldaps_train),
        ("gfs_train", gfs_train),
        ("ldaps_test", ldaps_test),
        ("gfs_test", gfs_test),
    ]:
        audit_grid_coordinates(frame, name)
        audit_availability_time(frame, name)

    audit_source_train_test_match(ldaps_train, ldaps_test, "ldaps")
    audit_source_train_test_match(gfs_train, gfs_test, "gfs")

    train_forecast_index = extract_forecast_index(ldaps_train)
    test_forecast_index = extract_forecast_index(ldaps_test)
    if not train_forecast_index.equals(extract_forecast_index(gfs_train)):
        raise ValueError("LDAPS와 GFS Train 시간키가 다릅니다.")
    if not test_forecast_index.equals(extract_forecast_index(gfs_test)):
        raise ValueError("LDAPS와 GFS Test 시간키가 다릅니다.")
    audit["train_forecast_index"] = audit_forecast_availability(train_forecast_index, "train_forecast_index")
    audit["test_forecast_index"] = audit_forecast_availability(test_forecast_index, "test_forecast_index")

    ldaps_grid_metadata = extract_grid_metadata(ldaps_train, "ldaps")
    gfs_grid_metadata = extract_grid_metadata(gfs_train, "gfs")
    turbine_metadata = read_turbine_metadata(project_root / "info.xlsx")

    ldaps_value_columns = [column for column in ldaps_train.columns if column not in META_COLUMNS]
    gfs_value_columns = [column for column in gfs_train.columns if column not in META_COLUMNS]
    ldaps_grid_ids = ldaps_grid_metadata["grid_id"].tolist()
    gfs_grid_ids = gfs_grid_metadata["grid_id"].tolist()

    ldaps_train_wide = raw_weather_to_wide(ldaps_train, "ldaps", ldaps_grid_ids, ldaps_value_columns)
    ldaps_test_wide = raw_weather_to_wide(ldaps_test, "ldaps", ldaps_grid_ids, ldaps_value_columns)
    gfs_train_wide = raw_weather_to_wide(gfs_train, "gfs", gfs_grid_ids, gfs_value_columns)
    gfs_test_wide = raw_weather_to_wide(gfs_test, "gfs", gfs_grid_ids, gfs_value_columns)

    weather_train = ldaps_train_wide.merge(gfs_train_wide, on=TIME_KEYS, how="inner", validate="one_to_one")
    weather_test = ldaps_test_wide.merge(gfs_test_wide, on=TIME_KEYS, how="inner", validate="one_to_one")
    if set(weather_train.columns) != set(weather_test.columns):
        train_only = sorted(set(weather_train.columns) - set(weather_test.columns))
        test_only = sorted(set(weather_test.columns) - set(weather_train.columns))
        raise ValueError(f"Train/Test raw-wide 열 집합이 다릅니다.\nTrain only: {train_only}\nTest only: {test_only}")
    weather_test = weather_test[weather_train.columns]

    labels = read_labels(project_root / "train" / "train_labels.csv")
    if set(labels["kst_dtm"]) != set(weather_train["forecast_kst_dtm"]):
        raise ValueError("Train 기상시각과 Label 시각이 다릅니다.")

    master_train_with_labels = (
        weather_train
        .merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="left", validate="one_to_one")
        .drop(columns="kst_dtm")
    )
    label_availability = build_label_availability(labels)

    manifest = {
        "time_keys": TIME_KEYS,
        "ldaps_grid_ids": ldaps_grid_ids,
        "gfs_grid_ids": gfs_grid_ids,
        "ldaps_value_columns": ldaps_value_columns,
        "gfs_value_columns": gfs_value_columns,
        "raw_weather_columns": [column for column in weather_train.columns if column not in TIME_KEYS],
        "targets": TARGET_COLUMNS,
        "weather_train_shape": list(weather_train.shape),
        "weather_test_shape": list(weather_test.shape),
        "master_train_with_labels_shape": list(master_train_with_labels.shape),
        "label_availability_shape": list(label_availability.shape),
    }

    audit.update(
        {
            "labels": {
                "rows": int(len(labels)),
                "columns": int(len(labels.columns)),
                "duplicated_kst_dtm": int(labels["kst_dtm"].duplicated().sum()),
                "missing_by_target": {target: int(labels[target].isna().sum()) for target in TARGET_COLUMNS},
                "valid_by_target": {target: int(labels[target].notna().sum()) for target in TARGET_COLUMNS},
                "all_targets_available_rows": int(labels[TARGET_COLUMNS].notna().all(axis=1).sum()),
            },
            "wide_shapes": {
                "ldaps_train_raw_wide": list(ldaps_train_wide.shape),
                "ldaps_test_raw_wide": list(ldaps_test_wide.shape),
                "gfs_train_raw_wide": list(gfs_train_wide.shape),
                "gfs_test_raw_wide": list(gfs_test_wide.shape),
                "weather_train_raw_wide": list(weather_train.shape),
                "weather_test_raw_wide": list(weather_test.shape),
                "master_train_with_labels": list(master_train_with_labels.shape),
                "label_availability": list(label_availability.shape),
            },
            "wide_missing_cells": {
                "weather_train_raw_wide": int(weather_train.isna().sum().sum()),
                "weather_test_raw_wide": int(weather_test.isna().sum().sum()),
                "master_train_with_labels": int(master_train_with_labels.isna().sum().sum()),
            },
            "turbine_metadata": {
                "rows": int(len(turbine_metadata)),
                "columns": int(len(turbine_metadata.columns)),
            },
        }
    )

    return {
        "tables": {
            "ldaps_train_long": ldaps_train,
            "gfs_train_long": gfs_train,
            "ldaps_test_long": ldaps_test,
            "gfs_test_long": gfs_test,
            "train_forecast_index": train_forecast_index,
            "test_forecast_index": test_forecast_index,
            "ldaps_grid_metadata": ldaps_grid_metadata,
            "gfs_grid_metadata": gfs_grid_metadata,
            "turbine_metadata": turbine_metadata,
            "weather_train_raw_wide": weather_train,
            "weather_test_raw_wide": weather_test,
            "labels": labels,
            "label_availability": label_availability,
            "master_train_with_labels": master_train_with_labels,
        },
        "manifest": manifest,
        "audit": audit,
    }


def write_outputs(package: dict[str, object], output_dir: Path, project_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    master_dir = output_dir / "master"
    reports_dir = output_dir / "reports"
    for directory in [metadata_dir, master_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    table_locations = {
        "train_forecast_index": metadata_dir / "train_forecast_index",
        "test_forecast_index": metadata_dir / "test_forecast_index",
        "ldaps_grid_metadata": metadata_dir / "ldaps_grid_metadata",
        "gfs_grid_metadata": metadata_dir / "gfs_grid_metadata",
        "turbine_metadata": metadata_dir / "turbine_metadata",
        "weather_test_raw_wide": master_dir / "weather_test_raw_wide",
        "label_availability": master_dir / "label_availability",
        "master_train_with_labels": master_dir / "master_train_with_labels",
    }

    saved_files = {}
    tables: dict[str, pd.DataFrame] = package["tables"]
    for name, path_without_suffix in table_locations.items():
        saved_files[name] = save_table(tables[name], path_without_suffix)

    with (master_dir / "schema_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(package["manifest"], file, ensure_ascii=False, indent=2, default=str)

    with (reports_dir / "audit_summary.json").open("w", encoding="utf-8") as file:
        json.dump(package["audit"], file, ensure_ascii=False, indent=2, default=str)

    write_markdown_summary(package, reports_dir / "audit_summary.md", saved_files, project_root)


def write_markdown_summary(
    package: dict[str, object],
    path: Path,
    saved_files: dict[str, list[str]],
    project_root: Path,
) -> None:
    audit = package["audit"]
    manifest = package["manifest"]
    lines = [
        "# Lecture 05 Master Data Package Audit",
        "",
        "## Shapes",
        "",
        "| table | rows | columns |",
        "|---|---:|---:|",
    ]
    for name, shape in audit["wide_shapes"].items():
        lines.append(f"| `{name}` | {shape[0]:,} | {shape[1]:,} |")

    lines.extend(
        [
            "",
            "## Label Missingness",
            "",
            "| target | missing | valid |",
            "|---|---:|---:|",
        ]
    )
    for target in TARGET_COLUMNS:
        missing = audit["labels"]["missing_by_target"][target]
        valid = audit["labels"]["valid_by_target"][target]
        lines.append(f"| `{target}` | {missing:,} | {valid:,} |")

    lines.extend(
        [
            "",
            f"- All targets available rows: {audit['labels']['all_targets_available_rows']:,}",
            f"- Raw weather feature columns: {len(manifest['raw_weather_columns']):,}",
            f"- Weather train missing cells: {audit['wide_missing_cells']['weather_train_raw_wide']:,}",
            f"- Weather test missing cells: {audit['wide_missing_cells']['weather_test_raw_wide']:,}",
            "",
            "## Saved Files",
            "",
        ]
    )
    for name, files in saved_files.items():
        for file_path in files:
            lines.append(f"- `{name}`: `{relative_display_path(file_path, project_root)}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relative_display_path(path: str | Path, project_root: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Lecture 05 raw-preserving master data package.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "lecture05_master_data_package",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = build_raw_master_package(args.project_root)
    write_outputs(package, args.output_dir, args.project_root)
    print(json.dumps(package["audit"]["wide_shapes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
