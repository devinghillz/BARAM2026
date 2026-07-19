from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from lecture08_data import (
    GROUP_KEYS,
    audit_forecast_availability,
    attach_labels,
    build_feature_tables,
    ensure_lecture06_package,
    ensure_lecture07_package,
    expected_oof_times,
    experiment_config,
    feature_spec,
    load_module,
    load_primary_assignments,
    load_year_block_assignments,
    missing_report,
    package_versions,
    write_json,
)
from lecture08_disagreement import build_oof_disagreement, disagreement_summary
from lecture08_models import MODEL_NAMES, choose_final_iterations, fit_fixed_iteration_model, require_model_dependencies, run_primary_cv, save_final_models, to_float32_matrix, train_final_models


def load_baram_metric(project_root: Path):
    return load_module("baram_metric", project_root / "lectures" / "lecture07" / "baram_metric.py")


def source_commit(project_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=project_root, text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    return {"commit": commit, "dirty": dirty}


def prepare_inputs(project_root: Path):
    ensure_lecture06_package(project_root)
    ensure_lecture07_package(project_root)
    lecture05 = load_module("lecture05_builder", project_root / "lectures" / "lecture05" / "build_lecture05_master_data.py")
    package = lecture05.build_raw_master_package(project_root)
    train_features, test_features, feature_columns = build_feature_tables(project_root)
    train_frame = attach_labels(train_features, package["tables"]["labels"])
    assignment = load_primary_assignments(project_root)
    year_assignment = load_year_block_assignments(project_root)
    capacity = {int(k): float(v) for k, v in json.loads((project_root / "lectures" / "lecture07" / "lecture07_validation" / "metadata" / "capacity_by_group.json").read_text(encoding="utf-8")).items()}
    preflight = {
        "train_forecast": audit_forecast_availability(package["tables"]["train_forecast_index"], 26304),
        "test_forecast": audit_forecast_availability(package["tables"]["test_forecast_index"], 8760),
        "train_shape": list(train_frame.shape),
        "test_shape": list(test_features.shape),
        "feature_count": len(feature_columns),
        "train_missing": missing_report(train_frame, feature_columns),
        "test_missing": missing_report(test_features, feature_columns),
        "expected_oof_time_count": int(expected_oof_times(assignment).nunique()),
    }
    return train_frame, test_features, feature_columns, assignment, year_assignment, capacity, preflight


def evaluate_model_oof(oof: pd.DataFrame, expected_times: pd.Series, capacity_by_group: dict[int, float], metric):
    score_input = oof[["forecast_kst_dtm", "group_id", "y_true", "y_pred"]].copy()
    return metric.evaluate_concatenated_oof([score_input], expected_times, capacity_by_group)


def evaluate_fold_scores(oof: pd.DataFrame, capacity_by_group: dict[int, float], metric) -> pd.DataFrame:
    rows = []
    for fold_name, part in oof.groupby("fold_name", sort=True):
        summary, _ = metric.evaluate_baram_score(part[["forecast_kst_dtm", "group_id", "y_true", "y_pred"]], capacity_by_group)
        rows.append({"fold_name": fold_name, **summary})
    return pd.DataFrame(rows)


def run_year_block(model_name: str, train_frame: pd.DataFrame, year_assignment: pd.DataFrame, feature_columns: list[str], final_iterations: pd.DataFrame, capacity: dict[int, float], metric) -> dict[str, object]:
    predictions = []
    expected_times = year_assignment.loc[year_assignment["split_role"].eq("validation"), "forecast_kst_dtm"].drop_duplicates().sort_values()
    for group_id in [1, 2, 3]:
        group = train_frame.loc[train_frame["group_id"].eq(group_id)].copy()
        train_times = year_assignment.loc[year_assignment["split_role"].eq("train"), ["forecast_kst_dtm", "data_available_kst_dtm"]].drop_duplicates()
        valid_times = year_assignment.loc[year_assignment["split_role"].eq("validation"), ["forecast_kst_dtm", "data_available_kst_dtm"]].drop_duplicates()
        train_part = group.merge(train_times, on=["forecast_kst_dtm", "data_available_kst_dtm"], how="inner", validate="many_to_one")
        train_part = train_part.loc[train_part["y_true"].notna()].copy()
        valid_part = group.merge(valid_times, on=["forecast_kst_dtm", "data_available_kst_dtm"], how="inner", validate="many_to_one")
        n_iter = int(final_iterations.loc[final_iterations["model_name"].eq(model_name) & final_iterations["group_id"].eq(group_id), "final_iteration"].iloc[0])
        model = fit_fixed_iteration_model(model_name, n_iter, to_float32_matrix(train_part, feature_columns), train_part["y_true"].to_numpy("float32"))
        out = valid_part[[*GROUP_KEYS, "y_true"]].copy()
        out["model_name"], out["y_pred"] = model_name, model.predict(to_float32_matrix(valid_part, feature_columns))
        predictions.append(out)
    summary, _ = metric.evaluate_concatenated_oof([pd.concat(predictions)[["forecast_kst_dtm", "group_id", "y_true", "y_pred"]]], expected_times, capacity)
    return {
        "model_name": model_name,
        **summary,
        "year_block_role": "diagnostic_only",
        "year_block_independent": False,
        "used_for_model_selection": False,
    }


def write_preflight(output_root: Path, feature_columns: list[str], capacity: dict[int, float], preflight: dict[str, object]) -> None:
    write_json(experiment_config(), output_root / "metadata" / "experiment_config.json")
    write_json(package_versions(), output_root / "metadata" / "package_versions.json")
    write_json(feature_spec(), output_root / "metadata" / "feature_spec.json")
    write_json(feature_columns, output_root / "metadata" / "feature_columns.json")
    write_json(capacity, output_root / "metadata" / "capacity_by_group.json")
    write_json(preflight, output_root / "reports" / "preflight_audit.json")
    lines = ["# Lecture 08 Baseline Audit", "", f"- Feature count: {preflight['feature_count']:,}", f"- Train shape: {preflight['train_shape']}", f"- Test shape: {preflight['test_shape']}", f"- Expected OOF times: {preflight['expected_oof_time_count']:,}"]
    (output_root / "reports" / "baseline_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "reports" / "baseline_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Lecture 08 boosting baselines.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "lecture08_baselines")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in ["metadata", "models", "oof", "reports"]:
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    metric = load_baram_metric(args.project_root)
    train_frame, _test_frame, feature_columns, assignment, year_assignment, capacity, preflight = prepare_inputs(args.project_root)
    commit_info = source_commit(args.project_root)
    write_preflight(args.output_dir, feature_columns, capacity, preflight)
    write_json(commit_info, args.output_dir / "metadata" / "source_commit.json")
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, **preflight}, ensure_ascii=False, indent=2, default=str))
        return

    require_model_dependencies(args.models)
    expected_times = expected_oof_times(assignment)
    oof_by_model, logs, fold_metric_rows, oof_summary_rows, year_rows = {}, [], [], [], []
    for model_name in args.models:
        oof, training_log = run_primary_cv(model_name, train_frame, assignment, feature_columns)
        summary, group_metrics = evaluate_model_oof(oof, expected_times, capacity, metric)
        fold_scores = evaluate_fold_scores(oof, capacity, metric)
        oof.to_csv(args.output_dir / "oof" / f"{model_name}_oof.csv", index=False, encoding="utf-8-sig")
        training_log["oof_score"] = summary["score"]
        logs.append(training_log)
        fold_metric_rows.append(fold_scores.assign(model_name=model_name))
        oof_summary_rows.append({"model_name": model_name, **summary, "fold_score_mean": float(fold_scores["score"].mean()), "fold_score_std": float(fold_scores["score"].std(ddof=0)), "fold_score_min": float(fold_scores["score"].min()), "worst_group_score": float(group_metrics["score"].min())})
        oof_by_model[model_name] = oof

    training_log = pd.concat(logs, ignore_index=True)
    final_iterations = choose_final_iterations(training_log)
    write_json(final_iterations.to_dict("records"), args.output_dir / "metadata" / "final_iterations.json")
    training_log.to_csv(args.output_dir / "reports" / "training_log.csv", index=False, encoding="utf-8-sig")
    pd.concat(fold_metric_rows, ignore_index=True).to_csv(args.output_dir / "reports" / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    for model_name in args.models:
        year_rows.append(run_year_block(model_name, train_frame, year_assignment, feature_columns, final_iterations, capacity, metric))
        models = train_final_models(model_name, train_frame, feature_columns, final_iterations)
        save_final_models(models, model_name, args.output_dir / "models")

    year_metrics = pd.DataFrame(year_rows)
    oof_summary = pd.DataFrame(oof_summary_rows).merge(year_metrics[["model_name", "score"]].rename(columns={"score": "year_block_score"}), on="model_name", how="left")
    oof_summary.to_csv(args.output_dir / "reports" / "oof_summary.csv", index=False, encoding="utf-8-sig")
    year_metrics.to_csv(args.output_dir / "reports" / "year_block_metrics.csv", index=False, encoding="utf-8-sig")
    if set(args.models) == set(MODEL_NAMES):
        oof_disagreement = build_oof_disagreement(oof_by_model, capacity)
        oof_disagreement.to_csv(args.output_dir / "oof" / "oof_disagreement.csv", index=False, encoding="utf-8-sig")
        write_json(disagreement_summary(oof_disagreement, None), args.output_dir / "reports" / "disagreement_summary.json")
    pd.DataFrame(columns=["submission_name", "model_name", "local_oof_score", "fold_score_mean", "fold_score_std", "fold_score_min", "worst_group_score", "year_block_score", "public_score", "submission_date", "notes"]).to_csv(args.output_dir / "reports" / "public_submission_log.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"trained_models": args.models, "oof_summary": oof_summary.to_dict("records")}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
