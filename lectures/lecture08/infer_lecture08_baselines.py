from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lecture08_data import build_feature_tables, ensure_lecture06_package, write_json
from lecture08_disagreement import build_test_disagreement, disagreement_summary, summarize_test_disagreement_by_issue
from lecture08_models import MODEL_NAMES, load_final_models, predict_test_long, require_model_dependencies
from submission_utils import audit_submission, build_submission, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer Lecture 08 boosting baselines and write submissions.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "lecture08_baselines")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in ["test_predictions", "submissions", "reports"]:
        (args.input_dir / directory).mkdir(parents=True, exist_ok=True)
    require_model_dependencies(args.models)
    ensure_lecture06_package(args.project_root)
    _, test_frame, feature_columns = build_feature_tables(args.project_root)
    saved_columns = json.loads((args.input_dir / "metadata" / "feature_columns.json").read_text(encoding="utf-8"))
    if feature_columns != saved_columns:
        raise ValueError("저장된 feature column과 현재 Test feature가 다릅니다.")
    capacity = {int(k): float(v) for k, v in json.loads((args.input_dir / "metadata" / "capacity_by_group.json").read_text(encoding="utf-8")).items()}
    sample = pd.read_csv(args.project_root / "sample_submission.csv", encoding="utf-8-sig")
    submission_audit, prediction_by_model = {}, {}
    for model_name in args.models:
        models = load_final_models(model_name, args.input_dir / "models")
        prediction = predict_test_long(model_name, models, test_frame, feature_columns)
        prediction_by_model[model_name] = prediction
        prediction.to_csv(args.input_dir / "test_predictions" / f"{model_name}_test_predictions.csv", index=False, encoding="utf-8-sig")
        submission = build_submission(args.project_root / "sample_submission.csv", prediction)
        submission_audit[model_name] = audit_submission(submission, sample, capacity)
        write_submission(submission, args.input_dir / "submissions" / f"lecture08_{model_name}_submit.csv")
    if set(args.models) == set(MODEL_NAMES):
        test_disagreement = build_test_disagreement(prediction_by_model, capacity)
        test_disagreement.to_csv(args.input_dir / "test_predictions" / "test_disagreement.csv", index=False, encoding="utf-8-sig")
        summarize_test_disagreement_by_issue(test_disagreement).to_csv(args.input_dir / "test_predictions" / "test_disagreement_by_issue.csv", index=False, encoding="utf-8-sig")
        write_json(disagreement_summary(None, test_disagreement), args.input_dir / "reports" / "test_disagreement_summary.json")
    write_json(submission_audit, args.input_dir / "reports" / "submission_audit.json")
    print(json.dumps({"submissions": list(submission_audit)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
