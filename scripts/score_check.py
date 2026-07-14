"""로컬 hold-out 스코어 확인.

사용법:
  python scripts/score_check.py
  python scripts/score_check.py --submission submissions/baseline_ldaps_nearest.csv
  python scripts/score_check.py --valid-start "2024-07-01 01:00:00"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import GROUP_COLUMNS, VALID_START
from src.data_loader import load_labels
from src.metrics import evaluate_submission, metric


def score_holdout(submission_path: Path, valid_start: str) -> dict[str, float]:
    labels = load_labels()
    valid_start_ts = pd.Timestamp(valid_start)

    # hold-out 구간 라벨로 로컬 채점 (test 라벨은 비공개)
    answer = labels.loc[labels["kst_dtm"] >= valid_start_ts, ["kst_dtm", *GROUP_COLUMNS]]
    pred = pd.read_csv(submission_path, encoding="utf-8-sig")
    pred["forecast_kst_dtm"] = pd.to_datetime(pred["forecast_kst_dtm"])

    # submission이 2025 test 전체면 hold-out과 겹치는 구간이 없음 →
    # 학습 라벨 기간 내에서만 검증하려면 baseline이 train 기간 예측을 내야 함.
    # 여기서는 submission 시각이 answer와 겹치는 부분만 채점.
    merged = answer.merge(
        pred,
        left_on="kst_dtm",
        right_on="forecast_kst_dtm",
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            "submission과 hold-out 라벨의 시간이 겹치지 않습니다. "
            "2025 test submission은 로컬 라벨로 직접 채점할 수 없습니다. "
            "baseline_train_predict.py 의 validation 출력을 참고하세요."
        )

    answer_df = merged[GROUP_COLUMNS]
    pred_df = merged[[c for c in GROUP_COLUMNS if c in merged.columns]]
    total, nmae, ficr = metric(answer_df, pred_df)
    return {
        "score": total,
        "1_minus_nmae": nmae,
        "ficr": ficr,
        "n_rows": len(merged),
        "valid_start": str(valid_start_ts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--valid-start", default=VALID_START)
    args = parser.parse_args()

    if args.submission is None:
        print("submission 파일 없음 — baseline validation을 실행하세요:")
        print("  python scripts/baseline_train_predict.py")
        return

    result = score_holdout(args.submission, args.valid_start)
    print(f"hold-out from: {result['valid_start']}")
    print(f"rows scored : {result['n_rows']}")
    print(f"1-NMAE      : {result['1_minus_nmae']:.6f}")
    print(f"FICR        : {result['ficr']:.6f}")
    print(f"Score       : {result['score']:.6f}")


if __name__ == "__main__":
    main()
