# BARAM 2026 Lecture 07 Validation

제7강 시간순서 validation protocol과 Dacon 공식 평가식 재현 코드를 둔다.

## 실행

```bash
python3 lectures/lecture07/build_lecture07_validation.py
```

기본 출력 위치:

```text
lectures/lecture07/lecture07_validation/
```

## 산출물

```text
folds/
  primary_fold_assignments.csv
  primary_fold_manifest.json
  year_block_assignments.csv
  year_block_manifest.json
metadata/
  validation_protocol.json
  capacity_by_group.json
reports/
  metric_unit_tests.json
  label_coverage_by_fold.json
  validation_audit.json
  validation_audit.md
```

제7강은 모델을 학습하지 않는다. 2024년 issue quarter 기준 expanding window fold, year block stress split, 공식 score 함수, OOF 예측 key 검사를 고정하고 이후 강의의 모델 비교는 이 protocol을 재사용한다.
