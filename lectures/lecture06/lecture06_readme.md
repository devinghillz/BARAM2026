# BARAM 2026 Lecture 06 Feature Candidates

제6강 feature 후보군 생성 코드와 작은 검증 산출물을 둔다.

## 실행

```bash
python3 lectures/lecture06/build_lecture06_feature_candidates.py
```

기본 출력 위치:

```text
lectures/lecture06/lecture06_feature_candidates/
```

## 산출물

대용량 feature block CSV는 로컬 재생성 산출물이라 git에는 올리지 않는다.

```text
shared/
  time_features_train.csv
  time_features_test.csv
  wind_grid_features_train.csv
  wind_grid_features_test.csv
  grid_statistics_train.csv
  grid_statistics_test.csv
  physical_grid_features_train.csv
  physical_grid_features_test.csv
group/
  center_nearest_train.csv
  center_nearest_test.csv
  turbine_nearest_train.csv
  turbine_nearest_test.csv
  idw_p1_train.csv
  idw_p1_test.csv
  idw_p2_train.csv
  idw_p2_test.csv
  model_difference_*_train.csv
  model_difference_*_test.csv
metadata/
  spatial_weights.csv
  feature_registry.csv
  block_manifest.json
reports/
  audit_summary.json
  audit_summary.md
```

제6강은 feature를 선택하지 않는다. Label을 쓰지 않는 deterministic 후보 block만 만들고, 최종 조합은 이후 validation 단계에서 결정한다.
