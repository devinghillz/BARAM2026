# BARAM 2026 Lecture 05 Master Data Package

이 폴더는 제5강 문서의 원본 보존형 마스터 데이터 패키지를 재현하기 위한 코드와 산출물을 둔다.

## 실행

```bash
python3 lectures/lecture05/build_lecture05_master_data.py
```

기본 출력 위치:

```text
lectures/lecture05/lecture05_master_data_package/
```

## 기본 산출물

```text
metadata/
  train_forecast_index.csv
  test_forecast_index.csv
  ldaps_grid_metadata.csv
  gfs_grid_metadata.csv
  turbine_metadata.csv
master/
  weather_test_raw_wide.csv
  label_availability.csv
  master_train_with_labels.csv
  schema_manifest.json
reports/
  audit_summary.json
  audit_summary.md
```

현재 기본 환경에는 `openpyxl`, `pyarrow`, `fastparquet`가 없으므로 CSV가 저장된다. Parquet 엔진을 설치한 환경에서 실행하면 같은 위치에 Parquet 파일도 추가 저장된다.

원본 정렬 long 테이블과 중간 train weather/label 테이블까지 모두 저장하려면 다음 옵션을 붙인다.

```bash
python3 lectures/lecture05/build_lecture05_master_data.py --save-full-package
```

## 원칙

- 원본 `train/`, `test/`, `info.xlsx`는 덮어쓰지 않는다.
- LDAPS와 GFS는 모든 격자와 원 기상변수를 wide 열로 보존한다.
- Label availability mask는 feature 테이블에 붙이지 않고 별도 파일로 둔다.
- 결측값은 보간, 0 대치, 삭제 없이 그대로 보존한다.
