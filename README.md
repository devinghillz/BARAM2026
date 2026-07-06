# BARAM 2026 — 풍력발전량 예측 AI 경진대회

제3회 풍력발전량 예측 AI 경진대회 (BARAM 2026) 팀 저장소입니다.

## 데이터

대회 제공 데이터는 `train/`, `test/` 디렉터리에 있습니다. 컬럼·기간·제약 사항은 [`data_description.md`](data_description.md)를 참고하세요.

| 경로 | 설명 |
|------|------|
| `train/train_labels.csv` | 학습용 KPX 그룹별 실제 발전량 |
| `train/ldaps_train.csv` | LDAPS 기상예보 (학습) |
| `train/gfs_train.csv` | GFS 기상예보 (학습) |
| `train/scada_*_train.csv` | SCADA 실측 (학습·분석용) |
| `test/ldaps_test.csv` | LDAPS 기상예보 (평가) |
| `test/gfs_test.csv` | GFS 기상예보 (평가) |
| `sample_submission.csv` | 제출 양식 |
| `info.xlsx` | 터빈·KPX 그룹 메타 정보 |

## 평가

- **Score** = 0.5 × (1 − NMAE) + 0.5 × FICR
- 예측 대상: 2025년 전체, 3개 KPX 그룹 시간별 발전량 (kWh)

[대회 페이지](https://dacon.io/competitions/official/236727/overview/description)
