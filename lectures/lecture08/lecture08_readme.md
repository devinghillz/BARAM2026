# BARAM 2026 Lecture 08 Boosting Baselines

제8강 130-feature reference baseline으로 LightGBM, XGBoost, CatBoost 학습과 모델별 Dacon 제출 생성을 위한 코드다.

## Preflight

```bash
python3 -B lectures/lecture08/train_lecture08_baselines.py --preflight-only
```

현재 환경에 모델 패키지가 없어도 데이터, feature, fold audit와 metadata를 생성한다.

## Train

```bash
python3 -B lectures/lecture08/train_lecture08_baselines.py
```

필요 패키지:

```bash
python3 -m pip install -r lectures/lecture08/requirements-lecture08.txt
```

## Infer

```bash
python3 -B lectures/lecture08/infer_lecture08_baselines.py
```

## 산출물

```text
lectures/lecture08/lecture08_baselines/
  metadata/
  models/
  oof/
  test_predictions/
  submissions/
  reports/
```

제8강 reference baseline feature는 `time_features`, `calendar_features`, `center_nearest`만 사용한다. `grid_statistics`는 제9강 ablation에서 검증한다. SCADA, clipping, imputation, ensemble은 사용하지 않는다.
