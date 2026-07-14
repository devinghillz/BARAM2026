"""
v0.9 전략 비교 — mild / fullws+mildslot / LGBM 시드 앙상블.

로컬 hold-out(2024)으로 Score·NMAE·FICR 비교 후 상위 제출 CSV 생성.

사용법:
  python scripts/eval_v09_strategies.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_slot_multipliers, apply_v04_calibrations, apply_ws_band_multipliers
from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, OUTPUT_DIR, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.eval_local_all import build_predictions
from scripts.train_v03 import (
    _filter_group_train,
    blend_wide,
    build_dataset,
    ficr_sample_weights,
    get_feature_columns,
    predict_wide,
)

COLS = GROUP_COLUMNS
Q_V05 = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW = 0.9
WS_FULL = {1: 1.10, 2: 1.08, 3: 1.0}
WS_MILD = {1: 1.08, 2: 1.06, 3: 1.0}
SLOT_FULL = {1: 1.06, 2: 1.04, 3: 1.06}
SLOT_MILD = {1: 1.04, 2: 1.03, 3: 1.04}
SEEDS = [42, 7, 123]
LB_V05B = 0.615


def make_lgbm_mean(seed: int):
    import lightgbm as lgb

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                lgb.LGBMRegressor(
                    objective="regression",
                    n_estimators=900,
                    learning_rate=0.03,
                    num_leaves=63,
                    min_child_samples=25,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=0.2,
                    random_state=seed,
                    verbose=-1,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_lgbm_quantile(seed: int, alpha: float = 0.58):
    import lightgbm as lgb

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    n_estimators=700,
                    learning_rate=0.035,
                    num_leaves=47,
                    min_child_samples=30,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_alpha=0.15,
                    reg_lambda=0.25,
                    random_state=seed,
                    verbose=-1,
                    n_jobs=-1,
                ),
            ),
        ]
    )


class GroupBlendModelSeed:
    def __init__(self, q_weight: float, seed: int):
        self.q_weight = q_weight
        self.seed = seed
        self.mean_model = make_lgbm_mean(seed)
        self.q_model = make_lgbm_quantile(seed)
        self.capacity = 21600.0

    def fit(self, X, y, capacity: float):
        self.capacity = capacity
        w = ficr_sample_weights(y.to_numpy(), capacity)
        self.mean_model.fit(X, y, regressor__sample_weight=w)
        self.q_model.fit(X, y, regressor__sample_weight=w)
        return self

    def predict(self, X) -> np.ndarray:
        pred = (1 - self.q_weight) * self.mean_model.predict(X) + self.q_weight * self.q_model.predict(X)
        return np.clip(pred, 0, self.capacity)


def train_seed_ensemble(train_df, feat, q_weights: dict[int, float], seeds: list[int]):
    """시드별 모델 dict 반환: {seed: {gid: model}}."""
    out = {}
    for seed in seeds:
        models = {}
        for gid in [1, 2, 3]:
            cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
            part = _filter_group_train(train_df, gid)
            m = GroupBlendModelSeed(q_weight=q_weights[gid], seed=seed)
            m.fit(part[feat], part["power_kwh"], cap)
            models[gid] = m
        out[seed] = models
    return out


def predict_seed_ensemble(ensemble, infer_df, feat) -> pd.DataFrame:
    """시드 평균 wide 예측."""
    acc = None
    for seed, models in ensemble.items():
        p = predict_wide(models, infer_df, feat)
        if acc is None:
            acc = p.copy()
        else:
            for c in COLS:
                acc[c] = acc[c] + p[c]
    for c in COLS:
        acc[c] = acc[c] / len(ensemble)
    return acc


def apply_calib(raw, long_df, ws, slot):
    return apply_slot_multipliers(apply_ws_band_multipliers(raw, long_df, ws), slot)


def score_case(name, pred, truth, note: str = "") -> dict:
    s = evaluate_submission(truth, pred, time_col="kst_dtm")
    return {
        "case": name,
        "local": round(s["score"], 6),
        "nmae": round(s["1_minus_nmae"], 6),
        "ficr": round(s["ficr"], 6),
        "note": note,
    }


def save_submission(pred, name):
    sub = load_submission_template().drop(columns=COLS)
    sub.merge(pred, on="forecast_kst_dtm", how="left").to_csv(
        SUBMISSION_DIR / name, index=False, encoding="utf-8-sig"
    )


def main():
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)
    truth = labels.loc[labels.kst_dtm >= vs, ["kst_dtm", *COLS]]

    print("=" * 60)
    print("[1/3] 기준선 + mild / fullws+mildslot (기존 v05b 학습)")
    print("=" * 60)
    _, preds = build_predictions(labels, scada, vs)
    raw = preds["v05b_raw"]
    long_val = build_dataset(labels, "idw", vs, scada, "train")
    long_val = long_val[long_val.forecast_kst_dtm >= vs]

    results = []
    results.append(score_case("A_v05b_제출본", preds["v05b"], truth, f"LB확인 {LB_V05B}"))
    results.append(score_case("B_mild_ws+slot", apply_calib(raw, long_val, WS_MILD, SLOT_MILD), truth))
    results.append(score_case("C_fullws+mildslot", apply_calib(raw, long_val, WS_FULL, SLOT_MILD), truth))

    print("=" * 60)
    print(f"[2/3] LGBM 시드 앙상블 ({len(SEEDS)} seeds: {SEEDS})")
    print("=" * 60)
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    feat = get_feature_columns(train_idw)
    fit_n = train_near.forecast_kst_dtm < vs
    fit_i = train_idw.forecast_kst_dtm < vs
    val_n = train_near.forecast_kst_dtm >= vs
    val_i = train_idw.forecast_kst_dtm >= vs

    print("  nearest 앙상블 학습...")
    ens_n = train_seed_ensemble(train_near.loc[fit_n], feat, Q_V05, SEEDS)
    print("  idw 앙상블 학습...")
    ens_i = train_seed_ensemble(train_idw.loc[fit_i], feat, Q_V05, SEEDS)

    raw_ens = blend_wide(
        predict_seed_ensemble(ens_n, train_near.loc[val_n], feat),
        predict_seed_ensemble(ens_i, long_val, feat),
        W_IDW,
    )
    results.append(score_case("D_seed3_보정전", raw_ens, truth))
    results.append(score_case("E_seed3_mild", apply_calib(raw_ens, long_val, WS_MILD, SLOT_MILD), truth))
    results.append(score_case("F_seed3_fullws+mildslot", apply_calib(raw_ens, long_val, WS_FULL, SLOT_MILD), truth))
    results.append(
        score_case(
            "G_seed3_v05b_full",
            apply_v04_calibrations(raw_ens, long_val, WS_FULL, slot_mult=SLOT_FULL),
            truth,
        )
    )

    # 단일모델 vs 앙상블 50:50
    blend50 = blend_wide(raw, raw_ens, 0.5)
    results.append(score_case("H_blend50_single+seed", apply_calib(blend50, long_val, WS_FULL, SLOT_MILD), truth))

    ranked = sorted(results, key=lambda x: x["local"], reverse=True)

    print("\n" + "=" * 60)
    print("[3/3] 결과 (로컬 hold-out 2024)")
    print("=" * 60)
    print(f"\n{'케이스':<28} {'Score':>7} {'NMAE':>7} {'FICR':>7}  비고")
    print("-" * 68)
    for r in ranked:
        print(f"{r['case']:<28} {r['local']:7.4f} {r['nmae']:7.4f} {r['ficr']:7.4f}  {r['note']}")

    best = ranked[0]
    gap = LB_V05B - results[0]["local"]
    print(f"\n★ 로컬 1위: {best['case']}  Score={best['local']:.4f}")
    print(f"  LB 추정 (v05b 갭 {gap:+.3f} 적용): {best['local'] + gap:.4f}")
    print(f"  ※ mild/fullws+mildslot는 갭이 더 작을 수 있음 (~{gap + 0.006:.3f})")

    # 상위 2개 전략 test CSV (전체 학습)
    print("\n테스트 예측 생성 (상위 전략)...")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")

    # v05b baseline test (from train_v05b logic)
    from scripts.train_v04 import train_all_groups_qmap

    f5n = train_all_groups_qmap(train_near, feat, Q_V05)
    f5i = train_all_groups_qmap(train_idw, feat, Q_V05)
    raw_test = blend_wide(predict_wide(f5n, test_near, feat), predict_wide(f5i, test_idw, feat), W_IDW)

    ens_n_full = train_seed_ensemble(train_near, feat, Q_V05, SEEDS)
    ens_i_full = train_seed_ensemble(train_idw, feat, Q_V05, SEEDS)
    raw_ens_test = blend_wide(
        predict_seed_ensemble(ens_n_full, test_near, feat),
        predict_seed_ensemble(ens_i_full, test_idw, feat),
        W_IDW,
    )

    SUBMISSION_DIR.mkdir(exist_ok=True)
    submissions = {
        "v09_mild_calib.csv": apply_calib(raw_test, test_idw, WS_MILD, SLOT_MILD),
        "v09_fullws_mildslot.csv": apply_calib(raw_test, test_idw, WS_FULL, SLOT_MILD),
        "v09_seed3_fullws_mildslot.csv": apply_calib(raw_ens_test, test_idw, WS_FULL, SLOT_MILD),
        "v09_seed3_mild.csv": apply_calib(raw_ens_test, test_idw, WS_MILD, SLOT_MILD),
    }
    for fname, pred in submissions.items():
        save_submission(pred, fname)
        print(f"  -> submissions/{fname}")

    out = OUTPUT_DIR / "v09_strategies_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ranked": ranked, "best": best, "gap_v05b": round(gap, 4), "seeds": SEEDS}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
