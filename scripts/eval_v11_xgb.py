"""
v1.1 XGBoost + LGBM 블렌드 후 최적 후처리 적용.

사용법:
  python scripts/eval_v11_xgb.py
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
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_slot_multipliers, apply_ws_band_multipliers
from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, OUTPUT_DIR, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.eval_v10_priorities import LB_V05B, SLOT_MILD, WS_FULL, apply_util_tier_boost
from scripts.eval_v11_grid import apply_pipeline, high_mult
from scripts.train_v03 import (
    _filter_group_train,
    blend_wide,
    build_dataset,
    ficr_sample_weights,
    get_feature_columns,
    predict_wide,
)
from scripts.train_v04 import train_all_groups_qmap

COLS = GROUP_COLUMNS
Q_V05 = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW = 0.9
BEST = {"high_s": 0.0, "mid_m": 1.0, "high_lo": 0.60, "slot_type": "mild"}


def make_xgb():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                xgb.XGBRegressor(
                    n_estimators=800,
                    learning_rate=0.03,
                    max_depth=8,
                    min_child_weight=25,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=0.2,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


def train_xgb_groups(train_df, feat, q_weights):
    models = {}
    for gid in [1, 2, 3]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        w = ficr_sample_weights(part["power_kwh"].to_numpy(), cap)
        m = make_xgb()
        m.fit(part[feat], part["power_kwh"], regressor__sample_weight=w)
        models[gid] = (m, cap)
    return models


def predict_xgb_wide(models, infer_df, feat):
    rows = []
    for gid, (m, cap) in models.items():
        p = infer_df[infer_df.group_id == gid].copy()
        pred = m.predict(p[feat])
        p["pred_kwh"] = np.clip(pred, 0, cap)
        rows.append(p)
    long = pd.concat(rows, ignore_index=True)
    return (
        long.pivot(index="forecast_kst_dtm", columns="group_id", values="pred_kwh")
        .rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
        .reset_index()
    )


def blend_raw(lgbm, xgb_p, alpha_xgb):
    out = lgbm.copy()
    for c in COLS:
        out[c] = (1 - alpha_xgb) * lgbm[c] + alpha_xgb * xgb_p[c]
    return out


def apply_best_post(raw, long_df):
    return apply_pipeline(
        raw,
        long_df,
        BEST["high_s"],
        BEST["mid_m"],
        BEST["high_lo"],
        slot=SLOT_MILD,
    )


def score(name, pred, truth):
    s = evaluate_submission(truth, pred, time_col="kst_dtm")
    return {"case": name, "local": round(s["score"], 6), "nmae": round(s["1_minus_nmae"], 6), "ficr": round(s["ficr"], 6)}


def save_sub(pred, fname):
    sub = load_submission_template().drop(columns=COLS)
    sub.merge(pred, on="forecast_kst_dtm", how="left").to_csv(
        SUBMISSION_DIR / fname, index=False, encoding="utf-8-sig"
    )


def main():
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)
    truth = labels.loc[labels.kst_dtm >= vs, ["kst_dtm", *COLS]]

    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    feat = get_feature_columns(train_idw)
    fit_n = train_near.forecast_kst_dtm < vs
    fit_i = train_idw.forecast_kst_dtm < vs
    val_n = train_near.forecast_kst_dtm >= vs
    val_i = train_idw.forecast_kst_dtm >= vs
    long_val = train_idw.loc[val_i]

    print("LGBM 학습...")
    ln = train_all_groups_qmap(train_near.loc[fit_n], feat, Q_V05)
    li = train_all_groups_qmap(train_idw.loc[fit_i], feat, Q_V05)
    lgbm_val = blend_wide(predict_wide(ln, train_near.loc[val_n], feat), predict_wide(li, long_val, feat), W_IDW)

    print("XGBoost 학습...")
    xn = train_xgb_groups(train_near.loc[fit_n], feat, Q_V05)
    xi = train_xgb_groups(train_idw.loc[fit_i], feat, Q_V05)
    xgb_val = blend_wide(predict_xgb_wide(xn, train_near.loc[val_n], feat), predict_xgb_wide(xi, long_val, feat), W_IDW)

    R = []
    R.append(score("lgbm+best_post", apply_best_post(lgbm_val, long_val), truth))

    for ax in [0.10, 0.15, 0.20, 0.25]:
        raw = blend_raw(lgbm_val, xgb_val, ax)
        R.append(score(f"blend_xgb{int(ax*100)}+best", apply_best_post(raw, long_val), truth))

    R.append(score("xgb_only+best", apply_best_post(xgb_val, long_val), truth))

    ranked = sorted(R, key=lambda x: x["local"], reverse=True)
    print(f"\n{'케이스':<28} {'Score':>7} {'NMAE':>7} {'FICR':>7}")
    print("-" * 54)
    for r in ranked:
        print(f"{r['case']:<28} {r['local']:7.4f} {r['nmae']:7.4f} {r['ficr']:7.4f}")

    best = ranked[0]
    gap = LB_V05B - 0.633473
    print(f"\n1위: {best['case']}  LB추정: {best['local'] + gap:.4f}")

    # test export for best blend if beats lgbm baseline
    best_ax = 0.15
    if ranked[0]["case"].startswith("blend_xgb"):
        best_ax = int(ranked[0]["case"].split("xgb")[1].split("+")[0]) / 100

    print("\n테스트 CSV...")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    fln = train_all_groups_qmap(train_near, feat, Q_V05)
    fli = train_all_groups_qmap(train_idw, feat, Q_V05)
    fxn = train_xgb_groups(train_near, feat, Q_V05)
    fxi = train_xgb_groups(train_idw, feat, Q_V05)
    lgbm_test = blend_wide(predict_wide(fln, test_near, feat), predict_wide(fli, test_idw, feat), W_IDW)
    xgb_test = blend_wide(predict_xgb_wide(fxn, test_near, feat), predict_xgb_wide(fxi, test_idw, feat), W_IDW)

    SUBMISSION_DIR.mkdir(exist_ok=True)
    save_sub(apply_best_post(lgbm_test, test_idw), "v11_lgbm_bestpost.csv")
    save_sub(apply_best_post(blend_raw(lgbm_test, xgb_test, best_ax), test_idw), f"v11_lgbm_xgb{int(best_ax*100)}_bestpost.csv")

    out = OUTPUT_DIR / "v11_xgb_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ranked": ranked, "best": best, "best_ax": best_ax}, f, ensure_ascii=False, indent=2)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
