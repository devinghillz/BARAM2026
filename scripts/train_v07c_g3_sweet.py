"""v07c: g3 sweet-spot 가중만 (α=0.58 유지, v05b 구조)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_v04_calibrations, tune_slot_multipliers
from src.config import GROUP_CAPACITY_KWH, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.train_v03 import (
    GroupBlendModel,
    blend_wide,
    build_dataset,
    ficr_sample_weights,
    get_feature_columns,
    predict_wide,
    _filter_group_train,
)
from scripts.train_v04 import train_all_groups_qmap

LB_OFFSET = -0.008
Q_WEIGHTS = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW = 0.9
WS_MULT = {1: 1.10, 2: 1.08, 3: 1.0}


def weights_g3_sweet(y, capacity, X):
    w = ficr_sample_weights(y, capacity)
    if "in_ws_sweet_spot" in X.columns:
        util = y / capacity
        mask = X["in_ws_sweet_spot"].astype(bool) & (util >= 0.15)
        w[mask.to_numpy()] *= 1.2
    return w


def train_with_g3_sweet(train_df, feat):
    from scripts.train_v04 import GroupBlendModel as GM
    models = {}
    for gid in [1, 2]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        m = GM(q_weight=Q_WEIGHTS[gid])
        m.fit(part[feat], part["power_kwh"], cap)
        models[gid] = m
    cap3 = GROUP_CAPACITY_KWH["kpx_group_3"]
    part3 = _filter_group_train(train_df, 3)
    m3 = GM(q_weight=0.8)
    X3, y3 = part3[feat], part3["power_kwh"]
    w3 = weights_g3_sweet(y3.to_numpy(), cap3, X3)
    m3.mean_model.fit(X3, y3, regressor__sample_weight=w3)
    m3.q_model.fit(X3, y3, regressor__sample_weight=w3)
    models[3] = m3
    return models


def predict_wide_mixed(models, infer_df, feat):
    rows = []
    for gid, m in models.items():
        p = infer_df[infer_df.group_id == gid].copy()
        p["pred_kwh"] = m.predict(p[feat])
        rows.append(p)
    long = pd.concat(rows, ignore_index=True)
    return (
        long.pivot(index="forecast_kst_dtm", columns="group_id", values="pred_kwh")
        .rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
        .reset_index()
    )


def run():
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)
    fit_near, fit_idw = train_near.forecast_kst_dtm < vs, train_idw.forecast_kst_dtm < vs
    valid_near, valid_idw = train_near.forecast_kst_dtm >= vs, train_idw.forecast_kst_dtm >= vs
    valid_true = labels.loc[labels.kst_dtm >= vs, ["kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]]

    print("Train v07c...")
    mn = train_with_g3_sweet(train_near.loc[fit_near], feat)
    mi = train_with_g3_sweet(train_idw.loc[fit_idw], feat)
    val = blend_wide(
        predict_wide_mixed(mn, train_near.loc[valid_near], feat),
        predict_wide_mixed(mi, train_idw.loc[valid_idw], feat),
        W_IDW,
    )
    val = apply_v04_calibrations(val, train_idw.loc[valid_idw], WS_MULT)
    slot, scores = tune_slot_multipliers(val, valid_true)
    print(f"Score: {scores['score']:.6f}  FICR: {scores['ficr']:.6f}  slot: {slot}")

    fn = train_with_g3_sweet(train_near, feat)
    fi = train_with_g3_sweet(train_idw, feat)
    test = blend_wide(
        predict_wide_mixed(fn, test_near, feat),
        predict_wide_mixed(fi, test_idw, feat),
        W_IDW,
    )
    test = apply_v04_calibrations(test, test_idw, WS_MULT, slot)
    out = SUBMISSION_DIR / "v07c_g3_sweet_weight.csv"
    SUBMISSION_DIR.mkdir(exist_ok=True)
    load_submission_template().drop(columns=["kpx_group_1", "kpx_group_2", "kpx_group_3"]).merge(
        test, on="forecast_kst_dtm", how="left"
    ).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved: {out}  LB예상: ~{scores['score'] + LB_OFFSET:.3f}")


if __name__ == "__main__":
    run()
