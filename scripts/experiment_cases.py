"""
다중 케이스 실험 — EDA 기반 15개 전략 로컬 검증.

사용법:
  python scripts/experiment_cases.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import (
    WINTER_MONTHS,
    WORST_HOURS,
    apply_slot_multipliers,
    apply_ws_band_multipliers,
    apply_ws_band_multipliers_conditional,
)
from src.config import GROUP_COLUMNS, GROUP_CAPACITY_KWH, OUTPUT_DIR, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.train_v03 import (
    GroupBlendModel,
    blend_wide,
    build_dataset,
    get_feature_columns,
    predict_wide,
    train_all_groups,
    _filter_group_train,
)
from scripts.train_v04 import train_all_groups_qmap

LB_V03, LB_V05B = 0.611, 0.61494  # 실제 LB
COLS = GROUP_COLUMNS


def blend(a, b, alpha):
    out = a.copy()
    for c in COLS:
        out[c] = alpha * a[c] + (1 - alpha) * b[c]
    return out


def sc(pred, truth, name, lb_est=None):
    s = evaluate_submission(truth, pred, time_col="kst_dtm")
    if lb_est is None:
        lb_est = s["score"] - 0.015  # default gap
    return {
        "case": name,
        "local": round(s["score"], 6),
        "nmae": round(s["1_minus_nmae"], 6),
        "ficr": round(s["ficr"], 6),
        "lb_est": round(lb_est, 6),
    }


def lb_blend(alpha_v03, local_score, case_type="blend"):
    """실제 LB 기준 블렌드 추정."""
    if case_type == "blend":
        return alpha_v03 * LB_V03 + (1 - alpha_v03) * LB_V05B
    if case_type == "v03":
        return LB_V03
    if case_type == "v05b":
        return LB_V05B
    if case_type == "mild":
        return local_score - 0.012  # mild 갭 가정
    return local_score - 0.015


def apply_scada_floor(pred, long_df, ratio=0.90, ws_lo=5, ws_hi=12):
    out = pred.copy()
    if "scada_prior_kwh" not in long_df.columns:
        return out
    long = long_df[["forecast_kst_dtm", "group_id", "scada_prior_kwh"]].copy()
    ws = long_df.get("ldaps_ws_hub_blend", long_df.get("ldaps_ws10", 0))
    long["ws"] = ws
    for gid in [1, 2, 3]:
        col = f"kpx_group_{gid}"
        cap = GROUP_CAPACITY_KWH[col]
        m = (long["group_id"] == gid) & (long["ws"] >= ws_lo) & (long["ws"] <= ws_hi)
        for t, prior in long.loc[m, ["forecast_kst_dtm", "scada_prior_kwh"]].itertuples(index=False):
            idx = out["forecast_kst_dtm"] == t
            if not idx.any():
                continue
            floor = prior * ratio
            out.loc[idx, col] = np.clip(max(out.loc[idx, col].iloc[0], floor), 0, cap)
    return out


def apply_scada_blend(pred, long_df, w_scada=0.10, ws_lo=5, ws_hi=12):
    out = pred.copy()
    if "scada_prior_kwh" not in long_df.columns:
        return out
    long = long_df[["forecast_kst_dtm", "group_id", "scada_prior_kwh"]].copy()
    ws = long_df.get("ldaps_ws_hub_blend", long_df.get("ldaps_ws10", 0))
    long["ws"] = ws
    for gid in [1, 2, 3]:
        col = f"kpx_group_{gid}"
        cap = GROUP_CAPACITY_KWH[col]
        m = (long["group_id"] == gid) & (long["ws"] >= ws_lo) & (long["ws"] <= ws_hi)
        for t, prior in long.loc[m, ["forecast_kst_dtm", "scada_prior_kwh"]].itertuples(index=False):
            idx = out["forecast_kst_dtm"] == t
            if not idx.any():
                continue
            raw = out.loc[idx, col].iloc[0]
            out.loc[idx, col] = np.clip((1 - w_scada) * raw + w_scada * prior, 0, cap)
    return out


def train_ldaps_only(train_df, feat_ldaps, q=0.8):
    models = {}
    for gid in [1, 2, 3]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        m = GroupBlendModel(q_weight=q if gid == 3 else 0.6)
        m.fit(part[feat_ldaps], part["power_kwh"], cap)
        models[gid] = m
    return models


def predict_split(models, infer, feat):
    rows = []
    for gid, m in models.items():
        p = infer[infer.group_id == gid].copy()
        cols = [c for c in feat if c in p.columns]
        p["pred_kwh"] = m.predict(p[cols])
        rows.append(p)
    long = pd.concat(rows)
    return (
        long.pivot(index="forecast_kst_dtm", columns="group_id", values="pred_kwh")
        .rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
        .reset_index()
    )


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
    print("[1/3] 데이터 로드 + 베이스 모델 학습")
    print("=" * 60)
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)
    feat_ldaps = get_feature_columns(train_idw, prefixes=["ldaps"])
    fit_n = train_near.forecast_kst_dtm < vs
    fit_i = train_idw.forecast_kst_dtm < vs
    val_n = train_near.forecast_kst_dtm >= vs
    val_i = train_idw.forecast_kst_dtm >= vs
    long_val = train_idw.loc[val_i]

    print("  v03...")
    t3n = train_all_groups(train_near.loc[fit_n], feat, 0.65)
    t3i = train_all_groups(train_idw.loc[fit_i], feat, 0.65)
    v03_val = blend_wide(predict_wide(t3n, train_near.loc[val_n], feat), predict_wide(t3i, long_val, feat), 0.8)
    f3n, f3i = train_all_groups(train_near, feat, 0.65), train_all_groups(train_idw, feat, 0.65)

    print("  v05b...")
    q5 = {1: 0.6, 2: 0.6, 3: 0.8}
    t5n = train_all_groups_qmap(train_near.loc[fit_n], feat, q5)
    t5i = train_all_groups_qmap(train_idw.loc[fit_i], feat, q5)
    raw_val = blend_wide(predict_wide(t5n, train_near.loc[val_n], feat), predict_wide(t5i, long_val, feat), 0.9)
    f5n = train_all_groups_qmap(train_near, feat, q5)
    f5i = train_all_groups_qmap(train_idw, feat, q5)

    print("  LDAPS-only...")
    tl_n = train_ldaps_only(train_near.loc[fit_n], feat_ldaps)
    tl_i = train_ldaps_only(train_idw.loc[fit_i], feat_ldaps)
    ldaps_val = blend_wide(
        predict_split(tl_n, train_near.loc[val_n], feat_ldaps),
        predict_split(tl_i, long_val, feat_ldaps),
        0.9,
    )
    fl_n = train_ldaps_only(train_near, feat_ldaps)
    fl_i = train_ldaps_only(train_idw, feat_ldaps)

    def full_v05b(raw):
        return apply_slot_multipliers(
            apply_ws_band_multipliers(raw, long_val, {1: 1.10, 2: 1.08, 3: 1.0}),
            {1: 1.06, 2: 1.04, 3: 1.06},
        )

    def test_v05b(raw):
        return apply_slot_multipliers(
            apply_ws_band_multipliers(raw, test_idw, {1: 1.10, 2: 1.08, 3: 1.0}),
            {1: 1.06, 2: 1.04, 3: 1.06},
        )

    raw_test = blend_wide(
        predict_wide(f5n, test_near, feat), predict_wide(f5i, test_idw, feat), 0.9
    )
    v03_test = blend_wide(predict_wide(f3n, test_near, feat), predict_wide(f3i, test_idw, feat), 0.8)

    print("\n[2/3] 15개 케이스 검증")
    R = []

    # A. 기준선
    R.append(sc(full_v05b(raw_val), truth, "A_v05b_제출본", LB_V05B))
    R.append(sc(v03_val, truth, "B_v03_단독", LB_V03))

    # B. 블렌드
    for a, name in [(0.15, "15"), (0.25, "25"), (0.35, "35")]:
        p = blend(v03_val, full_v05b(raw_val), a)
        R.append(sc(p, truth, f"C_blend_{name}pct_v03", lb_blend(a, 0, "blend")))

    # C. 완화 보정
    mild_ws = {1: 1.08, 2: 1.06, 3: 1.0}
    mild_slot = {1: 1.04, 2: 1.03, 3: 1.04}
    p_mild = apply_slot_multipliers(apply_ws_band_multipliers(raw_val, long_val, mild_ws), mild_slot)
    r_mild = sc(p_mild, truth, "D_mild_calib", 0.618)
    r_mild["lb_est"] = round(r_mild["local"] - 0.012, 6)
    R.append(r_mild)

    p_mild_full_slot = apply_slot_multipliers(
        apply_ws_band_multipliers(raw_val, long_val, {1: 1.10, 2: 1.08, 3: 1.0}), mild_slot
    )
    R.append(sc(p_mild_full_slot, truth, "E_fullws_mildslot", R[-2]["local"] - 0.01))

    # D. 조건부 보정
    p_cond = apply_ws_band_multipliers_conditional(raw_val, long_val, {1: 1.08, 2: 1.06, 3: 1.0})
    R.append(sc(p_cond, truth, "F_cond_ws_prior", LB_V05B - 0.005))

    # E. SCADA
    R.append(sc(apply_scada_floor(raw_val, long_val, 0.92), truth, "G_scada_floor_92", LB_V05B))
    R.append(sc(apply_scada_blend(full_v05b(raw_val), long_val, 0.08), truth, "H_scada_blend8", LB_V05B))

    # F. 겨울 슬롯
    p_winter = apply_slot_multipliers(
        apply_ws_band_multipliers(raw_val, long_val, mild_ws),
        mild_slot,
        worst_months=WINTER_MONTHS,
        worst_hours=WORST_HOURS,
    )
    R.append(sc(p_winter, truth, "I_winter_mild_slot", LB_V05B - 0.003))

    # G. g2만 보정
    ws_g2 = {1: 1.0, 2: 1.08, 3: 1.0}
    p_g2 = apply_slot_multipliers(apply_ws_band_multipliers(raw_val, long_val, ws_g2), {1: 1.0, 2: 1.04, 3: 1.0})
    R.append(sc(p_g2, truth, "J_g2_only_boost", LB_V05B - 0.002))

    # H. LDAPS-only + mild cal
    p_ldaps = apply_slot_multipliers(apply_ws_band_multipliers(ldaps_val, long_val, mild_ws), mild_slot)
    R.append(sc(p_ldaps, truth, "K_ldaps_mild", LB_V05B - 0.01))

    # I. blend mild
    p_blend_mild = blend(v03_val, p_mild, 0.25)
    R.append(sc(p_blend_mild, truth, "L_blend25_v03+mild", lb_blend(0.25, 0, "blend")))

    # J. sweet spot only boost
    p_sweet = apply_ws_band_multipliers_conditional(
        full_v05b(raw_val), long_val, {1: 1.05, 2: 1.04, 3: 1.0}, prior_ratio=0.95
    )
    R.append(sc(p_sweet, truth, "M_sweet_cond_light", LB_V05B))

    by_local = sorted(R, key=lambda x: x["local"], reverse=True)
    by_lb = sorted(R, key=lambda x: x["lb_est"], reverse=True)

    print("\n" + "=" * 60)
    print("[3/3] 결과")
    print("=" * 60)
    print(f"\n{'케이스':<28} {'로컬':>7} {'NMAE':>7} {'FICR':>7} {'LB추정':>7}")
    print("-" * 60)
    for r in by_local:
        print(f"{r['case']:<28} {r['local']:7.4f} {r['nmae']:7.4f} {r['ficr']:7.4f} {r['lb_est']:7.4f}")

    best_local = by_local[0]
    best_lb = by_lb[0]

    # 최적 케이스 test CSV 생성
    SUBMISSION_DIR.mkdir(exist_ok=True)
    cases_test = {
        "A_v05b": test_v05b(raw_test),
        "D_mild": apply_slot_multipliers(apply_ws_band_multipliers(raw_test, test_idw, mild_ws), mild_slot),
        "C_blend25": blend(v03_test, test_v05b(raw_test), 0.25),
        "E_fullws_mild": apply_slot_multipliers(
            apply_ws_band_multipliers(raw_test, test_idw, {1: 1.10, 2: 1.08, 3: 1.0}), mild_slot
        ),
    }
    for k, p in cases_test.items():
        save_submission(p, f"exp_{k}.csv")

    out = OUTPUT_DIR / "experiment_cases_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"all": R, "best_local": best_local, "best_lb": best_lb, "ranked": by_local}, f, ensure_ascii=False, indent=2)

    print(f"\n★ 로컬 최고: {best_local['case']}  local={best_local['local']:.4f}  FICR={best_local['ficr']:.4f}")
    print(f"★ LB 추정 1위: {best_lb['case']}  lb_est={best_lb['lb_est']:.4f}")
    print(f"\n저장: {out}")
    print(f"제출 CSV: submissions/exp_*.csv")


if __name__ == "__main__":
    main()
