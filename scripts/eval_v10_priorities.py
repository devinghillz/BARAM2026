"""
우선순위 1~2 일괄 실험 — 후처리 + 학습 개선 로컬 검증.

사용법:
  python scripts/eval_v10_priorities.py
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

from src.calibration import (
    WORST_HOURS,
    apply_slot_multipliers,
    apply_v04_calibrations,
    apply_ws_band_multipliers,
    apply_ws_band_multipliers_conditional,
)
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
    make_lgbm_mean,
    make_lgbm_quantile,
    predict_wide,
    train_all_groups,
)
from scripts.train_v04 import train_all_groups_qmap

COLS = GROUP_COLUMNS
Q_V05 = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW = 0.9
WS_FULL = {1: 1.10, 2: 1.08, 3: 1.0}
WS_MILD = {1: 1.08, 2: 1.06, 3: 1.0}
SLOT_FULL = {1: 1.06, 2: 1.04, 3: 1.06}
SLOT_MILD = {1: 1.04, 2: 1.03, 3: 1.04}
NON_WINTER_MONTHS = {4, 5, 6, 7, 8, 9, 10, 11}
LB_V05B = 0.615


def ficr_boundary_weights(y: np.ndarray, capacity: float) -> np.ndarray:
    """FICR 경계(이용률 20~80%) 구간 가중 강화."""
    util = y / capacity
    w = ficr_sample_weights(y, capacity)
    boundary = (util >= 0.20) & (util <= 0.80)
    w[boundary] += 3.0
    edge = (util >= 0.28) & (util <= 0.72)
    w[edge] += 2.0
    return w


def shrink_mult(m: dict[int, float], factor: float = 0.9) -> dict[int, float]:
    return {g: 1.0 + (v - 1.0) * factor for g, v in m.items()}


def apply_util_tier_boost(
    pred_wide: pd.DataFrame,
    high_mult: dict[int, float] | None = None,
    mid_mult: dict[int, float] | None = None,
    high_lo: float = 0.65,
    mid_lo: float = 0.30,
    mid_hi: float = 0.50,
) -> pd.DataFrame:
    """예측 이용률 구간별 승수 (고발전 상향, 중간 하향)."""
    if high_mult is None:
        high_mult = {1: 1.05, 2: 1.04, 3: 1.03}
    if mid_mult is None:
        mid_mult = {1: 0.98, 2: 0.98, 3: 1.0}
    out = pred_wide.copy()
    for gid in [1, 2, 3]:
        col = f"kpx_group_{gid}"
        cap = GROUP_CAPACITY_KWH[col]
        util = out[col] / cap
        hi = util >= high_lo
        mid = (util >= mid_lo) & (util < mid_hi)
        out.loc[hi, col] = np.clip(out.loc[hi, col] * high_mult[gid], 0, cap)
        out.loc[mid, col] = np.clip(out.loc[mid, col] * mid_mult[gid], 0, cap)
    return out


def apply_calib(raw, long_df, ws, slot, slot_months=None):
    out = apply_ws_band_multipliers(raw, long_df, ws)
    if slot is None:
        return out
    return apply_slot_multipliers(out, slot, worst_months=slot_months)


class GroupBlendModelCustom:
    def __init__(self, q_weight: float, q_alpha: float = 0.58, weight_fn=ficr_sample_weights):
        self.q_weight = q_weight
        self.mean_model = make_lgbm_mean()
        self.q_model = make_lgbm_quantile(alpha=q_alpha)
        self.weight_fn = weight_fn
        self.capacity = 21600.0

    def fit(self, X, y, capacity: float):
        self.capacity = capacity
        w = self.weight_fn(y.to_numpy(), capacity)
        self.mean_model.fit(X, y, regressor__sample_weight=w)
        self.q_model.fit(X, y, regressor__sample_weight=w)
        return self

    def predict(self, X) -> np.ndarray:
        pred = (1 - self.q_weight) * self.mean_model.predict(X) + self.q_weight * self.q_model.predict(X)
        return np.clip(pred, 0, self.capacity)


class GroupResidualModel:
    """pred = scada_prior + residual_LGBM."""

    def __init__(self, q_weight: float = 0.5):
        self.q_weight = q_weight
        self.mean_model = make_lgbm_mean()
        self.q_model = make_lgbm_quantile(alpha=0.58)
        self.capacity = 21600.0

    def fit(self, X, y, prior: pd.Series, capacity: float):
        self.capacity = capacity
        residual = (y - prior).clip(-capacity, capacity)
        w = ficr_sample_weights(y.to_numpy(), capacity)
        self.mean_model.fit(X, residual, regressor__sample_weight=w)
        self.q_model.fit(X, residual, regressor__sample_weight=w)
        return self

    def predict(self, X, prior: np.ndarray) -> np.ndarray:
        r = (1 - self.q_weight) * self.mean_model.predict(X) + self.q_weight * self.q_model.predict(X)
        return np.clip(prior + r, 0, self.capacity)


def train_custom_qmap(train_df, feat, q_weights, q_alpha=0.58, weight_fn=ficr_sample_weights):
    models = {}
    for gid in [1, 2, 3]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        m = GroupBlendModelCustom(q_weights[gid], q_alpha=q_alpha, weight_fn=weight_fn)
        m.fit(part[feat], part["power_kwh"], cap)
        models[gid] = m
    return models


def train_residual_qmap(train_df, feat, q_weights):
    models = {}
    for gid in [1, 2, 3]:
        cap = GROUP_CAPACITY_KWH[f"kpx_group_{gid}"]
        part = _filter_group_train(train_df, gid)
        m = GroupResidualModel(q_weight=q_weights[gid])
        m.fit(part[feat], part["power_kwh"], part["scada_prior_kwh"], cap)
        models[gid] = m
    return models


def predict_residual_wide(models, infer_df, feat):
    rows = []
    for gid, m in models.items():
        p = infer_df[infer_df.group_id == gid].copy()
        prior = p["scada_prior_kwh"].to_numpy(float)
        p["pred_kwh"] = m.predict(p[feat], prior)
        rows.append(p)
    long = pd.concat(rows, ignore_index=True)
    return (
        long.pivot(index="forecast_kst_dtm", columns="group_id", values="pred_kwh")
        .rename(columns={1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
        .reset_index()
    )


def blend_preds(a, b, alpha):
    out = a.copy()
    for c in COLS:
        out[c] = alpha * a[c] + (1 - alpha) * b[c]
    return out


def score(name, pred, truth, tier="") -> dict:
    s = evaluate_submission(truth, pred, time_col="kst_dtm")
    return {
        "case": name,
        "tier": tier,
        "local": round(s["score"], 6),
        "nmae": round(s["1_minus_nmae"], 6),
        "ficr": round(s["ficr"], 6),
    }


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

    print("=" * 64)
    print("[1/4] 베이스 학습 (eval_local_all)")
    print("=" * 64)
    _, preds = build_predictions(labels, scada, vs)
    v03, v05b, raw = preds["v0.3"], preds["v05b"], preds["v05b_raw"]
    long_val = build_dataset(labels, "idw", vs, scada, "train")
    long_val = long_val[long_val.forecast_kst_dtm >= vs]

    R: list[dict] = []

    print("\n[2/4] 우선순위 1 - 후처리 실험")
    # 기준
    R.append(score("P1_00_v05b", v05b, truth, "P1"))
    R.append(score("P1_01_mild", apply_calib(raw, long_val, WS_MILD, SLOT_MILD), truth, "P1"))
    R.append(score("P1_02_fullws_mildslot", apply_calib(raw, long_val, WS_FULL, SLOT_MILD), truth, "P1"))

    # 조건부 ws
    R.append(
        score(
            "P1_03_cond_ws_mild",
            apply_ws_band_multipliers_conditional(raw, long_val, WS_MILD, prior_ratio=0.92),
            truth,
            "P1",
        )
    )
    R.append(
        score(
            "P1_04_cond_ws_full+mildslot",
            apply_calib(
                apply_ws_band_multipliers_conditional(raw, long_val, WS_FULL, prior_ratio=0.92),
                long_val,
                {1: 1.0, 2: 1.0, 3: 1.0},
                SLOT_MILD,
            ),
            truth,
            "P1",
        )
    )

    # ws만 / 겨울 slot 제외
    R.append(score("P1_05_ws_only", apply_ws_band_multipliers(raw, long_val, WS_FULL), truth, "P1"))
    R.append(
        score(
            "P1_06_ws+slot_no_winter",
            apply_calib(raw, long_val, WS_FULL, SLOT_MILD, slot_months=NON_WINTER_MONTHS),
            truth,
            "P1",
        )
    )
    R.append(
        score(
            "P1_07_cond_ws+no_winter_slot",
            apply_calib(
                apply_ws_band_multipliers_conditional(raw, long_val, WS_FULL, prior_ratio=0.92),
                long_val,
                {1: 1.0, 2: 1.0, 3: 1.0},
                SLOT_MILD,
                slot_months=NON_WINTER_MONTHS,
            ),
            truth,
            "P1",
        )
    )

    # 이용률 구간 보정
    base_ws = apply_ws_band_multipliers(raw, long_val, WS_FULL)
    R.append(
        score(
            "P1_08_ws+util_tier",
            apply_util_tier_boost(base_ws),
            truth,
            "P1",
        )
    )
    R.append(
        score(
            "P1_09_ws+util+fullslot",
            apply_slot_multipliers(apply_util_tier_boost(base_ws), SLOT_FULL),
            truth,
            "P1",
        )
    )

    # shrink 보정
    R.append(
        score(
            "P1_10_shrink90",
            apply_calib(raw, long_val, shrink_mult(WS_FULL), shrink_mult(SLOT_FULL)),
            truth,
            "P1",
        )
    )

    # combo
    R.append(
        score(
            "P1_11_cond+no_winter+util_high",
            apply_util_tier_boost(
                apply_calib(
                    apply_ws_band_multipliers_conditional(raw, long_val, WS_MILD, prior_ratio=0.92),
                    long_val,
                    {1: 1.0, 2: 1.0, 3: 1.0},
                    SLOT_MILD,
                    slot_months=NON_WINTER_MONTHS,
                ),
                high_mult={1: 1.04, 2: 1.03, 3: 1.02},
            ),
            truth,
            "P1",
        )
    )

    print("\n[3/4] 우선순위 2 - 학습 개선")
    # 블렌드
    for a, tag in [(0.10, "10"), (0.15, "15")]:
        R.append(score(f"P2_01_blend{tag}pct_v03", blend_preds(v03, v05b, a), truth, "P2"))

    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    feat = get_feature_columns(train_idw)
    fit_n = train_near.forecast_kst_dtm < vs
    fit_i = train_idw.forecast_kst_dtm < vs
    val_n = train_near.forecast_kst_dtm >= vs
    val_i = train_idw.forecast_kst_dtm >= vs

    print("  boundary-weight 학습...")
    bn = train_custom_qmap(train_near.loc[fit_n], feat, Q_V05, weight_fn=ficr_boundary_weights)
    bi = train_custom_qmap(train_idw.loc[fit_i], feat, Q_V05, weight_fn=ficr_boundary_weights)
    raw_bnd = blend_wide(predict_wide(bn, train_near.loc[val_n], feat), predict_wide(bi, long_val, feat), W_IDW)
    R.append(
        score(
            "P2_02_boundary_wt+fullws_mildslot",
            apply_calib(raw_bnd, long_val, WS_FULL, SLOT_MILD),
            truth,
            "P2",
        )
    )
    R.append(
        score(
            "P2_03_boundary_wt+cond_noslot",
            apply_ws_band_multipliers_conditional(raw_bnd, long_val, WS_MILD, prior_ratio=0.92),
            truth,
            "P2",
        )
    )

    print("  quantile alpha=0.60 학습...")
    an = train_custom_qmap(train_near.loc[fit_n], feat, Q_V05, q_alpha=0.60)
    ai = train_custom_qmap(train_idw.loc[fit_i], feat, Q_V05, q_alpha=0.60)
    raw_a60 = blend_wide(predict_wide(an, train_near.loc[val_n], feat), predict_wide(ai, long_val, feat), W_IDW)
    R.append(
        score(
            "P2_04_alpha060+fullws_mildslot",
            apply_calib(raw_a60, long_val, WS_FULL, SLOT_MILD),
            truth,
            "P2",
        )
    )

    print("  SCADA residual 학습...")
    rn = train_residual_qmap(train_near.loc[fit_n], feat, Q_V05)
    ri = train_residual_qmap(train_idw.loc[fit_i], feat, Q_V05)
    raw_res = blend_wide(
        predict_residual_wide(rn, train_near.loc[val_n], feat),
        predict_residual_wide(ri, long_val, feat),
        W_IDW,
    )
    R.append(score("P2_05_residual+mild", apply_calib(raw_res, long_val, WS_MILD, SLOT_MILD), truth, "P2"))
    R.append(
        score(
            "P2_06_residual+cond",
            apply_ws_band_multipliers_conditional(raw_res, long_val, WS_MILD, prior_ratio=0.92),
            truth,
            "P2",
        )
    )

    ranked = sorted(R, key=lambda x: x["local"], reverse=True)
    p1_best = max((r for r in R if r["tier"] == "P1"), key=lambda x: x["local"])
    p2_best = max((r for r in R if r["tier"] == "P2"), key=lambda x: x["local"])

    print("\n" + "=" * 64)
    print("[4/4] 결과")
    print("=" * 64)
    print(f"\n{'케이스':<36} {'Score':>7} {'NMAE':>7} {'FICR':>7}")
    print("-" * 64)
    for r in ranked:
        print(f"{r['case']:<36} {r['local']:7.4f} {r['nmae']:7.4f} {r['ficr']:7.4f}")

    gap = LB_V05B - next(r["local"] for r in R if r["case"] == "P1_00_v05b")
    print(f"\n★ 전체 1위: {ranked[0]['case']}  ({ranked[0]['local']:.4f})")
    print(f"★ P1 최고: {p1_best['case']}  ({p1_best['local']:.4f})")
    print(f"★ P2 최고: {p2_best['case']}  ({p2_best['local']:.4f})")
    print(f"  v05b LB갭: {gap:+.4f}  → 1위 LB추정: {ranked[0]['local'] + gap:.4f}")

    # 상위 3개 test CSV
    print("\n테스트 CSV 생성 (상위 전략)...")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    f5n = train_all_groups_qmap(train_near, feat, Q_V05)
    f5i = train_all_groups_qmap(train_idw, feat, Q_V05)
    raw_test = blend_wide(predict_wide(f5n, test_near, feat), predict_wide(f5i, test_idw, feat), W_IDW)
    f3n = train_all_groups(train_near, feat, 0.65)
    f3i = train_all_groups(train_idw, feat, 0.65)
    v03_test = blend_wide(predict_wide(f3n, test_near, feat), predict_wide(f3i, test_idw, feat), 0.8)
    v05b_test = apply_v04_calibrations(raw_test, test_idw, WS_FULL, slot_mult=SLOT_FULL)

    top_cases = {r["case"] for r in ranked[:3]}
    SUBMISSION_DIR.mkdir(exist_ok=True)
    exports = {
        "P1_00_v05b": v05b_test,
        "P1_02_fullws_mildslot": apply_calib(raw_test, test_idw, WS_FULL, SLOT_MILD),
        "P1_07_cond_ws+no_winter_slot": apply_calib(
            apply_ws_band_multipliers_conditional(raw_test, test_idw, WS_FULL, prior_ratio=0.92),
            test_idw,
            {1: 1.0, 2: 1.0, 3: 1.0},
            SLOT_MILD,
            slot_months=NON_WINTER_MONTHS,
        ),
        "P2_01_blend15pct_v03": blend_preds(v03_test, v05b_test, 0.15),
    }
    for case, pred in exports.items():
        if case in top_cases or case.startswith("P1_07") or case.startswith("P2_01"):
            fname = f"v10_{case}.csv"
            save_sub(pred, fname)
            print(f"  -> submissions/{fname}")

    out = OUTPUT_DIR / "v10_priorities_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ranked": ranked,
                "best_overall": ranked[0],
                "best_p1": p1_best,
                "best_p2": p2_best,
                "gap_v05b": round(gap, 4),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
