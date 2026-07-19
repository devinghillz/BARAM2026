"""
v13 — 1~5순위 통합 탐색 후 최적 제출 생성.

  P1: v12 주변 미세 그리드 (g2/g3 feb·aug·nov)
  P2: g1 핫스팟 추가
  P3: v03 raw 블렌드 (3~7%)
  P4: global slot g2/g3 완화 (핫스팟과 중복 완화)
  P5: 조건부 핫스팟 (prior 미만일 때만)

사용법:
  python scripts/eval_v13_comprehensive.py
"""

from __future__ import annotations

import copy
import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_v04_calibrations
from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, OUTPUT_DIR, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from src.split_utils import valid_split_frames
from scripts.train_v03 import blend_wide, build_dataset, get_feature_columns, predict_wide, train_all_groups
from scripts.train_v04 import train_all_groups_qmap

Q_WEIGHTS = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW_V05B = 0.9
W_IDW_V03 = 0.8
Q_V03 = 0.65
WS_MULT = {1: 1.10, 2: 1.08, 3: 1.0}

SLOT_VARIANTS = {
    "full": {1: 1.06, 2: 1.04, 3: 1.06},
    "mild23": {1: 1.06, 2: 1.02, 3: 1.02},
    "mild23_103": {1: 1.06, 2: 1.03, 3: 1.03},
}

HOTSPOT_G23 = {
    2: {2: {14, 19, 21}, 9: {7}, 11: {17}},
    3: {2: {7, 22}, 8: {5, 14}, 11: {0, 4, 8, 17, 19, 22}},
}

HOTSPOT_G1 = {
    1: {1: {17}, 7: {0}},
}

SPLITS = {
    "2024": pd.Timestamp(VALID_START),
    "2023h2": pd.Timestamp("2023-07-01 01:00:00"),
}

# LB 최고(v12) 기준 seed
LB_SEED = {"g2": 1.05, "g3_season": {2: 1.0, 8: 1.05, 11: 1.02}}


def prior_lookup(long_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.pivot_table(
        index="forecast_kst_dtm", columns="group_id", values="scada_prior_kwh", aggfunc="first"
    )


def _boost_col(
    out: pd.DataFrame,
    col: str,
    cap: float,
    ts: pd.Series,
    slot_mask: pd.Series,
    mult: float,
    conditional: bool,
    prior: pd.Series | None,
    prior_ratio: float,
) -> None:
    if mult == 1.0:
        return
    idx = slot_mask[slot_mask].index
    for i in idx:
        raw = float(out.loc[i, col])
        if conditional and prior is not None:
            t = ts.loc[i]
            if t not in prior.index or pd.isna(prior.loc[t]):
                continue
            if raw >= float(prior.loc[t]) * prior_ratio:
                continue
        out.loc[i, col] = np.clip(raw * mult, 0, cap)


def apply_hotspot_config(
    pred_wide: pd.DataFrame,
    config: dict,
    long_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """g2/g3/g1 핫스팟 + g3 월별 승수 + 조건부 prior."""
    out = pred_wide.copy()
    ts = pd.to_datetime(out["forecast_kst_dtm"])
    months, hours = ts.dt.month, ts.dt.hour
    conditional = config.get("conditional", False)
    prior_ratio = config.get("prior_ratio", 0.92)
    priors = prior_lookup(long_df) if long_df is not None and conditional else None

    g2_mult = config.get("g2_mult", 1.0)
    if g2_mult != 1.0 and 2 in HOTSPOT_G23:
        col = "kpx_group_2"
        cap = GROUP_CAPACITY_KWH[col]
        p2 = priors[2] if priors is not None and 2 in priors.columns else None
        mask = pd.Series(False, index=out.index)
        for mo, hrs in HOTSPOT_G23[2].items():
            mask |= (months == mo) & hours.isin(hrs)
        _boost_col(out, col, cap, ts, mask, g2_mult, conditional, p2, prior_ratio)

    g3_season = config.get("g3_season", {})
    if 3 in HOTSPOT_G23:
        col = "kpx_group_3"
        cap = GROUP_CAPACITY_KWH[col]
        p3 = priors[3] if priors is not None and 3 in priors.columns else None
        for mo, mult in g3_season.items():
            if mult == 1.0 or mo not in HOTSPOT_G23[3]:
                continue
            hrs = HOTSPOT_G23[3][mo]
            mask = (months == mo) & hours.isin(hrs)
            _boost_col(out, col, cap, ts, mask, mult, conditional, p3, prior_ratio)

    g1_mult = config.get("g1_mult", 1.0)
    if g1_mult != 1.0:
        col = "kpx_group_1"
        cap = GROUP_CAPACITY_KWH[col]
        p1 = priors[1] if priors is not None and 1 in priors.columns else None
        mask = pd.Series(False, index=out.index)
        for mo, hrs in HOTSPOT_G1[1].items():
            mask |= (months == mo) & hours.isin(hrs)
        _boost_col(out, col, cap, ts, mask, g1_mult, conditional, p1, prior_ratio)

    return out


def apply_pipeline(
    raw: pd.DataFrame,
    v03: pd.DataFrame,
    long_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    slot = SLOT_VARIANTS[config["slot_key"]]
    out = apply_v04_calibrations(raw, long_df, WS_MULT, slot)
    out = apply_hotspot_config(out, config, long_df)
    w = config.get("blend_v03", 0.0)
    if w > 0:
        out = blend_wide(out, v03, 1.0 - w)  # blend_wide: (1-w_idw)*a + w_idw*b → w on v03 side
        # blend_wide(a,b,w): (1-w)*a + w*b — we want (1-w)*hotspot + w*v03
        out = out.copy()
        for c in GROUP_COLUMNS:
            out[c] = (1 - w) * apply_hotspot_config(
                apply_v04_calibrations(raw, long_df, WS_MULT, slot), config, long_df
            )[c].values + w * v03[c].values
    return out


def blend_v03_fixed(v05b_part: pd.DataFrame, v03: pd.DataFrame, w: float) -> pd.DataFrame:
    out = v05b_part.copy()
    for c in GROUP_COLUMNS:
        out[c] = (1 - w) * v05b_part[c].to_numpy(float) + w * v03[c].to_numpy(float)
    return out


def apply_pipeline_fixed(raw, v03, long_df, config):
    slot = SLOT_VARIANTS[config["slot_key"]]
    cal = apply_v04_calibrations(raw, long_df, WS_MULT, slot)
    hot = apply_hotspot_config(cal, config, long_df)
    w = config.get("blend_v03", 0.0)
    if w <= 0:
        return hot
    return blend_v03_fixed(hot, v03, w)


def build_split_bundle(labels, vs):
    scada = build_scada_monthly_curve()
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    feat = get_feature_columns(train_idw)
    fit_near, fit_idw, near_val, idw_val = valid_split_frames(train_near, train_idw, vs)

    m5n = train_all_groups_qmap(fit_near, feat, Q_WEIGHTS)
    m5i = train_all_groups_qmap(fit_idw, feat, Q_WEIGHTS)
    raw = blend_wide(predict_wide(m5n, near_val, feat), predict_wide(m5i, idw_val, feat), W_IDW_V05B)

    m3n = train_all_groups(fit_near, feat, Q_V03)
    m3i = train_all_groups(fit_idw, feat, Q_V03)
    v03 = blend_wide(predict_wide(m3n, near_val, feat), predict_wide(m3i, idw_val, feat), W_IDW_V03)

    truth = labels.loc[labels["kst_dtm"] >= vs, ["kst_dtm", *GROUP_COLUMNS]]
    return {"raw": raw, "v03": v03, "long_val": idw_val, "truth": truth}


def score_cfg(bundle, config) -> dict:
    pred = apply_pipeline_fixed(bundle["raw"], bundle["v03"], bundle["long_val"], config)
    return evaluate_submission(bundle["truth"], pred, time_col="kst_dtm")


def config_label(cfg: dict) -> str:
    parts = [cfg["slot_key"][:4]]
    parts.append(f"g2{int(cfg['g2_mult']*100):02d}")
    for m in sorted(cfg["g3_season"]):
        parts.append(f"m{m}{int(cfg['g3_season'][m]*100):02d}")
    if cfg.get("g1_mult", 1.0) != 1.0:
        parts.append(f"g1{int(cfg['g1_mult']*100):02d}")
    if cfg.get("conditional"):
        parts.append("cond")
    if cfg.get("blend_v03", 0) > 0:
        parts.append(f"v03{int(cfg['blend_v03']*100):02d}")
    return "_".join(parts)


def main() -> None:
    labels = load_labels()
    bundles = {name: build_split_bundle(labels, vs) for name, vs in SPLITS.items()}
    base24 = score_cfg(bundles["2024"], {
        "slot_key": "full", "g2_mult": 1.0, "g3_season": {2: 1.0, 8: 1.0, 11: 1.0},
        "g1_mult": 1.0, "conditional": False, "blend_v03": 0.0,
    })
    base23 = score_cfg(bundles["2023h2"], {
        "slot_key": "full", "g2_mult": 1.0, "g3_season": {2: 1.0, 8: 1.0, 11: 1.0},
        "g1_mult": 1.0, "conditional": False, "blend_v03": 0.0,
    })
    print("=== v13 comprehensive (P1~P5) ===\n")
    print(f"  base 2024: {base24['score']:.6f}  2023h2: {base23['score']:.6f}")
    print(f"  LB seed replay target: g2=1.05 g3={{2:1.0,8:1.05,11:1.02}}\n")

    results = []

    # --- P4: slot variant (LB seed hotspot) ---
    print("[P4] global slot variants...")
    best_slot = "full"
    best_slot_score = -1.0
    seed_hs = {
        "g2_mult": LB_SEED["g2"],
        "g3_season": dict(LB_SEED["g3_season"]),
        "g1_mult": 1.0,
        "conditional": False,
        "blend_v03": 0.0,
    }
    for slot_key in SLOT_VARIANTS:
        cfg = {"slot_key": slot_key, **seed_hs}
        s24 = score_cfg(bundles["2024"], cfg)
        s23 = score_cfg(bundles["2023h2"], cfg)
        print(f"  {slot_key}: 2024={s24['score']:.6f}  2023h2={s23['score']:.6f}")
        results.append({"phase": "P4", "cfg": cfg, "2024": s24, "2023h2": s23})
        if s24["score"] > best_slot_score:
            best_slot_score = s24["score"]
            best_slot = slot_key

    # --- P1: fine grid around seed ---
    print(f"\n[P1] fine grid (slot={best_slot})...")
    g2_grid = [1.04, 1.05, 1.06]
    g3_feb = [1.0, 1.02, 1.03]
    g3_aug = [1.04, 1.05, 1.06]
    g3_nov = [1.01, 1.02, 1.03, 1.04]
    p1_best = None
    p1_best_score = -1.0
    for g2, gf, ga, gn in product(g2_grid, g3_feb, g3_aug, g3_nov):
        cfg = {
            "slot_key": best_slot,
            "g2_mult": g2,
            "g3_season": {2: gf, 8: ga, 11: gn},
            "g1_mult": 1.0,
            "conditional": False,
            "blend_v03": 0.0,
        }
        s24 = score_cfg(bundles["2024"], cfg)
        s23 = score_cfg(bundles["2023h2"], cfg)
        results.append({"phase": "P1", "cfg": cfg, "2024": s24, "2023h2": s23})
        if s24["score"] > p1_best_score:
            p1_best_score = s24["score"]
            p1_best = copy.deepcopy(cfg)

    s23_p1 = score_cfg(bundles["2023h2"], p1_best)
    print(f"  best P1: g2={p1_best['g2_mult']} g3={p1_best['g3_season']}")
    print(f"    2024={p1_best_score:.6f}  2023h2={s23_p1['score']:.6f}")

    # --- P2: g1 hotspot ---
    print("\n[P2] g1 hotspot...")
    p2_best = copy.deepcopy(p1_best)
    p2_best_score = p1_best_score
    for g1m in [1.0, 1.03, 1.04]:
        cfg = copy.deepcopy(p1_best)
        cfg["g1_mult"] = g1m
        s24 = score_cfg(bundles["2024"], cfg)
        s23 = score_cfg(bundles["2023h2"], cfg)
        results.append({"phase": "P2", "cfg": cfg, "2024": s24, "2023h2": s23})
        if s24["score"] > p2_best_score:
            p2_best_score = s24["score"]
            p2_best = copy.deepcopy(cfg)
    print(f"  best P2: g1={p2_best['g1_mult']}  2024={p2_best_score:.6f}")

    # --- P5: conditional ---
    print("\n[P5] conditional hotspot...")
    p5_best = copy.deepcopy(p2_best)
    p5_best_score = p2_best_score
    for cond, ratio in [(False, 0.92), (True, 0.92), (True, 0.90), (True, 0.95)]:
        cfg = copy.deepcopy(p2_best)
        cfg["conditional"] = cond
        cfg["prior_ratio"] = ratio
        s24 = score_cfg(bundles["2024"], cfg)
        s23 = score_cfg(bundles["2023h2"], cfg)
        tag = "off" if not cond else f"r{int(ratio*100)}"
        print(f"  cond={tag}: 2024={s24['score']:.6f}  2023h2={s23['score']:.6f}")
        results.append({"phase": "P5", "cfg": cfg, "2024": s24, "2023h2": s23})
        if s24["score"] > p5_best_score:
            p5_best_score = s24["score"]
            p5_best = copy.deepcopy(cfg)

    # --- P3: v03 blend ---
    print("\n[P3] v03 blend...")
    p3_best = copy.deepcopy(p5_best)
    p3_best_score = p5_best_score
    for w in [0.0, 0.03, 0.05, 0.07]:
        cfg = copy.deepcopy(p5_best)
        cfg["blend_v03"] = w
        s24 = score_cfg(bundles["2024"], cfg)
        s23 = score_cfg(bundles["2023h2"], cfg)
        print(f"  v03 {w:.0%}: 2024={s24['score']:.6f}  2023h2={s23['score']:.6f}")
        results.append({"phase": "P3", "cfg": cfg, "2024": s24, "2023h2": s23})
        if s24["score"] > p3_best_score:
            p3_best_score = s24["score"]
            p3_best = copy.deepcopy(cfg)

    # --- 최종 선정: 2023h2 안정 + 2024 최고 ---
    base23_score = base23["score"]
    finalists = [r for r in results if r["2023h2"]["score"] >= base23_score - 0.00015]
    if not finalists:
        finalists = results
    winner = max(finalists, key=lambda r: (r["2024"]["score"], r["2023h2"]["score"]))
    best_cfg = winner["cfg"]

    # greedy chain best도 비교
    chain_score24 = p3_best_score
    chain_s23 = score_cfg(bundles["2023h2"], p3_best)
    if (chain_score24, chain_s23["score"]) > (winner["2024"]["score"], winner["2023h2"]["score"]):
        best_cfg = p3_best
        winner = {"2024": score_cfg(bundles["2024"], p3_best), "2023h2": chain_s23, "phase": "chain"}

    print("\n=== BEST ===")
    print(f"  config: {best_cfg}")
    s24f = score_cfg(bundles["2024"], best_cfg)
    s23f = score_cfg(bundles["2023h2"], best_cfg)
    print(f"  2024: {s24f['score']:.6f} ({s24f['score']-base24['score']:+.6f})")
    print(f"        NMAE={s24f['1_minus_nmae']:.4f}  FICR={s24f['ficr']:.4f}")
    print(f"  2023h2: {s23f['score']:.6f} ({s23f['score']-base23['score']:+.6f})")

    # --- 제출 (full train) ---
    print("\n[Submit] full train + test...")
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)

    m5n = train_all_groups_qmap(train_near, feat, Q_WEIGHTS)
    m5i = train_all_groups_qmap(train_idw, feat, Q_WEIGHTS)
    raw_test = blend_wide(
        predict_wide(m5n, test_near, feat), predict_wide(m5i, test_idw, feat), W_IDW_V05B
    )
    m3n = train_all_groups(train_near, feat, Q_V03)
    m3i = train_all_groups(train_idw, feat, Q_V03)
    v03_test = blend_wide(
        predict_wide(m3n, test_near, feat), predict_wide(m3i, test_idw, feat), W_IDW_V03
    )

    test_out = apply_pipeline_fixed(raw_test, v03_test, test_idw, best_cfg)
    fname = f"v13_{config_label(best_cfg)}.csv"
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSION_DIR / fname
    sub = load_submission_template().drop(columns=GROUP_COLUMNS)
    sub.merge(test_out, on="forecast_kst_dtm", how="left").to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    print(f"  Saved: {out_path}")

    # LB seed 비교용도 저장
    seed_cfg = {"slot_key": best_slot, **seed_hs}
    seed_out = apply_pipeline_fixed(raw_test, v03_test, test_idw, seed_cfg)
    seed_path = SUBMISSION_DIR / "v13_lb_seed_replay.csv"
    sub.merge(seed_out, on="forecast_kst_dtm", how="left").to_csv(
        seed_path, index=False, encoding="utf-8-sig"
    )

    report = {
        "base": {"2024": base24, "2023h2": base23},
        "best_config": best_cfg,
        "best_scores": {"2024": s24f, "2023h2": s23f},
        "submission": str(out_path),
        "top10": sorted(
            [
                {
                    "phase": r["phase"],
                    "score_2024": r["2024"]["score"],
                    "score_2023h2": r["2023h2"]["score"],
                    "ficr": r["2024"]["ficr"],
                    "label": config_label(r["cfg"]),
                }
                for r in results
            ],
            key=lambda x: x["score_2024"],
            reverse=True,
        )[:10],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = OUTPUT_DIR / "v13_comprehensive_report.json"
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  Report: {rep_path}")


if __name__ == "__main__":
    main()
