"""v13 v03 blend × g2 hotspot 2×2 factorial — shared logic."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, SUBMISSION_DIR, VALID_START
from src.data_loader import load_submission_template
from src.metrics import (
    FICR_TIER_1,
    FICR_TIER_2,
    GENERATION_FLOOR_RATIO,
    PRICE_TIER_1,
    PRICE_TIER_2,
    evaluate_submission,
)
from scripts.eval_v13_comprehensive import (
    HOTSPOT_G23,
    SPLITS,
    apply_pipeline_fixed,
    build_split_bundle,
    score_cfg,
)
from scripts.submission_diff_lib import G3FEB_KEEP_CONFIG, V13_BEST_CONFIG, merge_submissions

G3_SEASON_FIXED = {2: 1.02, 8: 1.05, 11: 1.02}

FACTORIAL_CONFIGS: dict[str, dict] = {
    "A_v13_best": copy.deepcopy(V13_BEST_CONFIG),
    "B_v03_5_only": {
        "slot_key": "mild23",
        "blend_v03": 0.05,
        "g2_mult": 1.06,
        "g3_season": dict(G3_SEASON_FIXED),
        "g1_mult": 1.04,
        "conditional": False,
    },
    "C_g2_105_only": {
        "slot_key": "mild23",
        "blend_v03": 0.07,
        "g2_mult": 1.05,
        "g3_season": dict(G3_SEASON_FIXED),
        "g1_mult": 1.04,
        "conditional": False,
    },
    "D_both": copy.deepcopy(G3FEB_KEEP_CONFIG),
}

FACTORIAL_PATHS: dict[str, Any] = {
    "A_v13_best": SUBMISSION_DIR / "v13_best.csv",
    "B_v03_5_only": SUBMISSION_DIR / "v13_v0305_g2106_g3feb102.csv",
    "C_g2_105_only": SUBMISSION_DIR / "v13_v0307_g2105_g3feb102.csv",
    "D_both": SUBMISSION_DIR / "v13_hybrid_g105_g3feb102_v0305.csv",
}

PERIOD_DEFS: dict[str, dict[str, Any]] = {
    "2024_full": {
        "bundle_key": "2024",
        "start": pd.Timestamp(VALID_START),
        "end": None,
        "label": "2024 전체 hold-out",
    },
    "2024_h1": {
        "bundle_key": "2024",
        "start": pd.Timestamp("2024-01-01 01:00:00"),
        "end": pd.Timestamp("2024-07-01 01:00:00"),
        "label": "2024 H1 (2024-01-01 01:00 ~ 2024-06-30 23:00)",
    },
    "2024_h2": {
        "bundle_key": "2024",
        "start": pd.Timestamp("2024-07-01 01:00:00"),
        "end": pd.Timestamp("2025-01-01 01:00:00"),
        "label": "2024 H2 (2024-07-01 00:00 ~)",
    },
    "2023_h2": {
        "bundle_key": "2023h2",
        "start": pd.Timestamp("2023-07-01 01:00:00"),
        "end": pd.Timestamp("2024-01-01 01:00:00"),
        "label": "2023 H2 (2023-07-01 01:00 ~ 2023-12-31 23:00)",
    },
}

UTIL_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
UTIL_LABELS = ["0~0.1", "0.1~0.3", "0.3~0.5", "0.5~0.7", "0.7~1.0"]
WS_BINS = [0.0, 3.0, 5.0, 8.0, 12.0, np.inf]
WS_LABELS = ["0~3", "3~5", "5~8", "8~12", "12+"]

G2_HOTSPOT_SLICES = [
    ("g2_m2", 2, {2}, None),
    ("g2_m9", 2, {9}, None),
    ("g2_m11", 2, {11}, None),
    ("g2_m2_h19", 2, {2}, {19}),
    ("g2_m9_hs", 2, {9}, set(HOTSPOT_G23[2][9])),
    ("g2_m11_hs", 2, {11}, set(HOTSPOT_G23[2][11])),
]


def score_period(bundle: dict, config: dict, start: pd.Timestamp, end: pd.Timestamp | None) -> dict:
    pred = apply_pipeline_fixed(bundle["raw"], bundle["v03"], bundle["long_val"], config)
    truth = bundle["truth"]
    mask = truth["kst_dtm"] >= start
    if end is not None:
        mask &= truth["kst_dtm"] < end
    t = truth.loc[mask]
    p = pred.merge(t[["kst_dtm"]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
    p = p.drop(columns=["kst_dtm"])
    return evaluate_submission(t, p, time_col="kst_dtm")


def metric_single_group(actual: np.ndarray, forecast: np.ndarray, capacity: float) -> dict[str, float]:
    valid = actual >= capacity * GENERATION_FLOOR_RATIO
    actual = actual[valid]
    forecast = forecast[valid]
    if len(actual) == 0:
        return {"nmae": float("nan"), "ficr": float("nan"), "proxy_score": float("nan"), "n_rows": 0}

    error_rate = np.abs(forecast - actual) / capacity
    nmae = float(np.mean(error_rate))
    unit_price = np.select(
        [error_rate <= FICR_TIER_1, error_rate <= FICR_TIER_2],
        [PRICE_TIER_1, PRICE_TIER_2],
        default=0.0,
    )
    ficr = float(np.sum(actual * unit_price) / np.sum(actual * PRICE_TIER_1))
    proxy = 0.5 * (1.0 - nmae) + 0.5 * ficr
    return {"nmae": nmae, "ficr": ficr, "proxy_score": proxy, "n_rows": int(len(actual))}


def score_period_per_group(
    bundle: dict,
    config: dict,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> dict[str, dict]:
    pred = apply_pipeline_fixed(bundle["raw"], bundle["v03"], bundle["long_val"], config)
    truth = bundle["truth"]
    mask = truth["kst_dtm"] >= start
    if end is not None:
        mask &= truth["kst_dtm"] < end
    t = truth.loc[mask].copy()
    merged = t.merge(
        pred,
        left_on="kst_dtm",
        right_on="forecast_kst_dtm",
        how="inner",
        suffixes=("_true", "_pred"),
    )
    keep = merged[[f"{c}_true" for c in GROUP_COLUMNS]].notna().all(axis=1)
    merged = merged.loc[keep]

    out: dict[str, dict] = {}
    for col in GROUP_COLUMNS:
        gid = col.split("_")[-1]
        cap = GROUP_CAPACITY_KWH[col]
        true_col = f"{col}_true" if f"{col}_true" in merged.columns else col
        pred_col = f"{col}_pred" if f"{col}_pred" in merged.columns else col
        out[f"g{gid}"] = metric_single_group(
            merged[true_col].to_numpy(dtype=float),
            merged[pred_col].to_numpy(dtype=float),
            cap,
        )
    return out


def _with_deltas_vs_a(period_scores: dict[str, dict], ref: str = "A_v13_best") -> dict[str, dict]:
    base = period_scores[ref]
    enriched: dict[str, dict] = {}
    for name, s in period_scores.items():
        enriched[name] = {
            **s,
            "delta_score_vs_a": s["score"] - base["score"],
            "delta_nmae_vs_a": s["1_minus_nmae"] - base["1_minus_nmae"],
            "delta_ficr_vs_a": s["ficr"] - base["ficr"],
        }
    return enriched


def evaluate_all_periods(bundles: dict, configs: dict[str, dict] | None = None) -> dict:
    configs = configs or FACTORIAL_CONFIGS
    results: dict[str, Any] = {"periods": {}, "period_bounds": {}}

    for pname, pdef in PERIOD_DEFS.items():
        bundle = bundles[pdef["bundle_key"]]
        start, end = pdef["start"], pdef["end"]
        period_scores: dict[str, dict] = {}
        group_scores: dict[str, dict] = {}

        if end is None and pname == "2024_full":
            for cname, cfg in configs.items():
                period_scores[cname] = score_cfg(bundle, cfg)
                group_scores[cname] = score_period_per_group(bundle, cfg, start, None)
        else:
            for cname, cfg in configs.items():
                period_scores[cname] = score_period(bundle, cfg, start, end)
                group_scores[cname] = score_period_per_group(bundle, cfg, start, end)

        results["periods"][pname] = {
            "scores": _with_deltas_vs_a(period_scores),
            "groups": group_scores,
            "definition": pdef["label"],
            "start": str(start),
            "end": str(end) if end else None,
        }
        results["period_bounds"][pname] = {
            "n_rows_a": period_scores["A_v13_best"]["n_rows"],
            "start": str(start),
            "end": str(end) if end else "open",
        }

    return results


def factorial_effects(period_scores: dict[str, dict]) -> dict[str, float]:
    """Score / 1-NMAE / FICR main & interaction effects for one period."""
    A = period_scores["A_v13_best"]
    B = period_scores["B_v03_5_only"]
    C = period_scores["C_g2_105_only"]
    D = period_scores["D_both"]

    def eff(key: str, fn) -> float:
        return float(fn(
            A[key], B[key], C[key], D[key],
        ))

    return {
        "score": {
            "v03_main": eff("score", lambda a, b, c, d: ((b + d) / 2) - ((a + c) / 2)),
            "g2_main": eff("score", lambda a, b, c, d: ((c + d) / 2) - ((a + b) / 2)),
            "interaction": eff("score", lambda a, b, c, d: d - b - c + a),
        },
        "1_minus_nmae": {
            "v03_main": eff("1_minus_nmae", lambda a, b, c, d: ((b + d) / 2) - ((a + c) / 2)),
            "g2_main": eff("1_minus_nmae", lambda a, b, c, d: ((c + d) / 2) - ((a + b) / 2)),
            "interaction": eff("1_minus_nmae", lambda a, b, c, d: d - b - c + a),
        },
        "ficr": {
            "v03_main": eff("ficr", lambda a, b, c, d: ((b + d) / 2) - ((a + c) / 2)),
            "g2_main": eff("ficr", lambda a, b, c, d: ((c + d) / 2) - ((a + b) / 2)),
            "interaction": eff("ficr", lambda a, b, c, d: d - b - c + a),
        },
    }


def compute_factorial_effects_all(eval_results: dict) -> dict:
    out = {}
    for pname, pdata in eval_results["periods"].items():
        out[pname] = factorial_effects(pdata["scores"])
    return out


def _align_truth_pred_long(
    bundle: dict,
    config: dict,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    pred_wide = apply_pipeline_fixed(bundle["raw"], bundle["v03"], bundle["long_val"], config)
    truth = bundle["truth"]
    long_val = bundle["long_val"].copy()
    long_val["forecast_kst_dtm"] = pd.to_datetime(long_val["forecast_kst_dtm"])

    mask = truth["kst_dtm"] >= start
    if end is not None:
        mask &= truth["kst_dtm"] < end
    t = truth.loc[mask, ["kst_dtm", *GROUP_COLUMNS]].copy()
    t = t.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    t = t.melt(
        id_vars=["forecast_kst_dtm"],
        value_vars=GROUP_COLUMNS,
        var_name="group_col",
        value_name="actual",
    )
    t["group_id"] = t["group_col"].str.split("_").str[-1].astype(int)

    p = pred_wide.melt(
        id_vars=["forecast_kst_dtm"],
        value_vars=GROUP_COLUMNS,
        var_name="group_col",
        value_name="prediction",
    )
    p["group_id"] = p["group_col"].str.split("_").str[-1].astype(int)

    merged = t.merge(p, on=["forecast_kst_dtm", "group_col", "group_id"], how="inner")
    merged = merged.merge(
        long_val[["forecast_kst_dtm", "group_id", "ldaps_ws_hub_blend"]],
        on=["forecast_kst_dtm", "group_id"],
        how="left",
    )
    merged["capacity"] = merged["group_col"].map(GROUP_CAPACITY_KWH)
    merged["month"] = merged["forecast_kst_dtm"].dt.month
    merged["hour"] = merged["forecast_kst_dtm"].dt.hour
    merged = merged[merged["actual"].notna()].copy()
    merged["error"] = merged["prediction"] - merged["actual"]
    merged["abs_error"] = merged["error"].abs()
    merged["norm_error"] = merged["abs_error"] / merged["capacity"]
    merged["util_pred"] = merged["prediction"] / merged["capacity"]
    merged["under_pred"] = merged["error"] < 0
    merged["over_pred"] = merged["error"] > 0
    return merged


def slice_metrics(df: pd.DataFrame, ref_df: pd.DataFrame | None = None) -> dict:
    if df.empty:
        return {"row_count": 0}
    cap = df["capacity"].iloc[0]
    valid = df["actual"] >= cap * GENERATION_FLOOR_RATIO
    sub = df.loc[valid]
    if sub.empty:
        return {"row_count": 0}

    err = sub["error"]
    er = sub["abs_error"] / sub["capacity"]
    unit_price = np.select(
        [er <= FICR_TIER_1, er <= FICR_TIER_2],
        [PRICE_TIER_1, PRICE_TIER_2],
        default=0.0,
    )
    ficr = float(np.sum(sub["actual"] * unit_price) / np.sum(sub["actual"] * PRICE_TIER_1))

    out = {
        "row_count": int(len(sub)),
        "mean_prediction": float(sub["prediction"].mean()),
        "mean_actual": float(sub["actual"].mean()),
        "mean_bias": float(err.mean()),
        "mae": float(sub["abs_error"].mean()),
        "normalized_mae": float(er.mean()),
        "under_prediction_ratio": float((err < 0).mean()),
        "ficr": ficr,
    }
    if ref_df is not None and not ref_df.empty:
        ref = slice_metrics(ref_df)
        for k in ("mean_prediction", "mean_bias", "mae", "normalized_mae", "ficr"):
            if k in ref:
                out[f"delta_{k}_vs_a"] = out[k] - ref[k]
    return out


def g2_hotspot_slice_analysis(bundle: dict, configs: dict[str, dict] | None = None) -> dict:
    configs = configs or FACTORIAL_CONFIGS
    start = PERIOD_DEFS["2024_full"]["start"]
    long_by_cfg = {name: _align_truth_pred_long(bundle, cfg, start, None) for name, cfg in configs.items()}
    ref_long = long_by_cfg["A_v13_best"]

    slices_out = []
    for label, gid, months, hours in G2_HOTSPOT_SLICES:
        row = {"label": label}
        for cname, ldf in long_by_cfg.items():
            sub = ldf[ldf["group_id"] == gid]
            sub = sub[sub["month"].isin(months)]
            if hours is not None:
                sub = sub[sub["hour"].isin(hours)]
            ref_sub = ref_long[(ref_long["group_id"] == gid) & (ref_long["month"].isin(months))]
            if hours is not None:
                ref_sub = ref_sub[ref_sub["hour"].isin(hours)]
            row[cname] = slice_metrics(sub, ref_sub if cname != "A_v13_best" else None)
        slices_out.append(row)
    return {"slices": slices_out, "period": "2024_full"}


def v03_blend_effect_analysis(bundle: dict) -> dict:
    """A vs B: v03 7%→5% prediction impact."""
    start = PERIOD_DEFS["2024_full"]["start"]
    a_long = _align_truth_pred_long(bundle, FACTORIAL_CONFIGS["A_v13_best"], start, None)
    b_long = _align_truth_pred_long(bundle, FACTORIAL_CONFIGS["B_v03_5_only"], start, None)
    merged = a_long.merge(
        b_long[["forecast_kst_dtm", "group_id", "prediction"]],
        on=["forecast_kst_dtm", "group_id"],
        suffixes=("_a", "_b"),
    )
    merged["delta_pred"] = merged["prediction_b"] - merged["prediction_a"]
    merged["norm_delta"] = merged["delta_pred"] / merged["capacity"]

    def agg_frame(df: pd.DataFrame, group_cols: list[str]) -> list[dict]:
        rows = []
        if not group_cols:
            rows.append({
                "label": "overall",
                "row_count": len(df),
                "mean_delta_pred": float(df["delta_pred"].mean()),
                "mean_norm_delta": float(df["norm_delta"].mean()),
            })
            return rows
        for keys, g in df.groupby(group_cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rows.append({
                "label": "_".join(str(k) for k in keys),
                **dict(zip(group_cols, keys)),
                "row_count": len(g),
                "mean_delta_pred": float(g["delta_pred"].mean()),
                "mean_norm_delta": float(g["norm_delta"].mean()),
            })
        return rows

    merged["util_bin"] = pd.cut(
        merged["util_pred"],
        bins=UTIL_BINS,
        labels=UTIL_LABELS,
        include_lowest=True,
    )
    merged["ws_bin"] = pd.cut(
        merged["ldaps_ws_hub_blend"],
        bins=WS_BINS,
        labels=WS_LABELS,
        include_lowest=True,
    )
    merged["bias_bin"] = np.where(
        merged["error"] < 0,
        "under_prediction",
        np.where(merged["error"] > 0, "over_prediction", "exact"),
    )

    return {
        "overall": agg_frame(merged, []),
        "by_group": agg_frame(merged, ["group_id"]),
        "by_month": agg_frame(merged, ["month"]),
        "by_util_bin": agg_frame(merged, ["util_bin"]),
        "by_ws_bin": agg_frame(merged, ["ws_bin"]),
        "by_bias_bin": agg_frame(merged, ["bias_bin"]),
    }


def verify_sample_submission(path) -> dict:
    sample = load_submission_template()
    sub = pd.read_csv(path, encoding="utf-8-sig")
    return {
        "sample_rows": len(sample),
        "submission_rows": len(sub),
        "row_match": len(sample) == len(sub),
        "id_match": sample["forecast_id"].equals(sub["forecast_id"]),
        "columns_match": list(sample.columns) == list(sub.columns),
    }


def verify_submission_values(path) -> dict:
    from scripts.submission_diff_lib import load_submission_wide

    wide = load_submission_wide(path)
    long_df = wide.melt(
        id_vars=["forecast_id", "forecast_kst_dtm"],
        value_vars=["kpx_group_1", "kpx_group_2", "kpx_group_3"],
        var_name="group_col",
        value_name="prediction",
    )
    long_df["capacity"] = long_df["group_col"].map(GROUP_CAPACITY_KWH)
    pred = long_df["prediction"]
    cap = long_df["capacity"]
    eps = 1e-6
    return {
        "duplicate_forecast_id": int(wide["forecast_id"].duplicated().sum()),
        "missing_values": int(pred.isna().sum()),
        "negative_predictions": int((pred < -eps).sum()),
        "above_capacity": int((pred > cap + eps).sum()),
        "at_zero": int((pred <= eps).sum()),
        "at_capacity": int((pred >= cap - eps).sum()),
    }


def submissions_identical(path_a, path_b, atol: float = 1e-6) -> dict:
    from scripts.submission_diff_lib import load_submission_wide

    a = load_submission_wide(path_a)
    b = load_submission_wide(path_b)
    merged = a.merge(b, on="forecast_id", suffixes=("_a", "_b"), validate="one_to_one")
    diffs = {}
    for col in GROUP_COLUMNS:
        d = (merged[f"{col}_b"] - merged[f"{col}_a"]).abs()
        diffs[col] = {
            "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "n_differ": int((d > atol).sum()),
        }
    identical = all(v["n_differ"] == 0 for v in diffs.values())
    return {"identical": identical, "by_column": diffs, "n_rows": len(merged)}


def config_diff_only(a: dict, b: dict, allowed_keys: set[str]) -> bool:
    """True if every differing key is in allowed_keys."""
    for k in set(a) | set(b):
        if a.get(k) != b.get(k) and k not in allowed_keys:
            return False
    return True


def g2_only_prediction_diff(path_a, path_b) -> dict:
    from scripts.submission_diff_lib import is_hotspot_row, load_submission_wide

    merged = merge_submissions(load_submission_wide(path_a), load_submission_wide(path_b))
    g2_hs = merged.apply(
        lambda r: int(r["group_id"]) == 2 and is_hotspot_row(2, int(r["month"]), int(r["hour"])),
        axis=1,
    )
    non_g2_hs = ~g2_hs
    max_non = float(merged.loc[non_g2_hs, "delta"].abs().max()) if non_g2_hs.any() else 0.0
    return {
        "g2_hotspot_rows": int(g2_hs.sum()),
        "max_abs_delta_non_g2_hotspot": max_non,
        "non_g2_hotspot_identical": max_non <= 1e-4,
        "g2_hotspot_mean_delta": float(merged.loc[g2_hs, "delta"].mean()) if g2_hs.any() else 0.0,
    }


def recommend_candidate(eval_results: dict) -> dict:
    periods = eval_results["periods"]
    reasons: list[str] = []
    configs = ["B_v03_5_only", "C_g2_105_only", "D_both"]

    score_improve_periods = {c: 0 for c in configs}
    ficr_improve_periods = {c: 0 for c in configs}
    worst_drop = {c: 0.0 for c in configs}

    for pname, pdata in periods.items():
        scores = pdata["scores"]
        for c in configs:
            if scores[c]["delta_score_vs_a"] > 0:
                score_improve_periods[c] += 1
            if scores[c]["delta_ficr_vs_a"] > 0:
                ficr_improve_periods[c] += 1
            worst_drop[c] = min(worst_drop[c], scores[c]["delta_score_vs_a"])

    best = None
    best_score = -999.0
    for c in configs:
        s24 = periods["2024_full"]["scores"][c]
        if score_improve_periods[c] >= 2 and ficr_improve_periods[c] >= 2:
            if worst_drop[c] >= -0.0003 and s24["score"] > best_score:
                best = c
                best_score = s24["score"]

    a_wins_all = all(
        periods[p]["scores"]["A_v13_best"]["score"]
        >= max(periods[p]["scores"][c]["score"] for c in configs)
        for p in periods
    )

    if a_wins_all:
        return {
            "recommendation": "KEEP_V13_BEST",
            "file": str(FACTORIAL_PATHS["A_v13_best"]),
            "reasons": ["A가 모든 기간에서 최고 Score"],
        }

    if best is None:
        b_ok = score_improve_periods["B_v03_5_only"]
        c_ok = score_improve_periods["C_g2_105_only"]
        d_ok = score_improve_periods["D_both"]
        reasons.append(f"Score 개선 기간 수 — B:{b_ok} C:{c_ok} D:{d_ok}")
        reasons.append(f"FICR 개선 기간 수 — B:{ficr_improve_periods['B_v03_5_only']} "
                       f"C:{ficr_improve_periods['C_g2_105_only']} D:{ficr_improve_periods['D_both']}")
        reasons.append(f"최악 Score 하락 — B:{worst_drop['B_v03_5_only']:.6f} "
                       f"C:{worst_drop['C_g2_105_only']:.6f} D:{worst_drop['D_both']:.6f}")
        return {
            "recommendation": "KEEP_V13_BEST",
            "file": str(FACTORIAL_PATHS["A_v13_best"]),
            "reasons": reasons + ["신규 후보가 추천 기준(2+기간 Score·FICR 개선) 미충족"],
        }

    return {
        "recommendation": best,
        "file": str(FACTORIAL_PATHS[best]),
        "reasons": [
            f"{best}: {score_improve_periods[best]}개 기간 Score 개선, "
            f"{ficr_improve_periods[best]}개 기간 FICR 개선",
            f"최악 Score 하락 {worst_drop[best]:.6f}",
        ],
    }


def build_bundles(labels: pd.DataFrame) -> dict:
    return {name: build_split_bundle(labels, vs) for name, vs in SPLITS.items()}


def validate_period_keys(bundles: dict) -> dict:
    """Period masks: no duplicate timestamps, coverage check."""
    report = {}
    for pname, pdef in PERIOD_DEFS.items():
        truth = bundles[pdef["bundle_key"]]["truth"]
        ts = pd.to_datetime(truth["kst_dtm"])
        mask = ts >= pdef["start"]
        if pdef["end"] is not None:
            mask &= ts < pdef["end"]
        selected = ts[mask]
        report[pname] = {
            "n_rows": int(mask.sum()),
            "n_unique_ts": int(selected.nunique()),
            "duplicate_ts": int(selected.duplicated().sum()),
            "min_ts": str(selected.min()) if len(selected) else None,
            "max_ts": str(selected.max()) if len(selected) else None,
        }
    return report
