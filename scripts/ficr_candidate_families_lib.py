"""FICR candidate family validation — testable core logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import PRICE_TIER_1, PRICE_TIER_2, evaluate_submission
from scripts.ficr_boundary_lib import (
    GROUP_ID_TO_COL,
    SIM_MULTIPLIERS,
    baseline_metrics_by_period,
    build_all_period_frames,
    compute_row_ficr_fields,
    count_ficr_transitions,
    long_to_wide_pred,
    long_to_wide_truth,
)
from scripts.submission_diff_lib import V13_BEST_CONFIG

PERIODS = ["2024_full", "2024_h1", "2024_h2", "2023_h2"]

LOPO_FOLDS = [
    {"fold_id": 1, "select_period": "2023_h2", "eval_period": "2024_h2"},
    {"fold_id": 2, "select_period": "2024_h2", "eval_period": "2023_h2"},
    {"fold_id": 3, "select_period": "2024_h1", "eval_period": "2024_h2"},
]

MONTH_LOO_FOLDS = [
    {"fold_id": "dec_23sel_24eval", "select_year": 2023, "eval_year": 2024},
    {"fold_id": "dec_24sel_23eval", "select_year": 2024, "eval_year": 2023},
]


@dataclass
class CandidateSpec:
    candidate_id: str
    family: str
    group_id: int | None = None
    month: int | None = None
    hours: set[int] = field(default_factory=set)
    or_specs: list[dict] | None = None
    overfit_risk: bool = False
    exploratory: bool = False
    component_ids: list[str] = field(default_factory=list)


def build_candidate_specs() -> dict[str, CandidateSpec]:
    specs: dict[str, CandidateSpec] = {
        "A1": CandidateSpec("A1", "A", group_id=2, month=12, hours={9, 10}),
        "A2": CandidateSpec("A2", "A", group_id=2, month=12, hours={5, 9, 10}),
        "A3": CandidateSpec("A3", "A", group_id=2, month=12, hours={8, 9, 10}),
        "A4": CandidateSpec("A4", "A", group_id=2, month=12, hours={5, 8, 9, 10}),
        "B1": CandidateSpec(
            "B1", "B", group_id=2, month=12, hours={9, 10, 14, 15}, overfit_risk=True
        ),
        "B2": CandidateSpec(
            "B2", "B", group_id=2, month=12, hours={5, 8, 9, 10, 14, 15}, overfit_risk=True
        ),
        "C1": CandidateSpec("C1", "C", group_id=1, month=3, hours={4}),
        "C2": CandidateSpec("C2", "C", group_id=1, month=7, hours={14}),
        "C3": CandidateSpec(
            "C3",
            "C",
            exploratory=True,
            or_specs=[
                {"group_id": 1, "month": 3, "hours": {4}},
                {"group_id": 1, "month": 7, "hours": {14}},
            ],
        ),
    }
    specs["D1"] = CandidateSpec("D1", "D", component_ids=["A1", "C1"])
    specs["D2"] = CandidateSpec("D2", "D", component_ids=["A2", "C1"])
    return specs


def _single_slice_mask(df: pd.DataFrame, group_id: int, month: int, hours: set[int]) -> pd.Series:
    return (
        df["active"]
        & (df["group_id"] == group_id)
        & (df["month"] == month)
        & (df["hour"].isin(hours))
    )


def candidate_mask(df: pd.DataFrame, spec: CandidateSpec) -> pd.Series:
    if spec.or_specs:
        mask = pd.Series(False, index=df.index)
        for sl in spec.or_specs:
            mask |= _single_slice_mask(df, sl["group_id"], sl["month"], sl["hours"])
        return mask
    if spec.component_ids:
        return pd.Series(False, index=df.index)
    assert spec.group_id is not None and spec.month is not None
    return _single_slice_mask(df, spec.group_id, spec.month, spec.hours)


def resolve_mask(
    df: pd.DataFrame, spec: CandidateSpec, all_specs: dict[str, CandidateSpec]
) -> pd.Series:
    if spec.component_ids:
        mask = pd.Series(False, index=df.index)
        for cid in spec.component_ids:
            mask |= candidate_mask(df, all_specs[cid])
        return mask
    return candidate_mask(df, spec)


def price_to_tier(price: float) -> int:
    if price >= PRICE_TIER_1 - 1e-9:
        return 1
    if price >= PRICE_TIER_2 - 1e-9:
        return 2
    return 0


def count_tier_transitions(sub: pd.DataFrame, new_pred: pd.Series) -> dict[str, int]:
    old_tiers = sub["unit_price"].apply(price_to_tier).to_numpy()
    new_tiers = np.array([
        price_to_tier(
            compute_row_ficr_fields(float(a), float(p), float(c))["unit_price"]
        )
        for a, p, c in zip(sub["actual"], new_pred, sub["capacity"])
    ])
    return {
        "tier0_to_tier2": int(((old_tiers == 0) & (new_tiers == 2)).sum()),
        "tier0_to_tier1": int(((old_tiers == 0) & (new_tiers == 1)).sum()),
        "tier2_to_tier1": int(((old_tiers == 2) & (new_tiers == 1)).sum()),
        "tier1_to_tier2": int(((old_tiers == 1) & (new_tiers == 2)).sum()),
    }


def apply_multiplier_to_wide(
    df: pd.DataFrame,
    pred_wide: pd.DataFrame,
    row_mask: pd.Series,
    multiplier: float,
    util_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_copy = pred_wide.copy()
    sub = df.loc[row_mask].copy()
    if util_threshold is not None:
        sub = sub[sub["util_pred"] < util_threshold]
    if sub.empty:
        return pred_copy, sub

    sub = sub.copy()
    sub["new_prediction"] = sub["prediction"] * multiplier
    for gid, col in GROUP_ID_TO_COL.items():
        part = sub[sub["group_id"] == gid]
        if part.empty:
            continue
        lookup = part.set_index("forecast_kst_dtm")["new_prediction"]
        ts_mask = pred_copy["forecast_kst_dtm"].isin(lookup.index)
        pred_copy.loc[ts_mask, col] = pred_copy.loc[ts_mask, "forecast_kst_dtm"].map(lookup)
    return pred_copy, sub


def simulate_on_period(
    df: pd.DataFrame,
    row_mask: pd.Series,
    multiplier: float,
    baseline: dict,
    apply_mode: str = "uniform",
    util_threshold: float | None = None,
) -> dict[str, Any]:
    truth = long_to_wide_truth(df)
    pred_wide = long_to_wide_pred(df)
    util_th = util_threshold if apply_mode.startswith("util_lt_") else None
    pred_adj, sub = apply_multiplier_to_wide(df, pred_wide, row_mask, multiplier, util_th)
    if sub.empty:
        return {"valid": False, "reason": "empty_mask"}

    metrics = evaluate_submission(truth, pred_adj, time_col="kst_dtm")
    new_pred = sub["prediction"] * multiplier
    old_under = float((sub["signed_error"] < 0).mean())
    old_over = float((sub["signed_error"] > 0).mean())
    signed_new = new_pred.to_numpy() - sub["actual"].to_numpy()
    new_under = float((signed_new < 0).mean())
    new_over = float((signed_new > 0).mean())

    tier_t = count_tier_transitions(sub, new_pred)
    trans = count_ficr_transitions(sub, new_pred)
    fail_to_success = trans["fail_to_success"]
    success_to_fail = trans["success_to_fail"]
    net_transition = fail_to_success - success_to_fail

    return {
        "valid": True,
        "score": metrics["score"],
        "1_minus_nmae": metrics["1_minus_nmae"],
        "ficr": metrics["ficr"],
        "delta_score": metrics["score"] - baseline["score"],
        "delta_nmae": metrics["1_minus_nmae"] - baseline["1_minus_nmae"],
        "delta_ficr": metrics["ficr"] - baseline["ficr"],
        "changed_rows": int(len(sub)),
        "fail_to_success": fail_to_success,
        "success_to_fail": success_to_fail,
        "net_transition": net_transition,
        "mean_prediction_delta": float((new_pred - sub["prediction"]).mean()),
        "mean_norm_prediction_delta": float(((new_pred - sub["prediction"]) / sub["capacity"]).mean()),
        "under_ratio_before": old_under,
        "under_ratio_after": new_under,
        "under_ratio_change": new_under - old_under,
        "over_ratio_before": old_over,
        "over_ratio_after": new_over,
        "over_ratio_change": new_over - old_over,
        **tier_t,
    }


def apply_modes_for_spec(spec: CandidateSpec) -> list[tuple[str, float | None]]:
    modes = [("uniform", None)]
    if spec.family in {"A", "B"}:
        modes.append(("util_lt_0.3", 0.3))
        modes.append(("util_lt_0.5", 0.5))
    return modes


def evaluate_all_candidates(
    frames: dict[str, pd.DataFrame],
    baselines: dict[str, dict],
    specs: dict[str, CandidateSpec] | None = None,
) -> list[dict]:
    specs = specs or build_candidate_specs()
    rows: list[dict] = []
    for cid, spec in specs.items():
        for period, df in frames.items():
            mask = resolve_mask(df, spec, specs)
            for mult in SIM_MULTIPLIERS:
                for apply_mode, util_th in apply_modes_for_spec(spec):
                    r = simulate_on_period(
                        df, mask, mult, baselines[period], apply_mode, util_th
                    )
                    if not r.get("valid"):
                        continue
                    rows.append({
                        "candidate_id": cid,
                        "family": spec.family,
                        "period": period,
                        "multiplier": mult,
                        "apply_mode": apply_mode,
                        "util_threshold": util_th,
                        "mask_rows_total": int(mask.sum()),
                        "overfit_risk": spec.overfit_risk,
                        "exploratory": spec.exploratory,
                        **r,
                    })
    return rows


def _best_multiplier_on_period(
    results: list[dict],
    candidate_id: str,
    period: str,
    apply_mode: str = "uniform",
) -> dict | None:
    subset = [
        r
        for r in results
        if r["candidate_id"] == candidate_id
        and r["period"] == period
        and r["apply_mode"] == apply_mode
    ]
    if not subset:
        return None
    return max(subset, key=lambda x: (x["score"], x["ficr"]))


def leave_one_period_out(
    all_results: list[dict],
    specs: dict[str, CandidateSpec],
) -> list[dict]:
    rows = []
    for fold in LOPO_FOLDS:
        sel_p, ev_p = fold["select_period"], fold["eval_period"]
        for cid in specs:
            for apply_mode in ["uniform", "util_lt_0.3", "util_lt_0.5"]:
                if apply_mode != "uniform" and specs[cid].family not in {"A", "B"}:
                    continue
                best_sel = _best_multiplier_on_period(all_results, cid, sel_p, apply_mode)
                if best_sel is None:
                    continue
                mult = best_sel["multiplier"]
                eval_row = next(
                    (
                        r
                        for r in all_results
                        if r["candidate_id"] == cid
                        and r["period"] == ev_p
                        and r["multiplier"] == mult
                        and r["apply_mode"] == apply_mode
                    ),
                    None,
                )
                if eval_row is None:
                    continue
                rows.append({
                    "fold_id": fold["fold_id"],
                    "candidate_id": cid,
                    "family": specs[cid].family,
                    "apply_mode": apply_mode,
                    "select_period": sel_p,
                    "eval_period": ev_p,
                    "selected_multiplier": mult,
                    "select_score": best_sel["score"],
                    "select_delta_score": best_sel["delta_score"],
                    "eval_delta_score": eval_row["delta_score"],
                    "eval_delta_ficr": eval_row["delta_ficr"],
                    "eval_net_transition": eval_row["net_transition"],
                    "eval_success_to_fail": eval_row["success_to_fail"],
                    "eval_fail_to_success": eval_row["fail_to_success"],
                })
    return rows


def _december_timestamps(df: pd.DataFrame, year: int) -> np.ndarray:
    return df.loc[
        (df["forecast_kst_dtm"].dt.year == year) & (df["month"] == 12),
        "forecast_kst_dtm",
    ].unique()


def evaluate_december_year(
    df: pd.DataFrame,
    row_mask: pd.Series,
    multiplier: float,
    year: int,
    util_threshold: float | None = None,
) -> dict[str, Any]:
    dec_ts = _december_timestamps(df, year)
    if len(dec_ts) == 0:
        return {"valid": False, "reason": "no_december_data"}
    truth = long_to_wide_truth(df)
    pred_wide = long_to_wide_pred(df)
    pred_base = pred_wide[pred_wide["forecast_kst_dtm"].isin(dec_ts)].copy()
    truth_sub = truth[truth["kst_dtm"].isin(dec_ts)].copy()
    if truth_sub.empty:
        return {"valid": False, "reason": "empty_truth"}
    base_m = evaluate_submission(truth_sub, pred_base, time_col="kst_dtm")
    pred_adj, sub = apply_multiplier_to_wide(df, pred_wide, row_mask, multiplier, util_threshold)
    pred_sub = pred_adj[pred_adj["forecast_kst_dtm"].isin(dec_ts)].copy()
    metrics = evaluate_submission(truth_sub, pred_sub, time_col="kst_dtm")
    return {
        "valid": True,
        "year": year,
        "score": metrics["score"],
        "ficr": metrics["ficr"],
        "delta_score": metrics["score"] - base_m["score"],
        "delta_ficr": metrics["ficr"] - base_m["ficr"],
        "changed_rows_dec": int(sub[sub["forecast_kst_dtm"].isin(dec_ts)].shape[0]),
    }


def month_level_december_loo(
    frames: dict[str, pd.DataFrame],
    specs: dict[str, CandidateSpec],
) -> list[dict]:
    rows = []
    period_for_year = {2023: "2023_h2", 2024: "2024_h2"}
    for fold in MONTH_LOO_FOLDS:
        sel_y, ev_y = fold["select_year"], fold["eval_year"]
        sel_p = period_for_year.get(sel_y)
        ev_p = period_for_year.get(ev_y)
        if sel_p not in frames or ev_p not in frames:
            continue
        sel_df = frames[sel_p]
        ev_df = frames[ev_p]
        for cid, spec in specs.items():
            if spec.family not in {"A", "B"}:
                continue
            sel_mask = resolve_mask(sel_df, spec, specs)
            ev_mask = resolve_mask(ev_df, spec, specs)
            if _december_timestamps(sel_df, sel_y).size == 0 or _december_timestamps(ev_df, ev_y).size == 0:
                continue
            best_score = -999.0
            best_mult = None
            best_sel_delta = None
            for mult in SIM_MULTIPLIERS:
                sel_r = evaluate_december_year(sel_df, sel_mask, mult, sel_y)
                if not sel_r.get("valid"):
                    continue
                if sel_r["score"] > best_score:
                    best_score = sel_r["score"]
                    best_mult = mult
                    best_sel_delta = sel_r["delta_score"]
            if best_mult is None:
                continue
            ev_r = evaluate_december_year(ev_df, ev_mask, best_mult, ev_y)
            if not ev_r.get("valid"):
                continue
            rows.append({
                "fold_id": fold["fold_id"],
                "candidate_id": cid,
                "select_period": sel_p,
                "eval_period": ev_p,
                "select_year": sel_y,
                "eval_year": ev_y,
                "selected_multiplier": best_mult,
                "select_delta_score_dec": best_sel_delta,
                "eval_delta_score_dec": ev_r["delta_score"],
                "eval_delta_ficr_dec": ev_r["delta_ficr"],
            })
    return rows


def precompute_daily_deltas(
    df: pd.DataFrame,
    spec: CandidateSpec,
    all_specs: dict[str, CandidateSpec],
    multiplier: float,
    apply_mode: str = "uniform",
    util_threshold: float | None = None,
) -> dict[Any, tuple[float, float]]:
    df = df.copy()
    df["date"] = df["forecast_kst_dtm"].dt.date
    row_mask = resolve_mask(df, spec, all_specs)
    affected_dates = sorted(df.loc[row_mask, "date"].unique())
    out: dict[Any, tuple[float, float]] = {}
    for d in affected_dates:
        day_df = df[df["date"] == d]
        day_mask = resolve_mask(day_df, spec, all_specs)
        if not day_mask.any():
            continue
        base_day = evaluate_submission(
            long_to_wide_truth(day_df), long_to_wide_pred(day_df), time_col="kst_dtm"
        )
        r = simulate_on_period(
            day_df, day_mask, multiplier, base_day, apply_mode, util_threshold
        )
        if r.get("valid"):
            out[d] = (r["delta_score"], r["delta_ficr"])
    return out


def block_bootstrap_from_daily(
    daily_deltas: dict[Any, tuple[float, float]],
    n_iter: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    if not daily_deltas:
        return {"valid": False}
    dates = np.array(list(daily_deltas.keys()))
    rng = np.random.default_rng(seed)
    delta_scores: list[float] = []
    delta_ficrs: list[float] = []

    for _ in range(n_iter):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        ds = [daily_deltas[d][0] for d in sampled]
        dficr = [daily_deltas[d][1] for d in sampled]
        delta_scores.append(float(np.mean(ds)))
        delta_ficrs.append(float(np.mean(dficr)))

    ds_arr = np.array(delta_scores)
    dficr_arr = np.array(delta_ficrs)
    return {
        "valid": True,
        "n_iter": n_iter,
        "seed": seed,
        "n_dates": int(len(dates)),
        "delta_score_mean": float(ds_arr.mean()),
        "delta_score_median": float(np.median(ds_arr)),
        "delta_score_p05": float(np.quantile(ds_arr, 0.05)),
        "delta_score_p25": float(np.quantile(ds_arr, 0.25)),
        "delta_score_p75": float(np.quantile(ds_arr, 0.75)),
        "delta_score_p95": float(np.quantile(ds_arr, 0.95)),
        "delta_score_positive_ratio": float((ds_arr > 0).mean()),
        "delta_ficr_mean": float(dficr_arr.mean()),
        "delta_ficr_median": float(np.median(dficr_arr)),
        "delta_ficr_positive_ratio": float((dficr_arr > 0).mean()),
    }


def block_bootstrap_candidate(
    df: pd.DataFrame,
    spec: CandidateSpec,
    all_specs: dict[str, CandidateSpec],
    multiplier: float,
    apply_mode: str = "uniform",
    util_threshold: float | None = None,
    n_iter: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    daily = precompute_daily_deltas(
        df, spec, all_specs, multiplier, apply_mode, util_threshold
    )
    return block_bootstrap_from_daily(daily, n_iter=n_iter, seed=seed)


def top_candidate_configs(all_results: list[dict], limit: int = 10) -> list[dict]:
    full = [r for r in all_results if r["period"] == "2024_full"]
    ranked = sorted(full, key=lambda x: (x["delta_score"], x["delta_ficr"]), reverse=True)
    seen: set[tuple] = set()
    out = []
    for r in ranked:
        key = (r["candidate_id"], r["multiplier"], r["apply_mode"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def family_best(all_results: list[dict], family: str) -> dict | None:
    subset = [r for r in all_results if r["family"] == family and r["period"] == "2024_full"]
    if not subset:
        return None
    return max(subset, key=lambda x: (x["delta_score"], x["delta_ficr"]))


def grade_family_candidate(
    cid: str,
    spec: CandidateSpec,
    all_results: list[dict],
    lopo: list[dict],
    bootstrap: dict | None,
) -> tuple[str, dict | None]:
    full_rows = [r for r in all_results if r["candidate_id"] == cid and r["period"] == "2024_full"]
    if not full_rows:
        return "reject", None
    best_full = max(full_rows, key=lambda x: (x["delta_score"], x["delta_ficr"]))

    if best_full["delta_score"] <= 0:
        return "reject", best_full
    if best_full["net_transition"] <= 0:
        return "reject", best_full
    if best_full["success_to_fail"] >= best_full["fail_to_success"]:
        return "reject", best_full
    if best_full["delta_nmae"] < 0 and abs(best_full["delta_nmae"]) > best_full["delta_ficr"]:
        return "reject", best_full

    h23 = next(
        (
            r
            for r in all_results
            if r["candidate_id"] == cid
            and r["period"] == "2023_h2"
            and r["multiplier"] == best_full["multiplier"]
            and r["apply_mode"] == best_full["apply_mode"]
        ),
        None,
    )
    h24 = next(
        (
            r
            for r in all_results
            if r["candidate_id"] == cid
            and r["period"] == "2024_h2"
            and r["multiplier"] == best_full["multiplier"]
            and r["apply_mode"] == best_full["apply_mode"]
        ),
        None,
    )

    h2_ok = True
    if h23 and h24:
        h2_ok = (h23["delta_score"] >= -0.00005 and h24["delta_score"] >= -0.00005) or (
            h23["delta_score"] >= 0 and h24["delta_score"] >= 0
        )

    lopo_eval = [
        x
        for x in lopo
        if x["candidate_id"] == cid and x["apply_mode"] == best_full["apply_mode"]
    ]
    lopo_pos = sum(1 for x in lopo_eval if x["eval_delta_score"] > 0)
    lopo_neg = [x for x in lopo_eval if x["eval_delta_score"] < -0.00005]

    boot_pos = bootstrap.get("delta_score_positive_ratio", 0) if bootstrap else 0
    changed = best_full["changed_rows"]

    if (
        best_full["delta_score"] > 0
        and h2_ok
        and lopo_pos >= 2
        and best_full["net_transition"] > 0
        and best_full["success_to_fail"] < best_full["fail_to_success"]
        and boot_pos >= 0.70
        and best_full["multiplier"] <= 1.03
        and changed >= 50
    ):
        return "robust_candidate", best_full

    if lopo_neg and all(x["select_delta_score"] > 0 for x in lopo_neg):
        return "overfit_candidate", best_full

    if best_full["delta_score"] > 0 and changed >= 24 and 0.55 <= boot_pos < 0.70:
        return "weak_candidate", best_full

    if boot_pos < 0.55 and bootstrap is not None:
        return "reject", best_full

    if spec.exploratory and best_full["delta_score"] > 0:
        return "weak_candidate", best_full

    if best_full["delta_score"] > 0 and changed >= 24:
        return "weak_candidate", best_full

    return "reject", best_full


def analyze_hour_breakdown(
    df: pd.DataFrame,
    spec: CandidateSpec,
    all_specs: dict[str, CandidateSpec],
    multiplier: float,
    baseline: dict,
) -> list[dict]:
    rows = []
    hours = sorted(spec.hours) if spec.hours else []
    for h in hours:
        hmask = resolve_mask(df, spec, all_specs) & (df["hour"] == h)
        r = simulate_on_period(df, hmask, multiplier, baseline, "uniform")
        if r.get("valid"):
            rows.append({"hour": h, **r})
    return rows


def build_frames_and_baselines(bundles: dict) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    frames = build_all_period_frames(bundles, V13_BEST_CONFIG)
    baselines = baseline_metrics_by_period(frames)
    return frames, baselines
