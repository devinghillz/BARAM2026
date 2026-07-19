"""v14 FICR final candidate selection and submission generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS, SUBMISSION_DIR
from src.data_loader import load_submission_template
from scripts.ficr_candidate_families_lib import (
    CandidateSpec,
    block_bootstrap_candidate,
    build_candidate_specs,
    resolve_mask,
    simulate_on_period,
)
from scripts.submission_diff_lib import GROUP_ID_TO_COL, V13_BEST_CONFIG
from scripts.v13_factorial_lib import verify_sample_submission, verify_submission_values

ROOT = Path(__file__).resolve().parents[1]
PRIOR_JSON = ROOT / "outputs" / "v13_ficr_candidate_families.json"
PRIOR_RESULTS = ROOT / "outputs" / "v13_ficr_candidate_family_results.csv"
PRIOR_BOOTSTRAP = ROOT / "outputs" / "v13_ficr_candidate_bootstrap.csv"
V13_BEST_PATH = SUBMISSION_DIR / "v13_best.csv"

MULTIPLIER = 1.03
PERIODS = ["2024_full", "2024_h1", "2024_h2", "2023_h2"]
ROBUST_IDS = ["A1", "A2", "B1", "B2", "D1", "D2"]
P05_SERIOUS_NEGATIVE = -0.0001

CONDITION_LABELS = {
    "A1": "g2 m12 h9/h10",
    "A2": "g2 m12 h5/h9/h10",
    "B1": "g2 m12 h9/h10/h14/h15",
    "B2": "g2 m12 h5/h8/h9/h10/h14/h15",
    "D1": "g2 m12 h9/h10 + g1 m3 h4",
    "D2": "g2 m12 h5/h9/h10 + g1 m3 h4",
    "B1_minus_h14": "g2 m12 h9/h10/h15",
    "B1_minus_h15": "g2 m12 h9/h10/h14",
}

SIMPLICITY_SCORE = {
    "A1": 1,
    "B1_minus_h15": 2,
    "B1_minus_h14": 3,
    "D1": 4,
    "B1": 5,
    "A2": 6,
    "B2": 7,
    "D2": 8,
}

SUBMISSION_FILENAMES = {
    "SELECT_A1": "v14_ficr_g2m12_h9h10_x103.csv",
    "SELECT_B1": "v14_ficr_g2m12_h9h10h14h15_x103.csv",
    "SELECT_D1": "v14_ficr_g2m12_h9h10_g1m3h4_x103.csv",
    "SELECT_B1_MINUS_H14": "v14_ficr_g2m12_h9h10h15_x103.csv",
    "SELECT_B1_MINUS_H15": "v14_ficr_g2m12_h9h10h14_x103.csv",
}


@dataclass
class SelectionSpec:
    selection_id: str
    candidate_key: str
    spec: CandidateSpec
    multiplier: float = MULTIPLIER


def build_ablation_specs() -> dict[str, CandidateSpec]:
    base = build_candidate_specs()
    specs = dict(base)
    specs["B1_minus_h14"] = CandidateSpec(
        "B1_minus_h14", "B_ablation", group_id=2, month=12, hours={9, 10, 15}
    )
    specs["B1_minus_h15"] = CandidateSpec(
        "B1_minus_h15", "B_ablation", group_id=2, month=12, hours={9, 10, 14}
    )
    return specs


def selection_spec_for_choice(choice: str, specs: dict[str, CandidateSpec] | None = None) -> SelectionSpec | None:
    specs = specs or build_ablation_specs()
    mapping = {
        "SELECT_A1": "A1",
        "SELECT_B1": "B1",
        "SELECT_D1": "D1",
        "SELECT_B1_MINUS_H14": "B1_minus_h14",
        "SELECT_B1_MINUS_H15": "B1_minus_h15",
    }
    key = mapping.get(choice)
    if key is None:
        return None
    return SelectionSpec(choice, key, specs[key], MULTIPLIER)


def load_prior_results() -> dict[str, Any]:
    if not PRIOR_JSON.exists():
        raise FileNotFoundError(f"Missing prior results: {PRIOR_JSON}")
    report = json.loads(PRIOR_JSON.read_text(encoding="utf-8"))
    results_df = pd.read_csv(PRIOR_RESULTS) if PRIOR_RESULTS.exists() else pd.DataFrame()
    bootstrap_df = pd.read_csv(PRIOR_BOOTSTRAP) if PRIOR_BOOTSTRAP.exists() else pd.DataFrame()
    return {"report": report, "results_df": results_df, "bootstrap_df": bootstrap_df}


def _uniform_result(
    results_df: pd.DataFrame, candidate_id: str, period: str, multiplier: float = MULTIPLIER
) -> dict | None:
    if results_df.empty:
        return None
    rows = results_df[
        (results_df["candidate_id"] == candidate_id)
        & (results_df["period"] == period)
        & (results_df["multiplier"] == multiplier)
        & (results_df["apply_mode"] == "uniform")
    ]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _lopo_cross(report: dict, candidate_id: str) -> tuple[float | None, float | None, int]:
    lopo = [
        x
        for x in report.get("lopo", [])
        if x["candidate_id"] == candidate_id and x["apply_mode"] == "uniform"
    ]
    cross_23_24 = next((x for x in lopo if x["fold_id"] == 1), None)
    cross_24_23 = next((x for x in lopo if x["fold_id"] == 2), None)
    pos = sum(1 for x in lopo if x.get("eval_delta_score", 0) > 0)
    return (
        cross_23_24["eval_delta_score"] if cross_23_24 else None,
        cross_24_23["eval_delta_score"] if cross_24_23 else None,
        pos,
    )


def _bootstrap_stats(report: dict, bootstrap_df: pd.DataFrame, candidate_id: str) -> dict:
    grade = next((g for g in report.get("grades", []) if g["candidate_id"] == candidate_id), None)
    boot = grade.get("bootstrap") if grade else None
    if boot:
        return boot
    if not bootstrap_df.empty:
        row = bootstrap_df[
            (bootstrap_df["candidate_id"] == candidate_id)
            & (bootstrap_df["multiplier"] == MULTIPLIER)
            & (bootstrap_df["apply_mode"] == "uniform")
        ]
        if not row.empty:
            return row.iloc[0].to_dict()
    return {}


def build_robust_ranking_table(prior: dict) -> pd.DataFrame:
    report = prior["report"]
    results_df = prior["results_df"]
    bootstrap_df = prior["bootstrap_df"]
    rows = []
    for cid in ROBUST_IDS:
        grade = next((g for g in report["grades"] if g["candidate_id"] == cid), None)
        if not grade or grade["grade"] != "robust_candidate":
            continue
        bf = grade["best_full"]
        boot = _bootstrap_stats(report, bootstrap_df, cid)
        c23, c24, lopo_pos = _lopo_cross(report, cid)
        period_metrics = {p: _uniform_result(results_df, cid, p) for p in PERIODS}
        rows.append({
            "candidate_id": cid,
            "family": grade["family"],
            "condition": CONDITION_LABELS[cid],
            "multiplier": bf["multiplier"],
            "utilization_condition": bf.get("apply_mode", "uniform"),
            "delta_score_2024_full": bf["delta_score"],
            "delta_score_2024_h1": period_metrics["2024_h1"]["delta_score"] if period_metrics["2024_h1"] else np.nan,
            "delta_score_2024_h2": period_metrics["2024_h2"]["delta_score"] if period_metrics["2024_h2"] else np.nan,
            "delta_score_2023_h2": period_metrics["2023_h2"]["delta_score"] if period_metrics["2023_h2"] else np.nan,
            "cross_period_23_to_24": c23,
            "cross_period_24_to_23": c24,
            "lopo_positive_folds": lopo_pos,
            "bootstrap_delta_score_mean": boot.get("delta_score_mean"),
            "bootstrap_delta_score_median": boot.get("delta_score_median"),
            "bootstrap_delta_score_p05": boot.get("delta_score_p05"),
            "bootstrap_p_delta_score_positive": boot.get("delta_score_positive_ratio"),
            "bootstrap_p_delta_ficr_positive": boot.get("delta_ficr_positive_ratio"),
            "fail_to_success": bf["fail_to_success"],
            "success_to_fail": bf["success_to_fail"],
            "net_transition": bf["net_transition"],
            "tier0_to_tier1": bf.get("tier0_to_tier1", 0),
            "tier0_to_tier2": bf.get("tier0_to_tier2", 0),
            "tier1_to_tier2": bf.get("tier1_to_tier2", 0),
            "tier2_to_tier1": bf.get("tier2_to_tier1", 0),
            "changed_rows": bf["changed_rows"],
            "simplicity_score": SIMPLICITY_SCORE.get(cid, 99),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["cross_period_min"] = df[["cross_period_23_to_24", "cross_period_24_to_23"]].min(axis=1)
    return df.sort_values(
        by=[
            "cross_period_min",
            "bootstrap_p_delta_score_positive",
            "bootstrap_delta_score_p05",
            "delta_score_2024_full",
            "net_transition",
            "simplicity_score",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)


def passes_mandatory(row: pd.Series) -> tuple[bool, list[str]]:
    reasons = []
    if row["delta_score_2024_full"] <= 0:
        reasons.append("2024_full_delta_nonpos")
    if row["delta_score_2024_h2"] < -0.00005:
        reasons.append("2024_h2_drop")
    if row["delta_score_2023_h2"] < -0.00005:
        reasons.append("2023_h2_drop")
    if row["lopo_positive_folds"] < 2:
        reasons.append("lopo_insufficient")
    if (row.get("bootstrap_p_delta_score_positive") or 0) < 0.70:
        reasons.append("bootstrap_prob_low")
    p05 = row.get("bootstrap_delta_score_p05")
    if p05 is not None and not np.isnan(p05) and p05 < P05_SERIOUS_NEGATIVE:
        reasons.append("bootstrap_p05_serious_negative")
    if row["net_transition"] <= 0:
        reasons.append("net_transition_nonpos")
    if row["success_to_fail"] >= row["fail_to_success"]:
        reasons.append("success_to_fail_ge_fail_to_success")
    if row["multiplier"] > 1.03:
        reasons.append("multiplier_too_high")
    return len(reasons) == 0, reasons


def composite_rank_key(row: pd.Series) -> tuple:
    return (
        row.get("cross_period_min", -999),
        row.get("bootstrap_p_delta_score_positive", 0),
        row.get("bootstrap_delta_score_p05", -999),
        row.get("delta_score_2024_full", -999),
        row.get("net_transition", -999),
        -row.get("simplicity_score", 99),
    )


def rank_eligible_candidates(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in ranking.iterrows():
        ok, fails = passes_mandatory(row)
        rows.append({**row.to_dict(), "mandatory_pass": ok, "mandatory_failures": fails})
    df = pd.DataFrame(rows)
    passed = df[df["mandatory_pass"]].copy()
    if passed.empty:
        return passed
    passed["_rank_key"] = passed.apply(composite_rank_key, axis=1)
    return passed.sort_values("_rank_key", ascending=False).reset_index(drop=True)


def hour_slice_spec(hour: int) -> CandidateSpec:
    return CandidateSpec(f"g2_m12_h{hour}", "B_hour", group_id=2, month=12, hours={hour})


def analyze_b1_hour_contributions(
    frames: dict[str, pd.DataFrame],
    baselines: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    specs = build_ablation_specs()
    for hour in [9, 10, 14, 15]:
        spec = hour_slice_spec(hour)
        for period, df in frames.items():
            mask = resolve_mask(df, spec, specs)
            r = simulate_on_period(df, mask, MULTIPLIER, baselines[period], "uniform")
            if not r.get("valid"):
                continue
            rows.append({
                "slice": f"g2_m12_h{hour}",
                "hour": hour,
                "period": period,
                "changed_rows": r["changed_rows"],
                "delta_score": r["delta_score"],
                "delta_nmae": r["delta_nmae"],
                "delta_ficr": r["delta_ficr"],
                "fail_to_success": r["fail_to_success"],
                "success_to_fail": r["success_to_fail"],
                "mean_signed_error_before": float(df.loc[mask, "signed_error"].mean()),
                "mean_signed_error_after": float(df.loc[mask, "signed_error"].mean()) + r["mean_prediction_delta"],
                "under_ratio_before": r["under_ratio_before"],
                "under_ratio_after": r["under_ratio_after"],
                "over_ratio_before": r["over_ratio_before"],
                "over_ratio_after": r["over_ratio_after"],
            })
    return pd.DataFrame(rows)


def evaluate_h14_safety(hour_df: pd.DataFrame) -> dict[str, Any]:
    h14 = hour_df[hour_df["hour"] == 14]
    if h14.empty:
        return {"passes": False, "reasons": ["no_h14_data"]}
    reasons = []
    agg = {
        "fail_to_success": int(h14["fail_to_success"].sum()),
        "success_to_fail": int(h14["success_to_fail"].sum()),
        "delta_score_sum": float(h14["delta_score"].sum()),
        "delta_nmae_sum": float(h14["delta_nmae"].sum()),
        "delta_ficr_sum": float(h14["delta_ficr"].sum()),
    }
    if agg["fail_to_success"] <= agg["success_to_fail"]:
        reasons.append("h14_fail_to_success_not_greater")
    for period in ["2024_h2", "2023_h2"]:
        pr = h14[h14["period"] == period]
        if pr.empty:
            reasons.append(f"missing_{period}")
        elif pr.iloc[0]["delta_score"] <= 0:
            reasons.append(f"h14_delta_score_nonpos_{period}")
    if agg["delta_ficr_sum"] <= abs(agg["delta_nmae_sum"]):
        reasons.append("h14_ficr_gain_not_gt_nmae_loss")
    return {"passes": len(reasons) == 0, "reasons": reasons, "aggregate": agg}


def evaluate_ablation_candidate(
    candidate_key: str,
    frames: dict[str, pd.DataFrame],
    baselines: dict[str, dict],
    specs: dict[str, CandidateSpec],
    prior_results: pd.DataFrame | None = None,
    run_bootstrap: bool = True,
) -> dict[str, Any]:
    spec = specs[candidate_key]
    period_rows = []
    for period, df in frames.items():
        if prior_results is not None and candidate_key in ROBUST_IDS:
            cached = _uniform_result(prior_results, candidate_key, period, MULTIPLIER)
            if cached:
                period_rows.append({"period": period, **cached})
                continue
        mask = resolve_mask(df, spec, specs)
        r = simulate_on_period(df, mask, MULTIPLIER, baselines[period], "uniform")
        if r.get("valid"):
            period_rows.append({"period": period, "candidate_id": candidate_key, **r})
    boot = {}
    if run_bootstrap and "2024_full" in frames:
        boot = block_bootstrap_candidate(
            frames["2024_full"], spec, specs, MULTIPLIER, "uniform", None, 500, 42
        )
    full = next((x for x in period_rows if x["period"] == "2024_full"), {})
    return {"candidate_id": candidate_key, "period_results": period_rows, "bootstrap": boot, **full}


def run_ablation_study(
    frames: dict[str, pd.DataFrame],
    baselines: dict[str, dict],
    prior_results: pd.DataFrame,
) -> pd.DataFrame:
    specs = build_ablation_specs()
    rows = []
    for key in ["A1", "B1", "B1_minus_h14", "B1_minus_h15", "D1"]:
        res = evaluate_ablation_candidate(key, frames, baselines, specs, prior_results, True)
        for pr in res["period_results"]:
            rows.append({
                "candidate_id": key,
                "period": pr["period"],
                "multiplier": MULTIPLIER,
                "delta_score": pr.get("delta_score"),
                "delta_nmae": pr.get("delta_nmae"),
                "delta_ficr": pr.get("delta_ficr"),
                "changed_rows": pr.get("changed_rows"),
                "fail_to_success": pr.get("fail_to_success"),
                "success_to_fail": pr.get("success_to_fail"),
                "net_transition": pr.get("net_transition"),
            })
        boot = res.get("bootstrap") or {}
        rows.append({
            "candidate_id": key,
            "period": "bootstrap_summary",
            "multiplier": MULTIPLIER,
            "delta_score": boot.get("delta_score_mean"),
            "bootstrap_p05": boot.get("delta_score_p05"),
            "bootstrap_p_positive": boot.get("delta_score_positive_ratio"),
            "changed_rows": res.get("changed_rows"),
        })
    return pd.DataFrame(rows)


def ablation_rank_row(ablation_df: pd.DataFrame, candidate_key: str, choice: str) -> dict:
    sub = ablation_df[ablation_df["candidate_id"] == candidate_key]
    boot = sub[sub["period"] == "bootstrap_summary"]
    periods = {r["period"]: r for _, r in sub[sub["period"] != "bootstrap_summary"].iterrows()}
    return {
        "candidate_id": candidate_key,
        "selection_choice": choice,
        "delta_score_2024_full": periods.get("2024_full", {}).get("delta_score"),
        "delta_score_2024_h2": periods.get("2024_h2", {}).get("delta_score"),
        "delta_score_2023_h2": periods.get("2023_h2", {}).get("delta_score"),
        "bootstrap_p05": boot.iloc[0].get("bootstrap_p05") if not boot.empty else np.nan,
        "bootstrap_p_positive": boot.iloc[0].get("bootstrap_p_positive") if not boot.empty else np.nan,
        "net_transition": periods.get("2024_full", {}).get("net_transition"),
        "changed_rows": periods.get("2024_full", {}).get("changed_rows"),
        "simplicity_score": SIMPLICITY_SCORE.get(candidate_key, 99),
    }


def ablation_beats_b1_on_risk(ablation_df: pd.DataFrame, candidate_key: str, b1_full: float, tol: float = 0.00005) -> bool:
    row = ablation_df[(ablation_df["candidate_id"] == candidate_key) & (ablation_df["period"] == "2024_full")]
    if row.empty:
        return False
    if b1_full - float(row.iloc[0]["delta_score"]) > tol:
        return False
    b1b = ablation_df[(ablation_df["candidate_id"] == "B1") & (ablation_df["period"] == "bootstrap_summary")]
    cb = ablation_df[(ablation_df["candidate_id"] == candidate_key) & (ablation_df["period"] == "bootstrap_summary")]
    if b1b.empty or cb.empty:
        return True
    return float(cb.iloc[0]["bootstrap_p05"]) >= float(b1b.iloc[0]["bootstrap_p05"]) or float(
        cb.iloc[0]["bootstrap_p_positive"]
    ) >= float(b1b.iloc[0]["bootstrap_p_positive"])


def decide_final_selection(
    eligible: pd.DataFrame,
    h14_safety: dict[str, Any],
    ablation_df: pd.DataFrame,
) -> dict[str, Any]:
    if eligible.empty:
        return {"choice": "KEEP_V13_BEST", "reason": "no_mandatory_pass"}

    b1_full = float(
        ablation_df[(ablation_df["candidate_id"] == "B1") & (ablation_df["period"] == "2024_full")].iloc[0]["delta_score"]
    )

    if not h14_safety.get("passes", False):
        alts = []
        for key, choice in [
            ("B1_minus_h14", "SELECT_B1_MINUS_H14"),
            ("B1_minus_h15", "SELECT_B1_MINUS_H15"),
            ("A1", "SELECT_A1"),
            ("D1", "SELECT_D1"),
        ]:
            if ablation_beats_b1_on_risk(ablation_df, key, b1_full):
                alts.append(ablation_rank_row(ablation_df, key, choice))
        if alts:
            alt_df = pd.DataFrame(alts)
            alt_df["_rk"] = alt_df.apply(
                lambda r: (
                    min(r.get("delta_score_2024_h2", -9), r.get("delta_score_2023_h2", -9)),
                    r.get("bootstrap_p_positive", 0),
                    r.get("bootstrap_p05", -9),
                    r.get("delta_score_2024_full", -9),
                    -r.get("simplicity_score", 99),
                ),
                axis=1,
            )
            pick = alt_df.sort_values("_rk", ascending=False).iloc[0]
            return {
                "choice": pick["selection_choice"],
                "reason": "B1_h14_failed_pick_safer_ablation",
                "b1_h14_safety": h14_safety,
                "selected": pick.to_dict(),
            }
        non_b1 = eligible[eligible["candidate_id"] != "B1"]
        pick = non_b1.iloc[0]
        return {
            "choice": f"SELECT_{pick['candidate_id']}",
            "reason": "B1_h14_failed_fallback",
            "selected": pick.to_dict(),
        }

    top = eligible.iloc[0]
    choice_map = {"A1": "SELECT_A1", "B1": "SELECT_B1", "D1": "SELECT_D1"}
    if top["candidate_id"] in choice_map:
        return {
            "choice": choice_map[top["candidate_id"]],
            "reason": "top_composite_rank_h14_passed",
            "selected": top.to_dict(),
        }
    for fb in ["D1", "A1", "B1"]:
        sub = eligible[eligible["candidate_id"] == fb]
        if not sub.empty:
            return {"choice": choice_map[fb], "reason": f"mapped_{top['candidate_id']}_to_{fb}", "selected": sub.iloc[0].to_dict()}
    return {"choice": "KEEP_V13_BEST", "reason": "unmapped_top"}


def test_row_mask(ts: pd.Series, month: int, hours: set[int]) -> pd.Series:
    return (ts.dt.month == month) & (ts.dt.hour.isin(hours))


def apply_spec_to_submission(df: pd.DataFrame, spec: CandidateSpec, multiplier: float) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["forecast_kst_dtm"])
    specs = build_ablation_specs()

    def _apply(gid: int, month: int, hours: set[int]) -> None:
        col = GROUP_ID_TO_COL[gid]
        cap = GROUP_CAPACITY_KWH[col]
        mask = test_row_mask(ts, month, hours)
        current = out.loc[mask, col]
        scaled = current * multiplier
        safe = scaled <= cap + 1e-6
        out.loc[mask, col] = scaled.where(safe, current)

    if spec.component_ids:
        for cid in spec.component_ids:
            comp = specs[cid]
            _apply(comp.group_id, comp.month, comp.hours)
    else:
        _apply(spec.group_id, spec.month, spec.hours)
    return out


def generate_submission(choice: str, base_path: Path = V13_BEST_PATH) -> Path | None:
    sel = selection_spec_for_choice(choice)
    if sel is None:
        return None
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    patched = apply_spec_to_submission(base, sel.spec, sel.multiplier)
    fname = SUBMISSION_FILENAMES[choice]
    out_path = SUBMISSION_DIR / fname
    patched.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def expected_change_mask(cand: pd.DataFrame, choice: str) -> dict[str, pd.Series]:
    sel = selection_spec_for_choice(choice)
    ts = pd.to_datetime(cand["forecast_kst_dtm"])
    specs = build_ablation_specs()
    masks = {col: pd.Series(False, index=cand.index) for col in GROUP_COLUMNS}
    if not sel:
        return masks
    if sel.spec.component_ids:
        for cid in sel.spec.component_ids:
            comp = specs[cid]
            col = GROUP_ID_TO_COL[comp.group_id]
            masks[col] |= test_row_mask(ts, comp.month, comp.hours)
    else:
        col = GROUP_ID_TO_COL[sel.spec.group_id]
        masks[col] = test_row_mask(ts, sel.spec.month, sel.spec.hours)
    return masks


def verify_submission_patch(base_path: Path, cand_path: Path, choice: str) -> dict[str, Any]:
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    cand = pd.read_csv(cand_path, encoding="utf-8-sig")
    masks = expected_change_mask(cand, choice)
    slice_changes = {}
    for col in GROUP_COLUMNS:
        diff = (cand[col] - base[col]).abs() > 1e-6
        m = masks[col]
        inside = diff & m
        outside = diff & ~m
        entry = {"changed_inside": int(inside.sum()), "changed_outside": int(outside.sum())}
        if inside.any():
            ratios = (cand.loc[inside, col] / base.loc[inside, col]).dropna()
            entry["multiplier_min"] = float(ratios.min())
            entry["multiplier_max"] = float(ratios.max())
        slice_changes[col] = entry
    integrity = {
        **verify_sample_submission(cand_path),
        **verify_submission_values(cand_path),
        "changed_rows_total": int(sum(slice_changes[c]["changed_inside"] for c in GROUP_COLUMNS)),
        "slice_changes": slice_changes,
    }
    return integrity
