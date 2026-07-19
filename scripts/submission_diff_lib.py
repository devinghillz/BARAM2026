"""Submission diff analysis — testable core logic."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from src.config import GROUP_CAPACITY_KWH, GROUP_COLUMNS

# eval_v13_comprehensive.py 와 동일
HOTSPOT_G23: dict[int, dict[int, set[int]]] = {
    2: {2: {14, 19, 21}, 9: {7}, 11: {17}},
    3: {2: {7, 22}, 8: {5, 14}, 11: {0, 4, 8, 17, 19, 22}},
}
HOTSPOT_G1: dict[int, dict[int, set[int]]] = {
    1: {1: {17}, 7: {0}},
}

GROUP_ID_TO_COL = {1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"}
COL_TO_GROUP_ID = {v: k for k, v in GROUP_ID_TO_COL.items()}

# 파이프라인 설정 (eval_v13_comprehensive / eval_v13_refine)
V13_BEST_CONFIG = {
    "slot_key": "mild23",
    "blend_v03": 0.07,
    "g2_mult": 1.06,
    "g3_season": {2: 1.02, 8: 1.05, 11: 1.02},
    "g1_mult": 1.04,
    "conditional": False,
}
OLD_HYBRID_CONFIG = {
    "slot_key": "mild23",
    "blend_v03": 0.05,
    "g2_mult": 1.05,
    "g3_season": {2: 1.0, 8: 1.05, 11: 1.02},
    "g1_mult": 1.04,
    "conditional": False,
}
G3FEB_KEEP_CONFIG = {
    "slot_key": "mild23",
    "blend_v03": 0.05,
    "g2_mult": 1.05,
    "g3_season": {2: 1.02, 8: 1.05, 11: 1.02},
    "g1_mult": 1.04,
    "conditional": False,
}
V12_CONFIG = {
    "slot_key": "full",
    "blend_v03": 0.0,
    "g2_mult": 1.05,
    "g3_season": {2: 1.0, 8: 1.05, 11: 1.02},
    "g1_mult": 1.0,
    "conditional": False,
}

# 하위 호환 alias
BASE_CONFIG = {k: v for k, v in V13_BEST_CONFIG.items() if k != "slot_key"}
BASE_CONFIG["slot_key"] = V13_BEST_CONFIG["slot_key"]
CANDIDATE_CONFIG = {k: v for k, v in OLD_HYBRID_CONFIG.items() if k != "slot_key"}
CANDIDATE_CONFIG["slot_key"] = OLD_HYBRID_CONFIG["slot_key"]

USER_HOTSPOT_SLICES = [
    {"label": "g1_m1", "group_id": 1, "months": {1}, "hours": None},
    {"label": "g1_m7", "group_id": 1, "months": {7}, "hours": None},
    {"label": "g1_m7_h0", "group_id": 1, "months": {7}, "hours": {0}},
    {"label": "g2_m2", "group_id": 2, "months": {2}, "hours": None},
    {"label": "g2_m9", "group_id": 2, "months": {9}, "hours": None},
    {"label": "g2_m11", "group_id": 2, "months": {11}, "hours": None},
    {"label": "g2_m2_h19", "group_id": 2, "months": {2}, "hours": {19}},
    {"label": "g3_m2", "group_id": 3, "months": {2}, "hours": None},
    {"label": "g3_m8", "group_id": 3, "months": {8}, "hours": None},
    {"label": "g3_m11", "group_id": 3, "months": {11}, "hours": None},
    {"label": "g3_m2_h22", "group_id": 3, "months": {2}, "hours": {22}},
    {"label": "g3_m8_h5", "group_id": 3, "months": {8}, "hours": {5}},
    {"label": "g3_m8_h14", "group_id": 3, "months": {8}, "hours": {14}},
    {"label": "g3_m11_h0", "group_id": 3, "months": {11}, "hours": {0}},
    {"label": "g3_m11_h4", "group_id": 3, "months": {11}, "hours": {4}},
    {"label": "g3_m11_h22", "group_id": 3, "months": {11}, "hours": {22}},
]


def load_submission_wide(path: str | pd.PathLike) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    return df


def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.melt(
        id_vars=["forecast_id", "forecast_kst_dtm"],
        value_vars=GROUP_COLUMNS,
        var_name="group_col",
        value_name="prediction",
    )
    long["group_id"] = long["group_col"].map(COL_TO_GROUP_ID)
    long["capacity"] = long["group_col"].map(GROUP_CAPACITY_KWH)
    long["month"] = long["forecast_kst_dtm"].dt.month
    long["hour"] = long["forecast_kst_dtm"].dt.hour
    long["key"] = long["forecast_id"] + "|g" + long["group_id"].astype(str)
    return long


def check_key_integrity(base_long: pd.DataFrame, cand_long: pd.DataFrame) -> dict[str, Any]:
    base_keys = set(base_long["key"])
    cand_keys = set(cand_long["key"])
    common = base_keys & cand_keys
    only_base = base_keys - cand_keys
    only_cand = cand_keys - base_keys
    base_dup = int(base_long["key"].duplicated().sum())
    cand_dup = int(cand_long["key"].duplicated().sum())
    return {
        "base_rows": len(base_long),
        "candidate_rows": len(cand_long),
        "common_keys": len(common),
        "missing_in_candidate": len(only_base),
        "missing_in_base": len(only_cand),
        "duplicate_keys_base": base_dup,
        "duplicate_keys_candidate": cand_dup,
        "keys_fully_match": (
            len(only_base) == 0 and len(only_cand) == 0 and base_dup == 0 and cand_dup == 0
        ),
    }


def merge_submissions(base_wide: pd.DataFrame, cand_wide: pd.DataFrame) -> pd.DataFrame:
    bl = wide_to_long(base_wide).rename(columns={"prediction": "p_base"})
    cl = wide_to_long(cand_wide).rename(columns={"prediction": "p_candidate"})
    merged = bl.merge(
        cl[["key", "p_candidate"]],
        on="key",
        how="inner",
        validate="one_to_one",
    )
    merged["delta"] = merged["p_candidate"] - merged["p_base"]
    merged["normalized_delta"] = merged["delta"] / merged["capacity"]
    merged["absolute_normalized_delta"] = merged["delta"].abs() / merged["capacity"]
    merged["changed"] = merged["delta"].abs() > 1e-9
    merged["direction"] = np.where(
        merged["delta"] > 1e-9, "increase",
        np.where(merged["delta"] < -1e-9, "decrease", "same"),
    )
    return merged


def is_hotspot_row(group_id: int, month: int, hour: int) -> bool:
    if group_id in HOTSPOT_G1:
        for mo, hrs in HOTSPOT_G1[group_id].items():
            if month == mo and hour in hrs:
                return True
    if group_id in HOTSPOT_G23:
        for mo, hrs in HOTSPOT_G23[group_id].items():
            if month == mo and hour in hrs:
                return True
    return False


def infer_correction_types(
    row: pd.Series,
    base_config: dict | None = None,
    candidate_config: dict | None = None,
) -> str:
    """생성 스크립트 설정 차이로 확실히 판별 가능한 보정만 표기."""
    base_cfg = base_config or V13_BEST_CONFIG
    cand_cfg = candidate_config or OLD_HYBRID_CONFIG
    gid, mo, hr = int(row["group_id"]), int(row["month"]), int(row["hour"])
    parts: list[str] = []

    if base_cfg.get("blend_v03") != cand_cfg.get("blend_v03"):
        parts.append(
            f"v03_blend_{base_cfg.get('blend_v03', 0):.0%}_to_{cand_cfg.get('blend_v03', 0):.0%}"
        )

    if gid == 2 and is_hotspot_row(2, mo, hr):
        b2, c2 = base_cfg.get("g2_mult", 1.0), cand_cfg.get("g2_mult", 1.0)
        if b2 != c2:
            parts.append(f"g2_hotspot_{b2}_to_{c2}")
        else:
            parts.append("g2_hotspot_unchanged")

    if gid == 3 and mo in HOTSPOT_G23[3] and hr in HOTSPOT_G23[3][mo]:
        b3 = base_cfg.get("g3_season", {}).get(mo, 1.0)
        c3 = cand_cfg.get("g3_season", {}).get(mo, 1.0)
        if b3 != c3:
            parts.append(f"g3_m{mo}_hotspot_{b3}_to_{c3}")
        else:
            parts.append(f"g3_m{mo}_hotspot_unchanged")

    if gid == 1 and is_hotspot_row(1, mo, hr):
        b1, c1 = base_cfg.get("g1_mult", 1.0), cand_cfg.get("g1_mult", 1.0)
        if b1 != c1:
            parts.append(f"g1_hotspot_{b1}_to_{c1}")
        else:
            parts.append("g1_hotspot_unchanged")

    if not parts:
        return "no_config_diff"
    return "; ".join(parts)


def _ficr_slice_match(gid: int, mo: int, hr: int, slices: list[dict]) -> bool:
    for sl in slices:
        hours = sl["hours"]
        hour_set = set(hours) if not isinstance(hours, set) else hours
        if sl["group_id"] == gid and sl["month"] == mo and hr in hour_set:
            return True
    return False


def build_v14_ficr_config(filename: str) -> dict | None:
    name = filename.lower()
    if "v14_ficr" not in name:
        return None
    cfg = copy.deepcopy(V13_BEST_CONFIG)
    cfg["ficr_multiplier"] = 1.03
    if "g2m12_h9h10_g1m3h4" in name:
        cfg["ficr_slices"] = [
            {"group_id": 2, "month": 12, "hours": [9, 10]},
            {"group_id": 1, "month": 3, "hours": [4]},
        ]
    elif "g2m12_h9h10h14h15" in name:
        cfg["ficr_slices"] = [{"group_id": 2, "month": 12, "hours": [9, 10, 14, 15]}]
    elif "g2m12_h9h10h15" in name:
        cfg["ficr_slices"] = [{"group_id": 2, "month": 12, "hours": [9, 10, 15]}]
    elif "g2m12_h9h10h14" in name:
        cfg["ficr_slices"] = [{"group_id": 2, "month": 12, "hours": [9, 10, 14]}]
    elif "g2m12_h9h10" in name:
        cfg["ficr_slices"] = [{"group_id": 2, "month": 12, "hours": [9, 10]}]
    else:
        return None
    return cfg


def is_change_explainable(
    row: pd.Series,
    base_config: dict,
    candidate_config: dict,
) -> tuple[bool, list[str]]:
    """행 단위 변화가 base/candidate 설정 차이로 설명 가능한지."""
    gid, mo, hr = int(row["group_id"]), int(row["month"]), int(row["hour"])
    ad = float(row["absolute_normalized_delta"])
    tags: list[str] = []

    if base_config.get("blend_v03") != candidate_config.get("blend_v03"):
        tags.append("v03_blend")

    if gid == 2 and is_hotspot_row(2, mo, hr):
        if base_config.get("g2_mult") != candidate_config.get("g2_mult"):
            tags.append("g2_hotspot_mult")

    if gid == 3 and mo in HOTSPOT_G23[3] and hr in HOTSPOT_G23[3][mo]:
        b3 = base_config.get("g3_season", {}).get(mo, 1.0)
        c3 = candidate_config.get("g3_season", {}).get(mo, 1.0)
        if b3 != c3:
            tags.append("g3_hotspot_season")

    if gid == 1 and is_hotspot_row(1, mo, hr):
        if base_config.get("g1_mult") != candidate_config.get("g1_mult"):
            tags.append("g1_hotspot_mult")

    ficr_slices = candidate_config.get("ficr_slices")
    ficr_mult = candidate_config.get("ficr_multiplier")
    if ficr_slices and ficr_mult and _ficr_slice_match(gid, mo, hr, ficr_slices):
        tags.append("ficr_conditional_slice")
        return True, tags

    if ad < 0.01:
        return True, tags or ["global_micro_blend"]

    specific = [t for t in tags if t != "v03_blend"]
    if specific:
        return True, tags

    if is_hotspot_row(gid, mo, hr) and tags:
        return True, tags

    return False, tags


def local_large_change_analysis(
    merged: pd.DataFrame,
    base_config: dict,
    candidate_config: dict,
) -> dict[str, Any]:
    """≥1% capacity 변화 행의 집중도·설명 가능성."""
    large = merged[merged["absolute_normalized_delta"] >= 0.01].copy()
    warn_band = merged[
        (merged["absolute_normalized_delta"] >= 0.01)
        & (merged["absolute_normalized_delta"] < 0.03)
    ].copy()

    explainable_flags = []
    for _, row in large.iterrows():
        ok, _ = is_change_explainable(row, base_config, candidate_config)
        explainable_flags.append(ok)

    unexplained = large.iloc[[i for i, ok in enumerate(explainable_flags) if not ok]] if len(large) else large

    top_slice = None
    if not merged.empty:
        by_gmh = aggregate_slice(merged, ["group_id", "month", "hour"])
        top = max(by_gmh, key=lambda x: x["mean_absolute_normalized_delta"])
        top_slice = {
            "group_id": top["group_id"],
            "month": top["month"],
            "hour": top["hour"],
            "mean_absolute_normalized_delta": top["mean_absolute_normalized_delta"],
            "mean_delta": top["mean_delta"],
            "row_count": top["row_count"],
        }

    return {
        "pct_rows_ge_1pct_capacity": float((merged["absolute_normalized_delta"] >= 0.01).mean()),
        "pct_rows_ge_2pct_capacity": float((merged["absolute_normalized_delta"] >= 0.02).mean()),
        "pct_rows_1_to_3pct_capacity": float(len(warn_band) / len(merged)) if len(merged) else 0.0,
        "large_change_row_count": int(len(large)),
        "large_change_hotspot_ratio": float(large["is_pipeline_hotspot"].mean()) if len(large) else 0.0,
        "large_change_explainable_ratio": float(np.mean(explainable_flags)) if explainable_flags else 1.0,
        "unexplained_large_change_count": int(len(unexplained)),
        "top_change_slice": top_slice,
    }


def check_schema_integrity(base_wide: pd.DataFrame, cand_wide: pd.DataFrame) -> dict[str, Any]:
    required = ["forecast_id", "forecast_kst_dtm", *GROUP_COLUMNS]
    base_cols = list(base_wide.columns)
    cand_cols = list(cand_wide.columns)
    base_missing = [c for c in required if c not in base_cols]
    cand_missing = [c for c in required if c not in cand_cols]
    return {
        "base_row_count_wide": len(base_wide),
        "candidate_row_count_wide": len(cand_wide),
        "row_count_match": len(base_wide) == len(cand_wide),
        "base_columns": base_cols,
        "candidate_columns": cand_cols,
        "columns_match": base_cols == cand_cols,
        "required_columns_present": len(base_missing) == 0 and len(cand_missing) == 0,
        "missing_columns_base": base_missing,
        "missing_columns_candidate": cand_missing,
        "schema_valid": (
            len(base_missing) == 0
            and len(cand_missing) == 0
            and len(base_wide) == len(cand_wide)
        ),
    }


def _rule(
    rule_id: str,
    passed: bool,
    measured_value: Any,
    threshold: Any,
    reason: str,
    *,
    severity: str = "info",
    comparator: str = "<=",
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "passed": passed,
        "triggered": not passed and severity in {"reject", "review", "warning"},
        "measured_value": measured_value,
        "threshold": threshold,
        "comparator": comparator,
        "severity": severity,
        "reason": reason,
    }


def _format_triggered_rule(rule: dict[str, Any]) -> str:
    mv = rule["measured_value"]
    th = rule["threshold"]
    if isinstance(mv, float):
        mv_s = f"{mv:.4f}"
    else:
        mv_s = str(mv)
    if isinstance(th, float):
        th_s = f"{th:.4f}"
    else:
        th_s = str(th)
    return f"{rule['rule_id']}: {mv_s} {rule['comparator']} {th_s} — {rule['reason']}"


def compute_performance_recommendation(
    performance_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validation label 없이 기본 not_evaluated. 메트릭 입력 시에만 계산."""
    if not performance_metrics:
        return {
            "recommendation": "not_evaluated",
            "reason": "validation label 없음 — 별도 로컬 평가 스크립트 참조",
        }

    score_drop = performance_metrics.get("score_drop_vs_base")
    if score_drop is None:
        return {
            "recommendation": "not_evaluated",
            "reason": "score 메트릭 미제공",
        }

    if score_drop <= 0.0003:
        return {
            "recommendation": "consider_submit",
            "reason": f"로컬 score 하락 {score_drop:.6f} ≤ 0.0003",
            "metrics": performance_metrics,
        }
    if score_drop <= 0.001:
        return {
            "recommendation": "conditional",
            "reason": f"로컬 score 하락 {score_drop:.6f} — trade-off 확인 필요",
            "metrics": performance_metrics,
        }
    return {
        "recommendation": "keep_base",
        "reason": f"로컬 score 하락 {score_drop:.6f} > 0.001",
        "metrics": performance_metrics,
    }


def submission_verdict(
    integrity: dict,
    overall: dict,
    hotspot: dict,
    clipping: dict,
    base_config: dict,
    candidate_config: dict,
    schema: dict | None = None,
    merged: pd.DataFrame | None = None,
    performance_metrics: dict | None = None,
) -> dict[str, Any]:
    """
    Submission 파일 안전성·변화 설명 가능성 판정.
    로컬 성능 우위는 판정하지 않음 (performance_recommendation 별도).
    """
    rules: list[dict[str, Any]] = []
    warnings: list[str] = []

    ad_med = overall["absolute_normalized_delta_median"]
    ad_p95 = overall["absolute_normalized_delta_quantiles"]["p95"]
    pct_chg = abs(overall["global_mean_change_pct"])
    changed_ratio = overall["changed_ratio"]

    bad_clip = (
        clipping.get("candidate_negative", 0) > 0
        or clipping.get("candidate_above_capacity", 0) > 0
        or clipping.get("base_negative", 0) > 0
        or clipping.get("candidate_nan", 0) > 0
        or clipping.get("candidate_inf", 0) > 0
    )
    new_clip = clipping.get("newly_clipped_to_zero", 0) + clipping.get("newly_clipped_to_capacity", 0)
    schema = schema or {"schema_valid": True, "row_count_match": True, "columns_match": True}

    # --- Hard integrity rules (reject) ---
    rules.append(_rule(
        "keys_fully_match",
        integrity["keys_fully_match"],
        integrity["keys_fully_match"],
        True,
        "base/candidate 키·중복 완전 일치 필요",
        severity="reject",
        comparator="==",
    ))
    rules.append(_rule(
        "schema_valid",
        schema.get("schema_valid", True),
        schema.get("schema_valid", True),
        True,
        "행 수·필수 컬럼 일치 필요",
        severity="reject",
        comparator="==",
    ))
    rules.append(_rule(
        "duplicate_keys",
        integrity["duplicate_keys_base"] == 0 and integrity["duplicate_keys_candidate"] == 0,
        integrity["duplicate_keys_base"] + integrity["duplicate_keys_candidate"],
        0,
        "중복 키 없음",
        severity="reject",
        comparator="==",
    ))
    rules.append(_rule(
        "candidate_nan",
        clipping.get("candidate_nan", 0) == 0,
        clipping.get("candidate_nan", 0),
        0,
        "candidate NaN 없음",
        severity="reject",
        comparator="==",
    ))
    rules.append(_rule(
        "candidate_inf",
        clipping.get("candidate_inf", 0) == 0,
        clipping.get("candidate_inf", 0),
        0,
        "candidate inf 없음",
        severity="reject",
        comparator="==",
    ))
    rules.append(_rule(
        "candidate_negative",
        clipping.get("candidate_negative", 0) == 0,
        clipping.get("candidate_negative", 0),
        0,
        "candidate 음수 없음",
        severity="reject",
        comparator="==",
    ))
    rules.append(_rule(
        "candidate_above_capacity",
        clipping.get("candidate_above_capacity", 0) == 0,
        clipping.get("candidate_above_capacity", 0),
        0,
        "candidate capacity 초과 없음",
        severity="reject",
        comparator="==",
    ))
    rules.append(_rule(
        "p95_abs_norm_delta",
        ad_p95 <= 0.05,
        ad_p95,
        0.05,
        "p95 |normalized delta| ≤ 5% capacity",
        severity="reject",
    ))
    rules.append(_rule(
        "new_clipping_significant",
        new_clip <= 10,
        new_clip,
        10,
        "신규 clipping(0/capacity) ≤ 10행",
        severity="reject",
    ))

    # --- Global micro-change (informational + safety bounds) ---
    global_micro = (
        integrity["keys_fully_match"]
        and schema.get("schema_valid", True)
        and not bad_clip
        and new_clip == 0
        and pct_chg <= 0.5
        and ad_med <= 0.003
        and ad_p95 <= 0.03
    )
    rules.append(_rule(
        "global_mean_change",
        pct_chg <= 0.5,
        pct_chg,
        0.5,
        "전역 평균 prediction 변화 ≤ 0.5%",
        severity="review",
    ))
    rules.append(_rule(
        "median_abs_norm_delta",
        ad_med <= 0.003,
        ad_med,
        0.003,
        "median |normalized delta| ≤ 0.3% capacity",
        severity="review",
    ))
    rules.append(_rule(
        "p95_abs_norm_delta_review",
        ad_p95 <= 0.03,
        ad_p95,
        0.03,
        "p95 |normalized delta| ≤ 3% capacity (safe 상한)",
        severity="review",
    ))
    rules.append(_rule(
        "changed_row_ratio",
        True,
        changed_ratio,
        None,
        "v03 blend 시 높은 changed ratio는 정상 — 단독 reject 조건 아님",
        severity="info",
        comparator="info",
    ))
    rules.append(_rule(
        "hotspot_delta_share",
        True,
        hotspot["abs_delta_hotspot_share"],
        None,
        "낮은 hotspot delta share는 v03 전역 blend 시 정상 — 단독 reject 조건 아님",
        severity="info",
        comparator="info",
    ))
    rules.append(_rule(
        "non_hotspot_changes",
        True,
        hotspot["non_hotspot_changed_ratio"],
        None,
        "hotspot 외부 변화는 v03 blend로 설명 가능 — 단독 reject 조건 아님",
        severity="info",
        comparator="info",
    ))

    local = (
        local_large_change_analysis(merged, base_config, candidate_config)
        if merged is not None and not merged.empty
        else {
            "pct_rows_ge_1pct_capacity": overall.get("pct_abs_norm_delta_ge_1pct", 0.0),
            "pct_rows_ge_2pct_capacity": overall.get("pct_abs_norm_delta_ge_2pct", 0.0),
            "pct_rows_1_to_3pct_capacity": 0.0,
            "large_change_row_count": 0,
            "large_change_hotspot_ratio": 0.0,
            "large_change_explainable_ratio": 1.0,
            "unexplained_large_change_count": 0,
            "top_change_slice": None,
        }
    )

    rules.append(_rule(
        "large_changes_explainable",
        local["unexplained_large_change_count"] == 0,
        local["unexplained_large_change_count"],
        0,
        "≥1% capacity 변화가 설정으로 설명 가능",
        severity="reject",
        comparator="==",
    ))

    if local["pct_rows_1_to_3pct_capacity"] > 0:
        warnings.append(
            f"hotspot 1~3% capacity 변화 행 {local['pct_rows_1_to_3pct_capacity']:.2%}"
        )

    reject_rules = [r for r in rules if r["severity"] == "reject" and not r["passed"]]
    review_rules = [r for r in rules if r["severity"] == "review" and not r["passed"]]
    triggered_rules = [
        {
            **r,
            "formatted": _format_triggered_rule(r),
        }
        for r in rules
        if r.get("triggered") or (r["severity"] in {"reject", "review", "warning"} and not r["passed"])
    ]

    # --- Verdict decision ---
    reasons: list[str] = []
    if reject_rules:
        submission_safety_verdict = "reject"
        reasons.extend(r["reason"] for r in reject_rules)
    elif 0.03 < ad_p95 <= 0.05:
        submission_safety_verdict = "review"
        reasons.append(f"p95 |normalized delta| {ad_p95:.4f} — 3~5% 구간")
    elif new_clip > 0:
        submission_safety_verdict = "review"
        reasons.append(f"신규 clipping {new_clip}행 발생")
    elif local["unexplained_large_change_count"] > 0:
        submission_safety_verdict = "review"
        reasons.append(
            f"설명 불가 ≥1% 변화 {local['unexplained_large_change_count']}행"
        )
    elif global_micro and local["pct_rows_ge_1pct_capacity"] == 0:
        submission_safety_verdict = "safe"
        reasons.append("전역 미세 blend + 무결성 정상 + 큰 변화 없음")
    elif global_micro and (
        local["pct_rows_1_to_3pct_capacity"] > 0
        or local["pct_rows_ge_1pct_capacity"] > 0
    ):
        submission_safety_verdict = "safe_with_warnings"
        reasons.append("파일 안전; 전역 blend + 일부 hotspot 1~3% 변화 (설명 가능)")
        if changed_ratio >= 0.9:
            reasons.append(f"changed row ratio {changed_ratio:.2%} — v03 blend로 정상")
    elif review_rules:
        submission_safety_verdict = "review"
        reasons.extend(r["reason"] for r in review_rules)
    else:
        submission_safety_verdict = "safe_with_warnings"
        reasons.append("무결성 정상; 변화 규모는 허용 범위이나 전역 blend 영향 존재")

    perf = compute_performance_recommendation(performance_metrics)

    return {
        "submission_safety_verdict": submission_safety_verdict,
        "performance_recommendation": perf["recommendation"],
        "performance_detail": perf,
        "verdict": submission_safety_verdict,
        "change_pattern": {
            "global_micro_change": global_micro,
            "changed_row_ratio": changed_ratio,
            "local_large_change": local,
        },
        "rule_results": rules,
        "triggered_rules": triggered_rules,
        "warnings": warnings,
        "reasons": reasons,
        "disclaimer": (
            "submission_safety_verdict는 파일 안전성·변화 설명 가능성만 판정하며, "
            "로컬 성능 우위는 performance_recommendation(별도 평가)에서 확인한다."
        ),
    }


def compute_pct_change_metrics(
    p_base: pd.Series,
    p_candidate: pd.Series,
    zero_eps: float = 1e-6,
) -> dict[str, float]:
    """
    global: mean(candidate)/mean(base)-1 — 전체 평균 수준 변화율
    rowwise: mean((c-b)/b) — base>0 행만, 행별 % 변화의 산술평균
    """
    base_mean = float(p_base.mean())
    cand_mean = float(p_candidate.mean())
    global_pct = (cand_mean / base_mean - 1.0) * 100.0 if base_mean != 0 else float("nan")

    nonzero = p_base.abs() > zero_eps
    excluded = int((~nonzero).sum())
    if nonzero.any():
        rowwise_pct = float(
            ((p_candidate[nonzero] - p_base[nonzero]) / p_base[nonzero] * 100.0).mean()
        )
    else:
        rowwise_pct = float("nan")

    return {
        "global_mean_change_pct": global_pct,
        "rowwise_mean_change_pct": rowwise_pct,
        "rowwise_excluded_zero_base_rows": excluded,
    }


def _quantiles(series: pd.Series, ps: list[float]) -> dict[str, float]:
    if series.empty:
        return {f"p{int(p)}": float("nan") for p in ps}
    q = series.quantile([p / 100 for p in ps])
    return {f"p{p}": float(q.iloc[i]) for i, p in enumerate(ps)}


def overall_stats(df: pd.DataFrame) -> dict[str, Any]:
    d = df["delta"]
    ad = df["absolute_normalized_delta"]
    nd = df["normalized_delta"]
    changed = df["changed"]
    return {
        "row_count": len(df),
        "changed_rows": int(changed.sum()),
        "changed_ratio": float(changed.mean()),
        "mean_p_base": float(df["p_base"].mean()),
        "mean_p_candidate": float(df["p_candidate"].mean()),
        "mean_delta": float(d.mean()),
        "mean_absolute_delta": float(d.abs().mean()),
        "median_delta": float(d.median()),
        "std_delta": float(d.std()),
        "min_delta": float(d.min()),
        "max_delta": float(d.max()),
        "absolute_delta_quantiles": _quantiles(d.abs(), [50, 90, 95, 99]),
        "normalized_delta_mean": float(nd.mean()),
        "absolute_normalized_delta_mean": float(ad.mean()),
        "absolute_normalized_delta_median": float(ad.median()),
        "absolute_normalized_delta_quantiles": _quantiles(ad, [50, 90, 95, 99]),
        "pct_abs_norm_delta_ge_1pct": float((ad >= 0.01).mean()),
        "pct_abs_norm_delta_ge_2pct": float((ad >= 0.02).mean()),
        "pct_abs_norm_delta_ge_3pct": float((ad >= 0.03).mean()),
        "pct_abs_norm_delta_ge_5pct": float((ad >= 0.05).mean()),
        "increase_rows": int((df["direction"] == "increase").sum()),
        "decrease_rows": int((df["direction"] == "decrease").sum()),
        "same_rows": int((df["direction"] == "same").sum()),
        **compute_pct_change_metrics(df["p_base"], df["p_candidate"]),
    }


def aggregate_slice(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if not group_cols:
        return [summarize_group(df, {}, group_cols)]
    rows = []
    for keys, g in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = dict(zip(group_cols, keys))
        rows.append(summarize_group(g, label, group_cols))
    return rows


def summarize_group(g: pd.DataFrame, label: dict, group_cols: list[str]) -> dict[str, Any]:
    return {
        **label,
        "row_count": len(g),
        "mean_p_base": float(g["p_base"].mean()),
        "mean_p_candidate": float(g["p_candidate"].mean()),
        "mean_delta": float(g["delta"].mean()),
        "mean_absolute_delta": float(g["delta"].abs().mean()),
        "mean_normalized_delta": float(g["normalized_delta"].mean()),
        "mean_absolute_normalized_delta": float(g["absolute_normalized_delta"].mean()),
        "increase_ratio": float((g["direction"] == "increase").mean()),
        "decrease_ratio": float((g["direction"] == "decrease").mean()),
    }


def hotspot_analysis(df: pd.DataFrame) -> dict[str, Any]:
    df = df.copy()
    df["is_pipeline_hotspot"] = df.apply(
        lambda r: is_hotspot_row(int(r["group_id"]), int(r["month"]), int(r["hour"])), axis=1
    )
    changed = df[df["changed"]]
    total_abs = df["delta"].abs().sum()
    hs = df[df["is_pipeline_hotspot"]]
    non = df[~df["is_pipeline_hotspot"]]

    slice_stats = []
    for spec in USER_HOTSPOT_SLICES:
        mask = df["group_id"] == spec["group_id"]
        mask &= df["month"].isin(spec["months"])
        if spec["hours"] is not None:
            mask &= df["hour"].isin(spec["hours"])
        sub = df[mask]
        if len(sub) == 0:
            continue
        slice_stats.append({
            "label": spec["label"],
            **summarize_group(sub, {"label": spec["label"]}, []),
        })

    return {
        "pipeline_hotspot_row_ratio": float(df["is_pipeline_hotspot"].mean()),
        "changed_rows_hotspot_ratio": float(changed["is_pipeline_hotspot"].mean()) if len(changed) else 0.0,
        "abs_delta_hotspot_share": float(hs["delta"].abs().sum() / total_abs) if total_abs > 0 else 0.0,
        "hotspot_mean_normalized_delta": float(hs["normalized_delta"].mean()),
        "non_hotspot_mean_normalized_delta": float(non["normalized_delta"].mean()),
        "hotspot_mean_abs_normalized_delta": float(hs["absolute_normalized_delta"].mean()),
        "non_hotspot_mean_abs_normalized_delta": float(non["absolute_normalized_delta"].mean()),
        "hotspot_changed_ratio": float(hs["changed"].mean()),
        "non_hotspot_changed_ratio": float(non["changed"].mean()),
        "user_hotspot_slices": slice_stats,
    }


def clipping_analysis(df: pd.DataFrame) -> dict[str, Any]:
    eps = 1e-6

    def clip_stats(prefix: str, col: str) -> dict[str, int]:
        p = df[col]
        cap = df["capacity"]
        return {
            f"{prefix}_at_zero": int((p <= eps).sum()),
            f"{prefix}_at_capacity": int((p >= cap - eps).sum()),
            f"{prefix}_negative": int((p < -eps).sum()),
            f"{prefix}_above_capacity": int((p > cap + eps).sum()),
            f"{prefix}_nan": int(p.isna().sum()),
            f"{prefix}_inf": int(np.isinf(p).sum()),
        }

    out: dict[str, Any] = {}
    out.update(clip_stats("base", "p_base"))
    out.update(clip_stats("candidate", "p_candidate"))

    base_z = (df["p_base"] <= eps) & (df["p_candidate"] > eps)
    base_cap = (df["p_base"] >= df["capacity"] - eps) & (df["p_candidate"] < df["capacity"] - eps)
    cand_z = (df["p_candidate"] <= eps) & (df["p_base"] > eps)
    cand_cap = (df["p_candidate"] >= df["capacity"] - eps) & (df["p_base"] < df["capacity"] - eps)
    out["newly_clipped_to_zero"] = int(cand_z.sum())
    out["newly_clipped_to_capacity"] = int(cand_cap.sum())
    out["base_was_zero_candidate_not"] = int(base_z.sum())
    out["base_was_capacity_candidate_not"] = int(base_cap.sum())
    return out


def top_changes(
    df: pd.DataFrame,
    n: int = 200,
    base_config: dict | None = None,
    candidate_config: dict | None = None,
) -> pd.DataFrame:
    out = df.nlargest(n, "absolute_normalized_delta").copy()
    out["is_hotspot"] = out.apply(
        lambda r: is_hotspot_row(int(r["group_id"]), int(r["month"]), int(r["hour"])), axis=1
    )
    out["inferred_correction"] = out.apply(
        lambda r: infer_correction_types(r, base_config, candidate_config), axis=1
    )
    return out[
        [
            "key", "forecast_id", "forecast_kst_dtm", "group_id", "month", "hour",
            "capacity", "p_base", "p_candidate", "delta", "normalized_delta",
            "is_hotspot", "inferred_correction",
        ]
    ]


def analyze_slice(
    df: pd.DataFrame,
    group_id: int | None = None,
    months: set[int] | None = None,
    hours: set[int] | None = None,
    label: str = "",
) -> dict[str, Any]:
    sub = df.copy()
    if group_id is not None:
        sub = sub[sub["group_id"] == group_id]
    if months is not None:
        sub = sub[sub["month"].isin(months)]
    if hours is not None:
        sub = sub[sub["hour"].isin(hours)]
    if sub.empty:
        return {"label": label, "row_count": 0}

    clip_eps = 1e-6
    cap = sub["capacity"]
    clip_base = {
        "at_zero": int((sub["p_base"] <= clip_eps).sum()),
        "at_capacity": int((sub["p_base"] >= cap - clip_eps).sum()),
    }
    clip_cand = {
        "at_zero": int((sub["p_candidate"] <= clip_eps).sum()),
        "at_capacity": int((sub["p_candidate"] >= cap - clip_eps).sum()),
    }
    out = summarize_group(sub, {"label": label}, [])
    out["clip_base"] = clip_base
    out["clip_candidate"] = clip_cand
    out["newly_clipped_zero"] = int(
        ((sub["p_base"] > clip_eps) & (sub["p_candidate"] <= clip_eps)).sum()
    )
    out["newly_clipped_capacity"] = int(
        ((sub["p_base"] < cap - clip_eps) & (sub["p_candidate"] >= cap - clip_eps)).sum()
    )
    return out


def infer_submission_config(path: str | pd.PathLike) -> dict | None:
    """Guess pipeline config from submission filename."""
    name = str(path).lower()
    if "g3feb102" in name or "g3feb_keep" in name:
        return G3FEB_KEEP_CONFIG
    if "g105_g1" in name or "old_hybrid" in name:
        return OLD_HYBRID_CONFIG
    if "v13_best" in name:
        return V13_BEST_CONFIG
    v14 = build_v14_ficr_config(name)
    if v14:
        return v14
    if "v12" in name:
        return V12_CONFIG
    return None


def run_analysis(
    base_path: str | pd.PathLike,
    candidate_path: str | pd.PathLike,
    base_config: dict | None = None,
    candidate_config: dict | None = None,
    performance_metrics: dict | None = None,
) -> dict[str, Any]:
    base_wide = load_submission_wide(base_path)
    cand_wide = load_submission_wide(candidate_path)
    base_cfg = base_config or infer_submission_config(base_path) or V13_BEST_CONFIG
    cand_cfg = candidate_config or infer_submission_config(candidate_path) or OLD_HYBRID_CONFIG
    schema = check_schema_integrity(base_wide, cand_wide)
    base_long = wide_to_long(base_wide)
    cand_long = wide_to_long(cand_wide)
    integrity = check_key_integrity(base_long, cand_long)
    merged = merge_submissions(base_wide, cand_wide)
    merged["is_pipeline_hotspot"] = merged.apply(
        lambda r: is_hotspot_row(int(r["group_id"]), int(r["month"]), int(r["hour"])), axis=1
    )

    overall = overall_stats(merged)
    integrity["changed_rows"] = overall["changed_rows"]
    integrity["changed_ratio"] = overall["changed_ratio"]

    result = {
        "meta": {
            "base_path": str(base_path),
            "candidate_path": str(candidate_path),
            "base_config": base_cfg,
            "candidate_config": cand_cfg,
            "group_capacities_kwh": GROUP_CAPACITY_KWH,
        },
        "schema": schema,
        "integrity": integrity,
        "overall": overall,
        "by_group": aggregate_slice(merged, ["group_id"]),
        "by_month": aggregate_slice(merged, ["month"]),
        "by_hour": aggregate_slice(merged, ["hour"]),
        "by_group_month": aggregate_slice(merged, ["group_id", "month"]),
        "by_group_hour": aggregate_slice(merged, ["group_id", "hour"]),
        "by_group_month_hour": aggregate_slice(merged, ["group_id", "month", "hour"]),
        "hotspot": hotspot_analysis(merged),
        "clipping": clipping_analysis(merged),
    }
    result["verdict"] = submission_verdict(
        integrity,
        overall,
        result["hotspot"],
        result["clipping"],
        base_cfg,
        cand_cfg,
        schema=schema,
        merged=merged,
        performance_metrics=performance_metrics,
    )
    result["top_changes"] = top_changes(merged, base_config=base_cfg, candidate_config=cand_cfg)
    return result
