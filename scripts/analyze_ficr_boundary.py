"""
v13_best FICR near-boundary failure analysis.

사용법:
  python scripts/analyze_ficr_boundary.py
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

from src.config import OUTPUT_DIR
from src.data_loader import load_labels
from scripts.eval_v13_comprehensive import SPLITS, build_split_bundle
from scripts.ficr_boundary_lib import (
    FICR_DEFINITION,
    MIN_SAMPLES,
    SIM_MULTIPLIERS,
    aggregate_slice_stats,
    baseline_metrics_by_period,
    build_all_period_frames,
    build_split_bundle,
    classify_summary,
    filter_candidate_slices,
    grade_candidate,
    long_to_wide_truth,
    simulate_slice_multiplier,
    validate_period_keys,
)
from scripts.submission_diff_lib import V13_BEST_CONFIG

OUTPUT_JSON = OUTPUT_DIR / "v13_ficr_boundary_analysis.json"
OUTPUT_MD = OUTPUT_DIR / "v13_ficr_boundary_analysis.md"
OUTPUT_ROWS = OUTPUT_DIR / "v13_ficr_boundary_rows.csv"
OUTPUT_SLICES = OUTPUT_DIR / "v13_ficr_boundary_candidate_slices.csv"
OUTPUT_SIMS = OUTPUT_DIR / "v13_ficr_boundary_simulations.csv"


def write_markdown(report: dict) -> None:
    rec = report["recommendation"]
    lines = [
        "# v13 FICR Boundary 분석",
        "",
        "## 1. 결론",
        "",
        f"- **다음 submission 생성**: `{rec['create_submission']}`",
        f"- **근거**: {rec['summary']}",
        "",
    ]
    for r in rec.get("reasons", []):
        lines.append(f"- {r}")

    lines.extend(["", "## 2. 실제 FICR 계산 정의", ""])
    for k, v in FICR_DEFINITION.items():
        lines.append(f"- **{k}**: {v}")

    lines.extend(["", "## 3. 기간별 FICR 실패 구조", ""])
    for pname, summ in report["period_summaries"].items():
        lines.append(f"### {pname}")
        lines.append(f"- active rows: {summ['active_rows']}")
        lines.append(f"- failed ratio: {summ['failed_ratio']:.2%}")
        lines.append(f"- near-boundary failed: {summ['near_boundary_failed_ratio']:.2%}")
        lines.append(f"- under near: {summ['under_near_boundary_ratio']:.2%}")
        lines.append(f"- over near: {summ['over_near_boundary_ratio']:.2%}")
        lines.append("")

    lines.extend(["", "## 4. Near-boundary Under-prediction", ""])
    for row in report["top_under_slices"][:10]:
        lines.append(
            f"- `{row['slice']}`: under_near={row['under_near_count']} "
            f"({row['under_near_ratio']:.1%}), med_mult={row['median_min_multiplier']:.4f}"
        )

    lines.extend(["", "## 5. Near-boundary Over-prediction", ""])
    for row in report["top_over_slices"][:10]:
        lines.append(
            f"- `{row['slice']}`: over_near={row['over_near_count']} "
            f"({row['over_near_ratio']:.1%})"
        )

    lines.extend(["", "## 6. 그룹·월·시간 패턴", ""])
    for row in report["top_candidate_slices"][:10]:
        lines.append(
            f"- `{row['slice']}`: rows={row['total_rows']}, score={row['candidate_score']}, "
            f"hotspot={row['is_pipeline_hotspot']}, grade={row.get('grade', 'n/a')}"
        )

    lines.extend(["", "## 7. 풍속·Utilization 패턴", ""])
    for dim in ["ws_bin", "util_pred_bin", "util_actual_bin"]:
        lines.append(f"### {dim}")
        for row in report["pattern_slices"].get(dim, [])[:5]:
            lines.append(
                f"- {row['slice']}: near={row['near_boundary_ratio']:.1%}, "
                f"under={row['under_near_ratio']:.1%}"
            )
        lines.append("")

    lines.extend(["", "## 8. 기존 Hotspot과의 중복", ""])
    hs = report["hotspot_overlap"]
    lines.append(f"- hotspot slices: {hs['hotspot_slice_count']}")
    lines.append(f"- non-hotspot high-under slices: {hs['non_hotspot_top_count']}")
    for row in hs.get("non_hotspot_top", [])[:5]:
        lines.append(f"  - {row['slice']}: under_near_ratio={row['under_near_ratio']:.1%}")

    lines.extend(["", "## 9. 단일 Slice 시뮬레이션", ""])
    for row in report["best_simulations"][:10]:
        lines.append(
            f"- `{row['slice']}` ×{row['multiplier']:.3f} ({row['period']}): "
            f"ΔScore={row['delta_score']:+.6f}, ΔFICR={row['delta_ficr']:+.6f}, "
            f"net_gain={row['net_ficr_row_gain']}"
        )

    lines.extend(["", "## 10. 강건성 평가", ""])
    for grade, cnt in report["grade_counts"].items():
        lines.append(f"- {grade}: {cnt}")

    lines.extend(["", "## 11. 다음 Submission 후보", ""])
    lines.append(f"**{rec['create_submission']}**")
    for r in rec.get("reasons", []):
        lines.append(f"- {r}")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    labels = load_labels()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundles = {name: build_split_bundle(labels, vs) for name, vs in SPLITS.items()}

    print("=== Build aligned frames ===")
    frames = build_all_period_frames(bundles, V13_BEST_CONFIG)
    key_report = validate_period_keys(frames)
    baselines = baseline_metrics_by_period(frames)

    period_summaries = {p: classify_summary(df) for p, df in frames.items()}
    all_rows = pd.concat(frames.values(), ignore_index=True)

    slice_stats_by_period = {}
    pattern_slices = {}
    for pname, df in frames.items():
        slice_stats_by_period[pname] = aggregate_slice_stats(
            df, ["group_id", "month", "hour"]
        )
        for dim in ["ws_bin", "util_pred_bin", "util_actual_bin"]:
            pattern_slices.setdefault(dim, []).extend(
                aggregate_slice_stats(df, [dim], slice_label=dim)
            )

    pooled_gmh = {}
    for pname, slices in slice_stats_by_period.items():
        for s in slices:
            k = (s.get("group_id"), s.get("month"), s.get("hour"))
            if k not in pooled_gmh:
                pooled_gmh[k] = {**s, "slice": f"g{k[0]}_m{k[1]}_h{k[2]}", "periods": {}}
            pooled_gmh[k]["periods"][pname] = s

    top_under = sorted(
        [s for slist in slice_stats_by_period.values() for s in slist],
        key=lambda x: (-x["under_near_count"], -x["under_near_ratio"]),
    )[:20]
    top_over = sorted(
        [s for slist in slice_stats_by_period.values() for s in slist],
        key=lambda x: (-x["over_near_count"], -x["over_near_ratio"]),
    )[:20]

    candidates_50 = filter_candidate_slices(slice_stats_by_period, min_n=MIN_SAMPLES["medium"])
    candidates_24 = filter_candidate_slices(slice_stats_by_period, min_n=MIN_SAMPLES["exploratory"])
    top_candidates = candidates_50[:15] if candidates_50 else candidates_24[:15]

    print(f"=== Simulate top {len(top_candidates)} slices ===")
    sim_rows = []
    for cand in top_candidates:
        cand_sims = []
        for pname, df in frames.items():
            truth = long_to_wide_truth(df)
            for mult in SIM_MULTIPLIERS:
                r = simulate_slice_multiplier(
                    df,
                    truth,
                    int(cand["group_id"]),
                    int(cand["month"]),
                    int(cand["hour"]),
                    mult,
                    baselines[pname],
                )
                if r.get("valid"):
                    r["period"] = pname
                    r["slice"] = cand["slice"]
                    sim_rows.append(r)
                    cand_sims.append(r)
        cand["grade"] = grade_candidate(cand_sims, cand)

    sim_df = pd.DataFrame(sim_rows)
    if not sim_df.empty:
        sim_df.to_csv(OUTPUT_SIMS, index=False, encoding="utf-8-sig")

    grade_counts = {}
    for c in top_candidates:
        grade_counts[c.get("grade", "reject")] = grade_counts.get(c.get("grade", "reject"), 0) + 1

    robust = [c for c in top_candidates if c.get("grade") == "robust_candidate"]
    conditional = [c for c in top_candidates if c.get("grade") == "conditional_candidate"]

    best_sims = []
    if not sim_df.empty:
        best_sims = (
            sim_df.sort_values(["delta_score", "delta_ficr"], ascending=False)
            .head(20)
            .to_dict("records")
        )

    non_hotspot_top = [c for c in candidates_24 if not c["is_pipeline_hotspot"]][:10]
    hotspot_overlap = {
        "hotspot_slice_count": sum(1 for c in candidates_24 if c["is_pipeline_hotspot"]),
        "non_hotspot_top_count": len(non_hotspot_top),
        "non_hotspot_top": non_hotspot_top,
    }

    create_submission = "no"
    reasons = []
    if robust:
        create_submission = "conditional (robust slice simulation confirmed)"
        reasons.append(f"robust_candidate {len(robust)} found")
    elif conditional:
        create_submission = "no (conditional only, more validation needed)"
        reasons.append(f"conditional_candidate {len(conditional)} found")
    else:
        create_submission = "no"
        reasons.append(
            "no robust/conditional candidate; 1-3% single-slice mult did not improve Score"
        )

    overall = period_summaries["2024_full"]
    recommendation = {
        "create_submission": create_submission,
        "summary": (
            f"near-boundary under 비율 {overall['under_near_boundary_ratio']:.2%}, "
            f"robust={len(robust)}, conditional={len(conditional)}"
        ),
        "reasons": reasons,
    }

    row_cols = [
        "period", "key", "forecast_kst_dtm", "group_id", "month", "hour",
        "actual", "prediction", "capacity", "signed_error", "error_rate",
        "ficr_class", "boundary_band", "distance_to_nearest_ficr_boundary",
        "min_pred_delta_to_8pct", "min_multiplier_to_8pct",
        "is_under", "is_over", "ficr_success", "unit_price",
        "util_pred_bin", "util_actual_bin", "ws_bin", "is_pipeline_hotspot",
    ]
    all_rows[row_cols].to_csv(OUTPUT_ROWS, index=False, encoding="utf-8-sig")
    pd.DataFrame(top_candidates).to_csv(OUTPUT_SLICES, index=False, encoding="utf-8-sig")

    report = {
        "ficr_definition": FICR_DEFINITION,
        "config": V13_BEST_CONFIG,
        "period_key_validation": key_report,
        "baselines": baselines,
        "period_summaries": period_summaries,
        "overall_near_boundary": {
            "pooled_active": int(all_rows["active"].sum()),
            "near_boundary_failed_ratio": float(
                (all_rows["ficr_class"].isin(["under_near_boundary", "over_near_boundary"])).sum()
                / max(all_rows["active"].sum(), 1)
            ),
            "under_near_ratio": float(
                (all_rows["ficr_class"] == "under_near_boundary").sum()
                / max(all_rows["active"].sum(), 1)
            ),
            "over_near_ratio": float(
                (all_rows["ficr_class"] == "over_near_boundary").sum()
                / max(all_rows["active"].sum(), 1)
            ),
        },
        "top_under_slices": top_under,
        "top_over_slices": top_over,
        "top_candidate_slices": top_candidates,
        "pattern_slices": pattern_slices,
        "hotspot_overlap": hotspot_overlap,
        "best_simulations": best_sims,
        "grade_counts": grade_counts,
        "recommendation": recommendation,
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_markdown(report)

    print("\n=== 2024 Full Summary ===")
    s = period_summaries["2024_full"]
    print(f"  failed ratio: {s['failed_ratio']:.2%}")
    print(f"  under near: {s['under_near_boundary_ratio']:.2%}")
    print(f"  over near: {s['over_near_boundary_ratio']:.2%}")
    print(f"  robust candidates: {len(robust)}")
    print(f"  Recommendation: {create_submission}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
