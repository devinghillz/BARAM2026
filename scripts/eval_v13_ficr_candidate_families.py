"""
v13 FICR candidate family robustness validation (simulation only).

사용법:
  python scripts/eval_v13_ficr_candidate_families.py
  python scripts/eval_v13_ficr_candidate_families.py --eval-only
"""

from __future__ import annotations

import argparse
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
from scripts.ficr_boundary_lib import SIM_MULTIPLIERS, validate_period_keys
from scripts.ficr_candidate_families_lib import (
    PERIODS,
    analyze_hour_breakdown,
    block_bootstrap_candidate,
    build_candidate_specs,
    build_frames_and_baselines,
    evaluate_all_candidates,
    family_best,
    grade_family_candidate,
    leave_one_period_out,
    month_level_december_loo,
    top_candidate_configs,
)
from scripts.submission_diff_lib import V13_BEST_CONFIG

OUTPUT_JSON = OUTPUT_DIR / "v13_ficr_candidate_families.json"
OUTPUT_MD = OUTPUT_DIR / "v13_ficr_candidate_families.md"
OUTPUT_RESULTS = OUTPUT_DIR / "v13_ficr_candidate_family_results.csv"
OUTPUT_BOOTSTRAP = OUTPUT_DIR / "v13_ficr_candidate_bootstrap.csv"
OUTPUT_CROSS = OUTPUT_DIR / "v13_ficr_candidate_cross_period.csv"


def _interpretation_questions(report: dict) -> dict[str, str]:
    fam = report["family_best"]
    util = report["utilization_comparison"]
    grades = report["grades"]
    robust = [g for g in grades if g["grade"] == "robust_candidate"]

    a1 = fam.get("A")
    a2 = fam.get("A")
    b1 = fam.get("B")
    c1 = fam.get("C")
    d1 = next((g for g in grades if g["candidate_id"] == "D1"), None)
    d2 = next((g for g in grades if g["candidate_id"] == "D2"), None)

    q1 = "g2 12월 h9/h10은 2023 H2와 2024 H2에서 under-near가 반복되나, LOO/bootstrap 기준으로는 단독 robust 패턴으로 확정하기 어렵다."
    if a1 and a1["delta_score"] > 0:
        q1 = (
            f"A1(h9/h10)은 2024 full DeltaScore {a1['delta_score']:+.6f}로 개선되나 "
            f"cross-period/bootstrap 안정성은 {grades[0]['grade'] if grades else 'reject'} 수준이다."
        )

    q2 = "h5/h8 추가는 표본은 늘리지만 일반화 이득이 제한적이다."
    if a1 and a2:
        if a2["delta_score"] > a1["delta_score"] and a2.get("lopo_positive_folds", 0) >= a1.get("lopo_positive_folds", 0):
            q2 = "h5/h8 추가(A2/A4)가 A1 대비 Score/FICR 개선과 LOO 안정성 모두에서 우위를 보인다."
        elif a2["delta_score"] <= a1["delta_score"]:
            q2 = "h5/h8 추가는 표본은 늘리지만 2024 full Score 개선폭이 A1 대비 크지 않아 일반화 이득이 제한적이다."

    q3 = "h14/h15 포함(B family)은 over-correction(success->fail) 위험이 높다."
    if b1:
        if b1.get("success_to_fail", 0) > b1.get("fail_to_success", 0):
            q3 = f"h14/h15 포함(B1)에서 success->fail({b1['success_to_fail']}) > fail->success({b1['fail_to_success']})로 over-correction 징후가 있다."
        elif b1["delta_score"] > 0:
            q3 = "h14/h15 포함 시 Score 개선은 있으나 overfit_risk 후보로 분류되어 제출용으로는 보수적 접근이 필요하다."

    q4 = "utilization 조건은 변경 행 수를 줄이지만, 본 데이터에서는 NMAE 손실 완화 효과가 없고 FICR 이득만 감소하는 trade-off가 관측된다."
    if util.get("util_lt_0.3_vs_uniform") or util.get("util_lt_0.5_vs_uniform"):
        u = util.get("util_lt_0.3_vs_uniform") or util.get("util_lt_0.5_vs_uniform")
        if u.get("mean_nmae_loss_reduction", 0) > 0:
            q4 = (
                f"util 조건 적용 시 평균 Delta(1-NMAE) 손실이 {u['mean_nmae_loss_reduction']:.6f} 줄어 "
                f"NMAE 보호 효과가 확인된다 (FICR delta trade-off: {u.get('mean_ficr_tradeoff', 0):+.6f})."
            )
        else:
            q4 = (
                f"util 조건은 FICR 이득을 평균 {u.get('mean_ficr_tradeoff', 0):+.6f} 포기하는 대신 "
                f"NMAE 손실 완화는 관측되지 않았다 (mean_nmae_loss_reduction={u.get('mean_nmae_loss_reduction', 0):+.6f})."
            )

    q5 = "g1_m3_h4(C1)는 소표본이지만 2024 full에서 독립적 FICR 개선 신호가 있다."
    if c1:
        if c1["delta_score"] > 0 and c1["changed_rows"] < 50:
            q5 = (
                f"C1(g1_m3_h4)는 changed_rows={c1['changed_rows']}로 소표본이나 "
                f"DeltaScore {c1['delta_score']:+.6f}, DeltaFICR {c1['delta_ficr']:+.6f}로 "
                "독립 conditional 후보 가치는 있으나 robust 기준(50행) 미달."
            )
        elif c1["delta_score"] <= 0:
            q5 = "g1_m3_h4(C1)는 2024 full 기준 독립 유지 가치가 낮다."

    q6 = "g2 12월 + g1_m3 조합(D1/D2)은 개별 효과 합보다 우수하지 않다."
    if d1 and c1 and a1:
        sum_delta = a1["delta_score"] + c1["delta_score"]
        d_delta = max(d1.get("best_full", {}).get("delta_score", 0), 0)
        if d_delta > sum_delta:
            q6 = f"D1 조합 DeltaScore({d_delta:+.6f})가 개별 합({sum_delta:+.6f})보다 크다 — limited synergy."
        else:
            q6 = f"D1/D2 조합은 개별 효과 합({sum_delta:+.6f}) 대비 추가 이득이 없거나 상쇄된다."

    q7 = "robust_candidate가 없어 이번 단계에서 test submission 생성을 권장하지 않는다."
    if robust:
        q7 = f"robust_candidate {len(robust)}개 존재 — 다음 단계에서 submission 생성 검토 가능."

    return {
        "q1_g2_dec_h9_h10_reproducible": q1,
        "q2_h5_h8_generalization": q2,
        "q3_h14_h15_over_correction": q3,
        "q4_utilization_nmae": q4,
        "q5_g1_m3_h4_standalone": q5,
        "q6_combination_synergy": q6,
        "q7_submission_ready": q7,
    }


def write_markdown(report: dict) -> None:
    lines = [
        "# v13 FICR Candidate Family 검증",
        "",
        "## 1. 결론",
        "",
        f"- **robust_candidate**: {report['summary']['robust_count']}개",
        f"- **weak_candidate**: {report['summary']['weak_count']}개",
        f"- **overfit_candidate**: {report['summary']['overfit_count']}개",
        f"- **reject**: {report['summary']['reject_count']}개",
        f"- **submission 생성**: `{report['recommendation']['create_submission']}`",
        f"- **근거**: {report['recommendation']['summary']}",
        "",
        "## 2. 후보군 정의",
        "",
    ]
    for cid, desc in report["candidate_definitions"].items():
        lines.append(f"- **{cid}**: {desc}")

    lines.extend(["", "## 3. 전체 기간 결과", ""])
    lines.append("| candidate | mult | mode | DeltaScore | DeltaFICR | changed | net_trans | grade |")
    lines.append("|-----------|------|------|------------|-----------|---------|-----------|-------|")
    for g in sorted(report["grades"], key=lambda x: -x.get("best_full", {}).get("delta_score", 0)):
        bf = g.get("best_full") or {}
        lines.append(
            f"| {g['candidate_id']} | {bf.get('multiplier', '-')} | {bf.get('apply_mode', '-')} | "
            f"{bf.get('delta_score', 0):+.6f} | {bf.get('delta_ficr', 0):+.6f} | "
            f"{bf.get('changed_rows', 0)} | {bf.get('net_transition', 0)} | {g['grade']} |"
        )

    for section, family in [
        ("## 4. g2 12월 오전 패턴", "A"),
        ("## 5. g2 12월 넓은 패턴", "B"),
        ("## 6. g1 독립 패턴", "C"),
        ("## 7. 제한적 결합 결과", "D"),
    ]:
        lines.extend(["", section, ""])
        fb = report["family_best"].get(family)
        if fb:
            lines.append(
                f"- 최고: `{fb['candidate_id']}` mult={fb['multiplier']} mode={fb['apply_mode']} "
                f"DeltaScore={fb['delta_score']:+.6f} DeltaFICR={fb['delta_ficr']:+.6f}"
            )
        if family == "B" and report.get("hour_breakdown_b"):
            lines.append("- h14/h15 방향성:")
            for hb in report["hour_breakdown_b"]:
                lines.append(
                    f"  - h{hb['hour']}: DeltaScore={hb['delta_score']:+.6f} "
                    f"fail->succ={hb['fail_to_success']} succ->fail={hb['success_to_fail']} "
                    f"over_ratio_change={hb['over_ratio_change']:+.3f}"
                )

    lines.extend(["", "## 8. Utilization 조건 효과", ""])
    for k, v in report["utilization_comparison"].items():
        lines.append(f"- **{k}**: {v}")

    lines.extend(["", "## 9. Leave-one-period-out", ""])
    lines.append("| fold | candidate | mode | sel_mult | sel_dS | eval_dS | eval_dFICR |")
    lines.append("|------|-----------|------|----------|--------|---------|------------|")
    for row in report["lopo"][:30]:
        lines.append(
            f"| {row['fold_id']} | {row['candidate_id']} | {row['apply_mode']} | "
            f"{row['selected_multiplier']} | {row['select_delta_score']:+.6f} | "
            f"{row['eval_delta_score']:+.6f} | {row['eval_delta_ficr']:+.6f} |"
        )

    lines.extend(["", "## 10. Month-level 검증", ""])
    for row in report["month_loo"]:
        lines.append(
            f"- {row['fold_id']} `{row['candidate_id']}` mult={row['selected_multiplier']} "
            f"sel_dec_dS={row['select_delta_score_dec']:+.6f} eval_dec_dS={row['eval_delta_score_dec']:+.6f}"
        )

    lines.extend(["", "## 11. Bootstrap 안정성", ""])
    lines.append("| candidate | mult | mode | dS mean | dS>0% | dFICR>0% |")
    lines.append("|-----------|------|------|---------|-------|----------|")
    for b in report["bootstrap"]:
        lines.append(
            f"| {b['candidate_id']} | {b['multiplier']} | {b['apply_mode']} | "
            f"{b['delta_score_mean']:+.6f} | {b['delta_score_positive_ratio']:.1%} | "
            f"{b['delta_ficr_positive_ratio']:.1%} |"
        )

    lines.extend(["", "## 12. 후보 등급", ""])
    for grade in ["robust_candidate", "weak_candidate", "overfit_candidate", "reject"]:
        ids = [g["candidate_id"] for g in report["grades"] if g["grade"] == grade]
        lines.append(f"- **{grade}**: {', '.join(ids) if ids else '(none)'}")

    lines.extend(["", "## 13. Submission 생성 여부", ""])
    lines.append(f"- **create_submission**: `{report['recommendation']['create_submission']}`")
    for r in report["recommendation"]["reasons"]:
        lines.append(f"- {r}")

    iq = report["interpretation"]
    lines.extend(["", "## 해석", ""])
    for k, v in iq.items():
        lines.append(f"- **{k}**: {v}")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="Skip bundle rebuild if cached")
    args = parser.parse_args()

    import sys as _sys
    try:
        _sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_candidate_specs()

    print("Building split bundles...")
    labels = load_labels()
    bundles = {name: build_split_bundle(labels, vs) for name, vs in SPLITS.items()}

    print("Building period frames (v13_best)...")
    frames, baselines = build_frames_and_baselines(bundles)
    key_report = validate_period_keys(frames)

    print("Evaluating candidate families...")
    all_results = evaluate_all_candidates(frames, baselines, specs)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_RESULTS, index=False)

    print("Leave-one-period-out...")
    lopo = leave_one_period_out(all_results, specs)
    lopo_df = pd.DataFrame(lopo)

    print("Month-level December LOO...")
    month_loo = month_level_december_loo(frames, specs)
    month_df = pd.DataFrame(month_loo) if month_loo else pd.DataFrame()

    print("Bootstrap (top 10 configs)...")
    top10 = top_candidate_configs(all_results, limit=10)
    bootstrap_rows = []
    boot_map: dict[str, dict] = {}
    df_full = frames["2024_full"]
    for cfg in top10:
        spec = specs[cfg["candidate_id"]]
        b = block_bootstrap_candidate(
            df_full,
            spec,
            specs,
            cfg["multiplier"],
            cfg["apply_mode"],
            cfg.get("util_threshold"),
            n_iter=500,
            seed=42,
        )
        if b.get("valid"):
            row = {
                "candidate_id": cfg["candidate_id"],
                "family": cfg["family"],
                "multiplier": cfg["multiplier"],
                "apply_mode": cfg["apply_mode"],
                **b,
            }
            bootstrap_rows.append(row)
            boot_map[f"{cfg['candidate_id']}|{cfg['multiplier']}|{cfg['apply_mode']}"] = b
    boot_df = pd.DataFrame(bootstrap_rows)
    if not boot_df.empty:
        boot_df.to_csv(OUTPUT_BOOTSTRAP, index=False)

    print("Grading candidates...")
    grades = []
    for cid, spec in specs.items():
        bf_rows = [r for r in all_results if r["candidate_id"] == cid and r["period"] == "2024_full"]
        best_full = max(bf_rows, key=lambda x: (x["delta_score"], x["delta_ficr"])) if bf_rows else None
        boot_key = (
            f"{cid}|{best_full['multiplier']}|{best_full['apply_mode']}" if best_full else None
        )
        boot = boot_map.get(boot_key)
        grade, _ = grade_family_candidate(cid, spec, all_results, lopo, boot)
        lopo_c = [x for x in lopo if x["candidate_id"] == cid and best_full and x["apply_mode"] == best_full["apply_mode"]]
        grades.append({
            "candidate_id": cid,
            "family": spec.family,
            "grade": grade,
            "best_full": best_full,
            "bootstrap": boot,
            "lopo_positive_folds": sum(1 for x in lopo_c if x["eval_delta_score"] > 0),
            "overfit_risk": spec.overfit_risk,
            "exploratory": spec.exploratory,
        })

    cross_rows = []
    for g in grades:
        bf = g.get("best_full")
        if not bf:
            continue
        for period in PERIODS:
            pr = next(
                (
                    r for r in all_results
                    if r["candidate_id"] == g["candidate_id"]
                    and r["period"] == period
                    and r["multiplier"] == bf["multiplier"]
                    and r["apply_mode"] == bf["apply_mode"]
                ),
                None,
            )
            if pr:
                cross_rows.append({
                    "candidate_id": g["candidate_id"],
                    "family": g["family"],
                    "grade": g["grade"],
                    "period": period,
                    "multiplier": bf["multiplier"],
                    "apply_mode": bf["apply_mode"],
                    **{k: pr[k] for k in [
                        "score", "delta_score", "1_minus_nmae", "delta_nmae",
                        "ficr", "delta_ficr", "changed_rows", "net_transition",
                        "fail_to_success", "success_to_fail",
                        "tier0_to_tier1", "tier0_to_tier2", "tier1_to_tier2", "tier2_to_tier1",
                    ]},
                })
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(OUTPUT_CROSS, index=False)

    family_best_map = {}
    for fam in ["A", "B", "C", "D"]:
        fb = family_best(all_results, fam)
        if fb:
            family_best_map[fam] = fb

    util_comp = {}
    g2_uniform = [
        r for r in all_results
        if r["family"] in {"A", "B"} and r["period"] == "2024_full" and r["apply_mode"] == "uniform"
    ]
    for mode, label in [("util_lt_0.3", "util_lt_0.3_vs_uniform"), ("util_lt_0.5", "util_lt_0.5_vs_uniform")]:
        adaptive = [r for r in all_results if r["family"] in {"A", "B"} and r["period"] == "2024_full" and r["apply_mode"] == mode]
        nmae_diffs = []
        ficr_diffs = []
        for ar in adaptive:
            ur = next(
                (u for u in g2_uniform if u["candidate_id"] == ar["candidate_id"] and u["multiplier"] == ar["multiplier"]),
                None,
            )
            if ur:
                nmae_diffs.append(ur["delta_nmae"] - ar["delta_nmae"])
                ficr_diffs.append(ur["delta_ficr"] - ar["delta_ficr"])
        if nmae_diffs:
            util_comp[label] = {
                "mean_nmae_loss_reduction": float(sum(nmae_diffs) / len(nmae_diffs)),
                "mean_ficr_tradeoff": float(sum(ficr_diffs) / len(ficr_diffs)),
                "n_pairs": len(nmae_diffs),
            }

    b_spec = specs["B1"]
    hour_bd = analyze_hour_breakdown(
        frames["2024_full"], b_spec, specs,
        family_best_map.get("B", {}).get("multiplier", 1.03),
        baselines["2024_full"],
    )

    robust_count = sum(1 for g in grades if g["grade"] == "robust_candidate")
    create_sub = "yes" if robust_count > 0 else "no"
    rec_reasons = []
    if robust_count == 0:
        rec_reasons.append("robust_candidate 기준(LOO 2+/bootstrap 70%+/50행+) 충족 후보 없음")
        rec_reasons.append("simulation-only 단계 — test submission CSV 미생성")
    else:
        rec_reasons.append(f"robust_candidate {robust_count}개 — 다음 단계 submission 생성 검토 가능")
        rec_reasons.append("이번 단계에서는 test submission CSV를 생성하지 않음")

    report = {
        "baseline": V13_BEST_CONFIG,
        "baseline_metrics": baselines,
        "period_key_validation": key_report,
        "candidate_definitions": {
            cid: (
                f"family={s.family} g={s.group_id} m={s.month} h={sorted(s.hours)} "
                f"components={s.component_ids} or={s.or_specs is not None}"
            )
            for cid, s in specs.items()
        },
        "summary": {
            "total_candidates": len(specs),
            "total_simulations": len(all_results),
            "robust_count": robust_count,
            "weak_count": sum(1 for g in grades if g["grade"] == "weak_candidate"),
            "overfit_count": sum(1 for g in grades if g["grade"] == "overfit_candidate"),
            "reject_count": sum(1 for g in grades if g["grade"] == "reject"),
        },
        "family_best": family_best_map,
        "grades": grades,
        "lopo": lopo,
        "month_loo": month_loo,
        "bootstrap": bootstrap_rows,
        "utilization_comparison": util_comp,
        "hour_breakdown_b": hour_bd,
        "recommendation": {
            "create_submission": create_sub,
            "summary": "conditional slice family validation complete (simulation only)",
            "reasons": rec_reasons,
        },
    }
    report["interpretation"] = _interpretation_questions(report)

    def _json_default(obj):
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(type(obj))

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, default=_json_default, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(report)

    print("\n=== v13 FICR Candidate Family Validation ===")
    print(f"Candidates: {len(specs)} | Simulations: {len(all_results)}")
    print(f"robust={robust_count} weak={report['summary']['weak_count']} "
          f"overfit={report['summary']['overfit_count']} reject={report['summary']['reject_count']}")
    for fam in ["A", "B", "C", "D"]:
        fb = family_best_map.get(fam)
        if fb:
            print(f"  Family {fam} best: {fb['candidate_id']} x{fb['multiplier']} dS={fb['delta_score']:+.6f}")
    print(f"create_submission: {create_sub}")
    print(f"Outputs: {OUTPUT_JSON.name}, {OUTPUT_MD.name}")


if __name__ == "__main__":
    main()
