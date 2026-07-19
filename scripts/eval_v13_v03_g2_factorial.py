"""
v13 v03 blend × g2 hotspot 2×2 factorial 실험.

사용법:
  python scripts/eval_v13_v03_g2_factorial.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, SUBMISSION_DIR
from src.data_loader import load_labels
from scripts.eval_v13_refine import save_test
from scripts.v13_factorial_lib import (
    FACTORIAL_CONFIGS,
    FACTORIAL_PATHS,
    build_bundles,
    compute_factorial_effects_all,
    evaluate_all_periods,
    g2_hotspot_slice_analysis,
    g2_only_prediction_diff,
    recommend_candidate,
    submissions_identical,
    validate_period_keys,
    verify_sample_submission,
    verify_submission_values,
    v03_blend_effect_analysis,
)

OUTPUT_JSON = OUTPUT_DIR / "v13_v03_g2_factorial_eval.json"
OUTPUT_MD = OUTPUT_DIR / "v13_v03_g2_factorial_eval.md"

GENERATE_CONFIGS = {
    "B_v03_5_only": "v13_v0305_g2106_g3feb102.csv",
    "C_g2_105_only": "v13_v0307_g2105_g3feb102.csv",
}


def _score_table(period_data: dict) -> list[str]:
    lines = [
        "| config | Score | ΔScore | 1−NMAE | ΔNMAE | FICR | ΔFICR |",
        "|--------|-------|--------|--------|-------|------|-------|",
    ]
    for name in ["A_v13_best", "B_v03_5_only", "C_g2_105_only", "D_both"]:
        s = period_data["scores"][name]
        lines.append(
            f"| {name} | {s['score']:.6f} | {s['delta_score_vs_a']:+.6f} | "
            f"{s['1_minus_nmae']:.6f} | {s['delta_nmae_vs_a']:+.6f} | "
            f"{s['ficr']:.6f} | {s['delta_ficr_vs_a']:+.6f} |"
        )
    return lines


def _effects_table(effects: dict) -> list[str]:
    lines = [
        "| metric | v03 main | g2 main | interaction |",
        "|--------|----------|---------|-------------|",
    ]
    for metric in ("score", "1_minus_nmae", "ficr"):
        e = effects[metric]
        lines.append(
            f"| {metric} | {e['v03_main']:+.6f} | {e['g2_main']:+.6f} | {e['interaction']:+.6f} |"
        )
    return lines


def write_markdown(report: dict) -> None:
    rec = report["recommendation"]
    lines = [
        "# v13 v03 Blend × g2 Hotspot Factorial 평가",
        "",
        "## 1. 결론",
        "",
        f"- **추천**: `{rec['recommendation']}`",
        f"- **파일**: `{rec['file']}`",
        "",
    ]
    for r in rec["reasons"]:
        lines.append(f"- {r}")

    lines.extend(["", "## 2. 실험 설정과 파일 무결성", ""])
    lines.append("### 2×2 Factorial 설정")
    lines.append("")
    lines.append("| config | v03 blend | g2 hotspot | g3 2월 | 파일 |")
    lines.append("|--------|----------|------------|--------|------|")
    for name, cfg in FACTORIAL_CONFIGS.items():
        lines.append(
            f"| {name} | {cfg['blend_v03']:.2f} | {cfg['g2_mult']:.2f} | "
            f"{cfg['g3_season'][2]:.2f} | `{FACTORIAL_PATHS[name].name}` |"
        )

    lines.extend(["", "### Submission 무결성", ""])
    for name, check in report["integrity"]["submissions"].items():
        lines.append(f"**{name}**")
        for k, v in check.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.extend(["", "### 기간 정의", ""])
    for pname, bounds in report["period_key_validation"].items():
        lines.append(
            f"- **{pname}**: {bounds['min_ts']} ~ {bounds['max_ts']} "
            f"({bounds['n_rows']} rows, dup_ts={bounds['duplicate_ts']})"
        )

    period_sections = [
        ("2024_full", "## 3. 2024 전체 결과"),
        ("2024_h1", "## 4. 2024 H1 결과"),
        ("2024_h2", "## 5. 2024 H2 결과"),
        ("2023_h2", "## 6. 2023 H2 결과"),
    ]
    for pname, header in period_sections:
        lines.extend(["", header, ""])
        lines.append(f"*{report['eval']['periods'][pname]['definition']}*")
        lines.append("")
        lines.extend(_score_table(report["eval"]["periods"][pname]))

    lines.extend(["", "## 7. 그룹별 성능 (2024 전체)", ""])
    groups = report["eval"]["periods"]["2024_full"]["groups"]
    lines.append("| config | g1 proxy | g2 proxy | g3 proxy |")
    lines.append("|--------|----------|----------|----------|")
    ref_g = groups["A_v13_best"]
    for name in ["A_v13_best", "B_v03_5_only", "C_g2_105_only", "D_both"]:
        g = groups[name]
        d1 = g["g1"]["proxy_score"] - ref_g["g1"]["proxy_score"]
        d2 = g["g2"]["proxy_score"] - ref_g["g2"]["proxy_score"]
        d3 = g["g3"]["proxy_score"] - ref_g["g3"]["proxy_score"]
        lines.append(
            f"| {name} | {g['g1']['proxy_score']:.6f} ({d1:+.6f}) | "
            f"{g['g2']['proxy_score']:.6f} ({d2:+.6f}) | "
            f"{g['g3']['proxy_score']:.6f} ({d3:+.6f}) |"
        )

    lines.extend(["", "### 그룹별 NMAE / FICR (2024 전체)", ""])
    for gid in ("g1", "g2", "g3"):
        lines.append(f"**{gid}**")
        lines.append("| config | NMAE | ΔNMAE | FICR | ΔFICR |")
        lines.append("|--------|------|-------|------|-------|")
        for name in ["A_v13_best", "B_v03_5_only", "C_g2_105_only", "D_both"]:
            g = groups[name][gid]
            ref = ref_g[gid]
            lines.append(
                f"| {name} | {g['nmae']:.6f} | {g['nmae']-ref['nmae']:+.6f} | "
                f"{g['ficr']:.6f} | {g['ficr']-ref['ficr']:+.6f} |"
            )
        lines.append("")

    lines.extend(["", "## 8. g2 Hotspot Slice 분석 (2024 전체)", ""])
    for sl in report["g2_slices"]["slices"]:
        lines.append(f"### {sl['label']}")
        lines.append("| config | rows | pred_mean | actual_mean | bias | MAE | norm_MAE | under% | FICR |")
        lines.append("|--------|------|-----------|-------------|------|-----|----------|--------|------|")
        for cname in ["A_v13_best", "B_v03_5_only", "C_g2_105_only", "D_both"]:
            m = sl.get(cname, {})
            if m.get("row_count", 0) == 0:
                lines.append(f"| {cname} | 0 | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {cname} | {m['row_count']} | {m['mean_prediction']:.1f} | {m['mean_actual']:.1f} | "
                f"{m['mean_bias']:.1f} | {m['mae']:.1f} | {m['normalized_mae']:.4f} | "
                f"{m['under_prediction_ratio']:.1%} | {m['ficr']:.4f} |"
            )
        lines.append("")

    fx = report["factorial_effects"]
    lines.extend(["", "## 9. v03 Blend Main Effect", ""])
    for pname in ["2024_full", "2024_h1", "2024_h2", "2023_h2"]:
        e = fx[pname]["score"]["v03_main"]
        lines.append(f"- {pname}: Score {e:+.6f}, NMAE {fx[pname]['1_minus_nmae']['v03_main']:+.6f}, "
                     f"FICR {fx[pname]['ficr']['v03_main']:+.6f}")

    lines.extend(["", "## 10. g2 Hotspot Main Effect", ""])
    for pname in ["2024_full", "2024_h1", "2024_h2", "2023_h2"]:
        e = fx[pname]["score"]["g2_main"]
        lines.append(f"- {pname}: Score {e:+.6f}, NMAE {fx[pname]['1_minus_nmae']['g2_main']:+.6f}, "
                     f"FICR {fx[pname]['ficr']['g2_main']:+.6f}")

    lines.extend(["", "## 11. Interaction Effect", ""])
    for pname in ["2024_full", "2024_h1", "2024_h2", "2023_h2"]:
        e = fx[pname]["score"]["interaction"]
        lines.append(f"- {pname}: Score {e:+.6f}, NMAE {fx[pname]['1_minus_nmae']['interaction']:+.6f}, "
                     f"FICR {fx[pname]['ficr']['interaction']:+.6f}")

    lines.extend(["", "## 12. 기간별 안정성", ""])
    for cname in ["B_v03_5_only", "C_g2_105_only", "D_both"]:
        deltas = [
            report["eval"]["periods"][p]["scores"][cname]["delta_score_vs_a"]
            for p in ["2024_full", "2024_h1", "2024_h2", "2023_h2"]
        ]
        lines.append(f"- {cname}: ΔScore = [{', '.join(f'{d:+.6f}' for d in deltas)}]")

    lines.extend(["", "## 13. 최종 제출 추천", ""])
    lines.append(f"**{rec['recommendation']}** — `{rec['file']}`")
    for r in rec["reasons"]:
        lines.append(f"- {r}")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="v13 v03 x g2 factorial experiment")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip submission generation and A/D identity re-check",
    )
    args = parser.parse_args()

    labels = load_labels()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. Generate B, C submissions ===")
    generated = {}
    if args.eval_only:
        print("  (--eval-only: skip generation)")
        for cname, fname in GENERATE_CONFIGS.items():
            generated[cname] = str(FACTORIAL_PATHS[cname])
    else:
        for cname, fname in GENERATE_CONFIGS.items():
            path = save_test(FACTORIAL_CONFIGS[cname], labels, fname)
            generated[cname] = str(path)
            print(f"  {cname}: {path}")

    print("\n=== 2. Integrity checks ===")
    integrity = {"submissions": {}, "identity": {}}

    for cname, path in FACTORIAL_PATHS.items():
        p = Path(path)
        if not p.exists():
            integrity["submissions"][cname] = {"error": "file missing"}
            continue
        integrity["submissions"][cname] = {
            **verify_sample_submission(p),
            **verify_submission_values(p),
        }

    if args.eval_only:
        print("  (--eval-only: skip A/D pipeline identity check)")
    else:
        print("  Verifying A/D pipeline identity (temp files)...")
        tmp_a = save_test(FACTORIAL_CONFIGS["A_v13_best"], labels, "_tmp_factorial_a_check.csv")
        tmp_d = save_test(FACTORIAL_CONFIGS["D_both"], labels, "_tmp_factorial_d_check.csv")
        integrity["identity"]["A_pipeline_vs_v13_best"] = submissions_identical(
            tmp_a, FACTORIAL_PATHS["A_v13_best"]
        )
        integrity["identity"]["D_pipeline_vs_g3feb_keep"] = submissions_identical(
            tmp_d, FACTORIAL_PATHS["D_both"]
        )
        tmp_a.unlink(missing_ok=True)
        tmp_d.unlink(missing_ok=True)

    if FACTORIAL_PATHS["C_g2_105_only"].exists() and FACTORIAL_PATHS["A_v13_best"].exists():
        integrity["identity"]["C_vs_A_g2_only"] = g2_only_prediction_diff(
            FACTORIAL_PATHS["A_v13_best"], FACTORIAL_PATHS["C_g2_105_only"]
        )

    print("\n=== 3. Build validation bundles ===")
    bundles = build_bundles(labels)
    period_key_validation = validate_period_keys(bundles)
    print(f"  Period rows: { {k: v['n_rows'] for k, v in period_key_validation.items()} }")

    print("\n=== 4. Evaluate all periods ===")
    eval_results = evaluate_all_periods(bundles)
    factorial_effects = compute_factorial_effects_all(eval_results)

    print("\n=== 5. Group & slice analysis ===")
    g2_slices = g2_hotspot_slice_analysis(bundles["2024"])
    v03_effect = v03_blend_effect_analysis(bundles["2024"])

    recommendation = recommend_candidate(eval_results)

    report = {
        "configs": FACTORIAL_CONFIGS,
        "paths": {k: str(v) for k, v in FACTORIAL_PATHS.items()},
        "generated": generated,
        "integrity": integrity,
        "period_key_validation": period_key_validation,
        "eval": eval_results,
        "factorial_effects": factorial_effects,
        "g2_slices": g2_slices,
        "v03_blend_effect": v03_effect,
        "recommendation": recommendation,
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_markdown(report)

    print("\n=== 2024 Full Scores ===")
    for name in ["A_v13_best", "B_v03_5_only", "C_g2_105_only", "D_both"]:
        s = eval_results["periods"]["2024_full"]["scores"][name]
        print(
            f"  {name}: Score={s['score']:.6f}  1-NMAE={s['1_minus_nmae']:.6f}  "
            f"FICR={s['ficr']:.6f}  ΔScore={s['delta_score_vs_a']:+.6f}"
        )

    fx = factorial_effects["2024_full"]
    print("\n=== 2024 Full Factorial Effects ===")
    print(f"  v03 main (Score): {fx['score']['v03_main']:+.6f}")
    print(f"  g2 main  (Score): {fx['score']['g2_main']:+.6f}")
    print(f"  interaction    : {fx['score']['interaction']:+.6f}")

    print(f"\n=== Recommendation: {recommendation['recommendation']} ===")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
