"""
v14 FICR final candidate selection and single submission generation.

사용법:
  python scripts/select_and_generate_v14_ficr.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, SUBMISSION_DIR
from src.data_loader import load_labels
from scripts.eval_v13_comprehensive import SPLITS, build_split_bundle
from scripts.ficr_boundary_lib import build_all_period_frames
from scripts.ficr_candidate_families_lib import baseline_metrics_by_period
from scripts.submission_diff_lib import V13_BEST_CONFIG, run_analysis
from scripts.v14_ficr_selection_lib import (
    CONDITION_LABELS,
    MULTIPLIER,
    PRIOR_JSON,
    SUBMISSION_FILENAMES,
    V13_BEST_PATH,
    analyze_b1_hour_contributions,
    build_robust_ranking_table,
    decide_final_selection,
    evaluate_h14_safety,
    generate_submission,
    load_prior_results,
    passes_mandatory,
    rank_eligible_candidates,
    run_ablation_study,
    verify_submission_patch,
)

OUTPUT_JSON = OUTPUT_DIR / "v14_ficr_final_selection.json"
OUTPUT_MD = OUTPUT_DIR / "v14_ficr_final_selection.md"
OUTPUT_ABLATION = OUTPUT_DIR / "v14_ficr_final_ablation.csv"
OUTPUT_HOUR = OUTPUT_DIR / "v14_ficr_final_hour_contributions.csv"
OUTPUT_DIFF_JSON = OUTPUT_DIR / "v14_ficr_final_vs_v13_diff.json"
OUTPUT_DIFF_MD = OUTPUT_DIR / "v14_ficr_final_vs_v13_diff.md"


def write_markdown(report: dict) -> None:
    sel = report["final_selection"]
    lines = [
        "# v14 FICR Final Selection",
        "",
        "## 1. 결론",
        "",
        f"- **최종 선택**: `{sel['choice']}`",
        f"- **이유**: {sel['reason']}",
        f"- **submission 생성**: {report.get('submission_path', 'none')}",
        "",
        "## 2. Robust 후보 순위표",
        "",
    ]
    if report.get("ranking_table"):
        lines.append(report["ranking_markdown"])
    lines.extend(["", "## 3. B1 시간별 기여", ""])
    if report.get("hour_contributions"):
        hdf = pd.DataFrame(report["hour_contributions"])
        for hour in [9, 10, 14, 15]:
            sub = hdf[hdf["hour"] == hour]
            if sub.empty:
                continue
            lines.append(f"### g2_m12_h{hour}")
            for _, r in sub.iterrows():
                lines.append(
                    f"- {r['period']}: dS={r['delta_score']:+.6f} dFICR={r['delta_ficr']:+.6f} "
                    f"f2s={r['fail_to_success']} s2f={r['success_to_fail']}"
                )
    lines.extend(["", "## 4. B1 h14 안전성", ""])
    h14 = report.get("b1_h14_safety", {})
    lines.append(f"- passes: {h14.get('passes')}")
    for r in h14.get("reasons", []):
        lines.append(f"- {r}")
    lines.extend(["", "## 5. Ablation", ""])
    lines.append(report.get("ablation_markdown", ""))
    lines.extend(["", "## 6. Submission 무결성", ""])
    for k, v in (report.get("integrity") or {}).items():
        if k != "slice_changes":
            lines.append(f"- {k}: {v}")
    lines.extend(["", "## 7. Diff verdict", ""])
    diff = report.get("diff_analysis", {})
    if diff:
        v = diff.get("verdict", {})
        lines.append(f"- safety: {v.get('submission_safety_verdict')}")
        lines.append(f"- performance: {v.get('performance_recommendation')}")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _df_to_md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    cols = cols or list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c])[:12] for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading prior candidate family results...")
    prior = load_prior_results()

    ranking = build_robust_ranking_table(prior)
    eligible = rank_eligible_candidates(ranking)

    print("Building validation frames for ablation / hour decomposition...")
    labels = load_labels()
    bundles = {name: build_split_bundle(labels, vs) for name, vs in SPLITS.items()}
    frames = build_all_period_frames(bundles, V13_BEST_CONFIG)
    baselines = baseline_metrics_by_period(frames)

    hour_df = analyze_b1_hour_contributions(frames, baselines)
    hour_df.to_csv(OUTPUT_HOUR, index=False)
    h14_safety = evaluate_h14_safety(hour_df)

    print("Running ablation study (A1, B1, B1-h14, B1-h15, D1)...")
    ablation_df = run_ablation_study(frames, baselines, prior["results_df"])
    ablation_df.to_csv(OUTPUT_ABLATION, index=False)

    selection = decide_final_selection(eligible, h14_safety, ablation_df)
    choice = selection["choice"]

    submission_path = None
    integrity = None
    diff_analysis = None

    if choice != "KEEP_V13_BEST":
        print(f"Generating submission for {choice}...")
        submission_path = generate_submission(choice, V13_BEST_PATH)
        integrity = verify_submission_patch(V13_BEST_PATH, submission_path, choice)
        diff_analysis = run_analysis(
            str(V13_BEST_PATH),
            str(submission_path),
            candidate_config={**V13_BEST_CONFIG, "ficr_choice": choice},
        )
        OUTPUT_DIFF_JSON.write_text(
            json.dumps(diff_analysis, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_submission_diff.py"),
                "--base",
                str(V13_BEST_PATH),
                "--candidate",
                str(submission_path),
                "--output",
                str(OUTPUT_DIFF_JSON),
            ],
            check=False,
        )
        if OUTPUT_DIFF_JSON.exists():
            diff_analysis = json.loads(OUTPUT_DIFF_JSON.read_text(encoding="utf-8"))
        verdict = diff_analysis.get("verdict", {}) if diff_analysis else {}
        OUTPUT_DIFF_MD.write_text(
            f"# v14 vs v13_best diff\n\n- safety: {verdict.get('submission_safety_verdict')}\n"
            f"- performance: {verdict.get('performance_recommendation')}\n",
            encoding="utf-8",
        )
    else:
        print("KEEP_V13_BEST — no submission generated.")

    ranking_display = ranking.copy()
    mandatory_rows = []
    for _, row in ranking.iterrows():
        ok, fails = passes_mandatory(row)
        mandatory_rows.append({"candidate_id": row["candidate_id"], "pass": ok, "failures": fails})
    mandatory_summary = {r["candidate_id"]: {"pass": r["pass"], "failures": r["failures"]} for r in mandatory_rows}

    report = {
        "prior_source": str(PRIOR_JSON),
        "multiplier": MULTIPLIER,
        "ranking_table": ranking.to_dict(orient="records"),
        "eligible_table": eligible.to_dict(orient="records"),
        "mandatory_summary": mandatory_summary,
        "b1_h14_safety": h14_safety,
        "hour_contributions": hour_df.to_dict(orient="records"),
        "ablation": ablation_df.to_dict(orient="records"),
        "final_selection": selection,
        "submission_path": str(submission_path) if submission_path else None,
        "submission_filename": SUBMISSION_FILENAMES.get(choice),
        "integrity": integrity,
        "diff_analysis": {
            "verdict": diff_analysis.get("verdict") if diff_analysis else None,
            "overall": diff_analysis.get("overall") if diff_analysis else None,
        } if diff_analysis else None,
        "ranking_markdown": _df_to_md_table(
            ranking,
            ["candidate_id", "delta_score_2024_full", "cross_period_min", "bootstrap_p_delta_score_positive",
             "bootstrap_delta_score_p05", "net_transition", "changed_rows"],
        ),
        "ablation_markdown": _df_to_md_table(
            ablation_df[ablation_df["period"] == "2024_full"],
            ["candidate_id", "delta_score", "delta_ficr", "net_transition", "changed_rows"],
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    write_markdown(report)

    print("\n=== v14 FICR Final Selection ===")
    print(f"Choice: {choice}")
    print(f"Reason: {selection['reason']}")
    print(f"B1 h14 safety: {h14_safety.get('passes')}")
    if submission_path:
        print(f"Submission: {submission_path}")
        print(f"Changed rows: {integrity.get('changed_rows_total')}")
        if diff_analysis:
            print(f"Safety: {diff_analysis.get('verdict', {}).get('submission_safety_verdict')}")


if __name__ == "__main__":
    main()
