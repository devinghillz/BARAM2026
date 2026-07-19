"""
v13 hybrid 가설 분리: g3 2월 보정 유지 후보 생성·로컬 검증·3-way diff.

사용법:
  python scripts/eval_v13_g3feb_keep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import GROUP_COLUMNS, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template, load_submission_template
from src.metrics import evaluate_submission
from scripts.eval_v13_comprehensive import apply_pipeline_fixed, build_split_bundle, score_cfg
from scripts.eval_v13_refine import save_test
from scripts.submission_diff_lib import (
    G3FEB_KEEP_CONFIG,
    OLD_HYBRID_CONFIG,
    V12_CONFIG,
    V13_BEST_CONFIG,
    analyze_slice,
    merge_submissions,
    run_analysis,
    load_submission_wide,
    wide_to_long,
)

OUTPUT = ROOT / "outputs"
SPLITS = {
    "2024": pd.Timestamp(VALID_START),
    "2023h2": pd.Timestamp("2023-07-01 01:00:00"),
}
PERIODS_2024 = {
    "2024_h1": (pd.Timestamp("2024-01-01 01:00:00"), pd.Timestamp("2024-07-01 01:00:00")),
    "2024_h2": (pd.Timestamp("2024-07-01 01:00:00"), pd.Timestamp("2025-01-01 01:00:00")),
}

CONFIGS = {
    "v13_best": V13_BEST_CONFIG,
    "old_hybrid": OLD_HYBRID_CONFIG,
    "g3feb_keep": G3FEB_KEEP_CONFIG,
    "v12": V12_CONFIG,
}

PATHS = {
    "v13_best": SUBMISSION_DIR / "v13_best.csv",
    "old_hybrid": SUBMISSION_DIR / "v13_hybrid_g105_g1_v0305.csv",
    "g3feb_keep": SUBMISSION_DIR / "v13_hybrid_g105_g3feb102_v0305.csv",
}

B_VS_C_SLICES = [
    ("g3_m2_all", 3, {2}, None),
    ("g3_m2_h20", 3, {2}, {20}),
    ("g3_m2_h21", 3, {2}, {21}),
    ("g3_m2_h22", 3, {2}, {22}),
    ("g3_m2_h23", 3, {2}, {23}),
]


def score_period(bundle: dict, config: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    pred = apply_pipeline_fixed(bundle["raw"], bundle["v03"], bundle["long_val"], config)
    truth = bundle["truth"]
    mask = (truth["kst_dtm"] >= start) & (truth["kst_dtm"] < end)
    t = truth.loc[mask]
    p = pred.merge(t[["kst_dtm"]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
    p = p.drop(columns=["kst_dtm"])
    return evaluate_submission(t, p, time_col="kst_dtm")


def local_eval_all(labels: pd.DataFrame) -> dict:
    bundles = {name: build_split_bundle(labels, vs) for name, vs in SPLITS.items()}
    results: dict = {"splits": {}, "periods_2024": {}}

    for split_name, bundle in bundles.items():
        results["splits"][split_name] = {}
        for cfg_name, cfg in CONFIGS.items():
            if cfg_name == "v12" and split_name == "2023h2":
                pass
            s = score_cfg(bundle, cfg)
            results["splits"][split_name][cfg_name] = s

    b2024 = bundles["2024"]
    for pname, (start, end) in PERIODS_2024.items():
        results["periods_2024"][pname] = {}
        for cfg_name, cfg in CONFIGS.items():
            results["periods_2024"][pname][cfg_name] = score_period(b2024, cfg, start, end)

    return results


def delta_vs_ref(scores: dict, ref: str, key: str) -> dict:
    base = scores[ref]
    out = {}
    for name, s in scores.items():
        if name == ref:
            continue
        out[name] = {
            f"score_delta_vs_{ref}": s["score"] - base["score"],
            f"ficr_delta_vs_{ref}": s["ficr"] - base["ficr"],
            f"nmae_delta_vs_{ref}": s["1_minus_nmae"] - base["1_minus_nmae"],
        }
    return out


def b_vs_c_slices(old_path: Path, new_path: Path) -> list[dict]:
    merged = merge_submissions(load_submission_wide(old_path), load_submission_wide(new_path))
    rows = []
    for label, gid, months, hours in B_VS_C_SLICES:
        rows.append(analyze_slice(merged, gid, months, hours, label))
    return rows


def recommend_submission(eval_results: dict, b_vs_c: list[dict]) -> dict:
    s24 = eval_results["splits"]["2024"]
    s23 = eval_results["splits"]["2023h2"]
    ref = s24["v13_best"]
    new = s24["g3feb_keep"]
    old = s24["old_hybrid"]

    reasons = []
    score_drop = ref["score"] - new["score"]
    ficr_vs_old = new["ficr"] - old["ficr"]
    ficr_vs_ref = new["ficr"] - ref["ficr"]
    nmae_vs_ref = new["1_minus_nmae"] - ref["1_minus_nmae"]
    ficr_23 = new["ficr"] - s23["v13_best"]["ficr"]

    g3_m2 = next(x for x in b_vs_c if x.get("label") == "g3_m2_all")
    g3_h22 = next(x for x in b_vs_c if x.get("label") == "g3_m2_h22")

    verdict = "keep_v13_best"
    if (
        score_drop <= 0.0003
        and ficr_vs_old >= 0
        and ficr_23 >= -0.0001
        and g3_m2.get("mean_delta", 0) >= old.get("mean_delta", -999)  # less decrease vs old hybrid
    ):
        verdict = "submit_g3feb_keep"
        reasons.append("2024 총점 하락 ≤0.0003, FICR old hybrid 대비 개선/동일, g3 2월 보정 유지")
    elif ficr_vs_old < 0 and score_drop > 0.0003:
        verdict = "submit_old_hybrid"
        reasons.append("g3feb_keep 총점/FICR trade-off 불리, old hybrid 우세")
    else:
        reasons.append(f"2024 score drop vs v13_best: {score_drop:.6f}")
        reasons.append(f"FICR vs old hybrid: {ficr_vs_old:+.6f}, vs v13_best: {ficr_vs_ref:+.6f}")
        reasons.append(f"NMAE vs v13_best: {nmae_vs_ref:+.6f}")
        if score_drop <= 0.0003 and ficr_vs_old > 0:
            verdict = "submit_g3feb_keep"
            reasons.append("총점 유지 + FICR old hybrid 개선 → g3feb_keep 권장")
        elif ficr_vs_ref > 0 and nmae_vs_ref > -0.001:
            verdict = "conditional_g3feb_keep"
            reasons.append("조건부: FICR 개선 있으나 총점 trade-off 확인 필요")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "metrics": {
            "score_drop_vs_v13_best": score_drop,
            "ficr_delta_vs_old_hybrid": ficr_vs_old,
            "ficr_delta_vs_v13_best": ficr_vs_ref,
            "nmae_delta_vs_v13_best": nmae_vs_ref,
            "g3_m2_mean_delta_b_vs_c": g3_h22.get("mean_delta"),
        },
    }


def write_main_report(
    eval_results: dict,
    rec: dict,
    ab: dict,
    ac: dict,
    bc: dict,
    b_vs_c: list[dict],
    pct_fix_note: str,
) -> None:
    s24 = eval_results["splits"]["2024"]
    s23 = eval_results["splits"]["2023h2"]

    def row(name):
        s = s24[name]
        d = s["score"] - s24["v13_best"]["score"]
        return f"| {name} | {s['score']:.6f} | {s['1_minus_nmae']:.4f} | {s['ficr']:.4f} | {d:+.6f} |"

    lines = [
        "# v13 Hybrid 가설 분리 실험",
        "",
        "## 1. 결론",
        "",
        f"- **추천**: `{rec['verdict']}`",
        "",
    ]
    for r in rec["reasons"]:
        lines.append(f"- {r}")

    lines.extend([
        "",
        "## 2. 비교 리포트 계산 오류 수정",
        "",
        pct_fix_note,
        "",
        "## 3. 신규 후보 설정",
        "",
        "| 설정 | v13_best | old hybrid | **g3feb_keep (신규)** |",
        "|------|----------|------------|----------------------|",
        "| v03 blend | 7% | 5% | 5% |",
        "| g2 hotspot | ×1.06 | ×1.05 | ×1.05 |",
        "| g3 2월 | ×1.02 | ×1.00 | **×1.02** |",
        "| g3 8·11월 | ×1.05/1.02 | 동일 | 동일 |",
        "| g1 | ×1.04 | ×1.04 | ×1.04 |",
        "| mild slot | ✓ | ✓ | ✓ |",
        "",
        f"신규 파일: `{PATHS['g3feb_keep']}`",
        "",
        "## 4. 2024 Hold-out 결과",
        "",
        "| config | Score | 1−NMAE | FICR | Δ vs v13_best |",
        "|--------|-------|--------|------|---------------|",
    ])
    for name in ["v13_best", "old_hybrid", "g3feb_keep", "v12"]:
        lines.append(row(name))

    lines.extend([
        "",
        "## 5. 2023 H2 결과",
        "",
        "| config | Score | 1−NMAE | FICR | Δ vs v13_best |",
        "|--------|-------|--------|------|---------------|",
    ])
    for name in ["v13_best", "old_hybrid", "g3feb_keep", "v12"]:
        s = s23[name]
        d = s["score"] - s23["v13_best"]["score"]
        lines.append(
            f"| {name} | {s['score']:.6f} | {s['1_minus_nmae']:.4f} | {s['ficr']:.4f} | {d:+.6f} |"
        )

    lines.extend(["", "## 6. 기간별 안정성 (2024)", ""])
    for pname in ["2024_h1", "2024_h2"]:
        lines.append(f"### {pname}")
        lines.append("| config | Score | FICR |")
        lines.append("|--------|-------|------|")
        for name in ["v13_best", "old_hybrid", "g3feb_keep"]:
            s = eval_results["periods_2024"][pname][name]
            lines.append(f"| {name} | {s['score']:.6f} | {s['ficr']:.4f} |")
        lines.append("")

    o_ac = ac["overall"]
    lines.extend([
        "## 7. v13 Best vs 신규 후보",
        "",
        f"- global mean change %: {o_ac['global_mean_change_pct']:.4f}%",
        f"- rowwise mean change %: {o_ac['rowwise_mean_change_pct']:.4f}%",
        f"- median |norm delta|: {o_ac['absolute_normalized_delta_median']:.6f}",
        "",
        "## 8. 기존 Hybrid vs 신규 후보",
        "",
        "차이는 **g3 2월 hotspot (×1.0 vs ×1.02)** 에 집중.",
        "",
        "| slice | rows | mean delta | mean norm delta | increase% | newly clip |",
        "|-------|------|------------|-----------------|-----------|------------|",
    ])
    for sl in b_vs_c:
        if sl.get("row_count", 0) == 0:
            continue
        lines.append(
            f"| {sl['label']} | {sl['row_count']} | {sl.get('mean_delta', 0):.2f} | "
            f"{sl.get('mean_normalized_delta', 0):.6f} | {sl.get('increase_ratio', 0):.1%} | "
            f"z={sl.get('newly_clipped_zero', 0)} cap={sl.get('newly_clipped_capacity', 0)} |"
        )

    lines.extend([
        "",
        "## 9. g3 2월 보정 효과",
        "",
        "old hybrid는 g3 2월 ×1.0으로 **예측을 하향** → g3feb_keep은 v13_best와 동일 ×1.02 유지.",
        "B vs C에서 g3 2월 slice mean delta가 양수면 g3feb_keep이 old hybrid보다 해당 구간 prediction이 높음.",
        "",
        "## 10. LB 제출 추천",
        "",
        f"**{rec['verdict']}** — 로컬 dual hold-out 기준.",
        "",
        "| 우선순위 | 파일 |",
        "|----------|------|",
        f"| 1 | `{PATHS['g3feb_keep']}` (가설 분리 후보) |" if rec["verdict"].startswith("submit_g3feb") else "",
        f"| 2 | `{PATHS['v13_best']}` (현재 LB 최고) |",
        "",
    ])
    (OUTPUT / "v13_hybrid_g3feb_keep_local_eval.md").write_text(
        "\n".join(l for l in lines if l is not None), encoding="utf-8"
    )


def write_b_vs_c_md(b_vs_c: list[dict], merged_non_feb_check: dict) -> None:
    lines = [
        "# Old Hybrid vs g3feb_keep (g3 2월 보정 차이)",
        "",
        "## Non-g3-Feb hotspot rows",
        "",
        f"- rows compared: {merged_non_feb_check['non_g3_feb_hotspot_rows']}",
        f"- max |delta|: {merged_non_feb_check['max_abs_delta_non_g3_feb']:.6e}",
        f"- identical: {merged_non_feb_check['identical_non_g3_feb']}",
        "",
        "## g3 February slices",
        "",
        "| slice | rows | mean p_base | mean p_cand | mean delta | norm delta | inc% | dec% |",
        "|-------|------|-------------|-------------|------------|------------|------|------|",
    ]
    for sl in b_vs_c:
        if sl.get("row_count", 0) == 0:
            continue
        lines.append(
            f"| {sl['label']} | {sl['row_count']} | {sl.get('mean_p_base', 0):.1f} | "
            f"{sl.get('mean_p_candidate', 0):.1f} | {sl.get('mean_delta', 0):.2f} | "
            f"{sl.get('mean_normalized_delta', 0):.6f} | {sl.get('increase_ratio', 0):.1%} | "
            f"{sl.get('decrease_ratio', 0):.1%} |"
        )
    (OUTPUT / "v13_old_hybrid_vs_g3feb_keep_diff.md").write_text("\n".join(lines), encoding="utf-8")


def check_b_vs_c_only_g3_feb(old_path: Path, new_path: Path) -> dict:
    from scripts.submission_diff_lib import HOTSPOT_G23

    merged = merge_submissions(load_submission_wide(old_path), load_submission_wide(new_path))
    feb_hs = (
        (merged["group_id"] == 3)
        & (merged["month"] == 2)
        & merged["hour"].isin(HOTSPOT_G23[3][2])
    )
    non_feb_hs = ~feb_hs
    sub = merged.loc[non_feb_hs]
    max_d = float(sub["delta"].abs().max()) if len(sub) else 0.0
    identical = bool((sub["delta"].abs() <= 1e-6).all()) if len(sub) else True
    return {
        "non_g3_feb_hotspot_rows": int(non_feb_hs.sum()),
        "max_abs_delta_non_g3_feb": max_d,
        "identical_non_g3_feb": identical,
        "g3_feb_hotspot_rows": int(feb_hs.sum()),
        "g3_feb_mean_delta": float(merged.loc[feb_hs, "delta"].mean()) if feb_hs.any() else 0.0,
    }


def verify_sample_submission(path: Path) -> dict:
    sample = load_submission_template()
    sub = pd.read_csv(path, encoding="utf-8-sig")
    return {
        "sample_rows": len(sample),
        "submission_rows": len(sub),
        "row_match": len(sample) == len(sub),
        "id_match": sample["forecast_id"].equals(sub["forecast_id"]),
        "columns_match": list(sample.columns) == list(sub.columns),
    }


def main() -> None:
    labels = load_labels()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=== 1. Generate g3feb_keep submission ===")
    out_path = save_test(G3FEB_KEEP_CONFIG, labels, PATHS["g3feb_keep"].name)
    print(f"  Saved: {out_path}")
    verify = verify_sample_submission(out_path)
    print(f"  Sample check: {verify}")

    print("\n=== 2. Local evaluation ===")
    eval_results = local_eval_all(labels)
    (OUTPUT / "v13_hybrid_g3feb_keep_local_eval.json").write_text(
        json.dumps(eval_results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print("\n=== 3. Submission diffs (fixed pct metrics) ===")
    ab = run_analysis(PATHS["v13_best"], PATHS["old_hybrid"], V13_BEST_CONFIG, OLD_HYBRID_CONFIG)
    ac = run_analysis(PATHS["v13_best"], PATHS["g3feb_keep"], V13_BEST_CONFIG, G3FEB_KEEP_CONFIG)
    bc_merged_check = check_b_vs_c_only_g3_feb(PATHS["old_hybrid"], PATHS["g3feb_keep"])
    b_vs_c = b_vs_c_slices(PATHS["old_hybrid"], PATHS["g3feb_keep"])

    (OUTPUT / "v13_best_vs_g3feb_keep_diff.json").write_text(
        json.dumps({k: v for k, v in ac.items() if k != "top_changes"}, indent=2, default=str),
        encoding="utf-8",
    )
    ac["top_changes"].to_csv(
        OUTPUT / "v13_best_vs_g3feb_keep_top_changes.csv", index=False, encoding="utf-8-sig"
    )

    from scripts.analyze_submission_diff import write_markdown_report

    write_markdown_report(
        ac,
        OUTPUT / "v13_best_vs_g3feb_keep_diff.md",
        OUTPUT / "v13_best_vs_g3feb_keep_top_changes.csv",
    )

    pct_note = (
        "기존 `mean_pct_change_vs_base`는 rowwise mean((c−b)/b) (base=0 제외)였음. "
        f"A vs old hybrid: global={ab['overall']['global_mean_change_pct']:.4f}%, "
        f"rowwise={ab['overall']['rowwise_mean_change_pct']:.4f}% "
        f"(제외 {ab['overall']['rowwise_excluded_zero_base_rows']}행). "
        "mean delta>0인데 rowwise가 음수: base≈0인 218행 제외 후 잔여 행의 상대변화율 평균이 "
        "전체 mean delta와 다른 가중치를 갖기 때문."
    )

    rec = recommend_submission(eval_results, b_vs_c)
    write_main_report(eval_results, rec, ab, ac, bc_merged_check, b_vs_c, pct_note)
    write_b_vs_c_md(b_vs_c, bc_merged_check)

    s24 = eval_results["splits"]["2024"]
    print("\n=== 2024 Scores ===")
    for name in ["v13_best", "old_hybrid", "g3feb_keep", "v12"]:
        s = s24[name]
        print(f"  {name}: {s['score']:.6f}  NMAE={s['1_minus_nmae']:.4f}  FICR={s['ficr']:.4f}")

    print(f"\n=== Recommendation: {rec['verdict']} ===")
    for r in rec["reasons"]:
        print(f"  - {r}")


if __name__ == "__main__":
    main()
