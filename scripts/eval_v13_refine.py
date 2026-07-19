"""
v13 미세 조정 — v03 5~6% + g2×1.05(v12) + g1×1.04 유지.

사용법:
  python scripts/eval_v13_refine.py
"""

from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import GROUP_COLUMNS, OUTPUT_DIR, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.power_curve import build_scada_monthly_curve
from scripts.eval_v13_comprehensive import (
    SPLITS,
    apply_pipeline_fixed,
    build_split_bundle,
    config_label,
    score_cfg,
)
from scripts.train_v03 import blend_wide, build_dataset, get_feature_columns, predict_wide, train_all_groups
from scripts.train_v04 import train_all_groups_qmap

Q_WEIGHTS = {1: 0.6, 2: 0.6, 3: 0.8}
W_IDW_V05B = 0.9
W_IDW_V03 = 0.8
Q_V03 = 0.65

V13_LB = {
    "slot_key": "mild23",
    "g2_mult": 1.06,
    "g3_season": {2: 1.02, 8: 1.05, 11: 1.02},
    "g1_mult": 1.04,
    "conditional": False,
    "blend_v03": 0.07,
}

V12_G23 = {
    "g2_mult": 1.05,
    "g3_season": {2: 1.0, 8: 1.05, 11: 1.02},
}


def make_candidates() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for w in [0.05, 0.06, 0.07]:
        # A: v13 핫스팟 + blend만 변경
        c = copy.deepcopy(V13_LB)
        c["blend_v03"] = w
        out.append((f"A_v13hot_v03{int(w*100):02d}", c))

    for w in [0.05, 0.06, 0.07]:
        # B: v12 g2/g3 + g1×1.04 + mild23
        c = {
            "slot_key": "mild23",
            "g1_mult": 1.04,
            "conditional": False,
            "blend_v03": w,
            **V12_G23,
        }
        out.append((f"B_v12g23_g1_v03{int(w*100):02d}", c))

    for w in [0.05, 0.06]:
        # C: v12 g2/g3 + g1×1.04 + full slot (v12 LB 원본 slot)
        c = {
            "slot_key": "full",
            "g1_mult": 1.04,
            "conditional": False,
            "blend_v03": w,
            **V12_G23,
        }
        out.append((f"C_v12slot_g1_v03{int(w*100):02d}", c))

    return out


def save_test(cfg: dict, labels, fname: str) -> Path:
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)
    m5n = train_all_groups_qmap(train_near, feat, Q_WEIGHTS)
    m5i = train_all_groups_qmap(train_idw, feat, Q_WEIGHTS)
    raw = blend_wide(
        predict_wide(m5n, test_near, feat), predict_wide(m5i, test_idw, feat), W_IDW_V05B
    )
    m3n = train_all_groups(train_near, feat, Q_V03)
    m3i = train_all_groups(train_idw, feat, Q_V03)
    v03 = blend_wide(
        predict_wide(m3n, test_near, feat), predict_wide(m3i, test_idw, feat), W_IDW_V03
    )
    test_out = apply_pipeline_fixed(raw, v03, test_idw, cfg)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / fname
    sub = load_submission_template().drop(columns=GROUP_COLUMNS)
    sub.merge(test_out, on="forecast_kst_dtm", how="left").to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    labels = load_labels()
    bundles = {n: build_split_bundle(labels, vs) for n, vs in SPLITS.items()}

    print("=== v13 refine (v03 5~6%, g2×1.05 hybrid) ===\n")
    rows = []
    for name, cfg in make_candidates():
        s24 = score_cfg(bundles["2024"], cfg)
        s23 = score_cfg(bundles["2023h2"], cfg)
        rows.append({"name": name, "cfg": cfg, "2024": s24, "2023h2": s23})
        print(
            f"  {name}: score={s24['score']:.6f}  "
            f"NMAE={s24['1_minus_nmae']:.4f}  FICR={s24['ficr']:.4f}  "
            f"| 2023h2={s23['score']:.6f}"
        )

    # 1) 총점 최우선 (2023h2 -0.00015 이내)
    base23 = score_cfg(bundles["2023h2"], V13_LB)["score"]
    stable = [r for r in rows if r["2023h2"]["score"] >= base23 - 0.00015]
    if not stable:
        stable = rows
    best_score = max(stable, key=lambda r: (r["2024"]["score"], r["2024"]["ficr"]))

    # 2) FICR 우선 (총점 0.6198 이상 — v13 7% 대비 -0.0004 허용)
    score_floor = best_score["2024"]["score"] - 0.0004
    ficr_candidates = [r for r in stable if r["2024"]["score"] >= score_floor]
    best_ficr = max(ficr_candidates, key=lambda r: (r["2024"]["ficr"], r["2024"]["score"]))

    print(f"\n--- Best total: {best_score['name']} ---")
    print(f"  2024: {best_score['2024']['score']:.6f}  FICR={best_score['2024']['ficr']:.4f}")
    print(f"\n--- Best FICR:  {best_ficr['name']} ---")
    print(f"  2024: {best_ficr['2024']['score']:.6f}  FICR={best_ficr['2024']['ficr']:.4f}")

    p1 = save_test(best_score["cfg"], labels, f"v13_refine_{best_score['name']}.csv")
    p2 = save_test(best_ficr["cfg"], labels, f"v13_refine_{best_ficr['name']}.csv")
    # 제출용 짧은 이름
    save_test(best_score["cfg"], labels, "v13_refine_best.csv")
    if best_ficr["name"] != best_score["name"]:
        save_test(best_ficr["cfg"], labels, "v13_refine_ficr.csv")

    print(f"\nSubmit (total):  {SUBMISSION_DIR / 'v13_refine_best.csv'}")
    if best_ficr["name"] != best_score["name"]:
        print(f"Submit (FICR):   {SUBMISSION_DIR / 'v13_refine_ficr.csv'}")

    report = {
        "candidates": [
            {
                "name": r["name"],
                "score_2024": r["2024"]["score"],
                "nmae": r["2024"]["1_minus_nmae"],
                "ficr": r["2024"]["ficr"],
                "score_2023h2": r["2023h2"]["score"],
                "cfg": r["cfg"],
            }
            for r in sorted(rows, key=lambda x: x["2024"]["score"], reverse=True)
        ],
        "best_total": {"name": best_score["name"], "cfg": best_score["cfg"], "scores": best_score["2024"]},
        "best_ficr": {"name": best_ficr["name"], "cfg": best_ficr["cfg"], "scores": best_ficr["2024"]},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "v13_refine_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"Report: {OUTPUT_DIR / 'v13_refine_report.json'}")


if __name__ == "__main__":
    main()
