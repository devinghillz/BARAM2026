"""신뢰 파이프라인(eval_local_all) 기반 케이스 재검증."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_slot_multipliers, apply_ws_band_multipliers
from src.config import GROUP_COLUMNS, OUTPUT_DIR, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.metrics import evaluate_submission
from src.power_curve import build_scada_monthly_curve
from scripts.eval_local_all import build_predictions
from scripts.train_v03 import blend_wide, build_dataset, get_feature_columns, predict_wide, train_all_groups
from scripts.train_v04 import train_all_groups_qmap

LB_V03, LB_V05B = 0.611, 0.615
COLS = GROUP_COLUMNS
WS_FULL = {1: 1.10, 2: 1.08, 3: 1.0}
WS_MILD = {1: 1.08, 2: 1.06, 3: 1.0}
SLOT_FULL = {1: 1.06, 2: 1.04, 3: 1.06}
SLOT_MILD = {1: 1.04, 2: 1.03, 3: 1.04}


def blend(a, b, alpha):
    out = a.copy()
    for c in COLS:
        out[c] = alpha * a[c] + (1 - alpha) * b[c]
    return out


def lb_blend(alpha):
    return alpha * LB_V03 + (1 - alpha) * LB_V05B


def save_submission(pred, name):
    sub = load_submission_template().drop(columns=COLS)
    sub.merge(pred, on="forecast_kst_dtm", how="left").to_csv(
        SUBMISSION_DIR / name, index=False, encoding="utf-8-sig"
    )


def main():
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)
    truth = labels.loc[labels.kst_dtm >= vs, ["kst_dtm", *COLS]]

    print("학습 (eval_local_all 동일)...")
    _, preds = build_predictions(labels, scada, vs)
    v03, v05b, raw = preds["v0.3"], preds["v05b"], preds["v05b_raw"]
    long_valid = build_dataset(labels, "idw", vs, scada, "train")
    long_valid = long_valid[long_valid.forecast_kst_dtm >= vs]

    cases = []

    def add(name, p, extra=None):
        s = evaluate_submission(truth, p, time_col="kst_dtm")
        row = {
            "case": name,
            "local": round(s["score"], 6),
            "nmae": round(s["1_minus_nmae"], 6),
            "ficr": round(s["ficr"], 6),
        }
        if extra:
            row.update(extra)
        cases.append(row)

    add("A_v05b_제출본", v05b, {"lb_actual": LB_V05B})
    add("B_v03_단독", v03, {"lb_actual": LB_V03})
    for a in [0.15, 0.20, 0.25, 0.30, 0.35]:
        add(f"C_blend_{int(a * 100)}pct_v03", blend(v03, v05b, a), {"lb_actual": round(lb_blend(a), 6)})
    add("D_mild_calib", apply_slot_multipliers(apply_ws_band_multipliers(raw, long_valid, WS_MILD), SLOT_MILD))
    add("E_fullws_mildslot", apply_slot_multipliers(apply_ws_band_multipliers(raw, long_valid, WS_FULL), SLOT_MILD))

    ranked = sorted(cases, key=lambda x: x["local"], reverse=True)
    print(f"\n{'케이스':<24} {'로컬':>7} {'NMAE':>7} {'FICR':>7} {'LB실측':>7}")
    print("-" * 58)
    for r in ranked:
        lb = r.get("lb_actual", "")
        print(f"{r['case']:<24} {r['local']:7.4f} {r['nmae']:7.4f} {r['ficr']:7.4f} {str(lb):>7}")

    # test CSV (full train)
    print("\n테스트 예측 생성...")
    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)
    f3n = train_all_groups(train_near, feat, 0.65)
    f3i = train_all_groups(train_idw, feat, 0.65)
    f5n = train_all_groups_qmap(train_near, feat, {1: 0.6, 2: 0.6, 3: 0.8})
    f5i = train_all_groups_qmap(train_idw, feat, {1: 0.6, 2: 0.6, 3: 0.8})
    t3 = blend_wide(predict_wide(f3n, test_near, feat), predict_wide(f3i, test_idw, feat), 0.8)
    raw_test = blend_wide(predict_wide(f5n, test_near, feat), predict_wide(f5i, test_idw, feat), 0.9)
    t5 = apply_slot_multipliers(apply_ws_band_multipliers(raw_test, test_idw, WS_FULL), SLOT_FULL)
    t_mild = apply_slot_multipliers(apply_ws_band_multipliers(raw_test, test_idw, WS_MILD), SLOT_MILD)
    t_mildslot = apply_slot_multipliers(apply_ws_band_multipliers(raw_test, test_idw, WS_FULL), SLOT_MILD)

    SUBMISSION_DIR.mkdir(exist_ok=True)
    save_submission(t5, "exp_A_v05b.csv")
    save_submission(blend(t3, t5, 0.15), "exp_C_blend15.csv")
    save_submission(blend(t3, t5, 0.20), "exp_C_blend20.csv")
    save_submission(t_mild, "exp_D_mild.csv")
    save_submission(t_mildslot, "exp_E_fullws_mildslot.csv")

    out = OUTPUT_DIR / "experiment_cases_v2_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ranked": ranked, "best": ranked[0]}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out}")
    print("제출 CSV: submissions/exp_*.csv")


if __name__ == "__main__":
    main()
