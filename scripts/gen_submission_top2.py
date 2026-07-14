"""제출용 CSV 생성: blend30 + mild 보정."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_slot_multipliers, apply_ws_band_multipliers
from src.config import GROUP_COLUMNS, SUBMISSION_DIR, VALID_START
from src.data_loader import load_labels, load_submission_template
from src.power_curve import build_scada_monthly_curve
from scripts.train_v03 import blend_wide, build_dataset, get_feature_columns, predict_wide, train_all_groups
from scripts.train_v04 import train_all_groups_qmap

W_IDW_V03, W_IDW_V05 = 0.8, 0.9
Q_V03, Q_V05 = 0.65, {1: 0.6, 2: 0.6, 3: 0.8}
WS_FULL = {1: 1.10, 2: 1.08, 3: 1.0}
WS_MILD = {1: 1.08, 2: 1.06, 3: 1.0}
SLOT_FULL = {1: 1.06, 2: 1.04, 3: 1.06}
SLOT_MILD = {1: 1.04, 2: 1.03, 3: 1.04}


def blend_test(a, b, alpha):
    out = a.copy()
    for c in GROUP_COLUMNS:
        out[c] = alpha * a[c] + (1 - alpha) * b[c]
    return out


def train_and_predict_test():
    labels = load_labels()
    scada = build_scada_monthly_curve()
    vs = pd.Timestamp(VALID_START)

    train_near = build_dataset(labels, "nearest", vs, scada, "train")
    train_idw = build_dataset(labels, "idw", vs, scada, "train")
    test_near = build_dataset(labels, "nearest", None, scada, "test")
    test_idw = build_dataset(labels, "idw", None, scada, "test")
    feat = get_feature_columns(train_idw)

    print("v03 학습...")
    f3n = train_all_groups(train_near, feat, Q_V03)
    f3i = train_all_groups(train_idw, feat, Q_V03)
    t3 = blend_wide(predict_wide(f3n, test_near, feat), predict_wide(f3i, test_idw, feat), W_IDW_V03)

    print("v05b 학습...")
    f5n = train_all_groups_qmap(train_near, feat, Q_V05)
    f5i = train_all_groups_qmap(train_idw, feat, Q_V05)
    raw = blend_wide(predict_wide(f5n, test_near, feat), predict_wide(f5i, test_idw, feat), W_IDW_V05)

    t5_full = apply_slot_multipliers(apply_ws_band_multipliers(raw, test_idw, WS_FULL), SLOT_FULL)
    t5_mild = apply_slot_multipliers(apply_ws_band_multipliers(raw, test_idw, WS_MILD), SLOT_MILD)
    t_blend30 = blend_test(t3, t5_full, 0.30)
    t_blend20 = blend_test(t3, t5_full, 0.20)

    return {"blend30": t_blend30, "blend20": t_blend20, "mild": t5_mild, "v05b": t5_full}


def save(name, pred):
    sub = load_submission_template().drop(columns=GROUP_COLUMNS)
    sub = sub.merge(pred, on="forecast_kst_dtm", how="left")
    path = SUBMISSION_DIR / name
    SUBMISSION_DIR.mkdir(exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  -> {path}")


if __name__ == "__main__":
    preds = train_and_predict_test()
    print("저장:")
    save("v08_blend30_v03_v05b.csv", preds["blend30"])
    save("v08_mild_calib.csv", preds["mild"])
    save("v08_blend20_v03_v05b.csv", preds["blend20"])
