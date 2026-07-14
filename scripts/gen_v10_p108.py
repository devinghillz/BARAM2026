"""P1_08 ws+util_tier 제출 CSV 생성."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_v10_priorities import (
    Q_V05, W_IDW, WS_FULL, apply_util_tier_boost, save_sub,
)
from src.calibration import apply_ws_band_multipliers
from src.config import VALID_START
from src.data_loader import load_labels
from src.power_curve import build_scada_monthly_curve
from scripts.train_v03 import blend_wide, build_dataset, get_feature_columns, predict_wide
from scripts.train_v04 import train_all_groups_qmap

labels = load_labels()
scada = build_scada_monthly_curve()
vs = pd.Timestamp(VALID_START)
train_near = build_dataset(labels, "nearest", vs, scada, "train")
train_idw = build_dataset(labels, "idw", vs, scada, "train")
test_near = build_dataset(labels, "nearest", None, scada, "test")
test_idw = build_dataset(labels, "idw", None, scada, "test")
feat = get_feature_columns(train_idw)
f5n = train_all_groups_qmap(train_near, feat, Q_V05)
f5i = train_all_groups_qmap(train_idw, feat, Q_V05)
raw = blend_wide(predict_wide(f5n, test_near, feat), predict_wide(f5i, test_idw, feat), W_IDW)
pred = apply_util_tier_boost(apply_ws_band_multipliers(raw, test_idw, WS_FULL))
save_sub(pred, "v10_P1_08_ws_util_tier.csv")
print("saved submissions/v10_P1_08_ws_util_tier.csv")
