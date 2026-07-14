from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT_DIR / "train"
TEST_DIR = ROOT_DIR / "test"

PATHS = {
    "labels": TRAIN_DIR / "train_labels.csv",
    "ldaps_train": TRAIN_DIR / "ldaps_train.csv",
    "ldaps_test": TEST_DIR / "ldaps_test.csv",
    "gfs_train": TRAIN_DIR / "gfs_train.csv",
    "gfs_test": TEST_DIR / "gfs_test.csv",
    "scada_vestas": TRAIN_DIR / "scada_vestas_train.csv",
    "scada_unison": TRAIN_DIR / "scada_unison_train.csv",
    "submission": ROOT_DIR / "sample_submission.csv",
    "info": ROOT_DIR / "info.xlsx",
}

GROUP_COLUMNS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
GROUP_CAPACITY_KWH = {
    "kpx_group_1": 21_600,
    "kpx_group_2": 21_600,
    "kpx_group_3": 21_000,
}

# 로컬 검증: 2024년을 hold-out (시간 기준 split)
VALID_START = "2024-01-01 01:00:00"

# 기본 실험 출력 경로 (gitignore 대상)
OUTPUT_DIR = ROOT_DIR / "outputs"
SUBMISSION_DIR = ROOT_DIR / "submissions"
