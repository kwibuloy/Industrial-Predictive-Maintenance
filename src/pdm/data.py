"""Load the AI4I 2020 CSV and build the tool-run split."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pdm.config import N_FOLDS, RAW_CSV, SEED, SPLIT_CSV, ensure_dirs

RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

TARGET = "machine_failure"
TYPE_FLAGS = ["twf", "hdf", "pwf", "osf", "rnf"]
# RNF is random by design and is not a stage 2 class (see docs/decisions.md).
STAGE2_TYPES = ["twf", "hdf", "pwf", "osf"]
TYPE_LABELS = {
    "twf": "Tool wear",
    "hdf": "Heat dissipation",
    "pwf": "Power",
    "osf": "Overstrain",
    "rnf": "Random",
}


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the CSV, rename columns to snake_case and label each tool run.

    Rows are in time order. Tool wear rises row by row and drops back when the tool is
    changed, so each drop starts a new `tool_run`. It is a grouping key, not a feature.
    """
    df = pd.read_csv(path)
    missing = sorted(set(RENAME) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    df = df.rename(columns=RENAME).sort_values("udi").reset_index(drop=True)
    df["tool_run"] = (df["tool_wear_min"].diff() < 0).cumsum().astype(int)
    return df


def make_split(df: pd.DataFrame, seed: int = SEED, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Assign whole tool runs to development (about 80%) or test, then to CV folds.

    Returns one row per input row with `split` ("dev" or "test") and `fold`
    (0..n_folds-1 for development rows, -1 for test rows).
    """
    y = df[TARGET].to_numpy()
    groups = df["tool_run"].to_numpy()

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(df, y, groups))

    split = np.full(len(df), "dev", dtype=object)
    split[test_idx] = "test"
    fold = np.full(len(df), -1)

    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(inner.split(dev_idx, y[dev_idx], groups[dev_idx])):
        fold[dev_idx[val_idx]] = k

    return pd.DataFrame(
        {"udi": df["udi"].to_numpy(), "tool_run": groups, "split": split, "fold": fold}
    )


def load_dataset(path: Path = RAW_CSV, split_path: Path = SPLIT_CSV) -> pd.DataFrame:
    """Raw data plus `split` and `fold`. The split is created once and then reused."""
    df = load_raw(path)
    if split_path.exists():
        split = pd.read_csv(split_path)
    else:
        ensure_dirs()
        split = make_split(df)
        split.to_csv(split_path, index=False)
    merged = df.merge(split[["udi", "split", "fold"]], on="udi", how="left", validate="1:1")
    if merged["split"].isna().any():
        raise ValueError(f"{split_path} does not cover every row; delete it to rebuild.")
    return merged
