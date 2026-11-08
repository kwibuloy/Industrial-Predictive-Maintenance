"""Feature engineering and the preprocessing step shared by every model."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SENSORS = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]
ENGINEERED = ["temp_diff_k", "power_w", "wear_torque"]
NUMERIC = SENSORS + ENGINEERED
CATEGORICAL = ["type"]
FEATURES = CATEGORICAL + NUMERIC

FEATURE_LABELS = {
    "air_temp_k": "Air temperature",
    "process_temp_k": "Process temperature",
    "rot_speed_rpm": "Rotational speed",
    "torque_nm": "Torque",
    "tool_wear_min": "Tool wear",
    "temp_diff_k": "Temperature difference",
    "power_w": "Power",
    "wear_torque": "Wear x torque",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features computed from each row's own readings, so nothing leaks across rows."""
    out = df.copy()
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rot_speed_rpm"] * 2 * np.pi / 60
    out["wear_torque"] = out["tool_wear_min"] * out["torque_nm"]
    return out


def make_preprocessor(scale: bool, numeric: list[str] | None = None) -> ColumnTransformer:
    """One-hot encode product type; optionally standardise the numeric columns.

    It is fitted inside a Pipeline, so scaling statistics come from training rows only.
    """
    numeric = NUMERIC if numeric is None else numeric
    encoder = OneHotEncoder(
        categories=[["L", "M", "H"]], sparse_output=False, handle_unknown="ignore"
    )
    transformer = ColumnTransformer(
        [
            ("type", encoder, CATEGORICAL),
            ("num", StandardScaler() if scale else "passthrough", numeric),
        ],
        verbose_feature_names_out=False,
    )
    return transformer.set_output(transform="pandas")
