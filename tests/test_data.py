import numpy as np
import pandas as pd
import pytest

from pdm.data import RENAME, TARGET, load_raw, make_split
from pdm.features import FEATURES, NUMERIC, add_features, make_preprocessor


def synthetic_raw(n_runs: int = 30, rows_per_run: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_runs * rows_per_run
    wear = np.tile(np.arange(rows_per_run) * 3, n_runs)
    failure = (rng.random(n) < 0.15).astype(int)
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"L{i}" for i in range(n)],
            "Type": rng.choice(["L", "M", "H"], n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1.5, n),
            "Rotational speed [rpm]": rng.normal(1500, 180, n),
            "Torque [Nm]": rng.normal(40, 10, n),
            "Tool wear [min]": wear,
            "Machine failure": failure,
            "TWF": 0,
            "HDF": failure,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
    )


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "raw.csv"
    synthetic_raw().to_csv(path, index=False)
    return path


def test_load_raw_renames_columns_and_labels_tool_runs(raw_csv):
    df = load_raw(raw_csv)
    assert set(RENAME.values()) <= set(df.columns)
    assert not any("[" in c for c in df.columns)
    assert df["tool_run"].nunique() == 30
    assert (df.groupby("tool_run").size() == 20).all()


def test_load_raw_rejects_a_file_with_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    synthetic_raw().drop(columns=["Torque [Nm]"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Torque"):
        load_raw(path)


def test_split_keeps_each_tool_run_on_one_side(raw_csv):
    df = load_raw(raw_csv)
    split = make_split(df)
    assert (split.groupby("tool_run")["split"].nunique() == 1).all()
    assert (split.groupby("tool_run")["fold"].nunique() == 1).all()
    assert 0.1 < (split["split"] == "test").mean() < 0.3


def test_split_folds_cover_development_rows_only(raw_csv):
    split = make_split(load_raw(raw_csv))
    dev, test = split[split["split"] == "dev"], split[split["split"] == "test"]
    assert sorted(dev["fold"].unique()) == [0, 1, 2, 3, 4]
    assert (test["fold"] == -1).all()


def test_split_is_reproducible(raw_csv):
    df = load_raw(raw_csv)
    pd.testing.assert_frame_equal(make_split(df), make_split(df))


def test_engineered_features_use_only_the_row_itself(raw_csv):
    df = add_features(load_raw(raw_csv))
    row = df.iloc[7]
    assert row["temp_diff_k"] == pytest.approx(row["process_temp_k"] - row["air_temp_k"])
    assert row["power_w"] == pytest.approx(row["torque_nm"] * row["rot_speed_rpm"] * 2 * np.pi / 60)
    assert row["wear_torque"] == pytest.approx(row["tool_wear_min"] * row["torque_nm"])


def test_features_exclude_labels_and_identifiers():
    forbidden = {"udi", "product_id", "tool_run", TARGET, "twf", "hdf", "pwf", "osf", "rnf"}
    assert not forbidden & set(FEATURES)


def test_preprocessor_outputs_model_safe_names(raw_csv):
    df = add_features(load_raw(raw_csv))
    out = make_preprocessor(scale=True).fit_transform(df[FEATURES])
    assert list(out.columns) == ["type_L", "type_M", "type_H", *NUMERIC]
    assert out[NUMERIC].mean().abs().max() < 1e-9
