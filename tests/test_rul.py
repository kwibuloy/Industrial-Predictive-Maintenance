import numpy as np
import pandas as pd
import pytest

from pdm.rul import RUL_TARGET, AverageLifeBaseline, build_rul_frame


def two_and_a_half_runs() -> pd.DataFrame:
    # Run 0 ends at wear 9 in a tool wear failure, run 1 ends at 6, run 2 is cut off.
    wear = [0, 3, 6, 9, 0, 3, 6, 0, 3]
    n = len(wear)
    return pd.DataFrame(
        {
            "udi": np.arange(1, n + 1),
            "type": ["L"] * n,
            "air_temp_k": 300.0,
            "process_temp_k": 310.0,
            "rot_speed_rpm": 1500.0,
            "torque_nm": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "tool_wear_min": wear,
            "twf": [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "tool_run": [0, 0, 0, 0, 1, 1, 1, 2, 2],
        }
    )


def test_target_is_wear_left_until_the_run_ends():
    frame = build_rul_frame(two_and_a_half_runs())
    assert frame[RUL_TARGET].tolist() == [9, 6, 3, 0, 6, 3, 0]


def test_cut_off_last_run_is_excluded():
    frame = build_rul_frame(two_and_a_half_runs())
    assert sorted(frame["tool_run"].unique()) == [0, 1]


def test_run_end_in_failure_is_flagged_per_run():
    frame = build_rul_frame(two_and_a_half_runs())
    assert frame.groupby("tool_run")["run_ended_in_failure"].first().tolist() == [1, 0]


def test_run_history_features_never_look_ahead():
    frame = build_rul_frame(two_and_a_half_runs())
    # Row 2 of run 0: mean of torques 10 and 20 only. Run 1 restarts its own history.
    assert frame["run_mean_torque"].tolist()[:5] == pytest.approx([10, 15, 20, 25, 50])
    assert frame["run_rows_so_far"].tolist() == [0, 1, 2, 3, 0, 1, 2]


def test_baseline_predicts_average_life_minus_wear_and_never_negative():
    X = pd.DataFrame({"tool_wear_min": [0, 100, 200]})
    model = AverageLifeBaseline().fit(X, [210, 110, 10])
    assert model.mean_life_ == pytest.approx(210)
    assert model.predict(pd.DataFrame({"tool_wear_min": [50, 250]})).tolist() == [160, 0]
