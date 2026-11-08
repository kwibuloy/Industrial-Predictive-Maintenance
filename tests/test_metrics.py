import numpy as np
import pytest

from pdm.metrics import (
    bootstrap_by_group,
    cap_verdict,
    classification_summary,
    fdr,
    select_threshold,
    threshold_table,
)

# Sorted by probability: labels are 1 1 1 0 1 0 0 0.
Y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
P = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])


def test_fdr_is_false_alarms_over_all_alarms():
    assert fdr(tp=19, fp=1) == pytest.approx(0.05)


def test_fdr_is_undefined_without_alarms():
    assert fdr(tp=0, fp=0) is None


def test_cap_verdict_needs_the_whole_interval_under_the_cap():
    assert cap_verdict(0.035, [0.0, 0.04], cap=0.05) == "met"
    assert cap_verdict(0.035, [0.0, 0.08], cap=0.05) == "not demonstrable"
    assert cap_verdict(0.035, None, cap=0.05) == "not demonstrable"
    assert cap_verdict(0.07, [0.0, 0.12], cap=0.05) == "not met"
    assert cap_verdict(None, None, cap=0.05) == "no alarms"


def test_summary_counts_at_threshold():
    s = classification_summary(Y, P, threshold=0.5)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (4, 1, 0, 3)
    assert s["alarms"] == 5
    assert s["fdr"] == pytest.approx(1 / 5)
    assert s["precision"] == pytest.approx(4 / 5)
    assert s["recall"] == pytest.approx(1.0)
    assert s["accuracy"] == pytest.approx(7 / 8)


def test_summary_reports_no_alarms_as_undefined_not_zero():
    s = classification_summary(Y, P, threshold=0.95)
    assert s["alarms"] == 0
    assert s["fdr"] is None
    assert s["precision"] is None
    assert s["recall"] == 0


def test_threshold_table_matches_hand_counts():
    table = threshold_table(Y, P).set_index("threshold")
    assert table.loc[0.7, ["tp", "fp"]].tolist() == [3, 0]
    assert table.loc[0.6, "fdr"] == pytest.approx(1 / 4)
    assert table.loc[0.5, "recall"] == pytest.approx(1.0)


def test_threshold_table_groups_tied_probabilities():
    table = threshold_table([1, 0, 1], [0.5, 0.5, 0.2])
    assert table["threshold"].tolist() == [0.5, 0.2]
    assert table["alarms"].tolist() == [2, 3]


def test_select_threshold_takes_most_recall_under_the_cap():
    # FDR is 0 down to 0.7 (3 caught) and 0.25 at 0.6, so a 5% cap stops at 0.7.
    assert select_threshold(Y, P, cap=0.05) == pytest.approx(0.7)
    # A 20% cap allows 0.5: 4 caught, 1 false alarm, FDR exactly 0.20.
    assert select_threshold(Y, P, cap=0.20) == pytest.approx(0.5)


def test_select_threshold_breaks_recall_ties_with_fewer_false_alarms():
    # 0.5 and 0.4 both catch 4 failures; 0.5 raises one false alarm fewer.
    assert select_threshold(Y, P, cap=0.40) == pytest.approx(0.5)


def test_select_threshold_returns_none_when_cap_cannot_be_met():
    assert select_threshold([0, 1, 1], [0.9, 0.5, 0.4], cap=0.05) is None


def test_bootstrap_interval_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 25)
    y = rng.random(1000) < 0.1
    alarm = (y & (rng.random(1000) < 0.7)) | (~y & (rng.random(1000) < 0.01))
    first = bootstrap_by_group(y, alarm, groups, n_boot=500, seed=1)
    second = bootstrap_by_group(y, alarm, groups, n_boot=500, seed=1)
    assert first == second
    point = (alarm & ~y).sum() / alarm.sum()
    low, high = first["fdr_ci95"]
    assert low <= point <= high
    assert first["n_groups"] == 40
