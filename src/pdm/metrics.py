"""False Discovery Rate, threshold selection and bootstrap intervals.

An alarm is raised when the predicted failure probability is >= the threshold.
FDR = FP / (FP + TP): of the alarms raised, the share that were false. With no alarms
it is undefined and returned as None, never 0.
"""

import numpy as np
import pandas as pd

from pdm.config import FDR_CAP, N_BOOTSTRAP, SEED


def fdr(tp: int, fp: int) -> float | None:
    alarms = tp + fp
    return None if alarms == 0 else fp / alarms


def cap_verdict(rate: float | None, ci95: list[float] | None, cap: float = FDR_CAP) -> str:
    """Whether a measured FDR supports the claim that the cap holds.

    "met" needs the whole 95% interval under the cap, not just the estimate. An estimate
    under the cap with an interval reaching above it is "not demonstrable".
    """
    if rate is None:
        return "no alarms"
    if rate > cap:
        return "not met"
    if ci95 is not None and ci95[1] <= cap:
        return "met"
    return "not demonstrable"


def classification_summary(y_true, proba, threshold: float) -> dict:
    """Confusion counts and rates for alarms raised at `threshold`."""
    y = np.asarray(y_true).astype(bool)
    alarm = np.asarray(proba) >= threshold
    tp = int((alarm & y).sum())
    fp = int((alarm & ~y).sum())
    fn = int((~alarm & y).sum())
    tn = int((~alarm & ~y).sum())
    rate = fdr(tp, fp)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "alarms": tp + fp,
        "fdr": rate,
        "precision": None if rate is None else 1 - rate,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "accuracy": (tp + tn) / len(y),
    }


def threshold_table(y_true, proba) -> pd.DataFrame:
    """TP, FP, FDR and recall at every distinct probability, highest threshold first."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    order = np.argsort(-p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    last_of_value = np.r_[np.diff(p_sorted) != 0, True]
    tp, fp, thresholds = tp[last_of_value], fp[last_of_value], p_sorted[last_of_value]
    positives = int(y.sum())
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "tp": tp,
            "fp": fp,
            "alarms": tp + fp,
            "fdr": fp / (tp + fp),
            "recall": tp / positives if positives else np.nan,
        }
    )


def select_threshold(y_true, proba, cap: float = FDR_CAP) -> float | None:
    """Threshold with the most failures caught among those with FDR <= cap.

    Ties go to the higher threshold, which has fewer false alarms. Returns None when no
    threshold meets the cap.
    """
    table = threshold_table(y_true, proba)
    valid = table[(table["fdr"] <= cap) & (table["tp"] > 0)]
    if valid.empty:
        return None
    return float(valid.loc[valid["tp"].idxmax(), "threshold"])


def bootstrap_by_group(y_true, alarm, groups, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """95% intervals for FDR and recall, resampling whole groups (tool runs).

    Rows from one tool run are not independent, so runs are resampled, not rows.
    Resamples that raise no alarms have no FDR and are left out of its interval.
    """
    frame = pd.DataFrame(
        {
            "g": np.asarray(groups),
            "y": np.asarray(y_true).astype(int),
            "a": np.asarray(alarm).astype(int),
        }
    )
    frame["tp"] = frame["y"] * frame["a"]
    frame["fp"] = (1 - frame["y"]) * frame["a"]
    frame["fn"] = frame["y"] * (1 - frame["a"])
    per_group = frame.groupby("g")[["tp", "fp", "fn"]].sum().to_numpy()

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_group), size=(n_boot, len(per_group)))
    tp, fp, fn = per_group[draws].sum(axis=1).T

    alarms = tp + fp
    positives = tp + fn
    fdr_draws = fp[alarms > 0] / alarms[alarms > 0]
    recall_draws = tp[positives > 0] / positives[positives > 0]

    def interval(values: np.ndarray) -> list[float] | None:
        if len(values) == 0:
            return None
        return [float(v) for v in np.percentile(values, [2.5, 97.5])]

    return {
        "fdr_ci95": interval(fdr_draws),
        "recall_ci95": interval(recall_draws),
        "n_groups": int(len(per_group)),
        "n_boot": int(n_boot),
        "share_resamples_without_alarms": float((alarms == 0).mean()),
    }
