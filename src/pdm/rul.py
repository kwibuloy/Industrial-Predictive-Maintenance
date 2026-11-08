"""Time-to-failure proxy: remaining tool life in minutes of wear.

Run with `python -m pdm.rul`. The dataset has no timestamps, so this is a proxy: for each
row, the target is how many more minutes of wear the tool accumulates before its run
ends, in failure or in replacement. The last run is cut off by the end of the file and
is excluded because its end is unknown.
"""

import json

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from pdm import plotting
from pdm.config import METRICS_DIR, MODELS_DIR, SEED, ensure_dirs
from pdm.data import load_dataset
from pdm.features import CATEGORICAL, NUMERIC, add_features, make_preprocessor

RUL_TARGET = "remaining_life_min"
RUN_FEATURES = ["run_mean_torque", "run_mean_power", "run_rows_so_far"]
RUL_NUMERIC = NUMERIC + RUN_FEATURES
RUL_FEATURES = CATEGORICAL + RUL_NUMERIC
LATE_LIFE_WEAR_MIN = 150

MODEL_LABELS = {
    "baseline": "Average tool life minus wear",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
MODEL_COLORS = {"baseline": plotting.BLUE, "xgboost": plotting.ORANGE, "lightgbm": plotting.AQUA}


def build_rul_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add the target and run-history features. History uses the current and earlier rows only."""
    out = add_features(df)
    out = out[out["tool_run"] != out["tool_run"].max()].copy()
    runs = out.groupby("tool_run")
    out["run_end_wear_min"] = runs["tool_wear_min"].transform("max")
    out[RUL_TARGET] = out["run_end_wear_min"] - out["tool_wear_min"]
    out["run_ended_in_failure"] = runs["twf"].transform("last")
    out["run_mean_torque"] = runs["torque_nm"].transform(lambda s: s.expanding().mean())
    out["run_mean_power"] = runs["power_w"].transform(lambda s: s.expanding().mean())
    out["run_rows_so_far"] = runs.cumcount()
    return out.reset_index(drop=True)


class AverageLifeBaseline:
    """Predict the average wear at which training tools ended, minus the wear so far."""

    def fit(self, X: pd.DataFrame, y) -> "AverageLifeBaseline":
        self.mean_life_ = float((X["tool_wear_min"] + np.asarray(y)).mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.mean_life_ - X["tool_wear_min"].to_numpy(), 0, None)


def make_regressor(name: str):
    if name == "baseline":
        return AverageLifeBaseline()
    prep = make_preprocessor(scale=False, numeric=RUL_NUMERIC)
    if name == "xgboost":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=SEED,
            n_jobs=4,
        )
    elif name == "lightgbm":
        model = LGBMRegressor(
            n_estimators=300,
            num_leaves=8,
            learning_rate=0.03,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=4,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown regressor: {name}")
    return Pipeline([("prep", prep), ("model", model)])


def scores(y_true: np.ndarray, y_pred: np.ndarray, wear: np.ndarray) -> dict:
    late = wear >= LATE_LIFE_WEAR_MIN
    return {
        "mae_min": float(mean_absolute_error(y_true, y_pred)),
        "rmse_min": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "late_life_mae_min": float(mean_absolute_error(y_true[late], y_pred[late])),
        "late_life_rows": int(late.sum()),
    }


def out_of_fold(dev: pd.DataFrame, name: str) -> np.ndarray:
    folds = dev["fold"].to_numpy()
    pred = np.zeros(len(dev))
    for k in np.unique(folds):
        train, held_out = folds != k, folds == k
        model = make_regressor(name).fit(dev.loc[train, RUL_FEATURES], dev.loc[train, RUL_TARGET])
        pred[held_out] = model.predict(dev.loc[held_out, RUL_FEATURES])
    return pred


def plot_predicted_vs_actual(test: pd.DataFrame, pred: np.ndarray, name: str) -> None:
    limit = float(max(test[RUL_TARGET].max(), pred.max())) * 1.03
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test[RUL_TARGET], pred, s=14, alpha=0.35, color=MODEL_COLORS[name], linewidth=0)
    ax.plot([0, limit], [0, limit], color=plotting.INK_SECONDARY, linewidth=1)
    ax.text(
        limit * 0.97,
        limit * 0.93,
        "perfect prediction",
        ha="right",
        fontsize=9,
        color=plotting.INK_SECONDARY,
    )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Actual remaining tool life (min of wear)")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{MODEL_LABELS[name]} on the test runs")
    plotting.save(fig, "rul_predicted_vs_actual.png")


def plot_error_by_wear(test: pd.DataFrame, preds: dict) -> None:
    edges = np.arange(0, 251, 25)
    centres = (edges[:-1] + edges[1:]) / 2
    bins = pd.cut(test["tool_wear_min"], edges, right=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, pred in preds.items():
        error = pd.Series(np.abs(test[RUL_TARGET].to_numpy() - pred)).groupby(bins.to_numpy())
        mae = error.mean().reindex(bins.cat.categories)
        ax.plot(
            centres,
            mae.to_numpy(),
            marker="o",
            markersize=6,
            color=MODEL_COLORS[name],
            label=MODEL_LABELS[name],
        )
    ax.set_ylim(0, None)
    ax.set_xlabel("Tool wear so far (min)")
    ax.set_ylabel("Mean absolute error (min)")
    ax.set_title("Prediction error across the tool's life (test runs)")
    ax.legend(loc="lower left")
    plotting.save(fig, "rul_error_by_wear.png")


def main() -> None:
    matplotlib.use("Agg")
    plotting.apply_style()
    ensure_dirs()

    frame = build_rul_frame(load_dataset())
    dev = frame[frame["split"] == "dev"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    ends = frame.groupby("tool_run")["run_end_wear_min"].first()

    report = {
        "target": "Minutes of wear left until the tool's run ends (failure or replacement)",
        "is_proxy": True,
        "late_life_wear_min": LATE_LIFE_WEAR_MIN,
        "features": RUL_FEATURES,
        "runs": {
            "used": int(frame["tool_run"].nunique()),
            "dev": int(dev["tool_run"].nunique()),
            "test": int(test["tool_run"].nunique()),
            "ended_in_tool_wear_failure": int(
                frame.groupby("tool_run")["run_ended_in_failure"].first().sum()
            ),
            "end_wear_min": {
                "min": float(ends.min()),
                "mean": float(ends.mean()),
                "max": float(ends.max()),
                "std": float(ends.std()),
            },
        },
        "models": {},
    }

    test_preds = {}
    for name in MODEL_LABELS:
        oof = out_of_fold(dev, name)
        model = make_regressor(name).fit(dev[RUL_FEATURES], dev[RUL_TARGET])
        test_preds[name] = model.predict(test[RUL_FEATURES])
        report["models"][name] = {
            "dev_oof": scores(dev[RUL_TARGET].to_numpy(), oof, dev["tool_wear_min"].to_numpy()),
            "test": scores(
                test[RUL_TARGET].to_numpy(), test_preds[name], test["tool_wear_min"].to_numpy()
            ),
        }
        if name != "baseline":
            joblib.dump(model, MODELS_DIR / f"rul_{name}.joblib")

    winner = min(MODEL_LABELS, key=lambda n: report["models"][n]["dev_oof"]["mae_min"])
    report["winner"] = winner
    (METRICS_DIR / "rul.json").write_text(json.dumps(report, indent=2))

    plot_predicted_vs_actual(test, test_preds[winner], winner)
    plot_error_by_wear(test, test_preds)

    print(f"runs used {report['runs']['used']}  end wear {report['runs']['end_wear_min']}")
    for name, result in report["models"].items():
        print(f"{name:9} dev {result['dev_oof']}\n{'':9} test {result['test']}")
    print(f"winner by development MAE: {winner}")


if __name__ == "__main__":
    main()
