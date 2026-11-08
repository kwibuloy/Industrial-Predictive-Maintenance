"""Train stage 1 (failure detector) and stage 2 (failure-type classifier).

Run with `python -m pdm.train`. Only development rows are used. No decision threshold is
fixed here: this step saves out-of-fold probabilities, and `pdm.evaluate` picks the
threshold from them.
"""

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from pdm.config import (
    FAMILIES,
    FDR_CAP,
    METRICS_DIR,
    MODELS_DIR,
    N_FOLDS,
    OOF_STAGE1_CSV,
    SEED,
    ensure_dirs,
)
from pdm.data import STAGE2_TYPES, TARGET, load_dataset
from pdm.features import FEATURES, add_features, make_preprocessor
from pdm.metrics import classification_summary, select_threshold


def make_classifier(family: str, pos_weight: float = 1.0, seed: int = SEED):
    """`pos_weight` is how much more a missed failure costs than a false alarm in training."""
    if family == "logreg":
        return LogisticRegression(
            class_weight={0: 1.0, 1: pos_weight}, max_iter=5000, random_state=seed
        )
    if family == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
        )
    if family == "lightgbm":
        return LGBMClassifier(
            n_estimators=300,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
    raise ValueError(f"Unknown model family: {family}")


def make_stage1_pipeline(family: str, pos_weight: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(scale=family == "logreg")),
            ("model", make_classifier(family, pos_weight)),
        ]
    )


def make_stage2_pipeline(family: str) -> Pipeline:
    """One yes/no classifier per failure type, so a row can have several types."""
    return Pipeline(
        [
            ("prep", make_preprocessor(scale=family == "logreg")),
            ("model", MultiOutputClassifier(make_classifier(family))),
        ]
    )


def out_of_fold(frame: pd.DataFrame, make_pipeline, targets, predict) -> np.ndarray:
    """Predict every row with a model that never saw that row's fold (whole tool runs)."""
    folds = frame["fold"].to_numpy()
    y = frame[targets].to_numpy()
    out = np.zeros(y.shape, dtype=float)
    for k in np.unique(folds):
        train, held_out = folds != k, folds == k
        pipeline = make_pipeline().fit(frame.loc[train, FEATURES], y[train])
        out[held_out] = predict(pipeline, frame.loc[held_out, FEATURES])
    return out


def train_stage1(dev: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y = dev[TARGET].to_numpy()
    balanced = round(float((y == 0).sum() / (y == 1).sum()), 2)
    weight_grid = [1.0, 5.0, balanced]

    report = {"weight_grid": weight_grid, "models": {}}
    oof = dev[["udi", "tool_run", "fold", TARGET]].copy()

    for family in FAMILIES:
        trials = []
        for weight in weight_grid:
            proba = out_of_fold(
                dev,
                lambda family=family, weight=weight: make_stage1_pipeline(family, weight),
                TARGET,
                lambda pipe, X: pipe.predict_proba(X)[:, 1],
            )
            threshold = select_threshold(y, proba, FDR_CAP)
            at_cap = None if threshold is None else classification_summary(y, proba, threshold)
            trials.append({"pos_weight": weight, "at_cap": at_cap, "proba": proba})

        # Score = failures caught while at most 5% of alarms are false. Ties: fewer false
        # alarms, then the lighter weight.
        def score(trial):
            at_cap = trial["at_cap"]
            if at_cap is None:
                return (-1.0, 0.0, 0.0)
            return (at_cap["recall"], -at_cap["fdr"], -trial["pos_weight"])

        best = max(trials, key=score)
        oof[family] = best["proba"]
        report["models"][family] = {
            "selected_pos_weight": best["pos_weight"],
            "oof_at_cap": best["at_cap"],
            "trials": [{"pos_weight": t["pos_weight"], "oof_at_cap": t["at_cap"]} for t in trials],
        }

        final = make_stage1_pipeline(family, best["pos_weight"])
        final.fit(dev[FEATURES], y)
        joblib.dump(final, MODELS_DIR / f"stage1_{family}.joblib")

    return report, oof


def train_stage2(dev: pd.DataFrame) -> dict:
    """Train on development failures that have at least one of the four type flags."""
    typed = dev[(dev[TARGET] == 1) & (dev[STAGE2_TYPES].sum(axis=1) > 0)].reset_index(drop=True)
    y = typed[STAGE2_TYPES].to_numpy()

    report = {
        "training_rows": int(len(typed)),
        "positives_per_type": {t: int(typed[t].sum()) for t in STAGE2_TYPES},
        "models": {},
    }
    for family in FAMILIES:
        pred = out_of_fold(
            typed,
            lambda family=family: make_stage2_pipeline(family),
            STAGE2_TYPES,
            lambda pipe, X: pipe.predict(X),
        ).astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(
            y, pred, average=None, zero_division=0
        )
        report["models"][family] = {
            "oof_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "oof_exact_match": float((pred == y).all(axis=1).mean()),
            "oof_per_type": {
                t: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, t in enumerate(STAGE2_TYPES)
            },
        }
        final = make_stage2_pipeline(family).fit(typed[FEATURES], y)
        joblib.dump(final, MODELS_DIR / f"stage2_{family}.joblib")

    report["selected"] = max(FAMILIES, key=lambda f: report["models"][f]["oof_macro_f1"])
    return report


def main() -> None:
    ensure_dirs()
    df = add_features(load_dataset())
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    test = df[df["split"] == "test"]

    stage1, oof = train_stage1(dev)
    oof.to_csv(OOF_STAGE1_CSV, index=False)
    stage2 = train_stage2(dev)

    report = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "fdr_cap": FDR_CAP,
        "features": FEATURES,
        "split": {
            "dev": {
                "rows": int(len(dev)),
                "failures": int(dev[TARGET].sum()),
                "tool_runs": int(dev["tool_run"].nunique()),
            },
            "test": {
                "rows": int(len(test)),
                "failures": int(test[TARGET].sum()),
                "tool_runs": int(test["tool_run"].nunique()),
            },
        },
        "stage1": stage1,
        "stage2": stage2,
    }
    (METRICS_DIR / "training.json").write_text(json.dumps(report, indent=2))

    print(f"dev {report['split']['dev']}  test {report['split']['test']}")
    for family, result in stage1["models"].items():
        cap = result["oof_at_cap"]
        line = "cap not met" if cap is None else f"recall {cap['recall']:.3f} fdr {cap['fdr']:.3f}"
        print(f"stage1 {family:9} pos_weight {result['selected_pos_weight']:>6}  {line}")
    for family, result in stage2["models"].items():
        print(f"stage2 {family:9} oof macro F1 {result['oof_macro_f1']:.3f}")
    print(f"stage2 selected: {stage2['selected']}")


if __name__ == "__main__":
    main()
