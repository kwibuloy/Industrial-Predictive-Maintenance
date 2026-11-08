"""Pick the stage 1 threshold under the FDR cap and write the test report.

Run with `python -m pdm.evaluate` after `pdm.train`. Thresholds and the winning model
are decided from out-of-fold development predictions. The test split is only scored.
"""

import json

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from pdm import plotting
from pdm.config import (
    FAMILIES,
    FAMILY_LABELS,
    FDR_CAP,
    METRICS_DIR,
    MODELS_DIR,
    OOF_STAGE1_CSV,
    ensure_dirs,
)
from pdm.data import STAGE2_TYPES, TARGET, TYPE_LABELS, load_dataset
from pdm.features import FEATURES, add_features
from pdm.metrics import (
    bootstrap_by_group,
    cap_verdict,
    classification_summary,
    select_threshold,
    threshold_table,
)

DEFAULT_THRESHOLD = 0.5


def evaluate_stage1(oof: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict]:
    y_dev, y_test = oof[TARGET].to_numpy(), test[TARGET].to_numpy()
    results, test_proba = {}, {}
    for family in FAMILIES:
        threshold = select_threshold(y_dev, oof[family], FDR_CAP)
        model = joblib.load(MODELS_DIR / f"stage1_{family}.joblib")
        proba = model.predict_proba(test[FEATURES])[:, 1]
        test_proba[family] = proba

        entry = {
            "threshold": threshold,
            "dev_oof_default": classification_summary(y_dev, oof[family], DEFAULT_THRESHOLD),
            "test_default": classification_summary(y_test, proba, DEFAULT_THRESHOLD),
            "dev_oof_tuned": None,
            "test_tuned": None,
        }
        if threshold is not None:
            entry["dev_oof_tuned"] = classification_summary(y_dev, oof[family], threshold)
            entry["test_tuned"] = classification_summary(y_test, proba, threshold)
            bootstrap = bootstrap_by_group(y_test, proba >= threshold, test["tool_run"])
            entry["test_tuned"]["bootstrap"] = bootstrap
            entry["test_tuned"]["cap_verdict"] = cap_verdict(
                entry["test_tuned"]["fdr"], bootstrap["fdr_ci95"]
            )
        results[family] = entry
    return results, test_proba


def recall_by_type(test: pd.DataFrame, alarm: np.ndarray) -> dict:
    """Of the test failures of each type, how many stage 1 flagged.

    Stage 2 can only name a type for rows stage 1 flags, so this is the ceiling on
    end-to-end recall for each failure type.
    """
    failed = test[TARGET].to_numpy() == 1
    groups = {t: test[t].to_numpy() == 1 for t in STAGE2_TYPES}
    groups["untyped"] = test[STAGE2_TYPES].sum(axis=1).to_numpy() == 0
    out = {}
    for name, member in groups.items():
        rows = failed & member
        caught = int((rows & alarm).sum())
        out[name] = {
            "failures": int(rows.sum()),
            "caught": caught,
            "missed": int(rows.sum()) - caught,
            "recall": caught / int(rows.sum()) if rows.any() else None,
        }
    return out


def check_artefacts_match_split(df: pd.DataFrame, oof: pd.DataFrame) -> None:
    """Stop if the saved predictions were made under a different split than the current one."""
    dev_ids = set(df.loc[df["split"] == "dev", "udi"])
    if set(oof["udi"]) != dev_ids:
        raise RuntimeError(
            "Out-of-fold predictions do not match the current development split. "
            "Rerun `python -m pdm.train` before evaluating."
        )


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def pick_winner(results: dict) -> str | None:
    """Most development failures caught under the cap; ties go to the lower FDR."""
    eligible = [f for f in FAMILIES if results[f]["dev_oof_tuned"] is not None]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda f: (results[f]["dev_oof_tuned"]["recall"], -results[f]["dev_oof_tuned"]["fdr"]),
    )


def per_type_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        t: {
            "precision": float(precision[i]) if y_pred[:, i].sum() else None,
            "recall": float(recall[i]) if y_true[:, i].sum() else None,
            "actual": int(y_true[:, i].sum()),
            "predicted": int(y_pred[:, i].sum()),
        }
        for i, t in enumerate(STAGE2_TYPES)
    }


def evaluate_stage2(test: pd.DataFrame, alarm: np.ndarray, family: str) -> tuple[dict, np.ndarray]:
    """Score failure-type predictions on the test rows that stage 1 flagged."""
    model = joblib.load(MODELS_DIR / f"stage2_{family}.joblib")
    pred = np.zeros((len(test), len(STAGE2_TYPES)), dtype=int)
    if alarm.any():
        pred[alarm] = model.predict(test.loc[alarm, FEATURES])
    truth = test[STAGE2_TYPES].to_numpy()

    flagged_failures = alarm & (test[TARGET].to_numpy() == 1)
    typed_failures = (test[TARGET].to_numpy() == 1) & (truth.sum(axis=1) > 0)
    all_typed_pred = model.predict(test.loc[typed_failures, FEATURES])
    report = {
        "model": family,
        "flagged_rows": int(alarm.sum()),
        "flagged_true_failures": int(flagged_failures.sum()),
        "flagged_false_alarms": int((alarm & (test[TARGET].to_numpy() == 0)).sum()),
        "flagged_failures_without_type": int((flagged_failures & (truth.sum(axis=1) == 0)).sum()),
        "on_flagged_rows": per_type_scores(truth[alarm], pred[alarm]),
        "exact_match_on_flagged_true_failures": (
            float((pred[flagged_failures] == truth[flagged_failures]).all(axis=1).mean())
            if flagged_failures.any()
            else None
        ),
        "on_all_typed_test_failures": {
            "note": "Reference only: assumes stage 1 caught every failure.",
            "rows": int(typed_failures.sum()),
            "per_type": per_type_scores(truth[typed_failures], all_typed_pred),
        },
    }
    return report, pred


def plot_precision_recall(oof: pd.DataFrame, results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for family in FAMILIES:
        table = threshold_table(oof[TARGET], oof[family])
        color = plotting.FAMILY_COLORS[family]
        ax.plot(table["recall"], 1 - table["fdr"], color=color, label=FAMILY_LABELS[family])
        tuned = results[family]["dev_oof_tuned"]
        if tuned is not None:
            ax.scatter(
                tuned["recall"],
                tuned["precision"],
                s=70,
                color=color,
                zorder=3,
                edgecolor=plotting.SURFACE,
                linewidth=2,
            )
    ax.axhline(1 - FDR_CAP, color=plotting.INK_SECONDARY, linewidth=1)
    ax.text(
        0.32,
        1 - FDR_CAP - 0.012,
        "FDR cap: at most 5% of alarms false",
        va="top",
        color=plotting.INK_SECONDARY,
        fontsize=9,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall: share of real failures caught")
    ax.set_ylabel("Precision: share of alarms that are real (1 - FDR)")
    ax.set_title("Trade-off on development data; dots mark each chosen threshold")
    ax.legend(loc="lower left")
    plotting.save(fig, "eval_precision_recall.png")


def plot_threshold_tradeoff(oof: pd.DataFrame, family: str, threshold: float) -> None:
    table = threshold_table(oof[TARGET], oof[family]).sort_values("threshold")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(table["threshold"], table["recall"], color=plotting.BLUE, label="Recall")
    ax.plot(table["threshold"], table["fdr"], color=plotting.ORANGE, label="False Discovery Rate")
    ax.axhline(FDR_CAP, color=plotting.INK_SECONDARY, linewidth=1)
    ax.axvline(threshold, color=plotting.INK_SECONDARY, linewidth=1)
    ax.text(
        threshold,
        1.03,
        f"chosen threshold {threshold:.2f}",
        ha="center",
        color=plotting.INK_SECONDARY,
        fontsize=9,
    )
    ax.text(0.01, FDR_CAP + 0.015, "5% cap", color=plotting.INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Decision threshold on predicted failure probability")
    ax.set_ylabel("Rate")
    ax.set_title(
        f"{FAMILY_LABELS[family]}: raising the threshold trades recall for fewer false alarms"
    )
    ax.legend(loc="center", bbox_to_anchor=(0.45, 0.5))
    plotting.save(fig, "eval_threshold_tradeoff.png")


def plot_confusion(default: dict, tuned: dict, family: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    panels = [
        (f"Default threshold {DEFAULT_THRESHOLD:.2f}", default),
        (f"Tuned threshold {tuned['threshold']:.2f}", tuned),
    ]
    for ax, (title, s) in zip(axes, panels, strict=True):
        grid = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
        ax.imshow(np.log1p(grid), cmap=plotting.SEQUENTIAL, vmin=0, vmax=np.log1p(grid.max()) * 1.4)
        names = [["Correctly quiet", "False alarm"], ["Missed failure", "Failure caught"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i - 0.08,
                    f"{grid[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=plotting.INK,
                )
                ax.text(
                    j,
                    i + 0.22,
                    names[i][j],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=plotting.INK_SECONDARY,
                )
        ax.set_xticks([0, 1], ["No alarm", "Alarm"])
        ax.set_yticks([0, 1], ["Healthy", "Failed"])
        ax.set_xlabel("Model output")
        ax.grid(visible=False)
        ax.spines[:].set_visible(False)
        rate = "no alarms" if s["fdr"] is None else f"FDR {s['fdr']:.1%}"
        ax.set_title(f"{title}\n{rate}, recall {s['recall']:.1%}", fontsize=10)
    axes[0].set_ylabel("Actual")
    fig.suptitle(f"{FAMILY_LABELS[family]} on the test set", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, "eval_confusion_matrices.png")


def main() -> None:
    matplotlib.use("Agg")
    plotting.apply_style()
    ensure_dirs()

    df = add_features(load_dataset())
    test = df[df["split"] == "test"].reset_index(drop=True)
    oof = pd.read_csv(OOF_STAGE1_CSV)
    training = json.loads((METRICS_DIR / "training.json").read_text())
    check_artefacts_match_split(df, oof)

    stage1, test_proba = evaluate_stage1(oof, test)
    winner = pick_winner(stage1)
    report = {
        "fdr_cap": FDR_CAP,
        "default_threshold": DEFAULT_THRESHOLD,
        "winner": winner,
        "note": (
            "The winner and every threshold were chosen on development data. Test results "
            "for the other models are shown for context and must not be used to switch model."
        ),
        "stage1": stage1,
        "stage1_recall_by_type": None,
        "stage2": None,
    }

    predictions = test[["udi", "tool_run", TARGET, *STAGE2_TYPES]].copy()
    for family in FAMILIES:
        predictions[f"proba_{family}"] = test_proba[family]

    plot_precision_recall(oof, stage1)
    if winner is not None:
        threshold = stage1[winner]["threshold"]
        alarm = test_proba[winner] >= threshold
        stage2, type_pred = evaluate_stage2(test, alarm, training["stage2"]["selected"])
        report["stage2"] = stage2
        report["stage1_recall_by_type"] = recall_by_type(test, alarm)
        predictions["alarm"] = alarm.astype(int)
        for i, t in enumerate(STAGE2_TYPES):
            predictions[f"pred_{t}"] = type_pred[:, i]
        plot_threshold_tradeoff(oof, winner, threshold)
        plot_confusion(stage1[winner]["test_default"], stage1[winner]["test_tuned"], winner)

    (METRICS_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    predictions.to_csv(METRICS_DIR / "test_predictions.csv", index=False)

    print(f"winner: {winner}")
    for family in FAMILIES:
        r = stage1[family]
        if r["test_tuned"] is None:
            print(f"{family:9} no threshold meets the {FDR_CAP:.0%} cap on development data")
            continue
        t, d = r["test_tuned"], r["test_default"]
        low, high = t["bootstrap"]["fdr_ci95"] or (None, None)
        print(
            f"{family:9} thr {r['threshold']:.3f} | test tuned: FDR {fmt(t['fdr'])} "
            f"[{fmt(low)}, {fmt(high)}] cap {t['cap_verdict']}, recall {fmt(t['recall'])} "
            f"alarms {t['alarms']} | test 0.5: FDR {fmt(d['fdr'])} recall {fmt(d['recall'])} "
            f"alarms {d['alarms']}"
        )
    if report["stage1_recall_by_type"] is not None:
        for name, s in report["stage1_recall_by_type"].items():
            print(f"stage1 caught {s['caught']}/{s['failures']} {name}")
    if report["stage2"] is not None:
        for t, s in report["stage2"]["on_flagged_rows"].items():
            print(f"stage2 {TYPE_LABELS[t]:17} {s}")


if __name__ == "__main__":
    main()
