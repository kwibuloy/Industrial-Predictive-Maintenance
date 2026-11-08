"""Exploratory analysis and the sensor lead-indicator correlation.

Run with `python -m pdm.eda`. The data profile uses every row. Anything that could steer
feature choices (correlations, distributions) uses development rows only.
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn.feature_selection import mutual_info_classif

from pdm import plotting
from pdm.config import METRICS_DIR, SEED, ensure_dirs
from pdm.data import STAGE2_TYPES, TARGET, TYPE_FLAGS, TYPE_LABELS, load_dataset
from pdm.features import FEATURE_LABELS, NUMERIC, SENSORS, add_features

TARGETS = [TARGET, *STAGE2_TYPES]
TARGET_LABELS = {TARGET: "Any failure", **TYPE_LABELS}


def profile(df: pd.DataFrame) -> dict:
    n_types = df[TYPE_FLAGS].sum(axis=1)
    failed = df[TARGET] == 1
    runs = df.groupby("tool_run")
    by_type = df.groupby("type")[TARGET].agg(["size", "sum", "mean"])
    combos = (
        df.loc[n_types > 1, TYPE_FLAGS]
        .apply(lambda r: "+".join(t.upper() for t in TYPE_FLAGS if r[t]), axis=1)
        .value_counts()
    )
    return {
        "rows": int(len(df)),
        "columns_raw": 14,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.drop(columns=["tool_run", "split", "fold"]).duplicated().sum()),
        "failures": int(failed.sum()),
        "failure_rate": float(failed.mean()),
        "failure_by_product_type": {
            t: {"rows": int(r["size"]), "failures": int(r["sum"]), "rate": float(r["mean"])}
            for t, r in by_type.iterrows()
        },
        "type_flag_counts": {t: int(df[t].sum()) for t in TYPE_FLAGS},
        "failures_with_no_type": int((failed & (n_types == 0)).sum()),
        "failures_with_multiple_types": int((failed & (n_types > 1)).sum()),
        "multiple_type_combinations": {k: int(v) for k, v in combos.items()},
        "rnf_rows_marked_healthy": int(((df["rnf"] == 1) & ~failed).sum()),
        "rnf_rows_marked_failed": int(((df["rnf"] == 1) & failed).sum()),
        "tool_runs": int(df["tool_run"].nunique()),
        "median_rows_per_run": float(runs.size().median()),
        "median_wear_at_run_end_min": float(runs["tool_wear_min"].max().median()),
        "runs_ending_in_tool_wear_failure": int(runs["twf"].last().sum()),
        "split_rows": {k: int(v) for k, v in df["split"].value_counts().items()},
        "split_failures": {k: int(v) for k, v in df.groupby("split")[TARGET].sum().items()},
    }


def correlation_table(dev: pd.DataFrame) -> pd.DataFrame:
    """Association of each numeric feature with each failure flag (development rows).

    Point-biserial measures a linear link, Spearman a monotonic one, and mutual
    information any dependence, including "fails at both extremes" patterns.
    """
    rows = []
    for target in TARGETS:
        y = dev[target].to_numpy()
        mi = mutual_info_classif(dev[NUMERIC], y, random_state=SEED)
        for feature, mi_value in zip(NUMERIC, mi, strict=True):
            x = dev[feature].to_numpy()
            rows.append(
                {
                    "feature": feature,
                    "kind": "sensor" if feature in SENSORS else "engineered",
                    "target": target,
                    "n_positive": int(y.sum()),
                    "point_biserial": float(pointbiserialr(y, x).statistic),
                    "spearman": float(spearmanr(x, y).statistic),
                    "mutual_info": float(mi_value),
                }
            )
    return pd.DataFrame(rows)


def lead_indicators(corr: pd.DataFrame) -> dict:
    """Strongest raw sensor per target, by mutual information and by |Spearman|."""
    sensors = corr[corr["kind"] == "sensor"]
    out = {}
    for target, group in sensors.groupby("target"):
        by_mi = group.sort_values("mutual_info", ascending=False)
        by_rho = group.reindex(group["spearman"].abs().sort_values(ascending=False).index)
        out[target] = {
            "n_positive": int(group["n_positive"].iloc[0]),
            "top_by_mutual_info": by_mi["feature"].iloc[0],
            "top_by_abs_spearman": by_rho["feature"].iloc[0],
            "ranking_by_mutual_info": by_mi["feature"].tolist(),
        }
    return out


def plot_failure_types(df: pd.DataFrame) -> None:
    counts = df.loc[df[TARGET] == 1, STAGE2_TYPES].sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh([TYPE_LABELS[t] for t in counts.index], counts.to_numpy(), height=0.55)
    for bar in bars:
        bar.set_color(plotting.BLUE)
    ax.bar_label(bars, padding=4, color=plotting.INK_SECONDARY)
    ax.set_title("Failures by type (a failure can have more than one)")
    ax.set_xlabel("Rows")
    ax.grid(axis="y", visible=False)
    plotting.save(fig, "eda_failure_types.png")


def plot_sensor_distributions(dev: pd.DataFrame) -> None:
    healthy, failed = dev[dev[TARGET] == 0], dev[dev[TARGET] == 1]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, feature in zip(axes.ravel(), NUMERIC, strict=True):
        bins = np.histogram_bin_edges(dev[feature], bins=30)
        ax.hist(
            healthy[feature],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=plotting.HEALTHY_COLOR,
            label=f"Healthy ({len(healthy):,})",
        )
        ax.hist(
            failed[feature],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=plotting.FAILED_COLOR,
            label=f"Failed ({len(failed):,})",
        )
        ax.set_title(FEATURE_LABELS[feature], fontsize=10)
        ax.set_yticks([])
        ax.grid(visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.suptitle(
        "Sensor readings for healthy and failed rows (development set, density)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    plotting.save(fig, "eda_sensor_distributions.png")


def plot_association(corr: pd.DataFrame, value: str, name: str, title: str, cmap, vmin, vmax):
    grid = corr.pivot(index="feature", columns="target", values=value).loc[NUMERIC, TARGETS]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(grid.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(TARGETS)), [TARGET_LABELS[t] for t in TARGETS])
    ax.set_yticks(range(len(NUMERIC)), [FEATURE_LABELS[f] for f in NUMERIC])
    ax.grid(visible=False)
    ax.spines[:].set_visible(False)
    span = max(abs(vmin), abs(vmax))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(v) > 0.6 * span else plotting.INK,
            )
    ax.axhline(len(SENSORS) - 0.5, color=plotting.SURFACE, linewidth=4)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8).outline.set_visible(False)
    plotting.save(fig, name)


def plot_tool_wear_runs(df: pd.DataFrame, n_rows: int = 420) -> None:
    head = df.head(n_rows)
    failed = head[head[TARGET] == 1]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(head["udi"], head["tool_wear_min"], color=plotting.BLUE, label="Tool wear")
    ax.scatter(
        failed["udi"],
        failed["tool_wear_min"],
        s=42,
        color=plotting.ORANGE,
        edgecolor=plotting.SURFACE,
        linewidth=1.5,
        zorder=3,
        label="Machine failure",
    )
    ax.set_title(f"Tool wear resets when the tool is changed (first {n_rows} rows)")
    ax.set_xlabel("Row (UDI)")
    ax.set_ylabel("Tool wear (min)")
    ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0, 1.02))
    plotting.save(fig, "eda_tool_wear_runs.png")


def main() -> None:
    matplotlib.use("Agg")
    plotting.apply_style()
    ensure_dirs()

    df = add_features(load_dataset())
    dev = df[df["split"] == "dev"]

    corr = correlation_table(dev)
    corr.to_csv(METRICS_DIR / "sensor_correlation.csv", index=False)
    summary = {"profile": profile(df), "lead_indicators": lead_indicators(corr)}
    (METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    plot_failure_types(df)
    plot_sensor_distributions(dev)
    plot_association(
        corr,
        "spearman",
        "eda_correlation_heatmap.png",
        "Spearman correlation with each failure type (development set)",
        plotting.DIVERGING,
        -0.2,
        0.2,
    )
    plot_association(
        corr,
        "mutual_info",
        "eda_mutual_information.png",
        "Mutual information with each failure type (development set)",
        plotting.SEQUENTIAL,
        0,
        float(corr["mutual_info"].max()),
    )
    plot_tool_wear_runs(df)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
