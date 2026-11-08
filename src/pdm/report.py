"""Turn the saved metrics into tables and a written summary.

`python -m pdm.report` writes `reports/summary.md`. The notebook calls the same
functions, so every number shown anywhere comes from `reports/metrics/`.
"""

import json

import pandas as pd

from pdm.config import FAMILIES, FAMILY_LABELS, METRICS_DIR, ROOT
from pdm.data import STAGE2_TYPES, TARGET, TYPE_LABELS
from pdm.features import FEATURE_LABELS
from pdm.rul import MODEL_LABELS

SUMMARY_MD = ROOT / "reports" / "summary.md"
TARGET_LABELS = {TARGET: "Any failure", **TYPE_LABELS}


def load_metrics() -> dict:
    names = ("eda_summary", "training", "evaluation", "rul")
    metrics = {name: json.loads((METRICS_DIR / f"{name}.json").read_text()) for name in names}
    metrics["correlation"] = pd.read_csv(METRICS_DIR / "sensor_correlation.csv")
    return metrics


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}%}"


def interval(bounds: list[float] | None) -> str:
    return "n/a" if bounds is None else f"{bounds[0]:.1%} to {bounds[1]:.1%}"


def data_profile(m: dict) -> pd.DataFrame:
    p = m["eda_summary"]["profile"]
    rows = [
        ("Rows", f"{p['rows']:,}"),
        ("Missing values", p["missing_values"]),
        ("Machine failures", f"{p['failures']} ({pct(p['failure_rate'])})"),
        ("Failures with more than one type", p["failures_with_multiple_types"]),
        ("Failures with no type recorded", p["failures_with_no_type"]),
        ("Random-failure rows labelled healthy", p["rnf_rows_marked_healthy"]),
        ("Tool runs", p["tool_runs"]),
        ("Median wear when a run ends (min)", f"{p['median_wear_at_run_end_min']:.0f}"),
        (
            "Development rows / failures",
            f"{p['split_rows']['dev']:,} / {p['split_failures']['dev']}",
        ),
        ("Test rows / failures", f"{p['split_rows']['test']:,} / {p['split_failures']['test']}"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def lead_indicator_table(m: dict) -> pd.DataFrame:
    """Strongest raw sensor and strongest feature overall, for each failure type."""
    corr = m["correlation"]
    rows = []
    for target, label in TARGET_LABELS.items():
        group = corr[corr["target"] == target]
        if group.empty:
            continue
        sensor = group[group["kind"] == "sensor"].sort_values("mutual_info").iloc[-1]
        overall = group.sort_values("mutual_info").iloc[-1]
        rows.append(
            {
                "Failure": label,
                "Failures (dev)": int(sensor["n_positive"]),
                "Strongest sensor": FEATURE_LABELS[sensor["feature"]],
                "Its Spearman correlation": f"{sensor['spearman']:+.2f}",
                "Strongest feature overall": FEATURE_LABELS[overall["feature"]],
            }
        )
    return pd.DataFrame(rows)


def stage1_comparison(m: dict) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        r = m["evaluation"]["stage1"][family]
        dev, test = r["dev_oof_tuned"], r["test_tuned"]
        weight = m["training"]["stage1"]["models"][family]["selected_pos_weight"]
        if test is None:
            rows.append({"Model": FAMILY_LABELS[family], "Threshold": "cap not reachable"})
            continue
        rows.append(
            {
                "Model": FAMILY_LABELS[family],
                "Failure weight": weight,
                "Threshold": f"{r['threshold']:.3f}",
                "Dev recall": pct(dev["recall"]),
                "Dev FDR": pct(dev["fdr"]),
                "Test recall": pct(test["recall"]),
                "Test FDR": pct(test["fdr"]),
                "Test FDR 95% interval": interval(test["bootstrap"]["fdr_ci95"]),
                "5% cap on test": test["cap_verdict"],
                "Test alarms": test["alarms"],
            }
        )
    return pd.DataFrame(rows)


def tradeoff_table(m: dict) -> pd.DataFrame:
    """The winner on the test set: default 0.5 threshold next to the tuned one."""
    r = m["evaluation"]["stage1"][m["evaluation"]["winner"]]
    rows = []
    for label, s in (("Default 0.50", r["test_default"]), ("Tuned", r["test_tuned"])):
        rows.append(
            {
                "Threshold": label if label != "Tuned" else f"Tuned {s['threshold']:.2f}",
                "Alarms": s["alarms"],
                "False alarms": s["fp"],
                "Failures caught": s["tp"],
                "Failures missed": s["fn"],
                "FDR": pct(s["fdr"]),
                "Recall": pct(s["recall"]),
                "Accuracy": pct(s["accuracy"], 2),
            }
        )
    return pd.DataFrame(rows)


def stage2_table(m: dict) -> pd.DataFrame:
    scores = m["evaluation"]["stage2"]["on_flagged_rows"]
    return pd.DataFrame(
        [
            {
                "Failure type": TYPE_LABELS[t],
                "Actual": scores[t]["actual"],
                "Predicted": scores[t]["predicted"],
                "Precision": pct(scores[t]["precision"]),
                "Recall": pct(scores[t]["recall"]),
            }
            for t in STAGE2_TYPES
        ]
    )


def caught_by_type_table(m: dict) -> pd.DataFrame:
    """Stage 1 on the test set: failures of each type caught and missed."""
    labels = {**TYPE_LABELS, "untyped": "No type recorded"}
    return pd.DataFrame(
        [
            {
                "Failure type": labels[name],
                "Test failures": s["failures"],
                "Caught": s["caught"],
                "Missed": s["missed"],
                "Recall": pct(s["recall"]),
            }
            for name, s in m["evaluation"]["stage1_recall_by_type"].items()
        ]
    )


def rul_table(m: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": MODEL_LABELS[name],
                "Dev error (min)": f"{r['dev_oof']['mae_min']:.1f}",
                "Test error (min)": f"{r['test']['mae_min']:.1f}",
                "Test error, late life (min)": f"{r['test']['late_life_mae_min']:.1f}",
            }
            for name, r in m["rul"]["models"].items()
        ]
    )


def headline(m: dict) -> str:
    """The main result in plain sentences, built from the saved numbers."""
    ev = m["evaluation"]
    winner = ev["winner"]
    if winner is None:
        return "No model reached the 5% False Discovery Rate cap on development data."
    r = ev["stage1"][winner]
    tuned, default = r["test_tuned"], r["test_default"]
    failures = tuned["tp"] + tuned["fn"]
    cap = ev["fdr_cap"]
    verdict = {
        "met": f"The whole interval is under the {cap:.0%} cap.",
        "not demonstrable": (
            f"The estimate is under the {cap:.0%} cap, but the interval reaches above it. With "
            f"only {failures} test failures, this sample is consistent with the cap being met "
            "and does not prove it."
        ),
        "not met": f"This is above the {cap:.0%} cap: the threshold did not hold up on test.",
        "no alarms": "With no alarms there is no False Discovery Rate to report.",
    }[tuned["cap_verdict"]]
    return (
        f"**{FAMILY_LABELS[winner]}** was selected, with an alarm threshold of "
        f"{r['threshold']:.2f}. On the held-out test set it raised {tuned['alarms']} alarms, of "
        f"which {tuned['fp']} were false: a False Discovery Rate of {pct(tuned['fdr'])} "
        f"(95% interval {interval(tuned['bootstrap']['fdr_ci95'])}). {verdict}\n\n"
        f"It caught {tuned['tp']} of {failures} failures ({pct(tuned['recall'])}) and missed "
        f"{tuned['fn']}. At the default 0.50 threshold the same model raised "
        f"{default['alarms']} alarms with {default['fp']} false ({pct(default['fdr'])} FDR) and "
        f"caught {default['tp']}. Accuracy was {pct(default['accuracy'], 2)} at 0.50 and "
        f"{pct(tuned['accuracy'], 2)} at the tuned threshold."
    )


def rul_headline(m: dict) -> str:
    rul = m["rul"]
    ends = rul["runs"]["end_wear_min"]
    base = rul["models"]["baseline"]["test"]["mae_min"]
    best_ml = min(
        (n for n in rul["models"] if n != "baseline"),
        key=lambda n: rul["models"][n]["test"]["mae_min"],
    )
    ml = rul["models"][best_ml]["test"]["mae_min"]
    verdict = (
        "The learned models did not beat that simple rule"
        if rul["winner"] == "baseline"
        else f"{MODEL_LABELS[rul['winner']]} beat that simple rule on development data"
    )
    return (
        f"Tools ended their runs between {ends['min']:.0f} and {ends['max']:.0f} minutes of wear "
        f"(average {ends['mean']:.0f}). Subtracting current wear from that average predicts "
        f"remaining life with an average error of {base:.1f} minutes on the test runs. {verdict}: "
        f"the best one, {MODEL_LABELS[best_ml]}, had an error of {ml:.1f} minutes."
    )


def to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, rule, *body])


def main() -> None:
    m = load_metrics()
    sections = [
        "# Results summary",
        "Generated by `python -m pdm.report` from `reports/metrics/`. Do not edit by hand.",
        "## Headline",
        headline(m),
        "## Cost of the constraint (winning model, test set)",
        to_markdown(tradeoff_table(m)),
        "## Model comparison (stage 1)",
        to_markdown(stage1_comparison(m)),
        "## What the alarm catches and misses, by failure type (stage 1, test set)",
        "A failure with two types is counted under each.",
        to_markdown(caught_by_type_table(m)),
        "## Failure type (stage 2, on the test rows stage 1 flagged)",
        to_markdown(stage2_table(m)),
        "## Lead indicators (development set)",
        to_markdown(lead_indicator_table(m)),
        "## Remaining tool life (proxy for time to failure)",
        rul_headline(m),
        to_markdown(rul_table(m)),
        "## Data",
        to_markdown(data_profile(m)),
    ]
    SUMMARY_MD.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(SUMMARY_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
