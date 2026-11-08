# Industrial Predictive Maintenance

Elevvo internship, Industry Level Task 9, on the
[AI4I 2020 Predictive Maintenance dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).

A false alarm stops a production line that was working, so this project does not chase
accuracy. It builds a failure detector whose alarms can be trusted, measured by the
**False Discovery Rate (FDR)**: of all alarms raised, the share that were false. The
alarm threshold is the one that catches the most failures while at most 5% of alarms are
false.

## Results

Full write-up: [notebooks/predictive_maintenance.ipynb](notebooks/predictive_maintenance.ipynb).
All figures below are from the held-out test set (24 tool runs, 1,951 rows, 65 failures)
and are generated into [reports/summary.md](reports/summary.md).

**The alarm (stage 1).** LightGBM was selected. Tuning the threshold cut false alarms
from 7 to 2 at the cost of 2 more missed failures.

| Threshold | Alarms | False alarms | Failures caught | Failures missed | FDR | Recall | Accuracy |
|---|---|---|---|---|---|---|---|
| Default 0.50 | 64 | 7 | 57 | 8 | 10.9% | 87.7% | 99.23% |
| Tuned 0.75 | 57 | 2 | 55 | 10 | 3.5% | 84.6% | 99.38% |

The 3.5% is under the 5% cap, but its 95% interval is 0.0% to 8.0%. With 65 test
failures the result is consistent with the cap being met and does not prove it.

**What it misses.** It catches 22 of 22 heat dissipation, 16 of 16 power and 22 of 23
overstrain failures, but only 1 of 9 tool wear failures. Tool wear failures happen at an
unpredictable point late in a tool's life and the sensors do not mark them out.

**Failure type (stage 2).** For flagged rows, a second model names the failure type.
Heat dissipation, power and overstrain are identified with 88% to 100% precision and 94%
to 100% recall.

**Lead indicator (bonus).** Torque is the sensor most strongly associated with failure
overall, then rotational speed; temperature alone is weakest. Each failure type has its
own driver: power (torque x speed) for power failures, low speed with a small temperature
gap for heat dissipation, wear x torque for overstrain.

**Time to failure (bonus).** The data has no timestamps, so the target is a proxy:
minutes of tool wear left before the tool's run ends. A one-line rule (average tool life
minus current wear) has an average error of 9.2 minutes; XGBoost and LightGBM did not
beat it (9.7 and 9.9). The sensors carry no information about when a tool's run ends.

## How it works

- **Two stages.** A binary detector raises the alarm, and FDR is controlled there with a
  single threshold. A multi-label classifier then names the failure type or types.
- **Three model families** per stage: logistic regression, XGBoost, LightGBM.
- **Split by tool run.** Rows are in time order in 120 tool runs. Whole runs go to
  development (80%) or test (20%), and cross-validation folds are whole runs too, so
  near-identical neighbouring rows never sit on both sides.
- **The test set makes no decisions.** Thresholds, class weights and the winning model
  come from cross-validated development predictions.
- **Engineered features** from each row's own readings: power, temperature difference,
  wear x torque.

Design decisions and the reasons for them are in [docs/decisions.md](docs/decisions.md);
the rules the code follows are in [docs/rules.md](docs/rules.md).

## Run it

Download `ai4i2020.csv` from Kaggle into `data/raw/`, then (Windows):

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .

.venv\Scripts\python.exe -m pdm.eda
.venv\Scripts\python.exe -m pdm.train
.venv\Scripts\python.exe -m pdm.evaluate
.venv\Scripts\python.exe -m pdm.rul
.venv\Scripts\python.exe -m pdm.report

.venv\Scripts\python.exe -m pytest
```

Results are seeded and reproduce exactly from a clean run.

## Limitations

- The dataset is synthetic and its failures follow fixed rules; real sensor data would
  be noisier and harder.
- The test set is small, so the FDR interval is wide. Re-check the cap on more data
  before relying on it.
- The threshold sits at the edge of the cap by design, so on new data the false alarm
  share will land a little above 5% about as often as below it.
- Lead indicators are associations within a cycle, and time to failure is a proxy,
  because the data has no time dimension.
