# Decisions

## Agreed with the user

| Topic | Decision |
|---|---|
| Deliverable | `src/pdm/` package + a narrative notebook that imports from it |
| Framing | Two-stage. Stage 1: binary failure detector. Stage 2: failure-type classifier for flagged rows |
| Models | scikit-learn baseline, XGBoost and LightGBM, compared; winner chosen on FDR-constrained recall |
| FDR rule | Stage 1 threshold = the one with highest recall subject to **FDR <= 5%** |
| Dataset | Kaggle `ai4i2020.csv` in `data/raw/` |
| Environment | `.venv` + `requirements.txt`, Python 3.12 |
| Order of work | 1. core classifier, 2. lead-indicator correlation, 3. time-to-failure regression |
| Stage 1 label | `Machine failure` exactly as given. The 18 RNF rows it marks as healthy stay healthy |
| RNF | Dropped from stage 2 (random by design; 18 of 19 are not machine failures) |
| Untyped failures | The 9 failures with no type flag stay in stage 1, and are left out of stage 2 training |
| Multi-type failures | Stage 2 is multi-label: a yes/no for each of TWF, HDF, PWF, OSF |
| Split | By tool run. 80% of runs development, 20% test, 5 CV folds also by run, seed 42 |
| Test uncertainty | Bootstrap confidence interval on test FDR, resampling whole tool runs |
| Time-to-failure | Remaining tool life: minutes of wear left until the tool's run ends (failure or replacement) |

## Facts these rest on (verified on the raw file)

- Rows in `UDI` order form 120 tool runs: wear rises 2 to 5 minutes per row, then resets
  to 0. `tool_run` is derived from those resets. It is a grouping key, never a feature.
- 339 machine failures. 24 carry more than one type flag, 9 carry none.
- The last run is cut off by the end of the file (11 rows), so its end is unknown and it
  is excluded from the time-to-failure target.

## Working defaults (proposed by Claude, not yet confirmed)

- Class weight for the failure class is tuned over a small grid on development data,
  scored by recall at the 5% FDR cap.
- Stage 2 model family is the one with the best out-of-fold macro F1.
- The threshold is the most permissive one that meets the cap on development data, with
  no safety margin (the agreed rule, taken literally). Both reviewers noted that test FDR
  will therefore land above 5% about as often as below. A stricter development target
  (for example 3%) would trade some recall for a firmer guarantee: ask the user.
