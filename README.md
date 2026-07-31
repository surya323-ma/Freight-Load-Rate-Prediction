# Freight Load Rate Prediction

Predicts `posted_rate` (the market rate for a freight load) from lane, equipment,
weight, distance, and market-condition features, using historical loads
(`data/train_test.csv`, 2025-01-01 – 2025-10-31) to predict rates for a future
window (`data/validation.csv`, 2025-11-01 – 2025-12-31).

## Repo layout

```
data/
  train_test.csv                     # labeled development data (given)
  validation.csv                     # 12,000 loads to score (given)
  validation_predictions_template.csv# load_id template (given)
  december_chart_inputs.csv          # fixed Lexington -> Fort Wayne lane, filled in by predict.py
src/
  features.py                        # shared feature engineering / cleaning
  train.py                           # validation split, training, metrics
  predict.py                         # scores validation.csv and the December lane
score.py                             # provided validator + chart generator
report.docx                          # write-up (approach, chart, findings)
reports/
  validation_metrics.json            # holdout + cross-validation metrics
  final_model.joblib                 # trained model + fitted feature builder
scorer_results/
  candidate_december.png             # chart produced by score.py
validation_predictions.csv           # final submission: load_id, predicted_rate
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run instructions

From the repo root:

```bash
# 1. Train the model (time-based holdout validation, then refit on all labeled data)
python3 src/train.py

# 2. Score validation.csv and the December fixed-lane scenario
python3 src/predict.py

# 3. Validate the submission files and produce the required December chart
python3 score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv \
  --output-dir scorer_results
```

`train.py` prints a time-based holdout metric set (the primary validation
number) and a 5-fold random cross-validation metric set (sanity check), and
writes both to `reports/validation_metrics.json`. It saves the final
model/feature-builder bundle to `reports/final_model.joblib`.

`predict.py` loads that bundle, fills in `predicted_rate` for every row of
`data/validation.csv` (written to `validation_predictions.csv` at the repo
root, matching the template's `load_id` order and two-column format), and
fills in `predicted_rate` for the 31-day fixed-lane scenario in
`data/december_chart_inputs.csv`.

`score.py` (provided) re-validates both output files against the assignment's
schema/ID/range checks and renders `scorer_results/candidate_december.png`.

## Approach summary

See `report.docx` for the full write-up. In short:

- **Validation split**: time-based holdout — train on data before
  2025-09-14, validate on 2025-09-14 through 2025-10-31 (the most recent
  ~15% of labeled days) — because the final scoring window
  (Nov–Dec 2025) is entirely after the labeled data, making this a
  forecasting problem rather than an interpolation problem. A 5-fold random
  CV is reported alongside as a secondary sanity check.
- **Model**: LightGBM gradient-boosted trees on `log1p(posted_rate)`, with
  pickup/delivery/equipment/lane as native categoricals, calendar features
  (month, day-of-week, cyclical encodings), and cleaned weight/market_index.
- **Data-quality fixes**: physically-impossible negative `weight` values
  (sign-corrected via absolute value), missing `weight` and `market_index`
  values (median-imputed, learned on the training split only, with
  "was missing" indicator flags).
