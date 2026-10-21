# Loom script (2-3 minutes)

Camera on for the full recording.

## 0:00-0:20 — Introduction

"Hi, I'm presenting my freight rate prediction solution. The repo contains the
training pipeline, the validation predictions, the completed December inputs, and
the report. I'll cover the data findings, the split, the model, and the results."

## 0:20-0:55 — Data findings and data quality

- Training data is January through October 2025; validation is November and
  December. That makes this a future-forecast problem, not a random-split problem.
- Found 300 missing weights in train and 165 in validation. I take absolute weight
  because 292 train rows and 145 validation rows have sign-flipped weights, then
  impute missing weights by equipment median.
- Found exactly 335 rates below $1.20 per mile and 335 above $4.50 per mile. These
  are symmetric, randomly distributed label anomalies. I exclude them from fitting
  and report the clean and full holdout metrics separately.
- Validation has 1,461 loads on unseen lanes, so lane effects need a fallback.
- Market index moves with rates in-sample but the relationship changes by
  September-October, so it is not used in the final model.

## 0:55-1:40 — Validation design and model reasoning

- I use a forward time split: fit January-August, validate September-October.
  A second fold fits January-July and validates August-September.
- The target is log rate. The main structure is log rate as a function of log
  distance, log weight, equipment, and squared/interaction terms.
- I add day-of-week indicators for the persistent weekly pattern.
- I add out-of-fold smoothed target encodings for pickup city, delivery city, and
  lane. These capture lane effects but never use the current row's target.
- I use Ridge regression. The selected model is Ridge with alpha 1.0. I tested
  gradient-boosted trees and market-index features; they overfit and generalized
  worse in the forward holdout.

## 1:40-2:15 — Results and December chart

- On the clean September-October holdout: RMSE about $113, MAE about $44, MAPE
  1.84%, R² 0.993. The base distance-weight-equipment model gets RMSE about $129,
  so the lane and weekday features add real value.
- The full holdout RMSE is about $631 because of the 142 corrupted labels in that
  period. Those are not predictable from the features.
- December predictions average about $807, ranging about $805 to $823. The repeating
  weekly shape is the learned weekday effect. The model does not need market index
  or quote signal, which is important because the December inputs do not include them.

## 2:15-2:45 — Code walkthrough

- `freight_rate.py` is the main pipeline. `prepare_frame` cleans and engineers
  features. `oof_encode` builds the out-of-fold target encodings. `evaluate_configurations`
  runs the model comparison. `fit_final_model` refits on all training data, and
  `write_predictions` produces both required output files.
- `score.py` is the supplied scorer. It validates the 12,000 predictions and the
  31 December rows and generates the official chart.
- `make_report.py` rebuilds the DOCX and PDF report from the saved metrics.
- Run with `python freight_rate.py`, then `python score.py ...`, then
  `python make_report.py`.

## 2:45-3:00 — Close

"The main takeaway is that a clean, regularized log-linear model with lane-level
target encoding fits this data very well. The largest remaining risk is the
unpredictable label anomalies, not the model structure. Thanks for watching."
