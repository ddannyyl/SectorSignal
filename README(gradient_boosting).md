# Gradient Boosting Model (XGBoost)

## Overview

This project implements a Gradient Boosting pipeline using XGBoost for binary classification.
Three models are compared:

* **Baseline model**
* **Manual tuned model**
* **Hyperparameter tuned model (GridSearchCV)**

## Features

* Reproducible training pipeline
* Hyperparameter tuning with GridSearchCV
* Runtime tracking and model persistence
* Evaluation metrics:

  * Accuracy
  * F1 Score
  * Precision
  * Recall
  * ROC-AUC
* Visualization outputs:

  * Confusion matrices
  * ROC curve comparison
  * Metric comparison bar chart
  * Sensitivity analysis (threshold tuning)
  * Feature importance

## Project Structure

```
GradientBoosting/
│── train_gb.py
│
data/
│── X_train(2016-2023).csv
│── y_train1(2016-2023).csv
│── X_val(2024).csv
│── y_val(2024).csv
│── X_test(2025).csv
│── y_test(2025).csv
│
GradientBoostingResults/
│── metrics_comparison.csv
│── *.png (plots)
│── xgb_*.json (models)
│── xgb_hypertuned_runtime.pkl
```

## How to Run

```bash
pip install -r requirements.txt
```

```bash
python GradientBoosting/train_gb.py
```

## Configuration Flags

Inside `train_gb.py`:

* `FAST_MODE = True`
  → Uses smaller grid for faster tuning (recommended for Codespaces)

* `LOAD_SAVED_MODEL = True`
  → Loads saved model instead of retraining

## Results Summary

The hypertuned model improves F1 score and recall compared to baseline and manual tuning, though ROC-AUC remains close to random (~0.5), suggesting limited feature separability.

## Notes

* Sensitivity analysis helps choose optimal classification threshold
* Designed for reproducibility and experimentation

```
```
