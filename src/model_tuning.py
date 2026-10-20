"""
AI-Based Phishing Website Detection
Phase 5: Controlled Hyperparameter Tuning Module

Performs Stratified 5-Fold Cross-Validation strictly on training data:
1. Random Forest Classifier tuning (on unscaled X_train)
2. Support Vector Machine / SVC tuning (on standardized X_train_scaled)

Key constraints:
- Primary selection metric: Mean Cross-Validation F1-score (positive class = phishing)
- Test set (X_test, y_test) remains completely untouched during tuning
- Retrains optimal models on full training set
- Evaluates tuned models once on the unseen test set
- Serializes tuned models and exports hyperparameter_tuning_report.json
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


def load_processed_data(data_dir: str = "data/processed") -> Dict[str, Any]:
    """Load train/test processed datasets."""
    p = Path(data_dir)
    return {
        "X_train": pd.read_csv(p / "X_train.csv", index_col=0),
        "X_test": pd.read_csv(p / "X_test.csv", index_col=0),
        "X_train_scaled": pd.read_csv(p / "X_train_scaled.csv", index_col=0),
        "X_test_scaled": pd.read_csv(p / "X_test_scaled.csv", index_col=0),
        "y_train": pd.read_csv(p / "y_train.csv", index_col=0).squeeze("columns"),
        "y_test": pd.read_csv(p / "y_test.csv", index_col=0).squeeze("columns"),
    }


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Dict[str, Any]:
    """
    Tune Random Forest using 5-fold CV on training data.
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }

    # Evaluate multiple metrics across folds; refit best by f1
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_idx = grid_search.best_index_
    cv_results = grid_search.cv_results_

    return {
        "best_estimator": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "cv_metrics": {
            "mean_f1": float(cv_results["mean_test_f1"][best_idx]),
            "std_f1": float(cv_results["std_test_f1"][best_idx]),
            "mean_accuracy": float(cv_results["mean_test_accuracy"][best_idx]),
            "std_accuracy": float(cv_results["std_test_accuracy"][best_idx]),
            "mean_precision": float(cv_results["mean_test_precision"][best_idx]),
            "std_precision": float(cv_results["std_test_precision"][best_idx]),
            "mean_recall": float(cv_results["mean_test_recall"][best_idx]),
            "std_recall": float(cv_results["std_test_recall"][best_idx]),
        },
    }


def tune_svm(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Dict[str, Any]:
    """
    Tune SVM (RBF kernel) using 5-fold CV on standardized training data.
    """
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1],
        "kernel": ["rbf"],
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    grid_search = GridSearchCV(
        estimator=SVC(random_state=42),
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train_scaled, y_train)

    best_idx = grid_search.best_index_
    cv_results = grid_search.cv_results_

    return {
        "best_estimator": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "cv_metrics": {
            "mean_f1": float(cv_results["mean_test_f1"][best_idx]),
            "std_f1": float(cv_results["std_test_f1"][best_idx]),
            "mean_accuracy": float(cv_results["mean_test_accuracy"][best_idx]),
            "std_accuracy": float(cv_results["std_test_accuracy"][best_idx]),
            "mean_precision": float(cv_results["mean_test_precision"][best_idx]),
            "std_precision": float(cv_results["std_test_precision"][best_idx]),
            "mean_recall": float(cv_results["mean_test_recall"][best_idx]),
            "std_recall": float(cv_results["std_test_recall"][best_idx]),
        },
    }


def evaluate_on_unseen_test(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Compute final evaluation metrics on the untouched test set."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "precision": round(float(precision_score(y_test, y_pred)), 6),
        "recall": round(float(recall_score(y_test, y_pred)), 6),
        "f1_score": round(float(f1_score(y_test, y_pred)), 6),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def run_tuning_pipeline() -> Dict[str, Any]:
    """Execute complete hyperparameter tuning process."""
    data = load_processed_data("data/processed")
    X_train = data["X_train"]
    X_test = data["X_test"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("--- 1. Tuning Random Forest (5-Fold CV on training data) ---")
    rf_tuning = tune_random_forest(X_train, y_train, cv)
    best_rf = rf_tuning["best_estimator"]

    print("--- 2. Tuning Support Vector Machine (5-Fold CV on standardized training data) ---")
    svm_tuning = tune_svm(X_train_scaled, y_train, cv)
    best_svm = svm_tuning["best_estimator"]

    print("--- 3. Evaluating Tuned Models on Untouched Test Set ---")
    rf_test_metrics = evaluate_on_unseen_test(best_rf, X_test, y_test)
    svm_test_metrics = evaluate_on_unseen_test(best_svm, X_test_scaled, y_test)

    # Save models
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_rf, models_dir / "random_forest_tuned.joblib")
    joblib.dump(best_svm, models_dir / "svm_tuned.joblib")

    # Compile report
    report = {
        "cv_setup": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "primary_metric": "f1",
        },
        "Random Forest Tuned": {
            "best_parameters": rf_tuning["best_params"],
            "cv_metrics": rf_tuning["cv_metrics"],
            "test_metrics": rf_test_metrics,
        },
        "SVM Tuned": {
            "best_parameters": svm_tuning["best_params"],
            "cv_metrics": svm_tuning["cv_metrics"],
            "test_metrics": svm_test_metrics,
        },
    }

    report_path = models_dir / "hyperparameter_tuning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Tuning complete. Report saved to: {report_path}")
    return report


if __name__ == "__main__":
    run_tuning_pipeline()
