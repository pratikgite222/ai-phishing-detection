"""
AI-Based Phishing Website Detection
Phase 4: Machine Learning Model Training and Evaluation Module

Trains and evaluates 4 standard classification models:
1. Logistic Regression (on scaled features)
2. Decision Tree Classifier (on unscaled features)
3. Random Forest Classifier (on unscaled features)
4. Support Vector Machine / SVC (on scaled features)

Features:
- Reproducible random_state=42
- Strict separation between train and unseen test sets
- Comprehensive evaluation (Train vs Test to inspect overfitting)
- Artifact serialization in models/ directory
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_datasets(processed_dir: str = "data/processed") -> Dict[str, Any]:
    """Load train/test processed datasets."""
    p = Path(processed_dir)
    return {
        "X_train": pd.read_csv(p / "X_train.csv", index_col=0),
        "X_test": pd.read_csv(p / "X_test.csv", index_col=0),
        "X_train_scaled": pd.read_csv(p / "X_train_scaled.csv", index_col=0),
        "X_test_scaled": pd.read_csv(p / "X_test_scaled.csv", index_col=0),
        "y_train": pd.read_csv(p / "y_train.csv", index_col=0).squeeze("columns"),
        "y_test": pd.read_csv(p / "y_test.csv", index_col=0).squeeze("columns"),
    }


def get_models() -> Dict[str, Dict[str, Any]]:
    """
    Define the four classifiers with standard parameters and specify their data scaling needs.
    """
    return {
        "Logistic Regression": {
            "model": LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            ),
            "use_scaled": True,
            "filename": "logistic_regression.joblib",
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                criterion="gini",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
            ),
            "use_scaled": False,
            "filename": "decision_tree.joblib",
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
            ),
            "use_scaled": False,
            "filename": "random_forest.joblib",
        },
        "Support Vector Machine": {
            "model": SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                random_state=42,
            ),
            "use_scaled": True,
            "filename": "svm.joblib",
        },
    }


def train_and_evaluate_all(
    models_dict: Dict[str, Dict[str, Any]],
    data: Dict[str, Any],
    models_dir: str = "models",
) -> Dict[str, Any]:
    """
    Train each model, compute train and test metrics, save models, and compile report.
    """
    m_dir = Path(models_dir)
    m_dir.mkdir(parents=True, exist_ok=True)

    y_train = data["y_train"]
    y_test = data["y_test"]

    evaluation_report = {}

    for name, config in models_dict.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]
        filename = config["filename"]

        X_tr = data["X_train_scaled"] if use_scaled else data["X_train"]
        X_te = data["X_test_scaled"] if use_scaled else data["X_test"]

        # Train model
        clf.fit(X_tr, y_train)

        # Predictions
        y_pred_tr = clf.predict(X_tr)
        y_pred_te = clf.predict(X_te)

        # Metrics on Train
        tr_acc = float(accuracy_score(y_train, y_pred_tr))
        tr_prec = float(precision_score(y_train, y_pred_tr))
        tr_rec = float(recall_score(y_train, y_pred_tr))
        tr_f1 = float(f1_score(y_train, y_pred_tr))
        tr_cm = confusion_matrix(y_train, y_pred_tr).tolist()

        # Metrics on Unseen Test
        te_acc = float(accuracy_score(y_test, y_pred_te))
        te_prec = float(precision_score(y_test, y_pred_te))
        te_rec = float(recall_score(y_test, y_pred_te))
        te_f1 = float(f1_score(y_test, y_pred_te))
        te_cm = confusion_matrix(y_test, y_pred_te).tolist()

        # Serialize model
        joblib.dump(clf, m_dir / filename)

        evaluation_report[name] = {
            "parameters": clf.get_params(),
            "use_scaled_data": use_scaled,
            "train_metrics": {
                "accuracy": round(tr_acc, 6),
                "precision": round(tr_prec, 6),
                "recall": round(tr_rec, 6),
                "f1_score": round(tr_f1, 6),
                "confusion_matrix": tr_cm,
            },
            "test_metrics": {
                "accuracy": round(te_acc, 6),
                "precision": round(te_prec, 6),
                "recall": round(te_rec, 6),
                "f1_score": round(te_f1, 6),
                "confusion_matrix": te_cm,
            },
            "overfitting_gap_accuracy": round(tr_acc - te_acc, 6),
        }

    # Save evaluation report
    with open(m_dir / "model_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    return evaluation_report


def main():
    print("Loading preprocessed datasets...")
    data = load_datasets()
    print("Initializing models...")
    models_dict = get_models()
    print("Training and evaluating models...")
    report = train_and_evaluate_all(models_dict, data)
    print("Model training, evaluation, and artifact saving complete.")


if __name__ == "__main__":
    main()
