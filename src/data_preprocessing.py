"""
AI-Based Phishing Website Detection
Phase 3: Data Preprocessing and Train/Test Split Module

Provides modular, reproducible data preparation pipeline:
- Raw dataset loading
- Feature matrix & target separation
- Zero-variance filtering (removal of constant 'nb_or')
- Stratified 80/20 train/test splitting
- StandardScaler fitting strictly on X_train
- Artifact serialization for modeling
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw_data(file_path: str = "data/raw/dataset_phishing.csv") -> pd.DataFrame:
    """Load the raw phishing dataset from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {file_path}")
    return pd.read_csv(path)


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Separate raw URL, encode target variable, and extract ML features.
    
    Target Encoding:
        legitimate -> 0
        phishing   -> 1
        
    Removed Columns:
        - 'url': Raw URL string (preserved separately for reference, not a direct tabular feature)
        - 'status': Target label
        - 'nb_or': Constant feature with zero variance (all values are 0)
    """
    # 1. Preserve URLs separately
    urls = df["url"].copy()

    # 2. Convert target: legitimate -> 0, phishing -> 1
    target_mapping = {"legitimate": 0, "phishing": 1}
    y = df["status"].map(target_mapping).astype(int)

    # 3. Create feature matrix (drop url, status, nb_or)
    columns_to_drop = ["url", "status", "nb_or"]
    X = df.drop(columns=columns_to_drop)

    feature_names = list(X.columns)
    return X, y, urls, feature_names


def inspect_feature_quality(X: pd.DataFrame, corr_threshold: float = 0.90) -> Dict:
    """
    Examine the feature matrix for:
    - Constant / zero-variance columns
    - Identical / duplicate columns
    - Highly correlated pairs (|r| >= corr_threshold)
    """
    # Check zero variance
    zero_var = [col for col in X.columns if X[col].nunique() <= 1]

    # Check identical columns
    dup_cols = []
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if X[cols[i]].equals(X[cols[j]]):
                dup_cols.append((cols[i], cols[j]))

    # Check pairwise correlations
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = []
    for i in range(len(upper.index)):
        for j in range(len(upper.columns)):
            val = upper.iloc[i, j]
            if val >= corr_threshold:
                high_corr.append({
                    "feature_1": upper.index[i],
                    "feature_2": upper.columns[j],
                    "correlation": round(float(val), 4),
                })

    return {
        "zero_variance_features": zero_var,
        "duplicate_columns": dup_cols,
        "high_correlation_pairs": high_corr,
    }


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    urls: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Split the dataset using stratified sampling.
    Preserves index alignment with raw URLs.
    """
    X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
        X,
        y,
        urls,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, urls_train, urls_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Fit StandardScaler strictly on X_train, and transform both X_train and X_test.
    Preserves column names and index structures.
    """
    scaler = StandardScaler()
    
    # Fit ONLY on X_train to prevent data leakage
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    
    # Transform X_test with fitted scaler
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return scaler, X_train_scaled, X_test_scaled


def save_artifacts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    scaler: StandardScaler,
    feature_names: List[str],
    processed_dir: str = "data/processed",
    models_dir: str = "models",
) -> None:
    """Save processed datasets, scaler, and feature metadata to disk."""
    proc_path = Path(processed_dir)
    mod_path = Path(models_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    mod_path.mkdir(parents=True, exist_ok=True)

    # Save unscaled datasets (for Tree-based models: Decision Tree, Random Forest)
    X_train.to_csv(proc_path / "X_train.csv", index=True)
    X_test.to_csv(proc_path / "X_test.csv", index=True)
    y_train.to_frame(name="status").to_csv(proc_path / "y_train.csv", index=True)
    y_test.to_frame(name="status").to_csv(proc_path / "y_test.csv", index=True)

    # Save scaled datasets (for linear/distance models: Logistic Regression, SVM)
    X_train_scaled.to_csv(proc_path / "X_train_scaled.csv", index=True)
    X_test_scaled.to_csv(proc_path / "X_test_scaled.csv", index=True)

    # Save fitted scaler and feature names
    joblib.dump(scaler, mod_path / "scaler.joblib")
    with open(mod_path / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)


def run_pipeline() -> Dict:
    """Execute complete Phase 3 data preprocessing pipeline."""
    df = load_raw_data("data/raw/dataset_phishing.csv")
    X, y, urls, feature_names = prepare_features_and_target(df)
    quality_report = inspect_feature_quality(X)

    X_train, X_test, y_train, y_test, urls_train, urls_test = split_dataset(
        X, y, urls, test_size=0.20, random_state=42
    )

    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    save_artifacts(
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
        feature_names,
    )

    return {
        "raw_shape": df.shape,
        "features_count": len(feature_names),
        "removed_columns": ["url", "status", "nb_or"],
        "target_mapping": {"legitimate": 0, "phishing": 1},
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts().to_dict(),
        "y_test_counts": y_test.value_counts().to_dict(),
        "quality_report": quality_report,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print("Preprocessing completed successfully.")
    print(json.dumps(results, indent=2))
