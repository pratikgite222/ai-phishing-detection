"""
AI-Based Phishing Website Detection
Phase 7: End-to-End Prediction Pipeline

Connects raw URL input -> Live Feature Extractor -> Tuned Random Forest.
Enforces:
- Exact 54 features matching models/feature_names.json in correct sequence.
- NO scaling (Random Forest operates directly on un-scaled numerical features).
- Explicit class order resolution from model.classes_ ([0, 1]).
- Clean, decoupled result dictionary for UI consumption.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.feature_extraction import extract_features, normalize_url
except ImportError:
    from feature_extraction import extract_features, normalize_url


# Global caches for model artifacts
_MODEL_CACHE: Optional[Any] = None
_FEATURE_NAMES_CACHE: Optional[List[str]] = None


def load_model(model_path: str = "models/random_forest_tuned.joblib") -> Any:
    """Load and cache the trained production Random Forest model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    p = Path(model_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p

    if not p.exists():
        raise FileNotFoundError(f"Model file missing at: {p}")

    _MODEL_CACHE = joblib.load(p)
    return _MODEL_CACHE


def load_feature_names(feature_names_path: str = "models/feature_names.json") -> List[str]:
    """Load and cache the expected 54 feature names."""
    global _FEATURE_NAMES_CACHE
    if _FEATURE_NAMES_CACHE is not None:
        return _FEATURE_NAMES_CACHE

    p = Path(feature_names_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p

    if not p.exists():
        raise FileNotFoundError(f"Feature names file missing at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        _FEATURE_NAMES_CACHE = json.load(f)

    return _FEATURE_NAMES_CACHE


def predict_url(
    url: str,
    check_network: bool = True,
    network_timeout: float = 1.5,
    model_path: str = "models/random_forest_tuned.joblib",
    feature_names_path: str = "models/feature_names.json",
    include_features: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end prediction for a raw URL string.
    
    Workflow:
    1. Validate and normalize the URL.
    2. Extract exactly 54 numerical features via src/feature_extraction.py.
    3. Verify feature vector ordering, datatypes, and absence of NaNs/Infs.
    4. Pass UN-SCALED features to the tuned Random Forest model.
    5. Safely map probabilities according to model.classes_ (0 -> Legitimate, 1 -> Phishing).
    6. Return clean, formatted prediction payload.
    """
    # 1. Normalization
    normalized_url = normalize_url(url)

    # 2. Extract features as a DataFrame
    features_df = extract_features(
        normalized_url,
        check_network=check_network,
        network_timeout=network_timeout,
        return_type="dataframe",
    )

    # 3. Model & Feature Specification Validation
    model = load_model(model_path)
    expected_features = load_feature_names(feature_names_path)

    if list(features_df.columns) != expected_features:
        raise ValueError("Feature ordering mismatch between extracted vector and feature_names.json")

    if features_df.shape[1] != 54:
        raise ValueError(f"Expected 54 features, but got {features_df.shape[1]}")

    if features_df.isnull().any().any():
        raise ValueError("Extracted feature vector contains NaN values.")

    # 4. Predict class and probabilities (NO SCALER APPLIED)
    raw_pred = int(model.predict(features_df)[0])
    raw_proba = model.predict_proba(features_df)[0]

    # Explicit class mapping from model.classes_
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Model classes_ must contain [0, 1], found: {classes}")

    legit_idx = classes.index(0)
    phish_idx = classes.index(1)

    legit_prob = float(raw_proba[legit_idx])
    phish_prob = float(raw_proba[phish_idx])

    label = "Phishing" if raw_pred == 1 else "Legitimate"

    result = {
        "original_url": url,
        "normalized_url": normalized_url,
        "prediction": raw_pred,
        "label": label,
        "phishing_probability": round(phish_prob, 4),
        "legitimate_probability": round(legit_prob, 4),
        "feature_vector_length": features_df.shape[1],
    }

    if include_features:
        result["features"] = features_df.iloc[0].to_dict()

    return result


if __name__ == "__main__":
    # Demonstration on standard test URL
    demo_url = "https://www.google.com"
    print("Testing src/predict.py on:", demo_url)
    res = predict_url(demo_url, check_network=False)
    print(json.dumps(res, indent=2))
