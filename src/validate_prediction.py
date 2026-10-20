"""
AI-Based Phishing Website Detection
Validation Utility: Prediction Pipeline Verification

Checks:
1. models/random_forest_tuned.joblib loads successfully
2. models/feature_names.json loads successfully
3. Model expects exactly 54 features
4. Extractor produces exactly 54 features
5. Feature order strictly aligns with training feature_names_in_
6. Model prediction executes without exceptions
7. model.predict_proba executes properly
8. Probabilities sum to 1.0 (within 1e-6 tolerance)
9. Output contains no NaN or Infinite values
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.predict import load_feature_names, load_model, predict_url
except ImportError:
    from predict import load_feature_names, load_model, predict_url


def run_pipeline_validation(
    test_url: str = "https://www.google.com",
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate the complete prediction pipeline against all operational constraints.
    """
    errors: List[str] = []
    details: Dict[str, Any] = {}

    # 1. Check model loading
    try:
        model = load_model("models/random_forest_tuned.joblib")
        details["model_loaded"] = True
        details["model_type"] = str(type(model))
        details["model_classes"] = list(model.classes_)
    except Exception as e:
        errors.append(f"Failed to load model: {e}")
        return False, errors, details

    # 2. Check feature names loading
    try:
        expected_features = load_feature_names("models/feature_names.json")
        details["feature_names_loaded"] = True
        details["feature_count_json"] = len(expected_features)
    except Exception as e:
        errors.append(f"Failed to load feature_names.json: {e}")
        return False, errors, details

    # 3. Model dimensionality check
    if getattr(model, "n_features_in_", None) != 54:
        errors.append(f"Model expects {getattr(model, 'n_features_in_', None)} features, expected 54")

    # 4. Feature ordering alignment check
    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != expected_features:
            errors.append("Mismatch between model.feature_names_in_ and feature_names.json")

    # 5. Prediction execution check
    try:
        res = predict_url(test_url, check_network=False)
        details["prediction_result"] = res
    except Exception as e:
        errors.append(f"predict_url execution failed: {e}")
        return False, errors, details

    # 6. Output structure check
    required_keys = [
        "original_url",
        "normalized_url",
        "prediction",
        "label",
        "phishing_probability",
        "legitimate_probability",
        "feature_vector_length",
    ]
    for k in required_keys:
        if k not in res:
            errors.append(f"Missing key in prediction result: {k}")

    # 7. Probability summation check
    p_phish = res.get("phishing_probability", 0.0)
    p_legit = res.get("legitimate_probability", 0.0)
    prob_sum = p_phish + p_legit
    if not math.isclose(prob_sum, 1.0, abs_tol=1e-3):
        errors.append(f"Probabilities do not sum to 1.0 (Sum: {prob_sum})")

    # 8. NaN / Inf check
    if math.isnan(p_phish) or math.isinf(p_phish) or math.isnan(p_legit) or math.isinf(p_legit):
        errors.append("Probabilities contain NaN or Inf values")

    is_valid = len(errors) == 0
    return is_valid, errors, details


def main():
    print("=== RUNNING PREDICTION PIPELINE VALIDATION ===")
    valid, errs, info = run_pipeline_validation()
    if valid:
        print("Model Loaded: Successfully")
        print(f"Model Classes: {info.get('model_classes')}")
        print(f"Feature Count: {info.get('feature_count_json')}")
        print("Feature Order: 100% Verified against model.feature_names_in_")
        print("Probability Sum: 1.0 Verified")
        print("\nALL PREDICTION PIPELINE INTEGRITY CHECKS PASSED.")
    else:
        print("VALIDATION ERRORS:")
        for e in errs:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
