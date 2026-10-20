"""
AI-Based Phishing Website Detection
Validation Utility: Feature Vector Integrity Checker

Validates that src/feature_extraction.py outputs precisely conform to:
1. models/feature_names.json (exact names and exact sequence)
2. Numerical validity (zero NaNs, zero Infs, correct numeric datatypes)
3. Dimensionality (exactly 54 features)
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.feature_extraction import extract_features, get_expected_feature_names


def validate_feature_vector(
    url: str,
    feature_names_path: str = "models/feature_names.json",
    check_network: bool = False,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate extracted features for a given URL against canonical requirements.
    
    Returns:
        (is_valid, error_list, details_dict)
    """
    errors: List[str] = []
    expected_names = get_expected_feature_names(feature_names_path)

    try:
        features_df = extract_features(
            url, check_network=check_network, return_type="dataframe"
        )
    except Exception as e:
        return False, [f"Extraction crashed with exception: {e}"], {}

    # 1. Dimensionality check
    if features_df.shape[1] != 54:
        errors.append(f"Expected 54 columns, but got {features_df.shape[1]}")

    # 2. Exact column name and order match
    actual_columns = list(features_df.columns)
    if actual_columns != expected_names:
        missing = set(expected_names) - set(actual_columns)
        extra = set(actual_columns) - set(expected_names)
        if missing:
            errors.append(f"Missing expected features: {missing}")
        if extra:
            errors.append(f"Unexpected extra features: {extra}")
        if not missing and not extra:
            errors.append("Feature names are present but in the wrong order.")

    # 3. Numeric values, NaN, and Inf checks
    row = features_df.iloc[0]
    nan_cols = []
    inf_cols = []
    non_numeric_cols = []

    for col in expected_names:
        if col in row:
            val = row[col]
            if not isinstance(val, (int, float)):
                non_numeric_cols.append((col, str(type(val))))
            elif math.isnan(val):
                nan_cols.append(col)
            elif math.isinf(val):
                inf_cols.append(col)

    if non_numeric_cols:
        errors.append(f"Non-numeric values found in: {non_numeric_cols}")
    if nan_cols:
        errors.append(f"NaN values found in: {nan_cols}")
    if inf_cols:
        errors.append(f"Infinite values found in: {inf_cols}")

    is_valid = len(errors) == 0
    details = {
        "url": url,
        "feature_count": len(actual_columns),
        "is_valid": is_valid,
        "errors": errors,
        "features": row.to_dict(),
    }

    return is_valid, errors, details


def run_self_check():
    """Run validation self-check on standard URLs."""
    test_urls = [
        "https://www.google.com",
        "https://www.amazon.com",
        "http://192.168.1.1/login.php",
        "http://paypal-login-security.example.com/verify/account",
        "https://example.com/login?user=test&id=123",
    ]

    print("=== RUNNING FEATURE VALIDATION SUITE ===")
    all_passed = True
    for u in test_urls:
        valid, errs, _ = validate_feature_vector(u, check_network=False)
        status = "PASSED" if valid else "FAILED"
        print(f"[{status}] {u}")
        if not valid:
            all_passed = False
            for err in errs:
                print(f"   -> Error: {err}")

    if all_passed:
        print("\nAll feature validation tests passed successfully.")
    else:
        print("\nValidation failures detected.")


if __name__ == "__main__":
    run_self_check()
